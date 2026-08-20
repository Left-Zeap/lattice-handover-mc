"""扫描场景的批量运输腿 Monte Carlo：多网格点同时在 GPU 上积分。

``run_leg_monte_carlo_batch`` 与 ``handover_batch`` 同型：把 P 个网格点
的 L1/L2 运输腿粒子摊平成单个 ``(P×N, 3)`` 粒子数组，用一个批量
mega-step 融合 kernel（与 ``transport_mc._get_fused_leg_step_kernel``
逐式同构的整步 velocity-Verlet）同时推进全部点的全部轨迹。

与批量 handover 的结构差异：运输腿的光学系数（双束包络/势/力组合系
数）随 z_L(t) 逐步变化且逐点不同，因此预先把全部步的 13 个系数在
host 按 ``(步, 13, 点)`` 表算好上传，每步取一行按 ``point_index``
gather 成逐粒子数组（表内存超限时自动减小分块点数）。

约束与口径：

- 同批要求所有点的 ``L1TransportInputs`` 一致（扫描天然满足；
  不一致抛 ``ValueError`` 回退逐点 ``simulate_leg_monte_carlo``）。
  逐点只变失谐/功率（→ 波长、强度剖面、偶极系数、散射系数）。
  例外：``initial_temperature_uK``/``initial_atom_number``/
  ``mot_atom_number`` 三个初态字段允许逐点不同（全链路 L2 腿的
  初温/原子数来自各点 handover 捕获样本）；它们只影响 host 端
  初态采样与 trace 组装，不进入批量动力学。
- ``conveyor_enabled=True`` 时束腰剖面经瑞利长度随波长逐点不同，
  批量路径未覆盖，直接抛 ``ValueError`` 回退逐点（docstring 注明）。
- 时序（z_L(t)、逃逸剔除间隔、快照网格）全批共享；初态逐点在 CPU
  上用各自 seed 的 NumPy RNG 采样（与逐点调用逐位一致）；散射反冲
  RNG 全批共享，结果与逐点仅统计一致。
- 逃逸剔除（每 200 步+末步）按逐点倾斜势垒在全批粒子数组上统一
  进行；快照（``_time_grid`` 网格）把粒子状态取回 host 逐点汇总，
  留存率 Jeffreys 标准误、温度、bound 比例、累计散射、瞬时损失率
  均与逐点 ``simulate_leg_monte_carlo`` 同口径；采样失败（浅阱/高温
  无束缚初态）的点按既有口径返回零留存 trace，不中断整批。
- 显存保护：P×N 超过 ``handover_batch._MAX_BATCH_PARTICLES`` 或系数
  表超过 ``_MAX_COEFF_TABLE_BYTES`` 时自动分块。

``backend="cpu"`` 时退化为逐点调用 ``simulate_leg_monte_carlo``。
"""

from __future__ import annotations

from dataclasses import fields
import math

import numpy as np

from .constants import BOLTZMANN, GRAVITY, HBAR
from .device_loop import (
    allocate_rng_states,
    get_leg_loop_kernels,
    launch_config,
)
from .dipole import scalar_potential_and_scattering
from .gpu_backend import (
    resolve_backend as _resolve_backend,
    scattering_kicks_gpu as _scattering_kicks_gpu,
    scattering_rng_gpu as _scattering_rng_gpu,
)
from .handover_batch import _MAX_BATCH_PARTICLES
from .lattice import (
    evaluate_lattice,
    gaussian_gravity_trap,
    tilted_lattice_barrier_fraction,
)
from .l1_transport import (
    L1DesignPoint,
    L1TransportInputs,
    L1TransportTrace,
    _atom_from_label,
    _kinematics,
    _time_grid,
    l1_timing,
)
from .phase_space import ParticleEnsemble
from .transport_mc import (
    _ESCAPE_CHECK_INTERVAL_STEPS,
    _double_beam_potential_and_force,
    _leg_optics_at,
    _leg_optics_profile,
    _sample_initial_ensemble,
    _stable_leg_step_s,
    _zero_retention_trace,
    simulate_leg_monte_carlo,
)


# 系数表 (步×13×点) 的显存上限（约 300 MB）。
_MAX_COEFF_TABLE_BYTES = 300_000_000

# 逐步系数表的列序（与批量 mega-step kernel 参数同序）。
_COEFF_COLUMNS = (
    "envelope_c1",
    "envelope_c2",
    "envelope_cc",
    "intensity1",
    "intensity2",
    "potential_c1",
    "potential_c2",
    "potential_cc",
    "axial_c",
    "radial_c1",
    "radial_c2",
    "radial_cc",
    "two_wave_number",
)


# 允许逐点不同的初态字段：只影响 host 端初态采样宽度与 trace 末端
# 原子数标量缩放，不影响批量动力学（光学/时序/数值参数仍必须全批一致）。
_PER_POINT_LEG_FIELDS = frozenset(
    {"initial_temperature_uK", "initial_atom_number", "mot_atom_number"}
)


def _check_leg_consistency(tasks) -> None:
    """校验同批所有点的输入一致（初态字段除外）且未启用 conveyor 几何。"""
    reference = tasks[0][1]
    for index, inputs, _, _ in tasks[1:]:
        differing = [
            field.name
            for field in fields(inputs)
            if field.name not in _PER_POINT_LEG_FIELDS
            and getattr(inputs, field.name) != getattr(reference, field.name)
        ]
        if differing:
            raise ValueError(
                f"批量运输腿要求所有点的 L1TransportInputs 全等（仅 "
                f"{sorted(_PER_POINT_LEG_FIELDS)} 允许逐点不同）："
                f"点 {index} 与点 {tasks[0][0]} 的字段 {differing} 不一致；"
                "请回退为逐点调用"
            )
    if reference.conveyor_enabled:
        raise ValueError(
            "conveyor 几何下束腰剖面经瑞利长度随波长逐点不同，批量运输腿"
            "未覆盖该情形；请回退为逐点调用（或将 conveyor 关闭）"
        )


_FUSED_BATCH_LEG_STEP_KERNEL = None


def _get_fused_batch_leg_step_kernel():
    """惰性创建批量运输腿整步 velocity-Verlet 融合 kernel（mega-step）。

    与 ``transport_mc._get_fused_leg_step_kernel`` 逐式同构，但 13 个
    双束系数全部是逐粒子数组（逐点经系数表 gather）；时序标量
    （lattice_position、半步系数、步长）全批共享。kernel 内只出现
    数组·数组 或 数组·标量 运算；就地更新列视图 ``p0..f2``，返回
    新位置处的 ``(势, 局域前向强度, 局域非相干强度和)``。相对相位
    φ=0（与逐点路径同一约定）。
    """
    global _FUSED_BATCH_LEG_STEP_KERNEL
    if _FUSED_BATCH_LEG_STEP_KERNEL is None:
        import cupy as cp

        @cp.fuse()
        def kernel(
            p0,
            p1,
            p2,
            v0,
            v1,
            v2,
            f0,
            f1,
            f2,
            envelope_c1,
            envelope_c2,
            envelope_cc,
            intensity1,
            intensity2,
            potential_c1,
            potential_c2,
            potential_cc,
            axial_c,
            radial_c1,
            radial_c2,
            radial_cc,
            two_wave_number,
            lattice_position,
            gravity_force_y,
            half_dt_over_mass,
            time_step,
        ):
            # 半步速度 + 整步位置。
            nv0 = v0 + f0 * half_dt_over_mass
            nv1 = v1 + f1 * half_dt_over_mass
            nv2 = v2 + f2 * half_dt_over_mass
            np0 = p0 + nv0 * time_step
            np1 = p1 + nv1 * time_step
            np2 = p2 + nv2 * time_step
            # 新位置、新时刻的双束势与力（逐粒子系数）。
            rho2 = np0 * np0 + np1 * np1
            zeta = np2 - lattice_position
            envelope1 = cp.exp(rho2 * envelope_c1)
            envelope2 = cp.exp(rho2 * envelope_c2)
            cross_envelope = cp.exp(rho2 * envelope_cc)
            theta = two_wave_number * zeta
            cosine = cp.cos(theta)
            potential = -(
                potential_c1 * envelope1
                + potential_c2 * envelope2
                + potential_cc * cross_envelope * cosine
            )
            axial_force = -(axial_c * cross_envelope * cp.sin(theta))
            radial_coefficient = -(
                radial_c1 * envelope1
                + radial_c2 * envelope2
                + radial_cc * cross_envelope * cosine
            )
            g0 = radial_coefficient * np0
            g1 = radial_coefficient * np1 + gravity_force_y
            g2 = axial_force
            # 半步速度（新力），全部状态就地写回。
            p0[...] = np0
            p1[...] = np1
            p2[...] = np2
            v0[...] = nv0 + g0 * half_dt_over_mass
            v1[...] = nv1 + g1 * half_dt_over_mass
            v2[...] = nv2 + g2 * half_dt_over_mass
            f0[...] = g0
            f1[...] = g1
            f2[...] = g2
            local_forward = intensity1 * envelope1
            local_incoherent = local_forward + intensity2 * envelope2
            return potential, local_forward, local_incoherent

        _FUSED_BATCH_LEG_STEP_KERNEL = kernel
    return _FUSED_BATCH_LEG_STEP_KERNEL


def _kinematics_arrays(
    inputs: L1TransportInputs,
    timing,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """向量化梯形轨迹（与 ``_kinematics`` 逐式同构，返回位置/速度/加速度）。"""
    if inputs.control_waveform is not None:
        control = inputs.control_waveform
        times_ms = np.asarray(times, dtype=float) * 1e3
        return (
            np.interp(times_ms, control.time_ms, control.position_m),
            np.interp(times_ms, control.time_ms, control.velocity_m_s),
            np.interp(times_ms, control.time_ms, control.acceleration_m_s2),
        )
    if inputs.kinematic_profile == "minimum_jerk":
        ta = timing.acceleration_time_s
        cruise_end = ta + timing.cruise_time_s
        total = timing.total_time_s
        velocity_max = timing.maximum_velocity_m_s
        launch_u = np.clip(times / ta, 0.0, 1.0)
        remaining = total - times
        brake_u = np.clip(remaining / ta, 0.0, 1.0)

        def shapes(u):
            velocity_shape = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            acceleration_shape = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
            position_shape = 2.5 * u**4 - 3.0 * u**5 + u**6
            return position_shape, velocity_shape, acceleration_shape

        launch_x, launch_v, launch_a = shapes(launch_u)
        brake_x, brake_v, brake_a = shapes(brake_u)
        acceleration_distance = 0.5 * velocity_max * ta
        position = np.where(
            times < ta,
            velocity_max * ta * launch_x,
            np.where(
                times < cruise_end,
                acceleration_distance + velocity_max * (times - ta),
                inputs.distance_m - velocity_max * ta * brake_x,
            ),
        )
        velocity = np.where(
            times < ta,
            velocity_max * launch_v,
            np.where(times < cruise_end, velocity_max, velocity_max * brake_v),
        )
        accel = np.where(
            times < ta,
            velocity_max / ta * launch_a,
            np.where(
                times < cruise_end,
                0.0,
                -velocity_max / ta * brake_a,
            ),
        )
        return position, velocity, accel
    ta = timing.acceleration_time_s
    cruise_end = ta + timing.cruise_time_s
    acceleration = inputs.acceleration_m_s2
    velocity_max = timing.maximum_velocity_m_s
    acceleration_distance = 0.5 * acceleration * ta**2
    remaining = timing.total_time_s - times

    position = np.where(
        times < ta,
        0.5 * acceleration * times**2,
        np.where(
            times < cruise_end,
            acceleration_distance + velocity_max * (times - ta),
            inputs.distance_m - 0.5 * acceleration * remaining**2,
        ),
    )
    velocity = np.where(
        times < ta,
        acceleration * times,
        np.where(times < cruise_end, velocity_max, acceleration * remaining),
    )
    accel = np.where(
        times < ta,
        acceleration,
        np.where(times < cruise_end, 0.0, -acceleration),
    )
    return position, velocity, accel


def _leg_coefficient_table(
    z_lattice: np.ndarray,
    step_times_s: np.ndarray,
    inputs: L1TransportInputs,
    source_power_w: float,
    profile,
    wave_number: float,
    potential_per_intensity: float,
) -> np.ndarray:
    """逐点逐步的 13 个双束系数表 ``(步, 13)``（与 host 单点同式）。"""
    intensity1 = np.interp(z_lattice, profile.position_m, profile.intensity1_w_m2)
    intensity2 = np.interp(z_lattice, profile.position_m, profile.intensity2_w_m2)
    waist1 = np.interp(z_lattice, profile.position_m, profile.waist1_m)
    waist2 = np.interp(z_lattice, profile.position_m, profile.waist2_m)
    if inputs.control_waveform is not None:
        control = inputs.control_waveform
        waist_control = control.sample_optional_array("waist_um", step_times_s)
        source_scale = control.sample_optional_array(
            "source_power_scale", step_times_s
        )
        delivery_scale = control.sample_optional_array(
            "delivery_efficiency_scale", step_times_s
        )
        if waist_control is not None:
            waist1 = waist_control * 1e-6
            waist2 = waist1.copy()
        if any(
            values is not None
            for values in (waist_control, source_scale, delivery_scale)
        ):
            source = np.interp(
                z_lattice, profile.position_m, profile.source_power_w
            )
            if source_scale is not None:
                source = source_power_w * source_scale
            delivery = inputs.delivery_efficiency * (
                1.0 if delivery_scale is None else delivery_scale
            )
            intensity1 = 2.0 * source * delivery / (math.pi * waist1**2)
            intensity2 = inputs.retro_power_ratio * intensity1
    geometric_mean = np.sqrt(intensity1 * intensity2)
    return np.stack(
        (
            -2.0 / waist1**2,
            -2.0 / waist2**2,
            -(1.0 / waist1**2 + 1.0 / waist2**2),
            intensity1,
            intensity2,
            potential_per_intensity * intensity1,
            potential_per_intensity * intensity2,
            potential_per_intensity * 2.0 * geometric_mean,
            potential_per_intensity * 2.0 * geometric_mean * 2.0 * wave_number,
            potential_per_intensity * 4.0 * intensity1 / waist1**2,
            potential_per_intensity * 4.0 * intensity2 / waist2**2,
            potential_per_intensity
            * 4.0
            * geometric_mean
            * (1.0 / waist1**2 + 1.0 / waist2**2),
            np.full_like(intensity1, 2.0 * wave_number),
        ),
        axis=1,
    )


class _LegPointContext:
    """一个网格点的 host 侧常量（逐点组装 trace 时复用）。"""

    def __init__(
        self,
        detuning_ghz: float,
        source_power_w: float,
        wavelength_nm: float,
        wave_number: float,
        potential_per_intensity: float,
        scattering_per_intensity: float,
        profile,
        start_source_power: float,
        handover_depth_uK: float,
        handover_scattering_rate_s: float,
        feasible: bool,
        inputs: L1TransportInputs,
    ) -> None:
        self.detuning_ghz = detuning_ghz
        self.source_power_w = source_power_w
        self.wavelength_nm = wavelength_nm
        self.wave_number = wave_number
        self.potential_per_intensity = potential_per_intensity
        self.scattering_per_intensity = scattering_per_intensity
        self.profile = profile
        self.start_source_power = start_source_power
        self.handover_depth_uK = handover_depth_uK
        self.handover_scattering_rate_s = handover_scattering_rate_s
        self.feasible = feasible
        self.inputs = inputs

    def optics_at(
        self, position_m: float, time_s: float
    ) -> tuple[float, float, float, float, float, float]:
        return _leg_optics_at(
            self.inputs,
            self.profile,
            self.source_power_w,
            position_m,
            time_s,
        )

    def axial_modulation_at(self, position_m: float, time_s: float) -> float:
        """z 处轴向调制深度 U_ax = |C_U|·4√(I₁I₂)（host 标量）。"""
        i1, i2, _, _, _, _ = self.optics_at(position_m, time_s)
        return self.potential_per_intensity * 4.0 * math.sqrt(i1 * i2)

    def radial_gravity_at(
        self, position_m: float, time_s: float
    ) -> tuple[float, float]:
        """Return the radial downhill barrier and sagged minimum potential."""
        i1, i2, _, _, waist_um, _ = self.optics_at(position_m, time_s)
        depth = self.potential_per_intensity * (
            i1 + i2 + 2.0 * math.sqrt(i1 * i2)
        )
        if not self.inputs.include_gravity:
            return depth, -depth
        barrier, minimum, _ = gaussian_gravity_trap(
            depth, waist_um * 1e-6, _atom_from_label(self.inputs.atom_label).mass_kg
        )
        return barrier, minimum


def _prepare_leg_point(
    inputs: L1TransportInputs,
    detuning_ghz: float,
    source_power_w: float,
    initial_ensemble: ParticleEnsemble | None = None,
):
    """逐点 host 前处理：光学、可行性、初态采样、t=0 势与力。

    与 ``simulate_leg_monte_carlo`` 同口径；采样失败返回
    ``(context, None)``（调用方按既有口径给零留存 trace）。
    """
    atom = _atom_from_label(inputs.atom_label)
    mass = atom.mass_kg
    wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning_ghz)
    wave_number = 2.0 * math.pi / (wavelength_nm * 1e-9)
    forward_power_w = source_power_w * inputs.delivery_efficiency

    unit_dipole = scalar_potential_and_scattering(atom, wavelength_nm, 1.0)
    potential_per_intensity = abs(unit_dipole.potential_j)
    scattering_per_intensity = unit_dipole.scattering_rate_s

    profile = _leg_optics_profile(inputs, wavelength_nm, source_power_w)

    # 端点量与可行性检查沿用解析腿同一口径（批量路径不支持 conveyor，
    # 已在一致性校验拦截）。
    timing = l1_timing(inputs)
    end_i1, _, _, _, end_waist_um, _ = _leg_optics_at(
        inputs,
        profile,
        source_power_w,
        inputs.distance_m,
        timing.total_time_s,
    )
    lattice = evaluate_lattice(
        atom,
        wavelength_nm,
        forward_power_w=(math.pi * (end_waist_um * 1e-6) ** 2 * end_i1 / 2.0),
        waist_um=end_waist_um,
        retro_power_ratio=inputs.retro_power_ratio,
    )
    start_source_power = _leg_optics_at(
        inputs, profile, source_power_w, 0.0, 0.0
    )[-1]
    handover_depth_uK = lattice.depth_uK
    handover_scattering_rate_s = lattice.scattering_rate_s
    minimum_critical_acceleration = lattice.critical_axial_acceleration_m_s2
    feasible = (
        (
            not inputs.require_minimum_depth
            or handover_depth_uK >= inputs.target_depth_uK
        )
        and (
            not inputs.require_maximum_start_power
            or start_source_power <= inputs.maximum_l1_source_power_w
        )
        and (
            not inputs.require_critical_acceleration
            or minimum_critical_acceleration > inputs.acceleration_m_s2
        )
    )
    context = _LegPointContext(
        detuning_ghz,
        source_power_w,
        wavelength_nm,
        wave_number,
        potential_per_intensity,
        scattering_per_intensity,
        profile,
        start_source_power,
        handover_depth_uK,
        handover_scattering_rate_s,
        feasible,
        inputs,
    )

    (
        intensity1_0,
        intensity2_0,
        waist1_0,
        waist2_0,
        _,
        _,
    ) = context.optics_at(0.0, 0.0)
    axial_modulation_0 = (
        potential_per_intensity * 4.0 * math.sqrt(intensity1_0 * intensity2_0)
    )
    # 初态采样与逐点调用同 seed 逐位一致。
    rng = np.random.default_rng(inputs.mc_seed)
    try:
        if initial_ensemble is None:
            positions, velocities = _sample_initial_ensemble(
                particle_count=inputs.mc_particle_count,
                atom_mass_kg=mass,
                temperature_uK=inputs.initial_temperature_uK,
                intensity1_w_m2=intensity1_0,
                intensity2_w_m2=intensity2_0,
                waist1_m=waist1_0,
                waist2_m=waist2_0,
                axial_modulation_j=axial_modulation_0,
                wave_number_m=wave_number,
                potential_per_intensity_j=potential_per_intensity,
                cloud_axial_sigma_mm=inputs.mc_cloud_axial_sigma_mm,
                include_gravity=inputs.include_gravity,
                rng=rng,
            )
        else:
            propagated = initial_ensemble.resampled(
                inputs.mc_particle_count, inputs.mc_seed
            )
            positions, velocities, _ = propagated.host_arrays()
    except ValueError:
        return context, None
    # t=0 的势与力在 host 用同一函数计算（z_L(0)=0、φ=0）。
    potential0, force0, _, _ = _double_beam_potential_and_force(
        positions,
        intensity1_w_m2=intensity1_0,
        intensity2_w_m2=intensity2_0,
        waist1_m=waist1_0,
        waist2_m=waist2_0,
        wave_number_m=wave_number,
        lattice_position_m=0.0,
        phase_rad=0.0,
        potential_per_intensity_j=potential_per_intensity,
    )
    if inputs.include_gravity:
        force0[:, 1] -= mass * GRAVITY
    return context, (positions, velocities, potential0, force0)


def _point_stats_at_snapshot(
    positions_h: np.ndarray,
    velocities_h: np.ndarray,
    potential_h: np.ndarray,
    scatter_h: np.ndarray,
    point_index_h: np.ndarray,
    point_count: int,
    lattice_velocity: float,
    axial_modulation: np.ndarray,
    effective_barrier: np.ndarray,
    radial_barrier: np.ndarray,
    radial_minimum: np.ndarray,
    mass: float,
    include_gravity: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """快照时刻的逐点统计：动能温度（去质心共动系）、bound 比例、
    逐点累计散射（含已被剔除粒子的部分由调用方加 base）。

    无幸存者的点温度为 NaN、bound 为 0（与逐点 ``_record`` 同口径）。
    """
    relative = velocities_h.copy()
    relative[:, 2] -= lattice_velocity
    alive_counts = np.bincount(
        point_index_h, minlength=point_count
    ).astype(float)
    temperature = np.full(point_count, np.nan)
    bound = np.zeros(point_count)
    if point_index_h.size:
        # 逐点质心速度（幸存者子样本）。
        means = np.stack(
            [
                np.bincount(
                    point_index_h,
                    weights=relative[:, axis],
                    minlength=point_count,
                )
                for axis in range(3)
            ],
            axis=1,
        ) / np.maximum(alive_counts, 1.0)[:, None]
        centered = relative - means[point_index_h]
        mean_speed2 = np.bincount(
            point_index_h,
            weights=(centered * centered).sum(axis=1),
            minlength=point_count,
        ) / np.maximum(alive_counts, 1.0)
        temperature = (
            mass * mean_speed2 / (3.0 * BOLTZMANN) * 1e6
        )
        temperature = np.where(alive_counts > 0, temperature, np.nan)
        kinetic = 0.5 * mass * (relative * relative).sum(axis=1)
        axial_excitation = (
            kinetic
            + potential_h
            + axial_modulation[point_index_h]
        )
        bound_mask = axial_excitation < effective_barrier[point_index_h]
        if include_gravity:
            radial_excitation = (
                kinetic
                + potential_h
                + mass * GRAVITY * positions_h[:, 1]
                - radial_minimum[point_index_h]
            )
            bound_mask &= radial_excitation < radial_barrier[point_index_h]
            bound_mask &= radial_barrier[point_index_h] > 0.0
        bound_counts = np.bincount(
            point_index_h,
            weights=bound_mask.astype(float),
            minlength=point_count,
        )
        bound = bound_counts / np.maximum(alive_counts, 1.0)
    scattering = np.bincount(
        point_index_h, weights=scatter_h, minlength=point_count
    )
    return temperature, bound, scattering


def _assemble_leg_trace(
    inputs: L1TransportInputs,
    timing,
    context: _LegPointContext,
    snapshot_times: np.ndarray,
    temperature_series: np.ndarray,
    bound_series: np.ndarray,
    scattering_series: np.ndarray,
    alive_series: np.ndarray,
    time_step_s: float,
) -> L1TransportTrace:
    """由快照序列组装逐点 L1TransportTrace（与逐点 ``_record`` 同口径）。"""
    atom = _atom_from_label(inputs.atom_label)
    mass = atom.mass_kg
    particle_count = inputs.mc_particle_count
    wavelength_m = context.wavelength_nm * 1e-9

    time_out: list[float] = []
    stage_out: list[str] = []
    position_out: list[float] = []
    velocity_out: list[float] = []
    acceleration_out: list[float] = []
    frequency_out: list[float] = []
    waist_out: list[float] = []
    power_out: list[float] = []
    barrier_out: list[float] = []
    temperature_out: list[float] = []
    retention_out: list[float] = []
    bound_out: list[float] = []
    scattering_out: list[float] = []
    loss_rate_out: list[float] = []
    previous_alive = particle_count
    previous_time = 0.0
    for snap_index, grid_time in enumerate(snapshot_times):
        grid_time = float(grid_time)
        position, velocity, acceleration, stage = _kinematics(
            min(grid_time, timing.total_time_s), inputs, timing
        )
        _, _, _, _, waist_um, source_power_w = context.optics_at(
            position, grid_time
        )
        axial_modulation = context.axial_modulation_at(position, grid_time)
        critical = axial_modulation * context.wave_number / mass
        barrier_fraction = tilted_lattice_barrier_fraction(
            acceleration, critical
        )
        effective_barrier_j = axial_modulation * barrier_fraction
        if inputs.include_gravity:
            radial_barrier_j, _ = context.radial_gravity_at(
                position, grid_time
            )
            effective_barrier_j = min(
                effective_barrier_j, radial_barrier_j
            )
        alive_now = int(alive_series[snap_index])
        interval = grid_time - previous_time
        lost = previous_alive - alive_now
        loss_rate = (
            lost / previous_alive / interval
            if lost > 0 and previous_alive > 0 and interval > 0.0
            else 0.0
        )
        time_out.append(grid_time * 1e3)
        stage_out.append(stage)
        position_out.append(position)
        velocity_out.append(velocity)
        acceleration_out.append(acceleration)
        frequency_out.append(
            2.0 * velocity / wavelength_m * 1e-6
            if inputs.control_waveform is None
            else float(
                inputs.control_waveform.sample(grid_time)[
                    "aom_frequency_difference_mhz"
                ]
            )
        )
        waist_out.append(waist_um)
        power_out.append(source_power_w)
        barrier_out.append(effective_barrier_j / BOLTZMANN * 1e6)
        temperature_out.append(float(temperature_series[snap_index]))
        retention_out.append(alive_now / particle_count)
        bound_out.append(float(bound_series[snap_index]))
        scattering_out.append(float(scattering_series[snap_index]))
        loss_rate_out.append(loss_rate)
        previous_alive = alive_now
        previous_time = grid_time

    survivor_count = int(alive_series[-1])
    final_retention = survivor_count / particle_count
    # Jeffreys Beta(1/2, 1/2) 后验标准差在 k=0 或 k=N 时仍为有限值。
    posterior_alpha = survivor_count + 0.5
    posterior_beta = particle_count - survivor_count + 0.5
    posterior_sum = posterior_alpha + posterior_beta
    standard_error = math.sqrt(
        posterior_alpha
        * posterior_beta
        / (posterior_sum**2 * (posterior_sum + 1.0))
    )
    final_temperature = temperature_out[-1]
    point = L1DesignPoint(
        detuning_ghz=context.detuning_ghz,
        handover_source_power_w=context.source_power_w,
        start_source_power_w=context.start_source_power,
        wavelength_nm=context.wavelength_nm,
        depth_uK=context.handover_depth_uK,
        scattering_rate_s=context.handover_scattering_rate_s,
        final_temperature_uK=final_temperature,
        final_temperature_rise_uK=(
            final_temperature - inputs.initial_temperature_uK
        ),
        final_retention_fraction=final_retention,
        total_retention_from_mot_fraction=(
            inputs.loading_efficiency * final_retention
        ),
        final_atom_number=inputs.initial_atom_number * final_retention,
        cumulative_scattering_events=scattering_out[-1],
        maximum_loss_rate_s=max(loss_rate_out),
        feasible_hardware_point=context.feasible,
        initial_temperature_uK=inputs.initial_temperature_uK,
        initial_atom_number=inputs.initial_atom_number,
        actual_time_step_us=time_step_s * 1e6,
    )
    return L1TransportTrace(
        point=point,
        time_ms=tuple(time_out),
        stage=tuple(stage_out),
        position_m=tuple(position_out),
        velocity_m_s=tuple(velocity_out),
        acceleration_m_s2=tuple(acceleration_out),
        aom_frequency_difference_mhz=tuple(frequency_out),
        waist_um=tuple(waist_out),
        source_power_w=tuple(power_out),
        effective_barrier_uK=tuple(barrier_out),
        temperature_uK=tuple(temperature_out),
        temperature_rise_uK=tuple(
            value - inputs.initial_temperature_uK for value in temperature_out
        ),
        retention_fraction=tuple(retention_out),
        bound_fraction=tuple(bound_out),
        cumulative_scattering_events=tuple(scattering_out),
        instantaneous_loss_rate_s=tuple(loss_rate_out),
        retention_standard_error=standard_error,
    )


def _run_leg_gpu_chunk(
    chunk,
    inputs,
    progress=None,
    initial_ensembles=None,
    return_final_ensembles: bool = False,
):
    """在 GPU 上一次性推进一批网格点的全部运输腿粒子。

    ``chunk`` 为 ``[(index, inputs, detuning, power), ...]``（inputs 已
    校验一致；时序/光学/数值参数取自共享 ``inputs``，初态字段逐点取
    各自值）。返回 ``{index: L1TransportTrace}``；采样失败的点按既有
    口径给零留存 trace。
    """
    import cupy as cp

    xp = cp
    atom = _atom_from_label(inputs.atom_label)
    mass = atom.mass_kg
    particle_count = inputs.mc_particle_count
    include_scattering = inputs.mc_include_scattering
    timing = l1_timing(inputs)
    total_time = timing.total_time_s
    # 步长经 ω_z·dt ≤ 1 精度守卫钳制（transport_mc._stable_leg_step_s，
    # 与 CPU 逐点路径同一判据）；分块内各点失谐/功率不同，取最严
    # （最深轴向调制）者，保证全批共享步长逐位一致。
    stable_step_s = math.inf
    for _, point_inputs, point_detuning, point_power in chunk:
        point_wavelength_nm = atom.laser_wavelength_red_of_d1_nm(point_detuning)
        point_profile = _leg_optics_profile(
            point_inputs, point_wavelength_nm, point_power
        )
        stable_step_s = min(
            stable_step_s,
            _stable_leg_step_s(
                point_inputs, atom, point_wavelength_nm, point_profile
            ),
        )
    requested_step_s = min(inputs.transport_time_step_us * 1e-6, stable_step_s)
    integration_steps = max(1, math.ceil(total_time / requested_step_s))
    time_step_s = total_time / integration_steps
    snapshot_times = _time_grid(inputs, timing)

    # 逐步运动学（全批共享时序，向量化预计算）。
    step_times = np.minimum(
        time_step_s * np.arange(1, integration_steps + 1), total_time
    )
    z_lattice, lattice_velocity, acceleration = _kinematics_arrays(
        inputs, timing, step_times
    )

    # 1. 逐点 host 前处理（光学/可行性/初态采样/t=0 势与力）。
    contexts: dict[int, _LegPointContext] = {}
    sampled: list[tuple[int, L1TransportInputs, _LegPointContext, tuple]] = []
    results: dict[int, L1TransportTrace] = {}
    final_ensembles: dict[object, ParticleEnsemble | None] = {}
    for index, point_inputs, detuning, power in chunk:
        context, ensemble = _prepare_leg_point(
            point_inputs,
            detuning,
            power,
            None if initial_ensembles is None else initial_ensembles.get(index),
        )
        contexts[index] = context
        if ensemble is None:
            # 浅阱/高温点几乎无束缚初态：零留存 trace，不进批量数组。
            results[index] = _zero_retention_trace(
                point_inputs,
                timing,
                context.profile,
                context.potential_per_intensity,
                detuning,
                power,
                context.wavelength_nm,
                context.start_source_power,
                context.handover_depth_uK,
                context.handover_scattering_rate_s,
                context.feasible,
                particle_count,
            )
            final_ensembles[index] = None
        else:
            sampled.append((index, point_inputs, context, ensemble))

    point_count = len(sampled)
    if point_count == 0:
        return (results, final_ensembles) if return_final_ensembles else results

    # 2. 系数表 (步, 13, 点) 与逐粒子常量；逐点初态拼接上传。
    positions_h = np.concatenate([item[3][0] for item in sampled], axis=0)
    velocities_h = np.concatenate([item[3][1] for item in sampled], axis=0)
    potential0_h = np.concatenate([item[3][2] for item in sampled])
    force_h = np.concatenate([item[3][3] for item in sampled], axis=0)
    table_h = np.stack(
        [
            _leg_coefficient_table(
                z_lattice,
                step_times,
                point_inputs,
                context.source_power_w,
                context.profile,
                context.wave_number,
                context.potential_per_intensity,
            )
            for _, point_inputs, context, _ in sampled
        ],
        axis=2,
    )
    recoil_h = np.concatenate(
        [
            np.full(particle_count, HBAR * context.wave_number / mass)
            for _, _, context, _ in sampled
        ]
    )
    scattering_coeff_h = np.concatenate(
        [
            np.full(particle_count, context.scattering_per_intensity)
            for _, _, context, _ in sampled
        ]
    )
    kick_seed = int(
        np.random.SeedSequence([int(inputs.mc_seed) for _ in sampled])
        .generate_state(1, dtype=np.uint64)[0]
        % 2**63
    )
    kick_rng = _scattering_rng_gpu(kick_seed)

    positions = xp.asarray(positions_h)
    velocities = xp.asarray(velocities_h)
    force = xp.asarray(force_h)
    table = xp.asarray(table_h)
    point_index = xp.asarray(
        np.repeat(np.arange(point_count), particle_count), dtype=xp.int32
    )
    point_index_h = np.repeat(np.arange(point_count), particle_count)
    recoil = xp.asarray(recoil_h)
    scattering_coeff = xp.asarray(scattering_coeff_h)
    scatter_counts = xp.zeros(positions.shape[0], dtype=xp.int64)
    scatter_base = np.zeros(point_count)

    # 3. t=0 快照（host 上的初态统计，与逐点 ``_record(0)`` 同口径）。
    axial_modulation0 = np.asarray(
        [context.axial_modulation_at(0.0, 0.0) for _, _, context, _ in sampled]
    )
    critical0 = axial_modulation0 * np.asarray(
        [context.wave_number for _, _, context, _ in sampled]
    ) / mass
    barrier0 = axial_modulation0 * np.asarray(
        [
            tilted_lattice_barrier_fraction(0.0, critical)
            for critical in critical0
        ]
    )
    radial0 = np.asarray(
        [context.radial_gravity_at(0.0, 0.0) for _, _, context, _ in sampled]
    )
    temperature0, bound0, _ = _point_stats_at_snapshot(
        positions_h,
        velocities_h,
        potential0_h,
        np.zeros(positions_h.shape[0]),
        point_index_h,
        point_count,
        0.0,
        axial_modulation0,
        barrier0,
        radial0[:, 0],
        radial0[:, 1],
        mass,
        inputs.include_gravity,
    )
    snapshot_count = len(snapshot_times)
    temperature_series = np.full((snapshot_count, point_count), np.nan)
    bound_series = np.zeros((snapshot_count, point_count))
    scattering_series = np.zeros((snapshot_count, point_count))
    alive_series = np.zeros((snapshot_count, point_count), dtype=np.int64)
    temperature_series[0] = temperature0
    bound_series[0] = bound0
    alive_series[0] = particle_count

    # 4. mega-step 主循环：优先设备端时间循环（段间才与 host 交互——
    #    进度、快照、逃逸剔除；逐步的 Python 调度、kernel 启动与散射
    #    标量同步全部消除，这是长步数运输腿的主要墙钟成本）；kernel
    #    编译失败时回退逐步融合 kernel（行为与既有路径一致）。
    half_dt_over_mass = 0.5 * time_step_s / mass
    potential = xp.zeros(positions.shape[0], dtype=xp.float64)
    snapshot_index = 1
    progress_stride = max(1, integration_steps // 20)
    # 边界步集合：逃逸剔除（每 200 步+末步）∪ 快照步（首个满足
    # step*dt >= t_s - 1e-12 的步，与原逐步 while 记录同一步）。
    escape_steps = set(
        range(
            _ESCAPE_CHECK_INTERVAL_STEPS,
            integration_steps + 1,
            _ESCAPE_CHECK_INTERVAL_STEPS,
        )
    )
    escape_steps.add(integration_steps)
    snapshot_step_set = set()
    for grid_time in snapshot_times[1:]:
        snapshot_step_set.add(
            min(
                integration_steps,
                max(
                    1,
                    math.ceil((float(grid_time) - 1e-12) / time_step_s),
                ),
            )
        )
    boundaries = sorted(escape_steps | snapshot_step_set)
    loop_kernels = get_leg_loop_kernels()
    rng_states = None
    if loop_kernels is not None:
        init_kernel, steps_kernel = loop_kernels
        rng_states = allocate_rng_states(
            xp, init_kernel, positions.shape[0], kick_seed
        )
        z_lattice_device = xp.asarray(z_lattice)
    else:
        step_kernel = _get_fused_batch_leg_step_kernel()
    p0 = positions[:, 0]
    p1 = positions[:, 1]
    p2 = positions[:, 2]
    v0 = velocities[:, 0]
    v1 = velocities[:, 1]
    v2 = velocities[:, 2]
    f0 = force[:, 0]
    f1 = force[:, 1]
    f2 = force[:, 2]
    last_report = 0
    previous = 0
    for boundary in boundaries:
        if loop_kernels is not None:
            total_now = positions.shape[0]
            grid, block = launch_config(total_now)
            steps_kernel(
                grid,
                block,
                (
                    positions,
                    velocities,
                    force,
                    potential,
                    table,
                    z_lattice_device,
                    point_index,
                    scattering_coeff,
                    recoil,
                    scatter_counts,
                    rng_states,
                    np.float64(time_step_s),
                    np.float64(half_dt_over_mass),
                    np.float64(-mass * GRAVITY if inputs.include_gravity else 0.0),
                    np.int64(point_count),
                    np.int64(previous),
                    np.int64(boundary),
                    np.int64(total_now),
                    np.int32(1 if include_scattering else 0),
                ),
            )
        else:
            for step in range(previous + 1, boundary + 1):
                # 本步系数行 (13, P) 按 point_index gather 成逐粒子 (13, M)。
                coefficients = table[step - 1][:, point_index]
                potential, local_forward, local_incoherent = step_kernel(
                    p0,
                    p1,
                    p2,
                    v0,
                    v1,
                    v2,
                    f0,
                    f1,
                    f2,
                    coefficients[0],
                    coefficients[1],
                    coefficients[2],
                    coefficients[3],
                    coefficients[4],
                    coefficients[5],
                    coefficients[6],
                    coefficients[7],
                    coefficients[8],
                    coefficients[9],
                    coefficients[10],
                    coefficients[11],
                    coefficients[12],
                    float(z_lattice[step - 1]),
                    -mass * GRAVITY if inputs.include_gravity else 0.0,
                    half_dt_over_mass,
                    time_step_s,
                )

                if include_scattering:
                    _scattering_kicks_gpu(
                        velocities,
                        shape1=local_incoherent,
                        coefficient1_s=scattering_coeff,
                        time_step_s=time_step_s,
                        axis2_0=0.0,
                        axis2_1=0.0,
                        axis2_2=1.0,
                        forward_probability=(local_forward / local_incoherent),
                        recoil_m_s=recoil,
                        rng=kick_rng,
                        accumulate_counts=scatter_counts,
                    )

        if progress is not None and (
            boundary - last_report >= progress_stride
            or boundary == integration_steps
        ):
            progress(f"GPU 批量运输腿积分 {boundary}/{integration_steps}")
            last_report = boundary

        if boundary in escape_steps:
            # 逐点轴向调制/临界加速度/倾斜势垒（host），再逐粒子 gather。
            potential_cc = table_h[boundary - 1, 7]
            axial_modulation = 2.0 * potential_cc
            critical = axial_modulation * np.asarray(
                [context.wave_number for _, _, context, _ in sampled]
            ) / mass
            barrier = axial_modulation * np.asarray(
                [
                    tilted_lattice_barrier_fraction(
                        float(acceleration[boundary - 1]), critical_value
                    )
                    for critical_value in critical
                ]
            )
            radial = np.asarray(
                [
                    context.radial_gravity_at(
                        float(z_lattice[boundary - 1]),
                        boundary * time_step_s,
                    )
                    for _, _, context, _ in sampled
                ]
            )
            relative_z = v2 - float(lattice_velocity[boundary - 1])
            kinetic = (
                0.5
                * mass
                * (v0 * v0 + v1 * v1 + relative_z * relative_z)
            )
            axial_excitation = (
                kinetic
                + potential
                + xp.asarray(axial_modulation)[point_index]
            )
            alive = axial_excitation < xp.asarray(barrier)[point_index]
            if inputs.include_gravity:
                radial_excitation = (
                    kinetic
                    + potential
                    + mass * GRAVITY * positions[:, 1]
                    - xp.asarray(radial[:, 1])[point_index]
                )
                alive &= radial_excitation < xp.asarray(radial[:, 0])[point_index]
                alive &= xp.asarray(radial[:, 0])[point_index] > 0.0
            if not bool(xp.all(alive)):
                alive_h = alive.get()
                scatter_h = scatter_counts.get()
                for point_number in range(point_count):
                    culled = (~alive_h) & (point_index_h == point_number)
                    scatter_base[point_number] += scatter_h[culled].sum()
                positions = positions[alive]
                velocities = velocities[alive]
                force = force[alive]
                potential = potential[alive]
                point_index = point_index[alive]
                point_index_h = point_index_h[alive_h]
                scatter_counts = scatter_counts[alive]
                recoil = recoil[alive]
                scattering_coeff = scattering_coeff[alive]
                if rng_states is not None:
                    rng_states = rng_states[alive]
                if not positions.shape[0]:
                    break
                p0 = positions[:, 0]
                p1 = positions[:, 1]
                p2 = positions[:, 2]
                v0 = velocities[:, 0]
                v1 = velocities[:, 1]
                v2 = velocities[:, 2]
                f0 = force[:, 0]
                f1 = force[:, 1]
                f2 = force[:, 2]

        time_s = boundary * time_step_s
        while (
            snapshot_index < snapshot_count
            and snapshot_times[snapshot_index] <= time_s + 1e-12
        ):
            # 快照：粒子状态取回 host 逐点统计（元数据在组装阶段按
            # _kinematics 重算，与逐点 ``_record`` 同式）。
            grid_time = float(snapshot_times[snapshot_index])
            axial_modulation_g = np.asarray(
                [
                    context.axial_modulation_at(
                        float(
                            np.interp(
                                min(grid_time, total_time),
                                step_times,
                                z_lattice,
                            )
                        ),
                        min(grid_time, total_time),
                    )
                    for _, _, context, _ in sampled
                ]
            )
            critical_g = axial_modulation_g * np.asarray(
                [context.wave_number for _, _, context, _ in sampled]
            ) / mass
            velocity_g = float(
                np.interp(min(grid_time, total_time), step_times, lattice_velocity)
            )
            acceleration_g = float(
                np.interp(min(grid_time, total_time), step_times, acceleration)
            )
            barrier_g = axial_modulation_g * np.asarray(
                [
                    tilted_lattice_barrier_fraction(acceleration_g, critical_value)
                    for critical_value in critical_g
                ]
            )
            position_g = float(
                np.interp(min(grid_time, total_time), step_times, z_lattice)
            )
            radial_g = np.asarray(
                [
                    context.radial_gravity_at(position_g, grid_time)
                    for _, _, context, _ in sampled
                ]
            )
            temperature, bound, scattering = _point_stats_at_snapshot(
                positions.get(),
                velocities.get(),
                potential.get(),
                scatter_counts.get(),
                point_index_h,
                point_count,
                velocity_g,
                axial_modulation_g,
                barrier_g,
                radial_g[:, 0],
                radial_g[:, 1],
                mass,
                inputs.include_gravity,
            )
            temperature_series[snapshot_index] = temperature
            bound_series[snapshot_index] = bound
            scattering_series[snapshot_index] = scatter_base + scattering
            alive_series[snapshot_index] = np.bincount(
                point_index_h, minlength=point_count
            )
            snapshot_index += 1
        previous = boundary

    # 末步浮点舍入导致快照未记满时，用末态补齐（与逐点路径的填充
    # ``_record`` 同口径）；提前全灭时剩余快照保持 NaN/0。
    if snapshot_index < snapshot_count:
        if positions.shape[0]:
            axial_modulation_end = np.asarray(
                [
                    context.axial_modulation_at(
                        float(z_lattice[-1]), total_time
                    )
                    for _, _, context, _ in sampled
                ]
            )
            critical_end = axial_modulation_end * np.asarray(
                [context.wave_number for _, _, context, _ in sampled]
            ) / mass
            barrier_end = axial_modulation_end * np.asarray(
                [
                    tilted_lattice_barrier_fraction(
                        float(acceleration[-1]), critical_value
                    )
                    for critical_value in critical_end
                ]
            )
            radial_end = np.asarray(
                [
                    context.radial_gravity_at(float(z_lattice[-1]), total_time)
                    for _, _, context, _ in sampled
                ]
            )
            temperature, bound, scattering = _point_stats_at_snapshot(
                positions.get(),
                velocities.get(),
                potential.get(),
                scatter_counts.get(),
                point_index_h,
                point_count,
                float(lattice_velocity[-1]),
                axial_modulation_end,
                barrier_end,
                radial_end[:, 0],
                radial_end[:, 1],
                mass,
                inputs.include_gravity,
            )
            alive_end = np.bincount(point_index_h, minlength=point_count)
            for remaining in range(snapshot_index, snapshot_count):
                temperature_series[remaining] = temperature
                bound_series[remaining] = bound
                scattering_series[remaining] = scatter_base + scattering
                alive_series[remaining] = alive_end
        else:
            for remaining in range(snapshot_index, snapshot_count):
                scattering_series[remaining] = scatter_base

    # 5. 逐点组装 L1TransportTrace（初态字段取各点自身 inputs）。
    for point_number, (index, point_inputs, context, _) in enumerate(sampled):
        results[index] = _assemble_leg_trace(
            point_inputs,
            timing,
            context,
            snapshot_times,
            temperature_series[:, point_number],
            bound_series[:, point_number],
            scattering_series[:, point_number] / particle_count,
            alive_series[:, point_number],
            time_step_s,
        )
    if return_final_ensembles:
        positions_end = positions.get()
        velocities_end = velocities.get()
        for point_number, (index, _, _, _) in enumerate(sampled):
            selected = point_index_h == point_number
            final_ensembles[index] = (
                None
                if not np.any(selected)
                else ParticleEnsemble(
                    positions_m=positions_end[selected].copy(),
                    velocities_m_s=velocities_end[selected].copy(),
                    frame="transport_lab",
                )
            )
        return results, final_ensembles
    return results


def _chunk_leg_tasks(tasks, particle_count: int, integration_steps: int):
    """按粒子数与系数表显存上限把网格点分块。"""
    by_particles = max(1, _MAX_BATCH_PARTICLES // particle_count)
    table_bytes_per_point = integration_steps * len(_COEFF_COLUMNS) * 8
    by_table = max(1, _MAX_COEFF_TABLE_BYTES // table_bytes_per_point)
    points_per_chunk = max(1, min(by_particles, by_table))
    return [
        tasks[start : start + points_per_chunk]
        for start in range(0, len(tasks), points_per_chunk)
    ]


def run_leg_monte_carlo_batch(
    tasks,
    *,
    backend: str = "gpu",
    progress=None,
    initial_ensembles: dict[object, ParticleEnsemble] | None = None,
    return_final_ensembles: bool = False,
) -> (
    list[L1TransportTrace]
    | tuple[list[L1TransportTrace], list[ParticleEnsemble | None]]
):
    """批量运行多个网格点的运输腿 Monte Carlo，按 tasks 顺序返回 trace。

    ``tasks`` 为 ``[(index, L1TransportInputs, detuning_ghz, power), ...]``；
    所有点的 inputs 除初态字段（``initial_temperature_uK``/
    ``initial_atom_number``/``mot_atom_number``，供全链路 L2 腿逐点
    使用 handover 捕获样本）外必须全等（否则抛 ``ValueError`` 回退
    逐点），conveyor 几何未覆盖（同样抛错回退）。``backend="gpu"`` 时全部点的
    粒子摊平后在 GPU 上用一个 mega-step kernel 同时推进（自动分块保护
    显存）；``backend="cpu"`` 时退化为逐点调用
    ``simulate_leg_monte_carlo``（结果与直接逐点调用逐位一致）。
    ``progress`` 给定时每个分块及积分过程周期性报告（消息含
    ``n/total`` 供 UI 解析）。
    """
    tasks = list(tasks)
    if not tasks:
        raise ValueError("批量运输腿的任务列表不能为空")
    if backend not in {"cpu", "gpu"}:
        raise ValueError("计算后端必须是 cpu 或 gpu")
    _check_leg_consistency(tasks)

    def _run_cpu():
        outputs = []
        for completed, (index, point_inputs, detuning, power) in enumerate(
            tasks, start=1
        ):
            outputs.append(
                simulate_leg_monte_carlo(
                    point_inputs,
                    detuning,
                    power,
                    initial_ensemble=(
                        None
                        if initial_ensembles is None
                        else initial_ensembles.get(index)
                    ),
                    return_final_ensemble=return_final_ensembles,
                )
            )
            if progress is not None:
                progress(f"CPU 逐点运输腿 {completed}/{len(tasks)}")
        if return_final_ensembles:
            return (
                [output[0] for output in outputs],
                [output[1] for output in outputs],
            )
        return outputs

    if backend == "cpu":
        return _run_cpu()
    try:
        _resolve_backend("gpu")
        inputs = tasks[0][1]
        timing = l1_timing(inputs)
        # 分块显存估算的步数必须与 _run_leg_gpu_chunk 实际步数一致：
        # 同样经 ω_z·dt ≤ 1 精度守卫钳制，取全批各点最严者。
        atom = _atom_from_label(inputs.atom_label)
        stable_step_s = math.inf
        for _, point_inputs, point_detuning, point_power in tasks:
            point_wavelength_nm = atom.laser_wavelength_red_of_d1_nm(
                point_detuning
            )
            point_profile = _leg_optics_profile(
                point_inputs, point_wavelength_nm, point_power
            )
            stable_step_s = min(
                stable_step_s,
                _stable_leg_step_s(
                    point_inputs, atom, point_wavelength_nm, point_profile
                ),
            )
        requested_step_s = min(
            inputs.transport_time_step_us * 1e-6, stable_step_s
        )
        integration_steps = max(
            1,
            math.ceil(timing.total_time_s / requested_step_s),
        )
        chunks = _chunk_leg_tasks(
            tasks, inputs.mc_particle_count, integration_steps
        )
        traces: dict[object, L1TransportTrace] = {}
        ensembles: dict[object, ParticleEnsemble | None] = {}
        for chunk_index, chunk in enumerate(chunks, start=1):
            if progress is not None and len(chunks) > 1:
                progress(f"GPU 批量运输腿分块 {chunk_index}/{len(chunks)}")
            chunk_output = _run_leg_gpu_chunk(
                chunk,
                inputs,
                progress=progress,
                initial_ensembles=initial_ensembles,
                return_final_ensembles=return_final_ensembles,
            )
            if return_final_ensembles:
                chunk_traces, chunk_ensembles = chunk_output
                traces.update(chunk_traces)
                ensembles.update(chunk_ensembles)
            else:
                traces.update(chunk_output)
        ordered_traces = [traces[index] for index, _, _, _ in tasks]
        if return_final_ensembles:
            return (
                ordered_traces,
                [ensembles[index] for index, _, _, _ in tasks],
            )
        return ordered_traces
    except Exception as exc:  # noqa: BLE001 - GPU 不可用/内核失败回退 CPU
        if progress is not None:
            progress(
                f"GPU 批量运输腿不可用（{exc}），回退 CPU 逐点计算"
            )
        return _run_cpu()
