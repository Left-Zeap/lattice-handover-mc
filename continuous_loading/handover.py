"""双光晶格交接的三维经典轨迹 Monte Carlo 模型。

模型直接积分时变双晶格势中的经典轨迹，并把交接率定义为
Lattice-1 关闭后仍束缚在 Lattice-2 有效势垒内的原子比例。它适用
于本文参数下 ``U/E_r >> 1``、单格点热占据较高的情形。

主函数 ``run_handover_monte_carlo`` 的执行顺序是：

1. 建立两条晶格轴和空间错位；
2. 从 Lattice-1 中仍束缚的热分布采样初始位置和速度；
3. 令 L1 深度线性降到零、L2 深度线性升到满功率；
4. 用 velocity-Verlet 逐步传播全部经典轨迹，并可全程叠加沿 -y 的重力；
5. 可选地按局域散射率加入随机光子反冲；
6. 在终点用 Lattice-2 总激发能以及轴向加速/径向重力中的较低势垒判断捕获；
7. 只对最终捕获子样本计算交接前后等效温度和净升温。

默认假设两套晶格的光学交叉项已由频差、偏振或时间平均消除，因此
总势能是两个偶极势之和。若实验中存在稳定的相干交叉干涉，需要改
用总电场计算势能。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable

import numpy as np

from .constants import BOLTZMANN, GRAVITY, HBAR
from .control_waveforms import HandoverControlWaveform
from .gpu_backend import (
    module_of as _module_of,
    resolve_backend as _resolve_backend,
    rng_standard_normal as _rng_standard_normal,
    scatter_add as _scatter_add,
    scattering_kicks_gpu as _scattering_kicks_gpu,
    scattering_rng_gpu as _scattering_rng_gpu,
)
from .lattice import gaussian_gravity_trap, tilted_lattice_barrier_fraction
from .phase_space import ParticleEnsemble, canonicalize_lattice_phase


@dataclass(frozen=True)
class HandoverParameters:
    """一次双晶格交接的物理和数值参数。

    ``lattice1_distance_cm`` 与 ``optimal_distance_cm`` 的差定义原子云
    中心沿 Lattice-1 光轴相对理想交接点的位置。两晶格有夹角时，
    这个纵向误差会自动转换为相对 Lattice-2 光轴的横向失配。
    """

    atom_mass_kg: float
    wavelength_nm: float
    depth1_uK: float
    depth2_uK: float
    waist1_um: float
    waist2_um: float
    scattering_rate1_s: float = 0.0
    scattering_rate2_s: float = 0.0
    retro_power_ratio: float = 1.0
    initial_atom_number: float = 4_000_000.0
    temperature_uK: float = 20.0
    duration_ms: float = 1.0
    crossing_angle_deg: float = 4.0
    lattice1_distance_cm: float = 38.85
    optimal_distance_cm: float = 38.85
    cloud_axial_sigma_mm: float = 0.0
    l2_transverse_offset_um: float = 0.0
    relative_phase_rad: float = 0.0
    randomize_relative_phase: bool = True
    lattice1_velocity_m_s: float = 0.0
    lattice2_velocity_m_s: float = 0.0
    post_handover_acceleration_m_s2: float = 4_000.0
    # 竖直方向与运输腿一致为 -y；默认 False 保持独立 handover API 兼容。
    include_gravity: bool = False
    include_scattering: bool = True
    particle_count: int = 2_000
    time_step_us: float = 0.1
    trace_points: int = 51
    seed: int = 20_250_902
    compute_backend: str = "cpu"
    control_waveform: HandoverControlWaveform | None = None

    def __post_init__(self) -> None:
        positive = {
            "原子质量": self.atom_mass_kg,
            "波长": self.wavelength_nm,
            "Lattice-1 阱深": self.depth1_uK,
            "Lattice-2 阱深": self.depth2_uK,
            "Lattice-1 束腰": self.waist1_um,
            "Lattice-2 束腰": self.waist2_um,
            "温度": self.temperature_uK,
            "交接时间": self.duration_ms,
            "时间步长": self.time_step_us,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name}必须是有限正数")

        nonnegative = {
            "Lattice-1 散射率": self.scattering_rate1_s,
            "Lattice-2 散射率": self.scattering_rate2_s,
            "原子云轴向尺寸": self.cloud_axial_sigma_mm,
        }
        for name, value in nonnegative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name}必须是有限非负数")

        finite = {
            "夹角": self.crossing_angle_deg,
            "Lattice-1 距离": self.lattice1_distance_cm,
            "最佳交接距离": self.optimal_distance_cm,
            "Lattice-2 横向偏移": self.l2_transverse_offset_um,
            "相对相位": self.relative_phase_rad,
            "Lattice-1 速度": self.lattice1_velocity_m_s,
            "Lattice-2 速度": self.lattice2_velocity_m_s,
            "交接后加速度": self.post_handover_acceleration_m_s2,
        }
        for name, value in finite.items():
            if not math.isfinite(value):
                raise ValueError(f"{name}必须是有限数")

        if not 0.0 <= self.retro_power_ratio <= 1.0:
            raise ValueError("回程/前向功率比必须位于 [0, 1]")
        if (
            not math.isfinite(self.initial_atom_number)
            or self.initial_atom_number <= 0.0
        ):
            raise ValueError("Lattice-1 初始原子数必须是有限正数")
        if not 0.0 <= self.crossing_angle_deg < 180.0:
            raise ValueError("两晶格夹角必须位于 [0, 180) 度")
        if self.particle_count <= 0:
            raise ValueError("Monte Carlo 粒子数必须为正整数")
        if self.trace_points < 2:
            raise ValueError("轨迹记录点数至少为 2")
        if self.compute_backend not in {"cpu", "gpu"}:
            raise ValueError("计算后端必须是 cpu 或 gpu")
        if self.control_waveform is not None and not math.isclose(
            self.control_waveform.duration_ms,
            self.duration_ms,
            rel_tol=0.0,
            abs_tol=max(1e-9, 1e-6 * self.duration_ms),
        ):
            raise ValueError("handover 实测波形时长必须与 duration_ms 一致")


@dataclass(frozen=True)
class HandoverTrace:
    """交接期间的低带宽诊断轨迹。"""

    time_ms: tuple[float, ...]
    lattice1_fraction: tuple[float, ...]
    lattice2_fraction: tuple[float, ...]
    kinetic_temperature_uK: tuple[float, ...]
    captured_kinetic_temperature_uK: tuple[float | None, ...]
    cloud_rms_radius_um: tuple[float, ...]


@dataclass(frozen=True)
class HandoverResult:
    """一次经典轨迹 Monte Carlo 交接的汇总结果。"""

    parameters: HandoverParameters
    captured_count: int
    transfer_efficiency: float
    transfer_standard_error: float
    estimated_captured_atom_number: float
    estimated_captured_atom_number_standard_error: float
    sampled_initial_temperature_uK: float
    captured_initial_temperature_uK: float | None
    final_temperature_uK: float | None
    final_kinetic_temperature_uK: float | None
    handover_heating_uK: float | None
    all_atom_final_temperature_uK: float
    all_atom_handover_heating_uK: float
    mean_scattering_events: float
    recoil_heating_estimate_uK: float
    critical_acceleration_m_s2: float
    barrier_fraction: float
    effective_barrier_uK: float
    integration_steps: int
    actual_time_step_us: float
    trace: HandoverTrace


@dataclass(frozen=True)
class HandoverScanPoint:
    """单个扫描参数值及其 Monte Carlo 结果。"""

    parameter_name: str
    parameter_value: float
    result: HandoverResult


_SCANNABLE_PARAMETERS = {
    "duration_ms",
    "lattice1_distance_cm",
    "post_handover_acceleration_m_s2",
    "temperature_uK",
    "depth1_uK",
    "depth2_uK",
    "crossing_angle_deg",
    "cloud_axial_sigma_mm",
    "l2_transverse_offset_um",
    "lattice1_velocity_m_s",
    "lattice2_velocity_m_s",
    "relative_phase_rad",
}


def _unit_axes(angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 L1、L2 光轴和垂直于交叉平面的单位向量。"""
    angle = math.radians(angle_deg)
    e1 = np.array((0.0, 0.0, 1.0))
    e2 = np.array((math.sin(angle), 0.0, math.cos(angle)))
    e_out = np.array((0.0, 1.0, 0.0))
    return e1, e2, e_out


def _lattice_potential_force(
    positions_m: np.ndarray,
    *,
    axis: np.ndarray,
    beam_offset_m: np.ndarray,
    phase_rad: float | np.ndarray,
    axial_velocity_m_s: float,
    time_s: float,
    wave_number_m: float,
    waist_m: float,
    depth_j: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算高斯驻波势、力和相对局域光强。

    返回的相对光强 ``shape`` 在束轴波腹处为 1，可直接用于把峰值
    散射率缩放到每条轨迹所在位置。数组运算使用输入数组所属的后端
    （NumPy 或 CuPy），GPU 模式下粒子数组驻留 GPU，且整个势-力
    计算被 ``cupy.fuse`` 融合为单个 kernel 以降低启动开销。
    """
    xp = _module_of(positions_m)
    if xp is not np:
        kernel = _get_fused_lattice_kernel()
        # CuPy 14 + CUDA 13.3 + sm_120 的融合代码生成对"标量-标量子
        # 表达式再与数组组合"有 bug（CUDA_ERROR_NO_BINARY_FOR_GPU），
        # 所有标量系数一律在 host 预计算，kernel 内只剩数组-标量运算。
        potential, shape, f0, f1, f2 = kernel(
            positions_m,
            axis,
            beam_offset_m,
            phase_rad,
            axial_velocity_m_s * time_s,
            wave_number_m,
            depth_j,
            -2.0 / waist_m**2,
            4.0 / waist_m**2,
        )
        force = xp.empty_like(positions_m)
        force[:, 0] = f0
        force[:, 1] = f1
        force[:, 2] = f2
        return potential, force, shape
    displacement = positions_m - beam_offset_m
    # 行内点积用逐元素乘加实现（不依赖 cuBLAS，GPU 上更直接）。
    axial_coordinate = (displacement * axis).sum(axis=1)
    transverse = displacement - axial_coordinate[:, None] * axis
    transverse_radius2 = (transverse * transverse).sum(axis=1)
    envelope = xp.exp(-2.0 * transverse_radius2 / waist_m**2)
    phase = (
        wave_number_m
        * (axial_coordinate - axial_velocity_m_s * time_s)
        + phase_rad
    )
    cosine2 = xp.cos(phase) ** 2
    shape = envelope * cosine2
    potential = -depth_j * shape

    axial_term = (
        wave_number_m * xp.sin(2.0 * phase)
    )[:, None] * axis
    radial_term = 4.0 * cosine2[:, None] * transverse / waist_m**2
    force = -depth_j * envelope[:, None] * (axial_term + radial_term)
    return potential, force, shape


_FUSED_LATTICE_KERNEL = None


def _get_fused_lattice_kernel():
    """惰性创建势-力融合的 CuPy kernel（与上方 NumPy 路径逐式同构）。

    cupy.fuse 不支持归约与 newaxis，点积按分量显式展开；返回
    ``(V, shape, F_x, F_y, F_z)``，力数组由调用方组装。
    """
    global _FUSED_LATTICE_KERNEL
    if _FUSED_LATTICE_KERNEL is None:
        import cupy as cp

        @cp.fuse()
        def kernel(
            positions,
            axis,
            beam_offset,
            phase_rad,
            velocity_time,
            wave_number,
            depth,
            neg2_over_waist2,
            four_over_waist2,
        ):
            d0 = positions[:, 0] - beam_offset[0]
            d1 = positions[:, 1] - beam_offset[1]
            d2 = positions[:, 2] - beam_offset[2]
            axial_coordinate = d0 * axis[0] + d1 * axis[1] + d2 * axis[2]
            t0 = d0 - axial_coordinate * axis[0]
            t1 = d1 - axial_coordinate * axis[1]
            t2 = d2 - axial_coordinate * axis[2]
            radius2 = t0 * t0 + t1 * t1 + t2 * t2
            envelope = cp.exp(radius2 * neg2_over_waist2)
            phase = wave_number * (axial_coordinate - velocity_time) + phase_rad
            cosine2 = cp.cos(phase) ** 2
            shape = envelope * cosine2
            potential = -(depth * shape)
            sin2phase = cp.sin(2.0 * phase)
            b0 = wave_number * sin2phase * axis[0] + cosine2 * t0 * four_over_waist2
            b1 = wave_number * sin2phase * axis[1] + cosine2 * t1 * four_over_waist2
            b2 = wave_number * sin2phase * axis[2] + cosine2 * t2 * four_over_waist2
            return (
                potential,
                shape,
                -(depth * envelope * b0),
                -(depth * envelope * b1),
                -(depth * envelope * b2),
            )

        _FUSED_LATTICE_KERNEL = kernel
    return _FUSED_LATTICE_KERNEL


_FUSED_VERLET_STEP_KERNEL = None


def _get_fused_verlet_step_kernel():
    """惰性创建整步 velocity-Verlet 融合的 CuPy kernel（mega-step）。

    一次 kernel 完成：半步速度 → 整步位置 → 两晶格合力（新位置、新时
    刻）→ 半步速度，并就地更新 ``positions``/``velocities``/``force``
    （cupy.fuse 只支持 ``[...]`` 整体赋值，因此以列视图 ``p0..f2``
    传入 (M,3) 数组的三列）。返回新位置处的两条晶格相对光强
    ``(shape1, shape2)``，供散射反冲使用。与 CPU 路径逐式同构；所有
    随时间变化的标量系数（深度×斜坡分数、速度×时刻等）一律在 host
    预计算后作为单标量传入（规避 CuPy 14 + sm_120 的标量-标量子
    表达式融合 bug，见 ``_lattice_potential_force`` 注释）。
    """
    global _FUSED_VERLET_STEP_KERNEL
    if _FUSED_VERLET_STEP_KERNEL is None:
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
            phase2,
            phase_control,
            k_e2_0,
            k_e2_1,
            k_e2_2,
            e2_0,
            e2_1,
            e2_2,
            off2_0,
            off2_1,
            off2_2,
            depth1_now,
            depth2_now,
            neg2_w1,
            four_w1,
            neg2_w2,
            four_w2,
            wave_number,
            phase_shift1,
            phase_shift2,
            phase1,
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
            # Lattice-1：轴 (0,0,1)、束偏移为零。
            radius1 = np0 * np0 + np1 * np1
            envelope1 = cp.exp(radius1 * neg2_w1)
            ph1 = wave_number * (np2 - phase_shift1) + phase1
            cos1 = cp.cos(ph1) ** 2
            shape1 = envelope1 * cos1
            sin1 = cp.sin(2.0 * ph1)
            # Lattice-2：轴 e2、束偏移 off2。
            d0 = np0 - off2_0
            d1 = np1 - off2_1
            d2 = np2 - off2_2
            axial2 = d0 * e2_0 + d1 * e2_1 + d2 * e2_2
            t0 = d0 - axial2 * e2_0
            t1 = d1 - axial2 * e2_1
            t2 = d2 - axial2 * e2_2
            radius2 = t0 * t0 + t1 * t1 + t2 * t2
            envelope2 = cp.exp(radius2 * neg2_w2)
            ph2 = (
                wave_number * (axial2 - phase_shift2)
                + phase2
                + phase_control
            )
            cos2 = cp.cos(ph2) ** 2
            shape2 = envelope2 * cos2
            sin2 = cp.sin(2.0 * ph2)
            # 合力：-d·env·(k·sin2φ·axis + 4cos²φ·t/w²)（逐分量展开）。
            g0 = -(depth1_now * envelope1 * (cos1 * np0 * four_w1)) - (
                depth2_now * envelope2 * (k_e2_0 * sin2 + cos2 * t0 * four_w2)
            )
            g1 = -(depth1_now * envelope1 * (cos1 * np1 * four_w1)) - (
                depth2_now * envelope2 * (k_e2_1 * sin2 + cos2 * t1 * four_w2)
            ) + gravity_force_y
            g2 = -(depth1_now * envelope1 * (wave_number * sin1)) - (
                depth2_now * envelope2 * (k_e2_2 * sin2 + cos2 * t2 * four_w2)
            )
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
            return shape1, shape2

        _FUSED_VERLET_STEP_KERNEL = kernel
    return _FUSED_VERLET_STEP_KERNEL


def _kinetic_temperature_uK(
    velocities_m_s: np.ndarray,
    atom_mass_kg: float,
) -> float:
    """由去除质心速度后的三维动能方差计算温度。"""
    xp = _module_of(velocities_m_s)
    centered = velocities_m_s - xp.mean(velocities_m_s, axis=0)
    mean_speed2 = float((centered * centered).sum(axis=1).mean())
    return atom_mass_kg * mean_speed2 / (3.0 * BOLTZMANN) * 1e6


def _stable_handover_step_s(parameters: HandoverParameters) -> float:
    """Velocity-Verlet accuracy guard from the fastest axial trap mode.

    The mathematical stability bound is ``omega*dt < 2`` but values near it
    exhibit severe secular-energy drift.  ``omega*dt <= 1`` keeps the existing
    0.25 us default unchanged at the nominal handover point while preventing
    user-selected multi-microsecond steps from producing false temperature
    spikes.  The GPU path uses the same scalar bound before launching kernels.
    """
    wave_number = 2.0 * math.pi / (parameters.wavelength_nm * 1e-9)
    if parameters.control_waveform is None:
        effective_depth_uK = max(parameters.depth1_uK, parameters.depth2_uK)
    else:
        effective_depth_uK = max(
            parameters.depth1_uK * fraction1
            + parameters.depth2_uK * fraction2
            for fraction1, fraction2 in zip(
                parameters.control_waveform.lattice1_fraction,
                parameters.control_waveform.lattice2_fraction,
            )
        )
    axial_omega = math.sqrt(
        2.0
        * effective_depth_uK
        * 1e-6
        * BOLTZMANN
        * wave_number**2
        / parameters.atom_mass_kg
    )
    return 1.0 / axial_omega


def _sample_initial_ensemble(
    parameters: HandoverParameters,
    rng: np.random.Generator,
    *,
    e1: np.ndarray,
    wave_number_m: float,
    cloud_center_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从条件为仍束缚在 L1 中的谐振热分布采样。"""
    count = parameters.particle_count
    mass = parameters.atom_mass_kg
    depth_j = parameters.depth1_uK * 1e-6 * BOLTZMANN
    temperature_k = parameters.temperature_uK * 1e-6
    waist_m = parameters.waist1_um * 1e-6
    omega_radial = math.sqrt(4.0 * depth_j / (mass * waist_m**2))
    omega_axial = math.sqrt(2.0 * depth_j * wave_number_m**2 / mass)
    sigma_radial = math.sqrt(BOLTZMANN * temperature_k / mass) / omega_radial
    sigma_axial = math.sqrt(BOLTZMANN * temperature_k / mass) / omega_axial
    sigma_velocity = math.sqrt(BOLTZMANN * temperature_k / mass)
    initial_barrier_j = depth_j
    minimum_potential_j = -depth_j
    gravity_sag_m = 0.0
    if parameters.include_gravity:
        initial_barrier_j, minimum_potential_j, gravity_sag_m = (
            gaussian_gravity_trap(depth_j, waist_m, mass)
        )

    phase1 = -wave_number_m * float(cloud_center_m @ e1)
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    energies: list[np.ndarray] = []
    accepted = 0
    total_drawn = 0

    while accepted < count:
        batch = max(256, 2 * (count - accepted))
        local = rng.normal(size=(batch, 3))
        local[:, 0] *= sigma_radial
        local[:, 1] *= sigma_radial
        local[:, 1] += gravity_sag_m
        local[:, 2] *= sigma_axial
        if parameters.cloud_axial_sigma_mm > 0.0:
            lattice_spacing = math.pi / wave_number_m
            site_coordinate = rng.normal(
                scale=parameters.cloud_axial_sigma_mm * 1e-3,
                size=batch,
            )
            local[:, 2] += (
                np.rint(site_coordinate / lattice_spacing)
                * lattice_spacing
            )
        trial_positions = local + cloud_center_m
        trial_velocities = rng.normal(
            scale=sigma_velocity,
            size=(batch, 3),
        )
        trial_velocities += parameters.lattice1_velocity_m_s * e1

        potential, _, _ = _lattice_potential_force(
            trial_positions,
            axis=e1,
            beam_offset_m=np.zeros(3),
            phase_rad=phase1,
            axial_velocity_m_s=parameters.lattice1_velocity_m_s,
            time_s=0.0,
            wave_number_m=wave_number_m,
            waist_m=waist_m,
            depth_j=depth_j,
        )
        relative_velocity = (
            trial_velocities - parameters.lattice1_velocity_m_s * e1
        )
        kinetic = 0.5 * mass * np.einsum(
            "ij,ij->i",
            relative_velocity,
            relative_velocity,
        )
        if parameters.include_gravity:
            excitation = (
                kinetic
                + potential
                + mass * GRAVITY * trial_positions[:, 1]
                - minimum_potential_j
            )
            bound = (initial_barrier_j > 0.0) & (
                excitation < initial_barrier_j
            )
        else:
            excitation = kinetic + potential + depth_j
            bound = excitation < depth_j

        if np.any(bound):
            positions.append(trial_positions[bound])
            velocities.append(trial_velocities[bound])
            energies.append(excitation[bound])
            accepted += int(np.count_nonzero(bound))

        total_drawn += batch
        if total_drawn > 1_000 * count:
            raise ValueError(
                "当前温度/阱深下的初始束缚比例过低，无法稳定采样"
            )

    return (
        np.concatenate(positions, axis=0)[:count].copy(),
        np.concatenate(velocities, axis=0)[:count].copy(),
        np.concatenate(energies, axis=0)[:count].copy(),
    )


def _apply_scattering_kicks(
    velocities_m_s: np.ndarray,
    *,
    rate1_s: np.ndarray,
    rate2_s: np.ndarray,
    time_step_s: float,
    e1: np.ndarray,
    e2: np.ndarray,
    wave_number_m: float,
    atom_mass_kg: float,
    forward_absorption_probability: float,
    rng: np.random.Generator,
) -> int:
    """按局域散射率施加吸收和各向同性自发辐射反冲（后端无关）。"""
    xp = _module_of(velocities_m_s)
    total_rate = rate1_s + rate2_s
    counts = rng.poisson(total_rate * time_step_s)
    atom_indices = xp.repeat(xp.flatnonzero(counts), counts[counts > 0])
    event_count = int(atom_indices.size)
    if event_count == 0:
        return 0

    local_total = total_rate[atom_indices]
    choose_l1 = rng.random(event_count) < (
        rate1_s[atom_indices] / local_total
    )
    absorption_axes = xp.where(
        choose_l1[:, None],
        xp.asarray(e1),
        xp.asarray(e2),
    )
    absorption_sign = xp.where(
        rng.random(event_count) < forward_absorption_probability,
        1.0,
        -1.0,
    )
    emission_direction = _rng_standard_normal(rng, size=(event_count, 3))
    emission_direction /= xp.sqrt(
        (emission_direction * emission_direction).sum(axis=1)
    )[:, None]

    recoil_velocity = HBAR * wave_number_m / atom_mass_kg
    kicks = recoil_velocity * (
        absorption_sign[:, None] * absorption_axes - emission_direction
    )
    _scatter_add(xp, velocities_m_s, atom_indices, kicks)
    return event_count


def zero_capture_handover_result(parameters: HandoverParameters) -> HandoverResult:
    """采样失败（束缚初态比例过低）时的零捕获结果。

    语义：该参数点无法为轨迹建立 L1 束缚初态，全部采样原子视为在
    handover 环节丢失（``transfer_efficiency=0``、捕获子样本温度无
    定义），不传播任何轨迹。用于单点与批量扫描的容错：不让一个浅阱/
    高温网格点中断整个二维扫描。
    """
    count = parameters.particle_count
    mass = parameters.atom_mass_kg
    wave_number = 2.0 * math.pi / (parameters.wavelength_nm * 1e-9)
    depth2_j = parameters.depth2_uK * 1e-6 * BOLTZMANN
    waist2_m = parameters.waist2_um * 1e-6
    critical = depth2_j * wave_number / mass
    axial_fraction = tilted_lattice_barrier_fraction(
        parameters.post_handover_acceleration_m_s2, critical
    )
    gravity_barrier_j = depth2_j
    if parameters.include_gravity:
        gravity_barrier_j, _, _ = gaussian_gravity_trap(
            depth2_j, waist2_m, mass
        )
    effective_barrier_j = min(depth2_j * axial_fraction, gravity_barrier_j)
    duration_s = parameters.duration_ms * 1e-3
    requested_step_s = min(
        parameters.time_step_us * 1e-6, _stable_handover_step_s(parameters)
    )
    integration_steps = max(1, math.ceil(duration_s / requested_step_s))
    time_step_s = duration_s / integration_steps
    # Jeffreys Beta(1/2, 1/2) 后验：k=0 时仍为有限标准误。
    posterior_alpha = 0.5
    posterior_beta = count + 0.5
    posterior_sum = posterior_alpha + posterior_beta
    standard_error = math.sqrt(
        posterior_alpha
        * posterior_beta
        / (posterior_sum**2 * (posterior_sum + 1.0))
    )
    return HandoverResult(
        parameters=parameters,
        captured_count=0,
        transfer_efficiency=0.0,
        transfer_standard_error=standard_error,
        estimated_captured_atom_number=0.0,
        estimated_captured_atom_number_standard_error=0.0,
        sampled_initial_temperature_uK=float("nan"),
        captured_initial_temperature_uK=None,
        final_temperature_uK=None,
        final_kinetic_temperature_uK=None,
        handover_heating_uK=None,
        all_atom_final_temperature_uK=float("nan"),
        all_atom_handover_heating_uK=float("nan"),
        mean_scattering_events=0.0,
        recoil_heating_estimate_uK=0.0,
        critical_acceleration_m_s2=critical,
        barrier_fraction=(
            effective_barrier_j / depth2_j if depth2_j > 0.0 else 0.0
        ),
        effective_barrier_uK=effective_barrier_j / BOLTZMANN * 1e6,
        integration_steps=integration_steps,
        actual_time_step_us=time_step_s * 1e6,
        trace=HandoverTrace(
            time_ms=(0.0, parameters.duration_ms),
            lattice1_fraction=(1.0, 0.0),
            lattice2_fraction=(0.0, 1.0),
            kinetic_temperature_uK=(float("nan"), float("nan")),
            captured_kinetic_temperature_uK=(None, None),
            cloud_rms_radius_um=(0.0, 0.0),
        ),
    )


def run_handover_monte_carlo(
    parameters: HandoverParameters,
    *,
    initial_ensemble: ParticleEnsemble | None = None,
    return_captured_ensemble: bool = False,
) -> HandoverResult | tuple[HandoverResult, ParticleEnsemble | None]:
    """积分交接轨迹并返回 Lattice-2 捕获率和温升。

    最终捕获判据为原子相对 Lattice-2 阱底的总激发能低于考虑后续
    加速度倾斜后的最低轴向势垒。交接期间没有假设热平衡。
    """
    # 1. 统一单位，并建立 L1、L2 光轴和垂直于交叉平面的方向。
    backend = _resolve_backend(parameters.compute_backend)
    if backend == "gpu":
        import cupy as cp

        xp = cp
    else:
        xp = np
    # 初态采样与相对相位始终在 CPU 上用同一 NumPy RNG，保证与 CPU
    # 后端逐位一致的初态；散射反冲的 RNG 按后端创建（GPU 序列与
    # CPU 不同，结果仅统计一致）。
    rng = np.random.default_rng(parameters.seed)
    kick_rng = rng if xp is np else _scattering_rng_gpu(parameters.seed)
    e1, e2, e_out = _unit_axes(parameters.crossing_angle_deg)
    mass = parameters.atom_mass_kg
    wavelength_m = parameters.wavelength_nm * 1e-9
    wave_number = 2.0 * math.pi / wavelength_m
    depth1_j = parameters.depth1_uK * 1e-6 * BOLTZMANN
    depth2_j = parameters.depth2_uK * 1e-6 * BOLTZMANN
    waist1_m = parameters.waist1_um * 1e-6
    waist2_m = parameters.waist2_um * 1e-6
    duration_s = parameters.duration_ms * 1e-3

    # 2. 将运输距离误差和显式横向偏移转换成三维几何位置。
    distance_offset_m = (
        parameters.lattice1_distance_cm - parameters.optimal_distance_cm
    ) * 1e-2
    cloud_center = distance_offset_m * e1
    l2_beam_offset = (
        parameters.l2_transverse_offset_um * 1e-6 * e_out
    )
    phase1 = -wave_number * float(cloud_center @ e1)

    # 3. 先从 L1 谐振热分布抽样，再用完整周期势拒绝未束缚样本。
    if initial_ensemble is None:
        try:
            positions, velocities, initial_excitation = _sample_initial_ensemble(
                parameters,
                rng,
                e1=e1,
                wave_number_m=wave_number,
                cloud_center_m=cloud_center,
            )
        except ValueError:
            # 束缚初态比例过低（浅阱/高温）：全部采样原子视为在 handover
            # 环节丢失，返回零捕获结果，不让扫描因单点失败而中断。
            return zero_capture_handover_result(parameters)
    else:
        propagated = initial_ensemble.resampled(
            parameters.particle_count, parameters.seed
        )
        positions, velocities, _ = propagated.host_arrays()
        initial_potential, _, _ = _lattice_potential_force(
            positions,
            axis=e1,
            beam_offset_m=np.zeros(3),
            phase_rad=phase1,
            axial_velocity_m_s=parameters.lattice1_velocity_m_s,
            time_s=0.0,
            wave_number_m=wave_number,
            waist_m=waist1_m,
            depth_j=depth1_j,
        )
        initial_relative_velocity = (
            velocities - parameters.lattice1_velocity_m_s * e1
        )
        initial_excitation = (
            0.5
            * mass
            * (initial_relative_velocity * initial_relative_velocity).sum(axis=1)
            + initial_potential
            + depth1_j
        )
    if parameters.randomize_relative_phase:
        phase2: float | np.ndarray = (
            parameters.relative_phase_rad
            + rng.uniform(0.0, math.pi, parameters.particle_count)
        )
    else:
        phase2 = parameters.relative_phase_rad

    # GPU 后端：采样完成后把粒子状态与几何量一次性传入 GPU。
    if xp is not np:
        positions = xp.asarray(positions)
        velocities = xp.asarray(velocities)
        initial_excitation = xp.asarray(initial_excitation)
        if isinstance(phase2, np.ndarray):
            phase2 = xp.asarray(phase2)
        e1 = xp.asarray(e1)
        e2 = xp.asarray(e2)
        l2_beam_offset = xp.asarray(l2_beam_offset)
    zero3 = xp.zeros(3)

    # 4. 调整实际步长，使整数个时间步精确落在 handover 终点。
    requested_step_s = min(
        parameters.time_step_us * 1e-6,
        _stable_handover_step_s(parameters),
    )
    integration_steps = max(1, math.ceil(duration_s / requested_step_s))
    time_step_s = duration_s / integration_steps
    step_times_s = np.arange(integration_steps + 1, dtype=float) * time_step_s
    if parameters.control_waveform is None:
        fraction2_steps = step_times_s / duration_s
        fraction1_steps = 1.0 - fraction2_steps
        phase_control_steps = np.zeros_like(step_times_s)
    else:
        (
            fraction1_steps,
            fraction2_steps,
            phase_control_steps,
        ) = parameters.control_waveform.sampled_arrays(step_times_s)
    record_steps = np.unique(
        np.linspace(
            0,
            integration_steps,
            parameters.trace_points,
            dtype=int,
        )
    )
    record_lookup = set(int(step) for step in record_steps)
    trace_time: list[float] = []
    trace_l1: list[float] = []
    trace_l2: list[float] = []
    trace_temperature: list[float] = []
    trace_velocities: list[np.ndarray] = []
    trace_radius: list[float] = []

    def record(step: int) -> None:
        fraction1 = float(fraction1_steps[step])
        fraction2 = float(fraction2_steps[step])
        centered = positions - xp.mean(positions, axis=0)
        rms_radius = math.sqrt(
            float((centered * centered).sum(axis=1).mean())
        )
        trace_time.append(step * time_step_s * 1e3)
        trace_l1.append(fraction1)
        trace_l2.append(fraction2)
        trace_temperature.append(_kinetic_temperature_uK(velocities, mass))
        trace_velocities.append(velocities.copy())
        trace_radius.append(rms_radius * 1e6)

    def combined_force(
        time_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        step_index = min(
            integration_steps,
            max(0, int(round(time_s / time_step_s))),
        )
        fraction1 = float(fraction1_steps[step_index])
        fraction2 = float(fraction2_steps[step_index])
        phase_control = float(phase_control_steps[step_index])
        _, force1, shape1 = _lattice_potential_force(
            positions,
            axis=e1,
            beam_offset_m=zero3,
            phase_rad=phase1,
            axial_velocity_m_s=parameters.lattice1_velocity_m_s,
            time_s=time_s,
            wave_number_m=wave_number,
            waist_m=waist1_m,
            depth_j=depth1_j * fraction1,
        )
        _, force2, shape2 = _lattice_potential_force(
            positions,
            axis=e2,
            beam_offset_m=l2_beam_offset,
            phase_rad=phase2 + phase_control,
            axial_velocity_m_s=parameters.lattice2_velocity_m_s,
            time_s=time_s,
            wave_number_m=wave_number,
            waist_m=waist2_m,
            depth_j=depth2_j * fraction2,
        )
        force = force1 + force2
        if parameters.include_gravity:
            force[:, 1] -= mass * GRAVITY
        return force, shape1, shape2

    record(0)
    force, _, _ = combined_force(0.0)
    total_scattering_events = 0
    forward_probability = 1.0 / (1.0 + parameters.retro_power_ratio)

    # 5. Velocity-Verlet：半步速度、整步位置、新力、半步速度。
    if xp is np:
        for step in range(1, integration_steps + 1):
            velocities += 0.5 * time_step_s * force / mass
            positions += time_step_s * velocities
            time_end = step * time_step_s
            force, shape1, shape2 = combined_force(time_end)
            velocities += 0.5 * time_step_s * force / mass

            # 散射反冲在每个完整动力学步之后按局域 Poisson 事件加入。
            if parameters.include_scattering:
                fraction1 = float(fraction1_steps[step])
                fraction2 = float(fraction2_steps[step])
                rate1 = (
                    parameters.scattering_rate1_s
                    * fraction1
                    * shape1
                )
                rate2 = (
                    parameters.scattering_rate2_s
                    * fraction2
                    * shape2
                )
                total_scattering_events += _apply_scattering_kicks(
                    velocities,
                    rate1_s=rate1,
                    rate2_s=rate2,
                    time_step_s=time_step_s,
                    e1=e1,
                    e2=e2,
                    wave_number_m=wave_number,
                    atom_mass_kg=mass,
                    forward_absorption_probability=forward_probability,
                    rng=kick_rng,
                )

            if step in record_lookup:
                record(step)
    else:
        # GPU：整个 velocity-Verlet 步融合为单个 mega-step kernel
        # （就地更新粒子数组，每步一次 kernel 启动）；随时间变化的
        # 标量系数全部 host 预计算。
        step_kernel = _get_fused_verlet_step_kernel()
        e2_0 = float(e2[0])
        e2_1 = float(e2[1])
        e2_2 = float(e2[2])
        off2_0 = float(l2_beam_offset[0])
        off2_1 = float(l2_beam_offset[1])
        off2_2 = float(l2_beam_offset[2])
        neg2_w1 = -2.0 / waist1_m**2
        four_w1 = 4.0 / waist1_m**2
        neg2_w2 = -2.0 / waist2_m**2
        four_w2 = 4.0 / waist2_m**2
        half_dt_over_mass = 0.5 * time_step_s / mass
        p0 = positions[:, 0]
        p1 = positions[:, 1]
        p2 = positions[:, 2]
        v0 = velocities[:, 0]
        v1 = velocities[:, 1]
        v2 = velocities[:, 2]
        f0 = force[:, 0]
        f1 = force[:, 1]
        f2 = force[:, 2]
        scatter_counts = xp.zeros(parameters.particle_count, dtype=xp.int64)
        for step in range(1, integration_steps + 1):
            time_end = step * time_step_s
            fraction1 = float(fraction1_steps[step])
            fraction2 = float(fraction2_steps[step])
            phase_control = float(phase_control_steps[step])
            shape1, shape2 = step_kernel(
                p0,
                p1,
                p2,
                v0,
                v1,
                v2,
                f0,
                f1,
                f2,
                phase2,
                phase_control,
                wave_number * e2_0,
                wave_number * e2_1,
                wave_number * e2_2,
                e2_0,
                e2_1,
                e2_2,
                off2_0,
                off2_1,
                off2_2,
                depth1_j * fraction1,
                depth2_j * fraction2,
                neg2_w1,
                four_w1,
                neg2_w2,
                four_w2,
                wave_number,
                parameters.lattice1_velocity_m_s * time_end,
                parameters.lattice2_velocity_m_s * time_end,
                phase1,
                -mass * GRAVITY if parameters.include_gravity else 0.0,
                half_dt_over_mass,
                time_step_s,
            )

            # 散射反冲在每个完整动力学步之后按局域 Poisson 事件加入
            # （GPU 路径用固定事件槽融合实现，避免逐事件设备同步）。
            if parameters.include_scattering:
                _scattering_kicks_gpu(
                    velocities,
                    shape1=shape1,
                    coefficient1_s=(
                        parameters.scattering_rate1_s * fraction1
                    ),
                    shape2=shape2,
                    coefficient2_s=(
                        parameters.scattering_rate2_s * fraction2
                    ),
                    time_step_s=time_step_s,
                    axis2_0=e2_0,
                    axis2_1=e2_1,
                    axis2_2=e2_2,
                    forward_probability=forward_probability,
                    recoil_m_s=HBAR * wave_number / mass,
                    rng=kick_rng,
                    accumulate_counts=scatter_counts,
                )

            if step in record_lookup:
                record(step)

        total_scattering_events = int(scatter_counts.sum())

    # 6. 终点只剩 L2；能量在 L2 共动系中、相对阱底计算。
    potential2, _, _ = _lattice_potential_force(
        positions,
        axis=e2,
        beam_offset_m=l2_beam_offset,
        phase_rad=phase2 + float(phase_control_steps[-1]),
        axial_velocity_m_s=parameters.lattice2_velocity_m_s,
        time_s=duration_s,
        wave_number_m=wave_number,
        waist_m=waist2_m,
        depth_j=depth2_j,
    )
    relative_velocity2 = (
        velocities - parameters.lattice2_velocity_m_s * e2
    )
    kinetic2 = 0.5 * mass * (relative_velocity2 * relative_velocity2).sum(axis=1)
    final_excitation = kinetic2 + potential2 + depth2_j

    # 后续运输加速度不参与 handover 传播，只用于降低末态捕获势垒。
    critical_acceleration = depth2_j * wave_number / mass
    barrier_fraction = tilted_lattice_barrier_fraction(
        parameters.post_handover_acceleration_m_s2,
        critical_acceleration,
    )
    effective_barrier_j = depth2_j * barrier_fraction
    if parameters.include_gravity:
        gravity_barrier_j, minimum_potential_j, _ = gaussian_gravity_trap(
            depth2_j, waist2_m, mass
        )
        final_excitation = (
            kinetic2
            + potential2
            + mass * GRAVITY * positions[:, 1]
            - minimum_potential_j
        )
        effective_barrier_j = min(effective_barrier_j, gravity_barrier_j)
        barrier_fraction = effective_barrier_j / depth2_j
    captured = xp.isfinite(final_excitation) & (
        final_excitation < effective_barrier_j
    ) & (effective_barrier_j > 0.0)
    captured_count = int(xp.count_nonzero(captured))
    efficiency = captured_count / parameters.particle_count
    # Jeffreys Beta(1/2, 1/2) 后验标准差在 k=0 或 k=N 时仍给出有限
    # Monte Carlo 不确定度，避免普通 sqrt[p(1-p)/N] 错报为零。
    posterior_alpha = captured_count + 0.5
    posterior_beta = parameters.particle_count - captured_count + 0.5
    posterior_sum = posterior_alpha + posterior_beta
    standard_error = math.sqrt(
        posterior_alpha
        * posterior_beta
        / (posterior_sum**2 * (posterior_sum + 1.0))
    )

    # 7. 为消除选择偏差，净升温前后都使用“最终被捕获的同一子样本”。
    sampled_initial_temperature = (
        float(xp.mean(initial_excitation)) / (3.0 * BOLTZMANN) * 1e6
    )
    all_atom_final_temperature = (
        float(xp.mean(final_excitation)) / (3.0 * BOLTZMANN) * 1e6
    )
    all_atom_heating = (
        all_atom_final_temperature - sampled_initial_temperature
    )
    captured_initial_temperature: float | None
    final_temperature: float | None
    final_kinetic_temperature: float | None
    heating: float | None
    if captured_count:
        captured_initial_temperature = (
            float(xp.mean(initial_excitation[captured]))
            / (3.0 * BOLTZMANN)
            * 1e6
        )
        final_temperature = (
            float(xp.mean(final_excitation[captured]))
            / (3.0 * BOLTZMANN)
            * 1e6
        )
        captured_relative_velocity = relative_velocity2[captured]
        captured_relative_velocity -= xp.mean(
            captured_relative_velocity,
            axis=0,
        )
        final_kinetic_temperature = (
            mass
            * float(
                (
                    captured_relative_velocity
                    * captured_relative_velocity
                ).sum(axis=1).mean()
            )
            / (3.0 * BOLTZMANN)
            * 1e6
        )
        heating = final_temperature - captured_initial_temperature
    else:
        captured_initial_temperature = None
        final_temperature = None
        final_kinetic_temperature = None
        heating = None

    captured_trace_temperature: list[float | None] = []
    for recorded_velocities in trace_velocities:
        if captured_count:
            captured_trace_temperature.append(
                _kinetic_temperature_uK(recorded_velocities[captured], mass)
            )
        else:
            captured_trace_temperature.append(None)

    mean_scattering_events = (
        total_scattering_events / parameters.particle_count
    )
    recoil_energy_j = HBAR**2 * wave_number**2 / (2.0 * mass)
    recoil_heating = (
        mean_scattering_events
        * 2.0
        * recoil_energy_j
        / (3.0 * BOLTZMANN)
        * 1e6
    )

    result = HandoverResult(
        parameters=parameters,
        captured_count=captured_count,
        transfer_efficiency=efficiency,
        transfer_standard_error=standard_error,
        estimated_captured_atom_number=(
            parameters.initial_atom_number * efficiency
        ),
        estimated_captured_atom_number_standard_error=(
            parameters.initial_atom_number * standard_error
        ),
        sampled_initial_temperature_uK=sampled_initial_temperature,
        captured_initial_temperature_uK=captured_initial_temperature,
        final_temperature_uK=final_temperature,
        final_kinetic_temperature_uK=final_kinetic_temperature,
        handover_heating_uK=heating,
        all_atom_final_temperature_uK=all_atom_final_temperature,
        all_atom_handover_heating_uK=all_atom_heating,
        mean_scattering_events=mean_scattering_events,
        recoil_heating_estimate_uK=recoil_heating,
        critical_acceleration_m_s2=critical_acceleration,
        barrier_fraction=barrier_fraction,
        effective_barrier_uK=effective_barrier_j / BOLTZMANN * 1e6,
        integration_steps=integration_steps,
        actual_time_step_us=time_step_s * 1e6,
        trace=HandoverTrace(
            time_ms=tuple(trace_time),
            lattice1_fraction=tuple(trace_l1),
            lattice2_fraction=tuple(trace_l2),
            kinetic_temperature_uK=tuple(trace_temperature),
            captured_kinetic_temperature_uK=tuple(
                captured_trace_temperature
            ),
            cloud_rms_radius_um=tuple(trace_radius),
        ),
    )
    if not return_captured_ensemble:
        return result
    if captured_count == 0:
        return result, None
    positions_h = positions[captured]
    velocities_h = velocities[captured]
    final_phase = phase2 + float(phase_control_steps[-1])
    if np.isscalar(final_phase):
        # 固定相位口径（randomize_relative_phase=False）下 phase2 是标量，
        # 即使在 GPU 后端该数组也始终建在 host 上。
        captured_phase_h = np.full(captured_count, float(final_phase))
    else:
        captured_phase_h = final_phase[captured]
    if xp is not np:
        positions_h = positions_h.get()
        velocities_h = velocities_h.get()
        # captured_phase_h 可能是 host NumPy 数组（固定相位口径）或
        # 设备 CuPy 数组（随机相位口径），只有后者需要 .get() 下载。
        if isinstance(captured_phase_h, xp.ndarray):
            captured_phase_h = captured_phase_h.get()
        e2_h = e2.get()
        l2_beam_offset_h = l2_beam_offset.get()
    else:
        e2_h = e2
        l2_beam_offset_h = l2_beam_offset
    # 交接计算允许逐轨迹随机 L2 相位；L2 运输固定使用零相位。
    # 把 cos²[k(q-vt)+phi] 精确映射到目标坐标
    # q'=q-vt+phi/k，同时转入 L2 晶格共动系。若直接丢弃 phi，
    # 下一阶段等价于一次瞬时相位淬火，深阱 Cs 参数下会表现为极端
    # 温度尖刺。该变换只在阶段边界对捕获粒子做一次 O(N) 向量运算。
    return result, canonicalize_lattice_phase(
        ParticleEnsemble(
            positions_m=np.asarray(positions_h).copy(),
            velocities_m_s=np.asarray(velocities_h).copy(),
            frame="handover",
        ),
        phase_rad=np.asarray(captured_phase_h, dtype=float),
        wave_number_m=wave_number,
        axis=np.asarray(e2_h, dtype=float),
        beam_offset_m=np.asarray(l2_beam_offset_h, dtype=float),
        lattice_displacement_m=(
            parameters.lattice2_velocity_m_s * duration_s
        ),
        lattice_velocity_m_s=parameters.lattice2_velocity_m_s,
        frame="handover_l2_canonical",
    )


def scan_handover_parameter(
    parameters: HandoverParameters,
    parameter_name: str,
    values: Iterable[float],
) -> list[HandoverScanPoint]:
    """使用公共随机数扫描一个 handover 参数。

    每个扫描点沿用同一个随机种子，从而降低有限 Monte Carlo 样本给
    相邻点差值带来的噪声。
    """
    if parameter_name not in _SCANNABLE_PARAMETERS:
        supported = ", ".join(sorted(_SCANNABLE_PARAMETERS))
        raise ValueError(f"不支持扫描 {parameter_name}；可选参数：{supported}")

    points: list[HandoverScanPoint] = []
    for value in values:
        numeric_value = float(value)
        updates = {parameter_name: numeric_value}
        if parameter_name == "depth1_uK":
            updates["scattering_rate1_s"] = (
                parameters.scattering_rate1_s
                * numeric_value
                / parameters.depth1_uK
            )
        elif parameter_name == "depth2_uK":
            updates["scattering_rate2_s"] = (
                parameters.scattering_rate2_s
                * numeric_value
                / parameters.depth2_uK
            )
        point_parameters = replace(
            parameters,
            **updates,
        )
        points.append(
            HandoverScanPoint(
                parameter_name=parameter_name,
                parameter_value=numeric_value,
                result=run_handover_monte_carlo(point_parameters),
            )
        )
    if not points:
        raise ValueError("扫描值不能为空")
    return points
