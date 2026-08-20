"""扫描场景的批量 handover Monte Carlo：多网格点同时在 GPU 上积分。

``run_handover_monte_carlo_batch`` 把 P 个 ``HandoverParameters`` 的粒子
集合摊平成单个 ``(P×N, 3)`` 粒子数组，逐点参数（波数、阱深、束腰、
散射率、相对相位、L2 束偏移、晶格速度等）gather 成逐粒子数组，用一
个 mega-step 融合 kernel（与 ``handover._get_fused_verlet_step_kernel``
同型的整步 velocity-Verlet：半步速度 → 整步位置 → 新力 → 半步速度）
同时推进全部网格点的全部轨迹，避免逐点串行的 kernel 启动开销。

约束与口径：

- 同批要求 ``particle_count``、``duration_ms``、``time_step_us``、
  ``crossing_angle_deg``、``retro_power_ratio``、``trace_points`` 完全
  一致（扫描天然满足；扫描只改功率/失谐等逐点物理量）。不一致时抛
  ``ValueError``，调用方应回退为逐点调用 ``run_handover_monte_carlo``。
- 夹角一致意味着 e1/e2 光轴与回程比（前向吸收概率）全批共享；深度
  斜坡分数 ``(1-t/τ, t/τ)`` 因 duration 一致也是全批标量，只有逐点
  物理系数留在逐粒子数组里。
- 初态采样与相对相位仍在 CPU 上按各点 seed 用 NumPy RNG 逐点生成，
  与同 seed 的逐点调用逐位一致；散射反冲的 GPU RNG 全批共用一个
  （由各点 seed 派生），与逐点调用的 RNG 顺序不同，结果仅统计一致。
- 扫描只需要逐点汇总，因此 trace 只保留 t=0 与终点两个端点（即使
  ``trace_points > 2``），capture 判据、Jeffreys 标准误、温度类汇总
  与逐点调用同口径。
- 显存保护：粒子数组约 20 个 (P×N) 分量，P×N 超过
  ``_MAX_BATCH_PARTICLES`` 时自动分块循环。

``backend="cpu"`` 时退化为逐点调用现有 ``run_handover_monte_carlo``
（结果与直接逐点调用逐位一致）。
"""

from __future__ import annotations

import math

import numpy as np

from .constants import BOLTZMANN, GRAVITY, HBAR
from .device_loop import (
    allocate_rng_states,
    get_handover_loop_kernels,
    launch_config,
)
from .gpu_backend import (
    resolve_backend as _resolve_backend,
    scattering_kicks_gpu as _scattering_kicks_gpu,
    scattering_rng_gpu as _scattering_rng_gpu,
)
from .handover import (
    HandoverParameters,
    HandoverResult,
    HandoverTrace,
    _sample_initial_ensemble,
    _stable_handover_step_s,
    _unit_axes,
    run_handover_monte_carlo,
    zero_capture_handover_result,
)
from .lattice import gaussian_gravity_trap, tilted_lattice_barrier_fraction
from .phase_space import ParticleEnsemble, canonicalize_lattice_phase


# 单批粒子上限：P×N×8B×约 20 个分量 ≈ 320 MB，留足显存余量。
_MAX_BATCH_PARTICLES = 2_000_000

# 同批必须一致的数值/几何字段（扫描中不变；逐点物理量不在此列）。
_CONSISTENCY_FIELDS = (
    "particle_count",
    "duration_ms",
    "time_step_us",
    "crossing_angle_deg",
    "retro_power_ratio",
    "trace_points",
    "control_waveform",
    "include_gravity",
)


def _check_consistency(parameters: list[HandoverParameters]) -> None:
    """校验同批网格点共享的数值/几何参数完全一致。"""
    reference = parameters[0]
    for point_index, point in enumerate(parameters[1:], start=1):
        for field in _CONSISTENCY_FIELDS:
            if getattr(point, field) != getattr(reference, field):
                raise ValueError(
                    f"批量 handover 要求 {field} 全批一致：点 0 为 "
                    f"{getattr(reference, field)!r}，点 {point_index} 为 "
                    f"{getattr(point, field)!r}；请回退为逐点调用"
                )


_FUSED_BATCH_STEP_KERNEL = None


def _get_fused_batch_step_kernel():
    """惰性创建批量整步 velocity-Verlet 融合 kernel（mega-step）。

    与 ``handover._get_fused_verlet_step_kernel`` 逐式同构，但逐点物理
    系数（波数、阱深、束腰系数、晶格速度、L2 束偏移、半步系数、相对
    相位）全部是逐粒子数组；夹角一致所以 e1=(0,0,1)、e2 分量与深度
    斜坡分数是全批标量。kernel 内只出现 数组·数组 或 数组·标量 运算
    （规避 CuPy 14 + sm_120 的标量-标量子表达式融合 bug）。就地更新
    列视图 ``p0..f2``，返回两条晶格的相对光强 ``(shape1, shape2)``。
    """
    global _FUSED_BATCH_STEP_KERNEL
    if _FUSED_BATCH_STEP_KERNEL is None:
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
            phase1,
            phase2,
            wave_number,
            depth1_j,
            depth2_j,
            neg2_w1,
            four_w1,
            neg2_w2,
            four_w2,
            velocity1,
            velocity2,
            offset2_1,
            gravity_force_y,
            half_dt_over_mass,
            e2_0,
            e2_1,
            e2_2,
            fraction1,
            fraction2,
            phase_control,
            time_end,
            time_step,
        ):
            # 半步速度 + 整步位置（逐粒子半步系数）。
            nv0 = v0 + f0 * half_dt_over_mass
            nv1 = v1 + f1 * half_dt_over_mass
            nv2 = v2 + f2 * half_dt_over_mass
            np0 = p0 + nv0 * time_step
            np1 = p1 + nv1 * time_step
            np2 = p2 + nv2 * time_step
            # Lattice-1：轴 (0,0,1)、束偏移为零。
            radius1 = np0 * np0 + np1 * np1
            envelope1 = cp.exp(radius1 * neg2_w1)
            ph1 = wave_number * (np2 - velocity1 * time_end) + phase1
            cos1 = cp.cos(ph1) ** 2
            shape1 = envelope1 * cos1
            sin1 = cp.sin(2.0 * ph1)
            # Lattice-2：共享轴 e2，束偏移只有 y 分量（沿 e_out）。
            d1 = np1 - offset2_1
            axial2 = np0 * e2_0 + d1 * e2_1 + np2 * e2_2
            t0 = np0 - axial2 * e2_0
            t1 = d1 - axial2 * e2_1
            t2 = np2 - axial2 * e2_2
            radius2 = t0 * t0 + t1 * t1 + t2 * t2
            envelope2 = cp.exp(radius2 * neg2_w2)
            ph2 = (
                wave_number * (axial2 - velocity2 * time_end)
                + phase2
                + phase_control
            )
            cos2 = cp.cos(ph2) ** 2
            shape2 = envelope2 * cos2
            sin2 = cp.sin(2.0 * ph2)
            # 深度×斜坡分数在 kernel 内做 数组·标量（全批共享分数）。
            depth1_now = depth1_j * fraction1
            depth2_now = depth2_j * fraction2
            # L2 轴向项系数 k·e2_i 为 数组·标量（k 逐粒子、e2 共享）。
            g0 = -(depth1_now * envelope1 * (cos1 * np0 * four_w1)) - (
                depth2_now
                * envelope2
                * ((wave_number * e2_0) * sin2 + cos2 * t0 * four_w2)
            )
            g1 = -(depth1_now * envelope1 * (cos1 * np1 * four_w1)) - (
                depth2_now
                * envelope2
                * ((wave_number * e2_1) * sin2 + cos2 * t1 * four_w2)
            ) + gravity_force_y
            g2 = -(depth1_now * envelope1 * (wave_number * sin1)) - (
                depth2_now
                * envelope2
                * ((wave_number * e2_2) * sin2 + cos2 * t2 * four_w2)
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

        _FUSED_BATCH_STEP_KERNEL = kernel
    return _FUSED_BATCH_STEP_KERNEL


def _point_means(
    values: np.ndarray,
    point_index: np.ndarray,
    point_count: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """逐点均值（``mask`` 给定时的子样本均值；空子样本为 NaN）。"""
    weights = values if mask is None else np.where(mask, values, 0.0)
    sums = np.bincount(point_index, weights=weights, minlength=point_count)
    if mask is None:
        counts = np.bincount(point_index, minlength=point_count).astype(float)
    else:
        counts = np.bincount(
            point_index, weights=mask.astype(float), minlength=point_count
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts


def _centered_speed2(
    velocities: np.ndarray,
    point_index: np.ndarray,
    point_count: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """逐点去质心后的 |v|²（``mask`` 子样本；未选中的粒子填 0）。"""
    means = np.stack(
        [
            _point_means(velocities[:, axis], point_index, point_count, mask)
            for axis in range(3)
        ],
        axis=1,
    )
    centered = velocities - means[point_index]
    speed2 = (centered * centered).sum(axis=1)
    if mask is not None:
        speed2 = np.where(mask, speed2, 0.0)
    return speed2


def _run_gpu_chunk(
    chunk: list[HandoverParameters],
    progress=None,
    requested_step_s: float | None = None,
    initial_ensembles: list[ParticleEnsemble | None] | None = None,
    return_captured_ensembles: bool = False,
) -> (
    list[HandoverResult]
    | tuple[list[HandoverResult], list[ParticleEnsemble | None]]
):
    """在 GPU 上一次性推进一批网格点的全部粒子并逐点组装结果。"""
    import cupy as cp

    xp = cp
    reference = chunk[0]
    point_count = len(chunk)
    count = reference.particle_count
    duration_s = reference.duration_ms * 1e-3
    e1_np, e2_np, _ = _unit_axes(reference.crossing_angle_deg)
    e2_0, e2_1, e2_2 = (float(component) for component in e2_np)

    # 1. 逐点 CPU 初态采样（与逐点调用同 seed 逐位一致），并 gather
    #    出逐粒子参数数组。
    positions_list: list[np.ndarray] = []
    velocities_list: list[np.ndarray] = []
    excitation_list: list[np.ndarray] = []
    per_particle: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "wave_number",
            "depth1_j",
            "depth2_j",
            "neg2_w1",
            "four_w1",
            "neg2_w2",
            "four_w2",
            "velocity1",
            "velocity2",
            "offset2_1",
            "gravity_force_y",
            "gravity_minimum_j",
            "mass",
            "phase1",
            "phase2",
            "scattering1",
            "scattering2",
        )
    }
    barrier_fractions: list[float] = []
    critical_accelerations: list[float] = []
    # 采样失败（束缚初态比例过低）的点：记为"零捕获"，全部原子视为在
    # handover 环节丢失，不进批量数组；组装时按 chunk 顺序占位。
    zero_results: dict[int, HandoverResult] = {}
    successful_points: list[int] = []
    if initial_ensembles is not None and len(initial_ensembles) != point_count:
        raise ValueError('initial ensemble list must match handover batch size')
    for point_number, point in enumerate(chunk):
        mass = point.atom_mass_kg
        wavelength_m = point.wavelength_nm * 1e-9
        wave_number = 2.0 * math.pi / wavelength_m
        depth1_j = point.depth1_uK * 1e-6 * BOLTZMANN
        depth2_j = point.depth2_uK * 1e-6 * BOLTZMANN
        waist1_m = point.waist1_um * 1e-6
        waist2_m = point.waist2_um * 1e-6
        distance_offset_m = (
            point.lattice1_distance_cm - point.optimal_distance_cm
        ) * 1e-2
        cloud_center = distance_offset_m * e1_np

        rng = np.random.default_rng(point.seed)
        supplied = (
            None if initial_ensembles is None else initial_ensembles[point_number]
        )
        try:
            if supplied is None:
                positions, velocities, initial_excitation = _sample_initial_ensemble(
                    point,
                    rng,
                    e1=e1_np,
                    wave_number_m=wave_number,
                    cloud_center_m=cloud_center,
                )
            else:
                propagated = supplied.resampled(point.particle_count, point.seed)
                positions, velocities, _ = propagated.host_arrays()
                axial = positions @ e1_np
                transverse = positions - axial[:, None] * e1_np
                envelope = np.exp(
                    -2.0 * np.sum(transverse * transverse, axis=1) / waist1_m**2
                )
                phase = (
                    wave_number * axial
                    - wave_number * float(cloud_center @ e1_np)
                )
                potential = -depth1_j * envelope * np.cos(phase) ** 2
                relative_velocity = (
                    velocities - point.lattice1_velocity_m_s * e1_np
                )
                initial_excitation = (
                    0.5
                    * mass
                    * np.sum(relative_velocity * relative_velocity, axis=1)
                    + potential
                    + depth1_j
                )
        except ValueError:
            # 浅阱/高温点无法建立束缚初态：该点全部原子视为在 handover
            # 环节丢失（零捕获），不中断整批扫描。
            zero_results[point_number] = zero_capture_handover_result(point)
            continue
        if point.randomize_relative_phase:
            phase2 = point.relative_phase_rad + rng.uniform(
                0.0, math.pi, point.particle_count
            )
        else:
            phase2 = np.full(point.particle_count, point.relative_phase_rad)
        positions_list.append(positions)
        velocities_list.append(velocities)
        excitation_list.append(initial_excitation)
        successful_points.append(point_number)

        critical = depth2_j * wave_number / mass
        critical_accelerations.append(critical)
        axial_fraction = tilted_lattice_barrier_fraction(
            point.post_handover_acceleration_m_s2,
            critical,
        )
        gravity_barrier_j = depth2_j
        gravity_minimum_j = -depth2_j
        if point.include_gravity:
            gravity_barrier_j, gravity_minimum_j, _ = gaussian_gravity_trap(
                depth2_j, waist2_m, mass
            )
        effective_barrier_j = min(
            depth2_j * axial_fraction, gravity_barrier_j
        )
        barrier_fractions.append(effective_barrier_j / depth2_j)
        constants = {
            "wave_number": wave_number,
            "depth1_j": depth1_j,
            "depth2_j": depth2_j,
            "neg2_w1": -2.0 / waist1_m**2,
            "four_w1": 4.0 / waist1_m**2,
            "neg2_w2": -2.0 / waist2_m**2,
            "four_w2": 4.0 / waist2_m**2,
            "velocity1": point.lattice1_velocity_m_s,
            "velocity2": point.lattice2_velocity_m_s,
            "offset2_1": point.l2_transverse_offset_um * 1e-6,
            "gravity_force_y": (
                -mass * GRAVITY if point.include_gravity else 0.0
            ),
            "gravity_minimum_j": gravity_minimum_j,
            "mass": mass,
            "phase1": -wave_number * float(cloud_center @ e1_np),
            "scattering1": point.scattering_rate1_s,
            "scattering2": point.scattering_rate2_s,
        }
        for name, value in constants.items():
            per_particle[name].append(np.full(count, value))
        per_particle["phase2"].append(phase2)

    # 批量数组只包含采样成功的点；统计接口（_point_means 等）按成功点
    # 数量对齐。组装时用 local_by_point 把批量索引映射回 chunk 原编号。
    if not positions_list:
        # 全部点采样失败：所有原子视为在 handover 环节丢失。
        results = [zero_results[index] for index in range(point_count)]
        if return_captured_ensembles:
            return results, [None] * point_count
        return results
    point_count = len(successful_points)
    local_by_point = {
        point_number: local
        for local, point_number in enumerate(successful_points)
    }
    positions_h = np.concatenate(positions_list, axis=0)
    velocities_h = np.concatenate(velocities_list, axis=0)
    initial_excitation_h = np.concatenate(excitation_list, axis=0)
    arrays_h = {
        name: np.concatenate(values) for name, values in per_particle.items()
    }
    point_index_h = np.repeat(np.arange(point_count), count)
    total_particles = point_count * count

    # 2. 上传 GPU；散射 RNG 全批共用一个（由各点 seed 确定派生）。
    kick_seed = int(
        np.random.SeedSequence([int(point.seed) for point in chunk])
        .generate_state(1, dtype=np.uint64)[0]
        % 2**63
    )
    kick_rng = _scattering_rng_gpu(kick_seed)
    positions = xp.asarray(positions_h)
    velocities = xp.asarray(velocities_h)
    gpu_arrays = {name: xp.asarray(values) for name, values in arrays_h.items()}

    if requested_step_s is None:
        requested_step_s = min(
            reference.time_step_us * 1e-6,
            *(_stable_handover_step_s(point) for point in chunk),
        )
    integration_steps = max(1, math.ceil(duration_s / requested_step_s))
    time_step_s = duration_s / integration_steps
    step_times_s = np.arange(integration_steps + 1, dtype=float) * time_step_s
    if reference.control_waveform is None:
        fraction2_steps_h = step_times_s / duration_s
        fraction1_steps_h = 1.0 - fraction2_steps_h
        phase_control_steps_h = np.zeros_like(step_times_s)
    else:
        (
            fraction1_steps_h,
            fraction2_steps_h,
            phase_control_steps_h,
        ) = reference.control_waveform.sampled_arrays(step_times_s)
    fraction1_steps = xp.asarray(fraction1_steps_h)
    fraction2_steps = xp.asarray(fraction2_steps_h)
    phase_control_steps = xp.asarray(phase_control_steps_h)

    # 3. t=0 初始力（L2 深度为零，只有 L1 贡献；普通逐元素运算）。
    p0 = positions[:, 0]
    p1 = positions[:, 1]
    p2 = positions[:, 2]
    radius1 = p0 * p0 + p1 * p1
    envelope1 = xp.exp(radius1 * gpu_arrays["neg2_w1"])
    ph1 = gpu_arrays["wave_number"] * p2 + gpu_arrays["phase1"]
    cos1 = xp.cos(ph1) ** 2
    sin1 = xp.sin(2.0 * ph1)
    prefactor = gpu_arrays["depth1_j"] * envelope1
    force = xp.empty_like(positions)
    force[:, 0] = -(prefactor * (cos1 * p0 * gpu_arrays["four_w1"]))
    force[:, 1] = (
        -(prefactor * (cos1 * p1 * gpu_arrays["four_w1"]))
        + gpu_arrays["gravity_force_y"]
    )
    force[:, 2] = -(prefactor * (gpu_arrays["wave_number"] * sin1))

    # 4. mega-step 主循环：优先设备端时间循环（每段一次 kernel 启动，
    #    消除逐步 Python 调度/kernel 启动/散射标量同步的开销——长步数
    #    积分的主要墙钟成本）；编译失败时回退逐步融合 kernel（行为与
    #    既有路径一致）。
    forward_probability = 1.0 / (1.0 + reference.retro_power_ratio)
    recoil_per_particle = (
        HBAR * gpu_arrays["wave_number"] / gpu_arrays["mass"]
    )
    scatter_counts = xp.zeros(total_particles, dtype=xp.int64)
    # 进度节流：每块最多约 20 次步进报告（消息含 n/total 供 UI 解析）。
    progress_stride = max(1, integration_steps // 20)
    loop_kernels = get_handover_loop_kernels()
    if loop_kernels is not None:
        init_kernel, steps_kernel = loop_kernels
        rng_states = allocate_rng_states(
            xp, init_kernel, total_particles, kick_seed
        )
        grid, block = launch_config(total_particles)
        step = 0
        while step < integration_steps:
            segment_end = min(integration_steps, step + progress_stride)
            steps_kernel(
                grid,
                block,
                (
                    positions,
                    velocities,
                    force,
                    gpu_arrays["wave_number"],
                    gpu_arrays["depth1_j"],
                    gpu_arrays["depth2_j"],
                    gpu_arrays["neg2_w1"],
                    gpu_arrays["four_w1"],
                    gpu_arrays["neg2_w2"],
                    gpu_arrays["four_w2"],
                    gpu_arrays["velocity1"],
                    gpu_arrays["velocity2"],
                    gpu_arrays["offset2_1"],
                    gpu_arrays["mass"],
                    gpu_arrays["gravity_force_y"],
                    gpu_arrays["phase1"],
                    gpu_arrays["phase2"],
                    gpu_arrays["scattering1"],
                    gpu_arrays["scattering2"],
                    fraction1_steps,
                    fraction2_steps,
                    phase_control_steps,
                    scatter_counts,
                    rng_states,
                    np.float64(e2_0),
                    np.float64(e2_1),
                    np.float64(e2_2),
                    np.float64(time_step_s),
                    np.float64(duration_s),
                    np.float64(forward_probability),
                    np.int64(step),
                    np.int64(segment_end),
                    np.int64(total_particles),
                    np.int32(1 if reference.include_scattering else 0),
                ),
            )
            step = segment_end
            if progress is not None:
                progress(f"GPU 批量 handover 积分 {step}/{integration_steps}")
    else:
        step_kernel = _get_fused_batch_step_kernel()
        half_dt_over_mass = (
            0.5 * time_step_s / gpu_arrays["mass"]
        )
        v0 = velocities[:, 0]
        v1 = velocities[:, 1]
        v2 = velocities[:, 2]
        f0 = force[:, 0]
        f1 = force[:, 1]
        f2 = force[:, 2]
        for step in range(1, integration_steps + 1):
            if progress is not None and step % progress_stride == 0:
                progress(
                    f"GPU 批量 handover 积分 {step}/{integration_steps}"
                )
            time_end = step * time_step_s
            fraction1 = float(fraction1_steps_h[step])
            fraction2 = float(fraction2_steps_h[step])
            phase_control = float(phase_control_steps_h[step])
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
                gpu_arrays["phase1"],
                gpu_arrays["phase2"],
                gpu_arrays["wave_number"],
                gpu_arrays["depth1_j"],
                gpu_arrays["depth2_j"],
                gpu_arrays["neg2_w1"],
                gpu_arrays["four_w1"],
                gpu_arrays["neg2_w2"],
                gpu_arrays["four_w2"],
                gpu_arrays["velocity1"],
                gpu_arrays["velocity2"],
                gpu_arrays["offset2_1"],
                gpu_arrays["gravity_force_y"],
                half_dt_over_mass,
                e2_0,
                e2_1,
                e2_2,
                fraction1,
                fraction2,
                phase_control,
                time_end,
                time_step_s,
            )

            if reference.include_scattering:
                _scattering_kicks_gpu(
                    velocities,
                    shape1=shape1,
                    coefficient1_s=gpu_arrays["scattering1"] * fraction1,
                    shape2=shape2,
                    coefficient2_s=gpu_arrays["scattering2"] * fraction2,
                    time_step_s=time_step_s,
                    axis2_0=e2_0,
                    axis2_1=e2_1,
                    axis2_2=e2_2,
                    forward_probability=forward_probability,
                    recoil_m_s=recoil_per_particle,
                    rng=kick_rng,
                    accumulate_counts=scatter_counts,
                )

    # 5. 终点评估整体取回 host，用 NumPy 逐点汇总（与逐点调用同式）。
    positions_end = positions.get()
    velocities_end = velocities.get()
    del positions, velocities, force, gpu_arrays
    wave_number_h = arrays_h["wave_number"]
    depth2_h = arrays_h["depth2_j"]
    mass_h = arrays_h["mass"]
    axial = (
        positions_end[:, 0] * e2_0
        + (positions_end[:, 1] - arrays_h["offset2_1"]) * e2_1
        + positions_end[:, 2] * e2_2
    )
    t0 = positions_end[:, 0] - axial * e2_0
    t1 = positions_end[:, 1] - arrays_h["offset2_1"] - axial * e2_1
    t2 = positions_end[:, 2] - axial * e2_2
    radius2 = t0 * t0 + t1 * t1 + t2 * t2
    envelope2 = np.exp(radius2 * arrays_h["neg2_w2"])
    phase2_end = (
        wave_number_h * (axial - arrays_h["velocity2"] * duration_s)
        + arrays_h["phase2"]
        + float(phase_control_steps_h[-1])
    )
    potential2 = -(depth2_h * (envelope2 * np.cos(phase2_end) ** 2))
    relative_velocity = velocities_end - (
        arrays_h["velocity2"][:, None] * e2_np
    )
    kinetic2 = (
        0.5 * mass_h * (relative_velocity * relative_velocity).sum(axis=1)
    )
    final_excitation = kinetic2 + potential2 + depth2_h
    if reference.include_gravity:
        final_excitation = (
            kinetic2
            + potential2
            + mass_h * GRAVITY * positions_end[:, 1]
            - arrays_h["gravity_minimum_j"]
        )
    barrier_fraction_h = np.repeat(
        np.asarray(barrier_fractions), count
    )
    effective_barrier = depth2_h * barrier_fraction_h
    captured = np.isfinite(final_excitation) & (
        final_excitation < effective_barrier
    ) & (effective_barrier > 0.0)

    # 6. 逐点汇总（Jeffreys 标准误、捕获子样本温度、端点 trace）。
    captured_counts = np.bincount(
        point_index_h, weights=captured.astype(float), minlength=point_count
    )
    sampled_initial_mean = _point_means(
        initial_excitation_h, point_index_h, point_count
    )
    all_atom_final_mean = _point_means(
        final_excitation, point_index_h, point_count
    )
    captured_initial_mean = _point_means(
        initial_excitation_h, point_index_h, point_count, captured
    )
    captured_final_mean = _point_means(
        final_excitation, point_index_h, point_count, captured
    )
    # trace 端点：t=0 用初态（host 副本），终点用末态；动能温度均为
    # 去质心口径，捕获子样本口径与逐点调用一致（captured 掩码作用于
    # 两个端点）。
    speed2_initial_all = _centered_speed2(
        velocities_h, point_index_h, point_count
    )
    speed2_initial_captured = _centered_speed2(
        velocities_h, point_index_h, point_count, captured
    )
    speed2_final_all = _centered_speed2(
        velocities_end, point_index_h, point_count
    )
    speed2_final_captured = _centered_speed2(
        relative_velocity, point_index_h, point_count, captured
    )
    mean_speed2_initial_all = _point_means(
        speed2_initial_all, point_index_h, point_count
    )
    mean_speed2_initial_captured = _point_means(
        speed2_initial_captured, point_index_h, point_count, captured
    )
    mean_speed2_final_all = _point_means(
        speed2_final_all, point_index_h, point_count
    )
    mean_speed2_final_captured = _point_means(
        speed2_final_captured, point_index_h, point_count, captured
    )
    radius2_initial_all = _centered_speed2(
        positions_h, point_index_h, point_count
    )
    radius2_final_all = _centered_speed2(
        positions_end, point_index_h, point_count
    )
    rms_initial = np.sqrt(
        _point_means(radius2_initial_all, point_index_h, point_count)
    )
    rms_final = np.sqrt(
        _point_means(radius2_final_all, point_index_h, point_count)
    )
    mass_per_point = mass_h[::count]
    mean_events = (
        np.bincount(
            point_index_h,
            weights=scatter_counts.get(),
            minlength=point_count,
        )
        / count
    )
    recoil_energy = (
        HBAR**2 * wave_number_h[::count] ** 2 / (2.0 * mass_per_point)
    )

    results: list[HandoverResult] = []
    for point_index, point in enumerate(chunk):
        if point_index in zero_results:
            # 采样失败点：零捕获结果（全部原子在 handover 环节丢失）。
            results.append(zero_results[point_index])
            continue
        local = local_by_point[point_index]
        captured_count = int(captured_counts[local])
        efficiency = captured_count / count
        posterior_alpha = captured_count + 0.5
        posterior_beta = count - captured_count + 0.5
        posterior_sum = posterior_alpha + posterior_beta
        standard_error = math.sqrt(
            posterior_alpha
            * posterior_beta
            / (posterior_sum**2 * (posterior_sum + 1.0))
        )
        sampled_initial_temperature = (
            sampled_initial_mean[local] / (3.0 * BOLTZMANN) * 1e6
        )
        all_atom_final_temperature = (
            all_atom_final_mean[local] / (3.0 * BOLTZMANN) * 1e6
        )
        if captured_count:
            captured_initial_temperature = (
                captured_initial_mean[local] / (3.0 * BOLTZMANN) * 1e6
            )
            final_temperature = (
                captured_final_mean[local] / (3.0 * BOLTZMANN) * 1e6
            )
            final_kinetic_temperature = (
                mass_per_point[local]
                * mean_speed2_final_captured[local]
                / (3.0 * BOLTZMANN)
                * 1e6
            )
            heating = final_temperature - captured_initial_temperature
            captured_trace = (
                mass_per_point[local]
                * mean_speed2_initial_captured[local]
                / (3.0 * BOLTZMANN)
                * 1e6,
                final_kinetic_temperature,
            )
        else:
            captured_initial_temperature = None
            final_temperature = None
            final_kinetic_temperature = None
            heating = None
            captured_trace = (None, None)
        kinetic_temperature_trace = (
            mass_per_point[local]
            * mean_speed2_initial_all[local]
            / (3.0 * BOLTZMANN)
            * 1e6,
            mass_per_point[local]
            * mean_speed2_final_all[local]
            / (3.0 * BOLTZMANN)
            * 1e6,
        )
        results.append(
            HandoverResult(
                parameters=point,
                captured_count=captured_count,
                transfer_efficiency=efficiency,
                transfer_standard_error=standard_error,
                estimated_captured_atom_number=(
                    point.initial_atom_number * efficiency
                ),
                estimated_captured_atom_number_standard_error=(
                    point.initial_atom_number * standard_error
                ),
                sampled_initial_temperature_uK=sampled_initial_temperature,
                captured_initial_temperature_uK=captured_initial_temperature,
                final_temperature_uK=final_temperature,
                final_kinetic_temperature_uK=final_kinetic_temperature,
                handover_heating_uK=heating,
                all_atom_final_temperature_uK=all_atom_final_temperature,
                all_atom_handover_heating_uK=(
                    all_atom_final_temperature - sampled_initial_temperature
                ),
                mean_scattering_events=float(mean_events[local]),
                recoil_heating_estimate_uK=(
                    mean_events[local]
                    * 2.0
                    * recoil_energy[local]
                    / (3.0 * BOLTZMANN)
                    * 1e6
                ),
                critical_acceleration_m_s2=critical_accelerations[local],
                barrier_fraction=barrier_fractions[local],
                effective_barrier_uK=(
                    point.depth2_uK * barrier_fractions[local]
                ),
                integration_steps=integration_steps,
                actual_time_step_us=time_step_s * 1e6,
                trace=HandoverTrace(
                    time_ms=(0.0, point.duration_ms),
                    lattice1_fraction=(
                        float(fraction1_steps_h[0]),
                        float(fraction1_steps_h[-1]),
                    ),
                    lattice2_fraction=(
                        float(fraction2_steps_h[0]),
                        float(fraction2_steps_h[-1]),
                    ),
                    kinetic_temperature_uK=kinetic_temperature_trace,
                    captured_kinetic_temperature_uK=captured_trace,
                    cloud_rms_radius_um=(
                        float(rms_initial[local]) * 1e6,
                        float(rms_final[local]) * 1e6,
                    ),
                ),
            )
        )
    if not return_captured_ensembles:
        return results
    captured_ensembles: list[ParticleEnsemble | None] = []
    final_phase_control = float(phase_control_steps_h[-1])
    angle = math.radians(reference.crossing_angle_deg)
    l2_offset = reference.l2_transverse_offset_um * 1e-6 * np.array(
        (math.cos(angle), 0.0, -math.sin(angle))
    )
    for point_number, point in enumerate(chunk):
        if point_number in zero_results:
            # 采样失败点：无捕获相空间。
            captured_ensembles.append(None)
            continue
        local = local_by_point[point_number]
        selected = (point_index_h == local) & captured
        if not np.any(selected):
            captured_ensembles.append(None)
            continue
        phase = arrays_h['phase2'][selected] + final_phase_control
        wave_number = 2.0 * math.pi / (point.wavelength_nm * 1e-9)
        captured_ensembles.append(
            canonicalize_lattice_phase(
                ParticleEnsemble(
                    positions_m=positions_end[selected].copy(),
                    velocities_m_s=velocities_end[selected].copy(),
                    frame='handover',
                ),
                phase_rad=phase,
                wave_number_m=wave_number,
                axis=e2_np,
                beam_offset_m=l2_offset,
                lattice_displacement_m=(
                    point.lattice2_velocity_m_s * duration_s
                ),
                lattice_velocity_m_s=point.lattice2_velocity_m_s,
                frame='handover_l2_canonical',
            )
        )
    return results, captured_ensembles


def _chunk_points(
    parameters: list[HandoverParameters],
) -> list[list[HandoverParameters]]:
    """按单批粒子上限把网格点分块（显存保护）。"""
    count = parameters[0].particle_count
    points_per_chunk = max(1, _MAX_BATCH_PARTICLES // count)
    return [
        parameters[start : start + points_per_chunk]
        for start in range(0, len(parameters), points_per_chunk)
    ]


def run_handover_monte_carlo_batch(
    parameters_list: list[HandoverParameters],
    *,
    backend: str = "gpu",
    progress=None,
    initial_ensembles: list[ParticleEnsemble | None] | None = None,
    return_captured_ensembles: bool = False,
) -> (
    list[HandoverResult]
    | tuple[list[HandoverResult], list[ParticleEnsemble | None]]
):
    """批量运行多个网格点的 handover Monte Carlo，逐点返回结果。

    ``backend="gpu"`` 时全部网格点的粒子摊平后在 GPU 上用一个
    mega-step kernel 同时推进（P×N 超阈值自动分块）；``backend="cpu"``
    时退化为逐点调用 ``run_handover_monte_carlo``。同批一致性要求见
    模块 docstring，不满足时抛 ``ValueError``。``progress`` 给定时
    每个分块及积分过程周期性报告（消息含 ``n/total`` 供 UI 解析）。
    """
    parameters = list(parameters_list)
    if not parameters:
        raise ValueError("批量 handover 的参数列表不能为空")
    if backend not in {"cpu", "gpu"}:
        raise ValueError("计算后端必须是 cpu 或 gpu")
    _check_consistency(parameters)
    if initial_ensembles is not None and len(initial_ensembles) != len(parameters):
        raise ValueError('initial ensemble list must match handover batch size')
    def _run_cpu():
        outputs = [
            run_handover_monte_carlo(
                point,
                initial_ensemble=(
                    None if initial_ensembles is None else initial_ensembles[index]
                ),
                return_captured_ensemble=return_captured_ensembles,
            )
            for index, point in enumerate(parameters)
        ]
        if return_captured_ensembles:
            return (
                [output[0] for output in outputs],
                [output[1] for output in outputs],
            )
        return outputs

    if backend == "cpu":
        return _run_cpu()
    try:
        _resolve_backend("gpu")
        chunks = _chunk_points(parameters)
        # 稳定步长按完整逻辑批次一次确定，再传给每个显存分块；否则改变
        # 分块阈值会令不同深度的块采用不同 dt，破坏分块确定性。
        requested_step_s = min(
            parameters[0].time_step_us * 1e-6,
            *(_stable_handover_step_s(point) for point in parameters),
        )
        results: list[HandoverResult] = []
        captured_ensembles: list[ParticleEnsemble | None] = []
        offset = 0
        for chunk_index, chunk in enumerate(chunks, start=1):
            if progress is not None and len(chunks) > 1:
                progress(
                    f"GPU 批量 handover 分块 {chunk_index}/{len(chunks)}"
                )
            chunk_output = _run_gpu_chunk(
                chunk,
                progress=progress,
                requested_step_s=requested_step_s,
                initial_ensembles=(
                    None
                    if initial_ensembles is None
                    else initial_ensembles[offset : offset + len(chunk)]
                ),
                return_captured_ensembles=return_captured_ensembles,
            )
            offset += len(chunk)
            if return_captured_ensembles:
                chunk_results, chunk_ensembles = chunk_output
                results.extend(chunk_results)
                captured_ensembles.extend(chunk_ensembles)
            else:
                results.extend(chunk_output)
        if return_captured_ensembles:
            return results, captured_ensembles
        return results
    except Exception as exc:  # noqa: BLE001 - GPU 不可用/内核失败回退 CPU
        if progress is not None:
            progress(
                f"GPU 批量 handover 不可用（{exc}），回退 CPU 逐点计算"
            )
        return _run_cpu()
