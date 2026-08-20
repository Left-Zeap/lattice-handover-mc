"""L1 宏观运输与 handover 经典轨迹 Monte Carlo 的逐点衔接。

本模块只负责模拟编排和二维扫描；绘图放在 ``l1_handover_plots.py``。
已有 ``simulate_l1_transport`` 和 ``run_handover_monte_carlo`` 接口保持不变。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import math
import os
from typing import Callable

import numpy as np

from .atomic import CS133, RB87, AlkaliAtom
from .constants import BOLTZMANN
from .control_waveforms import HandoverControlWaveform
from .handover import HandoverParameters, HandoverResult, run_handover_monte_carlo
from .handover_batch import run_handover_monte_carlo_batch
from .transport_batch import run_leg_monte_carlo_batch
from .l1_transport import (
    L1DesignPoint,
    L1TransportInputs,
    L1TransportTrace,
    L1_TRANSPORT_CONFIGURATION,
    simulate_l1_transport,
)
from .phase_space import (
    ParticleEnsemble,
    l1_transport_end_to_handover,
)


_MC = L1_TRANSPORT_CONFIGURATION["handover_monte_carlo"]
# 自适应粒子加密（两阶段扫描）：第一遍用基础粒子数，交接率标准误
# 超过目标的点用按 1/√N 标定的更大粒子数复算。默认关闭。
_MC_ADAPTIVE = _MC.get("adaptive_refinement", {})


@dataclass(frozen=True)
class L1HandoverInputs:
    """统一网格和 handover Monte Carlo 数值参数。"""

    transport: L1TransportInputs = L1TransportInputs()
    duration_us: float = float(_MC["duration_us"])
    particle_count: int = int(_MC["particle_count"])
    time_step_us: float = float(_MC["time_step_us"])
    trace_points: int = int(_MC["trace_points"])
    include_scattering: bool = bool(_MC["include_scattering"])
    seed: int = int(_MC["seed"])
    compute_backend: str = str(_MC["compute_backend"])
    parallel_backend: str = str(_MC["parallel_backend"])
    worker_count: int = int(_MC["worker_count"])
    crossing_angle_deg: float = float(_MC["crossing_angle_deg"])
    cloud_axial_sigma_mm: float = float(_MC["cloud_axial_sigma_mm"])
    l2_transverse_offset_um: float = float(_MC["l2_transverse_offset_um"])
    randomize_relative_phase: bool = bool(_MC["randomize_relative_phase"])
    # 固定相位口径（randomize_relative_phase=False）时使用的 L1/L2 相对
    # 相位；随机口径下该值叠加在逐粒子均匀随机相位上（通常保持 0）。
    relative_phase_rad: float = 0.0
    adaptive_refinement: bool = bool(_MC_ADAPTIVE.get("enabled", False))
    adaptive_target_standard_error: float = float(
        _MC_ADAPTIVE.get("target_standard_error", 0.005)
    )
    adaptive_max_particle_count: int = int(
        _MC_ADAPTIVE.get("max_particle_count", 100_000)
    )
    control_waveform: HandoverControlWaveform | None = None

    def __post_init__(self) -> None:
        if self.duration_us <= 0.0 or self.time_step_us <= 0.0:
            raise ValueError("handover 时间和时间步长必须为正")
        if self.particle_count <= 0 or self.trace_points < 2:
            raise ValueError("Monte Carlo 粒子数必须为正，轨迹点数至少为 2")
        if self.parallel_backend not in {"serial", "process"}:
            raise ValueError("并行后端必须是 serial 或 process")
        if self.compute_backend not in {"cpu", "gpu"}:
            raise ValueError("计算后端必须是 cpu 或 gpu")
        if self.worker_count <= 0:
            raise ValueError("CPU 工作进程数必须为正")
        if (
            not math.isfinite(self.adaptive_target_standard_error)
            or self.adaptive_target_standard_error <= 0.0
        ):
            raise ValueError("自适应加密的目标标准误必须是有限正数")
        if self.adaptive_max_particle_count < self.particle_count:
            raise ValueError("自适应加密的粒子数上限不能小于基础粒子数")
        if not math.isfinite(self.relative_phase_rad):
            raise ValueError("交接相对相位必须是有限数")
        if self.control_waveform is not None and not math.isclose(
            self.control_waveform.duration_ms,
            self.duration_us * 1e-3,
            rel_tol=0.0,
            abs_tol=max(1e-9, 1e-9 * self.duration_us),
        ):
            raise ValueError("handover 波形时长必须与 UI/配置交接时长一致")


@dataclass(frozen=True)
class L1HandoverPoint:
    """一个功率--失谐点的 L1 与 handover 联合汇总。"""

    detuning_ghz: float
    source_power_w: float
    transport: L1DesignPoint
    handover_transfer_efficiency: float
    handover_transfer_standard_error: float
    handover_heating_uK: float | None
    final_temperature_uK: float | None
    total_temperature_rise_uK: float | None
    final_retention_from_loaded_l1: float
    final_retention_from_mot: float
    final_atom_number: float
    # 该点 handover 实际使用的轨迹数（自适应加密后可大于基础值）。
    handover_particle_count: int | None = None


@dataclass(frozen=True)
class L1HandoverCombinedTrace:
    """L1 运输和 handover 拼接后的连续时间轨迹。

    LGM 装载模块已移除，轨迹从 L1 起点（静止晶格热平衡初态）开始；
    ``loading_start_ms``/``loading_end_ms`` 已 deprecated，恒为 None，
    仅为 UI 时间线消费者暂时保留。
    """

    time_ms: tuple[float, ...]
    phase: tuple[str, ...]
    temperature_uK: tuple[float, ...]
    retention_from_mot: tuple[float, ...]
    handover_start_ms: float
    handover_end_ms: float
    loading_start_ms: float | None = None
    loading_end_ms: float | None = None
    calculation_boundary: str = "static_lattice_thermal"


@dataclass(frozen=True)
class L1HandoverPointSimulation:
    point: L1HandoverPoint
    transport_trace: L1TransportTrace
    handover_result: HandoverResult
    combined_trace: L1HandoverCombinedTrace


@dataclass(frozen=True)
class L1HandoverScanResult:
    inputs: L1HandoverInputs
    detuning_ghz: tuple[float, ...]
    source_power_w: tuple[float, ...]
    transport_feasible: tuple[tuple[bool, ...], ...]
    handover_transfer_efficiency: tuple[tuple[float | None, ...], ...]
    handover_transfer_standard_error: tuple[
        tuple[float | None, ...], ...
    ]
    handover_heating_uK: tuple[tuple[float | None, ...], ...]
    total_temperature_rise_uK: tuple[tuple[float | None, ...], ...]
    final_retention_from_mot: tuple[tuple[float | None, ...], ...]
    evaluated_points: int
    optimal: L1HandoverPoint
    comparison: L1HandoverPoint
    # 全网格失败时 optimal/comparison 为哨兵点且 simulation 为 None
    # （绘图/导出/CLI 均已按 None 分支处理）。
    optimal_simulation: L1HandoverPointSimulation | None = None
    comparison_simulation: L1HandoverPointSimulation | None = None
    # 逐网格点的完整联合结果（含 L1 末态），供下游（如全链路 L2 补算）
    # 直接复用，避免按方法重新计算运输腿；失败点为 None。
    point_grid: tuple[tuple[L1HandoverPoint | None, ...], ...] | None = None
    # 自适应粒子加密第二遍复算的网格点数（未启用或全部达标时为 0）。
    refined_points: int = 0


def _atom(label: str) -> AlkaliAtom:
    normalized = label.lower().replace("-", "").replace("_", "")
    if normalized in {"rb", "rb87", "87rb"}:
        return RB87
    if normalized in {"cs", "cs133", "133cs"}:
        return CS133
    raise ValueError("原子必须是 Rb-87 或 Cs-133")


def failed_design_point(
    transport: L1TransportInputs,
) -> L1DesignPoint:
    """全网格失败时的哨兵 L1 设计点（扫描窗角点坐标，数值 NaN/0 占位）。

    只用于让扫描结果对象、绘图与 CLI 打印不因"无有效点"而崩溃；
    不参与任何物理解释。
    """
    return L1DesignPoint(
        detuning_ghz=transport.detuning_min_ghz,
        handover_source_power_w=transport.handover_source_power_min_w,
        start_source_power_w=transport.handover_source_power_min_w,
        wavelength_nm=float("nan"),
        depth_uK=float("nan"),
        scattering_rate_s=float("nan"),
        final_temperature_uK=float("nan"),
        final_temperature_rise_uK=float("nan"),
        final_retention_fraction=0.0,
        total_retention_from_mot_fraction=0.0,
        final_atom_number=0.0,
        cumulative_scattering_events=0.0,
        maximum_loss_rate_s=0.0,
        feasible_hardware_point=False,
    )


def failed_handover_point(
    inputs: L1HandoverInputs,
) -> L1HandoverPoint:
    """全网格失败时的哨兵 L1→handover 汇总点（见 ``failed_design_point``）。"""
    transport = inputs.transport
    return L1HandoverPoint(
        detuning_ghz=transport.detuning_min_ghz,
        source_power_w=transport.handover_source_power_min_w,
        transport=failed_design_point(transport),
        handover_transfer_efficiency=float("nan"),
        handover_transfer_standard_error=float("nan"),
        handover_heating_uK=float("nan"),
        final_temperature_uK=float("nan"),
        total_temperature_rise_uK=float("nan"),
        final_retention_from_loaded_l1=0.0,
        final_retention_from_mot=0.0,
        final_atom_number=0.0,
        handover_particle_count=None,
    )


def _handover_parameters(
    inputs: L1HandoverInputs,
    transport_trace: L1TransportTrace,
    *,
    trace_points: int,
    post_handover_acceleration_m_s2: float | None = None,
) -> HandoverParameters:
    """由 L1 末态构建 handover 参数。

    ``post_handover_acceleration_m_s2`` 为交接后 L2 的加速度（仅用于末
    态捕获判据的倾斜势垒）。默认 None 时取 L1 运输加速度（梯形 L2 的
    保守口径）；连续相空间模式（L2 为 minimum_jerk/实测波形）应显式传
    入 L2 轨迹在交接时刻的瞬时加速度（minimum_jerk 为 0），否则判据
    会系统性地用错势垒。
    """
    atom = _atom(inputs.transport.atom_label)
    point = transport_trace.point
    if post_handover_acceleration_m_s2 is None:
        post_handover_acceleration_m_s2 = inputs.transport.acceleration_m_s2
    return HandoverParameters(
        atom_mass_kg=atom.mass_kg,
        wavelength_nm=point.wavelength_nm,
        depth1_uK=point.depth_uK,
        depth2_uK=point.depth_uK,
        waist1_um=inputs.transport.handover_waist_um,
        waist2_um=inputs.transport.handover_waist_um,
        scattering_rate1_s=point.scattering_rate_s,
        scattering_rate2_s=point.scattering_rate_s,
        retro_power_ratio=inputs.transport.retro_power_ratio,
        initial_atom_number=point.final_atom_number,
        temperature_uK=point.final_temperature_uK,
        duration_ms=inputs.duration_us * 1e-3,
        crossing_angle_deg=inputs.crossing_angle_deg,
        lattice1_distance_cm=inputs.transport.distance_m * 100.0,
        optimal_distance_cm=inputs.transport.distance_m * 100.0,
        cloud_axial_sigma_mm=inputs.cloud_axial_sigma_mm,
        l2_transverse_offset_um=inputs.l2_transverse_offset_um,
        randomize_relative_phase=inputs.randomize_relative_phase,
        relative_phase_rad=inputs.relative_phase_rad,
        post_handover_acceleration_m_s2=post_handover_acceleration_m_s2,
        include_gravity=inputs.transport.include_gravity,
        include_scattering=inputs.include_scattering,
        compute_backend=inputs.compute_backend,
        particle_count=inputs.particle_count,
        time_step_us=inputs.time_step_us,
        trace_points=trace_points,
        seed=inputs.seed,
        control_waveform=inputs.control_waveform,
    )


def _effective_initial_state(
    inputs: L1HandoverInputs, point: L1DesignPoint
) -> tuple[float, float]:
    """返回该网格点实际使用的 L1 初温/原子数。

    初态为静止 L1 晶格热平衡图景的固定 (N, T)（记录在
    ``L1DesignPoint`` 的有效初态字段）；旧版本结果缺失字段时回退到
    输入的固定值。
    """
    temperature = point.initial_temperature_uK
    atom_number = point.initial_atom_number
    return (
        inputs.transport.initial_temperature_uK
        if temperature is None
        else temperature,
        inputs.transport.initial_atom_number
        if atom_number is None
        else atom_number,
    )


def _combined_trace(
    inputs: L1HandoverInputs,
    transport_trace: L1TransportTrace,
    handover: HandoverResult,
) -> L1HandoverCombinedTrace:
    _, initial_atom_number = _effective_initial_state(
        inputs, transport_trace.point
    )
    loading = initial_atom_number / inputs.transport.mot_atom_number
    l1_times = list(transport_trace.time_ms)
    start = l1_times[-1]
    handover_times = [start + value for value in handover.trace.time_ms]
    l1_retention = [loading * value for value in transport_trace.retention_fraction]
    pre_handover_retention = l1_retention[-1]
    handover_retention = [pre_handover_retention] * len(handover_times)
    if handover_retention:
        handover_retention[-1] = (
            pre_handover_retention * handover.transfer_efficiency
        )
    # handover 过程中画“当前全部输入样本”的三维动能温度；旧实现从
    # 第一帧起回溯筛选“最终会捕获的子样本”，会在 L1 边界制造选择
    # 偏差跳变。终点再切到真正传入 L2 的捕获子样本动能温度。
    handover_temperature = list(handover.trace.kinetic_temperature_uK)
    if handover_temperature and transport_trace.temperature_uK:
        handover_temperature[0] = transport_trace.temperature_uK[-1]
    final_kinetic = getattr(handover, "final_kinetic_temperature_uK", None)
    if handover_temperature and final_kinetic is not None:
        handover_temperature[-1] = float(final_kinetic)
    return L1HandoverCombinedTrace(
        time_ms=tuple(l1_times + handover_times),
        phase=tuple(
            ["L1 transport"] * len(l1_times)
            + ["handover"] * len(handover_times)
        ),
        temperature_uK=tuple(
            list(transport_trace.temperature_uK) + handover_temperature
        ),
        retention_from_mot=tuple(l1_retention + handover_retention),
        handover_start_ms=start,
        handover_end_ms=handover_times[-1],
    )


def _summary(
    inputs: L1HandoverInputs,
    transport_trace: L1TransportTrace,
    handover: HandoverResult,
) -> L1HandoverPoint:
    final_temperature = handover.final_temperature_uK
    final_loaded_retention = (
        transport_trace.point.final_retention_fraction
        * handover.transfer_efficiency
    )
    initial_temperature_uK, initial_atom_number = _effective_initial_state(
        inputs, transport_trace.point
    )
    # 装载效率取固定初态口径（L1 初态原子数 / MOT 原子数），只乘一次。
    final_mot_retention = (
        initial_atom_number / inputs.transport.mot_atom_number
    ) * final_loaded_retention
    handover_parameters = getattr(handover, "parameters", None)
    return L1HandoverPoint(
        detuning_ghz=transport_trace.point.detuning_ghz,
        source_power_w=transport_trace.point.handover_source_power_w,
        transport=transport_trace.point,
        handover_transfer_efficiency=handover.transfer_efficiency,
        handover_transfer_standard_error=handover.transfer_standard_error,
        handover_heating_uK=handover.handover_heating_uK,
        final_temperature_uK=final_temperature,
        total_temperature_rise_uK=(
            None
            if final_temperature is None
            else final_temperature - initial_temperature_uK
        ),
        final_retention_from_loaded_l1=final_loaded_retention,
        final_retention_from_mot=final_mot_retention,
        final_atom_number=(
            inputs.transport.mot_atom_number * final_mot_retention
        ),
        handover_particle_count=(
            None
            if handover_parameters is None
            else handover_parameters.particle_count
        ),
    )


def _validate_transport_trace(transport_trace: L1TransportTrace) -> None:
    """handover 前置校验（不可用时抛 ``ValueError``）。"""
    if not transport_trace.point.feasible_hardware_point:
        raise ValueError("该参数点不满足 L1 运输的阱深、加速度或功率限制")
    if (
        not math.isfinite(transport_trace.point.final_temperature_uK)
        or transport_trace.point.final_atom_number <= 0.0
    ):
        raise ValueError("L1 末端无存活原子，无法进入 handover Monte Carlo")
    if transport_trace.point.final_temperature_uK <= 0.0:
        # 幸存样本退化（如仅剩 1 个粒子、速度方差为零）时温度为 0：
        # 有限但非正，HandoverParameters 的"温度必须是有限正数"校验
        # 会拒绝；这类点按不可用处理，与零存活同一口径。
        raise ValueError(
            "L1 末端幸存样本温度退化（≤0），无法进入 handover Monte Carlo"
        )


def _validated_transport_trace(
    inputs: L1HandoverInputs,
    detuning_ghz: float,
    source_power_w: float,
) -> L1TransportTrace:
    """L1 运输腿 + handover 前置校验（不可用时抛 ``ValueError``）。"""
    if source_power_w <= 0.0:
        raise ValueError("零功率点没有束缚势阱，不能运行 handover Monte Carlo")
    transport_trace = simulate_l1_transport(
        inputs.transport,
        detuning_ghz,
        source_power_w,
    )
    _validate_transport_trace(transport_trace)
    return transport_trace


def simulate_l1_handover_point(
    inputs: L1HandoverInputs,
    detuning_ghz: float,
    source_power_w: float,
    *,
    trace_points: int | None = None,
) -> L1HandoverPointSimulation:
    """先完成 L1 宏观运输，再把末温和末态原子数传给 handover。"""
    transport_trace = _validated_transport_trace(
        inputs, detuning_ghz, source_power_w
    )
    parameters = _handover_parameters(
        inputs,
        transport_trace,
        trace_points=(inputs.trace_points if trace_points is None else trace_points),
    )
    handover = run_handover_monte_carlo(parameters)
    return L1HandoverPointSimulation(
        point=_summary(inputs, transport_trace, handover),
        transport_trace=transport_trace,
        handover_result=handover,
        combined_trace=_combined_trace(inputs, transport_trace, handover),
    )


def _sample_l1_initial_ensemble(
    transport: L1TransportInputs,
    detuning_ghz: float,
    source_power_w: float,
) -> ParticleEnsemble:
    """L1 起点静止 L1 晶格热平衡系综（chain_mc 同一初始晶格还原口径）。

    温度取 ``L1TransportInputs.initial_temperature_uK``；晶格参数取
    L1 腿 t=0 光学量——波腹阱深精确还原双束强度，束腰取强度加权有效
    束腰（与 ``transport_mc._leg_optics_profile`` 的 conveyor 有效束腰
    同一公式，w₁=w₂ 时退化为共同束腰）。浅阱/高温到几乎无束缚初态时
    采样抛 ``ValueError``，由调用方决定容错策略。
    """
    from .dipole import scalar_potential_and_scattering
    from .initial_state import (
        ThermalLatticeEnsembleInputs,
        sample_static_lattice_thermal_ensemble,
    )
    from .transport_mc import _leg_optics_at, _leg_optics_profile

    atom = _atom(transport.atom_label)
    wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning_ghz)
    profile = _leg_optics_profile(transport, wavelength_nm, source_power_w)
    i1, i2, w1, w2, _, _ = _leg_optics_at(
        transport, profile, source_power_w, 0.0, 0.0
    )
    potential_per_intensity = abs(
        scalar_potential_and_scattering(atom, wavelength_nm, 1.0).potential_j
    )
    antinode_depth_j = potential_per_intensity * (
        i1 + i2 + 2.0 * math.sqrt(i1 * i2)
    )
    effective_waist_m = math.sqrt(
        (i1 + i2) / (i1 / w1**2 + i2 / w2**2)
    )
    sampling_inputs = ThermalLatticeEnsembleInputs(
        atom_label=transport.atom_label,
        wavelength_nm=wavelength_nm,
        waist_um=effective_waist_m * 1e6,
        depth_uK=antinode_depth_j / BOLTZMANN * 1e6,
        temperature_uK=transport.initial_temperature_uK,
        particle_count=transport.mc_particle_count,
        seed=transport.mc_seed,
        retro_power_ratio=i2 / i1,
        cloud_axial_sigma_mm=transport.mc_cloud_axial_sigma_mm,
        include_gravity=transport.include_gravity,
    )
    return sample_static_lattice_thermal_ensemble(sampling_inputs)


def simulate_l1_handover_point_continuous(
    inputs: L1HandoverInputs,
    detuning_ghz: float,
    source_power_w: float,
    *,
    trace_points: int | None = None,
    post_handover_acceleration_m_s2: float | None = None,
) -> tuple[L1HandoverPointSimulation, ParticleEnsemble | None]:
    """以同一经验相空间样本贯通 L1→handover。

    初态为 L1 起点静止晶格热平衡系综（``initial_state`` 采样；LGM
    装载模块已移除）。阶段汇总结果仍沿用现有数据类，粒子数组只在
    进程内瞬态传递，不进入 JSON 或二维扫描。
    """
    if inputs.transport.transport_method != "monte_carlo":
        raise ValueError("相空间连续模式要求 transport_method=monte_carlo")
    if (
        inputs.transport.control_waveform is None
        and inputs.transport.kinematic_profile != "minimum_jerk"
    ):
        raise ValueError(
            "连续相空间模式禁止加速度阶跃：L1 请使用 minimum_jerk "
            "或提供连续的实测控制波形"
        )
    if source_power_w <= 0.0:
        raise ValueError("零功率点不能运行相空间连续链路")

    from .transport_mc import simulate_leg_monte_carlo

    initial_ensemble = _sample_l1_initial_ensemble(
        inputs.transport, detuning_ghz, source_power_w
    )
    transport_trace, transport_ensemble = simulate_leg_monte_carlo(
        inputs.transport,
        detuning_ghz,
        source_power_w,
        initial_ensemble=initial_ensemble,
        return_final_ensemble=True,
    )
    transport_trace = replace(
        transport_trace,
        pre_ramp_survival_fraction=(
            inputs.transport.pre_ramp_survival_fraction
        ),
    )
    _validate_transport_trace(transport_trace)
    if transport_ensemble is None:
        raise ValueError("L1 运输末端无幸存相空间样本")
    handover_initial = l1_transport_end_to_handover(
        transport_ensemble, inputs.transport.distance_m
    )
    parameters = _handover_parameters(
        inputs,
        transport_trace,
        trace_points=(inputs.trace_points if trace_points is None else trace_points),
        post_handover_acceleration_m_s2=post_handover_acceleration_m_s2,
    )
    handover, captured_ensemble = run_handover_monte_carlo(
        parameters,
        initial_ensemble=handover_initial,
        return_captured_ensemble=True,
    )
    simulation = L1HandoverPointSimulation(
        point=_summary(inputs, transport_trace, handover),
        transport_trace=transport_trace,
        handover_result=handover,
        combined_trace=_combined_trace(inputs, transport_trace, handover),
    )
    return simulation, captured_ensemble


def _scan_task(
    task: tuple[tuple[int, int], L1HandoverInputs, float, float],
) -> tuple[tuple[int, int], L1HandoverPoint | None]:
    """单网格点联合计算；单点失败返回 None，由扫描层记为无效点。"""
    index, inputs, detuning, power = task
    try:
        simulation = simulate_l1_handover_point(
            inputs,
            detuning,
            power,
            trace_points=2,
        )
    except Exception:  # noqa: BLE001 - 单点异常按该网格点无有效结果处理
        return index, None
    return index, simulation.point


def _scan_task_full(
    task: tuple[tuple[int, int], L1HandoverInputs, float, float],
) -> tuple[tuple[int, int], L1HandoverPointSimulation | None]:
    """与 ``_scan_task`` 相同，但返回完整模拟（自适应加密要保留逐点
    L1 trace 与 handover 参数供第二遍复算）。"""
    index, inputs, detuning, power = task
    try:
        simulation = simulate_l1_handover_point(
            inputs,
            detuning,
            power,
            trace_points=2,
        )
    except Exception:  # noqa: BLE001 - 单点异常按该网格点无有效结果处理
        return index, None
    return index, simulation


def _refine_task(
    task: tuple[
        tuple[int, int], L1HandoverInputs, L1TransportTrace, object, int
    ],
) -> tuple[tuple[int, int], L1HandoverPoint | None]:
    """自适应加密第二遍：同一 L1 末态上以更多轨迹重跑 handover。

    失败时返回 ``None``，调用方保留原网格点结果，不让加密步骤中断扫描。
    """
    index, inputs, transport_trace, parameters, particle_count = task
    try:
        result = run_handover_monte_carlo(
            replace(parameters, particle_count=particle_count)
        )
    except Exception:  # noqa: BLE001 - 加密失败不应中断二维扫描
        return index, None
    return index, _summary(inputs, transport_trace, result)


def _normalized(values: np.ndarray) -> np.ndarray:
    span = float(np.max(values) - np.min(values))
    if span <= 1e-15:
        return np.zeros_like(values)
    return (values - np.min(values)) / span


def analyze_l1_handover_scan(
    inputs: L1HandoverInputs = L1HandoverInputs(),
    *,
    progress: Callable[[str], None] | None = None,
) -> L1HandoverScanResult:
    """在 L1 网格上逐点运行 transport→handover 联合计算。"""
    transport = inputs.transport
    # 可行性预检只涉及光学量（阱深/功率/临界加速度），与运输动力学方法
    # 无关；强制用解析腿，避免对每个网格点跑昂贵的轨迹模拟（曾导致
    # 扫描看似无法启动）。
    precheck_transport = replace(
        transport,
        transport_method="analytic",
    )
    detunings = np.linspace(
        transport.detuning_min_ghz,
        transport.detuning_max_ghz,
        transport.detuning_points,
    )
    powers = np.linspace(
        transport.handover_source_power_min_w,
        transport.handover_source_power_max_w,
        transport.power_points,
    )
    feasible = np.zeros((transport.power_points, transport.detuning_points), dtype=bool)
    tasks: list[tuple[tuple[int, int], L1HandoverInputs, float, float]] = []
    for power_index, power in enumerate(powers):
        if power <= 0.0:
            continue
        for detuning_index, detuning in enumerate(detunings):
            try:
                trace = simulate_l1_transport(
                    precheck_transport,
                    float(detuning),
                    float(power),
                )
            except Exception:  # noqa: BLE001 - 预检失败点按无任务跳过
                continue
            if trace.point.feasible_hardware_point:
                feasible[power_index, detuning_index] = True
                tasks.append(
                    (
                        (power_index, detuning_index),
                        inputs,
                        float(detuning),
                        float(power),
                    )
                )
    if not tasks:
        # 无任何可连接 handover 的 L1 可行点：不抛错，返回空结果（矩阵
        # NaN + 哨兵点），保证二维扫描总能出图（全灰）。
        if progress is not None:
            progress(
                f"{transport.atom_label}: 统一扫描范围内没有可连接到 "
                "handover 的 L1 可行点，返回空结果（矩阵 NaN）"
            )
        shape = (transport.power_points, transport.detuning_points)
        empty = np.full(shape, np.nan)
        point_grid = tuple(
            tuple(None for _ in range(transport.detuning_points))
            for _ in range(transport.power_points)
        )
        failed_point = failed_handover_point(inputs)

        def optional_matrix(
            array: np.ndarray,
        ) -> tuple[tuple[float | None, ...], ...]:
            return tuple(
                tuple(
                    None if math.isnan(value) else float(value)
                    for value in row
                )
                for row in array
            )

        return L1HandoverScanResult(
            inputs=inputs,
            detuning_ghz=tuple(float(value) for value in detunings),
            source_power_w=tuple(float(value) for value in powers),
            transport_feasible=tuple(
                tuple(
                    False for _ in range(transport.detuning_points)
                )
                for _ in range(transport.power_points)
            ),
            handover_transfer_efficiency=optional_matrix(empty),
            handover_transfer_standard_error=optional_matrix(empty),
            handover_heating_uK=optional_matrix(empty),
            total_temperature_rise_uK=optional_matrix(empty),
            final_retention_from_mot=optional_matrix(empty),
            evaluated_points=0,
            optimal=failed_point,
            comparison=failed_point,
            optimal_simulation=None,
            comparison_simulation=None,
            point_grid=point_grid,
            refined_points=0,
        )

    shape = (transport.power_points, transport.detuning_points)
    efficiency = np.full(shape, np.nan)
    standard_error = np.full(shape, np.nan)
    handover_heating = np.full(shape, np.nan)
    total_heating = np.full(shape, np.nan)
    final_retention = np.full(shape, np.nan)
    points: dict[tuple[int, int], L1HandoverPoint] = {}
    failed_points: list[tuple[int, int]] = []
    total = len(tasks)
    # GPU 后端下禁止外层进程池：多个进程共享同一块 GPU 会竞争
    # CUDA 上下文与 ~/.cupy 内核缓存（并发编译可致缓存损坏和挂起）。
    # GPU 的并行由批量 handover 承担：全部网格点的粒子摊平后在
    # 单次批量调用中同时推进（见 handover_batch.py）。
    use_gpu_batch = inputs.compute_backend == "gpu"
    use_processes = (
        inputs.parallel_backend == "process"
        and inputs.worker_count > 1
        and total > 1
        and not use_gpu_batch
    )
    workers = min(inputs.worker_count, total, os.cpu_count() or 1)
    adaptive = inputs.adaptive_refinement
    task_fn = _scan_task_full if adaptive else _scan_task
    # 自适应加密需要逐点保留 L1 trace 与 handover 参数供第二遍复算。
    trace_by_index: dict[tuple[int, int], L1TransportTrace] = {}
    params_by_index: dict[tuple[int, int], HandoverParameters] = {}

    def _unpack(
        item: tuple[tuple[int, int], object],
    ) -> tuple[tuple[int, int], L1HandoverPoint | None]:
        index, payload = item
        if not adaptive:
            return item
        simulation = payload
        if simulation is None:
            return index, None
        trace_by_index[index] = simulation.transport_trace
        parameters = getattr(simulation.handover_result, "parameters", None)
        if parameters is not None:
            params_by_index[index] = parameters
        return index, simulation.point
    if progress is not None:
        if use_gpu_batch:
            mode = "GPU 批量（全部网格点单次批量调用）"
        else:
            mode = f"{workers} 个 CPU 进程" if use_processes else "串行"
        progress(
            f"{transport.atom_label}: {total} 个 L1 可行点进入 "
            f"N={inputs.particle_count} handover Monte Carlo（{mode}）"
        )

    def store(completed: int, item: tuple[tuple[int, int], L1HandoverPoint | None]) -> None:
        index, point = item
        if point is None:
            # 单点计算失败（如 L1 末端无存活原子、浅阱无法采样）：
            # 矩阵保持 NaN，不中断整个扫描。
            failed_points.append(index)
        else:
            points[index] = point
            efficiency[index] = point.handover_transfer_efficiency
            standard_error[index] = point.handover_transfer_standard_error
            handover_heating[index] = (
                np.nan if point.handover_heating_uK is None else point.handover_heating_uK
            )
            total_heating[index] = (
                np.nan
                if point.total_temperature_rise_uK is None
                else point.total_temperature_rise_uK
            )
            final_retention[index] = point.final_retention_from_mot
        if progress is not None and (
            completed == total or completed % max(1, total // 10) == 0
        ):
            progress(f"{transport.atom_label}: {completed}/{total}")

    if use_gpu_batch:
        # 先完成全部 L1 腿（解析腿逐点很快；MC 腿一次 GPU 批量），再
        # 一次批量调用完成全部 handover MC；全程 progress 反馈。
        pending: list[
            tuple[tuple[int, int], L1TransportTrace, HandoverParameters]
        ] = []
        failed: list[tuple[int, int]] = []
        if transport.transport_method == "monte_carlo":
            if progress is not None:
                progress(
                    f"{transport.atom_label}: 正在 GPU 批量运行 {total} "
                    "个点的 L1 运输 Monte Carlo（首次需编译内核）"
                )
            leg_tasks = [
                (index, transport, detuning, power)
                for index, _, detuning, power in tasks
            ]
            try:
                leg_traces = run_leg_monte_carlo_batch(
                    leg_tasks,
                    backend="gpu",
                    progress=progress,
                )
            except ValueError as exc:
                # 批量未覆盖的情形（如 conveyor 几何）：回退逐点 GPU。
                if progress is not None:
                    progress(
                        f"{transport.atom_label}: 批量运输腿不可用"
                        f"（{exc}），回退逐点 GPU 计算"
                    )
                leg_traces = []
                for leg_index, (_, leg_transport, detuning, power) in enumerate(
                    leg_tasks, start=1
                ):
                    if progress is not None:
                        progress(
                            f"{transport.atom_label}: L1 运输腿 "
                            f"{leg_index}/{len(leg_tasks)}"
                        )
                    try:
                        leg_traces.append(
                            simulate_l1_transport(leg_transport, detuning, power)
                        )
                    except Exception:  # noqa: BLE001 - 单点异常按该点无有效结果处理
                        leg_traces.append(None)
            for (index, _, _, _), trace in zip(leg_tasks, leg_traces):
                try:
                    if trace is None:
                        raise ValueError("L1 运输腿计算失败")
                    _validate_transport_trace(trace)
                    parameters = _handover_parameters(
                        inputs, trace, trace_points=2
                    )
                except Exception:  # noqa: BLE001 - 单点异常按该点无有效结果处理
                    failed.append(index)
                else:
                    pending.append((index, trace, parameters))
        else:
            leg_stride = max(1, total // 50)
            for leg_index, (index, _, detuning, power) in enumerate(
                tasks, start=1
            ):
                if progress is not None and (
                    leg_index % leg_stride == 0 or leg_index == total
                ):
                    progress(
                        f"{transport.atom_label}: L1 运输腿 "
                        f"{leg_index}/{total}"
                    )
                try:
                    trace = _validated_transport_trace(
                        inputs, detuning, power
                    )
                    parameters = _handover_parameters(
                        inputs, trace, trace_points=2
                    )
                except Exception:  # noqa: BLE001 - 单点异常按该点无有效结果处理
                    failed.append(index)
                else:
                    pending.append((index, trace, parameters))
        if progress is not None:
            progress(
                f"{transport.atom_label}: 正在 GPU 批量运行 "
                f"{len(pending)} 个点的 handover Monte Carlo"
                "（首次需编译内核）"
            )
        # 全部动态点都可能因 L1 零留存而被筛掉；空集合无需调用批量
        # API（其“参数列表不能为空”校验不是扫描错误），继续走统一的
        # 无有效点收尾并保留已计算的网格掩膜。
        results = (
            run_handover_monte_carlo_batch(
                [parameters for _, _, parameters in pending],
                backend="gpu",
                progress=progress,
            )
            if pending
            else []
        )
        completed = 0
        for (index, trace, parameters), result in zip(pending, results):
            completed += 1
            if adaptive:
                trace_by_index[index] = trace
                params_by_index[index] = parameters
            store(completed, (index, _summary(inputs, trace, result)))
        for index in failed:
            completed += 1
            store(completed, (index, None))
    elif use_processes:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(task_fn, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                store(completed, _unpack(future.result()))
    else:
        for completed, task in enumerate(tasks, start=1):
            store(completed, _unpack(task_fn(task)))

    # 自适应粒子加密第二遍：交接率标准误超标的点按 1/√N 标定更多
    # 轨迹复算（同一 L1 末态、同 seed；批量一致性要求统一的加密后
    # 粒子数，取各点需求的最大值并受上限约束）。
    refined_points = 0
    if adaptive:
        target = inputs.adaptive_target_standard_error
        refine_indices = [
            index
            for index, point in points.items()
            if point.handover_transfer_standard_error > target
            and index in params_by_index
        ]
        if refine_indices:
            required_count = max(
                math.ceil(
                    inputs.particle_count
                    * (points[index].handover_transfer_standard_error / target)
                    ** 2
                )
                for index in refine_indices
            )
            refined_count = min(
                inputs.adaptive_max_particle_count, required_count
            )
            if refined_count > inputs.particle_count:
                refined_points = len(refine_indices)
                if progress is not None:
                    progress(
                        f"{transport.atom_label}: 自适应粒子加密——"
                        f"{refined_points} 个点交接率标准误超过 {target:g}，"
                        f"以每点 N={refined_count} 复算"
                    )
                if use_gpu_batch:
                    pending2 = [
                        (
                            index,
                            trace_by_index[index],
                            replace(
                                params_by_index[index],
                                particle_count=refined_count,
                            ),
                        )
                        for index in refine_indices
                    ]
                    results2 = run_handover_monte_carlo_batch(
                        [parameters for _, _, parameters in pending2],
                        backend="gpu",
                        progress=progress,
                    )
                    refined_items = [
                        (index, _summary(inputs, trace, result))
                        for (index, trace, _), result in zip(pending2, results2)
                    ]
                else:
                    refine_tasks = [
                        (
                            index,
                            inputs,
                            trace_by_index[index],
                            params_by_index[index],
                            refined_count,
                        )
                        for index in refine_indices
                    ]
                    if use_processes:
                        refined_items = []
                        with ProcessPoolExecutor(max_workers=workers) as executor:
                            futures = [
                                executor.submit(_refine_task, task)
                                for task in refine_tasks
                            ]
                            for future in as_completed(futures):
                                refined_items.append(future.result())
                    else:
                        refined_items = [
                            _refine_task(task) for task in refine_tasks
                        ]
                for index, point in refined_items:
                    if point is None:
                        # 加密失败：保留原有网格点结果，不中断扫描。
                        continue
                    points[index] = point
                    efficiency[index] = point.handover_transfer_efficiency
                    standard_error[index] = point.handover_transfer_standard_error
                    handover_heating[index] = (
                        np.nan
                        if point.handover_heating_uK is None
                        else point.handover_heating_uK
                    )
                    total_heating[index] = (
                        np.nan
                        if point.total_temperature_rise_uK is None
                        else point.total_temperature_rise_uK
                    )
                    final_retention[index] = point.final_retention_from_mot
                if progress is not None:
                    progress(
                        f"{transport.atom_label}: 自适应粒子加密完成 "
                        f"{refined_points}/{refined_points}"
                    )

    def optional_matrix(array: np.ndarray) -> tuple[tuple[float | None, ...], ...]:
        return tuple(
            tuple(None if math.isnan(value) else float(value) for value in row)
            for row in array
        )

    point_grid = tuple(
        tuple(
            points.get((power_index, detuning_index))
            for detuning_index in range(transport.detuning_points)
        )
        for power_index in range(transport.power_points)
    )
    if progress is not None and failed_points:
        progress(
            f"{transport.atom_label}: {len(failed_points)} 个网格点计算失败，"
            "已按无效点（NaN）处理"
        )

    def _scan_result(
        best: L1HandoverPoint,
        comparison: L1HandoverPoint,
        best_simulation: L1HandoverPointSimulation | None,
        comparison_simulation: L1HandoverPointSimulation | None,
    ) -> L1HandoverScanResult:
        return L1HandoverScanResult(
            inputs=inputs,
            detuning_ghz=tuple(float(value) for value in detunings),
            source_power_w=tuple(float(value) for value in powers),
            transport_feasible=tuple(
                tuple(bool(value) for value in row) for row in feasible
            ),
            handover_transfer_efficiency=optional_matrix(efficiency),
            handover_transfer_standard_error=optional_matrix(standard_error),
            handover_heating_uK=optional_matrix(handover_heating),
            total_temperature_rise_uK=optional_matrix(total_heating),
            final_retention_from_mot=optional_matrix(final_retention),
            evaluated_points=total,
            optimal=best,
            comparison=comparison,
            optimal_simulation=best_simulation,
            comparison_simulation=comparison_simulation,
            point_grid=point_grid,
            refined_points=refined_points,
        )

    valid_indices = [
        index
        for index, point in points.items()
        if point.total_temperature_rise_uK is not None
    ]
    if not valid_indices:
        # 全网格无捕获/无存活点：不抛错中断扫描，返回哨兵结果（热力图
        # 全 NaN、最优/较差为哨兵点、simulation 为 None），保证总能出图。
        if progress is not None:
            progress(
                f"{transport.atom_label}: 所有网格点均无可选工作点，"
                "返回空结果（矩阵 NaN）"
            )
        failed_point = failed_handover_point(inputs)
        return _scan_result(failed_point, failed_point, None, None)
    heats = np.asarray(
        [points[index].total_temperature_rise_uK for index in valid_indices],
        dtype=float,
    )
    losses = np.asarray(
        [1.0 - points[index].final_retention_from_mot for index in valid_indices],
        dtype=float,
    )
    weight_sum = transport.temperature_weight + transport.retention_weight
    cost = (
        transport.temperature_weight / weight_sum * _normalized(heats)
        + transport.retention_weight / weight_sum * _normalized(losses)
    )
    best_index = valid_indices[int(np.argmin(cost))]
    comparison_index = valid_indices[int(np.argmax(cost))]
    best = points[best_index]
    comparison = points[comparison_index]
    if progress is not None:
        progress(
            f"{transport.atom_label}: 选点完成，正在为最优/较差工作点"
            "生成完整轨迹"
        )
    try:
        best_simulation = simulate_l1_handover_point(
            inputs,
            best.detuning_ghz,
            best.source_power_w,
        )
    except Exception:  # noqa: BLE001 - 完整轨迹重算失败时仍保留网格点结果
        best_simulation = None
    try:
        comparison_simulation = simulate_l1_handover_point(
            inputs,
            comparison.detuning_ghz,
            comparison.source_power_w,
        )
    except Exception:  # noqa: BLE001 - 完整轨迹重算失败时仍保留网格点结果
        comparison_simulation = None
    return _scan_result(
        best,
        comparison,
        best_simulation,
        comparison_simulation,
    )
