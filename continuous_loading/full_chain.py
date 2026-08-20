"""MOT→L1→handover→L2→科学区 全链路编排与二维扫描。

本模块把三段已有计算串成完整链路：

1. ``simulate_l1_transport``：L1 宏观运输（时序、升温、统计留存）；
2. ``run_handover_monte_carlo``：三维经典轨迹 Monte Carlo 交接；
3. ``simulate_l2_transport``：L2 宏观运输腿与科学区原子库汇总。

网格扫描直接复用 ``analyze_l1_handover_scan`` 的 Monte Carlo 结果
（解析 L2 腿只是毫秒级标量积分，逐点补算，不重复任何轨迹模拟），
再按科学区末态的总升温和总损失重新选择工作点。当运输腿为轨迹级
Monte Carlo 且后端为 GPU 时，全部候选点的 L2 腿改由
``transport_batch.run_leg_monte_carlo_batch`` 一次批量完成（各点
初温/原子数不同，属于该接口的逐点初态白名单字段），避免逐点单点
GPU 的每步固定开销；批量不可用（如 conveyor 几何）时回退逐点。

温度口径注意：默认 (N,T) 约化接口仍可用 handover 捕获样本总能量
温度；相空间连续接口没有插入瞬时再热化步骤，因此 L2 腿初温摘要与
轨迹都采用实际传入集合的三维动能温度。
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import os
import math
import re
from typing import Callable

import numpy as np

from .l1_handover import (
    L1HandoverInputs,
    L1HandoverPoint,
    L1HandoverPointSimulation,
    _combined_trace as _l1_handover_combined_trace,
    _handover_parameters,
    _sample_l1_initial_ensemble,
    _summary as _l1_handover_summary,
    _validate_transport_trace,
    analyze_l1_handover_scan,
    failed_handover_point,
    simulate_l1_handover_point,
    simulate_l1_handover_point_continuous,
)
from .l1_transport import L1TransportInputs
from .l2_transport import (
    L2TransportInputs,
    L2TransportResult,
    l2_end_source_power_w,
    l2_leg_inputs,
    l2_result_from_leg_trace,
    simulate_l2_transport,
)
from .transport_batch import run_leg_monte_carlo_batch
from .handover_batch import run_handover_monte_carlo_batch
from .phase_space import (
    handover_to_l2_local,
    l1_transport_end_to_handover,
)


@dataclass(frozen=True)
class FullChainInputs:
    """全链路的 L1、handover 和 L2 输入。"""

    handover: L1HandoverInputs = L1HandoverInputs()
    l2: L2TransportInputs = L2TransportInputs()
    # 默认启用相空间连续传递（单系综贯穿 L1→handover→L2，初态为 L1
    # 起点静止晶格热平衡系综）；显式关闭后回到阶段边界的 (N,T) 约化
    # 接口。GPU 扫描按 L1/handover/L2 三段固定形状批量推进，粒子集合
    # 只在阶段边界按网格索引传递。
    phase_space_continuity: bool = True

@dataclass(frozen=True)
class FullChainPoint:
    """一个失谐--功率点的全链路汇总。"""

    detuning_ghz: float
    source_power_w: float
    l1_handover: L1HandoverPoint
    l2_final_temperature_uK: float | None
    l2_temperature_rise_uK: float | None
    l2_retention_fraction: float | None
    l2_cumulative_scattering_events: float | None
    science_atom_number: float | None
    science_peak_density_m3: float | None
    science_atoms_per_site: float | None
    science_total_temperature_rise_uK: float | None
    final_retention_from_mot: float


@dataclass(frozen=True)
class FullChainCombinedTrace:
    """L1、handover、L2 拼接后的低带宽时间轨迹。

    LGM 装载模块已移除，轨迹从 L1 起点开始；``loading_start_ms``/
    ``loading_end_ms`` 已 deprecated，恒为 None，仅为 UI 时间线消费者
    暂时保留。
    """

    time_ms: tuple[float, ...]
    phase: tuple[str, ...]
    temperature_uK: tuple[float, ...]
    retention_from_mot: tuple[float, ...]
    handover_start_ms: float
    handover_end_ms: float
    l2_start_ms: float
    l2_end_ms: float
    loading_start_ms: float | None = None
    loading_end_ms: float | None = None
    calculation_boundary: str = "static_lattice_thermal"


@dataclass(frozen=True)
class FullChainPointSimulation:
    point: FullChainPoint
    l1_handover_simulation: L1HandoverPointSimulation
    l2_result: L2TransportResult | None
    combined_trace: FullChainCombinedTrace
    interface_mode: str = "reduced_temperature_number"


@dataclass(frozen=True)
class FullChainScanResult:
    inputs: FullChainInputs
    detuning_ghz: tuple[float, ...]
    source_power_w: tuple[float, ...]
    transport_feasible: tuple[tuple[bool, ...], ...]
    handover_transfer_efficiency: tuple[tuple[float | None, ...], ...]
    science_final_temperature_uK: tuple[tuple[float | None, ...], ...]
    science_total_temperature_rise_uK: tuple[tuple[float | None, ...], ...]
    final_retention_from_mot: tuple[tuple[float | None, ...], ...]
    science_peak_density_m3: tuple[tuple[float | None, ...], ...]
    evaluated_points: int
    optimal: FullChainPoint
    comparison: FullChainPoint
    # 全网格失败时 optimal/comparison 为哨兵点且 simulation 为 None
    # （绘图/导出/CLI 均已按 None 分支处理）。
    optimal_simulation: FullChainPointSimulation | None = None
    comparison_simulation: FullChainPointSimulation | None = None


@dataclass(frozen=True)
class _L2LegPlan:
    """一个候选网格点的 L2 腿计算计划（初态来自该点 handover 捕获样本）。"""

    power_index: int
    detuning_index: int
    detuning_ghz: float
    source_power_w: float
    end_source_power_w: float
    captured_temperature_uK: float
    captured_atom_number: float
    leg_inputs: L1TransportInputs


def _l2_boundary_acceleration(l2: L2TransportInputs) -> float:
    """L2 轨迹在 handover 交接时刻（t=0）的瞬时加速度。

    handover 捕获判据的倾斜势垒只应使用交接后 L2 的瞬时加速度：
    minimum_jerk 端点加速度为 0（后续加速逃逸由 L2 腿的瞬时势垒检查
    负责）；实测波形取 t=0 采样值；梯形取恒定加速度。
    """
    if l2.control_waveform is not None:
        return float(l2.control_waveform.sample(0.0)["acceleration_m_s2"])
    if l2.kinematic_profile == "minimum_jerk":
        return 0.0
    return l2.acceleration_m_s2


def failed_full_chain_point(inputs: FullChainInputs) -> FullChainPoint:
    """全网格失败时的哨兵全链路点（见 ``l1_handover.failed_handover_point``）。"""
    l1_point = failed_handover_point(inputs.handover)
    return FullChainPoint(
        detuning_ghz=l1_point.detuning_ghz,
        source_power_w=l1_point.source_power_w,
        l1_handover=l1_point,
        l2_final_temperature_uK=None,
        l2_temperature_rise_uK=None,
        l2_retention_fraction=None,
        l2_cumulative_scattering_events=None,
        science_atom_number=None,
        science_peak_density_m3=None,
        science_atoms_per_site=None,
        science_total_temperature_rise_uK=None,
        final_retention_from_mot=0.0,
    )


def _failed_full_chain_scan_result(
    inputs: FullChainInputs,
    detunings,
    powers,
    feasible,
) -> FullChainScanResult:
    """全网格失败时的空结果：矩阵 NaN/0 + 哨兵点 + simulation None。

    保证 CLI/UI/绘图总能生成热力图（全灰）、JSON 与 PNG，而不抛异常。
    """
    transport = inputs.handover.transport
    # 矩阵形状由传入的网格坐标推导（与 detunings/powers 严格一致），
    # 不依赖 transport 的默认网格点数。
    shape = (len(powers), len(detunings))
    empty = np.full(shape, np.nan)
    zeros = np.zeros(shape)

    def optional_matrix(array: np.ndarray) -> tuple[tuple[float | None, ...], ...]:
        return tuple(
            tuple(None if math.isnan(value) else float(value) for value in row)
            for row in array
        )

    point = failed_full_chain_point(inputs)
    return FullChainScanResult(
        inputs=inputs,
        detuning_ghz=tuple(float(value) for value in detunings),
        source_power_w=tuple(float(value) for value in powers),
        transport_feasible=tuple(
            tuple(bool(value) for value in row) for row in feasible
        ),
        handover_transfer_efficiency=optional_matrix(empty),
        science_final_temperature_uK=optional_matrix(empty),
        science_total_temperature_rise_uK=optional_matrix(empty),
        final_retention_from_mot=optional_matrix(zeros),
        science_peak_density_m3=optional_matrix(empty),
        evaluated_points=0,
        optimal=point,
        comparison=point,
        optimal_simulation=None,
        comparison_simulation=None,
    )


def _full_chain_point(
    inputs: FullChainInputs,
    l1_point: L1HandoverPoint,
    l2_result: L2TransportResult | None,
) -> FullChainPoint:
    if l2_result is None:
        return FullChainPoint(
            detuning_ghz=l1_point.detuning_ghz,
            source_power_w=l1_point.source_power_w,
            l1_handover=l1_point,
            l2_final_temperature_uK=None,
            l2_temperature_rise_uK=None,
            l2_retention_fraction=None,
            l2_cumulative_scattering_events=None,
            science_atom_number=None,
            science_peak_density_m3=None,
            science_atoms_per_site=None,
            science_total_temperature_rise_uK=None,
            final_retention_from_mot=0.0,
        )
    return FullChainPoint(
        detuning_ghz=l1_point.detuning_ghz,
        source_power_w=l1_point.source_power_w,
        l1_handover=l1_point,
        l2_final_temperature_uK=l2_result.final_temperature_uK,
        l2_temperature_rise_uK=l2_result.leg_temperature_rise_uK,
        l2_retention_fraction=l2_result.leg_retention_fraction,
        l2_cumulative_scattering_events=l2_result.cumulative_scattering_events,
        science_atom_number=l2_result.science.atom_number,
        science_peak_density_m3=l2_result.science.peak_density_m3,
        science_atoms_per_site=l2_result.science.atoms_per_site,
        science_total_temperature_rise_uK=(
            l2_result.final_temperature_uK
            - (
                inputs.handover.transport.initial_temperature_uK
                if l1_point.transport.initial_temperature_uK is None
                else l1_point.transport.initial_temperature_uK
            )
        ),
        final_retention_from_mot=(
            l1_point.final_retention_from_mot
            * l2_result.leg_retention_fraction
        ),
    )


def _run_l2_for_point(
    inputs: FullChainInputs,
    l1_point: L1HandoverPoint,
) -> L2TransportResult | None:
    """handover 有捕获原子时才运行 L2 腿。"""
    if l1_point.final_temperature_uK is None:
        return None
    if l1_point.final_atom_number <= 0.0:
        return None
    return simulate_l2_transport(
        inputs.handover.transport,
        inputs.l2,
        l1_point.detuning_ghz,
        l1_point.source_power_w,
        l1_point.final_temperature_uK,
        l1_point.final_atom_number,
    )


def _combined_trace(
    inputs: FullChainInputs,
    simulation: L1HandoverPointSimulation,
    l2_result: L2TransportResult | None,
) -> FullChainCombinedTrace:
    base = simulation.combined_trace
    time_ms = list(base.time_ms)
    phase = list(base.phase)
    temperature = list(base.temperature_uK)
    retention = list(base.retention_from_mot)
    l2_start = base.handover_end_ms
    l2_end = base.handover_end_ms
    if l2_result is not None:
        handover_retention = retention[-1]
        l2_times = [l2_start + value for value in l2_result.leg_trace.time_ms]
        l2_retention = [
            handover_retention * value
            for value in l2_result.leg_trace.retention_fraction
        ]
        l2_temperature = list(l2_result.leg_trace.temperature_uK)
        if inputs.phase_space_continuity and l2_temperature and temperature:
            # 相同捕获集合在固定粒子数 GPU 接口处会做低方差重采样；
            # 首点沿用 handover 末点，避免把有限样本波动画成物理脉冲。
            l2_temperature[0] = temperature[-1]
        time_ms.extend(l2_times)
        phase.extend(["L2 transport"] * len(l2_times))
        temperature.extend(l2_temperature)
        retention.extend(l2_retention)
        l2_end = l2_times[-1]
    return FullChainCombinedTrace(
        time_ms=tuple(time_ms),
        phase=tuple(phase),
        temperature_uK=tuple(temperature),
        retention_from_mot=tuple(retention),
        handover_start_ms=base.handover_start_ms,
        handover_end_ms=base.handover_end_ms,
        l2_start_ms=l2_start,
        l2_end_ms=l2_end,
        loading_start_ms=base.loading_start_ms,
        loading_end_ms=base.loading_end_ms,
        calculation_boundary=base.calculation_boundary,
    )


def simulate_full_chain_point(
    inputs: FullChainInputs,
    detuning_ghz: float,
    source_power_w: float,
    *,
    trace_points: int | None = None,
) -> FullChainPointSimulation:
    """运行一个失谐--功率点的完整 L1→handover→L2 链路。"""
    if inputs.phase_space_continuity:
        if (
            inputs.l2.control_waveform is None
            and inputs.l2.kinematic_profile != "minimum_jerk"
        ):
            raise ValueError(
                "连续相空间模式禁止加速度阶跃：L2 请使用 minimum_jerk "
                "或提供连续的实测控制波形"
            )
        simulation, captured_ensemble = (
            simulate_l1_handover_point_continuous(
                inputs.handover,
                detuning_ghz,
                source_power_w,
                trace_points=trace_points,
                post_handover_acceleration_m_s2=_l2_boundary_acceleration(
                    inputs.l2
                ),
            )
        )
        l2_result = None
        if (
            captured_ensemble is not None
            and simulation.point.final_temperature_uK is not None
            and simulation.point.final_atom_number > 0.0
        ):
            from .transport_mc import simulate_leg_monte_carlo

            l2_initial = handover_to_l2_local(
                captured_ensemble, inputs.handover.crossing_angle_deg
            )
            handover_kinetic = (
                simulation.handover_result.final_kinetic_temperature_uK
            )
            l2_initial_temperature = (
                simulation.point.final_temperature_uK
                if handover_kinetic is None
                else handover_kinetic
            )
            end_source_power = l2_end_source_power_w(
                inputs.handover.transport,
                inputs.l2,
                source_power_w,
            )
            leg_trace = simulate_leg_monte_carlo(
                l2_leg_inputs(
                    inputs.handover.transport,
                    inputs.l2,
                    l2_initial_temperature,
                    simulation.point.final_atom_number,
                ),
                detuning_ghz,
                end_source_power,
                initial_ensemble=l2_initial,
            )
            l2_result = l2_result_from_leg_trace(
                inputs.handover.transport,
                inputs.l2,
                detuning_ghz,
                end_source_power,
                l2_initial_temperature,
                simulation.point.final_atom_number,
                leg_trace,
            )
        return FullChainPointSimulation(
            point=_full_chain_point(inputs, simulation.point, l2_result),
            l1_handover_simulation=simulation,
            l2_result=l2_result,
            combined_trace=_combined_trace(inputs, simulation, l2_result),
            interface_mode="phase_space_continuous",
        )
    simulation = simulate_l1_handover_point(
        inputs.handover,
        detuning_ghz,
        source_power_w,
        trace_points=trace_points,
    )
    l2_result = _run_l2_for_point(inputs, simulation.point)
    return FullChainPointSimulation(
        point=_full_chain_point(inputs, simulation.point, l2_result),
        l1_handover_simulation=simulation,
        l2_result=l2_result,
        combined_trace=_combined_trace(inputs, simulation, l2_result),
    )


def _normalized(values: np.ndarray) -> np.ndarray:
    span = float(np.max(values) - np.min(values))
    if span <= 1e-15:
        return np.zeros_like(values)
    return (values - np.min(values)) / span


def _continuous_scan_point_task(task):
    index, inputs, detuning, power = task
    try:
        simulation = simulate_full_chain_point(
            inputs, detuning, power, trace_points=2
        )
    except Exception as exc:  # noqa: BLE001 - 单点异常按该点无有效结果处理
        # 异常原因一并返回，由扫描层汇总报告（此前静默丢弃，网格
        # 大面积无有效结果时无法定位原因）。
        return index, None, f"{type(exc).__name__}: {exc}"
    return index, simulation, None


def _report_point_failures(failures: Counter, progress, label: str) -> None:
    """把逐点失败原因 Top-3 经 progress 汇报（诊断可见性）。"""
    if not failures or progress is None:
        return
    details = "；".join(
        f"{reason} ×{count}" for reason, count in failures.most_common(3)
    )
    progress(
        f"{label}: {sum(failures.values())} 个网格点无有效结果，"
        f"主要原因：{details}"
    )


# 进度消息中的 "done/total" 计数（多阶段扫描的全局进度折算用）。
_PROGRESS_FRACTION = re.compile(r"(\d+)\s*/\s*(\d+)")


def _stage_weighted_progress(progress, label, stage_index, stage_count, total):
    """把单阶段 "done/total" 进度折算成全局网格进度 "x/total"。

    连续相空间 GPU 扫描按 初态采样→L1→handover→L2 四阶段批量推进，
    各阶段独立计数会让进度条在阶段切换时回退或提前到 100%；统一
    折算为全局完成点数（stage_index 为 0 基阶段序号）。
    """
    if progress is None:
        return None

    def wrapped(message: str) -> None:
        match = _PROGRESS_FRACTION.search(message)
        if match:
            done, sub_total = int(match.group(1)), int(match.group(2))
            if sub_total > 0:
                fraction = (
                    stage_index + min(done / sub_total, 1.0)
                ) / stage_count
                progress(
                    f"{label}: 连续相空间网格 "
                    f"{fraction * total:.0f}/{total}（{message}）"
                )
                return
        progress(message)

    return wrapped


def _continuous_scan_grid(inputs: FullChainInputs):
    transport = inputs.handover.transport
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
    shape = (transport.power_points, transport.detuning_points)
    feasible = np.zeros(shape, dtype=bool)
    tasks = []
    precheck = replace(transport, transport_method='analytic')
    from .l1_transport import simulate_l1_transport

    for power_index, power in enumerate(powers):
        for detuning_index, detuning in enumerate(detunings):
            if power <= 0.0:
                continue
            try:
                trace = simulate_l1_transport(
                    precheck, float(detuning), float(power)
                )
            except Exception:  # noqa: BLE001 - 预检失败点按无任务跳过
                continue
            feasible[power_index, detuning_index] = (
                trace.point.feasible_hardware_point
            )
            if trace.point.feasible_hardware_point:
                tasks.append(
                    (
                        (power_index, detuning_index),
                        inputs,
                        float(detuning),
                        float(power),
                    )
                )
    return detunings, powers, feasible, tasks


def _finish_continuous_scan(
    inputs, detunings, powers, feasible, points, efficiencies, progress
):
    shape = feasible.shape
    science_temperature = np.full(shape, np.nan)
    science_heating = np.full(shape, np.nan)
    final_retention = np.full(shape, np.nan)
    science_density = np.full(shape, np.nan)
    handover_efficiency = np.full(shape, np.nan)
    for index, point in points.items():
        science_temperature[index] = point.l2_final_temperature_uK
        science_heating[index] = point.science_total_temperature_rise_uK
        final_retention[index] = point.final_retention_from_mot
        science_density[index] = point.science_peak_density_m3
        handover_efficiency[index] = efficiencies[index]
    if not points:
        # 全网格失败：不抛错，返回空结果（矩阵 NaN + 哨兵点），保证
        # 热力图/JSON/PNG 总能生成。
        if progress is not None:
            progress(
                f"{inputs.handover.transport.atom_label}: 所有网格点均未"
                "完成连续相空间全链路，返回空结果（矩阵 NaN）"
            )
        return _failed_full_chain_scan_result(
            inputs, detunings, powers, feasible
        )
    valid_indices = [
        index
        for index, point in points.items()
        if point.science_total_temperature_rise_uK is not None
        and math.isfinite(point.science_total_temperature_rise_uK)
        and point.final_retention_from_mot > 0.0
    ]
    if not valid_indices:
        if progress is not None:
            progress(
                f"{inputs.handover.transport.atom_label}: 所有网格点均无"
                "有效连续相空间末态，返回空结果（矩阵 NaN）"
            )
        return _failed_full_chain_scan_result(
            inputs, detunings, powers, feasible
        )
    transport = inputs.handover.transport
    heats = np.asarray(
        [points[index].science_total_temperature_rise_uK for index in valid_indices]
    )
    losses = np.asarray(
        [1.0 - points[index].final_retention_from_mot for index in valid_indices]
    )
    weight_sum = transport.temperature_weight + transport.retention_weight
    cost = (
        transport.temperature_weight / weight_sum * _normalized(heats)
        + transport.retention_weight / weight_sum * _normalized(losses)
    )
    best_index = valid_indices[int(np.argmin(cost))]
    comparison_index = valid_indices[int(np.argmax(cost))]
    if progress is not None:
        progress(f'{transport.atom_label}: 连续相空间扫描选点完成')
    try:
        best_simulation = simulate_full_chain_point(
            inputs,
            points[best_index].detuning_ghz,
            points[best_index].source_power_w,
        )
    except Exception:  # noqa: BLE001 - 完整轨迹重算失败时仍保留网格点结果
        best_simulation = None
    try:
        comparison_simulation = simulate_full_chain_point(
            inputs,
            points[comparison_index].detuning_ghz,
            points[comparison_index].source_power_w,
        )
    except Exception:  # noqa: BLE001 - 完整轨迹重算失败时仍保留网格点结果
        comparison_simulation = None

    def optional_matrix(array):
        return tuple(
            tuple(None if math.isnan(value) else float(value) for value in row)
            for row in array
        )

    return FullChainScanResult(
        inputs=inputs,
        detuning_ghz=tuple(float(value) for value in detunings),
        source_power_w=tuple(float(value) for value in powers),
        transport_feasible=tuple(
            tuple(bool(value) for value in row) for row in feasible
        ),
        handover_transfer_efficiency=optional_matrix(handover_efficiency),
        science_final_temperature_uK=optional_matrix(science_temperature),
        science_total_temperature_rise_uK=optional_matrix(science_heating),
        final_retention_from_mot=optional_matrix(final_retention),
        science_peak_density_m3=optional_matrix(science_density),
        evaluated_points=len(points),
        optimal=best_simulation.point,
        comparison=comparison_simulation.point,
        optimal_simulation=best_simulation,
        comparison_simulation=comparison_simulation,
    )


def _analyze_continuous_scan_cpu(inputs, progress):
    detunings, powers, feasible, tasks = _continuous_scan_grid(inputs)
    points = {}
    efficiencies = {}
    failures: Counter = Counter()
    handover = inputs.handover
    use_processes = (
        handover.parallel_backend == 'process'
        and handover.worker_count > 1
        and len(tasks) > 1
    )

    def store(completed, item):
        index, simulation, error = item
        if simulation is not None and simulation.point.l2_final_temperature_uK is not None:
            points[index] = simulation.point
            efficiencies[index] = (
                simulation.point.l1_handover.handover_transfer_efficiency
            )
        elif error is not None:
            failures[error] += 1
        elif simulation is not None:
            failures["L2 无有效末态（handover 零捕获或全灭）"] += 1
        if progress is not None:
            progress(
                f'{handover.transport.atom_label}: 连续相空间网格 '
                f'{completed}/{len(tasks)}'
            )

    if use_processes:
        workers = min(
            handover.worker_count, len(tasks), os.cpu_count() or 1
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_continuous_scan_point_task, task)
                for task in tasks
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                store(completed, future.result())
    else:
        for completed, task in enumerate(tasks, start=1):
            store(completed, _continuous_scan_point_task(task))
    _report_point_failures(
        failures, progress, handover.transport.atom_label
    )
    return _finish_continuous_scan(
        inputs, detunings, powers, feasible, points, efficiencies, progress
    )


def _analyze_continuous_scan_gpu(inputs, progress):
    detunings, powers, feasible, grid_tasks = _continuous_scan_grid(inputs)
    transport = inputs.handover.transport
    if not grid_tasks:
        # 无可行网格点：不抛错，返回空结果。
        if progress is not None:
            progress(
                f'{transport.atom_label}: 连续相空间扫描无可行网格点，'
                '返回空结果'
            )
        return _failed_full_chain_scan_result(
            inputs, detunings, powers, feasible
        )
    indices = [task[0] for task in grid_tasks]
    pairs = [(task[2], task[3]) for task in grid_tasks]
    # 初态来源：逐点 CPU 静止 L1 晶格热平衡采样（LGM 装载模块已移除），
    # 系综经 initial_ensembles 喂入现有 GPU 批量 kernel（kernel 不动）。
    # 四阶段（采样/L1/handover/L2）进度统一折算为全局网格完成点数。
    stage_count = 4
    total_points = len(grid_tasks)
    sampling_progress = _stage_weighted_progress(
        progress, transport.atom_label, 0, stage_count, total_points
    )
    if sampling_progress is not None:
        sampling_progress(f'CPU 初态热平衡采样 0/{total_points}')
    leg_tasks = []
    initial_ensembles = {}
    sampling_failures = 0
    for sample_index, (index, (detuning, power)) in enumerate(
        zip(indices, pairs), start=1
    ):
        try:
            initial_ensembles[index] = _sample_l1_initial_ensemble(
                transport, detuning, power
            )
        except Exception:  # noqa: BLE001 - 浅阱/高温无束缚初态按该点无效处理
            sampling_failures += 1
            continue
        leg_tasks.append((index, transport, detuning, power))
        if sampling_progress is not None and (
            sample_index == len(grid_tasks)
            or sample_index % max(1, len(grid_tasks) // 10) == 0
        ):
            sampling_progress(
                f'CPU 初态热平衡采样 {sample_index}/{len(grid_tasks)}'
            )
    if sampling_failures and progress is not None:
        progress(
            f'{transport.atom_label}: {sampling_failures} 个网格点'
            '无法采样束缚初态（浅阱/高温），按无效点处理'
        )
    if not leg_tasks:
        # 全部网格点都无法采到束缚初态：原子视为全部在入口丢失，
        # 不抛错，返回空结果。
        if progress is not None:
            progress(
                f'{transport.atom_label}: 连续相空间扫描无可采样束缚'
                '初态的网格点，返回空结果（矩阵 NaN）'
            )
        return _failed_full_chain_scan_result(
            inputs, detunings, powers, feasible
        )
    l1_progress = _stage_weighted_progress(
        progress, transport.atom_label, 1, stage_count, total_points
    )
    if l1_progress is not None:
        l1_progress(f'GPU 批量 L1 运输 0/{len(leg_tasks)}')
    transport_traces, transport_ensembles = run_leg_monte_carlo_batch(
        leg_tasks,
        backend='gpu',
        progress=l1_progress,
        initial_ensembles=initial_ensembles,
        return_final_ensembles=True,
    )

    handover_parameters = []
    handover_initial = []
    handover_records = []
    l1_invalid_count = 0
    for task, trace, ensemble in zip(
        leg_tasks, transport_traces, transport_ensembles
    ):
        index, _, detuning, power = task
        trace = replace(
            trace,
            pre_ramp_survival_fraction=transport.pre_ramp_survival_fraction,
        )
        try:
            _validate_transport_trace(trace)
        except ValueError:
            l1_invalid_count += 1
            continue
        if ensemble is None:
            l1_invalid_count += 1
            continue
        handover_records.append((index, detuning, power, trace))
        handover_initial.append(
            l1_transport_end_to_handover(
                ensemble, transport.distance_m
            )
        )
        try:
            handover_parameters.append(
                _handover_parameters(
                    inputs.handover,
                    trace,
                    trace_points=2,
                    post_handover_acceleration_m_s2=_l2_boundary_acceleration(
                        inputs.l2
                    ),
                )
            )
        except ValueError:
            # 末态参数退化（温度/阱深非有限正数等）按该点无效处理：
            # 与 CPU 逐点路径的失败隔离口径一致，不中断整批扫描。
            handover_records.pop()
            handover_initial.pop()
            l1_invalid_count += 1
            continue
    if l1_invalid_count and progress is not None:
        progress(
            f'{transport.atom_label}: {l1_invalid_count} 个网格点在 L1 '
            '末端无存活原子或末态退化，按无效点处理'
        )
    if not handover_parameters:
        # L1 运输后无存活点：原子全部在 L1 环节丢失，返回空结果。
        if progress is not None:
            progress(
                f'{transport.atom_label}: 连续相空间扫描在 L1 运输后'
                '无存活点，返回空结果（矩阵 NaN）'
            )
        return _failed_full_chain_scan_result(
            inputs, detunings, powers, feasible
        )
    handover_progress = _stage_weighted_progress(
        progress, transport.atom_label, 2, stage_count, total_points
    )
    if handover_progress is not None:
        handover_progress(f'GPU 批量 handover 0/{len(handover_parameters)}')
    handover_results, captured_ensembles = run_handover_monte_carlo_batch(
        handover_parameters,
        backend='gpu',
        progress=handover_progress,
        initial_ensembles=handover_initial,
        return_captured_ensembles=True,
    )

    l2_tasks = []
    l2_initial = {}
    l2_meta = {}
    efficiencies = {}
    for record, result, ensemble in zip(
        handover_records, handover_results, captured_ensembles
    ):
        index, detuning, power, trace = record
        if ensemble is None or result.final_temperature_uK is None:
            continue
        l1_point = _l1_handover_summary(
            inputs.handover, trace, result
        )
        if l1_point.final_atom_number <= 0.0:
            continue
        temperature = (
            result.final_temperature_uK
            if result.final_kinetic_temperature_uK is None
            else result.final_kinetic_temperature_uK
        )
        end_power = l2_end_source_power_w(
            transport, inputs.l2, power
        )
        leg_inputs = l2_leg_inputs(
            transport, inputs.l2, temperature, l1_point.final_atom_number
        )
        l2_tasks.append(
            (index, leg_inputs, detuning, end_power)
        )
        l2_initial[index] = handover_to_l2_local(
            ensemble, inputs.handover.crossing_angle_deg
        )
        l2_meta[index] = (
            l1_point, temperature, l1_point.final_atom_number, detuning, end_power
        )
        efficiencies[index] = result.transfer_efficiency
    if not l2_tasks:
        # handover 后无 L2 捕获点：原子全部在 handover 环节丢失，返回空结果。
        if progress is not None:
            progress(
                f'{transport.atom_label}: 连续相空间扫描在 handover 后'
                '无 L2 捕获点，返回空结果（矩阵 NaN）'
            )
        return _failed_full_chain_scan_result(
            inputs, detunings, powers, feasible
        )
    l2_progress = _stage_weighted_progress(
        progress, transport.atom_label, 3, stage_count, total_points
    )
    if l2_progress is not None:
        l2_progress(f'GPU 批量 L2 运输 0/{len(l2_tasks)}')
    l2_traces = run_leg_monte_carlo_batch(
        l2_tasks,
        backend='gpu',
        progress=l2_progress,
        initial_ensembles=l2_initial,
    )
    points = {}
    for task, trace in zip(l2_tasks, l2_traces):
        index = task[0]
        l1_point, temperature, atom_number, detuning, end_power = l2_meta[index]
        l2_result = l2_result_from_leg_trace(
            transport,
            inputs.l2,
            detuning,
            end_power,
            temperature,
            atom_number,
            trace,
        )
        points[index] = _full_chain_point(
            inputs, l1_point, l2_result
        )
    return _finish_continuous_scan(
        inputs, detunings, powers, feasible, points, efficiencies, progress
    )


def analyze_full_chain_scan(
    inputs: FullChainInputs = FullChainInputs(),
    *,
    progress: Callable[[str], None] | None = None,
) -> FullChainScanResult:
    """复用 L1→handover 网格结果，补算 L2 腿并重新选点。

    约化接口沿用 L1→handover 网格后补算 L2；连续相空间接口在 CPU
    上按 serial/process 设置逐点运行，在 GPU 上按四个物理阶段批量
    推进，并在阶段边界传递逐点经验粒子集合。
    """
    if inputs.phase_space_continuity:
        transport = inputs.handover.transport
        if transport.transport_method != 'monte_carlo':
            raise ValueError(
                'continuous scan requires transport_method=monte_carlo '
                f"(got {transport.transport_method!r}); set the dataclass "
                "field transport_method='monte_carlo' on L1TransportInputs, "
                'or pass --transport-method monte_carlo on the CLI; '
                'alternatively use phase_space_continuity=False / '
                '--no-phase-space-continuity for the analytic reduced scan'
            )
        if (
            transport.control_waveform is None
            and transport.kinematic_profile != 'minimum_jerk'
        ):
            raise ValueError(
                'continuous L1 scan requires a smooth trajectory; set the '
                "dataclass field kinematic_profile='minimum_jerk' on "
                'L1TransportInputs, or pass --kinematic-profile minimum_jerk '
                'on the CLI (or use phase_space_continuity=False / '
                '--no-phase-space-continuity)'
            )
        if (
            inputs.l2.control_waveform is None
            and inputs.l2.kinematic_profile != 'minimum_jerk'
        ):
            raise ValueError(
                'continuous L2 scan requires a smooth trajectory; set the '
                "dataclass field kinematic_profile='minimum_jerk' on "
                'L2TransportInputs, or pass --kinematic-profile minimum_jerk '
                'on the CLI (or use phase_space_continuity=False / '
                '--no-phase-space-continuity)'
            )
        try:
            if inputs.handover.compute_backend == 'gpu':
                return _analyze_continuous_scan_gpu(inputs, progress)
            return _analyze_continuous_scan_cpu(inputs, progress)
        except Exception:  # noqa: BLE001 - 连续相空间扫描意外失败也返回可绘图结果
            if progress is not None:
                progress(
                    f"{transport.atom_label}: 连续相空间二维扫描发生未预期"
                    "错误，返回空结果（矩阵 NaN）"
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
            feasible = np.zeros(
                (transport.power_points, transport.detuning_points),
                dtype=bool,
            )
            return _failed_full_chain_scan_result(
                inputs, detunings, powers, feasible
            )
    try:
        base = analyze_l1_handover_scan(inputs.handover, progress=progress)
    except Exception:  # noqa: BLE001 - L1→handover 扫描意外失败也返回可绘图结果
        if progress is not None:
            progress(
                f"{inputs.handover.transport.atom_label}: L1→handover 二维"
                "扫描发生未预期错误，返回空结果（矩阵 NaN）"
            )
        transport = inputs.handover.transport
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
        feasible = np.zeros(
            (transport.power_points, transport.detuning_points),
            dtype=bool,
        )
        return _failed_full_chain_scan_result(
            inputs, detunings, powers, feasible
        )
    transport = inputs.handover.transport
    shape = (transport.power_points, transport.detuning_points)
    science_temperature = np.full(shape, np.nan)
    science_heating = np.full(shape, np.nan)
    final_retention = np.full(shape, np.nan)
    science_density = np.full(shape, np.nan)
    points: dict[tuple[int, int], FullChainPoint] = {}

    total_heating = np.asarray(base.total_temperature_rise_uK, dtype=float)
    base_retention = np.asarray(base.final_retention_from_mot, dtype=float)
    evaluated = 0
    # 先收集可进入 L2 的点并预构建各点 L2 腿输入（逐点初温/原子数
    # 来自该点 handover 捕获样本），批量或逐点补算时都能报告进度。
    plans: list[_L2LegPlan] = []
    for power_index, detuning_index in np.ndindex(shape):
        captured_rise = total_heating[power_index, detuning_index]
        if math.isnan(captured_rise):
            continue
        # handover 捕获末温优先取逐点联合结果；point_grid 缺失的旧
        # 结果回退到"网格总升温 + 统一固定初温"回加。
        l1_point = (
            base.point_grid[power_index][detuning_index]
            if base.point_grid is not None
            else None
        )
        if l1_point is not None and l1_point.final_temperature_uK is not None:
            captured_temperature = l1_point.final_temperature_uK
        else:
            captured_temperature = captured_rise + transport.initial_temperature_uK
        captured_number = transport.mot_atom_number * base_retention[
            power_index, detuning_index
        ]
        if captured_number <= 0.0:
            continue
        detuning = float(base.detuning_ghz[detuning_index])
        source_power = float(base.source_power_w[power_index])
        plans.append(
            _L2LegPlan(
                power_index=power_index,
                detuning_index=detuning_index,
                detuning_ghz=detuning,
                source_power_w=source_power,
                end_source_power_w=l2_end_source_power_w(
                    transport, inputs.l2, source_power
                ),
                captured_temperature_uK=captured_temperature,
                captured_atom_number=captured_number,
                leg_inputs=l2_leg_inputs(
                    transport, inputs.l2, captured_temperature, captured_number
                ),
            )
        )
    l2_total = len(plans)
    # 轨迹级运输 + GPU 后端时，全部候选点的 L2 腿合并为一次批量调用
    # （初温/原子数逐点不同，属 transport_batch 的逐点初态白名单）；
    # 解析腿（毫秒级）或 CPU 后端保持逐点，批量未覆盖的情形回退逐点。
    batched_leg_traces: list | None = None
    if (
        transport.transport_method == "monte_carlo"
        and inputs.handover.compute_backend == "gpu"
        and plans
    ):
        if progress is not None:
            progress(
                f"{transport.atom_label}: 正在 GPU 批量运行 {l2_total} "
                "个点的 L2 运输 Monte Carlo（首次需编译内核）"
            )
        try:
            batched_leg_traces = run_leg_monte_carlo_batch(
                [
                    (
                        (plan.power_index, plan.detuning_index),
                        plan.leg_inputs,
                        plan.detuning_ghz,
                        plan.end_source_power_w,
                    )
                    for plan in plans
                ],
                backend="gpu",
                progress=(
                    None
                    if progress is None
                    else lambda message: progress(
                        f"{transport.atom_label}: L2 {message}"
                    )
                ),
            )
        except ValueError as exc:
            if progress is not None:
                progress(
                    f"{transport.atom_label}: 批量 L2 运输腿不可用"
                    f"（{exc}），回退逐点计算"
                )
            batched_leg_traces = None
    for candidate_index, plan in enumerate(plans, start=1):
        try:
            if batched_leg_traces is not None:
                l2_result = l2_result_from_leg_trace(
                    transport,
                    inputs.l2,
                    plan.detuning_ghz,
                    plan.end_source_power_w,
                    plan.captured_temperature_uK,
                    plan.captured_atom_number,
                    batched_leg_traces[candidate_index - 1],
                )
            else:
                if progress is not None:
                    progress(
                        f"{transport.atom_label}: L2 运输腿 "
                        f"{candidate_index}/{l2_total}（逐点）"
                    )
                l2_result = simulate_l2_transport(
                    transport,
                    inputs.l2,
                    plan.detuning_ghz,
                    plan.source_power_w,
                    plan.captured_temperature_uK,
                    plan.captured_atom_number,
                )
        except Exception:  # noqa: BLE001 - L2 单点失败按该点原子后续丢失处理
            if progress is not None:
                progress(
                    f"{transport.atom_label}: L2 运输腿 "
                    f"{candidate_index}/{l2_total} 失败，"
                    "按该点原子后续丢失处理"
                )
            continue
        # L1 联合结果直接从 point_grid 复用（含按当前运输方法计算的
        # L1 末态），不再按方法重新计算运输腿。
        l1_point = (
            base.point_grid[plan.power_index][plan.detuning_index]
            if base.point_grid is not None
            else None
        )
        if l1_point is None:
            continue
        point = _full_chain_point(inputs, l1_point, l2_result)
        points[(plan.power_index, plan.detuning_index)] = point
        science_temperature[plan.power_index, plan.detuning_index] = (
            point.l2_final_temperature_uK
        )
        science_heating[plan.power_index, plan.detuning_index] = (
            point.science_total_temperature_rise_uK
        )
        final_retention[plan.power_index, plan.detuning_index] = (
            point.final_retention_from_mot
        )
        science_density[plan.power_index, plan.detuning_index] = (
            point.science_peak_density_m3
        )
        evaluated += 1

    if not points:
        # 全网格无捕获原子：不抛错，返回空结果（矩阵 NaN + 哨兵点），
        # 保证热力图/JSON/PNG 总能生成。
        if progress is not None:
            progress(
                f"{transport.atom_label}: 所有网格点均无捕获原子，"
                "返回空结果（矩阵 NaN）"
            )
        return _failed_full_chain_scan_result(
            inputs,
            base.detuning_ghz,
            base.source_power_w,
            np.asarray(base.transport_feasible, dtype=bool),
        )
    valid_indices = [
        index
        for index, point in points.items()
        if point.science_total_temperature_rise_uK is not None
        and math.isfinite(point.science_total_temperature_rise_uK)
        and point.final_retention_from_mot > 0.0
    ]
    if not valid_indices:
        if progress is not None:
            progress(
                f"{transport.atom_label}: 所有网格点均无有效科学区末态，"
                "返回空结果（矩阵 NaN）"
            )
        return _failed_full_chain_scan_result(
            inputs,
            base.detuning_ghz,
            base.source_power_w,
            np.asarray(base.transport_feasible, dtype=bool),
        )
    heats = np.asarray(
        [points[index].science_total_temperature_rise_uK for index in valid_indices],
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
    if progress is not None:
        progress(
            f"{transport.atom_label}: 全链路选点完成，"
            f"正在为最优/较差工作点生成完整轨迹"
        )
    try:
        best_simulation = simulate_full_chain_point(
            inputs,
            points[best_index].detuning_ghz,
            points[best_index].source_power_w,
        )
    except Exception:  # noqa: BLE001 - 完整轨迹重算失败时仍保留网格点结果
        best_simulation = None
    try:
        comparison_simulation = simulate_full_chain_point(
            inputs,
            points[comparison_index].detuning_ghz,
            points[comparison_index].source_power_w,
        )
    except Exception:  # noqa: BLE001 - 完整轨迹重算失败时仍保留网格点结果
        comparison_simulation = None

    def optional_matrix(array: np.ndarray) -> tuple[tuple[float | None, ...], ...]:
        return tuple(
            tuple(None if math.isnan(value) else float(value) for value in row)
            for row in array
        )

    return FullChainScanResult(
        inputs=inputs,
        detuning_ghz=base.detuning_ghz,
        source_power_w=base.source_power_w,
        transport_feasible=base.transport_feasible,
        handover_transfer_efficiency=base.handover_transfer_efficiency,
        science_final_temperature_uK=optional_matrix(science_temperature),
        science_total_temperature_rise_uK=optional_matrix(science_heating),
        final_retention_from_mot=optional_matrix(final_retention),
        science_peak_density_m3=optional_matrix(science_density),
        evaluated_points=evaluated,
        optimal=best_simulation.point,
        comparison=comparison_simulation.point,
        optimal_simulation=best_simulation,
        comparison_simulation=comparison_simulation,
    )
