"""失谐量--源端功率平面上的分段线性规划设计工具。

底层偶极势、散射率和加速势垒都不是失谐量与功率的严格线性函数。
本模块不改变这些原模型，而是在每个小失谐区间内构造保守的仿射
边界，并通过二维半平面相交求解线性规划可行域。所有推荐点最后再
代回原非线性模型核验。

默认约束包括：

* 静态晶格阱深下限；
* handover 时间内完成指定轴向振荡周数的工程绝热性下限；
* 后续加速度下达到目标热平衡束缚比例；
* 最大散射率；
* 最大源端功率。

``handover_min_axial_cycles`` 是工程判据，不是论文直接测量量。默认
80 周约对应论文 Extended Data Fig. 2b 中 0.3 ms 附近进入平台的
数量级，用于比较不同 handover 时间，不替代轨迹 Monte Carlo。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Iterable

from .atomic import CS133, RB87, AlkaliAtom
from .constants import BOLTZMANN
from .lattice import (
    evaluate_lattice,
    power_for_target_depth,
    tilted_lattice_barrier_fraction,
)
from .transport import thermal_bound_fraction_3d_harmonic


DEFAULT_HANDOVER_TIMES_MS = (0.2, 0.3, 0.4, 1.0)


@dataclass(frozen=True)
class LinearDesignInputs:
    """失谐--功率线性规划的公共输入。"""

    atom_label: str = "Cs-133"
    detuning_min_ghz: float = 300.0
    detuning_max_ghz: float = 1_000.0
    segment_count: int = 28
    waist_um: float = 250.0
    target_depth_uK: float = 500.0
    design_temperature_uK: float = 120.0
    target_bound_fraction: float = 0.80
    acceleration_m_s2: float = 4_000.0
    handover_min_axial_cycles: float = 80.0
    max_source_power_w: float = 6.0
    max_scattering_rate_s: float = 600.0
    delivery_efficiency: float = 0.70
    retro_power_ratio: float = 0.88**4
    detuning_objective_weight: float = 0.05

    def __post_init__(self) -> None:
        _atom_from_label(self.atom_label)
        positive = {
            "最小红失谐": self.detuning_min_ghz,
            "最大红失谐": self.detuning_max_ghz,
            "束腰": self.waist_um,
            "目标阱深": self.target_depth_uK,
            "设计温度": self.design_temperature_uK,
            "handover 轴向振荡周数": self.handover_min_axial_cycles,
            "最大源端功率": self.max_source_power_w,
            "最大散射率": self.max_scattering_rate_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name}必须是有限正数")
        if self.detuning_max_ghz <= self.detuning_min_ghz:
            raise ValueError("最大红失谐必须大于最小红失谐")
        if self.segment_count < 2:
            raise ValueError("线性分段数至少为 2")
        if not 0.0 < self.target_bound_fraction < 1.0:
            raise ValueError("目标束缚比例必须位于 (0, 1)")
        if not math.isfinite(self.acceleration_m_s2):
            raise ValueError("加速度必须是有限数")
        if not 0.0 < self.delivery_efficiency <= 1.0:
            raise ValueError("传输效率必须位于 (0, 1]")
        if not 0.0 <= self.retro_power_ratio <= 1.0:
            raise ValueError("回程/前向功率比必须位于 [0, 1]")
        if (
            not math.isfinite(self.detuning_objective_weight)
            or self.detuning_objective_weight < 0.0
        ):
            raise ValueError("失谐目标权重必须是有限非负数")


@dataclass(frozen=True)
class ConstraintBoundary:
    """一条原非线性约束边界，用于绘图和结果追溯。"""

    label: str
    sense: str
    detuning_ghz: tuple[float, ...]
    source_power_w: tuple[float, ...]


@dataclass(frozen=True)
class LinearCell:
    """一个失谐分段内的线性规划可行多边形。"""

    detuning_min_ghz: float
    detuning_max_ghz: float
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class LinearDesignPoint:
    """LP 推荐点代回完整非线性模型后的指标。"""

    detuning_ghz: float
    source_power_w: float
    wavelength_nm: float
    depth_uK: float
    scattering_rate_s: float
    effective_barrier_uK: float
    bound_fraction: float
    handover_axial_cycles: float
    exact_constraints_satisfied: bool
    active_constraints: tuple[str, ...]


@dataclass(frozen=True)
class HandoverLPResult:
    """一个 handover 时间对应的分段 LP 结果。"""

    handover_time_ms: float
    boundaries: tuple[ConstraintBoundary, ...]
    feasible_cells: tuple[LinearCell, ...]
    recommended: LinearDesignPoint | None

    @property
    def feasible(self) -> bool:
        return bool(self.feasible_cells)


@dataclass(frozen=True)
class LinearDesignResult:
    """若干默认 handover 时间的联合结果。"""

    inputs: LinearDesignInputs
    handover_results: tuple[HandoverLPResult, ...]


@dataclass(frozen=True)
class _AffineConstraint:
    """半平面 ``a*x + b*y <= c``。"""

    a: float
    b: float
    c: float


def _atom_from_label(label: str) -> AlkaliAtom:
    normalized = label.strip().lower().replace("_", "").replace("-", "")
    if normalized in {"cs", "cs133", "133cs"}:
        return CS133
    if normalized in {"rb", "rb87", "87rb"}:
        return RB87
    raise ValueError("原子必须是 Cs-133 或 Rb-87")


def _linspace(start: float, stop: float, count: int) -> tuple[float, ...]:
    step = (stop - start) / (count - 1)
    return tuple(start + index * step for index in range(count))


def _source_power_for_depth(
    atom: AlkaliAtom,
    detuning_ghz: float,
    depth_uK: float,
    inputs: LinearDesignInputs,
) -> float:
    wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning_ghz)
    forward_power = power_for_target_depth(
        atom,
        wavelength_nm,
        depth_uK,
        inputs.waist_um,
        inputs.retro_power_ratio,
    )
    return forward_power / inputs.delivery_efficiency


def handover_axial_depth_requirement_uK(
    atom: AlkaliAtom,
    detuning_ghz: float,
    handover_time_ms: float,
    minimum_cycles: float,
) -> float:
    """返回 handover 内完成指定轴向振荡周数所需的最小势深。"""
    if handover_time_ms <= 0.0 or minimum_cycles <= 0.0:
        raise ValueError("handover 时间和振荡周数必须为正")
    wavelength_m = atom.laser_wavelength_red_of_d1_nm(detuning_ghz) * 1e-9
    wave_number = 2.0 * math.pi / wavelength_m
    required_frequency_hz = minimum_cycles / (handover_time_ms * 1e-3)
    required_omega = 2.0 * math.pi * required_frequency_hz
    depth_j = atom.mass_kg * required_omega**2 / (2.0 * wave_number**2)
    return depth_j / BOLTZMANN * 1e6


def _barrier_ratio_for_bound_fraction(target_fraction: float) -> float:
    low = 0.0
    high = 8.0
    while thermal_bound_fraction_3d_harmonic(high, 1.0) < target_fraction:
        high *= 2.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if thermal_bound_fraction_3d_harmonic(middle, 1.0) < target_fraction:
            low = middle
        else:
            high = middle
    return high


def acceleration_bound_depth_requirement_uK(
    atom: AlkaliAtom,
    detuning_ghz: float,
    design_temperature_uK: float,
    target_bound_fraction: float,
    acceleration_m_s2: float,
) -> float:
    """求使加速后有效势垒达到目标束缚比例的最小静态深度。"""
    wavelength_m = atom.laser_wavelength_red_of_d1_nm(detuning_ghz) * 1e-9
    wave_number = 2.0 * math.pi / wavelength_m
    target_barrier_uK = (
        _barrier_ratio_for_bound_fraction(target_bound_fraction)
        * design_temperature_uK
    )

    def effective_barrier(depth_uK: float) -> float:
        depth_j = depth_uK * 1e-6 * BOLTZMANN
        critical = depth_j * wave_number / atom.mass_kg
        fraction = tilted_lattice_barrier_fraction(
            acceleration_m_s2,
            critical,
        )
        return depth_uK * fraction

    low = max(target_barrier_uK, 1e-9)
    high = low
    while effective_barrier(high) < target_barrier_uK:
        high *= 2.0
        if high > 1e9:
            raise ValueError("当前加速度下无法找到有限的束缚深度")
    for _ in range(80):
        middle = 0.5 * (low + high)
        if effective_barrier(middle) < target_barrier_uK:
            low = middle
        else:
            high = middle
    return high


def _boundary_functions(
    inputs: LinearDesignInputs,
    handover_time_ms: float,
) -> tuple[tuple[str, str, Callable[[float], float]], ...]:
    atom = _atom_from_label(inputs.atom_label)

    def target_depth(detuning: float) -> float:
        return _source_power_for_depth(
            atom,
            detuning,
            inputs.target_depth_uK,
            inputs,
        )

    def handover_cycles(detuning: float) -> float:
        depth = handover_axial_depth_requirement_uK(
            atom,
            detuning,
            handover_time_ms,
            inputs.handover_min_axial_cycles,
        )
        return _source_power_for_depth(atom, detuning, depth, inputs)

    def bound_fraction(detuning: float) -> float:
        depth = acceleration_bound_depth_requirement_uK(
            atom,
            detuning,
            inputs.design_temperature_uK,
            inputs.target_bound_fraction,
            inputs.acceleration_m_s2,
        )
        return _source_power_for_depth(atom, detuning, depth, inputs)

    def scattering(detuning: float) -> float:
        wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning)
        per_source_watt = evaluate_lattice(
            atom,
            wavelength_nm,
            forward_power_w=inputs.delivery_efficiency,
            waist_um=inputs.waist_um,
            retro_power_ratio=inputs.retro_power_ratio,
        )
        return inputs.max_scattering_rate_s / per_source_watt.scattering_rate_s

    def source_power(_: float) -> float:
        return inputs.max_source_power_w

    return (
        ("目标阱深下限", "lower", target_depth),
        (
            f"handover ≥ {inputs.handover_min_axial_cycles:g} 轴向周期",
            "lower",
            handover_cycles,
        ),
        (
            f"加速后束缚比例 ≥ {inputs.target_bound_fraction:.2f}",
            "lower",
            bound_fraction,
        ),
        ("最大散射率", "upper", scattering),
        ("最大源端功率", "upper", source_power),
    )


def _conservative_affine_on_interval(
    function: Callable[[float], float],
    start: float,
    stop: float,
    sense: str,
) -> tuple[float, float]:
    """返回 ``power = slope*detuning + intercept`` 的保守局部边界。"""
    first = function(start)
    last = function(stop)
    slope = (last - first) / (stop - start)
    intercept = first - slope * start
    samples = _linspace(start, stop, 65)
    residuals = [
        function(value) - (slope * value + intercept)
        for value in samples
    ]
    if sense == "lower":
        # 约束 power >= line；上移直线，避免低估真实下边界。
        intercept += max(residuals)
    elif sense == "upper":
        # 约束 power <= line；下移直线，避免高估真实上边界。
        intercept += min(residuals)
    else:
        raise ValueError("约束方向必须是 lower 或 upper")
    return slope, intercept


def _clip_polygon(
    polygon: list[tuple[float, float]],
    constraint: _AffineConstraint,
    *,
    tolerance: float = 1e-10,
) -> list[tuple[float, float]]:
    if not polygon:
        return []

    def value(point: tuple[float, float]) -> float:
        return (
            constraint.a * point[0]
            + constraint.b * point[1]
            - constraint.c
        )

    result: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_value = value(previous)
    previous_inside = previous_value <= tolerance
    for current in polygon:
        current_value = value(current)
        current_inside = current_value <= tolerance
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > 1e-30:
                fraction = previous_value / denominator
                result.append(
                    (
                        previous[0] + fraction * (current[0] - previous[0]),
                        previous[1] + fraction * (current[1] - previous[1]),
                    )
                )
        if current_inside:
            result.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return result


def _solve_cell(
    inputs: LinearDesignInputs,
    start: float,
    stop: float,
    boundary_functions: Iterable[
        tuple[str, str, Callable[[float], float]]
    ],
) -> LinearCell | None:
    polygon = [
        (start, 0.0),
        (stop, 0.0),
        (stop, inputs.max_source_power_w),
        (start, inputs.max_source_power_w),
    ]
    constraints = [
        _AffineConstraint(-1.0, 0.0, -start),
        _AffineConstraint(1.0, 0.0, stop),
        _AffineConstraint(0.0, -1.0, 0.0),
        _AffineConstraint(0.0, 1.0, inputs.max_source_power_w),
    ]
    for _, sense, function in boundary_functions:
        slope, intercept = _conservative_affine_on_interval(
            function,
            start,
            stop,
            sense,
        )
        if sense == "lower":
            constraints.append(
                _AffineConstraint(slope, -1.0, -intercept)
            )
        else:
            constraints.append(
                _AffineConstraint(-slope, 1.0, intercept)
            )
    for constraint in constraints:
        polygon = _clip_polygon(polygon, constraint)
        if len(polygon) < 3:
            return None
    return LinearCell(start, stop, tuple(polygon))


def _evaluate_point(
    inputs: LinearDesignInputs,
    handover_time_ms: float,
    detuning_ghz: float,
    source_power_w: float,
    functions: tuple[tuple[str, str, Callable[[float], float]], ...],
) -> LinearDesignPoint:
    atom = _atom_from_label(inputs.atom_label)
    wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning_ghz)
    lattice = evaluate_lattice(
        atom,
        wavelength_nm,
        forward_power_w=source_power_w * inputs.delivery_efficiency,
        waist_um=inputs.waist_um,
        retro_power_ratio=inputs.retro_power_ratio,
    )
    barrier_fraction = tilted_lattice_barrier_fraction(
        inputs.acceleration_m_s2,
        lattice.critical_axial_acceleration_m_s2,
    )
    effective_barrier = lattice.depth_uK * barrier_fraction
    bound_fraction = thermal_bound_fraction_3d_harmonic(
        effective_barrier,
        inputs.design_temperature_uK,
    )
    cycles = lattice.axial_frequency_hz * handover_time_ms * 1e-3
    exact_ok = (
        lattice.depth_uK >= inputs.target_depth_uK * (1.0 - 1e-9)
        and lattice.scattering_rate_s
        <= inputs.max_scattering_rate_s * (1.0 + 1e-9)
        and source_power_w <= inputs.max_source_power_w * (1.0 + 1e-9)
        and bound_fraction >= inputs.target_bound_fraction - 1e-9
        and cycles >= inputs.handover_min_axial_cycles * (1.0 - 1e-9)
    )
    active: list[str] = []
    for label, sense, function in functions:
        boundary = function(detuning_ghz)
        scale = max(abs(boundary), 1e-12)
        relative_gap = (
            source_power_w - boundary
            if sense == "lower"
            else boundary - source_power_w
        ) / scale
        if relative_gap <= 0.02:
            active.append(label)
    return LinearDesignPoint(
        detuning_ghz=detuning_ghz,
        source_power_w=source_power_w,
        wavelength_nm=wavelength_nm,
        depth_uK=lattice.depth_uK,
        scattering_rate_s=lattice.scattering_rate_s,
        effective_barrier_uK=effective_barrier,
        bound_fraction=bound_fraction,
        handover_axial_cycles=cycles,
        exact_constraints_satisfied=exact_ok,
        active_constraints=tuple(active),
    )


def analyze_detuning_power_lp(
    inputs: LinearDesignInputs = LinearDesignInputs(),
    *,
    handover_times_ms: Iterable[float] = DEFAULT_HANDOVER_TIMES_MS,
) -> LinearDesignResult:
    """构造并求解各默认 handover 时间下的分段二维 LP。"""
    times = tuple(float(value) for value in handover_times_ms)
    if not times or any(not math.isfinite(value) or value <= 0.0 for value in times):
        raise ValueError("handover 时间必须是非空的有限正数序列")
    nodes = _linspace(
        inputs.detuning_min_ghz,
        inputs.detuning_max_ghz,
        inputs.segment_count + 1,
    )
    plot_detuning = _linspace(
        inputs.detuning_min_ghz,
        inputs.detuning_max_ghz,
        241,
    )
    results: list[HandoverLPResult] = []
    detuning_span = inputs.detuning_max_ghz - inputs.detuning_min_ghz

    for handover_time in times:
        functions = _boundary_functions(inputs, handover_time)
        boundaries = tuple(
            ConstraintBoundary(
                label=label,
                sense=sense,
                detuning_ghz=plot_detuning,
                source_power_w=tuple(function(value) for value in plot_detuning),
            )
            for label, sense, function in functions
        )
        cells: list[LinearCell] = []
        for start, stop in zip(nodes[:-1], nodes[1:]):
            cell = _solve_cell(inputs, start, stop, functions)
            if cell is not None:
                cells.append(cell)

        recommended: LinearDesignPoint | None = None
        best_score = math.inf
        for cell in cells:
            for detuning, power in cell.polygon:
                score = (
                    power / inputs.max_source_power_w
                    + inputs.detuning_objective_weight
                    * (detuning - inputs.detuning_min_ghz)
                    / detuning_span
                )
                if score < best_score:
                    point = _evaluate_point(
                        inputs,
                        handover_time,
                        detuning,
                        power,
                        functions,
                    )
                    if point.exact_constraints_satisfied:
                        best_score = score
                        recommended = point
        results.append(
            HandoverLPResult(
                handover_time_ms=handover_time,
                boundaries=boundaries,
                feasible_cells=tuple(cells),
                recommended=recommended,
            )
        )
    return LinearDesignResult(inputs=inputs, handover_results=tuple(results))


def plot_detuning_power_lp(
    result: LinearDesignResult,
    output_path: str | Path,
) -> Path:
    """绘制各 handover 时间的约束边界、LP 可行域和推荐点。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(result.handover_results)
    columns = 2 if count > 1 else 1
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.2 * columns, 5.2 * rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    colors = ("#2563eb", "#7c3aed", "#d97706", "#dc2626", "#374151")
    plot_labels = (
        "Minimum trap depth",
        "Handover axial-cycle limit",
        "Accelerated bound-fraction limit",
        "Maximum scattering",
        "Maximum source power",
    )

    for index, time_result in enumerate(result.handover_results):
        axis = axes[index // columns][index % columns]
        for cell_index, cell in enumerate(time_result.feasible_cells):
            xs = [point[0] for point in cell.polygon]
            ys = [point[1] for point in cell.polygon]
            axis.fill(
                xs,
                ys,
                color="#22c55e",
                alpha=0.22,
                linewidth=0.0,
                label="LP feasible region" if cell_index == 0 else None,
            )
        for boundary, color, plot_label in zip(
            time_result.boundaries,
            colors,
            plot_labels,
        ):
            axis.plot(
                boundary.detuning_ghz,
                boundary.source_power_w,
                color=color,
                linewidth=1.8,
                label=plot_label,
            )
        if time_result.recommended is not None:
            point = time_result.recommended
            axis.scatter(
                [point.detuning_ghz],
                [point.source_power_w],
                marker="*",
                s=170,
                color="#111827",
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
                label="LP recommendation",
            )
        axis.set_title(
            f"{result.inputs.atom_label}: handover = "
            f"{time_result.handover_time_ms:g} ms"
        )
        axis.grid(alpha=0.22)
        axis.set_ylim(0.0, result.inputs.max_source_power_w * 1.08)
        axis.set_xlabel("D1 red detuning (GHz)")
        axis.set_ylabel("Source power per lattice branch (W)")
        axis.legend(fontsize=8, loc="best")

    for index in range(count, rows * columns):
        axes[index // columns][index % columns].set_visible(False)
    figure.suptitle(
        "Piecewise-linear detuning-power constraints and LP feasible regions",
        fontsize=14,
    )
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
