"""面向实验搭建的失谐--功率--束腰稳健设计优化。

时序量保持论文/现有程序默认值，只扫描会直接影响激光器采购和光路
设计的三个量。优化目标不是单点极值，而是寻找在三个变量各自发生
给定相对扰动时仍满足原始非线性限制的工作平台。最后只对少量稳健
候选运行现有三维 handover Monte Carlo，并对推荐点扫描相对相位。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np

from .atomic import CS133, RB87, AlkaliAtom
from .handover import HandoverParameters, run_handover_monte_carlo
from .linear_design import (
    LinearDesignInputs,
    LinearDesignPoint,
    _boundary_functions,
    _evaluate_point,
)


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "design_optimization_defaults.json"
)


def load_design_optimization_configuration(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, object]:
    """读取稳健设计优化的集中参数。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "atom",
        "scan",
        "fixed_handover",
        "constraints",
        "robustness",
        "species",
        "monte_carlo",
        "phase_scan",
        "plot",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError("设计优化配置缺少分组：" + ", ".join(sorted(missing)))
    return payload


DESIGN_OPTIMIZATION_CONFIGURATION = load_design_optimization_configuration()


def _section(name: str) -> dict[str, object]:
    value = DESIGN_OPTIMIZATION_CONFIGURATION[name]
    if not isinstance(value, dict):
        raise ValueError(f"设计优化配置 {name} 必须是对象")
    return value


_SCAN = _section("scan")
_FIXED = _section("fixed_handover")
_CONSTRAINTS = _section("constraints")
_ROBUSTNESS = _section("robustness")
_SPECIES = _section("species")
_MC = _section("monte_carlo")
_PHASE = _section("phase_scan")
_PLOT = _section("plot")


@dataclass(frozen=True)
class RobustDesignInputs:
    """三变量稳健优化输入；时序量只作为固定参数记录。"""

    atom_label: str = str(DESIGN_OPTIMIZATION_CONFIGURATION["atom"])
    detuning_min_ghz: float = float(_SCAN["detuning_min_ghz"])
    detuning_max_ghz: float = float(_SCAN["detuning_max_ghz"])
    detuning_points: int = int(_SCAN["detuning_points"])
    source_power_min_w: float = float(_SCAN["source_power_min_w"])
    source_power_max_w: float = float(_SCAN["source_power_max_w"])
    power_points: int = int(_SCAN["power_points"])
    waist_min_um: float = float(_SCAN["waist_min_um"])
    waist_max_um: float = float(_SCAN["waist_max_um"])
    waist_points: int = int(_SCAN["waist_points"])
    relative_tolerance: float = float(_ROBUSTNESS["relative_tolerance"])
    variation_mode: str = str(_ROBUSTNESS["variation_mode"])
    monte_carlo_candidate_count: int = int(
        _ROBUSTNESS["monte_carlo_candidate_count"]
    )
    minimum_transfer_efficiency: float = float(
        _ROBUSTNESS["minimum_transfer_efficiency"]
    )
    confidence_sigma: float = float(_ROBUSTNESS["confidence_sigma"])
    particle_count: int = int(_MC["particle_count"])
    time_step_us: float = float(_MC["time_step_us"])
    include_scattering: bool = bool(_MC["include_scattering"])
    seed: int = int(_MC["seed"])
    parallel_backend: str = str(_MC["parallel_backend"])
    worker_count: int = int(_MC["worker_count"])
    phase_points: int = int(_PHASE["phase_points"])

    def __post_init__(self) -> None:
        _atom_from_label(self.atom_label)
        ranges = (
            ("失谐", self.detuning_min_ghz, self.detuning_max_ghz),
            ("功率", self.source_power_min_w, self.source_power_max_w),
            ("束腰", self.waist_min_um, self.waist_max_um),
        )
        for name, lower, upper in ranges:
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError(f"{name}范围必须有限")
            if lower <= 0.0 or upper <= lower:
                raise ValueError(f"{name}范围必须满足 0 < min < max")
        if min(self.detuning_points, self.power_points, self.waist_points) < 2:
            raise ValueError("三个扫描维度都至少需要 2 个网格点")
        if not 0.0 <= self.relative_tolerance < 1.0:
            raise ValueError("相对容差必须位于 [0, 1)")
        if self.variation_mode not in {"one_at_a_time", "box_corners"}:
            raise ValueError("容差模式必须是 one_at_a_time 或 box_corners")
        if self.monte_carlo_candidate_count <= 0:
            raise ValueError("Monte Carlo 候选数必须为正")
        if not 0.0 <= self.minimum_transfer_efficiency <= 1.0:
            raise ValueError("最低交接率必须位于 [0, 1]")
        if self.confidence_sigma < 0.0:
            raise ValueError("置信系数不能为负")
        if self.particle_count <= 0 or self.time_step_us <= 0.0:
            raise ValueError("粒子数和时间步长必须为正")
        if self.parallel_backend not in {"serial", "process"}:
            raise ValueError("并行后端必须是 serial 或 process")
        if self.worker_count <= 0 or self.phase_points < 2:
            raise ValueError("进程数必须为正，相位点数至少为 2")


@dataclass(frozen=True)
class RobustCandidate:
    """经过配置所选容差扰动检查的候选工作点。"""

    detuning_ghz: float
    source_power_w: float
    waist_um: float
    wavelength_nm: float
    depth_uK: float
    scattering_rate_s: float
    bound_fraction: float
    handover_axial_cycles: float
    worst_constraint_margin: float
    worst_constraint: str
    scan_window_margin: float
    plateau_score: float
    nominal_constraints_satisfied: bool
    robust_constraints_satisfied: bool


@dataclass(frozen=True)
class MonteCarloCandidate:
    """稳健候选的轨迹验证结果。"""

    design: RobustCandidate
    transfer_efficiency: float
    transfer_standard_error: float
    conservative_efficiency: float
    final_temperature_uK: float | None
    handover_heating_uK: float | None


@dataclass(frozen=True)
class PhaseScanPoint:
    """固定相位下的交接率和预计保留原子数。"""

    phase_rad: float
    transfer_efficiency: float
    transfer_standard_error: float
    expected_atom_number: float


@dataclass(frozen=True)
class RobustDesignResult:
    """三维网格、候选、推荐点和相位扫描的联合结果。"""

    inputs: RobustDesignInputs
    fixed_parameters: dict[str, float | bool]
    detuning_ghz: tuple[float, ...]
    source_power_w: tuple[float, ...]
    waist_um: tuple[float, ...]
    nominal_feasible: tuple[tuple[tuple[bool, ...], ...], ...]
    robust_feasible: tuple[tuple[tuple[bool, ...], ...], ...]
    worst_constraint_margin: tuple[tuple[tuple[float, ...], ...], ...]
    robust_point_count: int
    monte_carlo_candidates: tuple[MonteCarloCandidate, ...]
    recommended: MonteCarloCandidate | None
    phase_scan: tuple[PhaseScanPoint, ...]


def _atom_from_label(label: str) -> AlkaliAtom:
    normalized = label.strip().lower().replace("-", "").replace("_", "")
    if normalized in {"cs", "cs133", "133cs"}:
        return CS133
    if normalized in {"rb", "rb87", "87rb"}:
        return RB87
    raise ValueError("原子必须是 Cs-133 或 Rb-87")


def _species_values(atom_label: str) -> tuple[float, float]:
    canonical = "Cs-133" if _atom_from_label(atom_label) is CS133 else "Rb-87"
    payload = _SPECIES[canonical]
    return (
        float(payload["initial_temperature_uK"]),
        float(payload["initial_atom_number"]),
    )


def _design_inputs(inputs: RobustDesignInputs, waist_um: float) -> LinearDesignInputs:
    temperature_uK, _ = _species_values(inputs.atom_label)
    return LinearDesignInputs(
        atom_label=inputs.atom_label,
        detuning_min_ghz=inputs.detuning_min_ghz * (1.0 - inputs.relative_tolerance),
        detuning_max_ghz=inputs.detuning_max_ghz * (1.0 + inputs.relative_tolerance),
        waist_um=waist_um,
        target_depth_uK=float(_CONSTRAINTS["target_depth_uK"]),
        design_temperature_uK=temperature_uK,
        target_bound_fraction=float(_CONSTRAINTS["target_bound_fraction"]),
        acceleration_m_s2=float(
            _FIXED["post_handover_acceleration_m_s2"]
        ),
        handover_min_axial_cycles=float(
            _CONSTRAINTS["minimum_axial_cycles"]
        ),
        max_source_power_w=inputs.source_power_max_w,
        max_scattering_rate_s=float(
            _CONSTRAINTS["max_scattering_rate_s"]
        ),
        delivery_efficiency=float(_CONSTRAINTS["delivery_efficiency"]),
        retro_power_ratio=float(_CONSTRAINTS["retro_power_ratio"]),
    )


def _constraint_margins(
    point: LinearDesignPoint,
    design: LinearDesignInputs,
) -> dict[str, float]:
    """返回正值为满足、负值为违反的无量纲裕量。"""
    return {
        "minimum trap depth": point.depth_uK / design.target_depth_uK - 1.0,
        "maximum scattering": (
            design.max_scattering_rate_s / max(point.scattering_rate_s, 1e-30)
            - 1.0
        ),
        "accelerated bound fraction": (
            (point.bound_fraction - design.target_bound_fraction)
            / (1.0 - design.target_bound_fraction)
        ),
        "handover axial cycles": (
            point.handover_axial_cycles / design.handover_min_axial_cycles - 1.0
        ),
        "maximum source power": (
            design.max_source_power_w / max(point.source_power_w, 1e-30) - 1.0
        ),
    }


def _evaluate_robust_candidate(
    inputs: RobustDesignInputs,
    detuning_ghz: float,
    source_power_w: float,
    waist_um: float,
) -> RobustCandidate:
    handover_time_ms = float(_FIXED["time_us"]) * 1e-3
    nominal_design = _design_inputs(inputs, waist_um)
    nominal_functions = _boundary_functions(nominal_design, handover_time_ms)
    nominal = _evaluate_point(
        nominal_design,
        handover_time_ms,
        detuning_ghz,
        source_power_w,
        nominal_functions,
    )
    tolerance = inputs.relative_tolerance
    if tolerance == 0.0:
        variations = ((1.0, 1.0, 1.0),)
    elif inputs.variation_mode == "one_at_a_time":
        variations = (
            (1.0, 1.0, 1.0),
            (1.0 - tolerance, 1.0, 1.0),
            (1.0 + tolerance, 1.0, 1.0),
            (1.0, 1.0 - tolerance, 1.0),
            (1.0, 1.0 + tolerance, 1.0),
            (1.0, 1.0, 1.0 - tolerance),
            (1.0, 1.0, 1.0 + tolerance),
        )
    else:
        factors = (1.0 - tolerance, 1.0, 1.0 + tolerance)
        variations = tuple(
            (detuning_factor, power_factor, waist_factor)
            for detuning_factor in factors
            for power_factor in factors
            for waist_factor in factors
        )
    worst_margin = math.inf
    worst_constraint = ""
    for detuning_factor, power_factor, waist_factor in variations:
        design = _design_inputs(inputs, waist_um * waist_factor)
        functions = _boundary_functions(design, handover_time_ms)
        point = _evaluate_point(
            design,
            handover_time_ms,
            detuning_ghz * detuning_factor,
            source_power_w * power_factor,
            functions,
        )
        for label, margin in _constraint_margins(point, design).items():
            if margin < worst_margin:
                worst_margin = margin
                worst_constraint = label
    tolerance = inputs.relative_tolerance
    scan_window_margins = (
        (detuning_ghz * (1.0 - tolerance) - inputs.detuning_min_ghz)
        / (inputs.detuning_max_ghz - inputs.detuning_min_ghz),
        (inputs.detuning_max_ghz - detuning_ghz * (1.0 + tolerance))
        / (inputs.detuning_max_ghz - inputs.detuning_min_ghz),
        (source_power_w * (1.0 - tolerance) - inputs.source_power_min_w)
        / (inputs.source_power_max_w - inputs.source_power_min_w),
        (inputs.source_power_max_w - source_power_w * (1.0 + tolerance))
        / (inputs.source_power_max_w - inputs.source_power_min_w),
        (waist_um * (1.0 - tolerance) - inputs.waist_min_um)
        / (inputs.waist_max_um - inputs.waist_min_um),
        (inputs.waist_max_um - waist_um * (1.0 + tolerance))
        / (inputs.waist_max_um - inputs.waist_min_um),
    )
    scan_window_margin = min(scan_window_margins)
    plateau_score = min(worst_margin, scan_window_margin)
    return RobustCandidate(
        detuning_ghz=detuning_ghz,
        source_power_w=source_power_w,
        waist_um=waist_um,
        wavelength_nm=nominal.wavelength_nm,
        depth_uK=nominal.depth_uK,
        scattering_rate_s=nominal.scattering_rate_s,
        bound_fraction=nominal.bound_fraction,
        handover_axial_cycles=nominal.handover_axial_cycles,
        worst_constraint_margin=worst_margin,
        worst_constraint=worst_constraint,
        scan_window_margin=scan_window_margin,
        plateau_score=plateau_score,
        nominal_constraints_satisfied=nominal.exact_constraints_satisfied,
        robust_constraints_satisfied=(
            worst_margin >= -1e-12 and scan_window_margin >= -1e-12
        ),
    )


def _handover_parameters(
    inputs: RobustDesignInputs,
    candidate: RobustCandidate,
    *,
    phase_rad: float = 0.0,
    randomize_relative_phase: bool = True,
) -> HandoverParameters:
    atom = _atom_from_label(inputs.atom_label)
    temperature_uK, atom_number = _species_values(inputs.atom_label)
    design = _design_inputs(inputs, candidate.waist_um)
    return HandoverParameters(
        atom_mass_kg=atom.mass_kg,
        wavelength_nm=candidate.wavelength_nm,
        depth1_uK=candidate.depth_uK,
        depth2_uK=candidate.depth_uK,
        waist1_um=candidate.waist_um,
        waist2_um=candidate.waist_um,
        scattering_rate1_s=candidate.scattering_rate_s,
        scattering_rate2_s=candidate.scattering_rate_s,
        retro_power_ratio=design.retro_power_ratio,
        initial_atom_number=atom_number,
        temperature_uK=temperature_uK,
        duration_ms=float(_FIXED["time_us"]) * 1e-3,
        crossing_angle_deg=float(_FIXED["crossing_angle_deg"]),
        cloud_axial_sigma_mm=float(_FIXED["cloud_axial_sigma_mm"]),
        l2_transverse_offset_um=float(_FIXED["l2_transverse_offset_um"]),
        relative_phase_rad=phase_rad,
        randomize_relative_phase=randomize_relative_phase,
        lattice1_velocity_m_s=float(_FIXED["lattice1_velocity_m_s"]),
        lattice2_velocity_m_s=float(_FIXED["lattice2_velocity_m_s"]),
        post_handover_acceleration_m_s2=float(
            _FIXED["post_handover_acceleration_m_s2"]
        ),
        include_scattering=inputs.include_scattering,
        particle_count=inputs.particle_count,
        time_step_us=inputs.time_step_us,
        trace_points=2,
        seed=inputs.seed,
    )


def _run_mc_task(
    task: tuple[str, HandoverParameters],
) -> tuple[str, float, float, float | None, float | None]:
    key, parameters = task
    result = run_handover_monte_carlo(parameters)
    return (
        key,
        result.transfer_efficiency,
        result.transfer_standard_error,
        result.final_temperature_uK,
        result.handover_heating_uK,
    )


def _execute_tasks(
    tasks: list[tuple[str, HandoverParameters]],
    inputs: RobustDesignInputs,
    progress: Callable[[str], None] | None,
) -> dict[str, tuple[float, float, float | None, float | None]]:
    results: dict[str, tuple[float, float, float | None, float | None]] = {}
    use_processes = (
        inputs.parallel_backend == "process"
        and inputs.worker_count > 1
        and len(tasks) > 1
    )
    workers = min(inputs.worker_count, len(tasks), os.cpu_count() or 1)
    if progress is not None:
        mode = f"{workers} 个 CPU 进程" if use_processes else "串行"
        progress(f"Monte Carlo: {len(tasks)} 个任务（{mode}）")

    def store(item: tuple[str, float, float, float | None, float | None]) -> None:
        key, efficiency, error, temperature, heating = item
        results[key] = (efficiency, error, temperature, heating)

    if use_processes:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_mc_task, task) for task in tasks]
            for future in as_completed(futures):
                store(future.result())
    else:
        for task in tasks:
            store(_run_mc_task(task))
    return results


def analyze_robust_design(
    inputs: RobustDesignInputs = RobustDesignInputs(),
    *,
    progress: Callable[[str], None] | None = None,
) -> RobustDesignResult:
    """扫描三项采购变量，验证稳健候选并生成相位调节曲线。"""
    detunings = np.linspace(
        inputs.detuning_min_ghz,
        inputs.detuning_max_ghz,
        inputs.detuning_points,
    )
    powers = np.linspace(
        inputs.source_power_min_w,
        inputs.source_power_max_w,
        inputs.power_points,
    )
    waists = np.linspace(inputs.waist_min_um, inputs.waist_max_um, inputs.waist_points)
    candidates: list[RobustCandidate] = []
    nominal: list[list[list[bool]]] = []
    robust: list[list[list[bool]]] = []
    margins: list[list[list[float]]] = []
    for waist_index, waist in enumerate(waists):
        nominal_plane: list[list[bool]] = []
        robust_plane: list[list[bool]] = []
        margin_plane: list[list[float]] = []
        for power in powers:
            nominal_row: list[bool] = []
            robust_row: list[bool] = []
            margin_row: list[float] = []
            for detuning in detunings:
                candidate = _evaluate_robust_candidate(
                    inputs,
                    float(detuning),
                    float(power),
                    float(waist),
                )
                candidates.append(candidate)
                nominal_row.append(candidate.nominal_constraints_satisfied)
                robust_row.append(candidate.robust_constraints_satisfied)
                margin_row.append(candidate.worst_constraint_margin)
            nominal_plane.append(nominal_row)
            robust_plane.append(robust_row)
            margin_plane.append(margin_row)
        nominal.append(nominal_plane)
        robust.append(robust_plane)
        margins.append(margin_plane)
        if progress is not None:
            progress(f"稳健约束扫描: {waist_index + 1}/{len(waists)} 个束腰")

    robust_candidates = sorted(
        (item for item in candidates if item.robust_constraints_satisfied),
        key=lambda item: (item.plateau_score, item.worst_constraint_margin),
        reverse=True,
    )
    shortlisted = robust_candidates[: inputs.monte_carlo_candidate_count]
    candidate_tasks = [
        (
            f"candidate:{index}",
            _handover_parameters(inputs, candidate),
        )
        for index, candidate in enumerate(shortlisted)
    ]
    candidate_results = _execute_tasks(candidate_tasks, inputs, progress)
    monte_carlo: list[MonteCarloCandidate] = []
    for index, candidate in enumerate(shortlisted):
        efficiency, error, temperature, heating = candidate_results[
            f"candidate:{index}"
        ]
        monte_carlo.append(
            MonteCarloCandidate(
                design=candidate,
                transfer_efficiency=efficiency,
                transfer_standard_error=error,
                conservative_efficiency=max(
                    0.0,
                    efficiency - inputs.confidence_sigma * error,
                ),
                final_temperature_uK=temperature,
                handover_heating_uK=heating,
            )
        )
    eligible = [
        item
        for item in monte_carlo
        if item.conservative_efficiency >= inputs.minimum_transfer_efficiency
    ]
    recommended = (
        max(
            eligible,
            key=lambda item: (
                item.design.plateau_score,
                item.design.worst_constraint_margin,
                item.conservative_efficiency,
            ),
        )
        if eligible
        else None
    )

    phase_scan: list[PhaseScanPoint] = []
    if recommended is not None:
        phases = np.linspace(
            float(_PHASE["phase_min_rad"]),
            float(_PHASE["phase_max_rad"]),
            inputs.phase_points,
        )
        phase_tasks = [
            (
                f"phase:{index}",
                _handover_parameters(
                    inputs,
                    recommended.design,
                    phase_rad=float(phase),
                    randomize_relative_phase=False,
                ),
            )
            for index, phase in enumerate(phases)
        ]
        phase_results = _execute_tasks(phase_tasks, inputs, progress)
        _, atom_number = _species_values(inputs.atom_label)
        for index, phase in enumerate(phases):
            efficiency, error, _, _ = phase_results[f"phase:{index}"]
            phase_scan.append(
                PhaseScanPoint(
                    phase_rad=float(phase),
                    transfer_efficiency=efficiency,
                    transfer_standard_error=error,
                    expected_atom_number=atom_number * efficiency,
                )
            )

    fixed_parameters: dict[str, float | bool] = {
        key: float(value) if isinstance(value, (int, float)) else bool(value)
        for key, value in _FIXED.items()
    }
    return RobustDesignResult(
        inputs=inputs,
        fixed_parameters=fixed_parameters,
        detuning_ghz=tuple(float(value) for value in detunings),
        source_power_w=tuple(float(value) for value in powers),
        waist_um=tuple(float(value) for value in waists),
        nominal_feasible=tuple(
            tuple(tuple(row) for row in plane) for plane in nominal
        ),
        robust_feasible=tuple(
            tuple(tuple(row) for row in plane) for plane in robust
        ),
        worst_constraint_margin=tuple(
            tuple(tuple(row) for row in plane) for plane in margins
        ),
        robust_point_count=len(robust_candidates),
        monte_carlo_candidates=tuple(monte_carlo),
        recommended=recommended,
        phase_scan=tuple(phase_scan),
    )


def plot_robust_design(
    result: RobustDesignResult,
    output_path: str | Path,
) -> Path:
    """绘制三维稳健优化的二维投影和实验相位扫描。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    margins = np.array(result.worst_constraint_margin, dtype=float)
    robust = np.array(result.robust_feasible, dtype=bool)
    tolerance = result.inputs.relative_tolerance
    detunings = np.asarray(result.detuning_ghz)[None, None, :]
    powers = np.asarray(result.source_power_w)[None, :, None]
    waists = np.asarray(result.waist_um)[:, None, None]
    detuning_span = (
        result.inputs.detuning_max_ghz - result.inputs.detuning_min_ghz
    )
    power_span = (
        result.inputs.source_power_max_w - result.inputs.source_power_min_w
    )
    waist_span = result.inputs.waist_max_um - result.inputs.waist_min_um
    scan_margin = np.minimum.reduce(
        (
            np.broadcast_to(
                (
                    detunings * (1.0 - tolerance)
                    - result.inputs.detuning_min_ghz
                )
                / detuning_span,
                margins.shape,
            ),
            np.broadcast_to(
                (
                    result.inputs.detuning_max_ghz
                    - detunings * (1.0 + tolerance)
                )
                / detuning_span,
                margins.shape,
            ),
            np.broadcast_to(
                (
                    powers * (1.0 - tolerance)
                    - result.inputs.source_power_min_w
                )
                / power_span,
                margins.shape,
            ),
            np.broadcast_to(
                (
                    result.inputs.source_power_max_w
                    - powers * (1.0 + tolerance)
                )
                / power_span,
                margins.shape,
            ),
            np.broadcast_to(
                (
                    waists * (1.0 - tolerance)
                    - result.inputs.waist_min_um
                )
                / waist_span,
                margins.shape,
            ),
            np.broadcast_to(
                (
                    result.inputs.waist_max_um
                    - waists * (1.0 + tolerance)
                )
                / waist_span,
                margins.shape,
            ),
        )
    )
    plateau_scores = np.minimum(margins, scan_margin)
    best_margin_dp = np.max(plateau_scores, axis=0)
    best_margin_dw = np.max(plateau_scores, axis=1)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(float(_PLOT["figure_width_in"]), float(_PLOT["figure_height_in"])),
        constrained_layout=True,
    )

    margin_scale = float(_PLOT["margin_color_limit"])
    mesh0 = axes[0].pcolormesh(
        result.detuning_ghz,
        result.source_power_w,
        best_margin_dp,
        shading="nearest",
        cmap=str(_PLOT["margin_colormap"]),
        vmin=-margin_scale,
        vmax=margin_scale,
    )
    axes[0].contour(
        result.detuning_ghz,
        result.source_power_w,
        np.any(robust, axis=0).astype(float),
        levels=(0.5,),
        colors=("black",),
        linewidths=(1.2,),
    )
    axes[0].set_title("Best platform score over waist")
    axes[0].set_xlabel("D1 red detuning (GHz)")
    axes[0].set_ylabel("Source power per branch (W)")
    figure.colorbar(mesh0, ax=axes[0], label="Platform score")

    mesh1 = axes[1].pcolormesh(
        result.detuning_ghz,
        result.waist_um,
        best_margin_dw,
        shading="nearest",
        cmap=str(_PLOT["margin_colormap"]),
        vmin=-margin_scale,
        vmax=margin_scale,
    )
    axes[1].contour(
        result.detuning_ghz,
        result.waist_um,
        np.any(robust, axis=1).astype(float),
        levels=(0.5,),
        colors=("black",),
        linewidths=(1.2,),
    )
    axes[1].set_title("Best platform score over power")
    axes[1].set_xlabel("D1 red detuning (GHz)")
    axes[1].set_ylabel("Waist (µm)")
    figure.colorbar(mesh1, ax=axes[1], label="Platform score")

    if result.recommended is not None:
        point = result.recommended.design
        axes[0].plot(
            point.detuning_ghz,
            point.source_power_w,
            marker="*",
            color="#2563eb",
            markersize=13,
            markeredgecolor="white",
        )
        axes[1].plot(
            point.detuning_ghz,
            point.waist_um,
            marker="*",
            color="#2563eb",
            markersize=13,
            markeredgecolor="white",
        )

    if result.phase_scan:
        phase_pi = [item.phase_rad / math.pi for item in result.phase_scan]
        atoms = [item.expected_atom_number for item in result.phase_scan]
        _, initial_atom_number = _species_values(result.inputs.atom_label)
        atom_errors = [
            initial_atom_number * item.transfer_standard_error
            for item in result.phase_scan
        ]
        axes[2].errorbar(
            phase_pi,
            atoms,
            yerr=atom_errors,
            fmt="o-",
            color="#2563eb",
            linewidth=1.8,
            capsize=2,
        )
        axes[2].axvline(0.0, color="#dc2626", linestyle="--", linewidth=1.2)
        axes[2].set_ylim(
            max(0.0, min(atoms) - 0.05 * initial_atom_number),
            max(atoms) + 0.02 * initial_atom_number,
        )
        axes[2].set_xlabel("AOM relative phase / π")
        axes[2].set_ylabel("Expected retained atoms")
        axes[2].set_title("Experimental phase scan")
        axes[2].text(
            0.03,
            0.05,
            "phase = 0: centers aligned",
            transform=axes[2].transAxes,
            fontsize=8.5,
            color="#991b1b",
        )
    else:
        axes[2].text(
            0.5,
            0.5,
            "No candidate passed\nrobust + MC thresholds",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_axis_off()

    for axis in axes[:2]:
        axis.grid(alpha=0.15)
    figure.suptitle(
        f"{result.inputs.atom_label}: robust detuning–power–waist design "
        f"(±{100 * result.inputs.relative_tolerance:g}% "
        f"{'one-at-a-time' if result.inputs.variation_mode == 'one_at_a_time' else 'joint box'}; "
        f"handover={result.fixed_parameters['time_us']:g} µs)",
        fontweight="bold",
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(_PLOT["dpi"]), facecolor="white")
    plt.close(figure)
    return output
