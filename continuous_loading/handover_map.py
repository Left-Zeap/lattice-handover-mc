"""失谐量--源端功率可行域内的 handover 交接率二维扫描。

每个网格点先用 ``linear_design`` 相同的完整非线性条件检查是否可行；
只有可行点才运行 ``handover`` 三维经典轨迹 Monte Carlo。L1 与 L2
采用相同的失谐、束腰和满功率。默认参数集中保存在
``data/handover_map_defaults.json``。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np

from .atomic import CS133, RB87, AlkaliAtom
from .handover import HandoverParameters, run_handover_monte_carlo
from .handover_batch import run_handover_monte_carlo_batch
from .lattice import LatticeMetrics, evaluate_lattice, tilted_lattice_barrier_fraction
from .linear_design import LinearDesignInputs, _boundary_functions
from .l1_handover import (
    L1HandoverInputs,
    L1HandoverScanResult,
    analyze_l1_handover_scan,
)
from .l1_transport import l1_transport_inputs_for_species
from .transport import thermal_bound_fraction_3d_harmonic


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "handover_map_defaults.json"
)


def load_handover_map_configuration(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, object]:
    """读取 handover 热力图的集中默认参数。"""
    configuration_path = Path(path)
    payload = json.loads(configuration_path.read_text(encoding="utf-8"))
    required = {
        "scan",
        "monte_carlo",
        "parallel",
        "handover",
        "preconditions",
        "pipeline",
        "shared_design",
        "species",
        "plot",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            "handover map 配置缺少分组：" + ", ".join(sorted(missing))
        )
    return payload


HANDOVER_MAP_CONFIGURATION = load_handover_map_configuration()
_SCAN_DEFAULTS = HANDOVER_MAP_CONFIGURATION["scan"]
_MONTE_CARLO_DEFAULTS = HANDOVER_MAP_CONFIGURATION["monte_carlo"]
_PARALLEL_DEFAULTS = HANDOVER_MAP_CONFIGURATION["parallel"]
_HANDOVER_DEFAULTS = HANDOVER_MAP_CONFIGURATION["handover"]
_PRECONDITION_DEFAULTS = HANDOVER_MAP_CONFIGURATION["preconditions"]
_PIPELINE_DEFAULTS = HANDOVER_MAP_CONFIGURATION["pipeline"]
_SHARED_DESIGN_DEFAULTS = HANDOVER_MAP_CONFIGURATION["shared_design"]
_SPECIES_DEFAULTS = HANDOVER_MAP_CONFIGURATION["species"]
_PLOT_DEFAULTS = HANDOVER_MAP_CONFIGURATION["plot"]

HANDOVER_TIME_US = float(_HANDOVER_DEFAULTS["time_us"])


@dataclass(frozen=True)
class SpeciesHandoverDefaults:
    """单个原子体系用于 LP 筛选和 Monte Carlo 的默认参数。"""

    atom_label: str
    temperature_uK: float
    crossing_angle_deg: float = 4.0
    cloud_axial_sigma_mm: float = 0.5
    l2_transverse_offset_um: float = 0.0
    randomize_relative_phase: bool = True


RB87_HANDOVER_DEFAULTS = SpeciesHandoverDefaults(
    atom_label="Rb-87",
    temperature_uK=float(_SPECIES_DEFAULTS["Rb-87"]["initial_temperature_uK"]),
    crossing_angle_deg=float(
        _SPECIES_DEFAULTS["Rb-87"]["crossing_angle_deg"]
    ),
    cloud_axial_sigma_mm=float(
        _SPECIES_DEFAULTS["Rb-87"]["cloud_axial_sigma_mm"]
    ),
    l2_transverse_offset_um=float(
        _SPECIES_DEFAULTS["Rb-87"]["l2_transverse_offset_um"]
    ),
    randomize_relative_phase=bool(
        _SPECIES_DEFAULTS["Rb-87"]["randomize_relative_phase"]
    ),
)
CS133_HANDOVER_DEFAULTS = SpeciesHandoverDefaults(
    atom_label="Cs-133",
    temperature_uK=float(
        _SPECIES_DEFAULTS["Cs-133"]["initial_temperature_uK"]
    ),
    crossing_angle_deg=float(
        _SPECIES_DEFAULTS["Cs-133"]["crossing_angle_deg"]
    ),
    cloud_axial_sigma_mm=float(
        _SPECIES_DEFAULTS["Cs-133"]["cloud_axial_sigma_mm"]
    ),
    l2_transverse_offset_um=float(
        _SPECIES_DEFAULTS["Cs-133"]["l2_transverse_offset_um"]
    ),
    randomize_relative_phase=bool(
        _SPECIES_DEFAULTS["Cs-133"]["randomize_relative_phase"]
    ),
)


@dataclass(frozen=True)
class HandoverMapInputs:
    """双原子体系 handover 热力图的扫描和数值输入。"""

    detuning_min_ghz: float = float(_SCAN_DEFAULTS["detuning_min_ghz"])
    detuning_max_ghz: float = float(_SCAN_DEFAULTS["detuning_max_ghz"])
    source_power_min_w: float = float(_SCAN_DEFAULTS["source_power_min_w"])
    source_power_max_w: float = float(_SCAN_DEFAULTS["source_power_max_w"])
    detuning_points: int = int(_SCAN_DEFAULTS["detuning_points"])
    power_points: int = int(_SCAN_DEFAULTS["power_points"])
    particle_count: int = int(_MONTE_CARLO_DEFAULTS["particle_count"])
    time_step_us: float = float(_MONTE_CARLO_DEFAULTS["time_step_us"])
    include_scattering: bool = bool(
        _MONTE_CARLO_DEFAULTS["include_scattering"]
    )
    seed: int = int(_MONTE_CARLO_DEFAULTS["seed"])
    compute_backend: str = str(
        _MONTE_CARLO_DEFAULTS.get("compute_backend", "cpu")
    )
    handover_time_us: float = HANDOVER_TIME_US
    parallel_backend: str = str(_PARALLEL_DEFAULTS["backend"])
    worker_count: int = int(_PARALLEL_DEFAULTS["worker_count"])
    require_minimum_depth: bool = bool(
        _PRECONDITION_DEFAULTS["minimum_depth"]
    )
    require_thermal_bound_fraction: bool = bool(
        _PRECONDITION_DEFAULTS["thermal_bound_fraction"]
    )
    require_minimum_axial_cycles: bool = bool(
        _PRECONDITION_DEFAULTS["minimum_axial_cycles"]
    )
    use_l1_transport: bool = bool(
        _PIPELINE_DEFAULTS["use_l1_transport"]
    )
    write_l1_outputs: bool = bool(
        _PIPELINE_DEFAULTS["write_l1_outputs"]
    )

    def __post_init__(self) -> None:
        finite = {
            "最小失谐": self.detuning_min_ghz,
            "最大失谐": self.detuning_max_ghz,
            "最小功率": self.source_power_min_w,
            "最大功率": self.source_power_max_w,
            "时间步长": self.time_step_us,
            "交接时间": self.handover_time_us,
        }
        for name, value in finite.items():
            if not math.isfinite(value):
                raise ValueError(f"{name}必须是有限数")
        if self.detuning_min_ghz <= 0.0:
            raise ValueError("最小红失谐必须为正")
        if self.detuning_max_ghz <= self.detuning_min_ghz:
            raise ValueError("最大红失谐必须大于最小红失谐")
        if self.source_power_min_w < 0.0:
            raise ValueError("最小功率不能为负")
        if self.source_power_max_w <= self.source_power_min_w:
            raise ValueError("最大功率必须大于最小功率")
        if self.detuning_points < 2 or self.power_points < 2:
            raise ValueError("失谐和功率网格点数都至少为 2")
        if self.particle_count <= 0:
            raise ValueError("Monte Carlo 粒子数必须为正")
        if self.time_step_us <= 0.0:
            raise ValueError("时间步长必须为正")
        if self.handover_time_us <= 0.0:
            raise ValueError("交接时间必须为正")
        if self.parallel_backend not in {"serial", "process"}:
            raise ValueError("并行后端必须是 serial 或 process")
        if self.compute_backend not in {"cpu", "gpu"}:
            raise ValueError("计算后端必须是 cpu 或 gpu")
        if self.worker_count <= 0:
            raise ValueError("CPU 工作进程数必须为正")


@dataclass(frozen=True)
class SpeciesHandoverMap:
    """一个原子体系的可行掩膜、交接率和 handover 升温。"""

    atom_label: str
    temperature_uK: float
    defaults: SpeciesHandoverDefaults
    design_inputs: LinearDesignInputs
    detuning_ghz: tuple[float, ...]
    source_power_w: tuple[float, ...]
    feasible: tuple[tuple[bool, ...], ...]
    transfer_efficiency: tuple[tuple[float | None, ...], ...]
    transfer_standard_error: tuple[tuple[float | None, ...], ...]
    handover_heating_uK: tuple[tuple[float | None, ...], ...]
    evaluated_points: int


@dataclass(frozen=True)
class DualSpeciesHandoverMap:
    """Rb-87 与 Cs-133 两张 handover 效率图的联合结果。"""

    inputs: HandoverMapInputs
    handover_time_us: float
    species: tuple[SpeciesHandoverMap, ...]


def _atom_from_defaults(defaults: SpeciesHandoverDefaults) -> AlkaliAtom:
    if defaults.atom_label == "Rb-87":
        return RB87
    if defaults.atom_label == "Cs-133":
        return CS133
    raise ValueError("只支持 Rb-87 或 Cs-133")


def _design_inputs(
    defaults: SpeciesHandoverDefaults,
    scan: HandoverMapInputs,
) -> LinearDesignInputs:
    """返回与该原子体系对应的完整非线性筛选参数。"""
    return LinearDesignInputs(
        atom_label=defaults.atom_label,
        detuning_min_ghz=scan.detuning_min_ghz,
        detuning_max_ghz=scan.detuning_max_ghz,
        waist_um=float(_SHARED_DESIGN_DEFAULTS["waist_um"]),
        target_depth_uK=float(
            _SHARED_DESIGN_DEFAULTS["target_depth_uK"]
        ),
        design_temperature_uK=defaults.temperature_uK,
        target_bound_fraction=float(
            _SHARED_DESIGN_DEFAULTS["target_bound_fraction"]
        ),
        acceleration_m_s2=float(
            _SHARED_DESIGN_DEFAULTS[
                "post_handover_acceleration_m_s2"
            ]
        ),
        handover_min_axial_cycles=float(
            _SHARED_DESIGN_DEFAULTS["minimum_axial_cycles"]
        ),
        max_source_power_w=scan.source_power_max_w,
        max_scattering_rate_s=float(
            _SHARED_DESIGN_DEFAULTS["max_scattering_rate_s"]
        ),
        delivery_efficiency=float(
            _SHARED_DESIGN_DEFAULTS["delivery_efficiency"]
        ),
        retro_power_ratio=float(
            _SHARED_DESIGN_DEFAULTS["retro_power_ratio"]
        ),
    )


def _lattice_if_feasible(
    atom: AlkaliAtom,
    design: LinearDesignInputs,
    detuning_ghz: float,
    source_power_w: float,
    handover_time_us: float = HANDOVER_TIME_US,
    *,
    require_minimum_depth: bool = True,
    require_thermal_bound_fraction: bool = True,
    require_minimum_axial_cycles: bool = True,
) -> LatticeMetrics | None:
    """检查可进入轨迹 MC 的动力学可行点。

    散射已由 Monte Carlo 显式计算，因此最大散射率在图中保留为工程
    参考线，不再提前删掉扫描点。
    """
    if source_power_w <= 0.0 or source_power_w > design.max_source_power_w:
        return None
    wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning_ghz)
    lattice = evaluate_lattice(
        atom,
        wavelength_nm,
        forward_power_w=source_power_w * design.delivery_efficiency,
        waist_um=design.waist_um,
        retro_power_ratio=design.retro_power_ratio,
    )
    barrier_fraction = tilted_lattice_barrier_fraction(
        design.acceleration_m_s2,
        lattice.critical_axial_acceleration_m_s2,
    )
    bound_fraction = thermal_bound_fraction_3d_harmonic(
        lattice.depth_uK * barrier_fraction,
        design.design_temperature_uK,
    )
    axial_cycles = lattice.axial_frequency_hz * handover_time_us * 1e-6
    feasible = (
        (
            not require_minimum_depth
            or lattice.depth_uK >= design.target_depth_uK
        )
        and (
            not require_thermal_bound_fraction
            or bound_fraction >= design.target_bound_fraction
        )
        and (
            not require_minimum_axial_cycles
            or axial_cycles >= design.handover_min_axial_cycles
        )
    )
    return lattice if feasible else None


def _handover_parameters(
    atom: AlkaliAtom,
    defaults: SpeciesHandoverDefaults,
    design: LinearDesignInputs,
    scan: HandoverMapInputs,
    lattice: LatticeMetrics,
) -> HandoverParameters:
    """把一个可行设计点转换为现有轨迹 Monte Carlo 的输入。"""
    return HandoverParameters(
        atom_mass_kg=atom.mass_kg,
        wavelength_nm=lattice.laser_wavelength_nm,
        depth1_uK=lattice.depth_uK,
        depth2_uK=lattice.depth_uK,
        waist1_um=design.waist_um,
        waist2_um=design.waist_um,
        scattering_rate1_s=lattice.scattering_rate_s,
        scattering_rate2_s=lattice.scattering_rate_s,
        retro_power_ratio=design.retro_power_ratio,
        temperature_uK=defaults.temperature_uK,
        duration_ms=scan.handover_time_us * 1e-3,
        crossing_angle_deg=defaults.crossing_angle_deg,
        cloud_axial_sigma_mm=defaults.cloud_axial_sigma_mm,
        l2_transverse_offset_um=defaults.l2_transverse_offset_um,
        randomize_relative_phase=defaults.randomize_relative_phase,
        post_handover_acceleration_m_s2=design.acceleration_m_s2,
        include_scattering=scan.include_scattering,
        compute_backend=scan.compute_backend,
        particle_count=scan.particle_count,
        time_step_us=scan.time_step_us,
        trace_points=2,
        seed=scan.seed,
    )


def _run_handover_map_point(
    task: tuple[tuple[int, int], HandoverParameters],
) -> tuple[tuple[int, int], float, float, float | None]:
    """可由独立 CPU 进程执行的单网格点 Monte Carlo。"""
    grid_index, parameters = task
    try:
        result = run_handover_monte_carlo(parameters)
    except Exception:  # noqa: BLE001 - 单点异常按该点原子全部丢失处理
        return grid_index, 0.0, 0.0, None
    return (
        grid_index,
        result.transfer_efficiency,
        result.transfer_standard_error,
        getattr(result, "handover_heating_uK", None),
    )


def analyze_species_handover_map(
    defaults: SpeciesHandoverDefaults,
    scan: HandoverMapInputs = HandoverMapInputs(),
    *,
    progress: Callable[[str], None] | None = None,
) -> SpeciesHandoverMap:
    """在一个原子体系的可行网格点运行 handover Monte Carlo。"""
    atom = _atom_from_defaults(defaults)
    design = _design_inputs(defaults, scan)
    detunings = np.linspace(
        scan.detuning_min_ghz,
        scan.detuning_max_ghz,
        scan.detuning_points,
    )
    powers = np.linspace(
        scan.source_power_min_w,
        scan.source_power_max_w,
        scan.power_points,
    )
    lattices: dict[tuple[int, int], LatticeMetrics] = {}
    for power_index, power in enumerate(powers):
        for detuning_index, detuning in enumerate(detunings):
            lattice = _lattice_if_feasible(
                atom,
                design,
                float(detuning),
                float(power),
                scan.handover_time_us,
                require_minimum_depth=scan.require_minimum_depth,
                require_thermal_bound_fraction=(
                    scan.require_thermal_bound_fraction
                ),
                require_minimum_axial_cycles=(
                    scan.require_minimum_axial_cycles
                ),
            )
            if lattice is not None:
                lattices[(power_index, detuning_index)] = lattice

    efficiency: list[list[float | None]] = [
        [None] * scan.detuning_points for _ in range(scan.power_points)
    ]
    standard_error: list[list[float | None]] = [
        [None] * scan.detuning_points for _ in range(scan.power_points)
    ]
    heating: list[list[float | None]] = [
        [None] * scan.detuning_points for _ in range(scan.power_points)
    ]
    total = len(lattices)
    # GPU 后端：禁止外层进程池（多进程共享单 GPU 会竞争 CUDA 上下
    # 文），改为把全部网格点的 handover Monte Carlo 摊平成一次批量
    # GPU 调用（见 handover_batch.py）。
    use_gpu_batch = scan.compute_backend == "gpu"
    use_processes = (
        scan.parallel_backend == "process"
        and scan.worker_count > 1
        and total > 1
        and not use_gpu_batch
    )
    effective_workers = (
        min(scan.worker_count, total, os.cpu_count() or 1)
        if total
        else 1
    )
    if progress is not None:
        if use_gpu_batch:
            mode = "GPU 批量（全部网格点单次批量调用）"
        else:
            mode = (
                f"{effective_workers} 个 CPU 进程"
                if use_processes
                else "串行"
            )
        progress(
            f"{defaults.atom_label}: {total} 个可行网格点需要 "
            f"Monte Carlo（{mode}）"
        )
    tasks = [
        (
            grid_index,
            _handover_parameters(
                atom,
                defaults,
                design,
                scan,
                lattice,
            ),
        )
        for grid_index, lattice in lattices.items()
    ]

    def store(
        completed: int,
        point: tuple[tuple[int, int], float, float, float | None],
    ) -> None:
        (
            (power_index, detuning_index),
            point_efficiency,
            point_error,
            point_heating,
        ) = point
        efficiency[power_index][detuning_index] = point_efficiency
        standard_error[power_index][detuning_index] = point_error
        heating[power_index][detuning_index] = point_heating
        if progress is not None and (
            completed == total or completed % max(1, total // 10) == 0
        ):
            progress(f"{defaults.atom_label}: {completed}/{total}")

    if use_gpu_batch:
        if progress is not None:
            progress(
                f"{defaults.atom_label}: 正在 GPU 批量运行 {total} "
                "个点的 handover Monte Carlo（首次需编译内核）"
            )
        results = run_handover_monte_carlo_batch(
            [parameters for _, parameters in tasks],
            backend="gpu",
            progress=progress,
        )
        for completed, ((grid_index, _), result) in enumerate(
            zip(tasks, results), start=1
        ):
            store(
                completed,
                (
                    grid_index,
                    result.transfer_efficiency,
                    result.transfer_standard_error,
                    getattr(result, "handover_heating_uK", None),
                ),
            )
    elif use_processes:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(_run_handover_map_point, task)
                for task in tasks
            ]
            for completed, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                store(completed, future.result())
    else:
        for completed, task in enumerate(tasks, start=1):
            store(completed, _run_handover_map_point(task))

    feasible = tuple(
        tuple((power_index, detuning_index) in lattices for detuning_index in range(scan.detuning_points))
        for power_index in range(scan.power_points)
    )
    return SpeciesHandoverMap(
        atom_label=defaults.atom_label,
        temperature_uK=defaults.temperature_uK,
        defaults=defaults,
        design_inputs=design,
        detuning_ghz=tuple(float(value) for value in detunings),
        source_power_w=tuple(float(value) for value in powers),
        feasible=feasible,
        transfer_efficiency=tuple(tuple(row) for row in efficiency),
        transfer_standard_error=tuple(tuple(row) for row in standard_error),
        handover_heating_uK=tuple(tuple(row) for row in heating),
        evaluated_points=total,
    )


def analyze_dual_species_handover_map(
    scan: HandoverMapInputs = HandoverMapInputs(),
    *,
    progress: Callable[[str], None] | None = None,
) -> DualSpeciesHandoverMap:
    """计算 Rb-87 和 Cs-133 的联合失谐--功率 handover 图。"""
    if scan.use_l1_transport:
        result, _ = analyze_dual_species_l1_handover_map(
            scan,
            progress=progress,
        )
        return result
    species = tuple(
        analyze_species_handover_map(defaults, scan, progress=progress)
        for defaults in (CS133_HANDOVER_DEFAULTS, RB87_HANDOVER_DEFAULTS)
    )
    return DualSpeciesHandoverMap(
        inputs=scan,
        handover_time_us=scan.handover_time_us,
        species=species,
    )


def _integrated_species_handover_map(
    defaults: SpeciesHandoverDefaults,
    scan: HandoverMapInputs,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[SpeciesHandoverMap, L1HandoverScanResult]:
    """先完成 L1 运输，再把逐点末态传入 handover。"""
    transport = replace(
        l1_transport_inputs_for_species(defaults.atom_label),
        detuning_min_ghz=scan.detuning_min_ghz,
        detuning_max_ghz=scan.detuning_max_ghz,
        detuning_points=scan.detuning_points,
        handover_source_power_min_w=scan.source_power_min_w,
        handover_source_power_max_w=scan.source_power_max_w,
        power_points=scan.power_points,
    )
    inputs = L1HandoverInputs(
        transport=transport,
        duration_us=scan.handover_time_us,
        particle_count=scan.particle_count,
        time_step_us=scan.time_step_us,
        trace_points=2,
        include_scattering=scan.include_scattering,
        seed=scan.seed,
        parallel_backend=scan.parallel_backend,
        worker_count=scan.worker_count,
    )
    integrated = analyze_l1_handover_scan(inputs, progress=progress)
    adjusted_defaults = replace(
        defaults,
        temperature_uK=transport.initial_temperature_uK,
        crossing_angle_deg=inputs.crossing_angle_deg,
        cloud_axial_sigma_mm=inputs.cloud_axial_sigma_mm,
        l2_transverse_offset_um=inputs.l2_transverse_offset_um,
        randomize_relative_phase=inputs.randomize_relative_phase,
    )
    design = _design_inputs(adjusted_defaults, scan)
    atom = _atom_from_defaults(adjusted_defaults)
    feasible: list[list[bool]] = [
        [False] * scan.detuning_points for _ in range(scan.power_points)
    ]
    efficiency: list[list[float | None]] = [
        [None] * scan.detuning_points for _ in range(scan.power_points)
    ]
    standard_error: list[list[float | None]] = [
        [None] * scan.detuning_points for _ in range(scan.power_points)
    ]
    heating: list[list[float | None]] = [
        [None] * scan.detuning_points for _ in range(scan.power_points)
    ]
    evaluated = 0
    for power_index, power in enumerate(integrated.source_power_w):
        for detuning_index, detuning in enumerate(integrated.detuning_ghz):
            point_efficiency = integrated.handover_transfer_efficiency[
                power_index
            ][detuning_index]
            if point_efficiency is None:
                continue
            lattice = _lattice_if_feasible(
                atom,
                design,
                detuning,
                power,
                scan.handover_time_us,
                require_minimum_depth=scan.require_minimum_depth,
                require_thermal_bound_fraction=(
                    scan.require_thermal_bound_fraction
                ),
                require_minimum_axial_cycles=(
                    scan.require_minimum_axial_cycles
                ),
            )
            if lattice is None:
                continue
            feasible[power_index][detuning_index] = True
            efficiency[power_index][detuning_index] = point_efficiency
            standard_error[power_index][detuning_index] = (
                integrated.handover_transfer_standard_error[
                    power_index
                ][detuning_index]
            )
            heating[power_index][detuning_index] = (
                integrated.handover_heating_uK[
                    power_index
                ][detuning_index]
            )
            evaluated += 1
    return (
        SpeciesHandoverMap(
            atom_label=adjusted_defaults.atom_label,
            temperature_uK=transport.initial_temperature_uK,
            defaults=adjusted_defaults,
            design_inputs=design,
            detuning_ghz=integrated.detuning_ghz,
            source_power_w=integrated.source_power_w,
            feasible=tuple(tuple(row) for row in feasible),
            transfer_efficiency=tuple(tuple(row) for row in efficiency),
            transfer_standard_error=tuple(
                tuple(row) for row in standard_error
            ),
            handover_heating_uK=tuple(tuple(row) for row in heating),
            evaluated_points=evaluated,
        ),
        integrated,
    )


def analyze_dual_species_l1_handover_map(
    scan: HandoverMapInputs = HandoverMapInputs(),
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[DualSpeciesHandoverMap, tuple[L1HandoverScanResult, ...]]:
    """用同一批 L1→handover 结果生成双物种效率图和全流程图。"""
    species_maps: list[SpeciesHandoverMap] = []
    integrated_results: list[L1HandoverScanResult] = []
    for defaults in (CS133_HANDOVER_DEFAULTS, RB87_HANDOVER_DEFAULTS):
        species_map, integrated = _integrated_species_handover_map(
            defaults,
            scan,
            progress=progress,
        )
        species_maps.append(species_map)
        integrated_results.append(integrated)
    return (
        DualSpeciesHandoverMap(
            inputs=scan,
            handover_time_us=scan.handover_time_us,
            species=tuple(species_maps),
        ),
        tuple(integrated_results),
    )


def plot_dual_species_handover_map(
    result: DualSpeciesHandoverMap,
    output_path: str | Path,
) -> Path:
    """绘制 Rb-87、Cs-133 可行区域内的 Monte Carlo 交接率热力图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(
            float(_PLOT_DEFAULTS["figure_width_in"]),
            1.65 * float(_PLOT_DEFAULTS["figure_height_in"]),
        ),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    boundary_colors = (
        "#2563eb",
        "#7c3aed",
        "#d97706",
        "#dc2626",
        "#374151",
    )
    boundary_labels = (
        "Minimum trap depth",
        "Handover axial-cycle limit",
        "Accelerated bound-fraction limit",
        "Maximum scattering (advisory)",
        "Maximum source power",
    )
    efficiency_mesh = None
    heating_mesh = None
    preconditions_enabled = any(
        (
            result.inputs.require_minimum_depth,
            result.inputs.require_thermal_bound_fraction,
            result.inputs.require_minimum_axial_cycles,
        )
    )
    efficiency_cmap = plt.get_cmap(
        str(_PLOT_DEFAULTS["colormap"])
    ).copy()
    efficiency_cmap.set_bad("#d1d5db")
    heating_cmap = plt.get_cmap(
        str(_PLOT_DEFAULTS["colormap"])
    ).copy()
    heating_cmap.set_bad("#d1d5db")
    for column, species in enumerate(result.species):
        axis = axes[0, column]
        values = np.array(species.transfer_efficiency, dtype=float)
        masked = np.ma.masked_invalid(values)
        efficiency_mesh = axis.pcolormesh(
            species.detuning_ghz,
            species.source_power_w,
            masked,
            shading="nearest",
            cmap=efficiency_cmap,
            vmin=float(_PLOT_DEFAULTS["efficiency_color_min"]),
            vmax=float(_PLOT_DEFAULTS["efficiency_color_max"]),
            zorder=1,
        )
        boundary_detuning = np.linspace(
            result.inputs.detuning_min_ghz,
            result.inputs.detuning_max_ghz,
            int(_PLOT_DEFAULTS["boundary_sample_points"]),
        )
        functions = _boundary_functions(
            species.design_inputs,
            result.handover_time_us * 1e-3,
        )
        for color, label, (_, _, function) in zip(
            boundary_colors,
            boundary_labels,
            functions,
        ):
            axis.plot(
                boundary_detuning,
                [function(float(value)) for value in boundary_detuning],
                color=color,
                linewidth=1.8,
                label=label,
                zorder=4,
            )
        feasible = np.array(species.feasible, dtype=float)
        if (
            preconditions_enabled
            and np.any(feasible)
            and np.any(feasible < 1.0)
        ):
            axis.contour(
                species.detuning_ghz,
                species.source_power_w,
                feasible,
                levels=(0.5,),
                colors=("#202936",),
                linewidths=(1.2,),
                linestyles=("--",),
                zorder=3,
            )
        axis.set_title(
            f"{species.atom_label}  (T={species.temperature_uK:g} µK)",
            fontweight="bold",
        )
        axis.set_xlabel("D1 red detuning (GHz)")
        axis.set_xlim(
            result.inputs.detuning_min_ghz,
            result.inputs.detuning_max_ghz,
        )
        axis.set_ylim(
            result.inputs.source_power_min_w,
            result.inputs.source_power_max_w,
        )
        axis.grid(alpha=0.15)
        axis.text(
            0.98,
            0.98,
            f"MC points: {species.evaluated_points}",
            transform=axis.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            color="#202936",
        )
        handles, labels = axis.get_legend_handles_labels()
        if preconditions_enabled:
            handles.append(
                Line2D(
                    (0,),
                    (0,),
                    color="#202936",
                    linewidth=1.2,
                    linestyle="--",
                )
            )
            labels.append("Exact feasible boundary")
        axis.legend(
            handles,
            labels,
            loc="upper left",
            fontsize=7.5,
            framealpha=0.9,
        )
        heating_values = np.array(species.handover_heating_uK, dtype=float)
        heating_axis = axes[1, column]
        heating_mesh = heating_axis.pcolormesh(
            species.detuning_ghz,
            species.source_power_w,
            np.ma.masked_invalid(heating_values),
            shading="nearest",
            cmap=heating_cmap,
            zorder=1,
        )
        heating_axis.set_xlabel("D1 red detuning (GHz)")
        heating_axis.set_xlim(
            result.inputs.detuning_min_ghz,
            result.inputs.detuning_max_ghz,
        )
        heating_axis.set_ylim(
            result.inputs.source_power_min_w,
            result.inputs.source_power_max_w,
        )
        heating_axis.set_title(f"{species.atom_label} handover heating")
        heating_axis.grid(alpha=0.15)
    axes[0, 0].set_ylabel("Source power per lattice branch (W)")
    axes[1, 0].set_ylabel("Source power per lattice branch (W)")
    if efficiency_mesh is None or heating_mesh is None:
        raise ValueError("没有可绘制的原子体系")
    colorbar = figure.colorbar(
        efficiency_mesh,
        ax=axes[0, :],
        shrink=0.92,
        pad=0.02,
    )
    colorbar.set_label("Handover transfer efficiency")
    heating_colorbar = figure.colorbar(
        heating_mesh,
        ax=axes[1, :],
        shrink=0.92,
        pad=0.02,
    )
    heating_colorbar.set_label("Handover heating (µK)")
    figure.suptitle(
        (
            (
                "Feasible detuning–power region"
                if preconditions_enabled
                else "Full detuning–power scan after L1 transport"
            )
            + " colored by "
            f"{result.handover_time_us:g} µs handover efficiency\n"
            f"Classical trajectory MC: N={result.inputs.particle_count}, "
            f"dt={result.inputs.time_step_us:g} µs; "
            f"backend={result.inputs.parallel_backend}, "
            f"workers={result.inputs.worker_count}; "
            + (
                "blank = constraint-infeasible"
                if preconditions_enabled
                else (
                    "gray = P=0 or no captured-temperature definition"
                )
            )
        ),
        fontweight="bold",
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=int(_PLOT_DEFAULTS["dpi"]),
        facecolor="white",
    )
    plt.close(figure)
    return output
