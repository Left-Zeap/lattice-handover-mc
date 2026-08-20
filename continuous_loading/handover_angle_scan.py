"""扫描 L1/L2 夹角并比较 Rb-87 与 Cs-133 的 handover 指标。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np

from .atomic import CS133, RB87, AlkaliAtom
from .handover import HandoverParameters, HandoverResult, run_handover_monte_carlo
from .handover_map import HANDOVER_MAP_CONFIGURATION
from .lattice import evaluate_lattice, power_for_target_depth


_ANGLE_DEFAULTS = HANDOVER_MAP_CONFIGURATION["angle_scan"]
_HANDOVER_DEFAULTS = HANDOVER_MAP_CONFIGURATION["handover"]
_DESIGN_DEFAULTS = HANDOVER_MAP_CONFIGURATION["shared_design"]
_SPECIES_DEFAULTS = HANDOVER_MAP_CONFIGURATION["species"]
_PLOT_DEFAULTS = HANDOVER_MAP_CONFIGURATION["plot"]


@dataclass(frozen=True)
class HandoverAngleScanInputs:
    """双物种夹角扫描的固定物理量和数值参数。"""

    angle_min_deg: float = float(_ANGLE_DEFAULTS["angle_min_deg"])
    angle_max_deg: float = float(_ANGLE_DEFAULTS["angle_max_deg"])
    angle_step_deg: float = float(_ANGLE_DEFAULTS["angle_step_deg"])
    target_depth_uK: float = float(_ANGLE_DEFAULTS["target_depth_uK"])
    waist_um: float = float(_ANGLE_DEFAULTS["waist_um"])
    particle_count: int = int(_ANGLE_DEFAULTS["particle_count"])
    time_step_us: float = float(_ANGLE_DEFAULTS["time_step_us"])
    trace_points: int = int(_ANGLE_DEFAULTS["trace_points"])
    include_scattering: bool = bool(_ANGLE_DEFAULTS["include_scattering"])
    seed: int = int(_ANGLE_DEFAULTS["seed"])
    parallel_backend: str = str(_ANGLE_DEFAULTS["parallel_backend"])
    worker_count: int = int(_ANGLE_DEFAULTS["worker_count"])

    def __post_init__(self) -> None:
        finite = {
            "最小夹角": self.angle_min_deg,
            "最大夹角": self.angle_max_deg,
            "夹角步长": self.angle_step_deg,
            "目标阱深": self.target_depth_uK,
            "束腰": self.waist_um,
            "时间步长": self.time_step_us,
        }
        for name, value in finite.items():
            if not math.isfinite(value):
                raise ValueError(f"{name}必须是有限数")
        if self.angle_min_deg < 0.0 or self.angle_max_deg >= 180.0:
            raise ValueError("夹角扫描范围必须位于 [0, 180) 度")
        if self.angle_max_deg < self.angle_min_deg:
            raise ValueError("最大夹角不能小于最小夹角")
        if self.angle_step_deg <= 0.0:
            raise ValueError("夹角步长必须为正")
        if self.target_depth_uK <= 0.0 or self.waist_um <= 0.0:
            raise ValueError("目标阱深和束腰必须为正")
        if self.particle_count <= 0 or self.trace_points < 2:
            raise ValueError("粒子数必须为正且轨迹记录点至少为 2")
        if self.time_step_us <= 0.0:
            raise ValueError("时间步长必须为正")
        if self.parallel_backend not in {"serial", "process"}:
            raise ValueError("并行后端必须是 serial 或 process")
        if self.worker_count <= 0:
            raise ValueError("CPU 工作进程数必须为正")


@dataclass(frozen=True)
class SpeciesHandoverAngleScan:
    """一个原子体系的夹角扫描结果。"""

    atom_label: str
    detuning_ghz: float
    wavelength_nm: float
    required_source_power_w: float
    initial_temperature_uK: float
    angle_deg: tuple[float, ...]
    transfer_efficiency: tuple[float, ...]
    transfer_standard_error: tuple[float, ...]
    handover_heating_uK: tuple[float | None, ...]
    all_atom_handover_heating_uK: tuple[float, ...]


@dataclass(frozen=True)
class DualSpeciesHandoverAngleScan:
    """Rb-87 与 Cs-133 的联合夹角扫描。"""

    inputs: HandoverAngleScanInputs
    species: tuple[SpeciesHandoverAngleScan, ...]


def _atom(atom_label: str) -> AlkaliAtom:
    if atom_label == "Rb-87":
        return RB87
    if atom_label == "Cs-133":
        return CS133
    raise ValueError("角度扫描只支持 Rb-87 和 Cs-133")


def _angles(inputs: HandoverAngleScanInputs) -> np.ndarray:
    count = int(
        math.floor(
            (inputs.angle_max_deg - inputs.angle_min_deg)
            / inputs.angle_step_deg
            + 1e-12
        )
    )
    return inputs.angle_min_deg + inputs.angle_step_deg * np.arange(count + 1)


def _base_parameters(
    atom_label: str,
    inputs: HandoverAngleScanInputs,
) -> tuple[HandoverParameters, float, float]:
    atom = _atom(atom_label)
    detuning_ghz = float(
        _ANGLE_DEFAULTS["species_detuning_ghz"][atom_label]
    )
    wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning_ghz)
    retro_power_ratio = float(_DESIGN_DEFAULTS["retro_power_ratio"])
    forward_power_w = power_for_target_depth(
        atom,
        wavelength_nm,
        inputs.target_depth_uK,
        inputs.waist_um,
        retro_power_ratio,
    )
    lattice = evaluate_lattice(
        atom,
        wavelength_nm,
        forward_power_w,
        inputs.waist_um,
        retro_power_ratio,
    )
    source_power_w = forward_power_w / float(
        _DESIGN_DEFAULTS["delivery_efficiency"]
    )
    species = _SPECIES_DEFAULTS[atom_label]
    parameters = HandoverParameters(
        atom_mass_kg=atom.mass_kg,
        wavelength_nm=wavelength_nm,
        depth1_uK=lattice.depth_uK,
        depth2_uK=lattice.depth_uK,
        waist1_um=inputs.waist_um,
        waist2_um=inputs.waist_um,
        scattering_rate1_s=lattice.scattering_rate_s,
        scattering_rate2_s=lattice.scattering_rate_s,
        retro_power_ratio=retro_power_ratio,
        temperature_uK=float(species["initial_temperature_uK"]),
        duration_ms=float(_HANDOVER_DEFAULTS["time_us"]) * 1e-3,
        crossing_angle_deg=inputs.angle_min_deg,
        cloud_axial_sigma_mm=float(species["cloud_axial_sigma_mm"]),
        l2_transverse_offset_um=float(species["l2_transverse_offset_um"]),
        randomize_relative_phase=bool(species["randomize_relative_phase"]),
        post_handover_acceleration_m_s2=float(
            _DESIGN_DEFAULTS["post_handover_acceleration_m_s2"]
        ),
        include_scattering=inputs.include_scattering,
        particle_count=inputs.particle_count,
        time_step_us=inputs.time_step_us,
        trace_points=inputs.trace_points,
        seed=inputs.seed,
    )
    return parameters, detuning_ghz, source_power_w


def _run_angle_point(
    task: tuple[int, float, HandoverParameters],
) -> tuple[int, HandoverResult]:
    index, angle_deg, base = task
    return index, run_handover_monte_carlo(
        replace(base, crossing_angle_deg=angle_deg)
    )


def analyze_handover_angle_scan(
    inputs: HandoverAngleScanInputs = HandoverAngleScanInputs(),
    *,
    progress: Callable[[str], None] | None = None,
) -> DualSpeciesHandoverAngleScan:
    """按 1° 默认步长扫描双晶格夹角。"""
    angles = _angles(inputs)
    species_results: list[SpeciesHandoverAngleScan] = []
    for atom_label in ("Rb-87", "Cs-133"):
        base, detuning_ghz, source_power_w = _base_parameters(
            atom_label,
            inputs,
        )
        tasks = [
            (index, float(angle), base)
            for index, angle in enumerate(angles)
        ]
        results: list[HandoverResult | None] = [None] * len(tasks)
        use_processes = (
            inputs.parallel_backend == "process"
            and inputs.worker_count > 1
            and len(tasks) > 1
        )
        workers = min(inputs.worker_count, len(tasks), os.cpu_count() or 1)
        if progress is not None:
            mode = f"{workers} 个 CPU 进程" if use_processes else "串行"
            progress(
                f"{atom_label}: {len(tasks)} 个夹角点，"
                f"N={inputs.particle_count}（{mode}）"
            )
        completed = 0
        if use_processes:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(_run_angle_point, task)
                    for task in tasks
                ]
                for future in as_completed(futures):
                    index, result = future.result()
                    results[index] = result
                    completed += 1
                    if progress is not None and (
                        completed == len(tasks) or completed % 10 == 0
                    ):
                        progress(f"{atom_label}: {completed}/{len(tasks)}")
        else:
            for task in tasks:
                index, result = _run_angle_point(task)
                results[index] = result
                completed += 1
                if progress is not None and (
                    completed == len(tasks) or completed % 10 == 0
                ):
                    progress(f"{atom_label}: {completed}/{len(tasks)}")
        completed_results = [result for result in results if result is not None]
        if len(completed_results) != len(tasks):
            raise RuntimeError("夹角扫描结果不完整")
        species_results.append(
            SpeciesHandoverAngleScan(
                atom_label=atom_label,
                detuning_ghz=detuning_ghz,
                wavelength_nm=base.wavelength_nm,
                required_source_power_w=source_power_w,
                initial_temperature_uK=base.temperature_uK,
                angle_deg=tuple(float(value) for value in angles),
                transfer_efficiency=tuple(
                    result.transfer_efficiency
                    for result in completed_results
                ),
                transfer_standard_error=tuple(
                    result.transfer_standard_error
                    for result in completed_results
                ),
                handover_heating_uK=tuple(
                    result.handover_heating_uK
                    for result in completed_results
                ),
                all_atom_handover_heating_uK=tuple(
                    result.all_atom_handover_heating_uK
                    for result in completed_results
                ),
            )
        )
    return DualSpeciesHandoverAngleScan(
        inputs=inputs,
        species=tuple(species_results),
    )


def plot_handover_angle_scan(
    result: DualSpeciesHandoverAngleScan,
    output_path: str | Path,
) -> Path:
    """绘制交接率和 handover 升温随夹角的双面板折线图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.0),
        sharex=True,
        constrained_layout=True,
    )
    colors = {"Rb-87": "#2563eb", "Cs-133": "#dc2626"}
    for species in result.species:
        angle = np.asarray(species.angle_deg)
        efficiency = np.asarray(species.transfer_efficiency)
        error = np.asarray(species.transfer_standard_error)
        heating = np.asarray(
            [
                np.nan if value is None else value
                for value in species.handover_heating_uK
            ]
        )
        all_atom_heating = np.asarray(
            species.all_atom_handover_heating_uK
        )
        label = (
            f"{species.atom_label}: {species.detuning_ghz:g} GHz, "
            f"T={species.initial_temperature_uK:g} µK"
        )
        color = colors[species.atom_label]
        axes[0].plot(angle, efficiency, color=color, linewidth=1.8, label=label)
        axes[0].fill_between(
            angle,
            np.clip(efficiency - error, 0.0, 1.0),
            np.clip(efficiency + error, 0.0, 1.0),
            color=color,
            alpha=0.14,
            linewidth=0.0,
        )
        axes[1].plot(
            angle,
            heating,
            color=color,
            linewidth=1.8,
            label=f"{species.atom_label}: captured by L2",
        )
        axes[1].plot(
            angle,
            all_atom_heating,
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"{species.atom_label}: all atoms",
        )
    axes[0].set_ylabel("Handover efficiency")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Handover efficiency versus L1/L2 crossing angle")
    axes[1].set_xlabel("L1/L2 crossing angle (deg)")
    axes[1].set_ylabel("Handover heating (µK)")
    axes[1].set_title("Handover heating versus L1/L2 crossing angle")
    for axis in axes:
        axis.set_xlim(
            result.inputs.angle_min_deg,
            result.inputs.angle_max_deg,
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=9)
    figure.suptitle(
        f"Fixed {result.inputs.target_depth_uK:g} µK lattice depth; "
        f"{float(_HANDOVER_DEFAULTS['time_us']):g} µs handover; "
        f"MC N={result.inputs.particle_count}",
        fontsize=12,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(_PLOT_DEFAULTS["dpi"]))
    plt.close(figure)
    return output
