"""L1 transport→handover 联合扫描的可视化。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .l1_handover import L1HandoverScanResult
from .l1_transport import L1_TRANSPORT_CONFIGURATION


_PLOT = L1_TRANSPORT_CONFIGURATION["plot"]


def plot_l1_handover_scan(
    result: L1HandoverScanResult,
    output_path: str | Path,
) -> Path:
    """绘制全流程升温/装载率热图和两组连续时间轨迹。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(float(_PLOT["figure_width_in"]), float(_PLOT["figure_height_in"])),
        constrained_layout=True,
    )
    detunings = np.asarray(result.detuning_ghz)
    powers = np.asarray(result.source_power_w)
    heating = np.asarray(result.total_temperature_rise_uK, dtype=float)
    retention = np.asarray(result.final_retention_from_mot, dtype=float)
    temperature_cmap = plt.get_cmap(
        str(_PLOT["temperature_colormap"])
    ).copy()
    temperature_cmap.set_bad("#d1d5db")
    retention_cmap = plt.get_cmap(
        str(_PLOT["retention_colormap"])
    ).copy()
    retention_cmap.set_bad("#d1d5db")
    heat_mesh = axes[0, 0].pcolormesh(
        detunings,
        powers,
        np.ma.masked_invalid(heating),
        shading="nearest",
        cmap=temperature_cmap,
    )
    retention_mesh = axes[0, 1].pcolormesh(
        detunings,
        powers,
        np.ma.masked_invalid(retention),
        shading="nearest",
        cmap=retention_cmap,
        vmin=0.0,
        vmax=result.inputs.transport.loading_efficiency,
    )
    # 全网格失败时 optimal/comparison 为哨兵点且 simulation 为 None：
    # 只画热力图（全灰），不画误导性的"最优点"散点与时间轨迹。
    have_work_point = result.optimal_simulation is not None
    labels = (
        (result.optimal, "Selected optimum", "#2563eb", "*", 14),
        (result.comparison, "Poor feasible comparison", "#dc2626", "X", 9),
    )
    for axis in axes[0]:
        if have_work_point:
            for point, label, color, marker, size in labels:
                axis.plot(
                    point.detuning_ghz,
                    point.source_power_w,
                    marker=marker,
                    color=color,
                    markeredgecolor="white",
                    markersize=size,
                    linestyle="none",
                    label=label,
                )
        axis.set_xlabel("D1 red detuning (GHz)")
        axis.set_ylabel("Handover-end source power per branch (W)")
        axis.set_xlim(result.inputs.transport.detuning_min_ghz, result.inputs.transport.detuning_max_ghz)
        axis.set_ylim(
            result.inputs.transport.handover_source_power_min_w,
            result.inputs.transport.handover_source_power_max_w,
        )
        if have_work_point:
            axis.legend(loc="upper left", fontsize=8)
    axes[0, 0].set_title("MOT → L1 transport → L2 total temperature rise")
    axes[0, 1].set_title("Final L2 atom fraction relative to MOT")
    figure.colorbar(heat_mesh, ax=axes[0, 0], label="Total temperature rise (µK)")
    figure.colorbar(retention_mesh, ax=axes[0, 1], label="Final atom fraction / MOT")

    simulations = (
        (result.optimal_simulation, "Selected optimum", "#2563eb"),
        (result.comparison_simulation, "Poor feasible comparison", "#dc2626"),
    )
    for simulation, label, color in simulations:
        if simulation is None:
            continue
        trace = simulation.combined_trace
        point = simulation.point
        full_label = f"{label}: {point.detuning_ghz:g} GHz, {point.source_power_w:g} W"
        axes[1, 0].plot(trace.time_ms, trace.temperature_uK, color=color, linewidth=1.8, label=full_label)
        axes[1, 1].plot(trace.time_ms, trace.retention_from_mot, color=color, linewidth=1.8, label=full_label)
        for axis in axes[1]:
            axis.axvspan(
                trace.handover_start_ms,
                trace.handover_end_ms,
                color="#a7f3d0",
                alpha=0.22,
                linewidth=0.0,
            )
    for axis in axes[1]:
        axis.set_xlabel("Time from L1 launch (ms)")
        axis.grid(alpha=0.2)
        if have_work_point:
            axis.legend(loc="best", fontsize=8)
    axes[1, 0].set_ylabel("Equivalent temperature (µK)")
    axes[1, 0].set_title("L1 transport + handover temperature trace")
    axes[1, 1].set_ylabel("Atom fraction relative to MOT")
    axes[1, 1].set_ylim(0.0, min(1.02, 1.08 * result.inputs.transport.loading_efficiency))
    axes[1, 1].set_title("L1 transport + endpoint L2 capture retention")
    figure.suptitle(
        f"{result.inputs.transport.atom_label}: shared detuning–power scan; "
        f"T_MOT={result.inputs.transport.initial_temperature_uK:g} µK, "
        f"handover MC N={result.inputs.particle_count}; "
        "gray = P=0 or no captured-temperature definition",
        fontweight="bold",
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(_PLOT["dpi"]), facecolor="white")
    plt.close(figure)
    return output
