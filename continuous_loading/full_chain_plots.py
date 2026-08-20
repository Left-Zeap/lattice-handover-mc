"""MOT→L1→handover→L2→科学区 全链路扫描的可视化。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .full_chain import FullChainScanResult
from .l1_transport import L1_TRANSPORT_CONFIGURATION


_PLOT = L1_TRANSPORT_CONFIGURATION["plot"]


def plot_full_chain_scan(
    result: FullChainScanResult,
    output_path: str | Path,
) -> Path:
    """绘制全链路升温/留存热图和两组可选四相（legacy 三相）连续时间轨迹。"""
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
    heating = np.asarray(result.science_total_temperature_rise_uK, dtype=float)
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
        vmax=result.inputs.handover.transport.loading_efficiency,
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
        axis.set_xlim(
            result.inputs.handover.transport.detuning_min_ghz,
            result.inputs.handover.transport.detuning_max_ghz,
        )
        axis.set_ylim(
            result.inputs.handover.transport.handover_source_power_min_w,
            result.inputs.handover.transport.handover_source_power_max_w,
        )
        if have_work_point:
            axis.legend(loc="upper left", fontsize=8)
    axes[0, 0].set_title("MOT → L1 → handover → L2 total temperature rise")
    axes[0, 1].set_title("Science-region atom fraction relative to MOT")
    figure.colorbar(heat_mesh, ax=axes[0, 0], label="Total temperature rise (µK)")
    figure.colorbar(retention_mesh, ax=axes[0, 1], label="Final atom fraction / MOT")

    simulations = (
        (result.optimal_simulation, "Selected optimum", "#2563eb"),
        (result.comparison_simulation, "Poor feasible comparison", "#dc2626"),
    )
    for sim_index, (simulation, label, color) in enumerate(simulations):
        if simulation is None:
            continue
        trace = simulation.combined_trace
        point = simulation.point
        full_label = f"{label}: {point.detuning_ghz:g} GHz, {point.source_power_w:g} W"
        axes[1, 0].plot(trace.time_ms, trace.temperature_uK, color=color, linewidth=1.8, label=full_label)
        axes[1, 1].plot(trace.time_ms, trace.retention_from_mot, color=color, linewidth=1.8, label=full_label)
        span_labels = ("handover", "L2 transport") if sim_index == 0 else (None, None)
        for axis in axes[1]:
            axis.axvspan(
                trace.handover_start_ms,
                trace.handover_end_ms,
                color="#a7f3d0",
                alpha=0.22,
                linewidth=0.0,
                label=span_labels[0],
            )
            axis.axvspan(
                trace.l2_start_ms,
                trace.l2_end_ms,
                color="#bfdbfe",
                alpha=0.22,
                linewidth=0.0,
                label=span_labels[1],
            )
    for axis in axes[1]:
        axis.set_xlabel("Time from L1 launch (ms)")
        axis.grid(alpha=0.2)
        if have_work_point:
            axis.legend(loc="best", fontsize=8)
    axes[1, 0].set_ylabel("Equivalent temperature (µK)")
    axes[1, 0].set_title("L1 + handover + L2 temperature trace")
    axes[1, 1].set_ylabel("Atom fraction relative to MOT")
    axes[1, 1].set_ylim(
        0.0,
        min(1.02, 1.08 * result.inputs.handover.transport.loading_efficiency),
    )
    axes[1, 1].set_title("Full-chain retention to the science region")
    figure.suptitle(
        f"{result.inputs.handover.transport.atom_label}: full chain to science region; "
        f"T_MOT={result.inputs.handover.transport.initial_temperature_uK:g} µK, "
        f"handover MC N={result.inputs.handover.particle_count}; "
        "gray = P=0 or no captured atoms",
        fontweight="bold",
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(_PLOT["dpi"]), facecolor="white")
    plt.close(figure)
    return output
