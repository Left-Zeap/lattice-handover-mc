"""生成论文复现与 Cs 参数选择图。"""

from __future__ import annotations

from pathlib import Path

from .scenarios import reproduce_paper_rb87, scan_cs_designs


def generate_plots(output_dir: str | Path) -> tuple[Path, Path]:
    """生成 Rb 温升路径图和 Cs 功率/散射折中图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    reproduction = reproduce_paper_rb87()
    first, second = reproduction.transport_budget.stages
    temperatures = [
        reproduction.transport_budget.initial_temperature_uK,
        first.output_temperature_uK,
        first.output_temperature_uK
        + reproduction.transport_budget.handover_heating_uK,
        second.output_temperature_uK,
    ]
    labels = ["L1 start", "After L1", "After handover", "Science reservoir"]

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.plot(labels, temperatures, marker="o", linewidth=2.5, color="#2463a6")
    axis.fill_between(
        range(len(temperatures)),
        temperatures,
        alpha=0.12,
        color="#2463a6",
    )
    for index, temperature in enumerate(temperatures):
        axis.annotate(
            f"{temperature:.1f} uK",
            (index, temperature),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
        )
    axis.set_ylim(0.0, max(temperatures) * 1.16)
    axis.set_ylabel("Equivalent temperature (uK)")
    axis.set_title("Rb-87 dual-lattice transport heating budget")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    rb_path = output / "rb87_transport_heating.png"
    fig.savefig(rb_path, dpi=180)
    plt.close(fig)

    candidates = scan_cs_designs(
        target_depth_uK=500.0,
        waist_um=250.0,
        detuning_min_ghz=100.0,
        detuning_max_ghz=2000.0,
        detuning_step_ghz=25.0,
    )
    detuning = [candidate.d1_red_detuning_ghz for candidate in candidates]
    power = [candidate.forward_power_at_atoms_w for candidate in candidates]
    scattering = [candidate.scattering_rate_s for candidate in candidates]

    fig, power_axis = plt.subplots(figsize=(8.2, 4.8))
    scatter_axis = power_axis.twinx()
    power_axis.plot(detuning, power, color="#d97706", linewidth=2.3, label="Power")
    scatter_axis.plot(
        detuning,
        scattering,
        color="#2563eb",
        linewidth=2.3,
        label="Scattering",
    )
    power_axis.axhline(2.0, color="#d97706", linestyle="--", alpha=0.55)
    scatter_axis.axhline(500.0, color="#2563eb", linestyle="--", alpha=0.55)
    power_axis.axvspan(575.0, 675.0, color="#16a34a", alpha=0.09)
    power_axis.set_xlabel("Red detuning from Cs D1 (GHz)")
    power_axis.set_ylabel("Forward power at atoms (W)", color="#d97706")
    scatter_axis.set_ylabel("Scattering rate (1/s)", color="#2563eb")
    power_axis.set_title("Cs-133: 500 uK lattice, 250 um waist, retro ratio 0.88^4")
    power_axis.grid(alpha=0.22)
    fig.tight_layout()
    cs_path = output / "cs133_power_scattering_tradeoff.png"
    fig.savefig(cs_path, dpi=180)
    plt.close(fig)

    return rb_path, cs_path
