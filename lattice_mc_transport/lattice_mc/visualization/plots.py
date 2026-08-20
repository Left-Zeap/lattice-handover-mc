from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_single(history_arrays, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_ms = history_arrays["time_s"] * 1e3

    fig = plt.figure(figsize=(7.2, 4.4))
    ax = fig.add_subplot(111)
    ax.plot(t_ms, history_arrays["T_K"] * 1e6, label="T")
    ax.plot(t_ms, history_arrays["Tx_K"] * 1e6, alpha=0.5, label="Tx")
    ax.plot(t_ms, history_arrays["Ty_K"] * 1e6, alpha=0.5, label="Ty")
    ax.plot(t_ms, history_arrays["Tz_K"] * 1e6, alpha=0.5, label="Tz")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Kinetic temperature (uK)")
    ax.set_title("Temperature during L1 -> handover -> L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "temperature_timeseries.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 4.4))
    ax = fig.add_subplot(111)
    ax.plot(t_ms, 100.0 * history_arrays["survival"])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Survival (%)")
    ax.set_ylim(0.0, 101.0)
    ax.set_title("Survival during L1 -> handover -> L2")
    fig.tight_layout()
    fig.savefig(outdir / "survival_timeseries.png", dpi=180)
    plt.close(fig)

def plot_scan(detunings, powers, final_T_uK, final_survival, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    extent = [powers[0], powers[-1], detunings[0], detunings[-1]]

    fig = plt.figure(figsize=(7.0, 5.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        final_T_uK,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    ax.set_xlabel("Forward power per lattice (W)")
    ax.set_ylabel("D1 red detuning (GHz)")
    ax.set_title("Final kinetic temperature (uK)")
    fig.colorbar(im, ax=ax, label="uK")
    fig.tight_layout()
    fig.savefig(outdir / "final_temperature_heatmap.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(7.0, 5.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        100.0 * final_survival,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=0.0,
        vmax=100.0,
    )
    ax.set_xlabel("Forward power per lattice (W)")
    ax.set_ylabel("D1 red detuning (GHz)")
    ax.set_title("Final survival (%)")
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(outdir / "final_survival_heatmap.png", dpi=180)
    plt.close(fig)
