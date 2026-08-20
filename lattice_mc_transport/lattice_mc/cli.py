from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .config import load_config, load_json, save_json
from .simulation.runner import run_single, history_arrays
from .visualization.plots import plot_single, plot_scan
from .scan import run_detuning_power_scan
from .backend import to_cpu

def cmd_single(args):
    cfg = load_config(args.config)
    result = run_single(cfg, backend_name=args.backend)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    arr = history_arrays(result["history"])
    np.savez_compressed(outdir / "timeseries.npz", **arr)
    plot_single(arr, outdir)

    last = result["history"][-1]
    summary = {
        "backend": result["backend"],
        "species": cfg["species"],
        "d1_red_detuning_GHz": cfg["laser"]["d1_red_detuning_GHz"],
        "laser_wavelength_nm": result["response"].wavelength_m * 1e9,
        "n_atoms": cfg["initial"]["n_atoms"],
        "final_survival": last["survival"],
        "final_temperature_uK": last["T_K"] * 1e6,
        "final_Txyz_uK": [
            last["Tx_K"] * 1e6,
            last["Ty_K"] * 1e6,
            last["Tz_K"] * 1e6,
        ],
        "total_scatter_events": last["scatter_events_total"],
    }
    save_json(summary, outdir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def cmd_scan(args):
    cfg = load_config(args.config)
    scan_cfg = load_json(args.scan)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    from .backend import get_backend
    xp, resolved = get_backend(args.backend)
    if resolved == "gpu":
        from .scan import run_detuning_power_scan_batched
        n_points = (int(scan_cfg["detuning_GHz"]["num"])
                    * int(scan_cfg["power_w"]["num"]))
        n_atoms = int(scan_cfg.get("n_atoms_override")
                      or cfg["initial"]["n_atoms"])
        print(f"batched scan: {n_points} points x {n_atoms} atoms "
              f"in one ensemble", flush=True)

        t0 = [None]
        import time
        t0[0] = time.perf_counter()

        def progress(done, total):
            el = time.perf_counter() - t0[0]
            print(f"  step {done}/{total} ({100.0*done/total:.0f}%), "
                  f"elapsed {el/60:.1f} min, "
                  f"eta {el/max(done,1)*(total-done)/60:.1f} min", flush=True)

        res = run_detuning_power_scan_batched(cfg, scan_cfg, xp,
                                              progress=progress)
    else:
        res = run_detuning_power_scan(cfg, scan_cfg, backend_name=args.backend)
    np.savez_compressed(outdir / "scan_results.npz", **res)
    plot_scan(
        res["detunings_GHz"], res["powers_W"],
        res["final_T_uK"], res["final_survival"], outdir
    )
    save_json({
        "detunings_GHz": res["detunings_GHz"].tolist(),
        "powers_W": res["powers_W"].tolist(),
        "final_temperature_uK": res["final_T_uK"].tolist(),
        "final_survival": res["final_survival"].tolist(),
    }, outdir / "scan_summary.json")

def build_parser():
    p = argparse.ArgumentParser(
        prog="lattice-mc",
        description="6D MC: L1 transport -> handover -> L2 transport"
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("single", help="run one parameter point")
    s.add_argument("--config", required=True)
    s.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto")
    s.add_argument("--out", default="output/single")
    s.set_defaults(func=cmd_single)

    s = sub.add_parser("scan", help="scan {D1 red detuning, power}")
    s.add_argument("--config", required=True)
    s.add_argument("--scan", required=True)
    s.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto")
    s.add_argument("--out", default="output/scan")
    s.set_defaults(func=cmd_scan)

    return p

def main():
    args = build_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
