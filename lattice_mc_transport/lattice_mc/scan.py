from __future__ import annotations
from pathlib import Path
import numpy as np
from .config import clone_with_overrides
from .simulation.runner import run_single

def _grid(spec):
    return np.linspace(float(spec["start"]), float(spec["stop"]), int(spec["num"]))

def run_detuning_power_scan(base_cfg, scan_cfg, backend_name="auto"):
    detunings = _grid(scan_cfg["detuning_GHz"])
    powers = _grid(scan_cfg["power_w"])
    n_override = scan_cfg.get("n_atoms_override")

    final_T = np.full((len(detunings), len(powers)), np.nan)
    final_S = np.full_like(final_T, np.nan)

    for i, d in enumerate(detunings):
        for j, p in enumerate(powers):
            cfg = clone_with_overrides(
                base_cfg, detuning_GHz=d, power_w=p, n_atoms=n_override
            )
            result = run_single(cfg, backend_name=backend_name)
            h = result["history"][-1]
            final_T[i, j] = h["T_K"] * 1e6
            final_S[i, j] = h["survival"]
            print(
                f"[{i+1}/{len(detunings)} {j+1}/{len(powers)}] "
                f"det={d:.3f} GHz, P={p:.3f} W -> "
                f"T={final_T[i,j]:.3f} uK, S={100*final_S[i,j]:.2f}%"
            )

    return {
        "detunings_GHz": detunings,
        "powers_W": powers,
        "final_T_uK": final_T,
        "final_survival": final_S,
    }

def run_detuning_power_scan_batched(base_cfg, scan_cfg, xp, progress=None):
    """
    GPU path: propagate the whole {detuning x power} grid in a single
    batched ensemble instead of looping over points sequentially.
    """
    from .physics.atom import load_atom
    from .simulation.runner import build_geometry
    from .simulation.batched import run_batched_scan

    detunings = _grid(scan_cfg["detuning_GHz"])
    powers = _grid(scan_cfg["power_w"])
    n_override = scan_cfg.get("n_atoms_override")
    if n_override is not None:
        base_cfg = clone_with_overrides(base_cfg, n_atoms=n_override)

    atom = load_atom(base_cfg["species"])
    axis1, axis2, geom1, geom2 = build_geometry(xp, base_cfg)
    final_T, final_S = run_batched_scan(
        xp, base_cfg, detunings, powers, atom, axis1, axis2, geom1, geom2,
        progress=progress,
    )
    return {
        "detunings_GHz": detunings,
        "powers_W": powers,
        "final_T_uK": final_T * 1e6,
        "final_survival": final_S,
    }

