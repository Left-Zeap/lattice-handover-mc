from __future__ import annotations
import math
from pathlib import Path
import numpy as np

from ..backend import get_backend, make_rng, to_cpu
from ..physics.atom import load_atom
from ..physics.dipole import response_from_d1_red_detuning
from ..physics.lattice import LatticeGeometry, unit_vector
from .initializer import sample_initial_ensemble
from .propagator import propagate_experiment

def build_geometry(xp, cfg):
    theta = math.radians(float(cfg["geometry"]["handover_angle_deg"]))
    axis1 = unit_vector(xp, [1.0, 0.0, 0.0])
    axis2 = unit_vector(xp, [math.cos(theta), math.sin(theta), 0.0])

    R = float(cfg["laser"]["retro_power_ratio"])
    l1 = cfg["geometry"]["l1"]
    l2 = cfg["geometry"]["l2"]
    geom1 = LatticeGeometry(
        axis=axis1,
        s_points=tuple(map(float, l1["waist_s_m"])),
        waist_points_m=tuple(float(x) * 1e-6 for x in l1["waist_um"]),
        retro_ratio=R,
        phase_offset_rad=float(cfg["laser"]["phase_offset_l1_rad"]),
    )
    geom2 = LatticeGeometry(
        axis=axis2,
        s_points=tuple(map(float, l2["waist_s_m"])),
        waist_points_m=tuple(float(x) * 1e-6 for x in l2["waist_um"]),
        retro_ratio=R,
        phase_offset_rad=float(cfg["laser"]["phase_offset_l2_rad"]),
    )
    return axis1, axis2, geom1, geom2

def run_single(cfg, backend_name=None, progress=None):
    requested = backend_name or cfg["simulation"].get("backend", "auto")
    xp, resolved = get_backend(requested)

    seed = int(cfg["initial"]["seed"])
    rng = make_rng(xp, seed)

    atom = load_atom(cfg["species"])
    response = response_from_d1_red_detuning(
        atom, float(cfg["laser"]["d1_red_detuning_GHz"])
    )
    axis1, axis2, geom1, geom2 = build_geometry(xp, cfg)

    l1 = cfg["geometry"]["l1"]
    q0 = -float(l1["distance_m"])
    state = sample_initial_ensemble(
        xp=xp,
        rng=rng,
        n_atoms=int(cfg["initial"]["n_atoms"]),
        temperature_K=float(cfg["initial"]["temperature_uK"]) * 1e-6,
        atom=atom,
        response=response,
        q0=q0,
        power_w=float(l1["power_w"]),
        geom=geom1,
    )

    history, state = propagate_experiment(
        xp, rng, state, cfg, atom, response, geom1, geom2, axis1, axis2,
        progress=progress,
    )
    return {
        "backend": resolved,
        "atom": atom,
        "response": response,
        "history": history,
        "state": state,
        "xp": xp,
    }

def history_arrays(history):
    out = {}
    numeric_keys = [
        "time_s", "survival", "n_alive", "T_K", "Tx_K", "Ty_K", "Tz_K",
        "scatter_events_total"
    ]
    for k in numeric_keys:
        out[k] = np.asarray([h[k] for h in history])
    out["stage"] = np.asarray([h["stage"] for h in history], dtype="U32")
    return out
