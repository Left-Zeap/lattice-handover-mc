"""
Batched propagation: run many parameter points (detuning, power) in one
simulation by stacking their atoms into a single ensemble.

All scan points share the same lattice geometry and timing; the
point-dependent laser quantities (wave number k, Stark shift, scattering
coefficient, power) are carried as per-atom arrays, so the fused GPU kernels
process every point in the same launch. Atoms are laid out contiguously per
point, which makes per-point statistics a plain reshape + axis reduction.
"""
from __future__ import annotations
import math
import numpy as np

from ..backend import make_rng, to_cpu
from ..constants import KB
from ..physics.dipole import response_from_d1_red_detuning
from ..physics.lattice import field_quantities_multi
from ..statistics.survival import update_transport_survival, handover_end_capture
from .initializer import sample_initial_ensemble
from .propagator import (
    runtime_l1, runtime_handover, runtime_l2, _kick,
    _apply_scattering_gpu_multi,
)

def build_batched_state(xp, rng, cfg, atom, points, geom1):
    """
    points: list of (detuning_GHz, power_w).

    Returns (state, per_atom) where state is the concatenated ensemble and
    per_atom holds the per-atom laser parameter arrays.
    """
    l1 = cfg["geometry"]["l1"]
    q0 = -float(l1["distance_m"])
    T = float(cfg["initial"]["temperature_uK"]) * 1e-6
    n = int(cfg["initial"]["n_atoms"])

    parts = []
    pa = {"k": [], "shift": [], "scatter": [], "vmin_sign": [], "c2P_base": []}
    for det_gHz, power_w in points:
        resp = response_from_d1_red_detuning(atom, float(det_gHz))
        st = sample_initial_ensemble(
            xp=xp, rng=rng, n_atoms=n, temperature_K=T, atom=atom,
            response=resp, q0=q0, power_w=float(power_w), geom=geom1,
        )
        parts.append(st)
        shift = float(resp.shift_per_intensity)
        pa["k"].append(xp.full(n, 2.0 * math.pi / resp.wavelength_m))
        pa["shift"].append(xp.full(n, shift))
        pa["scatter"].append(xp.full(n, float(resp.scatter_per_intensity)))
        pa["vmin_sign"].append(xp.full(n, -1.0 if shift < 0.0 else 0.0))
        pa["c2P_base"].append(xp.full(n, 2.0 * float(power_w) / math.pi))

    state = {
        "r": xp.concatenate([p["r"] for p in parts]),
        "v": xp.concatenate([p["v"] for p in parts]),
        "alive": xp.concatenate([p["alive"] for p in parts]),
        "unbound_time": xp.concatenate([p["unbound_time"] for p in parts]),
        "scatter_count": xp.concatenate([p["scatter_count"] for p in parts]),
    }
    per_atom = {k: xp.concatenate(v) for k, v in pa.items()}
    return state, per_atom

def _forces_and_fields_multi(xp, r, runtimes, per_atom):
    gradV = None
    fields = []
    active_rt = []
    for rt in runtimes:
        if rt.power_fraction <= 0.0:
            continue
        c2P_t = per_atom["c2P_base"] * rt.power_fraction
        field = field_quantities_multi(
            xp, r, rt.axis, rt.q, c2P_t, per_atom, rt.geom
        )
        gradV = field["gradV"] if gradV is None else gradV + field["gradV"]
        fields.append(field)
        active_rt.append(rt)
    return gradV, fields, active_rt

def _verlet_step_multi(
    xp, rng, state, runtimes_now, runtimes_next, per_atom, atom,
    gravity_host, dt, enable_scattering,
):
    gV0, fields0, active0 = _forces_and_fields_multi(
        xp, state["r"], runtimes_now, per_atom
    )
    _kick(
        xp, state, gV0, None, gravity_host, atom.mass_kg,
        h_kick=0.5 * dt, h_drift=dt,
    )
    gV1, fields1, active1 = _forces_and_fields_multi(
        xp, state["r"], runtimes_next, per_atom
    )
    _kick(
        xp, state, gV1, None, gravity_host, atom.mass_kg,
        h_kick=0.5 * dt, h_drift=0.0,
    )
    if enable_scattering and fields1:
        _apply_scattering_gpu_multi(
            xp, rng, state, fields1, active1, per_atom, atom, dt
        )
    return fields1, active1

def record_diag_multi(xp, state, atom, t_abs, stage, n_groups, n_per):
    """Per-point statistics; atoms are contiguous per point."""
    v3 = state["v"].reshape(n_groups, n_per, 3)
    alive2 = state["alive"].reshape(n_groups, n_per)
    n_alive = xp.sum(alive2, axis=1)
    cnt = xp.maximum(n_alive, 1).astype(float)[:, None]
    mask = alive2[:, :, None]
    mean = xp.sum(xp.where(mask, v3, 0.0), axis=1) / cnt
    diff = xp.where(mask, v3 - mean[:, None, :], 0.0)
    var = xp.sum(diff * diff, axis=1) / cnt
    Txyz = atom.mass_kg * var / KB
    sc = xp.sum(state["scatter_count"].reshape(n_groups, n_per), axis=1)

    n_alive_h = np.asarray(to_cpu(n_alive))
    Txyz_h = np.asarray(to_cpu(Txyz))
    sc_h = np.asarray(to_cpu(sc))
    T_mean = Txyz_h.mean(axis=1)
    # Match the single-point convention: T is NaN when fewer than 2 survive.
    dead = n_alive_h < 2
    Txyz_h[dead] = np.nan
    T_mean[dead] = np.nan
    return {
        "time_s": float(t_abs),
        "stage": stage,
        "survival": n_alive_h / n_per,
        "n_alive": n_alive_h,
        "T_K": T_mean,
        "Txyz_K": Txyz_h,
        "scatter_events_total": sc_h,
    }

def propagate_batched(
    xp, rng, state, per_atom, cfg, atom, geom1, geom2, axis1, axis2,
    n_groups, n_per, progress=None,
):
    dt = float(cfg["simulation"]["dt_s"])
    rec_dt = float(cfg["simulation"]["record_interval_s"])
    scat = bool(cfg["simulation"]["enable_scattering"])
    surv_cfg = cfg["simulation"]["survival"]
    check_every = int(surv_cfg.get("check_every_steps", 1))
    gravity_host = tuple(float(g) for g in cfg["geometry"]["gravity_m_s2"])

    history = []
    next_record = 0.0
    t_abs = 0.0
    history.append(record_diag_multi(xp, state, atom, t_abs, "L1", n_groups, n_per))

    total_steps = int(math.ceil(
        (float(cfg["geometry"]["l1"]["duration_s"])
         + float(cfg["geometry"]["handover"]["duration_s"])
         + float(cfg["geometry"]["l2"]["duration_s"])) / dt
    ))
    done_steps = 0

    # ---------- L1 ----------
    T1 = float(cfg["geometry"]["l1"]["duration_s"])
    nsteps = int(math.ceil(T1 / dt))
    for i in range(nsteps):
        t0 = min(i * dt, T1)
        t1 = min((i + 1) * dt, T1)
        h = t1 - t0
        rt0 = [runtime_l1(t0, cfg, axis1, geom1)]
        rt1 = [runtime_l1(t1, cfg, axis1, geom1)]
        fields, active = _verlet_step_multi(
            xp, rng, state, rt0, rt1, per_atom, atom, gravity_host, h, scat
        )
        if fields and (((i + 1) % check_every == 0) or i == nsteps - 1):
            elapsed_check = h * (check_every if (i + 1) % check_every == 0 else ((i + 1) % check_every))
            update_transport_survival(
                xp, state, fields[0], rt1[0].velocity, rt1[0].acceleration,
                atom, elapsed_check, surv_cfg
            )
        t_abs = t1
        if t_abs + 1e-15 >= next_record + rec_dt or i == nsteps - 1:
            history.append(record_diag_multi(xp, state, atom, t_abs, "L1", n_groups, n_per))
            next_record = t_abs
        done_steps += 1
        if progress and done_steps % 20000 == 0:
            progress(done_steps, total_steps)

    # ---------- handover ----------
    TH = float(cfg["geometry"]["handover"]["duration_s"])
    nsteps = int(math.ceil(TH / dt))
    ho_start = t_abs
    for i in range(nsteps):
        t0 = min(i * dt, TH)
        t1 = min((i + 1) * dt, TH)
        h = t1 - t0
        rt0 = runtime_handover(t0, cfg, axis1, axis2, geom1, geom2)
        rt1 = runtime_handover(t1, cfg, axis1, axis2, geom1, geom2)
        _verlet_step_multi(
            xp, rng, state, rt0, rt1, per_atom, atom, gravity_host, h, scat
        )
        t_abs = ho_start + t1
        if t_abs + 1e-15 >= next_record + rec_dt or i == nsteps - 1:
            history.append(record_diag_multi(xp, state, atom, t_abs, "handover", n_groups, n_per))
            next_record = t_abs
        done_steps += 1
        if progress and done_steps % 20000 == 0:
            progress(done_steps, total_steps)

    # End-of-handover L2 capture.
    rt2 = runtime_l2(0.0, cfg, axis2, geom2)
    field2 = field_quantities_multi(
        xp, state["r"], axis2, rt2.q,
        per_atom["c2P_base"] * rt2.power_fraction, per_atom, geom2,
    )
    handover_end_capture(xp, state, field2, 0.0, 0.0, atom, surv_cfg)
    history.append(record_diag_multi(xp, state, atom, t_abs, "handover_end", n_groups, n_per))

    # ---------- L2 ----------
    T2 = float(cfg["geometry"]["l2"]["duration_s"])
    nsteps = int(math.ceil(T2 / dt))
    l2_start = t_abs
    for i in range(nsteps):
        t0 = min(i * dt, T2)
        t1 = min((i + 1) * dt, T2)
        h = t1 - t0
        rt0 = [runtime_l2(t0, cfg, axis2, geom2)]
        rt1 = [runtime_l2(t1, cfg, axis2, geom2)]
        fields, active = _verlet_step_multi(
            xp, rng, state, rt0, rt1, per_atom, atom, gravity_host, h, scat
        )
        if fields and (((i + 1) % check_every == 0) or i == nsteps - 1):
            elapsed_check = h * (check_every if (i + 1) % check_every == 0 else ((i + 1) % check_every))
            update_transport_survival(
                xp, state, fields[0], rt1[0].velocity, rt1[0].acceleration,
                atom, elapsed_check, surv_cfg
            )
        t_abs = l2_start + t1
        if t_abs + 1e-15 >= next_record + rec_dt or i == nsteps - 1:
            history.append(record_diag_multi(xp, state, atom, t_abs, "L2", n_groups, n_per))
            next_record = t_abs
        done_steps += 1
        if progress and done_steps % 20000 == 0:
            progress(done_steps, total_steps)

    return history, state

def run_batched_scan(xp, base_cfg, detunings, powers, atom, axis1, axis2,
                     geom1, geom2, progress=None):
    """
    Propagate the full {detuning x power} grid in one batched ensemble.
    Returns (final_T_K, final_survival) with shape (n_detuning, n_power).
    """
    points = [(d, p) for d in detunings for p in powers]
    n_groups = len(points)
    n_per = int(base_cfg["initial"]["n_atoms"])

    seed = int(base_cfg["initial"]["seed"])
    rng = make_rng(xp, seed)
    state, per_atom = build_batched_state(
        xp, rng, base_cfg, atom, points, geom1
    )
    history, state = propagate_batched(
        xp, rng, state, per_atom, base_cfg, atom, geom1, geom2, axis1, axis2,
        n_groups, n_per, progress=progress,
    )
    last = history[-1]
    final_T = last["T_K"].reshape(len(detunings), len(powers))
    final_S = last["survival"].reshape(len(detunings), len(powers))
    return final_T, final_S
