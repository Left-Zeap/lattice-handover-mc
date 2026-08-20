from __future__ import annotations
from ..physics.lattice import tilted_barrier

def update_transport_survival(
    xp,
    state,
    field,
    reference_velocity,
    reference_acceleration,
    atom,
    dt,
    cfg,
):
    """
    Loose microscopic local-binding criterion.

    A particle is not removed at the first threshold crossing. It must remain
    above energy_factor * U_eff for loss_grace_s. This reduces false losses
    caused by transient non-adiabatic excursions.

    The criterion is local and deliberately permissive; full trajectories still
    determine whether the atom actually departs.
    """
    alive = state["alive"]
    # On GPU this early-exit forces a host sync every check; the masked ops
    # below are already correct (and cheap) when nothing is alive, so only
    # short-circuit on CPU where the check is free.
    if xp.__name__ != "cupy" and not bool(alive.any()):
        return

    e = field["axis"]
    vref = reference_velocity * e
    vrel = state["v"] - vref[None, :]
    K = 0.5 * atom.mass_kg * xp.sum(vrel * vrel, axis=1)

    # excitation relative to local antinode + beam center
    Eexc = K + (field["V"] - field["Vmin"])

    Ueff = tilted_barrier(
        xp, field["Uax"], field["k"], atom.mass_kg, reference_acceleration
    )
    threshold = float(cfg["energy_factor"]) * Ueff
    unbound = Eexc > threshold

    state["unbound_time"] = xp.where(
        alive & unbound,
        state["unbound_time"] + dt,
        xp.where(alive, 0.0, state["unbound_time"]),
    )
    lost_energy = state["unbound_time"] >= float(cfg["loss_grace_s"])

    hard_r2 = (float(cfg["hard_radial_waists"]) * field["w"])**2
    lost_domain = field["rho2"] > hard_r2

    state["alive"] = alive & ~(lost_energy | lost_domain)

def handover_end_capture(
    xp,
    state,
    field_l2,
    l2_velocity,
    l2_acceleration,
    atom,
    cfg,
):
    """
    Handover is propagated with the full V1+V2 time-dependent force.
    Only after the ramp ends do we decide whether the particle belongs to L2.
    """
    e = field_l2["axis"]
    vrel = state["v"] - l2_velocity * e[None, :]
    K = 0.5 * atom.mass_kg * xp.sum(vrel * vrel, axis=1)
    Eexc = K + (field_l2["V"] - field_l2["Vmin"])
    Ueff = tilted_barrier(
        xp, field_l2["Uax"], field_l2["k"], atom.mass_kg, l2_acceleration
    )
    threshold = float(cfg["energy_factor"]) * Ueff
    hard_r2 = (float(cfg["hard_radial_waists"]) * field_l2["w"])**2
    captured = (Eexc <= threshold) & (field_l2["rho2"] <= hard_r2)
    state["alive"] &= captured
    state["unbound_time"][:] = 0.0
