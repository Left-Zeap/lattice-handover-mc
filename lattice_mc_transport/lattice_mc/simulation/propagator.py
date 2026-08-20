from __future__ import annotations
import math
from dataclasses import dataclass
from ..physics.lattice import field_quantities, LatticeGeometry
from ..physics.waveforms import trapezoid_kinematics, linear_ramp
from ..statistics.survival import update_transport_survival, handover_end_capture
from ..statistics.diagnostics import kinetic_temperature
from ..backend import to_cpu
from ..physics.lattice import _normalized_axis_cached

def _is_cupy(xp):
    return xp.__name__ == "cupy"

_KICK_KERNEL = None

def _get_kick_kernel():
    """Fused half-kick (+ optional drift) with gravity and -gradV/m."""
    global _KICK_KERNEL
    if _KICK_KERNEL is None:
        import cupy as cp
        _KICK_KERNEL = cp.ElementwiseKernel(
            "float64 vx, float64 vy, float64 vz, "
            "float64 rx, float64 ry, float64 rz, "
            "float64 gvx, float64 gvy, float64 gvz, "
            "float64 gx, float64 gy, float64 gz, "
            "float64 h_kick, float64 h_drift, float64 inv_m, bool use_field",
            "float64 nvx, float64 nvy, float64 nvz, "
            "float64 nrx, float64 nry, float64 nrz",
            """
            double ax = gx - (use_field ? gvx * inv_m : 0.0);
            double ay = gy - (use_field ? gvy * inv_m : 0.0);
            double az = gz - (use_field ? gvz * inv_m : 0.0);
            nvx = vx + h_kick * ax;
            nvy = vy + h_kick * ay;
            nvz = vz + h_kick * az;
            nrx = rx + h_drift * nvx;
            nry = ry + h_drift * nvy;
            nrz = rz + h_drift * nvz;
            """,
            "kick_drift_fused",
        )
    return _KICK_KERNEL

def _kick(xp, state, gradV, gravity, gravity_host, mass, h_kick, h_drift):
    """v += h_kick * (g - gradV/m); then r += h_drift * v (if h_drift != 0)."""
    if _is_cupy(xp):
        v = state["v"]
        r = state["r"]
        if gradV is None:
            gvx, gvy, gvz, use = r[:, 0], r[:, 1], r[:, 2], False
        else:
            gvx, gvy, gvz, use = gradV[:, 0], gradV[:, 1], gradV[:, 2], True
        kern = _get_kick_kernel()
        kern(
            v[:, 0], v[:, 1], v[:, 2],
            r[:, 0], r[:, 1], r[:, 2],
            gvx, gvy, gvz,
            gravity_host[0], gravity_host[1], gravity_host[2],
            float(h_kick), float(h_drift), 1.0 / mass, use,
            v[:, 0], v[:, 1], v[:, 2],
            r[:, 0], r[:, 1], r[:, 2],
        )
        return
    a = gravity[None, :] if gradV is None else gravity[None, :] - gradV / mass
    state["v"] += h_kick * a
    if h_drift:
        state["r"] += h_drift * state["v"]

@dataclass
class RuntimeLattice:
    name: str
    axis: object
    geom: LatticeGeometry
    base_power_w: float
    q: float
    velocity: float
    acceleration: float
    power_fraction: float

def _random_unit_vectors(xp, rng, n):
    # Isotropic emission.
    z = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * math.pi, size=n)
    rxy = xp.sqrt(xp.maximum(0.0, 1.0 - z*z))
    return xp.stack((rxy*xp.cos(phi), rxy*xp.sin(phi), z), axis=1)

_SCATTER_KERNEL = None

def _get_scatter_kernel():
    """
    Fully masked quantum-jump scattering in one kernel.

    The generic path compacts event indices with nonzero()/to_cpu() every
    step, which forces a host sync and serializes the GPU pipeline. Here all
    random numbers are drawn up front (one (5, n) draw) and non-event
    particles are handled by the mask inside the kernel -- zero syncs.
    """
    global _SCATTER_KERNEL
    if _SCATTER_KERNEL is None:
        import cupy as cp
        _SCATTER_KERNEL = cp.ElementwiseKernel(
            "bool alive, float64 g1, float64 g2, "
            "float64 u1, float64 u2, float64 u3, float64 u4, float64 u5, "
            "float64 vx, float64 vy, float64 vz, int32 sc, "
            "float64 dt, float64 k, float64 hbm, float64 two_pi, "
            "float64 e1x, float64 e1y, float64 e1z, float64 pf1, "
            "float64 e2x, float64 e2y, float64 e2z, float64 pf2, "
            "bool two_lattices",
            "float64 nvx, float64 nvy, float64 nvz, int32 nsc",
            """
            nvx = vx; nvy = vy; nvz = vz; nsc = sc;
            double total = fmax(0.0, g1) + (two_lattices ? fmax(0.0, g2) : 0.0);
            double p = 1.0 - exp(-total * dt);
            if (alive && u1 < p) {
                bool second = two_lattices && (u2 * total >= fmax(0.0, g1));
                double ax = second ? e2x : e1x;
                double ay = second ? e2y : e1y;
                double az = second ? e2z : e1z;
                double pf = second ? pf2 : pf1;
                double sgn = (u3 < pf) ? 1.0 : -1.0;
                double z = 2.0 * u4 - 1.0;
                double rxy = sqrt(fmax(0.0, 1.0 - z * z));
                double phi = two_pi * u5;
                double dpx = k * (sgn * ax - rxy * cos(phi));
                double dpy = k * (sgn * ay - rxy * sin(phi));
                double dpz = k * (sgn * az - z);
                nvx = vx + hbm * dpx;
                nvy = vy + hbm * dpy;
                nvz = vz + hbm * dpz;
                nsc = sc + 1;
            }
            """,
            "scatter_fused",
        )
    return _SCATTER_KERNEL

def _apply_scattering_gpu(xp, rng, state, fields, runtimes, response, atom, dt):
    n = state["alive"].shape[0]
    # One RNG launch; rows serve as the five independent uniforms per atom.
    u = rng.random((5, n))
    k = 2.0 * math.pi / response.wavelength_m
    two = len(fields) == 2
    e1, e1h = _normalized_axis_cached(xp, runtimes[0].geom, runtimes[0].axis)
    pf1 = 1.0 / (1.0 + float(runtimes[0].geom.retro_ratio))
    if two:
        e2, e2h = _normalized_axis_cached(xp, runtimes[1].geom, runtimes[1].axis)
        pf2 = 1.0 / (1.0 + float(runtimes[1].geom.retro_ratio))
        g2 = fields[1]["gamma_sc"]
    else:
        e2h = e1h
        pf2 = pf1
        g2 = fields[0]["gamma_sc"]
    from ..constants import HBAR
    v = state["v"]
    sc = state["scatter_count"]
    kern = _get_scatter_kernel()
    kern(
        state["alive"], fields[0]["gamma_sc"], g2,
        u[0], u[1], u[2], u[3], u[4],
        v[:, 0], v[:, 1], v[:, 2], sc,
        float(dt), k, HBAR / atom.mass_kg, 2.0 * math.pi,
        e1h[0], e1h[1], e1h[2], pf1,
        e2h[0], e2h[1], e2h[2], pf2,
        two,
        v[:, 0], v[:, 1], v[:, 2], sc,
    )

_SCATTER_KERNEL_MULTI = None

def _get_scatter_kernel_multi():
    """Batched variant: wave number k is a per-atom array."""
    global _SCATTER_KERNEL_MULTI
    if _SCATTER_KERNEL_MULTI is None:
        import cupy as cp
        _SCATTER_KERNEL_MULTI = cp.ElementwiseKernel(
            "bool alive, float64 g1, float64 g2, "
            "float64 u1, float64 u2, float64 u3, float64 u4, float64 u5, "
            "float64 k, float64 vx, float64 vy, float64 vz, int32 sc, "
            "float64 dt, float64 hbm, float64 two_pi, "
            "float64 e1x, float64 e1y, float64 e1z, float64 pf1, "
            "float64 e2x, float64 e2y, float64 e2z, float64 pf2, "
            "bool two_lattices",
            "float64 nvx, float64 nvy, float64 nvz, int32 nsc",
            """
            nvx = vx; nvy = vy; nvz = vz; nsc = sc;
            double total = fmax(0.0, g1) + (two_lattices ? fmax(0.0, g2) : 0.0);
            double p = 1.0 - exp(-total * dt);
            if (alive && u1 < p) {
                bool second = two_lattices && (u2 * total >= fmax(0.0, g1));
                double ax = second ? e2x : e1x;
                double ay = second ? e2y : e1y;
                double az = second ? e2z : e1z;
                double pf = second ? pf2 : pf1;
                double sgn = (u3 < pf) ? 1.0 : -1.0;
                double z = 2.0 * u4 - 1.0;
                double rxy = sqrt(fmax(0.0, 1.0 - z * z));
                double phi = two_pi * u5;
                double dpx = k * (sgn * ax - rxy * cos(phi));
                double dpy = k * (sgn * ay - rxy * sin(phi));
                double dpz = k * (sgn * az - z);
                nvx = vx + hbm * dpx;
                nvy = vy + hbm * dpy;
                nvz = vz + hbm * dpz;
                nsc = sc + 1;
            }
            """,
            "scatter_fused_multi",
        )
    return _SCATTER_KERNEL_MULTI

def _apply_scattering_gpu_multi(xp, rng, state, fields, runtimes, per_atom, atom, dt):
    n = state["alive"].shape[0]
    u = rng.random((5, n))
    two = len(fields) == 2
    _, e1h = _normalized_axis_cached(xp, runtimes[0].geom, runtimes[0].axis)
    pf1 = 1.0 / (1.0 + float(runtimes[0].geom.retro_ratio))
    if two:
        _, e2h = _normalized_axis_cached(xp, runtimes[1].geom, runtimes[1].axis)
        pf2 = 1.0 / (1.0 + float(runtimes[1].geom.retro_ratio))
        g2 = fields[1]["gamma_sc"]
    else:
        e2h = e1h
        pf2 = pf1
        g2 = fields[0]["gamma_sc"]
    from ..constants import HBAR
    v = state["v"]
    sc = state["scatter_count"]
    kern = _get_scatter_kernel_multi()
    kern(
        state["alive"], fields[0]["gamma_sc"], g2,
        u[0], u[1], u[2], u[3], u[4],
        per_atom["k"],
        v[:, 0], v[:, 1], v[:, 2], sc,
        float(dt), HBAR / atom.mass_kg, 2.0 * math.pi,
        e1h[0], e1h[1], e1h[2], pf1,
        e2h[0], e2h[1], e2h[2], pf2,
        two,
        v[:, 0], v[:, 1], v[:, 2], sc,
    )

def _apply_scattering(
    xp, rng, state, fields, runtimes, response, atom, dt
):
    """
    Quantum-jump-style recoil model.

    Event rate is local total off-resonant scattering rate.
    On an event:
      1) choose L1/L2 proportional to that lattice's local rate;
      2) choose absorption from forward/retro beam weighted by powers 1:R;
      3) emit one photon isotropically.

    This is intentionally a first-order total-scattering model; it does not yet
    resolve Rayleigh/Raman or dipole radiation patterns.
    """
    alive = state["alive"]
    if not fields:
        return
    if _is_cupy(xp):
        _apply_scattering_gpu(
            xp, rng, state, fields, runtimes, response, atom, dt
        )
        return

    rates = xp.stack([xp.maximum(0.0, f["gamma_sc"]) for f in fields], axis=1)
    total = xp.sum(rates, axis=1)
    p = 1.0 - xp.exp(-total * dt)
    event = alive & (rng.random(total.shape[0]) < p)
    n_event = int(to_cpu(xp.sum(event)))
    if n_event == 0:
        return

    idx = xp.where(event)[0]
    rates_e = rates[idx]
    totals_e = xp.sum(rates_e, axis=1)

    if len(fields) == 1:
        chosen = xp.zeros(n_event, dtype=xp.int32)
    else:
        # General categorical choice for the current two-lattice case.
        u = rng.random(n_event) * totals_e
        c0 = rates_e[:, 0]
        chosen = (u >= c0).astype(xp.int32)

    k = 2.0 * math.pi / response.wavelength_m
    n_emit = _random_unit_vectors(xp, rng, n_event)
    dp = -k * n_emit  # in units of hbar

    for j, (field, rt) in enumerate(zip(fields, runtimes)):
        mask_local = chosen == j
        if not bool(mask_local.any()):
            continue
        ids = idx[mask_local]
        R = float(rt.geom.retro_ratio)
        p_forward = 1.0 / (1.0 + R)
        sign = xp.where(rng.random(ids.shape[0]) < p_forward, 1.0, -1.0)
        dp[mask_local] += sign[:, None] * k * field["axis"][None, :]

    from ..constants import HBAR
    state["v"][idx] += (HBAR / atom.mass_kg) * dp
    state["scatter_count"][idx] += 1

def _forces_and_fields(xp, r, runtimes, response, atom):
    gradV = None
    fields = []
    active_rt = []

    for rt in runtimes:
        if rt.power_fraction <= 0.0:
            continue
        field = field_quantities(
            xp=xp,
            r=r,
            axis=rt.axis,
            q=rt.q,
            power_w=rt.base_power_w * rt.power_fraction,
            response=response,
            geom=rt.geom,
        )
        gradV = field["gradV"] if gradV is None else gradV + field["gradV"]
        fields.append(field)
        active_rt.append(rt)
    return gradV, fields, active_rt

def velocity_verlet_step(
    xp, rng, state, runtimes_now, runtimes_next,
    response, atom, gravity, gravity_host, dt, enable_scattering
):
    gV0, fields0, active0 = _forces_and_fields(
        xp, state["r"], runtimes_now, response, atom
    )
    _kick(
        xp, state, gV0, gravity, gravity_host, atom.mass_kg,
        h_kick=0.5 * dt, h_drift=dt,
    )

    gV1, fields1, active1 = _forces_and_fields(
        xp, state["r"], runtimes_next, response, atom
    )
    _kick(
        xp, state, gV1, gravity, gravity_host, atom.mass_kg,
        h_kick=0.5 * dt, h_drift=0.0,
    )

    if enable_scattering:
        _apply_scattering(
            xp, rng, state, fields1, active1, response, atom, dt
        )
    return fields1, active1

def runtime_l1(t, cfg, axis, geom):
    g = cfg["geometry"]["l1"]
    q, v, a = trapezoid_kinematics(
        t, g["distance_m"], g["duration_s"], g["acceleration_m_s2"],
        start=-g["distance_m"]
    )
    return RuntimeLattice("L1", axis, geom, g["power_w"], q, v, a, 1.0)

def runtime_handover(t, cfg, axis1, axis2, geom1, geom2):
    ho = cfg["geometry"]["handover"]
    tau = ho["duration_s"]
    f2 = linear_ramp(t, tau, 0.0, 1.0)
    f1 = 1.0 - f2
    g1 = cfg["geometry"]["l1"]
    g2 = cfg["geometry"]["l2"]
    return [
        RuntimeLattice("L1", axis1, geom1, g1["power_w"], 0.0, 0.0, 0.0, f1),
        RuntimeLattice("L2", axis2, geom2, g2["power_w"], 0.0, 0.0, 0.0, f2),
    ]

def runtime_l2(t, cfg, axis, geom):
    g = cfg["geometry"]["l2"]
    q, v, a = trapezoid_kinematics(
        t, g["distance_m"], g["duration_s"], g["acceleration_m_s2"],
        start=0.0
    )
    return RuntimeLattice("L2", axis, geom, g["power_w"], q, v, a, 1.0)

def record_diag(xp, state, atom, t_abs, stage, n0):
    d = kinetic_temperature(xp, state["v"], state["alive"], atom.mass_kg)
    total_sc = int(to_cpu(xp.sum(state["scatter_count"])))
    return {
        "time_s": float(t_abs),
        "stage": stage,
        "survival": d["n_alive"] / n0,
        "n_alive": d["n_alive"],
        "T_K": d["T_K"],
        "Tx_K": d["Tx_K"],
        "Ty_K": d["Ty_K"],
        "Tz_K": d["Tz_K"],
        "scatter_events_total": total_sc,
    }

def propagate_experiment(
    xp, rng, state, cfg, atom, response, geom1, geom2, axis1, axis2,
    progress=None,
):
    dt = float(cfg["simulation"]["dt_s"])
    rec_dt = float(cfg["simulation"]["record_interval_s"])
    scat = bool(cfg["simulation"]["enable_scattering"])
    surv_cfg = cfg["simulation"]["survival"]
    check_every = int(surv_cfg.get("check_every_steps", 1))
    gravity = xp.asarray(cfg["geometry"]["gravity_m_s2"], dtype=float)
    gravity_host = tuple(float(g) for g in cfg["geometry"]["gravity_m_s2"])
    n0 = state["r"].shape[0]

    history = []
    next_record = 0.0
    t_abs = 0.0
    history.append(record_diag(xp, state, atom, t_abs, "L1", n0))

    # Optional progress reporting (UI progress bars); purely additive.
    _prog_every = 20000
    _done = 0
    _total = int(math.ceil((
        float(cfg["geometry"]["l1"]["duration_s"])
        + float(cfg["geometry"]["handover"]["duration_s"])
        + float(cfg["geometry"]["l2"]["duration_s"])
    ) / dt))

    def _report():
        if progress and _done % _prog_every == 0:
            progress(_done, _total)

    # ---------- L1 ----------
    T1 = float(cfg["geometry"]["l1"]["duration_s"])
    nsteps = int(math.ceil(T1 / dt))
    for i in range(nsteps):
        t0 = min(i * dt, T1)
        t1 = min((i + 1) * dt, T1)
        h = t1 - t0
        rt0 = [runtime_l1(t0, cfg, axis1, geom1)]
        rt1 = [runtime_l1(t1, cfg, axis1, geom1)]
        fields, active = velocity_verlet_step(
            xp, rng, state, rt0, rt1, response, atom, gravity, gravity_host, h, scat
        )

        if fields and (((i + 1) % check_every == 0) or i == nsteps - 1):
            elapsed_check = h * (check_every if (i + 1) % check_every == 0 else ((i + 1) % check_every))
            update_transport_survival(
                xp, state, fields[0], rt1[0].velocity, rt1[0].acceleration,
                atom, elapsed_check, surv_cfg
            )

        t_abs = t1
        if t_abs + 1e-15 >= next_record + rec_dt or i == nsteps - 1:
            history.append(record_diag(xp, state, atom, t_abs, "L1", n0))
            next_record = t_abs
        _done += 1
        _report()

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
        velocity_verlet_step(
            xp, rng, state, rt0, rt1, response, atom, gravity, gravity_host, h, scat
        )
        t_abs = ho_start + t1
        if t_abs + 1e-15 >= next_record + rec_dt or i == nsteps - 1:
            history.append(record_diag(xp, state, atom, t_abs, "handover", n0))
            next_record = t_abs
        _done += 1
        _report()

    # End-of-handover L2 capture.
    rt2 = runtime_l2(0.0, cfg, axis2, geom2)
    field2 = field_quantities(
        xp, state["r"], axis2, rt2.q, rt2.base_power_w,
        response, geom2
    )
    handover_end_capture(
        xp, state, field2, 0.0, 0.0, atom, surv_cfg
    )
    history.append(record_diag(xp, state, atom, t_abs, "handover_end", n0))

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
        fields, active = velocity_verlet_step(
            xp, rng, state, rt0, rt1, response, atom, gravity, gravity_host, h, scat
        )

        if fields and (((i + 1) % check_every == 0) or i == nsteps - 1):
            elapsed_check = h * (check_every if (i + 1) % check_every == 0 else ((i + 1) % check_every))
            update_transport_survival(
                xp, state, fields[0], rt1[0].velocity, rt1[0].acceleration,
                atom, elapsed_check, surv_cfg
            )

        t_abs = l2_start + t1
        if t_abs + 1e-15 >= next_record + rec_dt or i == nsteps - 1:
            history.append(record_diag(xp, state, atom, t_abs, "L2", n0))
            next_record = t_abs
        _done += 1
        _report()

    if progress:
        progress(_total, _total)

    return history, state
