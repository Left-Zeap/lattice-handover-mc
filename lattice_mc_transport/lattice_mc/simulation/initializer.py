from __future__ import annotations
import math
from ..constants import KB
from ..physics.lattice import field_quantities, LatticeGeometry

def orthonormal_basis(xp, axis):
    e = axis / xp.sqrt(xp.sum(axis * axis))
    helper = xp.asarray([0.0, 0.0, 1.0])
    if float(abs(e[2])) > 0.9:
        helper = xp.asarray([0.0, 1.0, 0.0])
    u = xp.cross(e, helper)
    u = u / xp.sqrt(xp.sum(u * u))
    v = xp.cross(e, u)
    return e, u, v

def sample_initial_ensemble(
    xp,
    rng,
    n_atoms,
    temperature_K,
    atom,
    response,
    q0,
    power_w,
    geom: LatticeGeometry,
):
    """
    Thermal harmonic sampling around one representative L1 lattice site.

    The lattice phase is periodic, so sampling a representative site is enough
    for translationally equivalent local dynamics; the Gaussian beam envelope
    is evaluated at q0.
    """
    e, u, v = orthonormal_basis(xp, xp.asarray(geom.axis, dtype=float))
    r0 = q0 * e

    probe = xp.tile(r0[None, :], (1, 1))
    f = field_quantities(xp, probe, e, q0, power_w, response, geom)
    Uax = float(f["Uax"][0])
    Upeak = float(f["Upeak"][0])
    w = float(f["w"][0])
    k = 2.0 * math.pi / response.wavelength_m

    omega_ax = math.sqrt(max(1e-300, 2.0 * Uax * k * k / atom.mass_kg))
    omega_r = math.sqrt(max(1e-300, 4.0 * Upeak / (atom.mass_kg * w * w)))

    sig_ax = math.sqrt(KB * temperature_K / (atom.mass_kg * omega_ax**2))
    sig_r = math.sqrt(KB * temperature_K / (atom.mass_kg * omega_r**2))
    sig_v = math.sqrt(KB * temperature_K / atom.mass_kg)

    da = rng.normal(0.0, sig_ax, size=n_atoms)
    du = rng.normal(0.0, sig_r, size=n_atoms)
    dv = rng.normal(0.0, sig_r, size=n_atoms)

    r = (
        r0[None, :]
        + da[:, None] * e[None, :]
        + du[:, None] * u[None, :]
        + dv[:, None] * v[None, :]
    )
    vel = rng.normal(0.0, sig_v, size=(n_atoms, 3))
    alive = xp.ones(n_atoms, dtype=bool)
    unbound_time = xp.zeros(n_atoms, dtype=float)
    scatter_count = xp.zeros(n_atoms, dtype=xp.int32)

    return {
        "r": r,
        "v": vel,
        "alive": alive,
        "unbound_time": unbound_time,
        "scatter_count": scatter_count,
    }
