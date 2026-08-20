from __future__ import annotations
from dataclasses import dataclass
from ..constants import C, HBAR, PI

@dataclass(frozen=True)
class LaserResponse:
    omega_laser: float
    wavelength_m: float
    shift_per_intensity: float       # J / (W m^-2)
    scatter_per_intensity: float     # s^-1 / (W m^-2)

def response_from_d1_red_detuning(atom, detuning_GHz: float) -> LaserResponse:
    """
    Scalar D1/D2 far-detuned response used by the project manual.

    d1_red_detuning_GHz > 0 means omega_L = omega_D1 - 2*pi*delta.
    Counter-rotating terms are retained.
    D1:D2 scalar weights are 1:2.
    """
    delta = 2.0 * PI * detuning_GHz * 1e9
    omega_l = atom.d1.omega - delta
    if omega_l <= 0:
        raise ValueError("laser frequency became non-positive")

    u_coeff = 0.0
    g_coeff = 0.0
    for weight, line in ((1.0, atom.d1), (2.0, atom.d2)):
        dj = omega_l - line.omega
        u_coeff += (
            weight * line.gamma_rad_s / line.omega**3
            * (1.0 / dj - 1.0 / (omega_l + line.omega))
        )
        g_coeff += (
            weight * line.gamma_rad_s**2 / line.omega**3
            * (1.0 / dj**2 + 1.0 / (omega_l + line.omega)**2)
        )

    pref_u = PI * C**2 / 2.0
    pref_g = PI * C**2 / (2.0 * HBAR)
    return LaserResponse(
        omega_laser=omega_l,
        wavelength_m=2.0 * PI * C / omega_l,
        shift_per_intensity=pref_u * u_coeff,
        scatter_per_intensity=pref_g * g_coeff,
    )
