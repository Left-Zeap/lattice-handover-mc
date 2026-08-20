from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass
class LatticeGeometry:
    axis: object
    s_points: tuple[float, float]
    waist_points_m: tuple[float, float]
    retro_ratio: float
    phase_offset_rad: float = 0.0

def unit_vector(xp, vec):
    v = xp.asarray(vec, dtype=float)
    return v / xp.sqrt(xp.sum(v * v))

def _normalized_axis_cached(xp, geom, axis):
    """
    Normalize the lattice axis once per (geometry, axis array) pair and cache
    both the backend array and its host-side components. Repeated per-step
    calls would otherwise cost several tiny kernels plus a host sync each.
    """
    cache = getattr(geom, "_axis_unit_cache", None)
    if cache is None or cache[0] is not axis or cache[1] is not xp:
        e = unit_vector(xp, axis)
        from ..backend import to_cpu
        host = tuple(float(v) for v in to_cpu(e))
        cache = (axis, xp, e, host)
        geom._axis_unit_cache = cache
    return cache[2], cache[3]

_FIELD_KERNEL = None

def _get_field_kernel():
    """
    Single fused CUDA kernel for the whole field_quantities computation.

    The generic path issues ~30 small array ops per call; on GPU that is pure
    launch overhead (measured ~2.6 ms for 4000 atoms while the actual math is
    microseconds). One ElementwiseKernel collapses it to a single launch.
    """
    global _FIELD_KERNEL
    if _FIELD_KERNEL is None:
        import cupy as cp
        _FIELD_KERNEL = cp.ElementwiseKernel(
            "float64 x, float64 y, float64 z, "
            "float64 ex, float64 ey, float64 ez, float64 q, "
            "float64 c2P, float64 s0, float64 inv_ds, float64 w0, float64 w1, "
            "float64 slope, float64 one_plus_R, float64 two_sqrtR, "
            "float64 four_sqrtR, float64 one_plus_sqrtR_sq, "
            "float64 k, float64 phase, float64 shift, float64 scatter, "
            "float64 vmin_sign",
            "float64 V, float64 gvx, float64 gvy, float64 gvz, "
            "float64 gamma, float64 w, float64 rho2, float64 s, "
            "float64 Uax, float64 Upeak, float64 Vmin",
            """
            double sv = x*ex + y*ey + z*ez;
            double rpx = x - sv*ex;
            double rpy = y - sv*ey;
            double rpz = z - sv*ez;
            double rho2v = rpx*rpx + rpy*rpy + rpz*rpz;
            double f = (sv - s0) * inv_ds;
            double fc = fmin(1.0, fmax(0.0, f));
            double wv = w0 + fc * (w1 - w0);
            double dwds = (f >= 0.0 && f <= 1.0) ? slope : 0.0;
            double w2 = wv * wv;
            double If0 = c2P / w2;
            double If = If0 * exp(-2.0 * rho2v / w2);
            double theta = k * (sv - q) + phase;
            double B = one_plus_R + two_sqrtR * cos(2.0 * theta);
            double dlnIf = dwds * (-2.0 / wv + 4.0 * rho2v / (w2 * wv));
            double dB = -four_sqrtR * k * sin(2.0 * theta);
            double gas = If * (dlnIf * B + dB);
            double gfac = If * B * (-4.0 / w2);
            double I = If * B;
            V = shift * I;
            gvx = shift * (gas * ex + gfac * rpx);
            gvy = shift * (gas * ey + gfac * rpy);
            gvz = shift * (gas * ez + gfac * rpz);
            gamma = scatter * I;
            double absC = fabs(shift);
            Uax = absC * If0 * four_sqrtR;
            Upeak = absC * If0 * one_plus_sqrtR_sq;
            Vmin = vmin_sign * Upeak;
            w = wv;
            rho2 = rho2v;
            s = sv;
            """,
            "lattice_field_fused",
        )
    return _FIELD_KERNEL

def _field_quantities_fused(xp, r, e, e_host, q, power_w, response, geom):
    kern = _get_field_kernel()
    n = r.shape[0]
    # One backing allocation for all outputs; slices are zero-cost views.
    buf = xp.empty(11 * n, dtype=float)
    V = buf[0 * n:1 * n]
    gamma = buf[1 * n:2 * n]
    w = buf[2 * n:3 * n]
    rho2 = buf[3 * n:4 * n]
    s = buf[4 * n:5 * n]
    Uax = buf[5 * n:6 * n]
    Upeak = buf[6 * n:7 * n]
    Vmin = buf[7 * n:8 * n]
    gradV = buf[8 * n:11 * n].reshape(n, 3)

    R = float(geom.retro_ratio)
    sr = math.sqrt(R)
    k = 2.0 * math.pi / response.wavelength_m
    s0, s1 = (float(v) for v in geom.s_points)
    w0, w1 = (float(v) for v in geom.waist_points_m)
    shift = float(response.shift_per_intensity)
    kern(
        r[:, 0], r[:, 1], r[:, 2],
        e_host[0], e_host[1], e_host[2],
        float(q), 2.0 * float(power_w) / math.pi,
        s0, 1.0 / (s1 - s0), w0, w1, (w1 - w0) / (s1 - s0),
        1.0 + R, 2.0 * sr, 4.0 * sr, (1.0 + sr) ** 2,
        k, float(geom.phase_offset_rad),
        shift, float(response.scatter_per_intensity),
        -1.0 if shift < 0.0 else 0.0,
        V, gradV[:, 0], gradV[:, 1], gradV[:, 2],
        gamma, w, rho2, s, Uax, Upeak, Vmin,
    )
    return {
        "V": V,
        "gradV": gradV,
        "gamma_sc": gamma,
        "w": w,
        "rho2": rho2,
        "s": s,
        "Uax": Uax,
        "Upeak": Upeak,
        "Vmin": Vmin,
        "axis": e,
        "k": k,
    }

def waist_and_slope(xp, s, s_points, waist_points_m):
    """
    Piecewise-linear waist model between two source-supported endpoint values.
    Outside the interval the waist is clamped and dw/ds = 0.

    This is deliberately configurable: replace with measured/Gaussian waist(z)
    when the exact optical geometry is available.
    """
    s0, s1 = s_points
    w0, w1 = waist_points_m
    slope = (w1 - w0) / (s1 - s0)
    f = (s - s0) / (s1 - s0)
    f_clip = xp.clip(f, 0.0, 1.0)
    w = w0 + f_clip * (w1 - w0)
    active = (f >= 0.0) & (f <= 1.0)
    dwds = xp.where(active, slope, 0.0)
    return w, dwds

_FIELD_KERNEL_MULTI = None

def _get_field_kernel_multi():
    """
    Batched variant of the fused field kernel: the detuning/power-dependent
    quantities (k, Stark shift, scattering coefficient, 2P/pi) are per-atom
    arrays instead of scalars, so many parameter points can be propagated in
    one launch. Used by the batched scan.
    """
    global _FIELD_KERNEL_MULTI
    if _FIELD_KERNEL_MULTI is None:
        import cupy as cp
        _FIELD_KERNEL_MULTI = cp.ElementwiseKernel(
            "float64 x, float64 y, float64 z, "
            "float64 c2P, float64 k, float64 shift, float64 scatter, "
            "float64 vmin_sign, "
            "float64 ex, float64 ey, float64 ez, float64 q, "
            "float64 s0, float64 inv_ds, float64 w0, float64 w1, "
            "float64 slope, float64 one_plus_R, float64 two_sqrtR, "
            "float64 four_sqrtR, float64 one_plus_sqrtR_sq, float64 phase",
            "float64 V, float64 gvx, float64 gvy, float64 gvz, "
            "float64 gamma, float64 w, float64 rho2, float64 s, "
            "float64 Uax, float64 Upeak, float64 Vmin",
            """
            double sv = x*ex + y*ey + z*ez;
            double rpx = x - sv*ex;
            double rpy = y - sv*ey;
            double rpz = z - sv*ez;
            double rho2v = rpx*rpx + rpy*rpy + rpz*rpz;
            double f = (sv - s0) * inv_ds;
            double fc = fmin(1.0, fmax(0.0, f));
            double wv = w0 + fc * (w1 - w0);
            double dwds = (f >= 0.0 && f <= 1.0) ? slope : 0.0;
            double w2 = wv * wv;
            double If0 = c2P / w2;
            double If = If0 * exp(-2.0 * rho2v / w2);
            double theta = k * (sv - q) + phase;
            double B = one_plus_R + two_sqrtR * cos(2.0 * theta);
            double dlnIf = dwds * (-2.0 / wv + 4.0 * rho2v / (w2 * wv));
            double dB = -four_sqrtR * k * sin(2.0 * theta);
            double gas = If * (dlnIf * B + dB);
            double gfac = If * B * (-4.0 / w2);
            double I = If * B;
            V = shift * I;
            gvx = shift * (gas * ex + gfac * rpx);
            gvy = shift * (gas * ey + gfac * rpy);
            gvz = shift * (gas * ez + gfac * rpz);
            gamma = scatter * I;
            double absC = fabs(shift);
            Uax = absC * If0 * four_sqrtR;
            Upeak = absC * If0 * one_plus_sqrtR_sq;
            Vmin = vmin_sign * Upeak;
            w = wv;
            rho2 = rho2v;
            s = sv;
            """,
            "lattice_field_fused_multi",
        )
    return _FIELD_KERNEL_MULTI

def field_quantities_multi(xp, r, axis, q, c2P_arr, per_atom, geom):
    """
    Fused field evaluation with per-atom laser parameters.

    per_atom must provide cupy arrays: k, shift, scatter, vmin_sign.
    c2P_arr is the per-atom 2*power/pi for this lattice at this time.
    """
    kern = _get_field_kernel_multi()
    n = r.shape[0]
    buf = xp.empty(11 * n, dtype=float)
    V = buf[0 * n:1 * n]
    gamma = buf[1 * n:2 * n]
    w = buf[2 * n:3 * n]
    rho2 = buf[3 * n:4 * n]
    s = buf[4 * n:5 * n]
    Uax = buf[5 * n:6 * n]
    Upeak = buf[6 * n:7 * n]
    Vmin = buf[7 * n:8 * n]
    gradV = buf[8 * n:11 * n].reshape(n, 3)

    e, e_host = _normalized_axis_cached(xp, geom, axis)
    R = float(geom.retro_ratio)
    sr = math.sqrt(R)
    s0, s1 = (float(v) for v in geom.s_points)
    w0, w1 = (float(v) for v in geom.waist_points_m)
    kern(
        r[:, 0], r[:, 1], r[:, 2],
        c2P_arr, per_atom["k"], per_atom["shift"], per_atom["scatter"],
        per_atom["vmin_sign"],
        e_host[0], e_host[1], e_host[2], float(q),
        s0, 1.0 / (s1 - s0), w0, w1, (w1 - w0) / (s1 - s0),
        1.0 + R, 2.0 * sr, 4.0 * sr, (1.0 + sr) ** 2,
        float(geom.phase_offset_rad),
        V, gradV[:, 0], gradV[:, 1], gradV[:, 2],
        gamma, w, rho2, s, Uax, Upeak, Vmin,
    )
    return {
        "V": V,
        "gradV": gradV,
        "gamma_sc": gamma,
        "w": w,
        "rho2": rho2,
        "s": s,
        "Uax": Uax,
        "Upeak": Upeak,
        "Vmin": Vmin,
        "axis": e,
        "k": per_atom["k"],
    }

def field_quantities(
    xp,
    r,
    axis,
    q,
    power_w,
    response,
    geom: LatticeGeometry,
):
    """
    Exact unequal-retro standing-wave intensity plus analytic gradient.

    I_f = 2P/(pi w^2) exp(-2 rho^2/w^2)
    I = I_f [(1+R)+2 sqrt(R) cos(2 theta)]
    theta = k(s-q)+phase_offset

    Returns dict with total intensity, gradI, local waist, local axial
    barrier U_ax, local antinode depth U_peak and phase coordinate.
    """
    e, e_host = _normalized_axis_cached(xp, geom, axis)
    if xp.__name__ == "cupy":
        return _field_quantities_fused(
            xp, r, e, e_host, q, power_w, response, geom
        )
    s = xp.sum(r * e[None, :], axis=1)
    r_perp = r - s[:, None] * e[None, :]
    rho2 = xp.sum(r_perp * r_perp, axis=1)

    w, dwds = waist_and_slope(xp, s, geom.s_points, geom.waist_points_m)
    If = 2.0 * power_w / (math.pi * w**2) * xp.exp(-2.0 * rho2 / w**2)

    R = float(geom.retro_ratio)
    sr = math.sqrt(R)
    k = 2.0 * math.pi / response.wavelength_m
    theta = k * (s - q) + float(geom.phase_offset_rad)
    B = (1.0 + R) + 2.0 * sr * xp.cos(2.0 * theta)
    intensity = If * B

    # d ln(If) / ds due to the configurable axial waist profile.
    dlnIf_ds = dwds * (-2.0 / w + 4.0 * rho2 / w**3)
    dB_ds = -4.0 * sr * k * xp.sin(2.0 * theta)

    grad_ax_scalar = If * (dlnIf_ds * B + dB_ds)
    grad_rad = If[:, None] * B[:, None] * (-4.0 * r_perp / w[:, None]**2)
    gradI = grad_ax_scalar[:, None] * e[None, :] + grad_rad

    # Conservative potential U = C_U I.
    V = response.shift_per_intensity * intensity
    gradV = response.shift_per_intensity * gradI

    # Local forward center intensity (rho=0) at this axial location.
    If0 = 2.0 * power_w / (math.pi * w**2)
    absC = abs(response.shift_per_intensity)
    Uax = absC * If0 * (4.0 * sr)
    Upeak = absC * If0 * (1.0 + sr)**2
    Vmin = -Upeak if response.shift_per_intensity < 0 else 0.0

    gamma_sc = response.scatter_per_intensity * intensity

    return {
        "V": V,
        "gradV": gradV,
        "gamma_sc": gamma_sc,
        "w": w,
        "rho2": rho2,
        "s": s,
        "Uax": Uax,
        "Upeak": Upeak,
        "Vmin": Vmin,
        "axis": e,
        "k": k,
    }

def tilted_barrier(xp, Uax, k, mass, acceleration):
    """
    Local downhill barrier for -U cos^2(kx) + m a x.

    Returns zero when |a| >= a_c.
    """
    ac = Uax * k / mass
    beta = xp.where(ac > 0.0, abs(acceleration) / ac, xp.inf)
    clipped = xp.clip(beta, 0.0, 1.0)
    F = xp.sqrt(xp.maximum(0.0, 1.0 - clipped**2)) - clipped * xp.arccos(clipped)
    return xp.where(beta < 1.0, Uax * F, 0.0)
