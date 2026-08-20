"""一维逆向反射高斯光晶格的工程模型。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .atomic import AlkaliAtom
from .constants import BOLTZMANN, GRAVITY, HBAR
from .dipole import DipoleResult, scalar_potential_and_scattering


def _positive_finite(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name}必须是有限正数")
    return value


def gaussian_peak_intensity(power_w: float, waist_m: float) -> float:
    """返回 TEM00 光束中心的行波峰值强度 ``2P/(pi*w^2)``。"""
    _positive_finite("行波功率", power_w)
    _positive_finite("束腰", waist_m)
    return 2.0 * power_w / (math.pi * waist_m**2)


def standing_wave_antinode_intensity(
    forward_power_w: float,
    waist_m: float,
    retro_power_ratio: float,
) -> float:
    """返回前向光和逆向光相干叠加后的波腹峰值强度。

    ``retro_power_ratio`` 定义为原子处逆向功率与前向功率之比。若两束
    功率相等，比值为 1，波腹强度是单束行波峰值强度的 4 倍。
    """
    _positive_finite("前向功率", forward_power_w)
    _positive_finite("束腰", waist_m)
    if (
        not math.isfinite(retro_power_ratio)
        or retro_power_ratio < 0.0
        or retro_power_ratio > 1.0
    ):
        raise ValueError("逆向/前向功率比必须位于 [0, 1]")

    forward_intensity = gaussian_peak_intensity(forward_power_w, waist_m)
    interference_factor = (1.0 + math.sqrt(retro_power_ratio)) ** 2
    return forward_intensity * interference_factor


def gaussian_clipping_loss(aperture_radius_m: float, beam_radius_m: float) -> float:
    """理想高斯光通过同心圆孔时，被截掉的功率比例。

    对 1/e² 强度半径为 ``w`` 的圆对称高斯光，半径 ``a`` 以外的功率
    比例为 ``exp(-2*a^2/w^2)``。该式不包含偏心、像差和机械公差。
    """
    _positive_finite("孔径半径", aperture_radius_m)
    _positive_finite("光束半径", beam_radius_m)
    return math.exp(-2.0 * (aperture_radius_m / beam_radius_m) ** 2)


@dataclass(frozen=True)
class LatticeMetrics:
    """一组可直接用于运输与温升估计的晶格指标。"""

    atom_label: str
    laser_wavelength_nm: float
    forward_power_w: float
    retro_power_ratio: float
    waist_um: float
    antinode_intensity_w_m2: float
    depth_uK: float
    scattering_rate_s: float
    recoil_temperature_uK: float
    depth_in_recoil: float
    radial_frequency_hz: float
    axial_frequency_hz: float
    critical_axial_acceleration_m_s2: float
    dipole: DipoleResult


def evaluate_lattice(
    atom: AlkaliAtom,
    laser_wavelength_nm: float,
    forward_power_w: float,
    waist_um: float,
    retro_power_ratio: float = 1.0,
) -> LatticeMetrics:
    """计算一维逆向反射高斯晶格在束腰处的主要指标。"""
    waist_m = _positive_finite("束腰", waist_um) * 1e-6
    intensity = standing_wave_antinode_intensity(
        forward_power_w,
        waist_m,
        retro_power_ratio,
    )
    dipole = scalar_potential_and_scattering(
        atom,
        laser_wavelength_nm,
        intensity,
    )
    depth_j = abs(dipole.potential_j)
    wave_number = 2.0 * math.pi / (laser_wavelength_nm * 1e-9)
    recoil_j = HBAR**2 * wave_number**2 / (2.0 * atom.mass_kg)

    # -U exp(-2r²/w²) cos²(kz) 在势阱底部的谐振近似。
    radial_omega = math.sqrt(4.0 * depth_j / (atom.mass_kg * waist_m**2))
    axial_omega = math.sqrt(
        2.0 * depth_j * wave_number**2 / atom.mass_kg
    )

    # 加速参考系中的线性势 ma z 不能超过晶格的最大轴向恢复力 U*k。
    critical_acceleration = depth_j * wave_number / atom.mass_kg

    return LatticeMetrics(
        atom_label=atom.label,
        laser_wavelength_nm=laser_wavelength_nm,
        forward_power_w=forward_power_w,
        retro_power_ratio=retro_power_ratio,
        waist_um=waist_um,
        antinode_intensity_w_m2=intensity,
        depth_uK=depth_j / BOLTZMANN * 1e6,
        scattering_rate_s=dipole.scattering_rate_s,
        recoil_temperature_uK=recoil_j / BOLTZMANN * 1e6,
        depth_in_recoil=depth_j / recoil_j,
        radial_frequency_hz=radial_omega / (2.0 * math.pi),
        axial_frequency_hz=axial_omega / (2.0 * math.pi),
        critical_axial_acceleration_m_s2=critical_acceleration,
        dipole=dipole,
    )


def power_for_target_depth(
    atom: AlkaliAtom,
    laser_wavelength_nm: float,
    target_depth_uK: float,
    waist_um: float,
    retro_power_ratio: float = 1.0,
) -> float:
    """反解达到目标晶格深度所需的原子处前向功率。"""
    _positive_finite("目标阱深", target_depth_uK)
    per_watt = evaluate_lattice(
        atom=atom,
        laser_wavelength_nm=laser_wavelength_nm,
        forward_power_w=1.0,
        waist_um=waist_um,
        retro_power_ratio=retro_power_ratio,
    )
    return target_depth_uK / per_watt.depth_uK


def tilted_lattice_barrier_fraction(acceleration_m_s2: float, critical_m_s2: float) -> float:
    """返回匀加速参考系中，下坡方向势垒相对静态阱深的比例。

    对势 ``-U cos²(kz) + m a z``，令 ``beta = |a|/a_critical``，则
    下坡势垒为

    ``U_eff/U = sqrt(1-beta²) - beta*acos(beta)``。

    当 ``beta >= 1`` 时局域极小值消失，返回 0。
    """
    _positive_finite("临界加速度", critical_m_s2)
    if not math.isfinite(acceleration_m_s2):
        raise ValueError("加速度必须是有限数")
    beta = abs(acceleration_m_s2) / critical_m_s2
    if beta >= 1.0:
        return 0.0
    return math.sqrt(1.0 - beta**2) - beta * math.acos(beta)


def gaussian_gravity_trap(
    depth_j: float,
    waist_m: float,
    atom_mass_kg: float,
) -> tuple[float, float, float]:
    """Return ``(downhill barrier, minimum potential, y_sag)`` for gravity.

    The radial potential is ``V(y)=-U exp(-2y²/w²)+mgy`` with gravity along
    ``-y``. A local trap exists only while the maximum Gaussian restoring
    force exceeds ``mg``. The two stationary points are found by bisection;
    this host-side scalar calculation is used only at trace/capture points and
    therefore does not add a per-particle or per-step GPU cost.
    """
    if not math.isfinite(depth_j) or depth_j < 0.0:
        raise ValueError("阱深必须是有限非负数")
    _positive_finite("束腰", waist_m)
    _positive_finite("原子质量", atom_mass_kg)
    if depth_j == 0.0:
        return 0.0, 0.0, 0.0

    weight = atom_mass_kg * GRAVITY
    maximum_restoring_force = 2.0 * depth_j / waist_m * math.exp(-0.5)
    if weight >= maximum_restoring_force:
        return 0.0, -depth_j, 0.0

    # s*exp(-2s²)=q has the closed form
    # s=0.5*sqrt(-W_{0,-1}(-4q²)).  A compact three-step Halley solve avoids
    # importing SciPy and is much cheaper than per-trace-point bisection in a
    # large analytic scan.
    q = weight * waist_m / (4.0 * depth_j)
    z = -4.0 * q * q
    if z == 0.0:  # Extreme floating-point limit: gravity is negligible.
        return depth_j, -depth_j, -q * waist_m

    def negative_lambert_w(branch: int) -> float:
        distance_from_branch = max(0.0, 1.0 + math.e * z)
        if distance_from_branch < 0.3:
            p = math.sqrt(2.0 * distance_from_branch)
            if branch == 0:
                value = (
                    -1.0 + p - p**2 / 3.0 + 11.0 * p**3 / 72.0
                    - 43.0 * p**4 / 540.0
                )
            else:
                value = (
                    -1.0 - p - p**2 / 3.0 - 11.0 * p**3 / 72.0
                    - 43.0 * p**4 / 540.0
                )
        elif branch == 0:
            value = z * (1.0 - z + 1.5 * z * z)
        else:
            log_z = math.log(-z)
            log_log = math.log(-log_z)
            value = log_z - log_log + log_log / log_z
        for _ in range(3):
            if abs(value + 1.0) < 1e-12:
                break
            exponential = math.exp(value)
            residual = value * exponential - z
            denominator = exponential * (value + 1.0) - (
                (value + 2.0) * residual / (2.0 * (value + 1.0))
            )
            value -= residual / denominator
        return value

    minimum_s = 0.5 * math.sqrt(-negative_lambert_w(0))
    saddle_s = 0.5 * math.sqrt(-negative_lambert_w(-1))

    def potential(s: float) -> float:
        return -depth_j * math.exp(-2.0 * s * s) - weight * waist_m * s

    minimum_potential = potential(minimum_s)
    barrier_j = max(0.0, potential(saddle_s) - minimum_potential)
    return barrier_j, minimum_potential, -waist_m * minimum_s


def gaussian_gravity_barriers_j(
    depth_j: float | np.ndarray,
    waist_m: float | np.ndarray,
    atom_mass_kg: float,
) -> np.ndarray:
    """Vectorized downhill gravity barrier for analytic time/grid scans.

    This is the array equivalent of :func:`gaussian_gravity_trap`.  It keeps
    the three Halley updates inside NumPy so enabling gravity does not add a
    Python root-solve loop at every analytic trace point.
    """
    depths, waists = np.broadcast_arrays(
        np.asarray(depth_j, dtype=float), np.asarray(waist_m, dtype=float)
    )
    if np.any(~np.isfinite(depths)) or np.any(depths < 0.0):
        raise ValueError("阱深必须是有限非负数")
    if np.any(~np.isfinite(waists)) or np.any(waists <= 0.0):
        raise ValueError("束腰必须是有限正数")
    _positive_finite("原子质量", atom_mass_kg)

    barriers = np.zeros_like(depths)
    positive = depths > 0.0
    if not np.any(positive):
        return barriers
    weight = atom_mass_kg * GRAVITY
    supported = positive & (
        weight < 2.0 * depths / waists * math.exp(-0.5)
    )
    if not np.any(supported):
        return barriers

    selected_depth = depths[supported]
    selected_waist = waists[supported]
    q = weight * selected_waist / (4.0 * selected_depth)
    z = -4.0 * q * q
    negligible = z == 0.0
    result = selected_depth.copy()
    active = ~negligible
    if np.any(active):
        za = z[active]
        distance = np.maximum(0.0, 1.0 + math.e * za)
        near = distance < 0.3
        p = np.sqrt(2.0 * distance)
        w0 = np.empty_like(za)
        wm = np.empty_like(za)
        w0[near] = (
            -1.0 + p[near] - p[near] ** 2 / 3.0
            + 11.0 * p[near] ** 3 / 72.0
            - 43.0 * p[near] ** 4 / 540.0
        )
        wm[near] = (
            -1.0 - p[near] - p[near] ** 2 / 3.0
            - 11.0 * p[near] ** 3 / 72.0
            - 43.0 * p[near] ** 4 / 540.0
        )
        far = ~near
        w0[far] = za[far] * (
            1.0 - za[far] + 1.5 * za[far] * za[far]
        )
        log_z = np.log(-za[far])
        log_log = np.log(-log_z)
        wm[far] = log_z - log_log + log_log / log_z
        for values in (w0, wm):
            for _ in range(3):
                safe = np.abs(values + 1.0) >= 1e-12
                exponential = np.exp(values[safe])
                residual = values[safe] * exponential - za[safe]
                denominator = exponential * (values[safe] + 1.0) - (
                    (values[safe] + 2.0)
                    * residual
                    / (2.0 * (values[safe] + 1.0))
                )
                values[safe] -= residual / denominator
        minimum_s = 0.5 * np.sqrt(-w0)
        saddle_s = 0.5 * np.sqrt(-wm)
        depth_active = selected_depth[active]
        waist_active = selected_waist[active]
        minimum_potential = (
            -depth_active * np.exp(-2.0 * minimum_s**2)
            - weight * waist_active * minimum_s
        )
        saddle_potential = (
            -depth_active * np.exp(-2.0 * saddle_s**2)
            - weight * waist_active * saddle_s
        )
        result[active] = np.maximum(
            0.0, saddle_potential - minimum_potential
        )
    barriers[supported] = result
    return barriers


def peak_density_harmonic_site_m3(
    atoms_per_site: float,
    radial_frequency_hz: float,
    axial_frequency_hz: float,
    atom_mass_kg: float,
    temperature_uK: float,
) -> float:
    """复现论文 Methods 中单晶格格点的谐振近似峰值密度。"""
    _positive_finite("每格点原子数", atoms_per_site)
    _positive_finite("径向阱频", radial_frequency_hz)
    _positive_finite("轴向阱频", axial_frequency_hz)
    _positive_finite("原子质量", atom_mass_kg)
    _positive_finite("温度", temperature_uK)

    omega_r = 2.0 * math.pi * radial_frequency_hz
    omega_z = 2.0 * math.pi * axial_frequency_hz
    temperature_k = temperature_uK * 1e-6
    return (
        atoms_per_site
        * omega_r**2
        * omega_z
        * (
            atom_mass_kg
            / (2.0 * math.pi * BOLTZMANN * temperature_k)
        )
        ** 1.5
    )


def atoms_per_site_for_peak_density(
    peak_density_m3: float,
    radial_frequency_hz: float,
    axial_frequency_hz: float,
    atom_mass_kg: float,
    temperature_uK: float,
) -> float:
    """反解达到给定峰值密度所需的平均单格点原子数。"""
    _positive_finite("峰值密度", peak_density_m3)
    density_per_atom = peak_density_harmonic_site_m3(
        atoms_per_site=1.0,
        radial_frequency_hz=radial_frequency_hz,
        axial_frequency_hz=axial_frequency_hz,
        atom_mass_kg=atom_mass_kg,
        temperature_uK=temperature_uK,
    )
    return peak_density_m3 / density_per_atom
