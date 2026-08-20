"""offset-waist 双束 conveyor 几何模型（可选，默认关闭）。

构成驻波的前向束与回程束的束腰不再重合，而是沿运输轴错开间距 s
（物理思路参照 Matthies et al., PRA 109, 023321 (2024)）。全部公式与
``reports/offset_waist双束conveyor几何理论框架.md`` §3 一致：

- 束腰对称布置在中点两侧：z1 = (L-s)/2，z2 = (L+s)/2，s 可大于 L；
- 两束半径独立自由传播 w_i(z) = w0*sqrt(1+((z-z_i)/z_R)^2)，
  z_R = pi*w0^2/lambda；
- 波腹强度 I_anti = (sqrt(I1)+sqrt(I2))^2，可见度
  V = 2*sqrt(I1*I2)/(I1+I2)，轴向调制 dI_ax = 4*sqrt(I1*I2)；
- 径向阱频按两束曲率精确相加，轴向量（势垒、阱频、临界加速度）用
  轴向调制深度 U_ax = |C_U|*dI_ax。

口径差异：本模型分离波腹势深 U_anti（径向束缚、handover 阱深）与轴向
调制深度 U_ax（逃逸势垒）；``lattice.evaluate_lattice`` 在等腰假设下把
U_anti 同时用作轴向势垒（等强近似下偏高约 1.6%）。错腰时两者差异可远
超 1.6%，因此必须分开。

功率口径：``forward_power_w`` 是**原子处前向功率**，即调用方需自行乘
源端到原子传输效率（P_f = eta_del * P_src），与 ``evaluate_lattice``
一致。

本模块默认不被启用：只有 ``L1TransportInputs.conveyor_enabled=True``
时 ``simulate_l1_transport`` 才走这里的剖面；关闭时现有行为逐位不变。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .atomic import AlkaliAtom
from .constants import BOLTZMANN
from .dipole import scalar_potential_and_scattering


def _positive_finite(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name}必须是有限正数")
    return value


def beam_radius_um(
    waist_um: float,
    wavelength_nm: float,
    z_m: float,
    focus_m: float,
) -> float:
    """高斯光束在轴上位置 ``z_m`` 处的 1/e² 半径（束腰位于 ``focus_m``）。

    ``w(z) = w0*sqrt(1+((z-z_focus)/z_R)^2)``，``z_R = pi*w0^2/lambda``。
    """
    waist_m = _positive_finite("束腰", waist_um) * 1e-6
    wavelength_m = _positive_finite("激光波长", wavelength_nm) * 1e-9
    if not math.isfinite(z_m) or not math.isfinite(focus_m):
        raise ValueError("位置和束腰位置必须是有限数")
    rayleigh_m = math.pi * waist_m**2 / wavelength_m
    return waist_m * math.sqrt(1.0 + ((z_m - focus_m) / rayleigh_m) ** 2) * 1e6


@dataclass(frozen=True)
class ConveyorPoint:
    """错腰双束驻波在运输轴上单个位置的局部晶格量。"""

    position_m: float
    forward_radius_um: float
    retro_radius_um: float
    antinode_intensity_w_m2: float
    visibility: float
    depth_uK: float
    axial_depth_uK: float
    scattering_rate_s: float
    radial_frequency_hz: float
    axial_frequency_hz: float
    critical_axial_acceleration_m_s2: float
    effective_waist_um: float


@dataclass(frozen=True)
class ConveyorProfile:
    """错腰双束驻波沿整条运输路径的剖面与聚合量。"""

    position_m: np.ndarray
    forward_radius_um: np.ndarray
    retro_radius_um: np.ndarray
    antinode_intensity_w_m2: np.ndarray
    visibility: np.ndarray
    depth_uK: np.ndarray
    axial_depth_uK: np.ndarray
    scattering_rate_s: np.ndarray
    radial_frequency_hz: np.ndarray
    axial_frequency_hz: np.ndarray
    critical_axial_acceleration_m_s2: np.ndarray
    effective_waist_um: np.ndarray
    minimum_depth_uK: float
    maximum_depth_uK: float
    minimum_axial_depth_uK: float
    minimum_critical_acceleration_m_s2: float
    minimum_visibility: float


def conveyor_point(
    atom: AlkaliAtom,
    wavelength_nm: float,
    forward_power_w: float,
    waist_um: float,
    separation_cm: float,
    distance_m: float,
    position_m: float,
    retro_power_ratio: float = 1.0,
) -> ConveyorPoint:
    """计算错腰双束 conveyor 在 ``position_m`` 处的局部晶格量。

    两束腰位于 z1=(L-s)/2、z2=(L+s)/2；前向束功率为 ``forward_power_w``
    （原子处口径），回程束携带功率比 ``retro_power_ratio``。偶极系数由
    ``dipole.scalar_potential_and_scattering`` 在单位强度下取得。
    """
    _positive_finite("运输距离", distance_m)
    _positive_finite("束腰", waist_um)
    _positive_finite("激光波长", wavelength_nm)
    _positive_finite("前向功率", forward_power_w)
    if not math.isfinite(separation_cm) or separation_cm < 0.0:
        raise ValueError("束腰间距必须是有限非负数")
    if not math.isfinite(position_m) or not 0.0 <= position_m <= distance_m:
        raise ValueError("位置必须位于 [0, 运输距离]")
    if (
        not math.isfinite(retro_power_ratio)
        or retro_power_ratio < 0.0
        or retro_power_ratio > 1.0
    ):
        raise ValueError("回程功率比必须位于 [0, 1]")

    separation_m = separation_cm * 1e-2
    focus_forward_m = 0.5 * (distance_m - separation_m)
    focus_retro_m = 0.5 * (distance_m + separation_m)
    w1_m = beam_radius_um(waist_um, wavelength_nm, position_m, focus_forward_m) * 1e-6
    w2_m = beam_radius_um(waist_um, wavelength_nm, position_m, focus_retro_m) * 1e-6

    intensity_1 = 2.0 * forward_power_w / (math.pi * w1_m**2)
    intensity_2 = retro_power_ratio * 2.0 * forward_power_w / (math.pi * w2_m**2)
    antinode_intensity = (math.sqrt(intensity_1) + math.sqrt(intensity_2)) ** 2
    geometric_mean = math.sqrt(intensity_1 * intensity_2)
    visibility = 2.0 * geometric_mean / (intensity_1 + intensity_2)
    axial_modulation = 4.0 * geometric_mean

    unit_dipole = scalar_potential_and_scattering(atom, wavelength_nm, 1.0)
    potential_per_intensity = abs(unit_dipole.potential_j)
    scattering_per_intensity = unit_dipole.scattering_rate_s

    antinode_depth_j = potential_per_intensity * antinode_intensity
    axial_depth_j = potential_per_intensity * axial_modulation
    wave_number = 2.0 * math.pi / (wavelength_nm * 1e-9)

    radial_omega_squared = (
        4.0
        * potential_per_intensity
        / atom.mass_kg
        * (intensity_1 / w1_m**2 + intensity_2 / w2_m**2)
    )
    radial_omega = math.sqrt(radial_omega_squared)
    axial_omega = math.sqrt(2.0 * axial_depth_j * wave_number**2 / atom.mass_kg)
    critical_acceleration = axial_depth_j * wave_number / atom.mass_kg
    effective_waist_m = math.sqrt(
        (intensity_1 + intensity_2)
        / (intensity_1 / w1_m**2 + intensity_2 / w2_m**2)
    )

    return ConveyorPoint(
        position_m=position_m,
        forward_radius_um=w1_m * 1e6,
        retro_radius_um=w2_m * 1e6,
        antinode_intensity_w_m2=antinode_intensity,
        visibility=visibility,
        depth_uK=antinode_depth_j / BOLTZMANN * 1e6,
        axial_depth_uK=axial_depth_j / BOLTZMANN * 1e6,
        scattering_rate_s=scattering_per_intensity * antinode_intensity,
        radial_frequency_hz=radial_omega / (2.0 * math.pi),
        axial_frequency_hz=axial_omega / (2.0 * math.pi),
        critical_axial_acceleration_m_s2=critical_acceleration,
        effective_waist_um=effective_waist_m * 1e6,
    )


def conveyor_profile(
    atom: AlkaliAtom,
    wavelength_nm: float,
    forward_power_w: float,
    waist_um: float,
    separation_cm: float,
    distance_m: float,
    positions: np.ndarray,
    retro_power_ratio: float = 1.0,
) -> ConveyorProfile:
    """``conveyor_point`` 的数组版，并聚合全程最浅阱深、最小临界加速度等。"""
    positions = np.asarray(positions, dtype=float)
    points = [
        conveyor_point(
            atom,
            wavelength_nm,
            forward_power_w,
            waist_um,
            separation_cm,
            distance_m,
            float(position),
            retro_power_ratio,
        )
        for position in positions
    ]

    def _column(name: str) -> np.ndarray:
        return np.asarray([getattr(point, name) for point in points], dtype=float)

    depths = _column("depth_uK")
    axial_depths = _column("axial_depth_uK")
    critical = _column("critical_axial_acceleration_m_s2")
    visibility = _column("visibility")
    return ConveyorProfile(
        position_m=positions,
        forward_radius_um=_column("forward_radius_um"),
        retro_radius_um=_column("retro_radius_um"),
        antinode_intensity_w_m2=_column("antinode_intensity_w_m2"),
        visibility=visibility,
        depth_uK=depths,
        axial_depth_uK=axial_depths,
        scattering_rate_s=_column("scattering_rate_s"),
        radial_frequency_hz=_column("radial_frequency_hz"),
        axial_frequency_hz=_column("axial_frequency_hz"),
        critical_axial_acceleration_m_s2=critical,
        effective_waist_um=_column("effective_waist_um"),
        minimum_depth_uK=float(np.min(depths)),
        maximum_depth_uK=float(np.max(depths)),
        minimum_axial_depth_uK=float(np.min(axial_depths)),
        minimum_critical_acceleration_m_s2=float(np.min(critical)),
        minimum_visibility=float(np.min(visibility)),
    )
