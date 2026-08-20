"""低温 s 波弹性碰撞的数量级模型。"""

from __future__ import annotations

import math

from .constants import BOLTZMANN


BOHR_RADIUS_M = 5.291_772_105_44e-11


def identical_boson_s_wave_cross_section_m2(scattering_length_bohr: float) -> float:
    """低能、非简并极限下相同玻色子的 ``8*pi*a^2`` 截面。"""
    if not math.isfinite(scattering_length_bohr):
        raise ValueError("散射长度必须是有限数")
    scattering_length_m = abs(scattering_length_bohr) * BOHR_RADIUS_M
    return 8.0 * math.pi * scattering_length_m**2


def mean_relative_speed_m_s(atom_mass_kg: float, temperature_uK: float) -> float:
    """同质量 Maxwell 气体的平均相对速度。"""
    if not math.isfinite(atom_mass_kg) or atom_mass_kg <= 0.0:
        raise ValueError("原子质量必须是有限正数")
    if not math.isfinite(temperature_uK) or temperature_uK <= 0.0:
        raise ValueError("温度必须是有限正数")
    temperature_k = temperature_uK * 1e-6
    return 4.0 * math.sqrt(BOLTZMANN * temperature_k / (math.pi * atom_mass_kg))


def two_body_collision_density_m3_s(
    number_density_m3: float,
    atom_mass_kg: float,
    temperature_uK: float,
    scattering_length_bohr: float,
) -> float:
    """返回论文定义的 ``gamma = 0.5*n^2*v_rel*sigma``。"""
    if not math.isfinite(number_density_m3) or number_density_m3 <= 0.0:
        raise ValueError("数密度必须是有限正数")
    relative_speed = mean_relative_speed_m_s(atom_mass_kg, temperature_uK)
    cross_section = identical_boson_s_wave_cross_section_m2(
        scattering_length_bohr
    )
    return 0.5 * number_density_m3**2 * relative_speed * cross_section
