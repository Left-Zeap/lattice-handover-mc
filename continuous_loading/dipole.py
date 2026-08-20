"""碱金属基态的 D1/D2 标量偶极势与散射近似。

模型采用 Grimm、Weidemüller 与 Ovchinnikov 的远失谐光学偶极阱
框架，并把碱金属 ``S1/2 -> P1/2, P3/2`` 的标量线强按 1:2 加权。
这里不包含超精细分辨、矢量/张量光移和 Raman 通道的相干干涉，
因此适合做工程量级估算，而不是最终原子态保真度计算。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .atomic import AlkaliAtom
from .constants import BOLTZMANN, HBAR, SPEED_OF_LIGHT


def _positive_finite(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name}必须是有限正数")
    return value


@dataclass(frozen=True)
class DipoleResult:
    """给定局域光强下的偶极势和总散射率近似。"""

    potential_j: float
    scattering_rate_s: float
    laser_wavelength_nm: float
    detuning_d1_ghz: float
    detuning_d2_ghz: float

    @property
    def depth_uK(self) -> float:
        return abs(self.potential_j) / BOLTZMANN * 1e6

    @property
    def is_red_detuned_from_both_lines(self) -> bool:
        return self.detuning_d1_ghz < 0.0 and self.detuning_d2_ghz < 0.0


def scalar_potential_and_scattering(
    atom: AlkaliAtom,
    laser_wavelength_nm: float,
    intensity_w_m2: float,
    *,
    include_counter_rotating: bool = True,
) -> DipoleResult:
    """计算线偏振、未分辨超精细态下的标量势和散射率。

    偶极势使用

    ``U = (pi*c^2/2) I sum_j[g_j Gamma_j/omega_j^3 * R_j]``

    其中旋波近似下 ``R_j = 1/Delta_j``。若启用反旋项，则
    ``R_j = 1/Delta_j - 1/(omega_L + omega_j)``。

    散射率使用各线非相干求和的工程近似。靠近共振、需要区分 Raman
    与 Rayleigh 散射或指定 ``F,mF`` 时，应换用完整多能级模型。
    """
    _positive_finite("激光波长", laser_wavelength_nm)
    _positive_finite("光强", intensity_w_m2)

    omega_laser = (
        2.0 * math.pi * SPEED_OF_LIGHT / (laser_wavelength_nm * 1e-9)
    )
    potential_sum = 0.0
    scattering_sum = 0.0
    detunings_ghz: dict[str, float] = {}

    for line in atom.lines:
        omega_line = line.angular_frequency_rad_s
        gamma = line.gamma_rad_s
        delta = omega_laser - omega_line
        if delta == 0.0:
            raise ValueError(f"激光与 {atom.label} {line.name} 完全共振")

        rotating_factor = 1.0 / delta
        scattering_factor = 1.0 / (delta * delta)
        if include_counter_rotating:
            rotating_factor -= 1.0 / (omega_laser + omega_line)
            scattering_factor += 1.0 / (omega_laser + omega_line) ** 2

        common = line.scalar_weight / omega_line**3
        potential_sum += common * gamma * rotating_factor
        scattering_sum += common * gamma**2 * scattering_factor
        detunings_ghz[line.name] = delta / (2.0 * math.pi * 1e9)

    potential = (
        math.pi
        * SPEED_OF_LIGHT**2
        / 2.0
        * intensity_w_m2
        * potential_sum
    )
    scattering_rate = (
        math.pi
        * SPEED_OF_LIGHT**2
        / (2.0 * HBAR)
        * intensity_w_m2
        * scattering_sum
    )

    return DipoleResult(
        potential_j=potential,
        scattering_rate_s=scattering_rate,
        laser_wavelength_nm=laser_wavelength_nm,
        detuning_d1_ghz=detunings_ghz["D1"],
        detuning_d2_ghz=detunings_ghz["D2"],
    )
