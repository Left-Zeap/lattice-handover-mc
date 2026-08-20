"""Rb-87 与 Cs-133 的 D 线数据和基本原子量。

数值取自仓库中的 ``atom/Rb87.md`` 与 ``atom/Cs133.md``，两份资料
引用 Daniel A. Steck 的碱金属 D 线数据。线宽字段采用实验文献常用
约定 ``Gamma/(2*pi)``，单位 MHz。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .constants import ATOMIC_MASS_UNIT, SPEED_OF_LIGHT


@dataclass(frozen=True)
class DLine:
    """一条碱金属 D 线的最低限度光谱数据。"""

    name: str
    wavelength_nm: float
    linewidth_over_2pi_mhz: float
    scalar_weight: float

    @property
    def wavelength_m(self) -> float:
        return self.wavelength_nm * 1e-9

    @property
    def frequency_hz(self) -> float:
        return SPEED_OF_LIGHT / self.wavelength_m

    @property
    def angular_frequency_rad_s(self) -> float:
        return 2.0 * math.pi * self.frequency_hz

    @property
    def gamma_rad_s(self) -> float:
        return 2.0 * math.pi * self.linewidth_over_2pi_mhz * 1e6


@dataclass(frozen=True)
class AlkaliAtom:
    """用于标量远失谐模型的碱金属原子。"""

    symbol: str
    isotope: int
    mass_u: float
    d1: DLine
    d2: DLine

    @property
    def label(self) -> str:
        return f"{self.symbol}-{self.isotope}"

    @property
    def mass_kg(self) -> float:
        return self.mass_u * ATOMIC_MASS_UNIT

    @property
    def lines(self) -> tuple[DLine, DLine]:
        return self.d1, self.d2

    def laser_wavelength_red_of_d1_nm(self, detuning_ghz: float) -> float:
        """返回相对 D1 线红失谐 ``detuning_ghz`` 对应的真空波长。"""
        if not math.isfinite(detuning_ghz) or detuning_ghz <= 0.0:
            raise ValueError("D1 红失谐量必须是有限正数")
        laser_frequency = self.d1.frequency_hz - detuning_ghz * 1e9
        if laser_frequency <= 0.0:
            raise ValueError("红失谐量过大，导致激光频率不为正")
        return SPEED_OF_LIGHT / laser_frequency * 1e9


RB87 = AlkaliAtom(
    symbol="Rb",
    isotope=87,
    mass_u=86.909_180_520,
    d1=DLine(
        name="D1",
        wavelength_nm=794.978_850_9,
        linewidth_over_2pi_mhz=5.746,
        scalar_weight=1.0,
    ),
    d2=DLine(
        name="D2",
        wavelength_nm=780.241_209_686,
        linewidth_over_2pi_mhz=6.065,
        scalar_weight=2.0,
    ),
)


CS133 = AlkaliAtom(
    symbol="Cs",
    isotope=133,
    mass_u=132.905_451_931,
    d1=DLine(
        name="D1",
        wavelength_nm=894.592_959_86,
        linewidth_over_2pi_mhz=4.5612,
        scalar_weight=1.0,
    ),
    d2=DLine(
        name="D2",
        wavelength_nm=852.347_275_82,
        linewidth_over_2pi_mhz=5.2227,
        scalar_weight=2.0,
    ),
)
