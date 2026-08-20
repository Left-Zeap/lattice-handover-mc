"""Measured transport and handover control waveforms.

The default models in :mod:`continuous_loading` keep their existing analytic
ramps.  This module adds an optional, immutable tuple-backed representation of
measured AOM/optics traces.  Tuple storage keeps dataclass equality, process
pool pickling and GPU batch consistency checks inexpensive and predictable.

Transport CSV columns
---------------------

``time_ms`` is required.  At least one of ``position_m``, ``velocity_m_s`` or
``aom_frequency_difference_mhz`` must be present.  Missing kinematic columns
are reconstructed with trapezoidal integration / finite differences.  The
optional optical columns are:

``source_power_scale``
    Per-branch source power relative to the nominal end power of that leg
    (L1 handover end or L2 science-region end).
``waist_um``
    Measured effective waist.  If absent, the existing geometric model is used.
``delivery_efficiency_scale``
    Multiplier on the configured source-to-atoms delivery efficiency.

Handover CSV columns
--------------------

``time_ms``, ``lattice1_fraction`` and ``lattice2_fraction`` are required.
``relative_phase_rad`` is optional and represents a shared electronic/optical
phase transient added to the per-trajectory static phase.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path

import numpy as np


def _finite_tuple(name: str, values) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) < 2:
        raise ValueError(f"{name}至少需要两个采样点")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name}必须全部为有限数")
    return result


def _optional_finite_tuple(name: str, values) -> tuple[float, ...] | None:
    if values is None:
        return None
    return _finite_tuple(name, values)


def _validate_time(time_ms: tuple[float, ...]) -> None:
    if abs(time_ms[0]) > 1e-12:
        raise ValueError("控制波形必须从 time_ms=0 开始")
    if any(right <= left for left, right in zip(time_ms, time_ms[1:])):
        raise ValueError("控制波形 time_ms 必须严格递增")


def _same_length(reference: tuple[float, ...], **columns) -> None:
    for name, values in columns.items():
        if values is not None and len(values) != len(reference):
            raise ValueError(f"控制波形列 {name} 与 time_ms 长度不一致")


def _read_csv_columns(path: str | Path) -> dict[str, list[float]]:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise ValueError(f"控制波形文件不存在：{csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("控制波形 CSV 缺少表头")
        names = tuple(str(name).strip() for name in reader.fieldnames)
        columns: dict[str, list[float]] = {name: [] for name in names}
        for row_number, row in enumerate(reader, start=2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            for name in names:
                raw = str(row.get(name, "") or "").strip()
                if raw == "":
                    columns[name].append(float("nan"))
                    continue
                try:
                    columns[name].append(float(raw))
                except ValueError as exc:
                    raise ValueError(
                        f"控制波形 CSV 第 {row_number} 行 {name} 不是数字"
                    ) from exc
    if not columns or not next(iter(columns.values()), []):
        raise ValueError("控制波形 CSV 没有数据行")
    return columns


def _required_column(columns: dict[str, list[float]], name: str) -> np.ndarray:
    if name not in columns:
        raise ValueError(f"控制波形 CSV 缺少必需列 {name}")
    values = np.asarray(columns[name], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"控制波形列 {name} 不能有空值或非有限数")
    return values


def _optional_column(
    columns: dict[str, list[float]], name: str
) -> np.ndarray | None:
    if name not in columns:
        return None
    values = np.asarray(columns[name], dtype=float)
    if np.all(np.isnan(values)):
        return None
    if not np.all(np.isfinite(values)):
        raise ValueError(f"控制波形列 {name} 不能部分留空")
    return values


def _gradient(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    edge_order = 2 if values.size >= 3 else 1
    return np.gradient(values, time_s, edge_order=edge_order)


def _integrate_trapezoid(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    result[1:] = np.cumsum(
        0.5 * (values[1:] + values[:-1]) * np.diff(time_s)
    )
    return result


@dataclass(frozen=True)
class TransportControlWaveform:
    """Measured transport kinematics and optional optical control channels."""

    time_ms: tuple[float, ...]
    position_m: tuple[float, ...]
    velocity_m_s: tuple[float, ...]
    acceleration_m_s2: tuple[float, ...]
    aom_frequency_difference_mhz: tuple[float, ...]
    source_power_scale: tuple[float, ...] | None = None
    waist_um: tuple[float, ...] | None = None
    delivery_efficiency_scale: tuple[float, ...] | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        time_ms = _finite_tuple("time_ms", self.time_ms)
        position = _finite_tuple("position_m", self.position_m)
        velocity = _finite_tuple("velocity_m_s", self.velocity_m_s)
        acceleration = _finite_tuple("acceleration_m_s2", self.acceleration_m_s2)
        frequency = _finite_tuple(
            "aom_frequency_difference_mhz", self.aom_frequency_difference_mhz
        )
        power = _optional_finite_tuple("source_power_scale", self.source_power_scale)
        waist = _optional_finite_tuple("waist_um", self.waist_um)
        delivery = _optional_finite_tuple(
            "delivery_efficiency_scale", self.delivery_efficiency_scale
        )
        _validate_time(time_ms)
        _same_length(
            time_ms,
            position_m=position,
            velocity_m_s=velocity,
            acceleration_m_s2=acceleration,
            aom_frequency_difference_mhz=frequency,
            source_power_scale=power,
            waist_um=waist,
            delivery_efficiency_scale=delivery,
        )
        if position[0] < -1e-12 or any(
            right + 1e-12 < left for left, right in zip(position, position[1:])
        ):
            raise ValueError("运输波形 position_m 必须从非负位置单调前进")
        if power is not None and any(value <= 0.0 for value in power):
            raise ValueError("source_power_scale 必须为正")
        if waist is not None and any(value <= 0.0 for value in waist):
            raise ValueError("waist_um 必须为正")
        if delivery is not None and any(value <= 0.0 for value in delivery):
            raise ValueError("delivery_efficiency_scale 必须为正")
        object.__setattr__(self, "time_ms", time_ms)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "velocity_m_s", velocity)
        object.__setattr__(self, "acceleration_m_s2", acceleration)
        object.__setattr__(self, "aom_frequency_difference_mhz", frequency)
        object.__setattr__(self, "source_power_scale", power)
        object.__setattr__(self, "waist_um", waist)
        object.__setattr__(self, "delivery_efficiency_scale", delivery)

    @property
    def duration_ms(self) -> float:
        return self.time_ms[-1]

    @property
    def distance_m(self) -> float:
        return self.position_m[-1]

    @property
    def maximum_velocity_m_s(self) -> float:
        return max(abs(value) for value in self.velocity_m_s)

    def sample(self, time_s: float) -> dict[str, float | None]:
        t_ms = min(max(float(time_s) * 1e3, 0.0), self.duration_ms)

        def value(values: tuple[float, ...] | None) -> float | None:
            if values is None:
                return None
            return float(np.interp(t_ms, self.time_ms, values))

        return {
            "position_m": float(np.interp(t_ms, self.time_ms, self.position_m)),
            "velocity_m_s": float(
                np.interp(t_ms, self.time_ms, self.velocity_m_s)
            ),
            "acceleration_m_s2": float(
                np.interp(t_ms, self.time_ms, self.acceleration_m_s2)
            ),
            "aom_frequency_difference_mhz": float(
                np.interp(t_ms, self.time_ms, self.aom_frequency_difference_mhz)
            ),
            "source_power_scale": value(self.source_power_scale),
            "waist_um": value(self.waist_um),
            "delivery_efficiency_scale": value(self.delivery_efficiency_scale),
        }

    def sample_optional_array(
        self, name: str, time_s: np.ndarray
    ) -> np.ndarray | None:
        values = getattr(self, name)
        if values is None:
            return None
        return np.interp(
            np.asarray(time_s, dtype=float) * 1e3,
            np.asarray(self.time_ms),
            np.asarray(values),
        )

    @classmethod
    def from_csv(
        cls, path: str | Path, *, wavelength_nm: float
    ) -> "TransportControlWaveform":
        if not math.isfinite(wavelength_nm) or wavelength_nm <= 0.0:
            raise ValueError("运输波形参考波长必须是有限正数")
        columns = _read_csv_columns(path)
        time_ms = _required_column(columns, "time_ms")
        time_s = time_ms * 1e-3
        if abs(time_s[0]) > 1e-15 or np.any(np.diff(time_s) <= 0.0):
            raise ValueError("控制波形 time_ms 必须从 0 开始并严格递增")
        position = _optional_column(columns, "position_m")
        velocity = _optional_column(columns, "velocity_m_s")
        frequency = _optional_column(columns, "aom_frequency_difference_mhz")
        if position is None and velocity is None and frequency is None:
            raise ValueError(
                "运输波形至少需要 position_m、velocity_m_s 或 "
                "aom_frequency_difference_mhz 之一"
            )
        if velocity is None:
            if frequency is not None:
                velocity = 0.5 * wavelength_nm * 1e-9 * frequency * 1e6
            else:
                velocity = _gradient(position, time_s)
        if position is None:
            position = _integrate_trapezoid(velocity, time_s)
        position = position - position[0]
        acceleration = _optional_column(columns, "acceleration_m_s2")
        if acceleration is None:
            acceleration = _gradient(velocity, time_s)
        if frequency is None:
            frequency = 2.0 * velocity / (wavelength_nm * 1e-9) * 1e-6
        return cls(
            time_ms=tuple(time_ms),
            position_m=tuple(position),
            velocity_m_s=tuple(velocity),
            acceleration_m_s2=tuple(acceleration),
            aom_frequency_difference_mhz=tuple(frequency),
            source_power_scale=(
                None
                if (value := _optional_column(columns, "source_power_scale")) is None
                else tuple(value)
            ),
            waist_um=(
                None
                if (value := _optional_column(columns, "waist_um")) is None
                else tuple(value)
            ),
            delivery_efficiency_scale=(
                None
                if (
                    value := _optional_column(
                        columns, "delivery_efficiency_scale"
                    )
                ) is None
                else tuple(value)
            ),
            source_path=str(Path(path).expanduser().resolve()),
        )


@dataclass(frozen=True)
class HandoverControlWaveform:
    """Measured L1/L2 handover depth fractions and optional phase transient."""

    time_ms: tuple[float, ...]
    lattice1_fraction: tuple[float, ...]
    lattice2_fraction: tuple[float, ...]
    relative_phase_rad: tuple[float, ...] | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        time_ms = _finite_tuple("time_ms", self.time_ms)
        fraction1 = _finite_tuple("lattice1_fraction", self.lattice1_fraction)
        fraction2 = _finite_tuple("lattice2_fraction", self.lattice2_fraction)
        phase = _optional_finite_tuple("relative_phase_rad", self.relative_phase_rad)
        _validate_time(time_ms)
        _same_length(
            time_ms,
            lattice1_fraction=fraction1,
            lattice2_fraction=fraction2,
            relative_phase_rad=phase,
        )
        if any(value < 0.0 for value in fraction1 + fraction2):
            raise ValueError("handover 深度分数不能为负")
        if abs(fraction1[0] - 1.0) > 1e-6 or abs(fraction2[0]) > 1e-6:
            raise ValueError("handover 波形起点必须为 L1=1、L2=0")
        if abs(fraction1[-1]) > 1e-6 or abs(fraction2[-1] - 1.0) > 1e-6:
            raise ValueError("handover 波形终点必须为 L1=0、L2=1")
        object.__setattr__(self, "time_ms", time_ms)
        object.__setattr__(self, "lattice1_fraction", fraction1)
        object.__setattr__(self, "lattice2_fraction", fraction2)
        object.__setattr__(self, "relative_phase_rad", phase)

    @property
    def duration_ms(self) -> float:
        return self.time_ms[-1]

    def sample(self, time_s: float) -> tuple[float, float, float]:
        t_ms = min(max(float(time_s) * 1e3, 0.0), self.duration_ms)
        phase = (
            0.0
            if self.relative_phase_rad is None
            else float(np.interp(t_ms, self.time_ms, self.relative_phase_rad))
        )
        return (
            float(np.interp(t_ms, self.time_ms, self.lattice1_fraction)),
            float(np.interp(t_ms, self.time_ms, self.lattice2_fraction)),
            phase,
        )

    def sampled_arrays(
        self, step_times_s: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        times_ms = np.asarray(step_times_s, dtype=float) * 1e3
        phase_values = (
            np.zeros_like(times_ms)
            if self.relative_phase_rad is None
            else np.interp(times_ms, self.time_ms, self.relative_phase_rad)
        )
        return (
            np.interp(times_ms, self.time_ms, self.lattice1_fraction),
            np.interp(times_ms, self.time_ms, self.lattice2_fraction),
            phase_values,
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> "HandoverControlWaveform":
        columns = _read_csv_columns(path)
        phase = _optional_column(columns, "relative_phase_rad")
        return cls(
            time_ms=tuple(_required_column(columns, "time_ms")),
            lattice1_fraction=tuple(
                _required_column(columns, "lattice1_fraction")
            ),
            lattice2_fraction=tuple(
                _required_column(columns, "lattice2_fraction")
            ),
            relative_phase_rad=None if phase is None else tuple(phase),
            source_path=str(Path(path).expanduser().resolve()),
        )
