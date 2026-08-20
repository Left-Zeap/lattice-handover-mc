"""Lattice-2 宏观运输腿与科学区原子库汇总。

handover 捕获的原子已经在 Lattice-2 中（handover 束腰处静止）。本模块
把这批原子作为初态，复用 ``simulate_l1_transport`` 的单段晶格运输积分器
计算 L2 段（论文：0.17 m / 21 ms、束腰 250→150 µm、加速度 4000 m/s²）
的宏观升温和统计留存率，并汇总科学区原子库的密度量级。

温度口径假设：handover Monte Carlo 的 ``final_temperature_uK`` 是捕获
样本的 ``<E>/(3 k_B)``，含势能激发、未必热化。把它当作 L2 腿的热平衡
初温，等价于假设 21 ms 运输中碰撞把系综再热化；当前密度下每原子约
一次碰撞，这是边际假设，引用结果时应显式说明。

功率口径与 L1 一致：恒阱深假设下源端功率随束腰平方缩放，L2 末端
（150 µm）源功率 = handover 端（250 µm）源功率 × (150/250)²。

若 ``L1TransportInputs.conveyor_enabled=True``，``replace`` 会把
conveyor 几何参数一并带入 L2 腿（L=0.17 m，束腰间距按同一 s 解释，
允许 s>L），此时 L2 腿同样走恒功率、逐点几何剖面的积分分支。

若 ``L1TransportInputs.transport_method="monte_carlo"``，``replace``
会把运输 MC 模式与步长一并带入 L2 腿，L2 腿同样走轨迹级 Monte Carlo
（``transport_mc.py``，与 L1 共用 ``handover_monte_carlo`` 配置组）。

``include_gravity`` 也由同一次 ``replace`` 继承；L2 局部坐标只绕公共
竖直 y 轴旋转，所以重力仍统一沿 -y，不做额外坐标投影。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .control_waveforms import TransportControlWaveform
from .lattice import evaluate_lattice
from .l1_transport import (
    L1TransportInputs,
    L1TransportTrace,
    L1_TRANSPORT_CONFIGURATION,
    _atom_from_label,
    _peak_density_m3,
    simulate_l1_transport,
)


_L2 = L1_TRANSPORT_CONFIGURATION["l2_transport"]


@dataclass(frozen=True)
class L2TransportInputs:
    """L2 段的固定时序、束腰和统计参数（论文工作点）。"""

    distance_m: float = float(_L2["distance_m"])
    acceleration_m_s2: float = float(_L2["acceleration_m_s2"])
    maximum_velocity_m_s: float = float(_L2["maximum_velocity_m_s"])
    kinematic_profile: str = str(_L2.get("kinematic_profile", "trapezoid"))
    end_waist_um: float = float(_L2["end_waist_um"])
    time_points: int = int(_L2["time_points"])
    occupied_lattice_sites: float = float(_L2["occupied_lattice_sites"])
    control_waveform: TransportControlWaveform | None = None

    def __post_init__(self) -> None:
        positive = {
            "L2 运输距离": self.distance_m,
            "L2 加速度": self.acceleration_m_s2,
            "L2 最大速度": self.maximum_velocity_m_s,
            "L2 末端束腰": self.end_waist_um,
            "L2 占据格点数": self.occupied_lattice_sites,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name}必须是有限正数")
        if self.time_points < 4:
            raise ValueError("L2 时间点数至少为 4")
        if self.kinematic_profile not in {"trapezoid", "minimum_jerk"}:
            raise ValueError("L2 理想运输轨迹必须是 trapezoid 或 minimum_jerk")
        distance_factor = (
            1.875 if self.kinematic_profile == "minimum_jerk" else 1.0
        )
        if (
            self.control_waveform is None
            and distance_factor * self.maximum_velocity_m_s**2
            > self.acceleration_m_s2 * self.distance_m
        ):
            raise ValueError("L2 运输距离不足以在给定加速度下达到指定速度")
        if self.control_waveform is not None:
            tolerance = max(1e-6, 5e-3 * self.distance_m)
            if abs(self.control_waveform.distance_m - self.distance_m) > tolerance:
                raise ValueError("L2 实测运输波形终点与配置运输距离不一致")


@dataclass(frozen=True)
class ScienceRegionSummary:
    """L2 送达科学区后的原子库状态汇总。"""

    atom_number: float
    temperature_uK: float
    depth_uK: float
    end_waist_um: float
    radial_frequency_hz: float
    axial_frequency_hz: float
    atoms_per_site: float
    occupied_lattice_sites: float
    peak_density_m3: float


@dataclass(frozen=True)
class L2TransportResult:
    """L2 腿的宏观运输结果与科学区汇总。"""

    inputs: L2TransportInputs
    leg_trace: L1TransportTrace
    input_temperature_uK: float
    input_atom_number: float
    end_source_power_w: float
    final_temperature_uK: float
    leg_temperature_rise_uK: float
    leg_retention_fraction: float
    final_atom_number: float
    cumulative_scattering_events: float
    minimum_effective_barrier_uK: float
    science: ScienceRegionSummary


def l2_leg_inputs(
    transport_inputs: L1TransportInputs,
    l2_inputs: L2TransportInputs,
    captured_temperature_uK: float,
    captured_atom_number: float,
) -> L1TransportInputs:
    """构造 L2 腿的运输输入：几何时序取 L2 段，初态为 handover 捕获样本。

    conveyor 几何与运输 Monte Carlo 模式（``transport_method``、
    ``mc_*`` 字段）经 ``replace`` 从 ``transport_inputs`` 自动继承。
    """
    return replace(
        transport_inputs,
        distance_m=l2_inputs.distance_m,
        acceleration_m_s2=l2_inputs.acceleration_m_s2,
        maximum_velocity_m_s=l2_inputs.maximum_velocity_m_s,
        kinematic_profile=l2_inputs.kinematic_profile,
        start_waist_um=transport_inputs.handover_waist_um,
        handover_waist_um=l2_inputs.end_waist_um,
        time_points=l2_inputs.time_points,
        initial_temperature_uK=captured_temperature_uK,
        initial_atom_number=captured_atom_number,
        mot_atom_number=captured_atom_number,
        occupied_lattice_sites=l2_inputs.occupied_lattice_sites,
        control_waveform=l2_inputs.control_waveform,
    )


def l2_end_source_power_w(
    transport_inputs: L1TransportInputs,
    l2_inputs: L2TransportInputs,
    handover_source_power_w: float,
) -> float:
    """恒阱深假设下 L2 末端束腰处的每分支源端功率（随束腰平方缩放）。"""
    waist_ratio = l2_inputs.end_waist_um / transport_inputs.handover_waist_um
    return handover_source_power_w * waist_ratio**2


def l2_result_from_leg_trace(
    transport_inputs: L1TransportInputs,
    l2_inputs: L2TransportInputs,
    detuning_ghz: float,
    end_source_power_w: float,
    captured_temperature_uK: float,
    captured_atom_number: float,
    leg_trace: L1TransportTrace,
) -> L2TransportResult:
    """由 L2 腿 trace 装配科学区汇总；逐点解析腿与批量 MC 腿共用口径。"""
    atom = _atom_from_label(transport_inputs.atom_label)
    wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning_ghz)
    end_control = (
        None
        if l2_inputs.control_waveform is None
        else l2_inputs.control_waveform.sample(
            l2_inputs.control_waveform.duration_ms * 1e-3
        )
    )
    end_waist_um = (
        l2_inputs.end_waist_um
        if end_control is None or end_control["waist_um"] is None
        else float(end_control["waist_um"])
    )
    actual_end_source_power_w = (
        end_source_power_w
        if end_control is None or end_control["source_power_scale"] is None
        else end_source_power_w * float(end_control["source_power_scale"])
    )
    end_delivery = transport_inputs.delivery_efficiency * (
        1.0
        if end_control is None
        or end_control["delivery_efficiency_scale"] is None
        else float(end_control["delivery_efficiency_scale"])
    )
    end_lattice = evaluate_lattice(
        atom,
        wavelength_nm,
        forward_power_w=actual_end_source_power_w * end_delivery,
        waist_um=end_waist_um,
        retro_power_ratio=transport_inputs.retro_power_ratio,
    )
    point = leg_trace.point
    atoms_per_site = point.final_atom_number / l2_inputs.occupied_lattice_sites
    peak_density = _peak_density_m3(
        point.final_atom_number,
        l2_inputs.occupied_lattice_sites,
        end_lattice.radial_frequency_hz,
        end_lattice.axial_frequency_hz,
        atom.mass_kg,
        point.final_temperature_uK,
    )
    science = ScienceRegionSummary(
        atom_number=point.final_atom_number,
        temperature_uK=point.final_temperature_uK,
        depth_uK=end_lattice.depth_uK,
        end_waist_um=end_waist_um,
        radial_frequency_hz=end_lattice.radial_frequency_hz,
        axial_frequency_hz=end_lattice.axial_frequency_hz,
        atoms_per_site=atoms_per_site,
        occupied_lattice_sites=l2_inputs.occupied_lattice_sites,
        peak_density_m3=peak_density,
    )
    return L2TransportResult(
        inputs=l2_inputs,
        leg_trace=leg_trace,
        input_temperature_uK=captured_temperature_uK,
        input_atom_number=captured_atom_number,
        end_source_power_w=end_source_power_w,
        final_temperature_uK=point.final_temperature_uK,
        leg_temperature_rise_uK=point.final_temperature_rise_uK,
        leg_retention_fraction=point.final_retention_fraction,
        final_atom_number=point.final_atom_number,
        cumulative_scattering_events=point.cumulative_scattering_events,
        minimum_effective_barrier_uK=min(leg_trace.effective_barrier_uK),
        science=science,
    )


def simulate_l2_transport(
    transport_inputs: L1TransportInputs,
    l2_inputs: L2TransportInputs,
    detuning_ghz: float,
    handover_source_power_w: float,
    captured_temperature_uK: float,
    captured_atom_number: float,
) -> L2TransportResult:
    """把 handover 捕获样本作为初态，积分 L2 段宏观运输。

    ``handover_source_power_w`` 是 handover 束腰（250 µm）处每条晶格
    分支的源端功率，与 L1 扫描网格同一口径；L2 腿内部按恒阱深把功率
    随束腰平方缩放到 150 µm。
    """
    if not math.isfinite(captured_temperature_uK) or captured_temperature_uK <= 0.0:
        raise ValueError("handover 捕获温度必须是有限正数")
    if not math.isfinite(captured_atom_number) or captured_atom_number <= 0.0:
        raise ValueError("handover 捕获原子数必须是有限正数")
    if handover_source_power_w <= 0.0:
        raise ValueError("handover 端源功率必须为正")

    end_source_power = l2_end_source_power_w(
        transport_inputs, l2_inputs, handover_source_power_w
    )
    leg_trace = simulate_l1_transport(
        l2_leg_inputs(
            transport_inputs,
            l2_inputs,
            captured_temperature_uK,
            captured_atom_number,
        ),
        detuning_ghz,
        end_source_power,
    )
    return l2_result_from_leg_trace(
        transport_inputs,
        l2_inputs,
        detuning_ghz,
        end_source_power,
        captured_temperature_uK,
        captured_atom_number,
        leg_trace,
    )
