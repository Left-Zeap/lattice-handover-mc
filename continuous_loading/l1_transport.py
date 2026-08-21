"""L1 晶格宏观时序、升温和统计留存率二维扫描。

扫描变量是每条晶格分支的固定源端功率和 D1 红失谐。固定运输距离、
加速度和最大速度，采用梯形速度轨迹。束腰沿程变化，前向/反射光强
按 1/w² 改变并形成驻波。模型只积分总体温度和原子数，不追踪单原子。

可选的 offset-waist 双束 conveyor 几何（``conveyor_enabled``，默认
关闭）把"标定高斯包络 + 等束腰反射光"替换为逐点几何剖面：源端
功率仍全程恒定，阱深/阱频/散射率由 ``conveyor_geometry`` 按错腰高斯
光束叠加给出，公式见
``reports/offset_waist双束conveyor几何理论框架.md`` §3。

可选的轨迹级 Monte Carlo 运输（``transport_method="monte_carlo"``，
默认关闭）在函数顶部分支到 ``transport_mc.simulate_leg_monte_carlo``，
用底层双束光场直接传播相空间系综，输出与本宏观腿完全同型，理论见
``reports/运输蒙特卡洛与双束底层光场理论框架.md``。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .atomic import CS133, RB87, AlkaliAtom
from .constants import BOLTZMANN
from .control_waveforms import TransportControlWaveform
from .conveyor_geometry import ConveyorPoint, conveyor_point, conveyor_profile
from .lattice import (
    evaluate_lattice,
    gaussian_gravity_barriers_j,
    gaussian_gravity_trap,
    power_for_target_depth,
    tilted_lattice_barrier_fraction,
)
from .transport import thermal_bound_fraction_3d_harmonic


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "l1_transport_defaults.json"
)


def load_l1_transport_configuration(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, object]:
    """读取 L1 宏观运输扫描配置。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "atom",
        "species_defaults",
        "scan",
        "initial_state",
        "transport",
        "l2_transport",
        "optics",
        "conveyor_geometry",
        "handover_preconditions",
        "loss",
        "noise",
        "selection",
        "handover_monte_carlo",
        "transport_monte_carlo",
        "plot",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError("L1 运输配置缺少分组：" + ", ".join(sorted(missing)))
    return payload


L1_TRANSPORT_CONFIGURATION = load_l1_transport_configuration()


def _section(name: str) -> dict[str, object]:
    value = L1_TRANSPORT_CONFIGURATION[name]
    if not isinstance(value, dict):
        raise ValueError(f"L1 运输配置 {name} 必须是对象")
    return value


_SCAN = _section("scan")
_SPECIES_DEFAULTS = _section("species_defaults")
_INITIAL = _section("initial_state")
_TRANSPORT = _section("transport")
_OPTICS = _section("optics")
_CONVEYOR = _section("conveyor_geometry")
_HANDOVER_PRECONDITIONS = _section("handover_preconditions")
_LOSS = _section("loss")
_NOISE = _section("noise")
_SELECTION = _section("selection")
_TRANSPORT_MC = _section("transport_monte_carlo")
_MC = _section("handover_monte_carlo")
_PLOT = _section("plot")


@dataclass(frozen=True)
class L1TransportInputs:
    """L1 二维扫描、固定时序和宏观损失参数。"""

    atom_label: str = str(L1_TRANSPORT_CONFIGURATION["atom"])
    # 全链路公共重力开关；竖直方向统一为 -y，L2 由 replace 自动继承。
    include_gravity: bool = True
    detuning_min_ghz: float = float(_SCAN["detuning_min_ghz"])
    detuning_max_ghz: float = float(_SCAN["detuning_max_ghz"])
    detuning_points: int = int(_SCAN["detuning_points"])
    handover_source_power_min_w: float = float(
        _SCAN["handover_source_power_min_w"]
    )
    handover_source_power_max_w: float = float(
        _SCAN["handover_source_power_max_w"]
    )
    power_points: int = int(_SCAN["power_points"])
    mot_atom_number: float = float(_INITIAL["mot_atom_number"])
    # 从 MOT 计数时刻经过 compress/idle 到 L1 起点的独立存活率。
    # 默认 1.0；实验测得前级损失后可显式填入。
    pre_ramp_survival_fraction: float = float(
        _INITIAL.get("pre_ramp_survival_fraction", 1.0)
    )
    initial_atom_number: float = float(_INITIAL["loaded_l1_atom_number"])
    initial_temperature_uK: float = float(_INITIAL["temperature_uK"])
    occupied_lattice_sites: float = float(_INITIAL["occupied_lattice_sites"])
    distance_m: float = float(_TRANSPORT["distance_m"])
    acceleration_m_s2: float = float(_TRANSPORT["acceleration_m_s2"])
    maximum_velocity_m_s: float = float(_TRANSPORT["maximum_velocity_m_s"])
    kinematic_profile: str = str(
        _TRANSPORT.get("kinematic_profile", "trapezoid")
    )
    start_waist_um: float = float(_TRANSPORT["start_waist_um"])
    # L1 标定高斯包络：起点半径由 UI 的起点直径换算；w0 与 z0
    # 分别是最小束腰半径和距 L1 起点的位置。两项均为 None 时回退到
    # 旧版 start_waist_um→handover_waist_um 线性插值，供旧调用兼容。
    minimum_waist_um: float | None = (
        None
        if _TRANSPORT.get("minimum_waist_um") is None
        else float(_TRANSPORT["minimum_waist_um"])
    )
    minimum_waist_position_m: float | None = (
        None
        if _TRANSPORT.get("minimum_waist_position_m") is None
        else float(_TRANSPORT["minimum_waist_position_m"])
    )
    # 兼容字段：启用标定高斯包络时由 __post_init__ 自动改写为 z=L
    # 的计算半径；handover、L2 和旧接口仍可沿用这个字段。
    handover_waist_um: float = float(_TRANSPORT["handover_waist_um"])
    time_points: int = int(_TRANSPORT["time_points"])
    delivery_efficiency: float = float(_OPTICS["delivery_efficiency"])
    retro_power_ratio: float = float(_OPTICS["retro_power_ratio"])
    target_depth_uK: float = float(_OPTICS["target_depth_uK"])
    maximum_l1_source_power_w: float = float(
        _OPTICS["maximum_l1_source_power_w"]
    )
    require_minimum_depth: bool = bool(
        _HANDOVER_PRECONDITIONS["minimum_depth"]
    )
    require_maximum_start_power: bool = bool(
        _HANDOVER_PRECONDITIONS["maximum_start_power"]
    )
    require_critical_acceleration: bool = bool(
        _HANDOVER_PRECONDITIONS["critical_acceleration"]
    )
    conveyor_enabled: bool = bool(_CONVEYOR["enabled"])
    conveyor_waist_um: float = float(_CONVEYOR["waist_um"])
    conveyor_waist_separation_cm: float = float(
        _CONVEYOR["waist_separation_cm"]
    )
    background_loss_rate_s: float = float(_LOSS["background_loss_rate_s"])
    internal_loss_probability_per_scatter: float = float(
        _LOSS["internal_loss_probability_per_scatter"]
    )
    two_body_loss_coefficient_m3_s: float = float(
        _LOSS["two_body_loss_coefficient_m3_s"]
    )
    three_body_loss_coefficient_m6_s: float = float(
        _LOSS["three_body_loss_coefficient_m6_s"]
    )
    intensity_noise_psd_1_hz: float = float(
        _NOISE["intensity_noise_psd_1_hz"]
    )
    position_noise_psd_m2_hz: float = float(
        _NOISE["position_noise_psd_m2_hz"]
    )
    temperature_weight: float = float(_SELECTION["temperature_weight"])
    retention_weight: float = float(_SELECTION["retention_weight"])
    transport_method: str = (
        "monte_carlo" if bool(_TRANSPORT_MC["enabled"]) else "analytic"
    )
    transport_time_step_us: float = float(_TRANSPORT_MC["time_step_us"])
    # 运输 Monte Carlo 数值参数：默认与 handover_monte_carlo 配置组同值，
    # 使 UI/CLI 的每调用设置（粒子数、种子、散射、云尺寸）对运输腿同样
    # 生效；transport_method == "analytic" 时不使用。
    mc_particle_count: int = int(_MC["particle_count"])
    mc_seed: int = int(_MC["seed"])
    mc_include_scattering: bool = bool(_MC["include_scattering"])
    mc_cloud_axial_sigma_mm: float = float(_MC["cloud_axial_sigma_mm"])
    mc_compute_backend: str = str(_MC["compute_backend"])
    # 可选实测运输波形。None 时沿用原梯形速度/理想光学跟随接口。
    control_waveform: TransportControlWaveform | None = None

    @property
    def calibrated_gaussian_geometry(self) -> bool:
        """是否使用由起点直径、最小束腰和焦点位置标定的 L1 包络。"""
        return (
            self.minimum_waist_um is not None
            and self.minimum_waist_position_m is not None
        )

    @property
    def start_beam_diameter_um(self) -> float:
        """L1 起点的 1/e² 光强直径 ``2w``。"""
        return 2.0 * self.start_waist_um

    @property
    def effective_rayleigh_range_m(self) -> float | None:
        """由三项实测几何反推的有效瑞利长度；线性兼容模式返回 None。"""
        if not self.calibrated_gaussian_geometry:
            return None
        assert self.minimum_waist_um is not None
        assert self.minimum_waist_position_m is not None
        expansion = (self.start_waist_um / self.minimum_waist_um) ** 2 - 1.0
        return self.minimum_waist_position_m / math.sqrt(expansion)

    def beam_radius_um_at(self, position_m: float) -> float:
        """返回 L1 运输轴任意位置的 1/e² 光强半径 ``w``。"""
        position = float(position_m)
        if not math.isfinite(position) or not 0.0 <= position <= self.distance_m:
            raise ValueError("L1 光束位置必须位于 [0, distance_m]")
        if not self.calibrated_gaussian_geometry:
            return self.start_waist_um + (
                self.handover_waist_um - self.start_waist_um
            ) * position / self.distance_m
        assert self.minimum_waist_um is not None
        assert self.minimum_waist_position_m is not None
        rayleigh_m = self.effective_rayleigh_range_m
        assert rayleigh_m is not None
        return self.minimum_waist_um * math.sqrt(
            1.0
            + ((position - self.minimum_waist_position_m) / rayleigh_m) ** 2
        )

    def beam_diameter_um_at(self, position_m: float) -> float:
        """返回 L1 运输轴任意位置的 1/e² 光强直径 ``2w``。"""
        return 2.0 * self.beam_radius_um_at(position_m)

    def __post_init__(self) -> None:
        _atom_from_label(self.atom_label)
        if (self.minimum_waist_um is None) != (
            self.minimum_waist_position_m is None
        ):
            raise ValueError("L1 最小束腰大小和位置必须同时提供或同时省略")
        if self.calibrated_gaussian_geometry:
            assert self.minimum_waist_um is not None
            assert self.minimum_waist_position_m is not None
            if (
                not math.isfinite(self.minimum_waist_um)
                or self.minimum_waist_um <= 0.0
            ):
                raise ValueError("L1 最小束腰必须是有限正数")
            if (
                not math.isfinite(self.minimum_waist_position_m)
                or not 0.0 < self.minimum_waist_position_m < self.distance_m
            ):
                raise ValueError("L1 最小束腰位置必须严格位于运输区间内")
            if self.start_waist_um <= self.minimum_waist_um:
                raise ValueError("L1 起点光束半径必须大于最小束腰半径")
            object.__setattr__(
                self,
                "handover_waist_um",
                self.beam_radius_um_at(self.distance_m),
            )
        positive = {
            "最小失谐": self.detuning_min_ghz,
            "最大失谐": self.detuning_max_ghz,
            "最大功率": self.handover_source_power_max_w,
            "MOT 原子数": self.mot_atom_number,
            "L1 初始原子数": self.initial_atom_number,
            "初始温度": self.initial_temperature_uK,
            "占据格点数": self.occupied_lattice_sites,
            "运输距离": self.distance_m,
            "加速度": self.acceleration_m_s2,
            "最大速度": self.maximum_velocity_m_s,
            "起点束腰": self.start_waist_um,
            "终点束腰": self.handover_waist_um,
            "目标阱深": self.target_depth_uK,
            "L1 最大源端功率": self.maximum_l1_source_power_w,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name}必须是有限正数")
        if self.detuning_max_ghz <= self.detuning_min_ghz:
            raise ValueError("最大失谐必须大于最小失谐")
        if self.handover_source_power_min_w < 0.0:
            raise ValueError("最小功率不能小于零")
        if self.handover_source_power_max_w <= self.handover_source_power_min_w:
            raise ValueError("最大功率必须大于最小功率")
        if min(self.detuning_points, self.power_points) < 2:
            raise ValueError("失谐和功率网格点数至少为 2")
        if self.time_points < 4:
            raise ValueError("时间点数至少为 4")
        if not 0.0 < self.pre_ramp_survival_fraction <= 1.0:
            raise ValueError("MOT/compress/idle → L1 起点存活率必须位于 (0,1]")
        if self.kinematic_profile not in {"trapezoid", "minimum_jerk"}:
            raise ValueError("理想运输轨迹必须是 trapezoid 或 minimum_jerk")
        distance_factor = (
            1.875 if self.kinematic_profile == "minimum_jerk" else 1.0
        )
        if (
            self.control_waveform is None
            and distance_factor * self.maximum_velocity_m_s**2
            > self.acceleration_m_s2 * self.distance_m
        ):
            suffix = (
                "（minimum_jerk 轨迹需要更长加速距离）"
                if distance_factor > 1.0
                else ""
            )
            raise ValueError(
                "运输距离不足以在给定加速度下达到指定速度" + suffix
            )
        if self.control_waveform is not None:
            if self.conveyor_enabled:
                raise ValueError(
                    "实测运输波形暂不与 offset-waist conveyor 几何叠加；"
                    "请二选一以避免重复定义束腰/光功率"
                )
            tolerance = max(1e-6, 5e-3 * self.distance_m)
            if abs(self.control_waveform.distance_m - self.distance_m) > tolerance:
                raise ValueError(
                    "实测运输波形终点与配置运输距离不一致："
                    f"{self.control_waveform.distance_m:.6g} m != "
                    f"{self.distance_m:.6g} m"
                )
        if not 0.0 < self.delivery_efficiency <= 1.0:
            raise ValueError("传输效率必须位于 (0,1]")
        if not 0.0 <= self.retro_power_ratio <= 1.0:
            raise ValueError("回程功率比必须位于 [0,1]")
        if not math.isfinite(self.conveyor_waist_um) or self.conveyor_waist_um <= 0.0:
            raise ValueError("conveyor 束腰必须是有限正数")
        if (
            not math.isfinite(self.conveyor_waist_separation_cm)
            or self.conveyor_waist_separation_cm < 0.0
        ):
            raise ValueError("conveyor 束腰间距必须是有限非负数")
        nonnegative = {
            "背景损失率": self.background_loss_rate_s,
            "每次散射内态损失概率": self.internal_loss_probability_per_scatter,
            "二体损失系数": self.two_body_loss_coefficient_m3_s,
            "三体损失系数": self.three_body_loss_coefficient_m6_s,
            "强度噪声 PSD": self.intensity_noise_psd_1_hz,
            "位置噪声 PSD": self.position_noise_psd_m2_hz,
            "温度权重": self.temperature_weight,
            "留存权重": self.retention_weight,
        }
        for name, value in nonnegative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name}必须是有限非负数")
        if self.internal_loss_probability_per_scatter > 1.0:
            raise ValueError("每次散射内态损失概率不能大于 1")
        if self.temperature_weight + self.retention_weight <= 0.0:
            raise ValueError("工作点选择权重之和必须为正")
        if self.transport_method not in {"analytic", "monte_carlo"}:
            raise ValueError("运输方法必须是 analytic 或 monte_carlo")
        if (
            not math.isfinite(self.transport_time_step_us)
            or self.transport_time_step_us <= 0.0
        ):
            raise ValueError("运输 MC 时间步长必须是有限正数")
        if self.mc_particle_count <= 0:
            raise ValueError("运输 MC 粒子数必须是正整数")
        if self.mc_compute_backend not in {"cpu", "gpu"}:
            raise ValueError("计算后端必须是 cpu 或 gpu")
        if (
            not math.isfinite(self.mc_cloud_axial_sigma_mm)
            or self.mc_cloud_axial_sigma_mm < 0.0
        ):
            raise ValueError("运输 MC 原子云轴向尺寸必须是有限非负数")

    @property
    def loading_efficiency(self) -> float:
        return self.initial_atom_number / self.mot_atom_number


@dataclass(frozen=True)
class L1Timing:
    acceleration_time_s: float
    cruise_time_s: float
    total_time_s: float
    maximum_velocity_m_s: float


@dataclass(frozen=True)
class L1DesignPoint:
    detuning_ghz: float
    handover_source_power_w: float
    start_source_power_w: float
    wavelength_nm: float
    depth_uK: float
    scattering_rate_s: float
    final_temperature_uK: float
    final_temperature_rise_uK: float
    final_retention_fraction: float
    total_retention_from_mot_fraction: float
    final_atom_number: float
    cumulative_scattering_events: float
    maximum_loss_rate_s: float
    feasible_hardware_point: bool
    quality_cost: float | None = None
    # 该点实际使用的 L1 初态（静止 L1 晶格热平衡图景，与
    # L1TransportInputs 相同）；旧版本反序列化缺失时下游回退到
    # inputs 的固定初态。loading_capture_efficiency /
    # loading_temperature_interface 已 deprecated：LGM 装载模块已移除，
    # 两字段恒为 None，仅为 UI/序列化消费者暂时保留。
    initial_temperature_uK: float | None = None
    initial_atom_number: float | None = None
    loading_capture_efficiency: float | None = None
    loading_temperature_interface: str | None = None
    # 运输腿 Monte Carlo 实际积分步长（经 _stable_leg_step_s 精度守卫
    # 钳制后的值）；解析腿无积分步长，恒为 None。
    actual_time_step_us: float | None = None


@dataclass(frozen=True)
class L1ReferencePoint:
    """论文实用点或 Cs 工程参考点及其功率定义。"""

    label: str
    note: str
    point: L1DesignPoint


@dataclass(frozen=True)
class L1TransportTrace:
    point: L1DesignPoint
    time_ms: tuple[float, ...]
    stage: tuple[str, ...]
    position_m: tuple[float, ...]
    velocity_m_s: tuple[float, ...]
    acceleration_m_s2: tuple[float, ...]
    aom_frequency_difference_mhz: tuple[float, ...]
    waist_um: tuple[float, ...]
    source_power_w: tuple[float, ...]
    effective_barrier_uK: tuple[float, ...]
    temperature_uK: tuple[float, ...]
    temperature_rise_uK: tuple[float, ...]
    retention_fraction: tuple[float, ...]
    bound_fraction: tuple[float, ...]
    cumulative_scattering_events: tuple[float, ...]
    instantaneous_loss_rate_s: tuple[float, ...]
    retention_standard_error: float | None = None
    # loading_trace 已 deprecated：LGM 装载模块已移除，恒为 None，
    # 仅为 UI（时间线/单点页）与序列化消费者暂时保留。
    loading_trace: object | None = None
    pre_ramp_survival_fraction: float = 1.0
    calculation_boundary: str = "static_lattice_thermal"

    @property
    def beam_diameter_um(self) -> tuple[float, ...]:
        """逐采样点的 1/e² 光强直径 ``2w``。"""
        return tuple(2.0 * value for value in self.waist_um)


@dataclass(frozen=True)
class L1TransportScanResult:
    inputs: L1TransportInputs
    timing: L1Timing
    detuning_ghz: tuple[float, ...]
    handover_source_power_w: tuple[float, ...]
    final_temperature_rise_uK: tuple[tuple[float, ...], ...]
    final_retention_fraction: tuple[tuple[float, ...], ...]
    feasible_hardware_point: tuple[tuple[bool, ...], ...]
    quality_cost: tuple[tuple[float | None, ...], ...]
    reference_points: tuple[L1ReferencePoint, ...]
    optimal: L1DesignPoint
    comparison: L1DesignPoint
    optimal_trace: L1TransportTrace
    comparison_trace: L1TransportTrace


def _atom_from_label(label: str) -> AlkaliAtom:
    normalized = label.strip().lower().replace("-", "").replace("_", "")
    if normalized in {"rb", "rb87", "87rb"}:
        return RB87
    if normalized in {"cs", "cs133", "133cs"}:
        return CS133
    raise ValueError("原子必须是 Rb-87 或 Cs-133")


def l1_transport_inputs_for_species(atom_label: str) -> L1TransportInputs:
    """返回带有物种专用初温和光路效率的 L1 默认输入。"""
    canonical_label = _atom_from_label(atom_label).label
    raw_profile = _SPECIES_DEFAULTS.get(canonical_label)
    if not isinstance(raw_profile, dict):
        raise ValueError(f"L1 配置缺少 {canonical_label} 物种默认值")
    return replace(
        L1TransportInputs(),
        atom_label=canonical_label,
        delivery_efficiency=float(raw_profile["delivery_efficiency"]),
    )


def l1_timing(inputs: L1TransportInputs) -> L1Timing:
    """由固定加速度、速度和距离生成理想运输时序。"""
    if inputs.control_waveform is not None:
        return L1Timing(
            acceleration_time_s=0.0,
            cruise_time_s=inputs.control_waveform.duration_ms * 1e-3,
            total_time_s=inputs.control_waveform.duration_ms * 1e-3,
            maximum_velocity_m_s=inputs.control_waveform.maximum_velocity_m_s,
        )
    acceleration_time = inputs.maximum_velocity_m_s / inputs.acceleration_m_s2
    if inputs.kinematic_profile == "minimum_jerk":
        # v=v_max(10u^3-15u^4+6u^5) 的峰值加速度为 1.875*v_max/T。
        acceleration_time *= 1.875
    cruise_distance = (
        inputs.distance_m
        - inputs.maximum_velocity_m_s * acceleration_time
    )
    cruise_time = cruise_distance / inputs.maximum_velocity_m_s
    return L1Timing(
        acceleration_time_s=acceleration_time,
        cruise_time_s=cruise_time,
        total_time_s=2.0 * acceleration_time + cruise_time,
        maximum_velocity_m_s=inputs.maximum_velocity_m_s,
    )


def _kinematics(
    time_s: float,
    inputs: L1TransportInputs,
    timing: L1Timing,
) -> tuple[float, float, float, str]:
    if inputs.control_waveform is not None:
        sample = inputs.control_waveform.sample(time_s)
        acceleration = float(sample["acceleration_m_s2"])
        if time_s >= timing.total_time_s:
            stage = "arrived"
        elif acceleration > 1e-9:
            stage = "acceleration"
        elif acceleration < -1e-9:
            stage = "deceleration"
        else:
            stage = "measured transport"
        return (
            float(sample["position_m"]),
            float(sample["velocity_m_s"]),
            acceleration,
            stage,
        )
    if inputs.kinematic_profile == "minimum_jerk":
        ta = timing.acceleration_time_s
        cruise_end = ta + timing.cruise_time_s
        total = timing.total_time_s
        velocity_max = inputs.maximum_velocity_m_s

        def smooth_launch(elapsed_s: float) -> tuple[float, float, float]:
            u = min(1.0, max(0.0, elapsed_s / ta))
            u2 = u * u
            u3 = u2 * u
            u4 = u3 * u
            u5 = u4 * u
            velocity_shape = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
            acceleration_shape = 30.0 * u2 - 60.0 * u3 + 30.0 * u4
            position_shape = 2.5 * u4 - 3.0 * u5 + u5 * u
            return (
                velocity_max * ta * position_shape,
                velocity_max * velocity_shape,
                velocity_max / ta * acceleration_shape,
            )

        if time_s < ta:
            position, velocity, acceleration = smooth_launch(time_s)
            return position, velocity, acceleration, "smooth acceleration"
        acceleration_distance = 0.5 * velocity_max * ta
        if time_s < cruise_end:
            return (
                acceleration_distance + velocity_max * (time_s - ta),
                velocity_max,
                0.0,
                "cruise",
            )
        if time_s < total:
            remaining = total - time_s
            distance, velocity, acceleration = smooth_launch(remaining)
            return (
                inputs.distance_m - distance,
                velocity,
                -acceleration,
                "smooth deceleration",
            )
        return inputs.distance_m, 0.0, 0.0, "arrived"
    ta = timing.acceleration_time_s
    cruise_end = ta + timing.cruise_time_s
    total = timing.total_time_s
    acceleration = inputs.acceleration_m_s2
    velocity_max = inputs.maximum_velocity_m_s
    if time_s < ta:
        return (
            0.5 * acceleration * time_s**2,
            acceleration * time_s,
            acceleration,
            "acceleration",
        )
    acceleration_distance = 0.5 * acceleration * ta**2
    if time_s < cruise_end:
        return (
            acceleration_distance + velocity_max * (time_s - ta),
            velocity_max,
            0.0,
            "cruise",
        )
    if time_s < total:
        remaining = total - time_s
        return (
            inputs.distance_m - 0.5 * acceleration * remaining**2,
            acceleration * remaining,
            -acceleration,
            "deceleration",
        )
    return inputs.distance_m, 0.0, 0.0, "arrived"


def _time_grid(inputs: L1TransportInputs, timing: L1Timing) -> np.ndarray:
    if inputs.control_waveform is not None:
        measured_times = np.asarray(inputs.control_waveform.time_ms) * 1e-3
        return np.unique(
            np.concatenate(
                (
                    np.linspace(0.0, timing.total_time_s, inputs.time_points),
                    measured_times,
                )
            )
        )
    boundaries = np.array(
        [
            0.0,
            timing.acceleration_time_s,
            timing.acceleration_time_s + timing.cruise_time_s,
            timing.total_time_s,
        ]
    )
    return np.unique(
        np.concatenate(
            (
                np.linspace(0.0, timing.total_time_s, inputs.time_points),
                boundaries,
            )
        )
    )


def _peak_density_m3(
    atom_number: float,
    occupied_sites: float,
    radial_frequency_hz: float,
    axial_frequency_hz: float,
    atom_mass_kg: float,
    temperature_uK: float,
) -> float:
    omega_r = 2.0 * math.pi * radial_frequency_hz
    omega_z = 2.0 * math.pi * axial_frequency_hz
    temperature_k = max(temperature_uK, 1e-12) * 1e-6
    return (
        atom_number
        / occupied_sites
        * omega_r**2
        * omega_z
        * (
            atom_mass_kg
            / (2.0 * math.pi * BOLTZMANN * temperature_k)
        )
        ** 1.5
    )


def simulate_l1_transport(
    inputs: L1TransportInputs,
    detuning_ghz: float,
    handover_source_power_w: float,
) -> L1TransportTrace:
    """积分一个失谐--功率点的宏观温度和留存率。

    初态为固定 (N, T)——静止 L1 晶格热平衡图景（LGM 装载模块已移除，
    不再有逐点初态替换分支）。
    """
    if inputs.transport_method == "monte_carlo":
        # 轨迹级 Monte Carlo 腿（局部 import 避免循环导入）。
        from .transport_mc import simulate_leg_monte_carlo

        trace = simulate_leg_monte_carlo(
            inputs,
            detuning_ghz,
            handover_source_power_w,
        )
        return replace(
            trace,
            pre_ramp_survival_fraction=inputs.pre_ramp_survival_fraction,
        )
    atom = _atom_from_label(inputs.atom_label)
    timing = l1_timing(inputs)
    wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning_ghz)
    lattice = evaluate_lattice(
        atom,
        wavelength_nm,
        forward_power_w=(
            handover_source_power_w * inputs.delivery_efficiency
        ),
        waist_um=inputs.handover_waist_um,
        retro_power_ratio=inputs.retro_power_ratio,
    )
    if inputs.control_waveform is not None:
        end_control = inputs.control_waveform.sample(timing.total_time_s)
        end_waist = (
            inputs.handover_waist_um
            if end_control["waist_um"] is None
            else float(end_control["waist_um"])
        )
        end_source = handover_source_power_w * (
            1.0
            if end_control["source_power_scale"] is None
            else float(end_control["source_power_scale"])
        )
        end_delivery = inputs.delivery_efficiency * (
            1.0
            if end_control["delivery_efficiency_scale"] is None
            else float(end_control["delivery_efficiency_scale"])
        )
        lattice = evaluate_lattice(
            atom,
            wavelength_nm,
            forward_power_w=end_source * end_delivery,
            waist_um=end_waist,
            retro_power_ratio=inputs.retro_power_ratio,
        )
    if inputs.conveyor_enabled:
        # offset-waist 几何：每分支源端功率全程恒定（跳过 w² 跟随），
        # 阱深随几何起伏；handover 点量取 z=L 处的剖面值。
        forward_power_w = handover_source_power_w * inputs.delivery_efficiency

        def _local_geometry(position_m: float) -> ConveyorPoint:
            return conveyor_point(
                atom,
                wavelength_nm,
                forward_power_w,
                inputs.conveyor_waist_um,
                inputs.conveyor_waist_separation_cm,
                inputs.distance_m,
                position_m,
                inputs.retro_power_ratio,
            )

        profile = conveyor_profile(
            atom,
            wavelength_nm,
            forward_power_w,
            inputs.conveyor_waist_um,
            inputs.conveyor_waist_separation_cm,
            inputs.distance_m,
            np.linspace(0.0, inputs.distance_m, 401),
            inputs.retro_power_ratio,
        )
        end_point = _local_geometry(inputs.distance_m)
        start_source_power = handover_source_power_w
        handover_depth_uK = end_point.depth_uK
        handover_scattering_rate_s = end_point.scattering_rate_s
        minimum_critical_acceleration = (
            profile.minimum_critical_acceleration_m_s2
        )
    else:
        if inputs.control_waveform is None:
            start_source_power = handover_source_power_w
            start_delivery = inputs.delivery_efficiency
            start_waist_um = inputs.beam_radius_um_at(0.0)
        else:
            start_control = inputs.control_waveform.sample(0.0)
            start_source_power = handover_source_power_w * (
                1.0 if start_control["source_power_scale"] is None
                else float(start_control["source_power_scale"])
            )
            start_delivery = inputs.delivery_efficiency * (
                1.0 if start_control["delivery_efficiency_scale"] is None
                else float(start_control["delivery_efficiency_scale"])
            )
            start_waist_um = (
                inputs.start_waist_um
                if start_control["waist_um"] is None
                else float(start_control["waist_um"])
            )
        start_lattice = evaluate_lattice(
            atom,
            wavelength_nm,
            forward_power_w=start_source_power * start_delivery,
            waist_um=start_waist_um,
            retro_power_ratio=inputs.retro_power_ratio,
        )
        handover_depth_uK = lattice.depth_uK
        handover_scattering_rate_s = lattice.scattering_rate_s
        if inputs.control_waveform is None:
            weakest_waist_um = max(
                inputs.beam_radius_um_at(0.0),
                inputs.beam_radius_um_at(inputs.distance_m),
            )
            weakest_lattice = evaluate_lattice(
                atom,
                wavelength_nm,
                forward_power_w=(
                    handover_source_power_w * inputs.delivery_efficiency
                ),
                waist_um=weakest_waist_um,
                retro_power_ratio=inputs.retro_power_ratio,
            )
            minimum_critical_acceleration = (
                weakest_lattice.critical_axial_acceleration_m_s2
            )
        else:
            minimum_critical_acceleration = (
                lattice.critical_axial_acceleration_m_s2
            )
    feasible = (
        (
            not inputs.require_minimum_depth
            or handover_depth_uK >= inputs.target_depth_uK
        )
        and (
            not inputs.require_maximum_start_power
            or start_source_power <= inputs.maximum_l1_source_power_w
        )
        and (
            not inputs.require_critical_acceleration
            or minimum_critical_acceleration
            > inputs.acceleration_m_s2
        )
    )
    times = _time_grid(inputs, timing)
    gravity_barrier_trace_uK: np.ndarray | None = None
    gravity_kinematics: list[tuple[float, float, float, str]] | None = None
    if (
        inputs.include_gravity
        and not inputs.conveyor_enabled
        and inputs.control_waveform is None
    ):
        gravity_kinematics = [
            _kinematics(float(value), inputs, timing) for value in times
        ]
        gravity_positions = np.asarray(
            [state[0] for state in gravity_kinematics]
        )
        gravity_waists_m = np.asarray(
            [
                inputs.beam_radius_um_at(float(position))
                for position in gravity_positions
            ]
        ) * 1e-6
        gravity_depths_j = (
            lattice.depth_uK
            * (inputs.handover_waist_um * 1e-6 / gravity_waists_m) ** 2
            * 1e-6
            * BOLTZMANN
        )
        gravity_barrier_trace_uK = (
            gaussian_gravity_barriers_j(
                gravity_depths_j,
                gravity_waists_m,
                atom.mass_kg,
            )
            / BOLTZMANN
            * 1e6
        )
    jump_times_and_sizes = (
        ()
        if (
            inputs.control_waveform is not None
            or inputs.kinematic_profile == "minimum_jerk"
        )
        else (
            (0.0, inputs.acceleration_m_s2),
            (timing.acceleration_time_s, -inputs.acceleration_m_s2),
            (
                timing.acceleration_time_s + timing.cruise_time_s,
                -inputs.acceleration_m_s2,
            ),
            (timing.total_time_s, inputs.acceleration_m_s2),
        )
    )
    if inputs.conveyor_enabled:
        # 加速度跳变升温用跳变时刻的局部轴向阱频。
        def jump_temperature(delta_a: float, time_s: float) -> float:
            jump_position = _kinematics(time_s, inputs, timing)[0]
            local_omega = (
                2.0 * math.pi * _local_geometry(jump_position).axial_frequency_hz
            )
            return (
                atom.mass_kg * delta_a**2 / (6.0 * BOLTZMANN * local_omega**2) * 1e6
            )

    else:
        def jump_temperature(delta_a: float, time_s: float) -> float:
            jump_position = _kinematics(time_s, inputs, timing)[0]
            jump_lattice = evaluate_lattice(
                atom,
                wavelength_nm,
                forward_power_w=(
                    handover_source_power_w * inputs.delivery_efficiency
                ),
                waist_um=inputs.beam_radius_um_at(jump_position),
                retro_power_ratio=inputs.retro_power_ratio,
            )
            axial_omega = 2.0 * math.pi * jump_lattice.axial_frequency_hz
            return (
                atom.mass_kg * delta_a**2
                / (6.0 * BOLTZMANN * axial_omega**2)
                * 1e6
            )

    if inputs.conveyor_enabled:
        # 参考束缚比例用起点（z=0）的静态最小逃逸势垒。
        start_local = _local_geometry(0.0)
        static_depth_uK = start_local.axial_depth_uK
        static_radial_depth_uK = start_local.depth_uK
        static_waist_um = start_local.effective_waist_um
    else:
        static_depth_uK = start_lattice.depth_uK
        static_radial_depth_uK = start_lattice.depth_uK
        static_waist_um = start_waist_um
    if inputs.include_gravity:
        gravity_barrier_j, _, _ = gaussian_gravity_trap(
            static_radial_depth_uK * 1e-6 * BOLTZMANN,
            static_waist_um * 1e-6,
            atom.mass_kg,
        )
        static_depth_uK = min(
            static_depth_uK,
            gravity_barrier_j / BOLTZMANN * 1e6,
        )
    static_bound = thermal_bound_fraction_3d_harmonic(
        static_depth_uK,
        inputs.initial_temperature_uK,
    )
    static_bound = max(static_bound, 1e-300)

    temperature = inputs.initial_temperature_uK
    if (
        inputs.control_waveform is None
        and inputs.kinematic_profile == "trapezoid"
    ):
        temperature += jump_temperature(inputs.acceleration_m_s2, 0.0)
    rate_survival = 1.0
    spill_survival = 1.0
    cumulative_scattering = 0.0
    current_atom_number = inputs.initial_atom_number

    time_out: list[float] = []
    stage_out: list[str] = []
    position_out: list[float] = []
    velocity_out: list[float] = []
    acceleration_out: list[float] = []
    frequency_out: list[float] = []
    waist_out: list[float] = []
    power_out: list[float] = []
    barrier_out: list[float] = []
    temperature_out: list[float] = []
    retention_out: list[float] = []
    bound_out: list[float] = []
    scattering_out: list[float] = []
    loss_rate_out: list[float] = []

    previous_time = 0.0
    previous_bar_frequency: float | None = None
    maximum_loss_rate = 0.0
    applied_jump_times = {0.0}
    for time_index, time_s in enumerate(times):
        if gravity_kinematics is None:
            position, velocity, acceleration, stage = _kinematics(
                float(time_s), inputs, timing
            )
        else:
            position, velocity, acceleration, stage = gravity_kinematics[
                time_index
            ]
        if inputs.conveyor_enabled:
            local = _local_geometry(position)
            waist = local.effective_waist_um
            source_power = handover_source_power_w
            radial_frequency = local.radial_frequency_hz
            axial_frequency = local.axial_frequency_hz
            scattering_rate = local.scattering_rate_s
            barrier_depth_uK = local.axial_depth_uK
            radial_depth_uK = local.depth_uK
            critical_acceleration = local.critical_axial_acceleration_m_s2
        else:
            control = (
                None
                if inputs.control_waveform is None
                else inputs.control_waveform.sample(float(time_s))
            )
            geometric_waist = inputs.beam_radius_um_at(position)
            waist = (
                geometric_waist
                if control is None or control["waist_um"] is None
                else float(control["waist_um"])
            )
            source_power = (
                handover_source_power_w
                if control is None or control["source_power_scale"] is None
                else handover_source_power_w
                * float(control["source_power_scale"])
            )
            delivery = inputs.delivery_efficiency * (
                1.0
                if control is None
                or control["delivery_efficiency_scale"] is None
                else float(control["delivery_efficiency_scale"])
            )
            local_lattice = evaluate_lattice(
                atom,
                wavelength_nm,
                forward_power_w=source_power * delivery,
                waist_um=waist,
                retro_power_ratio=inputs.retro_power_ratio,
            )
            radial_frequency = local_lattice.radial_frequency_hz
            axial_frequency = local_lattice.axial_frequency_hz
            scattering_rate = local_lattice.scattering_rate_s
            barrier_depth_uK = local_lattice.depth_uK
            radial_depth_uK = local_lattice.depth_uK
            critical_acceleration = (
                local_lattice.critical_axial_acceleration_m_s2
            )
        bar_frequency = (
            radial_frequency**2 * axial_frequency
        ) ** (1.0 / 3.0)
        if previous_bar_frequency is None:
            previous_bar_frequency = bar_frequency
        delta_t = float(time_s) - previous_time
        if delta_t > 0.0:
            temperature *= bar_frequency / previous_bar_frequency
            bar_omega = 2.0 * math.pi * bar_frequency
            parametric_rate = (
                bar_omega**2 * inputs.intensity_noise_psd_1_hz / 4.0
            )
            temperature *= math.exp(parametric_rate * delta_t)
            position_heating_rate = (
                atom.mass_kg
                * bar_omega**4
                * inputs.position_noise_psd_m2_hz
                / (12.0 * BOLTZMANN)
                * 1e6
            )
            recoil_heating_rate = (
                2.0 * lattice.recoil_temperature_uK / 3.0
                * scattering_rate
            )
            temperature += (
                recoil_heating_rate + position_heating_rate
            ) * delta_t
            for jump_time, delta_a in jump_times_and_sizes:
                if (
                    previous_time < jump_time <= time_s + 1e-15
                    and jump_time not in applied_jump_times
                ):
                    temperature += jump_temperature(delta_a, jump_time)
                    applied_jump_times.add(jump_time)
            cumulative_scattering += scattering_rate * delta_t

        barrier_fraction = tilted_lattice_barrier_fraction(
            acceleration,
            critical_acceleration,
        )
        effective_barrier = barrier_depth_uK * barrier_fraction
        if inputs.include_gravity:
            if gravity_barrier_trace_uK is not None:
                gravity_barrier_uK = float(
                    gravity_barrier_trace_uK[time_index]
                )
            else:
                gravity_barrier_j, _, _ = gaussian_gravity_trap(
                    radial_depth_uK * 1e-6 * BOLTZMANN,
                    waist * 1e-6,
                    atom.mass_kg,
                )
                gravity_barrier_uK = gravity_barrier_j / BOLTZMANN * 1e6
            effective_barrier = min(effective_barrier, gravity_barrier_uK)
        bound_fraction = thermal_bound_fraction_3d_harmonic(
            effective_barrier,
            temperature,
        )
        spill_survival = min(
            spill_survival,
            min(1.0, bound_fraction / static_bound),
        )
        peak_density = _peak_density_m3(
            current_atom_number,
            inputs.occupied_lattice_sites,
            radial_frequency,
            axial_frequency,
            atom.mass_kg,
            temperature,
        )
        loss_rate = (
            inputs.background_loss_rate_s
            + inputs.internal_loss_probability_per_scatter
            * scattering_rate
            + inputs.two_body_loss_coefficient_m3_s
            * peak_density
            / (2.0**1.5)
            + inputs.three_body_loss_coefficient_m6_s
            * peak_density**2
            / (3.0**1.5)
        )
        if delta_t > 0.0:
            rate_survival *= math.exp(-loss_rate * delta_t)
        retention = max(0.0, min(1.0, rate_survival * spill_survival))
        current_atom_number = inputs.initial_atom_number * retention
        maximum_loss_rate = max(maximum_loss_rate, loss_rate)

        time_out.append(float(time_s) * 1e3)
        stage_out.append(stage)
        position_out.append(position)
        velocity_out.append(velocity)
        acceleration_out.append(acceleration)
        frequency_out.append(
            (
                2.0 * velocity / (wavelength_nm * 1e-9) * 1e-6
                if inputs.control_waveform is None
                else float(
                    inputs.control_waveform.sample(float(time_s))[
                        "aom_frequency_difference_mhz"
                    ]
                )
            )
        )
        waist_out.append(waist)
        power_out.append(source_power)
        barrier_out.append(effective_barrier)
        temperature_out.append(temperature)
        retention_out.append(retention)
        bound_out.append(bound_fraction)
        scattering_out.append(cumulative_scattering)
        loss_rate_out.append(loss_rate)
        previous_time = float(time_s)
        previous_bar_frequency = bar_frequency

    final_temperature = temperature_out[-1]
    final_retention = retention_out[-1]
    point = L1DesignPoint(
        detuning_ghz=detuning_ghz,
        handover_source_power_w=handover_source_power_w,
        start_source_power_w=start_source_power,
        wavelength_nm=wavelength_nm,
        depth_uK=handover_depth_uK,
        scattering_rate_s=handover_scattering_rate_s,
        final_temperature_uK=final_temperature,
        final_temperature_rise_uK=(
            final_temperature - inputs.initial_temperature_uK
        ),
        final_retention_fraction=final_retention,
        total_retention_from_mot_fraction=(
            inputs.loading_efficiency * final_retention
        ),
        final_atom_number=inputs.initial_atom_number * final_retention,
        cumulative_scattering_events=scattering_out[-1],
        maximum_loss_rate_s=maximum_loss_rate,
        feasible_hardware_point=feasible,
        initial_temperature_uK=inputs.initial_temperature_uK,
        initial_atom_number=inputs.initial_atom_number,
    )
    return L1TransportTrace(
        point=point,
        time_ms=tuple(time_out),
        stage=tuple(stage_out),
        position_m=tuple(position_out),
        velocity_m_s=tuple(velocity_out),
        acceleration_m_s2=tuple(acceleration_out),
        aom_frequency_difference_mhz=tuple(frequency_out),
        waist_um=tuple(waist_out),
        source_power_w=tuple(power_out),
        effective_barrier_uK=tuple(barrier_out),
        temperature_uK=tuple(temperature_out),
        temperature_rise_uK=tuple(
            value - inputs.initial_temperature_uK for value in temperature_out
        ),
        retention_fraction=tuple(retention_out),
        bound_fraction=tuple(bound_out),
        cumulative_scattering_events=tuple(scattering_out),
        instantaneous_loss_rate_s=tuple(loss_rate_out),
        pre_ramp_survival_fraction=inputs.pre_ramp_survival_fraction,
    )


def _normalized(values: np.ndarray) -> np.ndarray:
    span = float(np.max(values) - np.min(values))
    if span <= 1e-15:
        return np.zeros_like(values)
    return (values - np.min(values)) / span


def _failed_l1_transport_trace(
    inputs: L1TransportInputs,
    detuning_ghz: float,
    handover_source_power_w: float,
    timing: L1Timing,
) -> L1TransportTrace:
    """构造一个“全部原子在该点丢失”的哨兵 trace。

    用于二维扫描中某个网格点因数值/采样问题无法继续时，仍保留一个
    可绘制的轨迹骨架（温度 NaN、留存 0、硬件可行 False），避免整个
    扫描因单点异常中断。
    """
    nan = float("nan")
    atom = _atom_from_label(inputs.atom_label)
    wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning_ghz)
    time_out: list[float] = []
    stage_out: list[str] = []
    position_out: list[float] = []
    velocity_out: list[float] = []
    acceleration_out: list[float] = []
    frequency_out: list[float] = []
    count = 0
    for grid_time_s in _time_grid(inputs, timing):
        position, velocity, acceleration, stage = _kinematics(
            min(float(grid_time_s), timing.total_time_s),
            inputs,
            timing,
        )
        time_out.append(float(grid_time_s) * 1e3)
        stage_out.append(stage)
        position_out.append(float(position))
        velocity_out.append(float(velocity))
        acceleration_out.append(float(acceleration))
        frequency_out.append(
            2.0 * float(velocity) / (wavelength_nm * 1e-9) * 1e-6
        )
        count += 1
    particle_count = max(1, inputs.mc_particle_count)
    standard_error = math.sqrt(
        (0.5 * (particle_count + 0.5))
        / ((particle_count + 1.0) ** 2 * (particle_count + 2.0))
    )
    return L1TransportTrace(
        point=L1DesignPoint(
            detuning_ghz=detuning_ghz,
            handover_source_power_w=handover_source_power_w,
            start_source_power_w=nan,
            wavelength_nm=wavelength_nm,
            depth_uK=nan,
            scattering_rate_s=nan,
            final_temperature_uK=nan,
            final_temperature_rise_uK=nan,
            final_retention_fraction=0.0,
            total_retention_from_mot_fraction=0.0,
            final_atom_number=0.0,
            cumulative_scattering_events=0.0,
            maximum_loss_rate_s=0.0,
            feasible_hardware_point=False,
            initial_temperature_uK=inputs.initial_temperature_uK,
            initial_atom_number=inputs.initial_atom_number,
        ),
        time_ms=tuple(time_out),
        stage=tuple(stage_out),
        position_m=tuple(position_out),
        velocity_m_s=tuple(velocity_out),
        acceleration_m_s2=tuple(acceleration_out),
        aom_frequency_difference_mhz=tuple(frequency_out),
        waist_um=(nan,) * count,
        source_power_w=(nan,) * count,
        effective_barrier_uK=(nan,) * count,
        temperature_uK=(nan,) * count,
        temperature_rise_uK=(nan,) * count,
        retention_fraction=(0.0,) * count,
        bound_fraction=(0.0,) * count,
        cumulative_scattering_events=(0.0,) * count,
        instantaneous_loss_rate_s=(0.0,) * count,
        retention_standard_error=standard_error,
    )


def _reference_points(inputs: L1TransportInputs) -> tuple[L1ReferencePoint, ...]:
    raw_profile = _SPECIES_DEFAULTS.get(inputs.atom_label)
    if not isinstance(raw_profile, dict):
        return ()
    raw_reference = raw_profile.get("reference_point")
    if not isinstance(raw_reference, dict):
        return ()
    detuning_ghz = float(raw_reference["detuning_ghz"])
    if "handover_source_power_w" in raw_reference:
        source_power_w = float(raw_reference["handover_source_power_w"])
    else:
        target_depth_uK = float(raw_reference["target_depth_uK"])
        atom = _atom_from_label(inputs.atom_label)
        wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning_ghz)
        forward_power_w = power_for_target_depth(
            atom,
            wavelength_nm,
            target_depth_uK,
            inputs.handover_waist_um,
            inputs.retro_power_ratio,
        )
        source_power_w = forward_power_w / inputs.delivery_efficiency
    try:
        trace = simulate_l1_transport(inputs, detuning_ghz, source_power_w)
    except Exception:  # noqa: BLE001 - 参考点失败不应中断二维扫描
        return ()
    return (
        L1ReferencePoint(
            label=str(raw_reference["label"]),
            note=str(raw_reference.get("note", "")),
            point=trace.point,
        ),
    )


def analyze_l1_transport_scan(
    inputs: L1TransportInputs = L1TransportInputs(),
    *,
    progress: Callable[[str], None] | None = None,
) -> L1TransportScanResult:
    """扫描失谐--功率并选择最优和较差的可用工作点。"""
    detunings = np.linspace(
        inputs.detuning_min_ghz,
        inputs.detuning_max_ghz,
        inputs.detuning_points,
    )
    powers = np.linspace(
        inputs.handover_source_power_min_w,
        inputs.handover_source_power_max_w,
        inputs.power_points,
    )
    timing = l1_timing(inputs)
    traces: dict[tuple[int, int], L1TransportTrace] = {}
    heating = np.full(
        (inputs.power_points, inputs.detuning_points),
        np.nan,
    )
    retention = np.zeros_like(heating)
    feasible = np.zeros_like(heating, dtype=bool)
    for power_index, power in enumerate(powers):
        for detuning_index, detuning in enumerate(detunings):
            if power <= 0.0:
                continue
            try:
                trace = simulate_l1_transport(
                    inputs,
                    float(detuning),
                    float(power),
                )
            except Exception:  # noqa: BLE001 - 单点异常按该点原子全部丢失处理
                trace = _failed_l1_transport_trace(
                    inputs,
                    float(detuning),
                    float(power),
                    timing,
                )
            traces[(power_index, detuning_index)] = trace
            heating[power_index, detuning_index] = (
                trace.point.final_temperature_rise_uK
            )
            retention[power_index, detuning_index] = (
                trace.point.final_retention_fraction
            )
            feasible[power_index, detuning_index] = (
                trace.point.feasible_hardware_point
                and math.isfinite(trace.point.final_temperature_rise_uK)
                and trace.point.final_atom_number > 0.0
            )
        if progress is not None and (
            power_index + 1 == len(powers)
            or (power_index + 1) % max(1, len(powers) // 10) == 0
        ):
            progress(f"L1 二维扫描: {power_index + 1}/{len(powers)} 行功率")
    if not traces:
        # 极端情况：所有网格点都因异常/零功率跳过。仍构造一个哨兵轨迹，
        # 保证返回结果可 JSON 序列化、绘图不会因空 traces 崩溃。
        traces[(0, 0)] = _failed_l1_transport_trace(
            inputs,
            float(detunings[0]),
            float(powers[0]),
            timing,
        )
    feasible_indices = np.argwhere(feasible)
    if not len(feasible_indices):
        # 全网格无可行点：不抛错中断扫描，用第一个网格点的解析 trace 作
        # 哨兵最优/较差（quality_cost=NaN），热力图/时间曲线仍可绘制。
        if progress is not None:
            progress(
                "L1 二维扫描: 没有满足目标阱深和最大功率的工作点，"
                "返回空结果（quality_cost=NaN）"
            )
        fallback_index = next(iter(traces))
        fallback_trace = traces[fallback_index]
        fallback_point = replace(
            fallback_trace.point,
            quality_cost=float("nan"),
        )
        fallback_trace = replace(fallback_trace, point=fallback_point)
        cost = np.full_like(heating, np.nan)
        return L1TransportScanResult(
            inputs=inputs,
            timing=l1_timing(inputs),
            detuning_ghz=tuple(float(value) for value in detunings),
            handover_source_power_w=tuple(float(value) for value in powers),
            final_temperature_rise_uK=tuple(
                tuple(float(value) for value in row) for row in heating
            ),
            final_retention_fraction=tuple(
                tuple(float(value) for value in row) for row in retention
            ),
            feasible_hardware_point=tuple(
                tuple(bool(value) for value in row) for row in feasible
            ),
            quality_cost=tuple(
                tuple(None if math.isnan(value) else float(value) for value in row)
                for row in cost
            ),
            reference_points=_reference_points(inputs),
            optimal=fallback_point,
            comparison=fallback_point,
            optimal_trace=fallback_trace,
            comparison_trace=fallback_trace,
        )
    feasible_heating = heating[feasible]
    feasible_loss = 1.0 - retention[feasible]
    temperature_weight = inputs.temperature_weight / (
        inputs.temperature_weight + inputs.retention_weight
    )
    retention_weight = 1.0 - temperature_weight
    feasible_cost = (
        temperature_weight * _normalized(feasible_heating)
        + retention_weight * _normalized(feasible_loss)
    )
    cost = np.full_like(heating, np.nan)
    cost[feasible] = feasible_cost
    best_flat = int(np.argmin(feasible_cost))
    poor_flat = int(np.argmax(feasible_cost))
    best_index = tuple(int(value) for value in feasible_indices[best_flat])
    poor_index = tuple(int(value) for value in feasible_indices[poor_flat])
    best_trace = traces[best_index]
    poor_trace = traces[poor_index]
    best_point = replace(
        best_trace.point,
        quality_cost=float(feasible_cost[best_flat]),
    )
    poor_point = replace(
        poor_trace.point,
        quality_cost=float(feasible_cost[poor_flat]),
    )
    best_trace = replace(best_trace, point=best_point)
    poor_trace = replace(poor_trace, point=poor_point)
    return L1TransportScanResult(
        inputs=inputs,
        timing=l1_timing(inputs),
        detuning_ghz=tuple(float(value) for value in detunings),
        handover_source_power_w=tuple(float(value) for value in powers),
        final_temperature_rise_uK=tuple(
            tuple(float(value) for value in row) for row in heating
        ),
        final_retention_fraction=tuple(
            tuple(float(value) for value in row) for row in retention
        ),
        feasible_hardware_point=tuple(
            tuple(bool(value) for value in row) for row in feasible
        ),
        quality_cost=tuple(
            tuple(None if math.isnan(value) else float(value) for value in row)
            for row in cost
        ),
        reference_points=_reference_points(inputs),
        optimal=best_point,
        comparison=poor_point,
        optimal_trace=best_trace,
        comparison_trace=poor_trace,
    )


def plot_l1_transport_scan(
    result: L1TransportScanResult,
    output_path: str | Path,
) -> Path:
    """绘制两张参数热力图和两张最优/较差时间曲线。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(float(_PLOT["figure_width_in"]), float(_PLOT["figure_height_in"])),
        constrained_layout=True,
    )
    detunings = np.asarray(result.detuning_ghz)
    powers = np.asarray(result.handover_source_power_w)
    heating = np.asarray(result.final_temperature_rise_uK)
    retention = np.asarray(result.final_retention_fraction)
    feasible = np.asarray(result.feasible_hardware_point, dtype=float)
    heat_mesh = axes[0, 0].pcolormesh(
        detunings,
        powers,
        heating,
        shading="nearest",
        cmap=str(_PLOT["temperature_colormap"]),
    )
    retention_mesh = axes[0, 1].pcolormesh(
        detunings,
        powers,
        retention,
        shading="nearest",
        cmap=str(_PLOT["retention_colormap"]),
    )
    if np.any(feasible) and np.any(feasible < 1.0):
        for axis in axes[0]:
            axis.contour(
                detunings,
                powers,
                feasible,
                levels=(0.5,),
                colors=("white",),
                linewidths=(1.2,),
                linestyles=("--",),
            )
    for axis in axes[0]:
        axis.plot(
            result.optimal.detuning_ghz,
            result.optimal.handover_source_power_w,
            marker="*",
            color="#2563eb",
            markeredgecolor="white",
            markersize=14,
            label="Selected optimum",
        )
        axis.plot(
            result.comparison.detuning_ghz,
            result.comparison.handover_source_power_w,
            marker="X",
            color="#dc2626",
            markeredgecolor="white",
            markersize=9,
            label="Poor feasible comparison",
        )
        axis.set_xlabel("D1 red detuning (GHz)")
        axis.set_ylabel("Handover-end source power per branch (W)")
        axis.set_xlim(result.inputs.detuning_min_ghz, result.inputs.detuning_max_ghz)
        axis.set_ylim(
            result.inputs.handover_source_power_min_w,
            result.inputs.handover_source_power_max_w,
        )
        axis.legend(fontsize=8, loc="best")
    axes[0, 0].set_title("Final L1 temperature rise")
    axes[0, 1].set_title("Final L1 retention fraction")
    figure.colorbar(heat_mesh, ax=axes[0, 0], label="Temperature rise (µK)")
    figure.colorbar(retention_mesh, ax=axes[0, 1], label="Retention in loaded L1")

    labels = (
        (result.optimal_trace, "Selected optimum", "#2563eb"),
        (result.comparison_trace, "Poor feasible comparison", "#dc2626"),
    )
    for trace, label, color in labels:
        axes[1, 0].plot(
            trace.time_ms,
            trace.temperature_rise_uK,
            color=color,
            linewidth=1.8,
            label=(
                f"{label}: {trace.point.detuning_ghz:g} GHz, "
                f"{trace.point.handover_source_power_w:g} W"
            ),
        )
        axes[1, 1].plot(
            trace.time_ms,
            trace.retention_fraction,
            color=color,
            linewidth=1.8,
            label=label,
        )
    transition_times = (
        result.timing.acceleration_time_s * 1e3,
        (
            result.timing.acceleration_time_s
            + result.timing.cruise_time_s
        )
        * 1e3,
    )
    for axis in axes[1]:
        for transition in transition_times:
            axis.axvline(
                transition,
                color="#6b7280",
                linewidth=1.0,
                linestyle="--",
            )
        axis.set_xlabel("Time after L1 launch (ms)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[1, 0].set_ylabel("Temperature rise (µK)")
    axes[1, 0].set_title("Heating through acceleration, cruise and deceleration")
    axes[1, 1].set_ylabel("Retention fraction in loaded L1")
    minimum_retention = min(
        min(result.optimal_trace.retention_fraction),
        min(result.comparison_trace.retention_fraction),
    )
    retention_margin = max(1e-5, 0.05 * (1.0 - minimum_retention))
    axes[1, 1].set_ylim(
        max(0.0, minimum_retention - retention_margin),
        min(1.005, 1.0 + 0.25 * retention_margin),
    )
    axes[1, 1].yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    axes[1, 1].set_title("Statistical retention through L1 transport")
    figure.suptitle(
        f"{result.inputs.atom_label} L1 macroscopic transport: "
        f"a={result.inputs.acceleration_m_s2:g} m/s², "
        f"v={result.inputs.maximum_velocity_m_s:g} m/s; "
        "rate-loss coefficients use configured values",
        fontweight="bold",
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(_PLOT["dpi"]), facecolor="white")
    plt.close(figure)
    return output
