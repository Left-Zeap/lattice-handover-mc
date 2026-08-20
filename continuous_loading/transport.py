"""MOT 装入晶格后、到科学区之前的运输温升预算。

模型把可定量的机制分开列账：

* 光束几何或阱深变化造成的绝热压缩/解压；
* 自发散射的反冲加热；
* 两套未锁相晶格交接时的相位失配能量；
* 加速倾斜导致的有效势垒降低；
* 强度噪声和位置噪声的参数加热接口。

论文未给出完整功率轨迹、相位噪声 PSD、强度噪声 PSD 和晶格相位，
所以后两类技术加热不能从论文数据唯一预测；框架会显式保留这些
输入，而不是把未知加热隐藏进一个“理论效率”常数。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .constants import BOLTZMANN
from .lattice import LatticeMetrics, tilted_lattice_barrier_fraction


@dataclass(frozen=True)
class TransportStage:
    """一段功率和束腰可缓慢变化的晶格运输。"""

    name: str
    duration_s: float
    distance_m: float
    start_lattice: LatticeMetrics
    end_lattice: LatticeMetrics
    acceleration_m_s2: float
    acceleration_jumps_m_s2: tuple[float, ...] = ()
    intensity_noise_psd_1_hz: float = 0.0
    position_noise_psd_m2_hz: float = 0.0


@dataclass(frozen=True)
class StageHeating:
    """单段运输的温度变化分解。"""

    name: str
    input_temperature_uK: float
    adiabatic_temperature_uK: float
    recoil_heating_uK: float
    parametric_heating_uK: float
    position_noise_heating_uK: float
    acceleration_jump_heating_uK: float
    output_temperature_uK: float
    scattering_events: float
    barrier_fraction: float
    effective_barrier_uK: float
    thermal_bound_fraction: float
    average_speed_m_s: float


@dataclass(frozen=True)
class TransportBudget:
    """完整双晶格运输的温升预算。"""

    initial_temperature_uK: float
    stages: tuple[StageHeating, ...]
    handover_heating_uK: float
    handover_random_phase_fraction: float
    final_temperature_uK: float


def geometric_mean_frequency_hz(lattice: LatticeMetrics) -> float:
    """返回轴对称三维谐振阱的几何平均频率。"""
    return (
        lattice.radial_frequency_hz**2 * lattice.axial_frequency_hz
    ) ** (1.0 / 3.0)


def adiabatic_temperature(
    input_temperature_uK: float,
    start_lattice: LatticeMetrics,
    end_lattice: LatticeMetrics,
) -> float:
    """按 ``T/omega_bar`` 不变量估算经典气体绝热温度变化。"""
    if not math.isfinite(input_temperature_uK) or input_temperature_uK <= 0.0:
        raise ValueError("输入温度必须是有限正数")
    ratio = (
        geometric_mean_frequency_hz(end_lattice)
        / geometric_mean_frequency_hz(start_lattice)
    )
    return input_temperature_uK * ratio


def recoil_temperature_per_scatter_uK(
    lattice: LatticeMetrics,
    *,
    thermal_energy_dof: float = 3.0,
) -> float:
    """把一次吸收+自发辐射的 ``2 E_r`` 转换为等效温升。

    对三维谐振束缚的经典气体，总能量约为 ``3 k_B T``，因此默认
    每次散射导致 ``2 E_r/(3 k_B)`` 的温升。若只分析自由粒子动能，
    可把 ``thermal_energy_dof`` 改为 1.5。
    """
    if not math.isfinite(thermal_energy_dof) or thermal_energy_dof <= 0.0:
        raise ValueError("热能自由度系数必须是有限正数")
    return 2.0 * lattice.recoil_temperature_uK / thermal_energy_dof


def random_phase_handover_heating_uK(
    receiving_depth_uK: float,
    random_phase_fraction: float,
    *,
    thermal_energy_dof: float = 3.0,
) -> float:
    """估计两个未锁相驻波交接的等效温升。

    若原子在接收晶格相位上均匀分布，平均新增势能为 ``U/2``。
    ``random_phase_fraction=1`` 表示完全随机相位，0 表示晶格极小值
    完全重合。将能量除以 ``thermal_energy_dof*k_B`` 得到等效温升。
    """
    if not math.isfinite(receiving_depth_uK) or receiving_depth_uK <= 0.0:
        raise ValueError("接收晶格阱深必须是有限正数")
    if (
        not math.isfinite(random_phase_fraction)
        or random_phase_fraction < 0.0
        or random_phase_fraction > 1.0
    ):
        raise ValueError("随机相位比例必须位于 [0, 1]")
    if not math.isfinite(thermal_energy_dof) or thermal_energy_dof <= 0.0:
        raise ValueError("热能自由度系数必须是有限正数")
    return (
        random_phase_fraction
        * receiving_depth_uK
        / (2.0 * thermal_energy_dof)
    )


def acceleration_jump_heating_uK(
    atom_mass_kg: float,
    axial_frequency_hz: float,
    acceleration_jumps_m_s2: tuple[float, ...],
    *,
    thermal_energy_dof: float = 3.0,
) -> float:
    """估计突然改变加速度造成的非绝热轴向激发。

    谐振阱平衡位置随加速度移动 ``x_eq=-a/omega_z^2``。若加速度发生
    瞬时跳变 ``Delta a``，注入能量为

    ``Delta E = m*(Delta a)^2/(2*omega_z^2)``。

    本函数把多次跳变的能量按非相干方式相加；真实结果还依赖各跳变
    间隔相对于轴向振荡相位，可能相消或相长。
    """
    if not math.isfinite(atom_mass_kg) or atom_mass_kg <= 0.0:
        raise ValueError("原子质量必须是有限正数")
    if not math.isfinite(axial_frequency_hz) or axial_frequency_hz <= 0.0:
        raise ValueError("轴向阱频必须是有限正数")
    if not math.isfinite(thermal_energy_dof) or thermal_energy_dof <= 0.0:
        raise ValueError("热能自由度系数必须是有限正数")
    if any(not math.isfinite(jump) for jump in acceleration_jumps_m_s2):
        raise ValueError("加速度跳变量必须是有限数")

    omega_z = 2.0 * math.pi * axial_frequency_hz
    energy_j = sum(
        atom_mass_kg * jump**2 / (2.0 * omega_z**2)
        for jump in acceleration_jumps_m_s2
    )
    return energy_j / (thermal_energy_dof * BOLTZMANN) * 1e6


def thermal_bound_fraction_3d_harmonic(
    barrier_uK: float,
    temperature_uK: float,
) -> float:
    """三维经典谐振气体能量低于有限势垒的比例。

    三维谐振子的总能量分布是形状参数 3 的 Gamma 分布。令
    ``eta=U/(k_B T)``，则累积分布为

    ``1-exp(-eta)*(1+eta+eta^2/2)``。

    这是假设瞬时热平衡且忽略隧穿的上限模型；真实运输损失还依赖
    轨迹、碰撞再热化、蒸发和技术噪声。
    """
    if not math.isfinite(barrier_uK) or barrier_uK < 0.0:
        raise ValueError("势垒必须是有限非负数")
    if not math.isfinite(temperature_uK) or temperature_uK <= 0.0:
        raise ValueError("温度必须是有限正数")
    eta = barrier_uK / temperature_uK
    return 1.0 - math.exp(-eta) * (1.0 + eta + 0.5 * eta**2)


def _stage_heating(stage: TransportStage, input_temperature_uK: float) -> StageHeating:
    if stage.duration_s <= 0.0 or stage.distance_m < 0.0:
        raise ValueError("运输时间必须为正，距离不能为负")

    adiabatic = adiabatic_temperature(
        input_temperature_uK,
        stage.start_lattice,
        stage.end_lattice,
    )

    # 散射率随强度线性变化。缺少完整功率轨迹时，用端点几何平均值
    # 近似积分，避免只取高强度端造成系统性高估。
    average_scattering = math.sqrt(
        stage.start_lattice.scattering_rate_s
        * stage.end_lattice.scattering_rate_s
    )
    scattering_events = average_scattering * stage.duration_s
    recoil_uK = scattering_events * recoil_temperature_per_scatter_uK(
        stage.end_lattice
    )

    # 强度噪声参数加热：d<E>/dt = (omega^2/4) S_epsilon(2omega) <E>。
    # 这里用几何平均角频率和白噪声 PSD 做小信号近似。
    omega_bar = 2.0 * math.pi * geometric_mean_frequency_hz(stage.end_lattice)
    parametric_rate = omega_bar**2 * stage.intensity_noise_psd_1_hz / 4.0
    parametric_uK = adiabatic * math.expm1(parametric_rate * stage.duration_s)

    # 位置噪声加热：dE/dt = m*omega^4*S_x(omega)/4。
    atom_mass = _infer_mass_from_critical_acceleration(stage.end_lattice)
    position_heating_j = (
        atom_mass
        * omega_bar**4
        * stage.position_noise_psd_m2_hz
        * stage.duration_s
        / 4.0
    )
    position_uK = position_heating_j / (3.0 * BOLTZMANN) * 1e6
    acceleration_uK = acceleration_jump_heating_uK(
        atom_mass,
        stage.end_lattice.axial_frequency_hz,
        stage.acceleration_jumps_m_s2,
    )

    barrier_fraction = tilted_lattice_barrier_fraction(
        stage.acceleration_m_s2,
        stage.end_lattice.critical_axial_acceleration_m_s2,
    )
    effective_barrier = stage.end_lattice.depth_uK * barrier_fraction
    output = (
        adiabatic
        + recoil_uK
        + parametric_uK
        + position_uK
        + acceleration_uK
    )
    bound_fraction = thermal_bound_fraction_3d_harmonic(
        effective_barrier,
        output,
    )

    return StageHeating(
        name=stage.name,
        input_temperature_uK=input_temperature_uK,
        adiabatic_temperature_uK=adiabatic,
        recoil_heating_uK=recoil_uK,
        parametric_heating_uK=parametric_uK,
        position_noise_heating_uK=position_uK,
        acceleration_jump_heating_uK=acceleration_uK,
        output_temperature_uK=output,
        scattering_events=scattering_events,
        barrier_fraction=barrier_fraction,
        effective_barrier_uK=effective_barrier,
        thermal_bound_fraction=bound_fraction,
        average_speed_m_s=stage.distance_m / stage.duration_s,
    )


def _infer_mass_from_critical_acceleration(lattice: LatticeMetrics) -> float:
    """由 ``a_c=U*k/m`` 反推出质量，避免 Stage 重复持有原子对象。"""
    depth_j = lattice.depth_uK * 1e-6 * BOLTZMANN
    wave_number = 2.0 * math.pi / (lattice.laser_wavelength_nm * 1e-9)
    return depth_j * wave_number / lattice.critical_axial_acceleration_m_s2


def estimate_transport_budget(
    initial_temperature_uK: float,
    stage1: TransportStage,
    stage2: TransportStage,
    *,
    handover_random_phase_fraction: float,
) -> TransportBudget:
    """依次计算 Lattice-1、交接、Lattice-2 的温升。"""
    first = _stage_heating(stage1, initial_temperature_uK)
    handover = random_phase_handover_heating_uK(
        stage2.start_lattice.depth_uK,
        handover_random_phase_fraction,
    )
    second = _stage_heating(
        stage2,
        first.output_temperature_uK + handover,
    )
    return TransportBudget(
        initial_temperature_uK=initial_temperature_uK,
        stages=(first, second),
        handover_heating_uK=handover,
        handover_random_phase_fraction=handover_random_phase_fraction,
        final_temperature_uK=second.output_temperature_uK,
    )


def infer_handover_phase_fraction(
    target_final_temperature_uK: float,
    initial_temperature_uK: float,
    stage1: TransportStage,
    stage2: TransportStage,
) -> float:
    """反推复现实测终温所需的交接相位随机化比例。

    温升模型在相位比例上是线性的，但交接热量还会被第二段绝热压缩
    放大。本函数分别计算 0 与 1 两个端点并作线性插值。
    """
    if target_final_temperature_uK <= 0.0:
        raise ValueError("目标终温必须为正")
    cold = estimate_transport_budget(
        initial_temperature_uK,
        stage1,
        stage2,
        handover_random_phase_fraction=0.0,
    )
    hot = estimate_transport_budget(
        initial_temperature_uK,
        stage1,
        stage2,
        handover_random_phase_fraction=1.0,
    )
    span = hot.final_temperature_uK - cold.final_temperature_uK
    if span <= 0.0:
        raise ValueError("当前模型中交接相位没有产生正温升")
    fraction = (target_final_temperature_uK - cold.final_temperature_uK) / span
    return fraction
