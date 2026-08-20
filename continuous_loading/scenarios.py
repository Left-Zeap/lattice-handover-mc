"""论文 Rb 基准场景和 Cs 参数扫描。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .atomic import CS133, RB87, AlkaliAtom
from .collisions import two_body_collision_density_m3_s
from .handover import HandoverParameters
from .lattice import (
    LatticeMetrics,
    atoms_per_site_for_peak_density,
    evaluate_lattice,
    power_for_target_depth,
)
from .transport import (
    TransportBudget,
    TransportStage,
    estimate_transport_budget,
    infer_handover_phase_fraction,
)


@dataclass(frozen=True)
class ExtendedFigure2ScanPreset:
    """Extended Data Fig. 2 中与交接模型相关的扫描范围。"""

    panel: str
    parameter_name: str
    values: tuple[float, ...]
    description: str


_EXTENDED_FIGURE2_PRESETS = {
    "a": ExtendedFigure2ScanPreset(
        panel="a",
        parameter_name="lattice1_distance_cm",
        values=tuple(round(38.0 + 0.1 * index, 1) for index in range(18)),
        description=(
            "Lattice-1 交接位置扫描；图中峰值约在 38.85 cm。"
        ),
    ),
    "b": ExtendedFigure2ScanPreset(
        panel="b",
        parameter_name="duration_ms",
        values=tuple(round(0.20 + 0.01 * index, 2) for index in range(21)),
        description=(
            "交接斜坡时间扫描；图中约 0.30-0.33 ms 快速进入平台，"
            "Methods 正式工作点为 1 ms。"
        ),
    ),
    "c": ExtendedFigure2ScanPreset(
        panel="c",
        parameter_name="post_handover_acceleration_m_s2",
        values=(
            300.0,
            500.0,
            800.0,
            1_000.0,
            1_500.0,
            2_200.0,
            3_200.0,
            4_500.0,
            6_500.0,
            10_000.0,
            15_000.0,
            22_000.0,
            32_000.0,
            50_000.0,
            75_000.0,
            100_000.0,
        ),
        description=(
            "运输加速度扫描；本 handover 模型只计算交接后倾斜势垒"
            "造成的存活，不包含 Fig. 2c 低加速度侧的全程运输损失。"
        ),
    ),
}


def extended_figure2_scan_preset(panel: str) -> ExtendedFigure2ScanPreset:
    """返回论文 Extended Data Fig. 2a-c 的可复现扫描网格。"""
    try:
        return _EXTENDED_FIGURE2_PRESETS[panel.lower()]
    except KeyError as exc:
        raise ValueError("Extended Data Fig. 2 预设面板必须是 a、b 或 c") from exc


def paper_handover_parameters(
    *,
    d1_red_detuning_ghz: float = 300.0,
    depth1_uK: float = 500.0,
    depth2_uK: float = 500.0,
    waist1_um: float = 250.0,
    waist2_um: float = 250.0,
    temperature_uK: float = 30.8,
) -> HandoverParameters:
    """构造论文双晶格交接的经典轨迹基准参数。

    ``30.8 µK`` 是现有恒定 500 µK 阱深运输预算给出的 Lattice-1
    末端温度；实验可直接用实测交接前温度覆盖。两晶格精确交叉角和
    交接处原子云长度未公开：默认分别采用腔体约 4° 倾角和 0.5 mm
    的轴向 1σ 尺寸作为显式、可扫描的工程假设。
    """
    wavelength_nm = RB87.laser_wavelength_red_of_d1_nm(
        d1_red_detuning_ghz
    )
    retro_ratio = 0.88**4
    lattice1 = _constant_depth_endpoint(
        RB87,
        wavelength_nm,
        depth1_uK,
        waist1_um,
        retro_ratio,
    )
    lattice2 = _constant_depth_endpoint(
        RB87,
        wavelength_nm,
        depth2_uK,
        waist2_um,
        retro_ratio,
    )
    return HandoverParameters(
        atom_mass_kg=RB87.mass_kg,
        wavelength_nm=wavelength_nm,
        depth1_uK=depth1_uK,
        depth2_uK=depth2_uK,
        waist1_um=waist1_um,
        waist2_um=waist2_um,
        scattering_rate1_s=lattice1.scattering_rate_s,
        scattering_rate2_s=lattice2.scattering_rate_s,
        retro_power_ratio=retro_ratio,
        temperature_uK=temperature_uK,
        duration_ms=1.0,
        crossing_angle_deg=4.0,
        lattice1_distance_cm=38.85,
        optimal_distance_cm=38.85,
        cloud_axial_sigma_mm=0.5,
        post_handover_acceleration_m_s2=4_000.0,
    )


@dataclass(frozen=True)
class PaperReproduction:
    """论文运输参数的主要可计算复现结果。"""

    laser_wavelength_nm: float
    lattice_at_250um: LatticeMetrics
    lattice_at_330um_same_power: LatticeMetrics
    required_power_330um_for_500uK: float
    lattice_at_150um_for_500uK: LatticeMetrics
    transport_budget: TransportBudget
    inferred_handover_fraction: float
    lattice1_average_speed_m_s: float
    lattice2_average_speed_m_s: float
    lattice1_frequency_shift_mhz_at_10m_s: float
    inferred_atoms_per_lattice_site: float
    inferred_occupied_lattice_sites: float
    stochastic_overlap_atoms: float
    collision_density_m3_s: float


def _constant_depth_endpoint(
    atom: AlkaliAtom,
    wavelength_nm: float,
    target_depth_uK: float,
    waist_um: float,
    retro_ratio: float,
) -> LatticeMetrics:
    power = power_for_target_depth(
        atom,
        wavelength_nm,
        target_depth_uK,
        waist_um,
        retro_ratio,
    )
    return evaluate_lattice(
        atom,
        wavelength_nm,
        power,
        waist_um,
        retro_ratio,
    )


def reproduce_paper_rb87() -> PaperReproduction:
    """用论文公开参数复现晶格深度、速度和 20->120 µK 温升预算。

    功率解释采用：

    * 前向入射功率 1 W；
    * 回程/前向功率比为论文给出的 quad-pass 效率 ``0.88^4``；
    * 250 µm 束腰处直接计算阱深；
    * 温升预算为了隔离几何效应，假设两段运输主动调功率，使阱深
      保持在 500 µK；交接相位随机化比例由终温 120 µK 反推。
    """
    wavelength_nm = RB87.laser_wavelength_red_of_d1_nm(300.0)
    retro_ratio = 0.88**4

    at_250 = evaluate_lattice(
        RB87,
        wavelength_nm,
        forward_power_w=1.0,
        waist_um=250.0,
        retro_power_ratio=retro_ratio,
    )
    at_330_same_power = evaluate_lattice(
        RB87,
        wavelength_nm,
        forward_power_w=1.0,
        waist_um=330.0,
        retro_power_ratio=retro_ratio,
    )
    required_330 = power_for_target_depth(
        RB87,
        wavelength_nm,
        target_depth_uK=500.0,
        waist_um=330.0,
        retro_power_ratio=retro_ratio,
    )
    at_150_target = _constant_depth_endpoint(
        RB87,
        wavelength_nm,
        target_depth_uK=500.0,
        waist_um=150.0,
        retro_ratio=retro_ratio,
    )

    # 对温升预算使用 500 µK 恒深度轨迹。这是一个明确的模型假设，
    # 因为论文没有给出两条晶格的完整功率随时间曲线。
    l1_start = _constant_depth_endpoint(
        RB87, wavelength_nm, 500.0, 330.0, retro_ratio
    )
    l1_end = _constant_depth_endpoint(
        RB87, wavelength_nm, 500.0, 250.0, retro_ratio
    )
    l2_start = _constant_depth_endpoint(
        RB87, wavelength_nm, 500.0, 250.0, retro_ratio
    )
    l2_end = at_150_target

    stage1 = TransportStage(
        name="Lattice-1",
        duration_s=0.050,
        distance_m=0.39,
        start_lattice=l1_start,
        end_lattice=l1_end,
        acceleration_m_s2=4000.0,
        acceleration_jumps_m_s2=(4000.0, -4000.0, -4000.0, 4000.0),
    )
    stage2 = TransportStage(
        name="Lattice-2",
        duration_s=0.021,
        distance_m=0.17,
        start_lattice=l2_start,
        end_lattice=l2_end,
        acceleration_m_s2=4000.0,
        acceleration_jumps_m_s2=(4000.0, -4000.0, -4000.0, 4000.0),
    )
    inferred_fraction = infer_handover_phase_fraction(
        target_final_temperature_uK=120.0,
        initial_temperature_uK=20.0,
        stage1=stage1,
        stage2=stage2,
    )
    budget = estimate_transport_budget(
        initial_temperature_uK=20.0,
        stage1=stage1,
        stage2=stage2,
        handover_random_phase_fraction=inferred_fraction,
    )

    frequency_shift_hz = 2.0 * 10.0 / (wavelength_nm * 1e-9)
    paper_peak_density_m3 = 5e11 * 1e6
    atoms_per_site = atoms_per_site_for_peak_density(
        peak_density_m3=paper_peak_density_m3,
        radial_frequency_hz=at_150_target.radial_frequency_hz,
        axial_frequency_hz=at_150_target.axial_frequency_hz,
        atom_mass_kg=RB87.mass_kg,
        temperature_uK=120.0,
    )
    collision_density = two_body_collision_density_m3_s(
        number_density_m3=paper_peak_density_m3,
        atom_mass_kg=RB87.mass_kg,
        temperature_uK=120.0,
        # Rb-87 |F=1,mF=-1> 常用三重态散射长度约 100 a0；
        # 精确值随内态和磁场变化，这里只做论文 3e19 量级复现。
        scattering_length_bohr=100.4,
    )
    return PaperReproduction(
        laser_wavelength_nm=wavelength_nm,
        lattice_at_250um=at_250,
        lattice_at_330um_same_power=at_330_same_power,
        required_power_330um_for_500uK=required_330,
        lattice_at_150um_for_500uK=at_150_target,
        transport_budget=budget,
        inferred_handover_fraction=inferred_fraction,
        lattice1_average_speed_m_s=0.39 / 0.050,
        lattice2_average_speed_m_s=0.17 / 0.021,
        lattice1_frequency_shift_mhz_at_10m_s=frequency_shift_hz / 1e6,
        inferred_atoms_per_lattice_site=atoms_per_site,
        inferred_occupied_lattice_sites=2.5e6 / atoms_per_site,
        stochastic_overlap_atoms=paper_peak_density_m3 * 1e-17,
        collision_density_m3_s=collision_density,
    )


@dataclass(frozen=True)
class CsCandidate:
    """一个满足指定 Cs 晶格目标的波长/功率组合。"""

    d1_red_detuning_ghz: float
    wavelength_nm: float
    forward_power_at_atoms_w: float
    source_power_w: float
    depth_uK: float
    scattering_rate_s: float
    recoil_heating_rate_uK_s: float
    radial_frequency_hz: float
    axial_frequency_hz: float
    critical_acceleration_m_s2: float
    feasible: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CsTransportPrediction:
    """给定 Cs 晶格设计下的双段运输功率与温升预测。"""

    candidate_at_250um: CsCandidate
    lattice1_start_power_w: float
    lattice1_end_power_w: float
    lattice2_start_power_w: float
    lattice2_end_power_w: float
    transport_budget: TransportBudget


def cs_design_candidate(
    *,
    d1_red_detuning_ghz: float,
    target_depth_uK: float,
    waist_um: float,
    retro_power_ratio: float = 0.88**4,
    delivery_efficiency: float = 1.0,
    max_source_power_w: float | None = None,
    max_scattering_rate_s: float | None = None,
) -> CsCandidate:
    """计算一个 Cs D1 红失谐候选并检查功率、散射约束。"""
    if (
        not math.isfinite(delivery_efficiency)
        or delivery_efficiency <= 0.0
        or delivery_efficiency > 1.0
    ):
        raise ValueError("传输效率必须位于 (0, 1]")

    wavelength_nm = CS133.laser_wavelength_red_of_d1_nm(d1_red_detuning_ghz)
    forward_power = power_for_target_depth(
        CS133,
        wavelength_nm,
        target_depth_uK,
        waist_um,
        retro_power_ratio,
    )
    metrics = evaluate_lattice(
        CS133,
        wavelength_nm,
        forward_power,
        waist_um,
        retro_power_ratio,
    )
    source_power = forward_power / delivery_efficiency

    # 与 transport.py 一致：三维谐振气体每次散射的等效温升为
    # 2 E_r/(3 k_B)。
    recoil_heating_rate = (
        2.0
        * metrics.recoil_temperature_uK
        / 3.0
        * metrics.scattering_rate_s
    )

    reasons: list[str] = []
    if max_source_power_w is not None and source_power > max_source_power_w:
        reasons.append("超过源端功率上限")
    if (
        max_scattering_rate_s is not None
        and metrics.scattering_rate_s > max_scattering_rate_s
    ):
        reasons.append("超过散射率上限")

    return CsCandidate(
        d1_red_detuning_ghz=d1_red_detuning_ghz,
        wavelength_nm=wavelength_nm,
        forward_power_at_atoms_w=forward_power,
        source_power_w=source_power,
        depth_uK=metrics.depth_uK,
        scattering_rate_s=metrics.scattering_rate_s,
        recoil_heating_rate_uK_s=recoil_heating_rate,
        radial_frequency_hz=metrics.radial_frequency_hz,
        axial_frequency_hz=metrics.axial_frequency_hz,
        critical_acceleration_m_s2=metrics.critical_axial_acceleration_m_s2,
        feasible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def scan_cs_designs(
    *,
    target_depth_uK: float,
    waist_um: float,
    detuning_min_ghz: float,
    detuning_max_ghz: float,
    detuning_step_ghz: float,
    retro_power_ratio: float = 0.88**4,
    delivery_efficiency: float = 1.0,
    max_source_power_w: float | None = None,
    max_scattering_rate_s: float | None = None,
) -> list[CsCandidate]:
    """扫描 Cs D1 红失谐并返回全部候选。"""
    if detuning_step_ghz <= 0.0 or detuning_max_ghz < detuning_min_ghz:
        raise ValueError("失谐扫描范围或步长无效")

    candidates: list[CsCandidate] = []
    detuning = detuning_min_ghz
    while detuning <= detuning_max_ghz + 0.5 * detuning_step_ghz:
        candidates.append(
            cs_design_candidate(
                d1_red_detuning_ghz=detuning,
                target_depth_uK=target_depth_uK,
                waist_um=waist_um,
                retro_power_ratio=retro_power_ratio,
                delivery_efficiency=delivery_efficiency,
                max_source_power_w=max_source_power_w,
                max_scattering_rate_s=max_scattering_rate_s,
            )
        )
        detuning += detuning_step_ghz
    return candidates


def predict_cs_transport(
    *,
    d1_red_detuning_ghz: float,
    target_depth_uK: float = 500.0,
    initial_temperature_uK: float = 20.0,
    lattice1_start_waist_um: float = 330.0,
    handover_waist_um: float = 250.0,
    lattice2_end_waist_um: float = 150.0,
    lattice1_distance_m: float = 0.39,
    lattice1_time_s: float = 0.050,
    lattice2_distance_m: float = 0.17,
    lattice2_time_s: float = 0.021,
    acceleration_m_s2: float = 4000.0,
    handover_random_phase_fraction: float = 0.63,
    retro_power_ratio: float = 0.88**4,
    delivery_efficiency: float = 1.0,
) -> CsTransportPrediction:
    """把论文同型双晶格几何应用到 Cs，预测功率轨迹和温升。

    两段运输均假设通过功率斜坡保持 ``target_depth_uK`` 恒定。返回的
    四个功率是原子处前向功率；若要得到激光源端功率，应除以
    ``delivery_efficiency``。
    """
    wavelength_nm = CS133.laser_wavelength_red_of_d1_nm(d1_red_detuning_ghz)
    l1_start = _constant_depth_endpoint(
        CS133,
        wavelength_nm,
        target_depth_uK,
        lattice1_start_waist_um,
        retro_power_ratio,
    )
    l1_end = _constant_depth_endpoint(
        CS133,
        wavelength_nm,
        target_depth_uK,
        handover_waist_um,
        retro_power_ratio,
    )
    l2_start = l1_end
    l2_end = _constant_depth_endpoint(
        CS133,
        wavelength_nm,
        target_depth_uK,
        lattice2_end_waist_um,
        retro_power_ratio,
    )
    stage1 = TransportStage(
        name="Cs Lattice-1",
        duration_s=lattice1_time_s,
        distance_m=lattice1_distance_m,
        start_lattice=l1_start,
        end_lattice=l1_end,
        acceleration_m_s2=acceleration_m_s2,
        acceleration_jumps_m_s2=(
            acceleration_m_s2,
            -acceleration_m_s2,
            -acceleration_m_s2,
            acceleration_m_s2,
        ),
    )
    stage2 = TransportStage(
        name="Cs Lattice-2",
        duration_s=lattice2_time_s,
        distance_m=lattice2_distance_m,
        start_lattice=l2_start,
        end_lattice=l2_end,
        acceleration_m_s2=acceleration_m_s2,
        acceleration_jumps_m_s2=(
            acceleration_m_s2,
            -acceleration_m_s2,
            -acceleration_m_s2,
            acceleration_m_s2,
        ),
    )
    budget = estimate_transport_budget(
        initial_temperature_uK,
        stage1,
        stage2,
        handover_random_phase_fraction=handover_random_phase_fraction,
    )
    candidate = cs_design_candidate(
        d1_red_detuning_ghz=d1_red_detuning_ghz,
        target_depth_uK=target_depth_uK,
        waist_um=handover_waist_um,
        retro_power_ratio=retro_power_ratio,
        delivery_efficiency=delivery_efficiency,
    )
    return CsTransportPrediction(
        candidate_at_250um=candidate,
        lattice1_start_power_w=l1_start.forward_power_w / delivery_efficiency,
        lattice1_end_power_w=l1_end.forward_power_w / delivery_efficiency,
        lattice2_start_power_w=l2_start.forward_power_w / delivery_efficiency,
        lattice2_end_power_w=l2_end.forward_power_w / delivery_efficiency,
        transport_budget=budget,
    )
