"""模拟层编排器：L1→handover→L2 单系综贯穿三段的全链路 Monte Carlo。

编排严格遵循用户要求的三条规则：

1. **光场时序先行**：传播开始前由 ``ChainLightField.precompute`` 把
   三段光场 I(x,y,z,t) 的逐步时序表一次性算好，之后只按 (xyz, t)
   查询。传播本身仍由 transport_mc/handover 的 velocity-Verlet
   积分器执行（本模块不改物理公式）；light_field 的逐位一致性测试
   保证积分器逐步取光学量与时序表查询结果相同，时序表随
   ``ChainMcResult.light_field`` 返回供核查。
2. **L1/L2 腿每步宽容捕获判定**：腿段调用
   ``transport_mc.simulate_leg_monte_carlo`` 固定传
   ``escape_check_interval_steps=1, escape_lenient=True``——每个
   动力学步都做一次捕获判定，判据为共动系轴向激发能 ε < U_ax
   （未缩减全深，不乘 F(a/a_c) 倾斜势垒缩减因子；重力径向下坡
   鞍点判据保留），脱捕轨迹立即剔除、不再参与运算。
3. **handover 段不做捕获判定**：全部粒子传播到底（现行为，段内
   粒子数恒定）；粒子数只在段末的捕获子样本选择时减少一次。

口径与注意事项：

- **剔除判据与旧"200 步 + 缩减势垒"判据的统计差异**：每步判定在
  轨迹首次越过势垒时立即剔除，宽容判据则用比缩减判据更高的全深
  势垒（保留更多瞬时过垒轨迹）；两者对留存/末温的影响方向相反，
  同一工作点下应在 MC 误差内统计一致，差异超 MC 误差时需停下来
  核对物理。**GPU 口径分裂**：每步+宽容判据只在 CPU 逐步路径生效；
  GPU 批量/融合 kernel 仍用旧判据（GPU 为性能层，kernel 公式不
  动），因此 GPU 后端的链式结果与 CPU 仅统计一致。
- **温度口径**：腿段温度 = 幸存者共动系去质心三维动能温度
  （``L1TransportTrace.temperature_uK``，与
  ``handover._kinetic_temperature_uK`` 同口径）；handover 的
  ``final_temperature_uK`` = 捕获子样本 ⟨ε⟩/3k_B（总激发能口
  径），``final_kinetic_temperature_uK`` = 捕获子样本 L2 共动系
  去质心动能温度；科学区汇总由
  ``l2_transport.l2_result_from_leg_trace`` 给出。
- **留存链**：S_total = S_L1·η_HO·S_L2（装载段已删，无 η_load），
  原子数从 ``L1TransportInputs.initial_atom_number`` 起算；各段
  幸存/捕获的 Monte Carlo 标准误沿用 Jeffreys Beta(1/2,1/2) 后验
  口径（腿段 ``retention_standard_error``、handover
  ``transfer_standard_error``，均由现有函数计算后透传）。
- **段边界 adapter**：L1→handover 为 z 平移（transport_lab →
  handover_l1_local），handover→L2 为正交旋转（handover_l2_
  canonical → l2_local，逐粒子晶格相位已在 handover 出口规范
  化）；adapter 由 phase_space 提供，不丢相位——瞬时相位淬火会
  制造伪温度尖刺（handover.py 出口注释的教训）。
- **初始系综**：``initial_state.sample_static_lattice_thermal_ensemble``
  在 L1 起点静晶格（z_L=0、φ=0）中做热平衡采样；晶格参数由 L1
  腿 t=0 光学量精确还原（波腹阱深 U=c_u(√I₁+√I₂)²、回程比
  R=I₂/I₁），束腰取强度加权有效束腰——非 conveyor 时 w₁=w₂，
  采样与腿内部分布逐位一致；conveyor 错腰几何为有效束腰近似
  （拒绝步骤仍用完整双束势，差异为提议分布宽度的二阶效应）。
  浅阱/高温到几乎无束缚初态时采样抛 ``ValueError``（与腿段零留
  存容错不同，链式入口直接上抛，由调用方决定容错策略）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .constants import BOLTZMANN
from .dipole import scalar_potential_and_scattering
from .handover import (
    HandoverParameters,
    HandoverResult,
    run_handover_monte_carlo,
    zero_capture_handover_result,
)
from .initial_state import (
    ThermalLatticeEnsembleInputs,
    ensemble_kinetic_temperature_uK,
    sample_static_lattice_thermal_ensemble,
)
from .l1_transport import (
    L1TransportInputs,
    L1TransportTrace,
    _atom_from_label,
)
from .l2_transport import (
    L2TransportInputs,
    ScienceRegionSummary,
    l2_end_source_power_w,
    l2_leg_inputs,
    l2_result_from_leg_trace,
)
from .light_field import ChainLightField
from .phase_space import (
    ParticleEnsemble,
    handover_to_l2_local,
    l1_transport_end_to_handover,
)
from .transport_mc import (
    _leg_optics_at,
    _leg_optics_profile,
    simulate_leg_monte_carlo,
)


@dataclass(frozen=True)
class ChainMcInputs:
    """L1→handover→L2 链式 Monte Carlo 的输入。

    三段参数复用现有构造入口：``transport`` 为
    ``l1_transport.L1TransportInputs``（L1 腿几何/时序/光学/conveyor
    与 mc_* 数值字段）、``handover`` 为 ``handover.HandoverParameters``
    （由 ``l1_handover._handover_parameters`` 或等价方式构造，阱深/
    束腰/波长须与 L1 工作点自洽）、``l2`` 为
    ``l2_transport.L2TransportInputs``。

    顶层标量字段在运行时对三段**透传**（``dataclasses.replace``，
    不修改调用方对象），保证链路口径一致：``atom_label`` 同时重写
    handover 的原子质量；``initial_temperature_uK``/``particle_count``/
    ``seed``/``include_gravity``/``include_scattering`` 写入腿段
    mc_* 与 include_* 字段及 handover 同名字段。初始采样的原子云
    轴向尺寸取 ``transport.mc_cloud_axial_sigma_mm``。
    """

    atom_label: str
    transport: L1TransportInputs
    handover: HandoverParameters
    l2: L2TransportInputs
    initial_temperature_uK: float = 20.0
    particle_count: int = 2_000
    seed: int = 20_250_902
    include_gravity: bool = True
    include_scattering: bool = True

    def __post_init__(self) -> None:
        _atom_from_label(self.atom_label)
        if (
            not math.isfinite(self.initial_temperature_uK)
            or self.initial_temperature_uK <= 0.0
        ):
            raise ValueError("初始温度必须是有限正数")
        if self.particle_count <= 0:
            raise ValueError("Monte Carlo 粒子数必须是正整数")


@dataclass(frozen=True)
class ChainMcResult:
    """L1→handover→L2 链式 Monte Carlo 的结果。

    留存链 S_total = S_L1·η_HO·S_L2（无 η_load，原子数从
    ``transport.initial_atom_number`` 起算）；幸存/捕获标准误为
    Jeffreys 口径（由各段 trace/result 透传）。``l2_trace`` 与
    ``science`` 在 L1 全灭或 handover 零捕获时为 None（对应留存
    记 0）。
    """

    inputs: ChainMcInputs
    detuning_ghz: float
    handover_source_power_w: float
    light_field: ChainLightField
    initial_ensemble: ParticleEnsemble
    initial_sampled_temperature_uK: float
    l1_trace: L1TransportTrace
    handover_result: HandoverResult
    l2_trace: L1TransportTrace | None
    science: ScienceRegionSummary | None
    # 逐段粒子记账：各段进入数恒为 inputs.particle_count（段边界
    # 等权重采样补齐，见 phase_space.ParticleEnsemble.resampled）；
    # handover 段内粒子数恒定，捕获子样本选择只发生在段末。
    l1_survivor_count: int
    handover_captured_count: int
    l2_survivor_count: int
    # 逐段留存与 Jeffreys 标准误。
    l1_retention_fraction: float
    l1_retention_standard_error: float | None
    handover_transfer_efficiency: float
    handover_transfer_standard_error: float
    l2_retention_fraction: float
    l2_retention_standard_error: float | None
    total_retention_fraction: float
    final_atom_number: float


def _chain_leg_inputs(inputs: ChainMcInputs) -> L1TransportInputs:
    """顶层标量透传到 L1 腿输入（replace，不修改调用方对象）。"""
    return replace(
        inputs.transport,
        atom_label=inputs.atom_label,
        initial_temperature_uK=inputs.initial_temperature_uK,
        mc_particle_count=inputs.particle_count,
        mc_seed=inputs.seed,
        mc_include_scattering=inputs.include_scattering,
        include_gravity=inputs.include_gravity,
    )


def _chain_handover_parameters(inputs: ChainMcInputs) -> HandoverParameters:
    """顶层标量透传到 handover 参数（原子质量由 atom_label 重写）。"""
    atom = _atom_from_label(inputs.atom_label)
    return replace(
        inputs.handover,
        atom_mass_kg=atom.mass_kg,
        particle_count=inputs.particle_count,
        seed=inputs.seed,
        include_gravity=inputs.include_gravity,
        include_scattering=inputs.include_scattering,
    )


def _sample_chain_initial_ensemble(
    inputs: ChainMcInputs,
    transport: L1TransportInputs,
    wavelength_nm: float,
    handover_source_power_w: float,
) -> ParticleEnsemble:
    """用 L1 腿 t=0 光学量还原静晶格参数并采样热平衡系综。

    波腹阱深与回程比精确还原双束强度（见模块 docstring）；束腰取
    强度加权有效束腰，与 ``transport_mc._leg_optics_profile`` 的
    conveyor 有效束腰同一公式（w₁=w₂ 时退化为共同束腰）。
    """
    atom = _atom_from_label(inputs.atom_label)
    profile = _leg_optics_profile(
        transport, wavelength_nm, handover_source_power_w
    )
    i1, i2, w1, w2, _, _ = _leg_optics_at(
        transport, profile, handover_source_power_w, 0.0, 0.0
    )
    potential_per_intensity = abs(
        scalar_potential_and_scattering(atom, wavelength_nm, 1.0).potential_j
    )
    antinode_depth_j = potential_per_intensity * (
        i1 + i2 + 2.0 * math.sqrt(i1 * i2)
    )
    effective_waist_m = math.sqrt(
        (i1 + i2) / (i1 / w1**2 + i2 / w2**2)
    )
    sampling_inputs = ThermalLatticeEnsembleInputs(
        atom_label=inputs.atom_label,
        wavelength_nm=wavelength_nm,
        waist_um=effective_waist_m * 1e6,
        depth_uK=antinode_depth_j / BOLTZMANN * 1e6,
        temperature_uK=inputs.initial_temperature_uK,
        particle_count=inputs.particle_count,
        seed=inputs.seed,
        retro_power_ratio=i2 / i1,
        cloud_axial_sigma_mm=transport.mc_cloud_axial_sigma_mm,
        include_gravity=inputs.include_gravity,
    )
    return sample_static_lattice_thermal_ensemble(sampling_inputs)


def run_chain_monte_carlo(
    inputs: ChainMcInputs,
    detuning_ghz: float,
    handover_source_power_w: float,
) -> ChainMcResult:
    """运行 L1→handover→L2 链式 Monte Carlo（流程见模块 docstring）。

    ``detuning_ghz`` 为 D1 线红失谐；``handover_source_power_w`` 为
    handover 束腰处每条晶格分支的源端功率（与解析腿/扫描网格同一
    口径）。L2 腿的源端功率按 ``l2_end_source_power_w`` 恒阱深口径
    随束腰平方缩放。
    """
    transport = _chain_leg_inputs(inputs)
    handover_parameters = _chain_handover_parameters(inputs)
    atom = _atom_from_label(inputs.atom_label)
    wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning_ghz)

    # 1. 光场时序先行：三段逐步时序表在传播开始前一次性算好。
    light_field = ChainLightField.precompute(
        transport,
        detuning_ghz,
        handover_source_power_w,
        handover_parameters,
        inputs.l2,
    )
    # 2. 初始系综：L1 起点静晶格（z_L=0、φ=0）热平衡采样。
    initial_ensemble = _sample_chain_initial_ensemble(
        inputs, transport, wavelength_nm, handover_source_power_w
    )
    initial_temperature_uK = ensemble_kinetic_temperature_uK(
        initial_ensemble, atom.mass_kg
    )
    # 3. L1 腿：每步 + 宽容捕获判定，脱捕轨迹立即剔除。
    l1_trace, l1_final = simulate_leg_monte_carlo(
        transport,
        detuning_ghz,
        handover_source_power_w,
        initial_ensemble=initial_ensemble,
        return_final_ensemble=True,
        escape_check_interval_steps=1,
        escape_lenient=True,
    )
    s_l1 = l1_trace.point.final_retention_fraction
    l1_survivors = 0 if l1_final is None else l1_final.particle_count

    # 4-5. handover：z 平移到交接局域系后整段传播，段内不做捕获
    # 判定（粒子数恒定）；L1 全灭时按零捕获容错（同浅阱容错语义）。
    if l1_final is None:
        handover_result = zero_capture_handover_result(handover_parameters)
        captured_ensemble = None
    else:
        handover_result, captured_ensemble = run_handover_monte_carlo(
            handover_parameters,
            initial_ensemble=l1_transport_end_to_handover(
                l1_final, transport.distance_m
            ),
            return_captured_ensemble=True,
        )
    eta_ho = handover_result.transfer_efficiency

    # 6-7. L2 腿：旋转到 L2 局部系（相位已在 handover 出口规范化），
    # 同样的每步 + 宽容剔除；腿输入由 l2_leg_inputs 构造，源端功率
    # 用 l2_end_source_power_w 口径。
    l2_trace: L1TransportTrace | None = None
    science: ScienceRegionSummary | None = None
    s_l2 = 0.0
    l2_standard_error: float | None = None
    l2_survivors = 0
    if captured_ensemble is not None:
        captured_temperature_uK = handover_result.final_kinetic_temperature_uK
        if captured_temperature_uK is None:
            captured_temperature_uK = handover_result.final_temperature_uK
        captured_atom_number = transport.initial_atom_number * s_l1 * eta_ho
        end_source_power_w = l2_end_source_power_w(
            transport, inputs.l2, handover_source_power_w
        )
        l2_leg = l2_leg_inputs(
            transport,
            inputs.l2,
            captured_temperature_uK,
            captured_atom_number,
        )
        l2_trace, l2_final = simulate_leg_monte_carlo(
            l2_leg,
            detuning_ghz,
            end_source_power_w,
            initial_ensemble=handover_to_l2_local(
                captured_ensemble, handover_parameters.crossing_angle_deg
            ),
            return_final_ensemble=True,
            escape_check_interval_steps=1,
            escape_lenient=True,
        )
        s_l2 = l2_trace.point.final_retention_fraction
        l2_standard_error = l2_trace.retention_standard_error
        l2_survivors = 0 if l2_final is None else l2_final.particle_count
        # 8. 科学区汇总（与解析/批量腿共用同一装配口径）。
        science = l2_result_from_leg_trace(
            transport,
            inputs.l2,
            detuning_ghz,
            end_source_power_w,
            captured_temperature_uK,
            captured_atom_number,
            l2_trace,
        ).science

    # 8. 留存链统计折算（温度口径见模块 docstring，均已在各段
    # trace/result 内按现状口径给出）。
    s_total = s_l1 * eta_ho * s_l2
    return ChainMcResult(
        inputs=inputs,
        detuning_ghz=detuning_ghz,
        handover_source_power_w=handover_source_power_w,
        light_field=light_field,
        initial_ensemble=initial_ensemble,
        initial_sampled_temperature_uK=initial_temperature_uK,
        l1_trace=l1_trace,
        handover_result=handover_result,
        l2_trace=l2_trace,
        science=science,
        l1_survivor_count=l1_survivors,
        handover_captured_count=handover_result.captured_count,
        l2_survivor_count=l2_survivors,
        l1_retention_fraction=s_l1,
        l1_retention_standard_error=l1_trace.retention_standard_error,
        handover_transfer_efficiency=eta_ho,
        handover_transfer_standard_error=(
            handover_result.transfer_standard_error
        ),
        l2_retention_fraction=s_l2,
        l2_retention_standard_error=l2_standard_error,
        total_retention_fraction=s_total,
        final_atom_number=transport.initial_atom_number * s_total,
    )
