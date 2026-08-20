"""初始条件层：静止 L1 光晶格中的束缚热平衡系综采样。

物理图景：原子被**静止** L1 光晶格（驻波相位 φ=0、晶格速度 0、
z_L(0)=0，格点位于 n·λ/2）束缚了足够长的时间（远大于碰撞/再热化
弛豫时间），达到温度 ``T``（默认 20 µK）的热平衡。本模块把这一图景
作为 L1→handover→L2 全链路的初始条件，替代已移除的 LGM 静止装载
模拟：宏观量（温度）只用于初始采样，之后逐粒子传播。

采样口径（配方以 ``transport_mc._sample_initial_ensemble`` 为蓝本
提取，物理一字未改；原函数保持不变）：

- 提议分布：谐振近似下的三维 Boltzmann 分布——位置高斯
  σ_ρ=√(k_B T/m)/ω_ρ、σ_z=√(k_B T/m)/ω_z，ω 由均分定理从双束
  参数曲率给出（ω_ρ²=(4|C_U|/m)(I₁/w₁²+I₂/w₂²)、
  ω_z²=2|C_U|ΔI_ax·k²/m，ΔI_ax=4√(I₁I₂)）；速度为各向同性
  Maxwell 分布 σ_v=√(k_B T/m)。
- 可选重力下垂：``include_gravity`` 时提议中心沿 -y 平移
  ``lattice.gaussian_gravity_trap`` 给出的下垂量。
- 可选轴向格点吸附：``cloud_axial_sigma_mm>0`` 时把 z 吸附到
  n·λ/2 格点链上，格点指标按宽度为该值的高斯云分布抽取。
- 拒绝判据（截断到束缚域）：用**完整双束 cos² 势**计算相对晶格系
  总激发能 ε=½m|v|²+V+U_ax，只保留 ε<U_ax 的样本；启用重力时再
  要求径向激发能低于高斯+重力下坡鞍点势垒。提议分布是谐振近似
  Boltzmann 而非完整势的严格 Boltzmann，因此深阱极限下才是严格
  截断 Boltzmann 分布的良好近似——这是与 transport_mc/handover
  现状一致的工程口径。拒绝超上限（浅阱/高温到几乎无束缚初态）抛
  ``ValueError``，由调用方决定容错策略。

晶格描述直接取波长、束腰和**波腹阱深** ``depth_uK``（与
``HandoverParameters.depth*_uK`` 同一口径，已含 (1+√R)² 波腹增强），
不经失谐+功率换算：这是给定"静止晶格束缚原子"图景时最简单可靠的
参数化。内部再按 R=``retro_power_ratio`` 还原双束强度
I₁=U/(|C_U|(1+√R)²)、I₂=R·I₁，使 R=1 时轴向调制深 U_ax 恰等于
``depth_uK``（与 handover 的 cos² 单包络口径一致）。

输出 ``ParticleEnsemble`` 的 ``frame="l1_local"``（L1 局部坐标：
z 沿运输轴、重力沿 -y），等权，逐粒子 ``site_index`` 记录格点吸附
的格点指标（无吸附时全 0）。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .constants import BOLTZMANN, GRAVITY
from .dipole import scalar_potential_and_scattering
from .lattice import gaussian_gravity_trap
from .l1_transport import _atom_from_label
from .phase_space import ParticleEnsemble
from .transport_mc import _double_beam_potential_and_force


@dataclass(frozen=True)
class ThermalLatticeEnsembleInputs:
    """静止 L1 晶格束缚热平衡系综的采样参数。

    晶格由波长、束腰、波腹阱深直接描述（见模块 docstring 口径）；
    ``cloud_axial_sigma_mm=0`` 表示全部原子位于中心格点链。
    """

    atom_label: str
    wavelength_nm: float
    waist_um: float
    depth_uK: float
    temperature_uK: float = 20.0
    particle_count: int = 2_000
    seed: int = 20_250_902
    retro_power_ratio: float = 1.0
    cloud_axial_sigma_mm: float = 0.0
    include_gravity: bool = True

    def __post_init__(self) -> None:
        _atom_from_label(self.atom_label)
        positive = {
            "波长": self.wavelength_nm,
            "束腰": self.waist_um,
            "波腹阱深": self.depth_uK,
            "温度": self.temperature_uK,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name}必须是有限正数")
        if self.particle_count <= 0:
            raise ValueError("粒子数必须是正整数")
        if not 0.0 <= self.retro_power_ratio <= 1.0:
            raise ValueError("回程功率比必须位于 [0,1]")
        if (
            not math.isfinite(self.cloud_axial_sigma_mm)
            or self.cloud_axial_sigma_mm < 0.0
        ):
            raise ValueError("原子云轴向尺寸必须是有限非负数")


def _standing_wave_beam_parameters(
    inputs: ThermalLatticeEnsembleInputs,
) -> tuple[float, float, float, float, float]:
    """由波腹阱深还原双束采样参数 ``(I1, I2, U_ax, k, |C_U|)``（SI）。

    静止驻波前向束腰中心强度 I₁、回程 I₂=R·I₁，波腹强度
    I₁(1+√R)² 对应给定波腹阱深；轴向调制深
    U_ax=4|C_U|√(I₁I₂)，R=1 时等于波腹阱深。
    """
    atom = _atom_from_label(inputs.atom_label)
    unit_dipole = scalar_potential_and_scattering(
        atom, inputs.wavelength_nm, 1.0
    )
    potential_per_intensity = abs(unit_dipole.potential_j)
    wave_number = 2.0 * math.pi / (inputs.wavelength_nm * 1e-9)
    depth_j = inputs.depth_uK * 1e-6 * BOLTZMANN
    interference_factor = (1.0 + math.sqrt(inputs.retro_power_ratio)) ** 2
    intensity1 = depth_j / (potential_per_intensity * interference_factor)
    intensity2 = inputs.retro_power_ratio * intensity1
    axial_modulation_j = (
        potential_per_intensity * 4.0 * math.sqrt(intensity1 * intensity2)
    )
    return (
        intensity1,
        intensity2,
        axial_modulation_j,
        wave_number,
        potential_per_intensity,
    )


def sample_static_lattice_thermal_ensemble(
    inputs: ThermalLatticeEnsembleInputs,
) -> ParticleEnsemble:
    """采样静止 L1 晶格中的束缚热平衡系综（配方见模块 docstring）。

    与 ``transport_mc._sample_initial_ensemble`` 同一配方（z_L=0、
    φ=0），逐位复现其 RNG 调用顺序；额外返回逐粒子 ``site_index``。
    """
    atom = _atom_from_label(inputs.atom_label)
    mass = atom.mass_kg
    (
        intensity1_w_m2,
        intensity2_w_m2,
        axial_modulation_j,
        wave_number_m,
        potential_per_intensity_j,
    ) = _standing_wave_beam_parameters(inputs)
    waist1_m = waist2_m = inputs.waist_um * 1e-6
    particle_count = inputs.particle_count
    rng = np.random.default_rng(inputs.seed)

    temperature_k = inputs.temperature_uK * 1e-6
    omega_radial = math.sqrt(
        4.0
        * potential_per_intensity_j
        / mass
        * (
            intensity1_w_m2 / waist1_m**2
            + intensity2_w_m2 / waist2_m**2
        )
    )
    omega_axial = math.sqrt(
        2.0 * axial_modulation_j * wave_number_m**2 / mass
    )
    sigma_radial = math.sqrt(BOLTZMANN * temperature_k / mass) / omega_radial
    sigma_axial = math.sqrt(BOLTZMANN * temperature_k / mass) / omega_axial
    sigma_velocity = math.sqrt(BOLTZMANN * temperature_k / mass)

    radial_depth_j = potential_per_intensity_j * (
        intensity1_w_m2
        + intensity2_w_m2
        + 2.0 * math.sqrt(intensity1_w_m2 * intensity2_w_m2)
    )
    radial_curvature_weight = potential_per_intensity_j * (
        intensity1_w_m2 / waist1_m**2
        + intensity2_w_m2 / waist2_m**2
        + math.sqrt(intensity1_w_m2 * intensity2_w_m2)
        * (1.0 / waist1_m**2 + 1.0 / waist2_m**2)
    )
    effective_waist_m = math.sqrt(radial_depth_j / radial_curvature_weight)
    gravity_barrier_j = radial_depth_j
    gravity_minimum_j = -radial_depth_j
    gravity_sag_m = 0.0
    if inputs.include_gravity:
        gravity_barrier_j, gravity_minimum_j, gravity_sag_m = (
            gaussian_gravity_trap(radial_depth_j, effective_waist_m, mass)
        )

    lattice_spacing = math.pi / wave_number_m
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    sites: list[np.ndarray] = []
    accepted = 0
    total_drawn = 0
    while accepted < particle_count:
        batch = max(256, 2 * (particle_count - accepted))
        trial_positions = rng.normal(size=(batch, 3))
        trial_positions[:, 0] *= sigma_radial
        trial_positions[:, 1] *= sigma_radial
        trial_positions[:, 1] += gravity_sag_m
        trial_positions[:, 2] *= sigma_axial
        if inputs.cloud_axial_sigma_mm > 0.0:
            site_coordinate = rng.normal(
                scale=inputs.cloud_axial_sigma_mm * 1e-3,
                size=batch,
            )
            trial_sites = np.rint(site_coordinate / lattice_spacing)
            trial_positions[:, 2] += trial_sites * lattice_spacing
        else:
            trial_sites = np.zeros(batch, dtype=float)
        trial_velocities = rng.normal(scale=sigma_velocity, size=(batch, 3))
        potential, _, _, _ = _double_beam_potential_and_force(
            trial_positions,
            intensity1_w_m2=intensity1_w_m2,
            intensity2_w_m2=intensity2_w_m2,
            waist1_m=waist1_m,
            waist2_m=waist2_m,
            wave_number_m=wave_number_m,
            lattice_position_m=0.0,
            phase_rad=0.0,
            potential_per_intensity_j=potential_per_intensity_j,
        )
        kinetic = 0.5 * mass * np.einsum(
            "ij,ij->i",
            trial_velocities,
            trial_velocities,
        )
        excitation = kinetic + potential + axial_modulation_j
        bound = excitation < axial_modulation_j
        if inputs.include_gravity:
            radial_excitation = (
                kinetic
                + potential
                + mass * GRAVITY * trial_positions[:, 1]
                - gravity_minimum_j
            )
            bound &= (
                gravity_barrier_j > 0.0
            ) & (radial_excitation < gravity_barrier_j)
        if np.any(bound):
            positions.append(trial_positions[bound])
            velocities.append(trial_velocities[bound])
            sites.append(trial_sites[bound])
            accepted += int(np.count_nonzero(bound))
        total_drawn += batch
        if total_drawn > 1_000 * particle_count:
            raise ValueError(
                "当前温度/阱深下的初始束缚比例过低，无法稳定采样"
            )
    count = particle_count
    return ParticleEnsemble(
        positions_m=np.concatenate(positions, axis=0)[:count].copy(),
        velocities_m_s=np.concatenate(velocities, axis=0)[:count].copy(),
        weights=np.ones(count, dtype=float),
        site_index=(
            np.concatenate(sites, axis=0)[:count].astype(np.int64).copy()
        ),
        frame="l1_local",
    )


def ensemble_kinetic_temperature_uK(
    ensemble: ParticleEnsemble,
    atom_mass_kg: float,
) -> float:
    """系综的去质心三维动能温度 ``m⟨|v−⟨v⟩|²⟩/(3 k_B)``（µK）。

    统计诊断口径与 ``handover._kinetic_temperature_uK`` 一致（等权
    平均）；只用于验证初始采样温度 ≈ 设定值，不参与传播。
    """
    if not math.isfinite(atom_mass_kg) or atom_mass_kg <= 0.0:
        raise ValueError("原子质量必须是有限正数")
    _, velocities, _ = ensemble.host_arrays()
    centered = velocities - np.mean(velocities, axis=0)
    mean_speed2 = float((centered * centered).sum(axis=1).mean())
    return atom_mass_kg * mean_speed2 / (3.0 * BOLTZMANN) * 1e6
