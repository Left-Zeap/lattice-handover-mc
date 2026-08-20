"""数值计算并交叉验证 ``handover升温分析.pdf`` 的式 (1)-(11)。

这个模块有两个目的：

1. 按原 PDF 的定义逐式计算阱频、突然切换温升、经验囚禁深度以及
   二能级功率-失谐关系；
2. 同时给出适用性诊断和完整周期势的有界对照，帮助区分“代数公式
   算对了”和“公式在当前参数下物理上可以使用”。

运行：

    python -m continuous_loading.handover_formula_validation

所有内部计算使用 SI 单位；命令行输入和主要输出使用 nm、GHz、µm、
µK、ms 等实验常用单位。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from .atomic import CS133
from .constants import (
    ATOMIC_MASS_UNIT,
    BOLTZMANN,
    HBAR,
    SPEED_OF_LIGHT,
)


@dataclass(frozen=True)
class FormulaInputs:
    """原 PDF 公式所需的物理参数。"""

    mass_u: float = CS133.mass_u
    resonance_wavelength_nm: float = CS133.d1.wavelength_nm
    linewidth_over_2pi_mhz: float = CS133.d1.linewidth_over_2pi_mhz
    red_detuning_ghz: float = 600.0
    depth_uK: float = 500.0
    initial_temperature_uK: float = 30.8
    waist1_um: float = 250.0
    waist2_um: float = 150.0
    angle_deg: float = 4.0
    duration_ms: float = 1.0
    safety_factor: float = 5.0
    total_laser_power_w: float = 5.0
    optical_efficiency: float = 0.70
    beam_count: int = 4
    samples: int = 200_000
    seed: int = 20_260_729

    def __post_init__(self) -> None:
        positive = {
            "原子质量": self.mass_u,
            "共振波长": self.resonance_wavelength_nm,
            "线宽": self.linewidth_over_2pi_mhz,
            "红失谐": self.red_detuning_ghz,
            "势阱深度": self.depth_uK,
            "初始温度": self.initial_temperature_uK,
            "Lattice-1 束腰": self.waist1_um,
            "Lattice-2 束腰": self.waist2_um,
            "交接时间": self.duration_ms,
            "安全因子": self.safety_factor,
            "激光器总功率": self.total_laser_power_w,
            "光路效率": self.optical_efficiency,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name}必须是有限正数")
        if not math.isfinite(self.angle_deg) or self.angle_deg < 0.0:
            raise ValueError("夹角必须是有限非负数")
        if self.optical_efficiency > 1.0:
            raise ValueError("光路效率必须位于 (0, 1]")
        if self.beam_count <= 0:
            raise ValueError("均分光束数必须为正整数")
        if self.samples <= 0:
            raise ValueError("Monte Carlo 样本数必须为正整数")


@dataclass(frozen=True)
class FormulaResults:
    """原版公式、独立回代验证和物理适用性诊断。"""

    inputs: FormulaInputs

    # 基础量
    mass_kg: float
    resonance_frequency_hz: float
    resonance_angular_frequency_rad_s: float
    linewidth_rad_s: float
    detuning_angular_rad_s: float
    laser_wavelength_nm: float
    wave_number_m: float
    lattice_spacing_nm: float
    depth_j: float

    # PDF 式 (1)
    omega_perp_initial_rad_s: float
    omega_perp_final_rad_s: float
    omega_parallel_rad_s: float
    radial_frequency_initial_hz: float
    radial_frequency_final_hz: float
    axial_frequency_hz: float

    # PDF 式 (2)-(3)
    y_rms_um: float
    x0_rms_final_um: float
    omega_perp_midpoint_rad_s: float

    # PDF 式 (4)
    delta_t_perp_original_uK: float
    delta_t_perp_direct_uK: float
    delta_t_perp_3d_rethermalized_uK: float
    equation4_relative_residual: float

    # PDF 式 (5)
    delta_t_parallel_original_uK: float
    delta_t_parallel_direct_uK: float
    equation5_relative_residual: float
    angle_locality_parameter: float
    geometric_shift_in_lattice_periods: float
    original_angle_energy_over_depth: float

    # 完整周期势的有界对照
    periodic_mean_energy_uK: float
    periodic_equivalent_temperature_3d_uK: float
    periodic_uniform_phase_equivalent_temperature_3d_uK: float
    periodic_max_equivalent_temperature_3d_uK: float

    # Monte Carlo 对照
    mc_y2_relative_error: float
    mc_harmonic_angle_temperature_uK: float
    mc_periodic_equivalent_temperature_3d_uK: float
    mc_periodic_relative_error: float

    # PDF 式 (6)-(7)
    delta_t_total_original_uK: float
    final_temperature_original_uK: float
    required_depth_original_uK: float
    bound_fraction_at_safety_factor: float

    # 有限时间诊断
    omega_parallel_tau: float
    omega_perp_initial_tau: float
    radial_adiabaticity_max: float
    true_moving_center_suppression_factor: float
    true_moving_center_finite_energy_over_kB_uK: float

    # PDF 式 (8)-(11)
    single_beam_power_at_atoms_w: float
    single_beam_peak_intensity_w_m2: float
    achieved_depth_eq8_uK: float
    achieved_depth_eq9_uK: float
    equation9_relative_residual: float
    achieved_equivalent_temperature_eq10_uK: float
    required_power_per_detuning_w_per_hz_eq11: float
    required_total_power_eq11_w: float
    required_total_power_by_inverting_eq9_w: float
    equation11_relative_residual: float
    total_power_for_input_depth_w: float

    # 二能级散射数量级
    scattering_rate_at_input_depth_s: float
    scattering_rate_at_original_required_depth_s: float
    mean_scattering_events_during_handover_at_input_depth: float

    # 结论标志
    single_site_angle_approximation_valid: bool
    sudden_axial_approximation_valid: bool
    original_required_depth_exceeds_one_kelvin: bool


def _relative_residual(first: float, second: float) -> float:
    scale = max(abs(first), abs(second), 1e-300)
    return abs(first - second) / scale


def _thermal_bound_fraction(eta: float) -> float:
    return 1.0 - math.exp(-eta) * (1.0 + eta + 0.5 * eta**2)


def calculate_formula_results(inputs: FormulaInputs) -> FormulaResults:
    """计算原 PDF 式 (1)-(11) 以及物理适用性诊断。"""
    mass = inputs.mass_u * ATOMIC_MASS_UNIT
    lambda_res = inputs.resonance_wavelength_nm * 1e-9
    frequency_res = SPEED_OF_LIGHT / lambda_res
    omega_res = 2.0 * math.pi * frequency_res
    gamma = 2.0 * math.pi * inputs.linewidth_over_2pi_mhz * 1e6
    detuning_hz = inputs.red_detuning_ghz * 1e9
    detuning_angular = 2.0 * math.pi * detuning_hz
    laser_frequency = frequency_res - detuning_hz
    if laser_frequency <= 0.0:
        raise ValueError("红失谐过大，导致激光频率不为正")
    laser_wavelength = SPEED_OF_LIGHT / laser_frequency
    wave_number = 2.0 * math.pi / laser_wavelength

    depth_j = inputs.depth_uK * 1e-6 * BOLTZMANN
    temperature_k = inputs.initial_temperature_uK * 1e-6
    waist1 = inputs.waist1_um * 1e-6
    waist2 = inputs.waist2_um * 1e-6
    theta = math.radians(inputs.angle_deg)
    duration = inputs.duration_ms * 1e-3

    # PDF 式 (1)
    omega_perp_i = math.sqrt(4.0 * depth_j / (mass * waist1**2))
    omega_perp_f = math.sqrt(4.0 * depth_j / (mass * waist2**2))
    omega_parallel = wave_number * math.sqrt(2.0 * depth_j / mass)

    # PDF 式 (2)-(3)
    y_variance = BOLTZMANN * temperature_k / (mass * omega_perp_i**2)
    y_rms = math.sqrt(y_variance)
    x0_rms = theta * y_rms
    omega_perp_midpoint = math.sqrt(
        0.5 * omega_perp_i**2 + 0.5 * omega_perp_f**2
    )

    # PDF 式 (4)：闭式结果与从能量均分直接回代的独立结果。
    frequency_ratio2 = (omega_perp_f / omega_perp_i) ** 2
    delta_t_perp_original = (
        0.5 * (inputs.waist1_um**2 / inputs.waist2_um**2 - 1.0)
        * inputs.initial_temperature_uK
    )
    delta_e_per_radial_dof = (
        0.5
        * BOLTZMANN
        * temperature_k
        * (frequency_ratio2 - 1.0)
    )
    delta_t_perp_direct = (
        delta_e_per_radial_dof / BOLTZMANN * 1e6
    )
    delta_t_perp_3d = (
        2.0 * delta_e_per_radial_dof / (3.0 * BOLTZMANN) * 1e6
    )

    # PDF 式 (5)：闭式结果与 <y^2> 回代的独立结果。
    delta_t_parallel_original = (
        0.25
        * wave_number**2
        * waist1**2
        * theta**2
        * inputs.initial_temperature_uK
    )
    angle_energy_direct = (
        0.5 * mass * omega_parallel**2 * theta**2 * y_variance
    )
    delta_t_parallel_direct = (
        angle_energy_direct / BOLTZMANN * 1e6
    )
    locality = wave_number * theta * y_rms
    periods = x0_rms / (0.5 * laser_wavelength)

    # 对 y~N(0,sigma^2)，<sin^2(k theta y)> =
    # [1-exp(-2(k theta sigma)^2)]/2。
    periodic_mean_energy = (
        0.5 * depth_j * (1.0 - math.exp(-2.0 * locality**2))
    )
    periodic_mean_energy_uK = (
        periodic_mean_energy / BOLTZMANN * 1e6
    )
    periodic_equivalent_temperature = (
        periodic_mean_energy / (3.0 * BOLTZMANN) * 1e6
    )

    # Monte Carlo 只用于验证热平均和周期势平均，不参与主闭式结果。
    rng = np.random.default_rng(inputs.seed)
    y_samples = rng.normal(scale=y_rms, size=inputs.samples)
    mc_y2 = float(np.mean(y_samples**2))
    mc_harmonic_energy = float(
        np.mean(
            0.5
            * mass
            * omega_parallel**2
            * (theta * y_samples) ** 2
        )
    )
    mc_periodic_energy = float(
        np.mean(depth_j * np.sin(wave_number * theta * y_samples) ** 2)
    )
    mc_periodic_temperature = (
        mc_periodic_energy / (3.0 * BOLTZMANN) * 1e6
    )

    # PDF 式 (6)-(7)
    delta_t_total_original = (
        delta_t_perp_original + delta_t_parallel_original
    )
    final_temperature_original = (
        inputs.initial_temperature_uK + delta_t_total_original
    )
    required_depth_original = (
        inputs.safety_factor * final_temperature_original
    )
    bound_fraction = _thermal_bound_fraction(inputs.safety_factor)

    # 有限时间适用性诊断。
    omega_parallel_tau = omega_parallel * duration
    omega_perp_i_tau = omega_perp_i * duration
    delta_omega2 = abs(omega_perp_f**2 - omega_perp_i**2)
    omega_min = min(omega_perp_i, omega_perp_f)
    radial_adiabaticity = (
        delta_omega2 / (2.0 * duration * omega_min**3)
    )
    if omega_parallel_tau == 0.0:
        center_suppression = 1.0
    else:
        center_suppression = (
            4.0
            * math.sin(0.5 * omega_parallel_tau) ** 2
            / omega_parallel_tau**2
        )

    # PDF 式 (8)-(10)。Eq. (9) 的简化严格要求 beam_count=4；
    # 对其他 beam_count，achieved_depth_eq9 保留 Eq. (8) 代入后的通式。
    single_beam_power = (
        inputs.optical_efficiency
        * inputs.total_laser_power_w
        / inputs.beam_count
    )
    single_beam_intensity = 2.0 * single_beam_power / (
        math.pi * waist1**2
    )
    achieved_depth_eq8_j = (
        6.0
        * math.pi
        * SPEED_OF_LIGHT**2
        * gamma
        * single_beam_intensity
        / (omega_res**3 * detuning_angular)
    )
    achieved_depth_substituted_j = (
        12.0
        * SPEED_OF_LIGHT**2
        * gamma
        * inputs.optical_efficiency
        * inputs.total_laser_power_w
        / (
            inputs.beam_count
            * omega_res**3
            * detuning_angular
            * waist1**2
        )
    )

    # PDF 式 (11)，温度必须先从 µK 转为 K。
    final_temperature_original_k = final_temperature_original * 1e-6
    required_power_per_detuning = (
        2.0
        * math.pi
        * BOLTZMANN
        * inputs.safety_factor
        * final_temperature_original_k
        * omega_res**3
        * waist1**2
        / (
            3.0
            * SPEED_OF_LIGHT**2
            * gamma
            * inputs.optical_efficiency
        )
    )
    required_total_power_eq11 = (
        required_power_per_detuning * detuning_hz
    )

    # 将 Eq. (9) 推广到任意 beam_count 后直接反解，作为 Eq. (11)
    # 的独立回代验证。beam_count=4 时与原 PDF 完全一致。
    required_depth_j = (
        required_depth_original * 1e-6 * BOLTZMANN
    )
    required_total_power_inverted = (
        required_depth_j
        * inputs.beam_count
        * omega_res**3
        * detuning_angular
        * waist1**2
        / (
            12.0
            * SPEED_OF_LIGHT**2
            * gamma
            * inputs.optical_efficiency
        )
    )
    input_depth_power = (
        depth_j
        * inputs.beam_count
        * omega_res**3
        * detuning_angular
        * waist1**2
        / (
            12.0
            * SPEED_OF_LIGHT**2
            * gamma
            * inputs.optical_efficiency
        )
    )

    # 旋波、二能级近似下 Gamma_sc = Gamma U/(hbar |Delta|)。
    scattering_at_input = gamma * depth_j / (
        HBAR * detuning_angular
    )
    scattering_at_required = gamma * required_depth_j / (
        HBAR * detuning_angular
    )

    return FormulaResults(
        inputs=inputs,
        mass_kg=mass,
        resonance_frequency_hz=frequency_res,
        resonance_angular_frequency_rad_s=omega_res,
        linewidth_rad_s=gamma,
        detuning_angular_rad_s=detuning_angular,
        laser_wavelength_nm=laser_wavelength * 1e9,
        wave_number_m=wave_number,
        lattice_spacing_nm=0.5 * laser_wavelength * 1e9,
        depth_j=depth_j,
        omega_perp_initial_rad_s=omega_perp_i,
        omega_perp_final_rad_s=omega_perp_f,
        omega_parallel_rad_s=omega_parallel,
        radial_frequency_initial_hz=omega_perp_i / (2.0 * math.pi),
        radial_frequency_final_hz=omega_perp_f / (2.0 * math.pi),
        axial_frequency_hz=omega_parallel / (2.0 * math.pi),
        y_rms_um=y_rms * 1e6,
        x0_rms_final_um=x0_rms * 1e6,
        omega_perp_midpoint_rad_s=omega_perp_midpoint,
        delta_t_perp_original_uK=delta_t_perp_original,
        delta_t_perp_direct_uK=delta_t_perp_direct,
        delta_t_perp_3d_rethermalized_uK=delta_t_perp_3d,
        equation4_relative_residual=_relative_residual(
            delta_t_perp_original,
            delta_t_perp_direct,
        ),
        delta_t_parallel_original_uK=delta_t_parallel_original,
        delta_t_parallel_direct_uK=delta_t_parallel_direct,
        equation5_relative_residual=_relative_residual(
            delta_t_parallel_original,
            delta_t_parallel_direct,
        ),
        angle_locality_parameter=locality,
        geometric_shift_in_lattice_periods=periods,
        original_angle_energy_over_depth=(
            delta_t_parallel_original / inputs.depth_uK
        ),
        periodic_mean_energy_uK=periodic_mean_energy_uK,
        periodic_equivalent_temperature_3d_uK=(
            periodic_equivalent_temperature
        ),
        periodic_uniform_phase_equivalent_temperature_3d_uK=(
            inputs.depth_uK / 6.0
        ),
        periodic_max_equivalent_temperature_3d_uK=(
            inputs.depth_uK / 3.0
        ),
        mc_y2_relative_error=_relative_residual(mc_y2, y_variance),
        mc_harmonic_angle_temperature_uK=(
            mc_harmonic_energy / BOLTZMANN * 1e6
        ),
        mc_periodic_equivalent_temperature_3d_uK=(
            mc_periodic_temperature
        ),
        mc_periodic_relative_error=_relative_residual(
            mc_periodic_temperature,
            periodic_equivalent_temperature,
        ),
        delta_t_total_original_uK=delta_t_total_original,
        final_temperature_original_uK=final_temperature_original,
        required_depth_original_uK=required_depth_original,
        bound_fraction_at_safety_factor=bound_fraction,
        omega_parallel_tau=omega_parallel_tau,
        omega_perp_initial_tau=omega_perp_i_tau,
        radial_adiabaticity_max=radial_adiabaticity,
        true_moving_center_suppression_factor=center_suppression,
        true_moving_center_finite_energy_over_kB_uK=(
            delta_t_parallel_original * center_suppression
        ),
        single_beam_power_at_atoms_w=single_beam_power,
        single_beam_peak_intensity_w_m2=single_beam_intensity,
        achieved_depth_eq8_uK=(
            achieved_depth_eq8_j / BOLTZMANN * 1e6
        ),
        achieved_depth_eq9_uK=(
            achieved_depth_substituted_j / BOLTZMANN * 1e6
        ),
        equation9_relative_residual=_relative_residual(
            achieved_depth_eq8_j,
            achieved_depth_substituted_j,
        ),
        achieved_equivalent_temperature_eq10_uK=(
            achieved_depth_eq8_j / BOLTZMANN * 1e6
        ),
        required_power_per_detuning_w_per_hz_eq11=(
            required_power_per_detuning
        ),
        required_total_power_eq11_w=required_total_power_eq11,
        required_total_power_by_inverting_eq9_w=(
            required_total_power_inverted
        ),
        equation11_relative_residual=_relative_residual(
            required_total_power_eq11,
            required_total_power_inverted,
        ),
        total_power_for_input_depth_w=input_depth_power,
        scattering_rate_at_input_depth_s=scattering_at_input,
        scattering_rate_at_original_required_depth_s=(
            scattering_at_required
        ),
        mean_scattering_events_during_handover_at_input_depth=(
            scattering_at_input * duration
        ),
        # ``<<1`` 没有唯一数值边界；0.3 是明确且偏保守的诊断阈值。
        single_site_angle_approximation_valid=locality < 0.3,
        sudden_axial_approximation_valid=omega_parallel_tau < 0.1,
        original_required_depth_exceeds_one_kelvin=(
            required_depth_original >= 1e6
        ),
    )


def make_trace_rows(
    inputs: FormulaInputs,
    results: FormulaResults,
    *,
    points: int = 101,
) -> list[dict[str, float]]:
    """生成式 (2)-(3) 的时间轨迹和式 (5) 的累计突然估计。"""
    if points < 2:
        raise ValueError("轨迹点数至少为 2")
    rows: list[dict[str, float]] = []
    for fraction in np.linspace(0.0, 1.0, points):
        omega2 = (
            results.omega_perp_initial_rad_s**2 * (1.0 - fraction)
            + results.omega_perp_final_rad_s**2 * fraction
        )
        rows.append(
            {
                "time_ms": fraction * inputs.duration_ms,
                "fraction_lattice1": 1.0 - fraction,
                "fraction_lattice2": fraction,
                "omega_perp_rad_s": math.sqrt(omega2),
                "frequency_perp_hz": math.sqrt(omega2)
                / (2.0 * math.pi),
                "conditional_x0_rms_um": (
                    fraction * results.x0_rms_final_um
                ),
                "original_harmonic_angle_energy_over_kB_uK": (
                    fraction**2
                    * results.delta_t_parallel_original_uK
                ),
            }
        )
    return rows


def _save_json(path: str | Path, results: FormulaResults) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def _save_csv(
    path: str | Path,
    rows: list[dict[str, float]],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return output


def _save_plot(
    path: str | Path,
    inputs: FormulaInputs,
    results: FormulaResults,
    rows: list[dict[str, float]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    time_ms = np.array([row["time_ms"] for row in rows])
    frequency_hz = np.array([row["frequency_perp_hz"] for row in rows])
    x0_um = np.array([row["conditional_x0_rms_um"] for row in rows])

    angle_max = max(8.0, 1.25 * inputs.angle_deg)
    angles_deg = np.linspace(0.0, angle_max, 240)
    angles_rad = np.deg2rad(angles_deg)
    sigma_y_m = results.y_rms_um * 1e-6
    harmonic_uK = (
        0.5
        * results.mass_kg
        * results.omega_parallel_rad_s**2
        * angles_rad**2
        * sigma_y_m**2
        / BOLTZMANN
        * 1e6
    )
    chi = results.wave_number_m * angles_rad * sigma_y_m
    periodic_eq_uK = (
        inputs.depth_uK
        / 6.0
        * (1.0 - np.exp(-2.0 * chi**2))
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
    axes[0].plot(time_ms, frequency_hz, color="#1f77b4")
    axes[0].set_xlabel("handover time (ms)")
    axes[0].set_ylabel("radial frequency (Hz)")
    axes[0].set_title("PDF Eq. (3)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(time_ms, x0_um, color="#2ca02c")
    axes[1].set_xlabel("handover time (ms)")
    axes[1].set_ylabel("conditional x0 RMS (um)")
    axes[1].set_title("PDF Eq. (2), not global minimum")
    axes[1].grid(alpha=0.25)

    axes[2].semilogy(
        angles_deg,
        np.maximum(harmonic_uK, 1e-12),
        label="original harmonic energy / kB",
        color="#d62728",
    )
    axes[2].semilogy(
        angles_deg,
        np.maximum(periodic_eq_uK, 1e-12),
        label="periodic 3D equivalent T",
        color="#9467bd",
    )
    axes[2].axvline(inputs.angle_deg, color="#555555", linestyle="--")
    axes[2].set_xlabel("crossing angle (deg)")
    axes[2].set_ylabel("temperature scale (uK)")
    axes[2].set_title("Eq. (5) validity check")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _status(condition: bool) -> str:
    return "PASS" if condition else "WARN"


def print_report(results: FormulaResults) -> None:
    """打印面向人工核对的逐式报告。"""
    p = results.inputs
    print("handover升温分析.pdf 式 (1)-(11) 数值验证")
    print("=" * 66)
    print(
        f"输入: Cs-like m={p.mass_u:.9f} u, "
        f"T_i={p.initial_temperature_uK:.3f} uK, "
        f"U/kB={p.depth_uK:.3f} uK"
    )
    print(
        f"      w01/w02={p.waist1_um:.3f}/{p.waist2_um:.3f} um, "
        f"angle={p.angle_deg:.4f} deg, tau={p.duration_ms:.4f} ms"
    )
    print(
        f"      D1 red detuning={p.red_detuning_ghz:.3f} GHz, "
        f"laser wavelength={results.laser_wavelength_nm:.6f} nm"
    )

    print("\n[Eq. (1)] trap frequencies")
    print(
        f"  radial initial/final = "
        f"{results.radial_frequency_initial_hz:.3f}/"
        f"{results.radial_frequency_final_hz:.3f} Hz"
    )
    print(f"  axial = {results.axial_frequency_hz / 1e3:.3f} kHz")

    print("\n[Eq. (2)-(3)] conditional minimum and time-varying frequency")
    print(f"  y_rms = {results.y_rms_um:.6f} um")
    print(
        f"  conditional x0_rms(t=tau) = "
        f"{results.x0_rms_final_um:.6f} um"
    )
    print(
        f"  x0 shift / lattice period = "
        f"{results.geometric_shift_in_lattice_periods:.6f}"
    )
    print(
        "  note: x0(y,t) is a fixed-y conditional minimum, "
        "not the global 2D trap center"
    )

    print("\n[Eq. (4)] sudden radial-frequency change")
    print(
        f"  original DeltaT_perp = "
        f"{results.delta_t_perp_original_uK:.6f} uK"
    )
    print(
        f"  direct equipartition check = "
        f"{results.delta_t_perp_direct_uK:.6f} uK, "
        f"residual={results.equation4_relative_residual:.3e}"
    )
    print(
        f"  corrected 3D rethermalized value = "
        f"{results.delta_t_perp_3d_rethermalized_uK:.6f} uK"
    )

    print("\n[Eq. (5)] angle heating and validity")
    print(
        f"  original DeltaT_parallel = "
        f"{results.delta_t_parallel_original_uK:.6f} uK"
    )
    print(
        f"  direct <y^2> check = "
        f"{results.delta_t_parallel_direct_uK:.6f} uK, "
        f"residual={results.equation5_relative_residual:.3e}"
    )
    print(
        f"  k*theta*sigma_y = {results.angle_locality_parameter:.6f} "
        f"[{_status(results.single_site_angle_approximation_valid)}; "
        "single-site harmonic needs << 1]"
    )
    print(
        f"  original angle energy / depth = "
        f"{results.original_angle_energy_over_depth:.6f}"
    )
    print(
        f"  bounded periodic <DeltaE>/kB = "
        f"{results.periodic_mean_energy_uK:.6f} uK"
    )
    print(
        f"  bounded periodic 3D equivalent DeltaT = "
        f"{results.periodic_equivalent_temperature_3d_uK:.6f} uK"
    )
    print(
        f"  Monte Carlo harmonic/periodic = "
        f"{results.mc_harmonic_angle_temperature_uK:.6f}/"
        f"{results.mc_periodic_equivalent_temperature_3d_uK:.6f} uK"
    )

    print("\n[Eq. (6)-(7)] original total and heuristic confinement")
    print(
        f"  original DeltaT_total = "
        f"{results.delta_t_total_original_uK:.6f} uK"
    )
    print(
        f"  original T_final = "
        f"{results.final_temperature_original_uK:.6f} uK"
    )
    print(
        f"  alpha*T_final required depth = "
        f"{results.required_depth_original_uK:.6f} uK"
    )
    print(
        f"  equilibrium P_bound(alpha={p.safety_factor:g}) = "
        f"{results.bound_fraction_at_safety_factor:.6f}"
    )

    print("\n[finite-time diagnostics]")
    print(
        f"  omega_parallel*tau = {results.omega_parallel_tau:.6f} "
        f"[{_status(results.sudden_axial_approximation_valid)}; "
        "sudden needs << 1]"
    )
    print(
        f"  omega_perp_i*tau = "
        f"{results.omega_perp_initial_tau:.6f}"
    )
    print(
        f"  max |omega_dot/omega^2| = "
        f"{results.radial_adiabaticity_max:.6e}"
    )
    print(
        "  counterfactual true-moving-center finite/sudden factor = "
        f"{results.true_moving_center_suppression_factor:.6e}"
    )

    print("\n[Eq. (8)-(10)] two-level depth from laser power")
    print(
        f"  source total power / efficiency / beam count = "
        f"{p.total_laser_power_w:.6f} W / "
        f"{p.optical_efficiency:.6f} / {p.beam_count}"
    )
    print(
        f"  single-beam power at atoms = "
        f"{results.single_beam_power_at_atoms_w:.6f} W"
    )
    print(
        f"  single-beam peak intensity I0 = "
        f"{results.single_beam_peak_intensity_w_m2:.6e} W/m^2"
    )
    print(
        f"  Eq. (8)/(9) achieved depth = "
        f"{results.achieved_depth_eq8_uK:.6f}/"
        f"{results.achieved_depth_eq9_uK:.6f} uK, "
        f"residual={results.equation9_relative_residual:.3e}"
    )

    print("\n[Eq. (11)] required laser power")
    print(
        f"  P_L/Delta_nu >= "
        f"{results.required_power_per_detuning_w_per_hz_eq11:.6e} W/Hz"
    )
    print(
        f"  required total power at {p.red_detuning_ghz:g} GHz = "
        f"{results.required_total_power_eq11_w:.6e} W"
    )
    print(
        f"  Eq. (9) inversion check = "
        f"{results.required_total_power_by_inverting_eq9_w:.6e} W, "
        f"residual={results.equation11_relative_residual:.3e}"
    )
    print(
        f"  power for the independently specified {p.depth_uK:g} uK depth = "
        f"{results.total_power_for_input_depth_w:.6f} W"
    )

    print("\n[two-level scattering scale]")
    print(
        f"  at input depth: "
        f"{results.scattering_rate_at_input_depth_s:.6f} 1/s"
    )
    print(
        f"  mean events in tau: "
        f"{results.mean_scattering_events_during_handover_at_input_depth:.6f}"
    )
    print(
        f"  at original Eq. (7) required depth: "
        f"{results.scattering_rate_at_original_required_depth_s:.6e} 1/s"
    )

    print("\nSummary")
    if results.single_site_angle_approximation_valid:
        print("  PASS: Eq. (5) single-site locality diagnostic is small.")
    else:
        print(
            "  WARN: Eq. (5) is algebraically reproduced but its "
            "single-site harmonic condition fails."
        )
    if results.sudden_axial_approximation_valid:
        print("  PASS: axial sudden-switch diagnostic is small.")
    else:
        print(
            "  WARN: the requested handover is not in the axial sudden limit."
        )
    if p.beam_count != 4:
        print(
            "  WARN: original PDF Eq. (9)-(11) assumes exactly four "
            "equally split beams; generalized checks were used."
        )
    print(
        "  NOTE: matching Eq. (4)/(5) residuals proves algebraic "
        "consistency, not physical validity."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "数值计算并交叉验证 handover升温分析.pdf 的式 (1)-(11)"
        )
    )
    parser.add_argument("--mass-u", type=float, default=CS133.mass_u)
    parser.add_argument(
        "--resonance-wavelength",
        type=float,
        default=CS133.d1.wavelength_nm,
        help="二能级共振波长 nm",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=CS133.d1.linewidth_over_2pi_mhz,
        help="Gamma/(2*pi), MHz",
    )
    parser.add_argument(
        "--detuning",
        type=float,
        default=600.0,
        help="红失谐量 GHz",
    )
    parser.add_argument("--depth", type=float, default=500.0, help="势深 uK")
    parser.add_argument(
        "--temperature",
        type=float,
        default=30.8,
        help="初始温度 uK",
    )
    parser.add_argument("--waist1", type=float, default=250.0, help="w01, um")
    parser.add_argument("--waist2", type=float, default=150.0, help="w02, um")
    parser.add_argument("--angle", type=float, default=4.0, help="夹角 deg")
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="handover 时间 ms",
    )
    parser.add_argument("--alpha", type=float, default=5.0, help="安全因子")
    parser.add_argument(
        "--laser-power",
        type=float,
        default=5.0,
        help="PDF 定义的激光器输出总功率 W",
    )
    parser.add_argument(
        "--efficiency",
        type=float,
        default=0.70,
        help="从总功率到四束总功率的效率",
    )
    parser.add_argument(
        "--beam-count",
        type=int,
        default=4,
        help="功率均分光束数；原 PDF 固定为 4",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200_000,
        help="热平均 Monte Carlo 样本数",
    )
    parser.add_argument("--seed", type=int, default=20_260_729)
    parser.add_argument("--trace-points", type=int, default=101)
    parser.add_argument("--json", help="保存全部结果到 JSON")
    parser.add_argument("--csv", help="保存式 (2)-(3) 时间轨迹到 CSV")
    parser.add_argument("--plot", help="保存诊断图")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = FormulaInputs(
        mass_u=args.mass_u,
        resonance_wavelength_nm=args.resonance_wavelength,
        linewidth_over_2pi_mhz=args.linewidth,
        red_detuning_ghz=args.detuning,
        depth_uK=args.depth,
        initial_temperature_uK=args.temperature,
        waist1_um=args.waist1,
        waist2_um=args.waist2,
        angle_deg=args.angle,
        duration_ms=args.duration,
        safety_factor=args.alpha,
        total_laser_power_w=args.laser_power,
        optical_efficiency=args.efficiency,
        beam_count=args.beam_count,
        samples=args.samples,
        seed=args.seed,
    )
    results = calculate_formula_results(inputs)
    rows = make_trace_rows(inputs, results, points=args.trace_points)
    print_report(results)

    if args.json:
        saved = _save_json(args.json, results)
        print(f"\nJSON saved: {saved}")
    if args.csv:
        saved = _save_csv(args.csv, rows)
        print(f"CSV saved: {saved}")
    if args.plot:
        saved = _save_plot(args.plot, inputs, results, rows)
        print(f"Plot saved: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
