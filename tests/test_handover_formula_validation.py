import pytest

from continuous_loading.handover_formula_validation import (
    FormulaInputs,
    calculate_formula_results,
    make_trace_rows,
)


def _fast_inputs(**updates):
    values = {
        "samples": 40_000,
        "seed": 12345,
    }
    values.update(updates)
    return FormulaInputs(**values)


def test_original_closed_forms_match_independent_substitutions():
    result = calculate_formula_results(_fast_inputs())

    assert result.equation4_relative_residual < 1e-14
    assert result.equation5_relative_residual < 1e-14
    assert result.equation9_relative_residual < 1e-14
    assert result.equation11_relative_residual < 1e-14


def test_default_angle_formula_is_algebraically_right_but_physically_invalid():
    result = calculate_formula_results(_fast_inputs())

    assert result.angle_locality_parameter > 10.0
    assert result.geometric_shift_in_lattice_periods > 4.0
    assert not result.single_site_angle_approximation_valid
    assert result.delta_t_parallel_original_uK > 1e5
    assert result.original_angle_energy_over_depth > 200.0

    # 完整周期势的固定零相位高斯平均不超过 U/2；换算三维等效温度
    # 后不超过 U/(6 k_B)。
    assert (
        result.periodic_mean_energy_uK
        <= result.inputs.depth_uK / 2.0 * (1.0 + 1e-12)
    )
    assert (
        result.periodic_equivalent_temperature_3d_uK
        <= result.inputs.depth_uK / 6.0 * (1.0 + 1e-12)
    )

    # 原版势深需求与输入 500 µK 深度相差约三个数量级。
    power_ratio = (
        result.required_total_power_eq11_w
        / result.total_power_for_input_depth_w
    )
    assert power_ratio > 1_000.0


def test_monte_carlo_checks_thermal_and_periodic_averages():
    result = calculate_formula_results(_fast_inputs(samples=120_000))

    assert result.mc_y2_relative_error < 0.02
    assert result.mc_periodic_relative_error < 0.02
    assert result.mc_harmonic_angle_temperature_uK == pytest.approx(
        result.delta_t_parallel_original_uK,
        rel=0.02,
    )


def test_zero_angle_removes_angle_heating_and_trace_has_correct_endpoints():
    inputs = _fast_inputs(angle_deg=0.0)
    result = calculate_formula_results(inputs)
    rows = make_trace_rows(inputs, result, points=5)

    assert result.angle_locality_parameter == 0.0
    assert result.delta_t_parallel_original_uK == 0.0
    assert result.periodic_mean_energy_uK == 0.0
    assert result.single_site_angle_approximation_valid
    assert rows[0]["fraction_lattice1"] == pytest.approx(1.0)
    assert rows[0]["fraction_lattice2"] == pytest.approx(0.0)
    assert rows[-1]["fraction_lattice1"] == pytest.approx(0.0)
    assert rows[-1]["fraction_lattice2"] == pytest.approx(1.0)
    assert rows[-1]["frequency_perp_hz"] == pytest.approx(
        result.radial_frequency_final_hz
    )
