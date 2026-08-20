from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

import continuous_loading.design_optimization as optimization
from continuous_loading.design_optimization import (
    RobustDesignInputs,
    _evaluate_robust_candidate,
    analyze_robust_design,
    load_design_optimization_configuration,
    plot_robust_design,
)


def _fake_handover(_parameters):
    return SimpleNamespace(
        transfer_efficiency=0.96,
        transfer_standard_error=0.01,
        final_temperature_uK=125.0,
        handover_heating_uK=5.0,
    )


def test_defaults_keep_timing_fixed_and_scan_only_hardware_variables():
    configuration = load_design_optimization_configuration()
    inputs = RobustDesignInputs()

    assert configuration["fixed_handover"]["time_us"] == pytest.approx(1000.0)
    assert inputs.relative_tolerance == pytest.approx(0.1)
    assert inputs.variation_mode == "one_at_a_time"
    assert inputs.detuning_points >= 2
    assert inputs.power_points >= 2
    assert inputs.waist_points >= 2


def test_joint_box_is_not_less_conservative_than_one_at_a_time():
    inputs = RobustDesignInputs(relative_tolerance=0.1)
    one_at_a_time = _evaluate_robust_candidate(
        inputs,
        detuning_ghz=850.0,
        source_power_w=4.75,
        waist_um=250.0,
    )
    joint_box = _evaluate_robust_candidate(
        replace(inputs, variation_mode="box_corners"),
        detuning_ghz=850.0,
        source_power_w=4.75,
        waist_um=250.0,
    )

    assert (
        joint_box.worst_constraint_margin
        <= one_at_a_time.worst_constraint_margin + 1e-12
    )


def test_small_robust_optimization_and_phase_scan(monkeypatch, tmp_path):
    monkeypatch.setattr(
        optimization,
        "run_handover_monte_carlo",
        _fake_handover,
    )
    inputs = RobustDesignInputs(
        detuning_min_ghz=750.0,
        detuning_max_ghz=950.0,
        detuning_points=3,
        source_power_min_w=3.5,
        source_power_max_w=5.5,
        power_points=3,
        waist_min_um=220.0,
        waist_max_um=280.0,
        waist_points=3,
        relative_tolerance=0.05,
        monte_carlo_candidate_count=2,
        minimum_transfer_efficiency=0.9,
        particle_count=8,
        time_step_us=1.0,
        phase_points=3,
        parallel_backend="serial",
        worker_count=1,
    )
    result = analyze_robust_design(inputs)

    assert result.robust_point_count > 0
    assert result.recommended is not None
    assert result.recommended.transfer_efficiency == pytest.approx(0.96)
    assert len(result.phase_scan) == 3
    assert result.phase_scan[0].phase_rad == pytest.approx(0.0)
    assert result.phase_scan[-1].phase_rad == pytest.approx(math.pi)
    assert result.phase_scan[0].expected_atom_number == pytest.approx(3_840_000.0)

    output = plot_robust_design(result, tmp_path / "robust.png")
    assert output.exists()
    assert output.stat().st_size > 0
