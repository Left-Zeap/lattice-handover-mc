import math
from dataclasses import replace

import numpy as np
import pytest

from continuous_loading.l1_transport import (
    L1TransportInputs,
    analyze_l1_transport_scan,
    l1_timing,
    l1_transport_inputs_for_species,
    plot_l1_transport_scan,
    simulate_l1_transport,
)


def test_trapezoidal_timing_matches_distance_and_default_scale():
    inputs = L1TransportInputs()
    timing = l1_timing(inputs)

    assert math.isclose(
        timing.acceleration_time_s,
        inputs.maximum_velocity_m_s / inputs.acceleration_m_s2,
    )
    reconstructed_distance = (
        inputs.maximum_velocity_m_s * timing.cruise_time_s
        + inputs.maximum_velocity_m_s**2 / inputs.acceleration_m_s2
    )
    assert math.isclose(reconstructed_distance, inputs.distance_m)
    assert 0.049 < timing.total_time_s < 0.051
    assert inputs.detuning_min_ghz == 100.0
    assert inputs.detuning_max_ghz == 800.0
    assert inputs.handover_source_power_min_w == 0.0
    assert inputs.handover_source_power_max_w == 1.5
    assert inputs.initial_temperature_uK == 20.0


def test_calibrated_l1_gaussian_geometry_and_diameter_interface():
    inputs = L1TransportInputs()

    assert inputs.calibrated_gaussian_geometry
    assert inputs.start_beam_diameter_um == 660.0
    assert inputs.minimum_waist_um == 250.0
    assert inputs.minimum_waist_position_m == 0.2
    assert inputs.effective_rayleigh_range_m == pytest.approx(0.2321191727)
    assert inputs.beam_radius_um_at(0.0) == pytest.approx(330.0)
    assert inputs.beam_radius_um_at(0.2) == pytest.approx(250.0)
    assert inputs.handover_waist_um == pytest.approx(323.0727472)
    assert inputs.beam_diameter_um_at(inputs.distance_m) == pytest.approx(
        646.1454944
    )

    trace = simulate_l1_transport(replace(inputs, time_points=41), 300.0, 1.0)
    assert trace.waist_um[0] == pytest.approx(330.0)
    assert trace.waist_um[-1] == pytest.approx(inputs.handover_waist_um)
    assert trace.beam_diameter_um == pytest.approx(
        tuple(2.0 * value for value in trace.waist_um)
    )


def test_calibrated_l1_geometry_rejects_inconsistent_inputs():
    with pytest.raises(ValueError, match="最小束腰位置"):
        replace(L1TransportInputs(), minimum_waist_position_m=0.4)
    with pytest.raises(ValueError, match="起点光束半径"):
        replace(L1TransportInputs(), minimum_waist_um=340.0)


def test_single_point_keeps_source_power_fixed_and_monotone_retention():
    inputs = L1TransportInputs(time_points=61)
    trace = simulate_l1_transport(inputs, 300.0, 1.5)

    assert all(math.isclose(power, 1.5) for power in trace.source_power_w)
    assert trace.point.final_temperature_uK > inputs.initial_temperature_uK
    assert np.all(np.diff(trace.retention_fraction) <= 1e-14)
    assert 0.0 <= trace.point.final_retention_fraction <= 1.0


def test_l1_handover_preconditions_can_be_disabled():
    inputs = L1TransportInputs(
        time_points=21,
        require_minimum_depth=True,
        require_maximum_start_power=True,
        require_critical_acceleration=True,
    )
    blocked = simulate_l1_transport(inputs, 800.0, 0.1)
    relaxed = simulate_l1_transport(
        replace(
            inputs,
            require_minimum_depth=False,
            require_maximum_start_power=False,
            require_critical_acceleration=False,
        ),
        800.0,
        0.1,
    )

    assert not blocked.point.feasible_hardware_point
    assert relaxed.point.feasible_hardware_point


def test_background_rate_multiplies_statistical_survival():
    baseline_inputs = L1TransportInputs(time_points=61)
    loss_inputs = L1TransportInputs(
        time_points=61,
        background_loss_rate_s=2.0,
    )
    baseline = simulate_l1_transport(baseline_inputs, 300.0, 1.5)
    with_loss = simulate_l1_transport(loss_inputs, 300.0, 1.5)
    expected_ratio = math.exp(-2.0 * l1_timing(loss_inputs).total_time_s)

    assert math.isclose(
        with_loss.point.final_retention_fraction
        / baseline.point.final_retention_fraction,
        expected_ratio,
        rel_tol=1e-10,
    )


def test_small_scan_selects_two_feasible_points_and_plots(tmp_path):
    inputs = L1TransportInputs(
        detuning_min_ghz=300.0,
        detuning_max_ghz=900.0,
        detuning_points=4,
        handover_source_power_min_w=1.0,
        handover_source_power_max_w=5.0,
        power_points=4,
        time_points=41,
    )
    result = analyze_l1_transport_scan(inputs)

    assert np.asarray(result.final_temperature_rise_uK).shape == (4, 4)
    assert np.asarray(result.final_retention_fraction).shape == (4, 4)
    assert result.optimal.feasible_hardware_point
    assert result.comparison.feasible_hardware_point
    assert result.optimal.quality_cost <= result.comparison.quality_cost
    output = plot_l1_transport_scan(result, tmp_path / "l1_scan.png")
    assert output.exists()
    assert output.stat().st_size > 0


def test_rb_paper_point_and_cs_reference_use_explicit_power_conventions():
    rb_inputs = replace(
        l1_transport_inputs_for_species("Rb-87"),
        detuning_points=3,
        power_points=3,
        time_points=21,
    )
    rb_result = analyze_l1_transport_scan(rb_inputs)
    rb_reference = rb_result.reference_points[0]
    assert rb_reference.label == "Paper operating point"
    assert rb_reference.point.detuning_ghz == 300.0
    assert rb_reference.point.handover_source_power_w == 1.0
    assert 300.0 < rb_reference.point.depth_uK < 312.0

    cs_inputs = replace(
        l1_transport_inputs_for_species("Cs-133"),
        detuning_points=3,
        power_points=3,
        time_points=21,
    )
    cs_result = analyze_l1_transport_scan(cs_inputs)
    cs_reference = cs_result.reference_points[0]
    assert cs_inputs.delivery_efficiency == 0.7
    assert cs_reference.point.detuning_ghz == 600.0
    assert math.isclose(cs_reference.point.depth_uK, 500.0, rel_tol=1e-12)
    assert cs_reference.point.handover_source_power_w > 2.0
