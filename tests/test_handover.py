from dataclasses import replace

import numpy as np
import pytest

from continuous_loading.handover import (
    run_handover_monte_carlo,
    scan_handover_parameter,
)
from continuous_loading.scenarios import (
    extended_figure2_scan_preset,
    paper_handover_parameters,
)


def _fast_parameters(**updates):
    base = paper_handover_parameters(
        depth1_uK=500.0,
        depth2_uK=500.0,
        temperature_uK=15.0,
    )
    values = {
        "duration_ms": 0.04,
        "time_step_us": 0.2,
        "particle_count": 384,
        "trace_points": 9,
        "include_scattering": False,
        "post_handover_acceleration_m_s2": 0.0,
    }
    values.update(updates)
    return replace(base, **values)


def test_identical_lattices_make_handover_an_identity():
    parameters = _fast_parameters(
        crossing_angle_deg=0.0,
        randomize_relative_phase=False,
        relative_phase_rad=0.0,
        time_step_us=0.05,
    )
    result = run_handover_monte_carlo(parameters)

    # U1(t) + U2(t) is exactly time independent in this limit.
    assert result.transfer_efficiency == pytest.approx(1.0)
    assert result.estimated_captured_atom_number == pytest.approx(
        parameters.initial_atom_number
    )
    assert result.transfer_standard_error > 0.0
    assert result.handover_heating_uK == pytest.approx(0.0, abs=2e-3)
    assert result.trace.time_ms[-1] == pytest.approx(parameters.duration_ms)


def test_spatial_mode_mismatch_reduces_transfer():
    matched = _fast_parameters(
        duration_ms=0.12,
        crossing_angle_deg=4.0,
        randomize_relative_phase=True,
    )
    mismatched = replace(matched, lattice1_distance_cm=39.85)

    matched_result = run_handover_monte_carlo(matched)
    mismatched_result = run_handover_monte_carlo(mismatched)

    assert matched_result.transfer_efficiency > 0.0
    assert (
        mismatched_result.transfer_efficiency
        < matched_result.transfer_efficiency
    )


def test_acceleration_scan_reduces_the_effective_barrier():
    parameters = _fast_parameters(
        crossing_angle_deg=0.0,
        randomize_relative_phase=False,
    )
    points = scan_handover_parameter(
        parameters,
        "post_handover_acceleration_m_s2",
        (0.0, 2.0 * 10**5),
    )

    assert points[0].result.barrier_fraction == pytest.approx(1.0)
    assert points[1].result.barrier_fraction < points[0].result.barrier_fraction
    assert (
        points[1].result.transfer_efficiency
        <= points[0].result.transfer_efficiency
    )


def test_seed_makes_monte_carlo_reproducible():
    parameters = _fast_parameters(
        randomize_relative_phase=True,
        include_scattering=True,
    )
    first = run_handover_monte_carlo(parameters)
    second = run_handover_monte_carlo(parameters)

    assert second.transfer_efficiency == first.transfer_efficiency
    assert second.final_temperature_uK == first.final_temperature_uK
    assert second.mean_scattering_events == first.mean_scattering_events


def test_verlet_step_is_automatically_limited_by_axial_trap_frequency():
    parameters = _fast_parameters(
        depth1_uK=1000.0,
        depth2_uK=1000.0,
        duration_ms=0.01,
        time_step_us=10.0,
        particle_count=32,
        trace_points=3,
    )
    result = run_handover_monte_carlo(parameters)
    assert result.actual_time_step_us < parameters.time_step_us
    assert np.isfinite(result.all_atom_final_temperature_uK)


def test_extended_figure2_presets_separate_scan_and_working_point():
    paper = paper_handover_parameters()
    panel_a = extended_figure2_scan_preset("a")
    panel_b = extended_figure2_scan_preset("b")
    panel_c = extended_figure2_scan_preset("c")

    assert paper.duration_ms == pytest.approx(1.0)
    assert paper.lattice1_distance_cm == pytest.approx(38.85)
    assert paper.post_handover_acceleration_m_s2 == pytest.approx(4_000.0)
    assert panel_a.parameter_name == "lattice1_distance_cm"
    assert min(panel_b.values) == pytest.approx(0.20)
    assert max(panel_b.values) == pytest.approx(0.40)
    assert panel_c.parameter_name == "post_handover_acceleration_m_s2"
