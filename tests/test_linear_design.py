import pytest

from continuous_loading.atomic import CS133
from continuous_loading.linear_design import (
    LinearDesignInputs,
    acceleration_bound_depth_requirement_uK,
    analyze_detuning_power_lp,
    handover_axial_depth_requirement_uK,
)


def test_longer_handover_reduces_axial_depth_requirement():
    short = handover_axial_depth_requirement_uK(
        CS133,
        detuning_ghz=600.0,
        handover_time_ms=0.2,
        minimum_cycles=80.0,
    )
    long = handover_axial_depth_requirement_uK(
        CS133,
        detuning_ghz=600.0,
        handover_time_ms=0.4,
        minimum_cycles=80.0,
    )
    assert short == pytest.approx(4.0 * long)


def test_acceleration_increases_required_static_depth():
    static = acceleration_bound_depth_requirement_uK(
        CS133,
        detuning_ghz=600.0,
        design_temperature_uK=120.0,
        target_bound_fraction=0.8,
        acceleration_m_s2=0.0,
    )
    accelerated = acceleration_bound_depth_requirement_uK(
        CS133,
        detuning_ghz=600.0,
        design_temperature_uK=120.0,
        target_bound_fraction=0.8,
        acceleration_m_s2=4_000.0,
    )
    assert accelerated > static


def test_piecewise_lp_recommendation_passes_full_model():
    inputs = LinearDesignInputs(segment_count=8)
    result = analyze_detuning_power_lp(
        inputs,
        handover_times_ms=(0.3, 1.0),
    )
    assert len(result.handover_results) == 2
    for time_result in result.handover_results:
        assert time_result.feasible
        assert time_result.recommended is not None
        assert time_result.recommended.exact_constraints_satisfied
        assert (
            time_result.recommended.scattering_rate_s
            <= inputs.max_scattering_rate_s
        )
        assert time_result.recommended.bound_fraction >= inputs.target_bound_fraction


def test_tight_power_limit_can_make_lp_infeasible():
    inputs = LinearDesignInputs(
        segment_count=6,
        max_source_power_w=1.0,
    )
    result = analyze_detuning_power_lp(
        inputs,
        handover_times_ms=(0.3,),
    )
    assert not result.handover_results[0].feasible
    assert result.handover_results[0].recommended is None
