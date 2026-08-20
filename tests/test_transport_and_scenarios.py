import pytest

from continuous_loading.transport import (
    acceleration_jump_heating_uK,
    thermal_bound_fraction_3d_harmonic,
)
from continuous_loading.scenarios import (
    cs_design_candidate,
    predict_cs_transport,
    reproduce_paper_rb87,
    scan_cs_designs,
)


def test_paper_speed_and_temperature_reproduction():
    result = reproduce_paper_rb87()
    assert result.lattice1_average_speed_m_s == pytest.approx(7.8)
    assert result.lattice2_average_speed_m_s == pytest.approx(8.095238)
    assert result.transport_budget.final_temperature_uK == pytest.approx(120.0)
    assert 0.0 < result.inferred_handover_fraction < 1.0


def test_paper_transport_has_small_static_tilt():
    result = reproduce_paper_rb87()
    for stage in result.transport_budget.stages:
        assert stage.barrier_fraction > 0.98
        assert stage.effective_barrier_uK > 490.0


def test_paper_reservoir_density_order_of_magnitude():
    result = reproduce_paper_rb87()
    assert result.stochastic_overlap_atoms == pytest.approx(5.0)
    assert result.inferred_atoms_per_lattice_site == pytest.approx(466.3, rel=0.02)
    assert result.collision_density_m3_s == pytest.approx(3e19, rel=0.35)


def test_cs_scattering_power_tradeoff():
    near = cs_design_candidate(
        d1_red_detuning_ghz=300.0,
        target_depth_uK=500.0,
        waist_um=250.0,
    )
    far = cs_design_candidate(
        d1_red_detuning_ghz=700.0,
        target_depth_uK=500.0,
        waist_um=250.0,
    )
    assert far.forward_power_at_atoms_w > near.forward_power_at_atoms_w
    assert far.scattering_rate_s < near.scattering_rate_s


def test_cs_constraints_mark_candidates():
    candidates = scan_cs_designs(
        target_depth_uK=500.0,
        waist_um=250.0,
        detuning_min_ghz=300.0,
        detuning_max_ghz=700.0,
        detuning_step_ghz=100.0,
        max_source_power_w=2.0,
        max_scattering_rate_s=500.0,
    )
    assert any(candidate.feasible for candidate in candidates)
    assert any(not candidate.feasible for candidate in candidates)


def test_cs_transport_constant_depth_power_scaling():
    result = predict_cs_transport(d1_red_detuning_ghz=600.0)
    assert result.lattice1_start_power_w > result.lattice1_end_power_w
    assert result.lattice2_end_power_w < result.lattice2_start_power_w
    assert result.transport_budget.final_temperature_uK > 20.0


def test_acceleration_jump_and_thermal_fraction_limits():
    heating = acceleration_jump_heating_uK(
        atom_mass_kg=1.44e-25,
        axial_frequency_hz=390e3,
        acceleration_jumps_m_s2=(4000.0, -4000.0),
    )
    # 即使把两个 4,000 m/s² 跳变视为完全非相干，加热仍低于 0.01 µK。
    assert heating < 0.01
    assert thermal_bound_fraction_3d_harmonic(0.0, 20.0) == pytest.approx(0.0)
    assert thermal_bound_fraction_3d_harmonic(500.0, 20.0) > 0.999
