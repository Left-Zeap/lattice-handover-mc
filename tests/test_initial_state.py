"""initial_state.py 静止晶格热平衡系综采样的物理与数值回归测试。

物理锚点：采样温度在去质心动能口径下 ≈ 设定值（MC 误差内）；全部
粒子满足束缚拒绝判据语义（完整 cos² 势 ε<U_ax，重力时另有径向
鞍点判据）；配方与 ``transport_mc._sample_initial_ensemble`` 在相同
物理参数与同 seed 下逐位一致（验证提取无漂移）。
"""

import math

import numpy as np
import pytest

from continuous_loading.atomic import RB87
from continuous_loading.constants import BOLTZMANN, GRAVITY
from continuous_loading.initial_state import (
    ThermalLatticeEnsembleInputs,
    _standing_wave_beam_parameters,
    ensemble_kinetic_temperature_uK,
    sample_static_lattice_thermal_ensemble,
)
from continuous_loading.lattice import gaussian_gravity_trap
from continuous_loading import transport_mc


_WAVELENGTH_NM = RB87.laser_wavelength_red_of_d1_nm(300.0)


def _inputs(**overrides):
    base = dict(
        atom_label="Rb-87",
        wavelength_nm=_WAVELENGTH_NM,
        waist_um=260.0,
        depth_uK=500.0,
        temperature_uK=20.0,
        particle_count=2_000,
        seed=12345,
        include_gravity=False,
    )
    base.update(overrides)
    return ThermalLatticeEnsembleInputs(**base)


def test_input_validation():
    with pytest.raises(ValueError):
        _inputs(atom_label="H-1")
    for field, bad in (
        ("wavelength_nm", 0.0),
        ("waist_um", -1.0),
        ("depth_uK", float("nan")),
        ("temperature_uK", 0.0),
        ("particle_count", 0),
        ("retro_power_ratio", 1.5),
        ("cloud_axial_sigma_mm", -0.1),
    ):
        with pytest.raises(ValueError):
            _inputs(**{field: bad})


def test_antinode_potential_anchors_depth():
    """波腹（原点）势必须等于 -depth_uK 对应的能量（R=1 时 U_ax=depth）。"""
    inputs = _inputs()
    i1, i2, u_ax, k, c_u = _standing_wave_beam_parameters(inputs)
    potential, _, _, _ = transport_mc._double_beam_potential_and_force(
        np.zeros((1, 3)),
        intensity1_w_m2=i1,
        intensity2_w_m2=i2,
        waist1_m=inputs.waist_um * 1e-6,
        waist2_m=inputs.waist_um * 1e-6,
        wave_number_m=k,
        lattice_position_m=0.0,
        phase_rad=0.0,
        potential_per_intensity_j=c_u,
    )
    depth_j = inputs.depth_uK * 1e-6 * BOLTZMANN
    assert potential[0] == pytest.approx(-depth_j, rel=1e-12)
    assert u_ax == pytest.approx(depth_j, rel=1e-12)


def test_sampled_temperature_matches_setpoint_within_mc_error():
    """去质心动能温度 ≈ 20 µK；N=2000 时 MC 相对标准差 ≈1.8%，取 5% 容差。"""
    inputs = _inputs()
    ensemble = sample_static_lattice_thermal_ensemble(inputs)
    temperature = ensemble_kinetic_temperature_uK(ensemble, RB87.mass_kg)
    assert temperature == pytest.approx(inputs.temperature_uK, rel=0.05)


def test_same_seed_reproducible_and_different_seed_differs():
    first = sample_static_lattice_thermal_ensemble(_inputs())
    second = sample_static_lattice_thermal_ensemble(_inputs())
    np.testing.assert_array_equal(first.positions_m, second.positions_m)
    np.testing.assert_array_equal(first.velocities_m_s, second.velocities_m_s)
    np.testing.assert_array_equal(first.site_index, second.site_index)

    third = sample_static_lattice_thermal_ensemble(_inputs(seed=54321))
    assert not np.array_equal(first.positions_m, third.positions_m)
    assert not np.array_equal(first.velocities_m_s, third.velocities_m_s)


def test_particle_ensemble_validation_and_metadata():
    inputs = _inputs(cloud_axial_sigma_mm=0.5)
    ensemble = sample_static_lattice_thermal_ensemble(inputs)
    positions, velocities, weights = ensemble.host_arrays()
    assert positions.shape == (inputs.particle_count, 3)
    assert velocities.shape == (inputs.particle_count, 3)
    np.testing.assert_array_equal(weights, np.ones(inputs.particle_count))
    assert ensemble.frame == "l1_local"
    assert ensemble.particle_count == inputs.particle_count
    site_index = np.asarray(ensemble.site_index)
    assert site_index.shape == (inputs.particle_count,)
    assert np.issubdtype(site_index.dtype, np.integer)
    # 格点吸附开启时云宽 0.5 mm ≫ λ/2，格点指标应有显著散布。
    assert site_index.std() > 1.0


def _bound_criterion_residuals(ensemble, inputs):
    """逐粒子总激发能与（重力时）径向激发能相对各自势垒的余量。"""
    i1, i2, u_ax, k, c_u = _standing_wave_beam_parameters(inputs)
    positions, velocities, _ = ensemble.host_arrays()
    potential, _, _, _ = transport_mc._double_beam_potential_and_force(
        positions,
        intensity1_w_m2=i1,
        intensity2_w_m2=i2,
        waist1_m=inputs.waist_um * 1e-6,
        waist2_m=inputs.waist_um * 1e-6,
        wave_number_m=k,
        lattice_position_m=0.0,
        phase_rad=0.0,
        potential_per_intensity_j=c_u,
    )
    kinetic = 0.5 * RB87.mass_kg * np.einsum("ij,ij->i", velocities, velocities)
    axial_margin = u_ax - (kinetic + potential + u_ax)
    radial_margin = None
    if inputs.include_gravity:
        radial_depth = c_u * (i1 + i2 + 2.0 * math.sqrt(i1 * i2))
        barrier, minimum, _ = gaussian_gravity_trap(
            radial_depth, inputs.waist_um * 1e-6, RB87.mass_kg
        )
        assert barrier > 0.0
        radial_excitation = (
            kinetic + potential + RB87.mass_kg * GRAVITY * positions[:, 1] - minimum
        )
        radial_margin = barrier - radial_excitation
    return axial_margin, radial_margin


def test_all_particles_satisfy_bound_criterion():
    """拒绝判据语义：不存在被接受但未束缚的样本（含重力与格点吸附变体）。"""
    for overrides in (
        {},
        {"include_gravity": True},
        {"include_gravity": True, "cloud_axial_sigma_mm": 0.5},
    ):
        inputs = _inputs(**overrides)
        ensemble = sample_static_lattice_thermal_ensemble(inputs)
        axial_margin, radial_margin = _bound_criterion_residuals(ensemble, inputs)
        assert np.all(axial_margin > 0.0)
        if radial_margin is not None:
            assert np.all(radial_margin > 0.0)


def test_recipe_matches_transport_mc_bitwise():
    """相同物理参数与同 seed 下与 transport_mc 原配方逐位一致。"""
    for overrides in (
        {"include_gravity": True},
        {"include_gravity": True, "cloud_axial_sigma_mm": 0.5},
    ):
        inputs = _inputs(**overrides)
        i1, i2, u_ax, k, c_u = _standing_wave_beam_parameters(inputs)
        ref_positions, ref_velocities = transport_mc._sample_initial_ensemble(
            particle_count=inputs.particle_count,
            atom_mass_kg=RB87.mass_kg,
            temperature_uK=inputs.temperature_uK,
            intensity1_w_m2=i1,
            intensity2_w_m2=i2,
            waist1_m=inputs.waist_um * 1e-6,
            waist2_m=inputs.waist_um * 1e-6,
            axial_modulation_j=u_ax,
            wave_number_m=k,
            potential_per_intensity_j=c_u,
            cloud_axial_sigma_mm=inputs.cloud_axial_sigma_mm,
            include_gravity=inputs.include_gravity,
            rng=np.random.default_rng(inputs.seed),
        )
        ensemble = sample_static_lattice_thermal_ensemble(inputs)
        np.testing.assert_array_equal(ensemble.positions_m, ref_positions)
        np.testing.assert_array_equal(ensemble.velocities_m_s, ref_velocities)
