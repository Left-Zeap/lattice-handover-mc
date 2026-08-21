"""light_field.py 光场时序表与查询接口的逐位一致性回归测试。

对照方式：在测试内复现 ``simulate_leg_monte_carlo`` CPU 主循环取光
学量的调用链（``l1_timing`` → ``_kinematics(min(t,T))`` →
``_leg_optics_at(z_L, t)`` → ``_double_beam_potential_and_force``）
与 ``run_handover_monte_carlo`` 的 ramp 预计算 + ``combined_force``
调用链（两次 ``_lattice_potential_force`` 求和），在随机
(positions, step) 上要求与查询接口逐位一致。
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from continuous_loading import handover, transport_mc
from continuous_loading.atomic import RB87
from continuous_loading.control_waveforms import (
    HandoverControlWaveform,
    TransportControlWaveform,
)
from continuous_loading.dipole import scalar_potential_and_scattering
from continuous_loading.handover import HandoverParameters
from continuous_loading.l1_transport import (
    _atom_from_label,
    _kinematics,
    l1_timing,
    l1_transport_inputs_for_species,
)
from continuous_loading.l2_transport import L2TransportInputs
from continuous_loading.light_field import (
    ChainLightField,
    build_handover_field_timeline,
    build_leg_field_timeline,
    handover_potential_and_force,
    leg_potential_and_force,
)


_DETUNING_GHZ = 300.0
_SOURCE_POWER_W = 1.0
_WAVELENGTH_NM = RB87.laser_wavelength_red_of_d1_nm(_DETUNING_GHZ)
_WAVE_NUMBER = 2.0 * math.pi / (_WAVELENGTH_NM * 1e-9)
_C_U = abs(
    scalar_potential_and_scattering(RB87, _WAVELENGTH_NM, 1.0).potential_j
)


def _small_leg_inputs(**overrides):
    """小距离运输腿：总时长约 2.25 ms，0.5 µs 步长下 4500 步。"""
    base = dict(
        distance_m=0.005,
        minimum_waist_um=None,
        minimum_waist_position_m=None,
        acceleration_m_s2=4000.0,
        maximum_velocity_m_s=4.0,
        time_points=21,
        transport_method="monte_carlo",
        transport_time_step_us=0.5,
    )
    base.update(overrides)
    return replace(l1_transport_inputs_for_species("Rb-87"), **base)


def _leg_step_reference(inputs, profile, timing, total_time, time_step_s, step):
    """复现主循环第 step 步取运动学与光学量的逐步调用链。"""
    time_s = step * time_step_s
    position, velocity, acceleration, _ = _kinematics(
        min(time_s, total_time), inputs, timing
    )
    i1, i2, w1, w2, _, _ = transport_mc._leg_optics_at(
        inputs, profile, _SOURCE_POWER_W, position, time_s
    )
    return time_s, position, velocity, acceleration, i1, i2, w1, w2


def _check_leg_timeline_against_stepwise(inputs, timeline):
    timing = l1_timing(inputs)
    total_time = timing.total_time_s
    atom = _atom_from_label(inputs.atom_label)
    profile = transport_mc._leg_optics_profile(
        inputs, _WAVELENGTH_NM, _SOURCE_POWER_W
    )
    # 步数/步长与主循环同一来源（含 ω_z·dt ≤ 1 精度守卫钳制）。
    integration_steps, time_step_s = transport_mc._leg_integration_schedule(
        inputs, atom, _WAVELENGTH_NM, profile, total_time
    )
    assert timeline.time_step_s == time_step_s
    assert timeline.step_times_s.shape == (integration_steps + 1,)
    for step in range(integration_steps + 1):
        time_s, z, v, a, i1, i2, w1, w2 = _leg_step_reference(
            inputs, profile, timing, total_time, time_step_s, step
        )
        assert timeline.step_times_s[step] == time_s
        assert timeline.z_lattice_m[step] == z
        assert timeline.lattice_velocity_m_s[step] == v
        assert timeline.acceleration_m_s2[step] == a
        assert timeline.intensity1_w_m2[step] == i1
        assert timeline.intensity2_w_m2[step] == i2
        assert timeline.waist1_m[step] == w1
        assert timeline.waist2_m[step] == w2
    return profile, timing, total_time, time_step_s


def test_leg_timeline_matches_stepwise_kinematics_and_optics():
    inputs = _small_leg_inputs()
    timeline = build_leg_field_timeline(inputs, _DETUNING_GHZ, _SOURCE_POWER_W)
    _check_leg_timeline_against_stepwise(inputs, timeline)


def test_leg_timeline_with_control_waveform():
    """实测波形：时序与 waist/source_power 覆盖逻辑逐步一致。"""
    duration_s = 2.25e-3
    waveform = TransportControlWaveform(
        time_ms=(0.0, 1.0, 2.25),
        position_m=(0.0, 0.002, 0.005),
        velocity_m_s=(0.0, 4.0, 0.0),
        acceleration_m_s2=(4000.0, 0.0, -4000.0),
        aom_frequency_difference_mhz=(
            0.0,
            2.0 * 4.0 / (_WAVELENGTH_NM * 1e-9) * 1e-6,
            0.0,
        ),
        source_power_scale=(0.5, 0.8, 1.0),
        waist_um=(240.0, 250.0, 260.0),
    )
    assert waveform.duration_ms == pytest.approx(duration_s * 1e3)
    inputs = _small_leg_inputs(control_waveform=waveform)
    timeline = build_leg_field_timeline(inputs, _DETUNING_GHZ, _SOURCE_POWER_W)
    _check_leg_timeline_against_stepwise(inputs, timeline)


def test_leg_query_matches_main_loop_call_chain():
    """随机 (positions, step) 上腿段查询与主循环逐步求势逐位一致。"""
    inputs = _small_leg_inputs()
    timeline = build_leg_field_timeline(inputs, _DETUNING_GHZ, _SOURCE_POWER_W)
    profile, timing, total_time, time_step_s = (
        _check_leg_timeline_against_stepwise(inputs, timeline)
    )
    rng = np.random.default_rng(20250818)
    steps = rng.integers(0, timeline.step_times_s.shape[0], size=8)
    for step in steps:
        _, z, _, _, i1, i2, w1, w2 = _leg_step_reference(
            inputs, profile, timing, total_time, time_step_s, int(step)
        )
        positions = np.empty((32, 3))
        positions[:, :2] = rng.normal(scale=50e-6, size=(32, 2))
        positions[:, 2] = z + rng.normal(scale=20e-6, size=32)
        reference = transport_mc._double_beam_potential_and_force(
            positions,
            intensity1_w_m2=i1,
            intensity2_w_m2=i2,
            waist1_m=w1,
            waist2_m=w2,
            wave_number_m=_WAVE_NUMBER,
            lattice_position_m=z,
            phase_rad=0.0,
            potential_per_intensity_j=_C_U,
        )
        queried = leg_potential_and_force(timeline, int(step), positions, _C_U)
        for ref, got in zip(reference, queried):
            np.testing.assert_array_equal(got, ref)
    with pytest.raises(IndexError):
        leg_potential_and_force(
            timeline, timeline.step_times_s.shape[0], positions, _C_U
        )


def _handover_parameters(**overrides):
    base = dict(
        atom_mass_kg=RB87.mass_kg,
        wavelength_nm=_WAVELENGTH_NM,
        depth1_uK=500.0,
        depth2_uK=600.0,
        waist1_um=260.0,
        waist2_um=250.0,
        duration_ms=0.5,
        time_step_us=0.5,
        particle_count=64,
        crossing_angle_deg=4.0,
        lattice1_distance_cm=38.9,
        optimal_distance_cm=38.85,
        l2_transverse_offset_um=2.0,
        relative_phase_rad=0.3,
        randomize_relative_phase=False,
        lattice1_velocity_m_s=1.5,
        lattice2_velocity_m_s=0.5,
        include_gravity=False,
        include_scattering=False,
    )
    base.update(overrides)
    return HandoverParameters(**base)


def _handover_step_reference(parameters, step_times_s, time_step_s, step):
    """复现 handover.py 的 ramp 取值与 combined_force（不含重力）。"""
    from continuous_loading.constants import BOLTZMANN as k_B

    e1, e2, e_out = handover._unit_axes(parameters.crossing_angle_deg)
    wave_number = 2.0 * math.pi / (parameters.wavelength_nm * 1e-9)
    duration_s = parameters.duration_ms * 1e-3
    distance_offset_m = (
        parameters.lattice1_distance_cm - parameters.optimal_distance_cm
    ) * 1e-2
    cloud_center = distance_offset_m * e1
    phase1 = -wave_number * float(cloud_center @ e1)
    l2_beam_offset = parameters.l2_transverse_offset_um * 1e-6 * e_out
    if parameters.control_waveform is None:
        fraction2_steps = step_times_s / duration_s
        fraction1_steps = 1.0 - fraction2_steps
        phase_control_steps = np.zeros_like(step_times_s)
    else:
        (
            fraction1_steps,
            fraction2_steps,
            phase_control_steps,
        ) = parameters.control_waveform.sampled_arrays(step_times_s)
    time_s = step * time_step_s
    fraction1 = float(fraction1_steps[step])
    fraction2 = float(fraction2_steps[step])
    phase_control = float(phase_control_steps[step])
    depth1_j = parameters.depth1_uK * 1e-6 * k_B
    depth2_j = parameters.depth2_uK * 1e-6 * k_B

    def evaluate(positions, phase2):
        potential1, force1, shape1 = handover._lattice_potential_force(
            positions,
            axis=e1,
            beam_offset_m=np.zeros(3),
            phase_rad=phase1,
            axial_velocity_m_s=parameters.lattice1_velocity_m_s,
            time_s=time_s,
            wave_number_m=wave_number,
            waist_m=parameters.waist1_um * 1e-6,
            depth_j=depth1_j * fraction1,
        )
        potential2, force2, shape2 = handover._lattice_potential_force(
            positions,
            axis=e2,
            beam_offset_m=l2_beam_offset,
            phase_rad=phase2 + phase_control,
            axial_velocity_m_s=parameters.lattice2_velocity_m_s,
            time_s=time_s,
            wave_number_m=wave_number,
            waist_m=parameters.waist2_um * 1e-6,
            depth_j=depth2_j * fraction2,
        )
        return potential1 + potential2, force1 + force2, shape1, shape2

    return (
        step_times_s,
        fraction1_steps,
        fraction2_steps,
        phase_control_steps,
        evaluate,
    )


def _handover_step_grid(parameters):
    duration_s = parameters.duration_ms * 1e-3
    requested_step_s = min(
        parameters.time_step_us * 1e-6,
        handover._stable_handover_step_s(parameters),
    )
    integration_steps = max(1, math.ceil(duration_s / requested_step_s))
    time_step_s = duration_s / integration_steps
    step_times_s = (
        np.arange(integration_steps + 1, dtype=float) * time_step_s
    )
    return step_times_s, time_step_s


def _check_handover_timeline_and_query(parameters):
    timeline = build_handover_field_timeline(parameters)
    step_times_s, time_step_s = _handover_step_grid(parameters)
    assert timeline.time_step_s == time_step_s
    np.testing.assert_array_equal(timeline.step_times_s, step_times_s)
    rng = np.random.default_rng(20250819)
    positions = np.empty((48, 3))
    positions[:, :2] = rng.normal(scale=30e-6, size=(48, 2))
    positions[:, 2] = rng.normal(scale=0.2e-3, size=48)
    steps = rng.integers(0, step_times_s.shape[0], size=8)
    for step in steps:
        (
            _,
            fraction1_steps,
            fraction2_steps,
            phase_control_steps,
            evaluate,
        ) = _handover_step_reference(
            parameters, step_times_s, time_step_s, int(step)
        )
        assert timeline.fraction1[int(step)] == fraction1_steps[int(step)]
        assert timeline.fraction2[int(step)] == fraction2_steps[int(step)]
        assert (
            timeline.phase_control_rad[int(step)]
            == phase_control_steps[int(step)]
        )
        reference = evaluate(positions, parameters.relative_phase_rad)
        queried = handover_potential_and_force(timeline, int(step), positions)
        for ref, got in zip(reference, queried):
            np.testing.assert_array_equal(got, ref)
        # 逐粒子随机相对相位经 replace 替换后同样逐位一致。
        phase2_array = parameters.relative_phase_rad + rng.uniform(
            0.0, math.pi, positions.shape[0]
        )
        per_particle = replace(timeline, phase2_rad=phase2_array)
        reference = evaluate(positions, phase2_array)
        queried = handover_potential_and_force(
            per_particle, int(step), positions
        )
        for ref, got in zip(reference, queried):
            np.testing.assert_array_equal(got, ref)
    return timeline


def test_handover_timeline_and_query_match_combined_force():
    _check_handover_timeline_and_query(_handover_parameters())


def test_handover_timeline_with_control_waveform():
    duration_ms = 0.5
    waveform = HandoverControlWaveform(
        time_ms=(0.0, 0.25, duration_ms),
        lattice1_fraction=(1.0, 0.5, 0.0),
        lattice2_fraction=(0.0, 0.5, 1.0),
        relative_phase_rad=(0.0, 0.1, 0.2),
    )
    _check_handover_timeline_and_query(
        _handover_parameters(control_waveform=waveform)
    )


def test_chain_light_field_precompute_endpoints():
    transport_inputs = _small_leg_inputs()
    handover_parameters = _handover_parameters()
    l2_inputs = L2TransportInputs(
        distance_m=0.003,
        acceleration_m_s2=2000.0,
        maximum_velocity_m_s=2.0,
        end_waist_um=150.0,
        time_points=11,
    )
    chain = ChainLightField.precompute(
        transport_inputs,
        _DETUNING_GHZ,
        _SOURCE_POWER_W,
        handover_parameters,
        l2_inputs,
    )
    l1 = chain.l1
    assert l1.z_lattice_m[0] == 0.0
    assert l1.z_lattice_m[-1] == transport_inputs.distance_m
    assert l1.lattice_velocity_m_s[-1] == 0.0
    assert chain.handover.fraction1[0] == 1.0
    assert chain.handover.fraction1[-1] == 0.0
    assert chain.handover.fraction2[0] == 0.0
    assert chain.handover.fraction2[-1] == 1.0
    l2 = chain.l2
    assert l2.z_lattice_m[0] == 0.0
    assert l2.z_lattice_m[-1] == l2_inputs.distance_m
    # 同一失谐量下三段波数一致；L2 末端束腰收敛到 end_waist_um。
    assert l1.wave_number_m == l2.wave_number_m
    assert chain.handover.wave_number_m == pytest.approx(l1.wave_number_m)
    assert l2.waist1_m[-1] == pytest.approx(l2_inputs.end_waist_um * 1e-6)
    # 时序先行：三段数组在传播前已全部就位。
    for timeline in (l1, chain.handover, l2):
        assert timeline.step_times_s.ndim == 1
        assert np.all(np.isfinite(timeline.step_times_s))
