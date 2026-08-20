from dataclasses import replace

import numpy as np
import pytest

from continuous_loading.atomic import RB87
from continuous_loading.constants import BOLTZMANN
from continuous_loading.gpu_backend import cupy_available
from continuous_loading.control_waveforms import (
    HandoverControlWaveform,
    TransportControlWaveform,
)
from continuous_loading.full_chain import (
    FullChainInputs,
    analyze_full_chain_scan,
    simulate_full_chain_point,
)
from continuous_loading.handover import HandoverParameters, run_handover_monte_carlo
from continuous_loading.l1_transport import (
    _kinematics,
    l1_timing,
    l1_transport_inputs_for_species,
    simulate_l1_transport,
)
from continuous_loading.transport_batch import _kinematics_arrays
from continuous_loading.l1_handover import (
    L1HandoverInputs,
    simulate_l1_handover_point_continuous,
)
from continuous_loading.l2_transport import L2TransportInputs
from continuous_loading.lattice import gaussian_gravity_trap
from continuous_loading.phase_space import (
    ParticleEnsemble,
    canonicalize_lattice_phase,
)
from continuous_loading.transport_mc import simulate_leg_monte_carlo
from ui.timeline import build_full_series, build_timeline


def test_lattice_phase_is_canonicalized_at_stage_boundary() -> None:
    """Dropping per-particle phase would turn antinodes into random positions.

    The general handover->L2 transform removes beam offset, lattice
    displacement and velocity without changing the optical phase.
    """
    count = 32
    wave_number = 2.0 * np.pi / (852e-9)
    phase = np.linspace(0.0, np.pi, count, endpoint=False)
    positions = np.zeros((count, 3))
    positions[:, 2] = -phase / wave_number
    axis = np.array((0.2, 0.0, np.sqrt(0.96)))
    offset = np.array((3e-6, -2e-6, 0.0))
    displacement = 4e-6
    velocity = 0.3
    raw = ParticleEnsemble(
        positions_m=positions + offset,
        velocities_m_s=np.tile(velocity * axis, (count, 1)),
    )
    canonical = canonicalize_lattice_phase(
        raw,
        phase_rad=phase,
        wave_number_m=wave_number,
        axis=axis,
        beam_offset_m=offset,
        lattice_displacement_m=displacement,
        lattice_velocity_m_s=velocity,
        frame="l2",
    )
    expected_axial = (
        (positions @ axis) - displacement + phase / wave_number
    )
    assert np.asarray(canonical.positions_m) @ axis == pytest.approx(
        expected_axial
    )
    assert np.max(np.abs(np.asarray(canonical.velocities_m_s))) < 1e-15


def test_global_gravity_switch_reaches_both_transport_legs() -> None:
    transport = replace(
        l1_transport_inputs_for_species("Rb-87"),
        include_gravity=True,
    )
    assert transport.include_gravity is True
    # L2 is made with dataclasses.replace and therefore inherits the same flag.
    from continuous_loading.l2_transport import l2_leg_inputs

    l2_inputs = l2_leg_inputs(
        transport,
        L2TransportInputs(),
        captured_temperature_uK=20.0,
        captured_atom_number=1e5,
    )
    assert l2_inputs.include_gravity is True


def test_gaussian_gravity_trap_has_downward_sag_and_lower_barrier() -> None:
    depth_j = 500.0e-6 * BOLTZMANN
    barrier_j, minimum_j, sag_m = gaussian_gravity_trap(
        depth_j, 250.0e-6, RB87.mass_kg
    )
    assert 0.0 < barrier_j < depth_j
    assert minimum_j < -depth_j
    assert sag_m < 0.0
    unsupported, _, _ = gaussian_gravity_trap(
        0.01e-6 * BOLTZMANN, 1.0e-3, RB87.mass_kg
    )
    assert unsupported == 0.0


def test_handover_gravity_changes_sag_and_effective_capture_barrier() -> None:
    base = HandoverParameters(
        atom_mass_kg=RB87.mass_kg,
        wavelength_nm=795.0,
        depth1_uK=500.0,
        depth2_uK=500.0,
        waist1_um=250.0,
        waist2_um=250.0,
        duration_ms=0.02,
        time_step_us=0.2,
        particle_count=128,
        trace_points=2,
        include_scattering=False,
        seed=1234,
    )
    no_gravity, ensemble0 = run_handover_monte_carlo(
        base, return_captured_ensemble=True
    )
    gravity, ensemble_g = run_handover_monte_carlo(
        replace(base, include_gravity=True), return_captured_ensemble=True
    )
    assert ensemble0 is not None and ensemble_g is not None
    assert gravity.effective_barrier_uK < no_gravity.effective_barrier_uK
    assert np.mean(ensemble_g.positions_m[:, 1]) < (
        np.mean(ensemble0.positions_m[:, 1]) - 1e-6
    )


def test_measured_transport_waveform_drives_timing_aom_and_power() -> None:
    waveform = TransportControlWaveform(
        time_ms=(0.0, 1.0, 2.0),
        position_m=(0.0, 0.001, 0.002),
        velocity_m_s=(0.0, 1.0, 0.0),
        acceleration_m_s2=(1000.0, 0.0, -1000.0),
        aom_frequency_difference_mhz=(0.0, 2.0, 0.0),
        source_power_scale=(2.0, 1.5, 1.0),
    )
    inputs = replace(
        l1_transport_inputs_for_species("Rb-87"),
        distance_m=0.002,
        time_points=5,
        control_waveform=waveform,
    )
    assert l1_timing(inputs).total_time_s == pytest.approx(0.002)
    trace = simulate_l1_transport(inputs, 300.0, 1.0)
    middle = trace.time_ms.index(1.0)
    assert trace.aom_frequency_difference_mhz[middle] == pytest.approx(2.0)
    assert trace.source_power_w[0] == pytest.approx(2.0)
    assert trace.source_power_w[-1] == pytest.approx(1.0)


def test_minimum_jerk_transport_has_zero_boundary_acceleration_and_gpu_parity() -> None:
    inputs = replace(
        l1_transport_inputs_for_species("Rb-87"),
        distance_m=0.1,
        acceleration_m_s2=100.0,
        maximum_velocity_m_s=1.0,
        kinematic_profile="minimum_jerk",
    )
    timing = l1_timing(inputs)
    times = np.linspace(0.0, timing.total_time_s, 101)
    scalar = np.asarray(
        [_kinematics(float(value), inputs, timing)[:3] for value in times]
    )
    position, velocity, acceleration = _kinematics_arrays(
        inputs, timing, times
    )
    assert position == pytest.approx(scalar[:, 0])
    assert velocity == pytest.approx(scalar[:, 1])
    assert acceleration == pytest.approx(scalar[:, 2])
    assert position[0] == pytest.approx(0.0)
    assert position[-1] == pytest.approx(inputs.distance_m)
    assert acceleration[0] == pytest.approx(0.0)
    assert acceleration[-1] == pytest.approx(0.0)
    assert np.max(np.abs(acceleration)) == pytest.approx(
        inputs.acceleration_m_s2, rel=2e-3
    )


def test_handover_waveform_is_sampled_on_actual_integrator_grid() -> None:
    waveform = HandoverControlWaveform(
        time_ms=(0.0, 0.05, 0.1),
        lattice1_fraction=(1.0, 0.7, 0.0),
        lattice2_fraction=(0.0, 0.2, 1.0),
        relative_phase_rad=(0.0, 0.1, 0.0),
    )
    parameters = HandoverParameters(
        atom_mass_kg=RB87.mass_kg,
        wavelength_nm=780.0,
        depth1_uK=500.0,
        depth2_uK=500.0,
        waist1_um=250.0,
        waist2_um=250.0,
        duration_ms=0.1,
        time_step_us=10.0,
        particle_count=20,
        trace_points=3,
        include_scattering=False,
        control_waveform=waveform,
    )
    result = run_handover_monte_carlo(parameters)
    for time_ms, fraction1, fraction2 in zip(
        result.trace.time_ms,
        result.trace.lattice1_fraction,
        result.trace.lattice2_fraction,
    ):
        expected1, expected2, _ = waveform.sample(time_ms * 1e-3)
        assert fraction1 == pytest.approx(expected1)
        assert fraction2 == pytest.approx(expected2)


def test_particle_ensemble_transport_keeps_default_trace_interface_optional() -> None:
    inputs = replace(
        l1_transport_inputs_for_species("Rb-87"),
        distance_m=1e-9,
        acceleration_m_s2=0.1,
        maximum_velocity_m_s=1e-5,
        transport_method="monte_carlo",
        transport_time_step_us=0.05,
        mc_particle_count=8,
        mc_include_scattering=False,
        mc_compute_backend="cpu",
        time_points=5,
    )
    ensemble = ParticleEnsemble(
        positions_m=np.zeros((8, 3)),
        velocities_m_s=np.zeros((8, 3)),
        frame="l1_local",
    )
    trace, final_ensemble = simulate_leg_monte_carlo(
        inputs,
        300.0,
        2.0,
        initial_ensemble=ensemble,
        return_final_ensemble=True,
    )
    assert trace.point.final_retention_fraction == pytest.approx(1.0)
    assert final_ensemble is not None
    assert final_ensemble.particle_count == 8


def test_phase_space_continuity_scan_validates_mc_preconditions() -> None:
    with pytest.raises(ValueError, match="transport_method"):
        analyze_full_chain_scan(FullChainInputs(phase_space_continuity=True))


def test_continuous_phase_space_rejects_l1_acceleration_step() -> None:
    """绕过 UI 的库调用也不能用梯形速度制造 t=0 加速度阶跃。"""
    transport = replace(
        l1_transport_inputs_for_species("Rb-87"),
        transport_method="monte_carlo",
        kinematic_profile="trapezoid",
    )
    inputs = L1HandoverInputs(transport=transport)
    with pytest.raises(ValueError, match="禁止加速度阶跃"):
        simulate_l1_handover_point_continuous(inputs, 300.0, 1.0)


def _small_continuous_transport(backend: str):
    """连续相空间测试共用的小型 L1 运输输入（初态为静止晶格热平衡采样）。"""
    return replace(
        l1_transport_inputs_for_species("Rb-87"),
        transport_method="monte_carlo",
        kinematic_profile='minimum_jerk',
        distance_m=1e-9,
        acceleration_m_s2=0.1,
        maximum_velocity_m_s=5e-6,
        transport_time_step_us=0.05,
        mc_particle_count=24,
        mc_include_scattering=False,
        mc_compute_backend=backend,
        mc_cloud_axial_sigma_mm=0.0,
        time_points=5,
    )


def test_small_real_chain_carries_phase_space_through_all_three_stages() -> None:
    transport = _small_continuous_transport("cpu")
    handover = L1HandoverInputs(
        transport=transport,
        duration_us=10.0,
        particle_count=24,
        time_step_us=1.0,
        trace_points=3,
        include_scattering=False,
        compute_backend="cpu",
        parallel_backend="serial",
        worker_count=1,
        cloud_axial_sigma_mm=0.0,
    )
    l2 = L2TransportInputs(
        distance_m=1e-9,
        acceleration_m_s2=0.1,
        maximum_velocity_m_s=5e-6,
        kinematic_profile='minimum_jerk',
        time_points=5,
    )
    result = simulate_full_chain_point(
        FullChainInputs(
            handover=handover,
            l2=l2,
            phase_space_continuity=True,
        ),
        300.0,
        2.0,
        trace_points=3,
    )
    assert result.interface_mode == "phase_space_continuous"
    assert set(result.combined_trace.phase) == {
        "L1 transport",
        "handover",
        "L2 transport",
    }
    # LGM 装载模块已移除：时间轴从 L1 起点（静止晶格热平衡初态）开始。
    assert result.combined_trace.loading_start_ms is None
    assert result.combined_trace.loading_end_ms is None
    phases = np.asarray(result.combined_trace.phase)
    temperatures = np.asarray(result.combined_trace.temperature_uK)
    l1_indices = np.flatnonzero(phases == "L1 transport")
    handover_indices = np.flatnonzero(phases == "handover")
    l2_indices = np.flatnonzero(phases == "L2 transport")
    assert temperatures[l1_indices[-1]] == pytest.approx(
        temperatures[handover_indices[0]]
    )
    assert temperatures[handover_indices[-1]] == pytest.approx(
        temperatures[l2_indices[0]]
    )
    timeline = build_timeline(result)
    series = build_full_series(result)
    assert timeline["phase"][0] == "L1 transport"
    assert len(timeline["time_ms"]) == len(timeline["position_m"])
    assert len(series["time_ms"]) == len(series["source_power_w"])


@pytest.mark.parametrize(
    'backend',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param(
            'gpu',
            id='gpu',
            marks=pytest.mark.skipif(
                not cupy_available(), reason='CuPy/CUDA unavailable'
            ),
        ),
    ],
)
def test_small_continuous_phase_space_scan_runs_two_by_two(backend) -> None:
    transport = replace(
        _small_continuous_transport(backend),
        detuning_min_ghz=299.0,
        detuning_max_ghz=300.0,
        detuning_points=2,
        handover_source_power_min_w=1.9,
        handover_source_power_max_w=2.0,
        power_points=2,
        mc_particle_count=12,
        time_points=4,
    )
    handover = L1HandoverInputs(
        transport=transport,
        duration_us=10.0,
        particle_count=12,
        time_step_us=1.0,
        trace_points=2,
        include_scattering=False,
        compute_backend=backend,
        parallel_backend='serial',
        worker_count=1,
        cloud_axial_sigma_mm=0.0,
    )
    l2 = L2TransportInputs(
        distance_m=1e-9,
        acceleration_m_s2=0.1,
        maximum_velocity_m_s=5e-6,
        kinematic_profile='minimum_jerk',
        time_points=4,
    )
    result = analyze_full_chain_scan(
        FullChainInputs(
            handover=handover, l2=l2, phase_space_continuity=True
        )
    )
    assert result.evaluated_points > 0
    assert result.optimal_simulation.interface_mode == 'phase_space_continuous'
