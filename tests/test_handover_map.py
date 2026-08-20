from types import SimpleNamespace
from dataclasses import replace

import pytest

import continuous_loading.handover_map as handover_map
import continuous_loading.l1_handover as integrated_handover
from continuous_loading.handover_map import (
    CS133_HANDOVER_DEFAULTS,
    HANDOVER_TIME_US,
    HandoverMapInputs,
    RB87_HANDOVER_DEFAULTS,
    analyze_dual_species_handover_map,
    analyze_species_handover_map,
    load_handover_map_configuration,
    plot_dual_species_handover_map,
)


def _fake_monte_carlo(calls):
    def run(parameters):
        calls.append(parameters)
        efficiency = min(1.0, 0.8 + parameters.depth1_uK / 10_000.0)
        return SimpleNamespace(
            transfer_efficiency=efficiency,
            transfer_standard_error=0.02,
        )

    return run


def test_species_defaults_and_fixed_handover_time():
    configuration = load_handover_map_configuration()
    scan = HandoverMapInputs()

    assert RB87_HANDOVER_DEFAULTS.temperature_uK == pytest.approx(30.8)
    assert CS133_HANDOVER_DEFAULTS.temperature_uK == pytest.approx(120.0)
    assert HANDOVER_TIME_US == pytest.approx(1_000.0)
    assert scan.detuning_min_ghz == pytest.approx(
        configuration["scan"]["detuning_min_ghz"]
    )
    assert scan.detuning_max_ghz == pytest.approx(
        configuration["scan"]["detuning_max_ghz"]
    )
    assert scan.particle_count == configuration["monte_carlo"]["particle_count"]
    assert scan.time_step_us == pytest.approx(
        configuration["monte_carlo"]["time_step_us"]
    )
    assert scan.worker_count == configuration["parallel"]["worker_count"]
    assert scan.parallel_backend == configuration["parallel"]["backend"]


def test_only_constraint_feasible_points_run_monte_carlo(monkeypatch):
    calls = []
    monkeypatch.setattr(
        handover_map,
        "run_handover_monte_carlo",
        _fake_monte_carlo(calls),
    )
    scan = HandoverMapInputs(
        detuning_points=5,
        power_points=5,
        particle_count=16,
        time_step_us=1.0,
        parallel_backend="serial",
        worker_count=1,
        require_minimum_depth=True,
        require_thermal_bound_fraction=True,
        require_minimum_axial_cycles=True,
        use_l1_transport=False,
    )
    result = analyze_species_handover_map(RB87_HANDOVER_DEFAULTS, scan)

    assert result.evaluated_points == len(calls)
    assert 0 < result.evaluated_points < 25
    assert len(result.transfer_efficiency) == scan.power_points
    assert len(result.transfer_efficiency[0]) == scan.detuning_points
    assert all(
        parameters.duration_ms == pytest.approx(1.0)
        for parameters in calls
    )
    for feasible_row, efficiency_row in zip(
        result.feasible,
        result.transfer_efficiency,
    ):
        assert all(
            (efficiency is not None) == feasible
            for feasible, efficiency in zip(feasible_row, efficiency_row)
        )


def test_handover_map_preconditions_can_be_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        handover_map,
        "run_handover_monte_carlo",
        _fake_monte_carlo(calls),
    )
    scan = HandoverMapInputs(
        detuning_points=3,
        power_points=3,
        particle_count=8,
        time_step_us=1.0,
        parallel_backend="serial",
        worker_count=1,
        use_l1_transport=False,
        require_minimum_depth=False,
        require_thermal_bound_fraction=False,
        require_minimum_axial_cycles=False,
    )
    result = analyze_species_handover_map(RB87_HANDOVER_DEFAULTS, scan)

    assert result.evaluated_points == 6
    assert len(calls) == 6


def test_dual_species_map_and_plot(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        handover_map,
        "run_handover_monte_carlo",
        _fake_monte_carlo(calls),
    )
    scan = HandoverMapInputs(
        detuning_points=5,
        power_points=5,
        parallel_backend="serial",
        worker_count=1,
        use_l1_transport=False,
    )
    result = analyze_dual_species_handover_map(scan)

    assert tuple(item.atom_label for item in result.species) == (
        "Cs-133",
        "Rb-87",
    )
    output = plot_dual_species_handover_map(
        result,
        tmp_path / "handover_map.png",
    )
    assert output.exists()
    assert output.stat().st_size > 0


def test_dual_species_map_can_use_l1_transport_endpoints(monkeypatch):
    def fake_handover(parameters):
        final_temperature = parameters.temperature_uK + 2.0
        return SimpleNamespace(
            transfer_efficiency=0.9,
            transfer_standard_error=0.01,
            handover_heating_uK=2.0,
            final_temperature_uK=final_temperature,
            trace=SimpleNamespace(
                time_ms=(0.0, parameters.duration_ms),
                kinetic_temperature_uK=(
                    parameters.temperature_uK,
                    final_temperature,
                ),
            ),
        )

    monkeypatch.setattr(
        integrated_handover,
        "run_handover_monte_carlo",
        fake_handover,
    )
    scan = HandoverMapInputs(
        detuning_points=3,
        power_points=3,
        particle_count=8,
        time_step_us=10.0,
        parallel_backend="serial",
        worker_count=1,
        require_minimum_depth=False,
        require_thermal_bound_fraction=False,
        require_minimum_axial_cycles=False,
        use_l1_transport=True,
    )
    result = analyze_dual_species_handover_map(scan)

    assert tuple(item.evaluated_points for item in result.species) == (6, 6)
    assert all(
        item.temperature_uK == pytest.approx(20.0)
        for item in result.species
    )


def test_process_backend_matches_serial_result():
    serial_inputs = HandoverMapInputs(
        detuning_points=3,
        power_points=3,
        particle_count=16,
        time_step_us=1.0,
        parallel_backend="serial",
        worker_count=1,
        use_l1_transport=False,
    )
    process_inputs = replace(
        serial_inputs,
        parallel_backend="process",
        worker_count=2,
    )

    serial = analyze_species_handover_map(
        CS133_HANDOVER_DEFAULTS,
        serial_inputs,
    )
    parallel = analyze_species_handover_map(
        CS133_HANDOVER_DEFAULTS,
        process_inputs,
    )

    assert parallel.feasible == serial.feasible
    assert parallel.transfer_efficiency == serial.transfer_efficiency
    assert parallel.transfer_standard_error == serial.transfer_standard_error
