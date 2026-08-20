from types import SimpleNamespace

import pytest

import continuous_loading.handover_angle_scan as angle_scan
from continuous_loading.handover_angle_scan import (
    HandoverAngleScanInputs,
    analyze_handover_angle_scan,
    plot_handover_angle_scan,
)


def test_default_angle_range_is_zero_to_ninety_degrees():
    inputs = HandoverAngleScanInputs()

    assert inputs.angle_min_deg == pytest.approx(0.0)
    assert inputs.angle_max_deg == pytest.approx(90.0)
    assert inputs.angle_step_deg == pytest.approx(1.0)


def test_dual_species_angle_scan_and_plot(monkeypatch, tmp_path):
    calls = []

    def fake_monte_carlo(parameters):
        calls.append(parameters)
        return SimpleNamespace(
            transfer_efficiency=1.0 - parameters.crossing_angle_deg / 180.0,
            transfer_standard_error=0.01,
            handover_heating_uK=0.1 * parameters.crossing_angle_deg,
            all_atom_handover_heating_uK=(
                0.2 * parameters.crossing_angle_deg
            ),
        )

    monkeypatch.setattr(
        angle_scan,
        "run_handover_monte_carlo",
        fake_monte_carlo,
    )
    inputs = HandoverAngleScanInputs(
        angle_min_deg=0.0,
        angle_max_deg=2.0,
        angle_step_deg=1.0,
        particle_count=8,
        time_step_us=1.0,
        parallel_backend="serial",
        worker_count=1,
    )
    result = analyze_handover_angle_scan(inputs)

    assert len(calls) == 6
    assert tuple(item.atom_label for item in result.species) == (
        "Rb-87",
        "Cs-133",
    )
    assert result.species[0].angle_deg == (0.0, 1.0, 2.0)
    assert result.species[0].all_atom_handover_heating_uK == (
        0.0,
        0.2,
        0.4,
    )
    assert all(item.required_source_power_w > 0.0 for item in result.species)
    output = plot_handover_angle_scan(
        result,
        tmp_path / "handover_angle_scan.png",
    )
    assert output.exists()
    assert output.stat().st_size > 0
