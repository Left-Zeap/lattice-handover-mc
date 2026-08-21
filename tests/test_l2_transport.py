import math

import pytest

from continuous_loading.l1_transport import l1_transport_inputs_for_species
from continuous_loading.l2_transport import (
    L2TransportInputs,
    simulate_l2_transport,
)


def test_l2_defaults_reproduce_paper_stage():
    inputs = L2TransportInputs()

    assert inputs.distance_m == pytest.approx(0.17)
    assert inputs.end_waist_um == pytest.approx(150.0)
    # 梯形时序应复现论文 21 ms 的 L2 段时长。
    acceleration_time = inputs.maximum_velocity_m_s / inputs.acceleration_m_s2
    cruise_distance = (
        inputs.distance_m
        - inputs.maximum_velocity_m_s**2 / inputs.acceleration_m_s2
    )
    total_time = 2.0 * acceleration_time + cruise_distance / inputs.maximum_velocity_m_s
    assert total_time == pytest.approx(0.021, rel=0.02)


def test_l2_keeps_source_power_fixed():
    transport = l1_transport_inputs_for_species("Rb-87")
    result = simulate_l2_transport(
        transport,
        L2TransportInputs(),
        detuning_ghz=300.0,
        handover_source_power_w=1.0,
        captured_temperature_uK=35.0,
        captured_atom_number=3.0e6,
    )

    assert result.end_source_power_w == pytest.approx(1.0, rel=1e-12)
    assert result.leg_trace.point.start_source_power_w == pytest.approx(1.0)


def test_l2_adiabatic_compression_heats_and_retention_bounds_atom_number():
    transport = l1_transport_inputs_for_species("Rb-87")
    captured_atoms = 3.0e6
    result = simulate_l2_transport(
        transport,
        L2TransportInputs(),
        detuning_ghz=300.0,
        handover_source_power_w=1.0,
        captured_temperature_uK=35.0,
        captured_atom_number=captured_atoms,
    )

    # 默认 L1 交接半径约 323 µm，L2 压缩至 150 µm；绝热压缩会显著升温，
    # 散射反冲只会在此基础上继续升温。
    assert result.final_temperature_uK > 35.0 * 1.3
    # 散射和噪声损失的默认系数为零，腿内留存只来自有限势垒热溢出。
    assert 0.0 < result.leg_retention_fraction <= 1.0
    assert result.final_atom_number == pytest.approx(
        captured_atoms * result.leg_retention_fraction
    )
    assert result.science.atom_number == pytest.approx(result.final_atom_number)
    assert result.science.atoms_per_site == pytest.approx(
        result.final_atom_number / result.science.occupied_lattice_sites
    )
    assert result.science.peak_density_m3 > 0.0


def test_l2_rejects_invalid_captured_state():
    transport = l1_transport_inputs_for_species("Rb-87")
    with pytest.raises(ValueError, match="捕获温度"):
        simulate_l2_transport(
            transport,
            L2TransportInputs(),
            detuning_ghz=300.0,
            handover_source_power_w=1.0,
            captured_temperature_uK=0.0,
            captured_atom_number=3.0e6,
        )
    with pytest.raises(ValueError, match="捕获原子数"):
        simulate_l2_transport(
            transport,
            L2TransportInputs(),
            detuning_ghz=300.0,
            handover_source_power_w=1.0,
            captured_temperature_uK=35.0,
            captured_atom_number=0.0,
        )


def test_l2_velocity_must_fit_distance():
    with pytest.raises(ValueError, match="距离不足"):
        L2TransportInputs(maximum_velocity_m_s=math.sqrt(4000.0 * 0.17) + 1.0)
