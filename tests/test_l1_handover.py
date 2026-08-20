from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import continuous_loading.l1_handover as integrated
from continuous_loading.l1_handover import (
    L1HandoverInputs,
    analyze_l1_handover_scan,
)
from continuous_loading.l1_handover_plots import plot_l1_handover_scan
from continuous_loading.l1_transport import l1_transport_inputs_for_species


def _fake_handover(calls):
    def run(parameters):
        calls.append(parameters)
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

    return run


def test_defaults_share_transport_grid_and_use_1000_particles():
    inputs = L1HandoverInputs()

    assert inputs.transport.detuning_min_ghz == 100.0
    assert inputs.transport.detuning_max_ghz == 800.0
    assert inputs.transport.handover_source_power_min_w == 0.0
    assert inputs.transport.handover_source_power_max_w == 1.5
    assert inputs.transport.initial_temperature_uK == 20.0
    assert inputs.particle_count == 1_000


def test_scan_passes_transport_endpoint_into_handover_and_plots(
    monkeypatch,
    tmp_path,
):
    calls = []
    monkeypatch.setattr(integrated, "run_handover_monte_carlo", _fake_handover(calls))
    transport = replace(
        l1_transport_inputs_for_species("Rb-87"),
        detuning_min_ghz=100.0,
        detuning_max_ghz=300.0,
        detuning_points=3,
        handover_source_power_min_w=0.0,
        handover_source_power_max_w=1.5,
        power_points=3,
        time_points=21,
        initial_temperature_uK=30.0,
    )
    inputs = L1HandoverInputs(
        transport=transport,
        particle_count=8,
        time_step_us=10.0,
        parallel_backend="serial",
        worker_count=1,
        trace_points=3,
    )

    result = analyze_l1_handover_scan(inputs)

    assert result.evaluated_points > 0
    assert np.asarray(result.final_retention_from_mot, dtype=float).shape == (3, 3)
    assert all(call.temperature_uK > transport.initial_temperature_uK for call in calls)
    assert all(call.initial_atom_number <= transport.initial_atom_number for call in calls)
    assert result.optimal_simulation.combined_trace.temperature_uK[0] >= 30.0
    assert result.optimal.final_retention_from_mot < transport.loading_efficiency
    output = plot_l1_handover_scan(result, tmp_path / "combined.png")
    assert output.exists()
    assert output.stat().st_size > 0


def _fake_handover_with_se(calls):
    """标准误随 1/√N 变化的 fake handover（供自适应加密测试）。"""

    def run(parameters):
        calls.append(parameters)
        final_temperature = parameters.temperature_uK + 2.0
        return SimpleNamespace(
            parameters=parameters,
            transfer_efficiency=0.9,
            transfer_standard_error=1.0 / np.sqrt(parameters.particle_count),
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

    return run


def _adaptive_test_inputs(**overrides):
    transport = replace(
        l1_transport_inputs_for_species("Rb-87"),
        detuning_min_ghz=250.0,
        detuning_max_ghz=350.0,
        detuning_points=2,
        handover_source_power_min_w=0.5,
        handover_source_power_max_w=1.0,
        power_points=2,
        time_points=11,
    )
    return L1HandoverInputs(
        transport=transport,
        particle_count=100,
        time_step_us=10.0,
        parallel_backend="serial",
        worker_count=1,
        trace_points=3,
        **overrides,
    )


def test_adaptive_refinement_reruns_noisy_points(monkeypatch):
    """SE 超标的点以 1/√N 标定的更大粒子数复算并更新结果矩阵。"""
    calls = []
    monkeypatch.setattr(
        integrated, "run_handover_monte_carlo", _fake_handover_with_se(calls)
    )
    inputs = _adaptive_test_inputs(
        adaptive_refinement=True,
        adaptive_target_standard_error=0.02,
        adaptive_max_particle_count=100_000,
    )
    messages: list[str] = []
    result = analyze_l1_handover_scan(inputs, progress=messages.append)

    # 第一遍 SE = 1/√100 = 0.1 > 0.02 → 全部 4 个有效点复算，
    # N2 = ceil(100 × (0.1/0.02)²) = 2500。
    assert result.evaluated_points == 4
    assert result.refined_points == 4
    counts = [
        point.handover_particle_count
        for row in result.point_grid
        for point in row
        if point is not None
    ]
    assert counts and all(count == 2500 for count in counts)
    for row in result.handover_transfer_standard_error:
        for value in row:
            if value is not None:
                assert value == pytest.approx(1.0 / np.sqrt(2500))
    assert any("自适应粒子加密" in message for message in messages)
    # 第一遍 4 点 + 最优/较差重算 2 点 + 第二遍 4 点。
    assert [call.particle_count for call in calls].count(2500) == 4


def test_adaptive_refinement_skipped_when_accurate(monkeypatch):
    """全部点 SE 达标时不触发第二遍。"""
    calls = []
    monkeypatch.setattr(
        integrated, "run_handover_monte_carlo", _fake_handover_with_se(calls)
    )
    inputs = _adaptive_test_inputs(
        adaptive_refinement=True,
        adaptive_target_standard_error=0.2,
    )
    result = analyze_l1_handover_scan(inputs)

    assert result.refined_points == 0
    counts = {
        point.handover_particle_count
        for row in result.point_grid
        for point in row
        if point is not None
    }
    assert counts == {100}


def test_adaptive_refinement_validation():
    with pytest.raises(ValueError, match="目标标准误"):
        _adaptive_test_inputs(adaptive_target_standard_error=0.0)
    with pytest.raises(ValueError, match="粒子数上限"):
        _adaptive_test_inputs(adaptive_max_particle_count=10)


def test_validate_transport_trace_rejects_degenerate_temperature():
    """L1 末态温度退化（0/NaN/inf）或零存活必须被前置校验拒绝。

    回归：Cs 连续相空间 GPU 扫描曾因个别点仅剩 1 个幸存粒子
    （速度方差为零、温度恰为 0）穿过校验，随后在
    HandoverParameters"温度必须是有限正数"处整批崩溃。
    """
    from continuous_loading.l1_handover import _validate_transport_trace

    def trace_with(temperature, atom_number=1.0e6):
        return SimpleNamespace(
            point=SimpleNamespace(
                feasible_hardware_point=True,
                final_temperature_uK=temperature,
                final_atom_number=atom_number,
            )
        )

    for bad_temperature in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            _validate_transport_trace(trace_with(bad_temperature))
    with pytest.raises(ValueError):
        _validate_transport_trace(trace_with(30.0, atom_number=0.0))
    # 正常温度与正原子数通过。
    _validate_transport_trace(trace_with(30.0))
