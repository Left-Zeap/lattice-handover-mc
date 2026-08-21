from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import continuous_loading.l1_handover as integrated
from continuous_loading.full_chain import (
    FullChainInputs,
    analyze_full_chain_scan,
    simulate_full_chain_point,
)
from continuous_loading.full_chain_plots import plot_full_chain_scan
from continuous_loading.gpu_backend import cupy_available
from continuous_loading.l1_handover import L1HandoverInputs
from continuous_loading.l1_transport import l1_transport_inputs_for_species
from continuous_loading.l2_transport import L2TransportInputs


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


def _small_inputs(transport):
    handover = L1HandoverInputs(
        transport=transport,
        particle_count=8,
        time_step_us=10.0,
        parallel_backend="serial",
        worker_count=1,
        trace_points=3,
    )
    # 本文件的 fake handover 走 (N,T) 约化接口；连续相空间路径在
    # test_sequence_improvements.py 覆盖。
    return FullChainInputs(
        handover=handover,
        l2=L2TransportInputs(),
        phase_space_continuity=False,
    )


def test_chain_passes_handover_captured_state_into_l2(monkeypatch):
    calls = []
    monkeypatch.setattr(integrated, "run_handover_monte_carlo", _fake_handover(calls))
    transport = replace(
        l1_transport_inputs_for_species("Rb-87"),
        initial_temperature_uK=30.0,
        time_points=21,
    )
    inputs = _small_inputs(transport)

    simulation = simulate_full_chain_point(inputs, 300.0, 1.0, trace_points=3)

    assert len(calls) == 1
    handover_call = calls[0]
    # fake handover 的捕获末温必须成为 L2 腿初温。
    expected_l2_input = handover_call.temperature_uK + 2.0
    assert simulation.l2_result.input_temperature_uK == pytest.approx(
        expected_l2_input
    )
    # 捕获原子数 = L1 末态原子数 × 交接率，必须成为 L2 腿初态原子数。
    captured = handover_call.initial_atom_number * 0.9
    assert simulation.l2_result.input_atom_number == pytest.approx(captured)
    point = simulation.point
    assert point.l2_final_temperature_uK > expected_l2_input
    assert point.final_retention_from_mot == pytest.approx(
        point.l1_handover.final_retention_from_mot
        * point.l2_retention_fraction
    )
    # 拼接轨迹含三相且时间单调（段交界处允许重复时间戳，与
    # l1_handover 的拼接约定一致）。
    phases = simulation.combined_trace.phase
    assert set(phases) == {"L1 transport", "handover", "L2 transport"}
    times = np.asarray(simulation.combined_trace.time_ms)
    assert np.all(np.diff(times) >= 0.0)
    assert simulation.combined_trace.l2_end_ms == pytest.approx(times[-1])


def test_scan_selects_working_points_and_plots(monkeypatch, tmp_path):
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
    inputs = _small_inputs(transport)

    result = analyze_full_chain_scan(inputs)

    assert result.evaluated_points > 0
    retention = np.asarray(result.final_retention_from_mot, dtype=float)
    assert retention.shape == (3, 3)
    heating = np.asarray(result.science_total_temperature_rise_uK, dtype=float)
    valid = np.isfinite(heating)
    assert np.any(valid)
    # 科学区总升温必须大于只到 handover 的总升温。
    assert result.optimal.science_total_temperature_rise_uK > (
        result.optimal.l1_handover.total_temperature_rise_uK
    )
    assert result.optimal.final_retention_from_mot < (
        result.optimal.l1_handover.final_retention_from_mot
    )
    assert result.optimal.science_atom_number > 0.0
    output = plot_full_chain_scan(result, tmp_path / "full_chain.png")
    assert output.exists()
    assert output.stat().st_size > 0


def test_zero_capture_point_is_excluded(monkeypatch):
    def zero_capture(parameters):
        return SimpleNamespace(
            transfer_efficiency=0.0,
            transfer_standard_error=0.01,
            handover_heating_uK=None,
            final_temperature_uK=None,
            trace=SimpleNamespace(
                time_ms=(0.0, parameters.duration_ms),
                kinetic_temperature_uK=(
                    parameters.temperature_uK,
                    parameters.temperature_uK,
                ),
            ),
        )

    monkeypatch.setattr(
        integrated, "run_handover_monte_carlo", zero_capture
    )
    transport = replace(
        l1_transport_inputs_for_species("Rb-87"),
        detuning_points=2,
        power_points=2,
        time_points=21,
        initial_temperature_uK=30.0,
    )
    inputs = _small_inputs(transport)

    simulation = simulate_full_chain_point(inputs, 300.0, 1.0, trace_points=3)

    assert simulation.l2_result is None
    assert simulation.point.l2_final_temperature_uK is None
    assert simulation.point.science_atom_number is None
    assert simulation.point.final_retention_from_mot == 0.0
    result = analyze_full_chain_scan(inputs)
    # 全网格失败不再抛错：返回哨兵结果（空矩阵 + 哨兵点），保证总能出图。
    assert result.optimal_simulation is None
    assert result.optimal.final_retention_from_mot == 0.0
    assert result.evaluated_points == 0


GPU_REQUIRED = pytest.mark.skipif(
    not cupy_available(),
    reason="未检测到可用的 CuPy/CUDA 环境",
)


@GPU_REQUIRED
def test_full_chain_gpu_mc_l2_legs_batched(monkeypatch):
    """GPU + MC 运输的全链路：全部候选点的 L2 腿合并为一次批量调用。

    L2 腿各点初温/原子数来自该点 handover 捕获样本（逐点不同），依赖
    transport_batch 的逐点初态白名单；用 spy 验证调度决策与进度文案。
    """
    import continuous_loading.full_chain as full_chain_module

    real_batch = full_chain_module.run_leg_monte_carlo_batch
    batch_calls: list[int] = []

    def spy_batch(tasks, *, backend, progress=None):
        tasks = list(tasks)
        assert backend == "gpu"
        batch_calls.append(len(tasks))
        return real_batch(tasks, backend=backend, progress=progress)

    monkeypatch.setattr(
        full_chain_module, "run_leg_monte_carlo_batch", spy_batch
    )
    transport = replace(
        l1_transport_inputs_for_species("Rb-87"),
        distance_m=0.005,
        minimum_waist_um=None,
        minimum_waist_position_m=None,
        acceleration_m_s2=4000.0,
        maximum_velocity_m_s=4.0,
        detuning_min_ghz=250.0,
        detuning_max_ghz=350.0,
        detuning_points=2,
        handover_source_power_min_w=0.5,
        handover_source_power_max_w=1.0,
        power_points=2,
        time_points=11,
        transport_method="monte_carlo",
        transport_time_step_us=0.5,
        mc_particle_count=300,
        mc_compute_backend="gpu",
    )
    inputs = FullChainInputs(
        handover=L1HandoverInputs(
            transport=transport,
            particle_count=300,
            time_step_us=0.5,
            trace_points=2,
            compute_backend="gpu",
        ),
        l2=L2TransportInputs(
            distance_m=0.005,
            maximum_velocity_m_s=4.0,
            time_points=11,
        ),
        # 本测试走 (N,T) 约化接口的 GPU 批量 L2 腿调度。
        phase_space_continuity=False,
    )
    messages: list[str] = []
    result = analyze_full_chain_scan(inputs, progress=messages.append)

    assert result.evaluated_points > 0
    # 全部候选点的 L2 腿恰好合并成一次批量调用（L1 腿走 l1_handover
    # 自己的批量路径，不经此 spy）。
    assert batch_calls == [result.evaluated_points]
    assert any(
        "L2" in message and "批量" in message for message in messages
    )
    retention = [
        value
        for row in result.final_retention_from_mot
        for value in row
        if value is not None
    ]
    assert retention
    assert all(0.0 <= value <= 1.0 for value in retention)
