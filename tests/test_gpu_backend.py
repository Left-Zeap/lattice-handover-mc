"""GPU（CuPy）计算后端的守卫与 CPU/GPU 统计一致性测试。

没有 CuPy/CUDA 的环境自动跳过 GPU 用例；CPU 与 GPU 使用不同随机
数生成器，一致性断言全部是统计容差而非逐位相等。
"""

import math
import os
from dataclasses import replace

# 必须在 import PySide6 之前设置离屏平台。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from continuous_loading import gpu_backend
from continuous_loading.gpu_backend import cupy_available, resolve_backend
from continuous_loading.handover import run_handover_monte_carlo
from continuous_loading.l1_handover import L1HandoverInputs
from continuous_loading.l1_transport import (
    l1_transport_inputs_for_species,
    simulate_l1_transport,
)
from continuous_loading.scenarios import paper_handover_parameters


GPU_REQUIRED = pytest.mark.skipif(
    not cupy_available(),
    reason="未检测到可用的 CuPy/CUDA 环境",
)


def test_resolve_backend_validation():
    """非法后端与无 CuPy 时的 gpu 请求都必须给出明确错误。"""
    assert resolve_backend("cpu") == "cpu"
    with pytest.raises(ValueError, match="cpu 或 gpu"):
        resolve_backend("tpu")
    monkeypatch_target = gpu_backend.cupy_available
    try:
        gpu_backend.cupy_available = lambda: False
        with pytest.raises(ValueError, match="CuPy"):
            resolve_backend("gpu")
    finally:
        gpu_backend.cupy_available = monkeypatch_target


@GPU_REQUIRED
def test_gpu_handover_matches_cpu_statistically():
    """论文工作点、N=2000：GPU 与 CPU 的交接率和末温统计一致。"""
    parameters = replace(
        paper_handover_parameters(),
        particle_count=2_000,
        time_step_us=0.5,
    )
    cpu_result = run_handover_monte_carlo(
        replace(parameters, compute_backend="cpu")
    )
    gpu_result = run_handover_monte_carlo(
        replace(parameters, compute_backend="gpu")
    )

    tolerance = 3.0 * (
        cpu_result.transfer_standard_error + gpu_result.transfer_standard_error
    )
    assert abs(cpu_result.transfer_efficiency - gpu_result.transfer_efficiency) <= (
        tolerance + 1e-12
    )
    relative_temperature_difference = abs(
        cpu_result.final_temperature_uK - gpu_result.final_temperature_uK
    ) / cpu_result.final_temperature_uK
    assert relative_temperature_difference < 0.10


@GPU_REQUIRED
def test_gpu_seed_reproducible_on_same_backend():
    """GPU 后端同参同种子两次运行结果逐等。"""
    parameters = replace(
        paper_handover_parameters(),
        particle_count=500,
        time_step_us=0.5,
        compute_backend="gpu",
    )
    first = run_handover_monte_carlo(parameters)
    second = run_handover_monte_carlo(parameters)

    assert first.transfer_efficiency == second.transfer_efficiency
    assert first.final_temperature_uK == second.final_temperature_uK


@GPU_REQUIRED
def test_gpu_handover_fixed_phase_returns_captured_ensemble():
    """固定相位口径（randomize_relative_phase=False）下 GPU 后端返回
    捕获相空间系综的回归测试。

    回归背景：固定相位时 phase2 为标量，捕获相位数组建在 host 上，
    旧代码在 GPU 路径对其调用 .get()（CuPy 下载方法）而抛
    AttributeError——Cs + 固定相位 + 连续相空间扫描因此无有效结果。
    """
    parameters = replace(
        paper_handover_parameters(),
        particle_count=500,
        time_step_us=0.5,
        compute_backend="gpu",
        randomize_relative_phase=False,
        relative_phase_rad=0.7,
    )
    result, captured = run_handover_monte_carlo(
        parameters, return_captured_ensemble=True
    )
    assert 0.0 <= result.transfer_efficiency <= 1.0
    if result.captured_count > 0:
        assert captured is not None
        assert captured.particle_count == result.captured_count
        captured.host_arrays()  # 校验宿主数组形状/有限性


@GPU_REQUIRED
def test_gpu_transport_leg_matches_cpu_statistically():
    """小距离运输腿：GPU 与 CPU 的留存率/末温统计一致。"""
    base = replace(
        l1_transport_inputs_for_species("Rb-87"),
        distance_m=0.005,
        minimum_waist_um=None,
        minimum_waist_position_m=None,
        acceleration_m_s2=4000.0,
        maximum_velocity_m_s=4.0,
        time_points=11,
        transport_method="monte_carlo",
        transport_time_step_us=0.5,
        mc_particle_count=500,
    )
    cpu_trace = simulate_l1_transport(
        replace(base, mc_compute_backend="cpu"), 300.0, 1.0
    )
    gpu_trace = simulate_l1_transport(
        replace(base, mc_compute_backend="gpu"), 300.0, 1.0
    )

    assert abs(
        cpu_trace.point.final_retention_fraction
        - gpu_trace.point.final_retention_fraction
    ) <= 3.0 * (
        cpu_trace.retention_standard_error
        + gpu_trace.retention_standard_error
    ) + 1e-12
    relative_difference = abs(
        cpu_trace.point.final_temperature_uK
        - gpu_trace.point.final_temperature_uK
    ) / cpu_trace.point.final_temperature_uK
    assert relative_difference < 0.15


def test_ui_form_compute_backend_roundtrip():
    """表单"计算设备"往返并传递到运输腿与 handover 两套输入。"""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui import controllers
    from ui.widgets.forms import ChainParameterForm

    form = ChainParameterForm()
    params = form.params()
    assert params["compute_backend"] == "cpu"

    combo = form._widgets["compute_backend"]
    combo.setCurrentIndex(combo.findData("gpu"))
    updated = form.params()
    assert updated["compute_backend"] == "gpu"

    inputs = controllers.build_full_chain_inputs(updated)
    assert inputs.handover.compute_backend == "gpu"
    assert inputs.handover.transport.mc_compute_backend == "gpu"
    form.close()
    form.deleteLater()


def test_l1_handover_inputs_backend_field():
    """L1HandoverInputs.compute_backend 传入 HandoverParameters。"""
    pytest.importorskip("continuous_loading.handover_map")
    inputs = L1HandoverInputs(compute_backend="gpu")
    assert inputs.compute_backend == "gpu"
    with pytest.raises(ValueError):
        L1HandoverInputs(compute_backend="tpu")


def test_gpu_scan_uses_batch_dispatch(monkeypatch):
    """GPU 后端的扫描必须走批量路径：一次 batch 调用覆盖全部网格点。

    不需要真实 GPU：fake batch/单点 handover 绕过后端检查，验证调度
    决策、进度文案与结果矩阵组装。
    """
    from types import SimpleNamespace

    import continuous_loading.l1_handover as integrated
    from continuous_loading.l1_handover import analyze_l1_handover_scan
    from continuous_loading.l1_transport import l1_transport_inputs_for_species

    def fake_result(parameters):
        return SimpleNamespace(
            transfer_efficiency=0.9,
            transfer_standard_error=0.01,
            handover_heating_uK=2.0,
            final_temperature_uK=parameters.temperature_uK + 2.0,
            trace=SimpleNamespace(
                time_ms=(0.0, parameters.duration_ms),
                kinetic_temperature_uK=(parameters.temperature_uK,) * 2,
            ),
        )

    batch_calls: list[int] = []

    def fake_batch(parameters_list, *, backend, progress=None):
        assert backend == "gpu"
        batch_calls.append(len(parameters_list))
        return [fake_result(parameters) for parameters in parameters_list]

    monkeypatch.setattr(
        integrated, "run_handover_monte_carlo", fake_result
    )
    monkeypatch.setattr(
        integrated, "run_handover_monte_carlo_batch", fake_batch
    )
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
    inputs = L1HandoverInputs(
        transport=transport,
        particle_count=20,
        time_step_us=1.0,
        trace_points=3,
        parallel_backend="process",
        worker_count=4,
        compute_backend="gpu",
    )
    messages: list[str] = []
    result = analyze_l1_handover_scan(
        inputs, progress=messages.append
    )

    # 全部网格点一次批量调用；best/comparison 复算走单点 fake。
    assert batch_calls == [result.evaluated_points]
    assert result.evaluated_points > 0
    assert any("GPU 批量" in message for message in messages)
    feasible = [
        result.handover_transfer_efficiency[p][d]
        for p in range(2)
        for d in range(2)
        if result.transport_feasible[p][d]
    ]
    assert feasible and all(value == 0.9 for value in feasible)


@GPU_REQUIRED
def test_gpu_mega_step_handover_matches_cpu_strict():
    """mega-step kernel：GPU 与 CPU 同 seed 交接率差 ≤3σ、末温 <2%。"""
    parameters = replace(
        paper_handover_parameters(),
        particle_count=2_000,
        time_step_us=0.5,
    )
    cpu_result = run_handover_monte_carlo(
        replace(parameters, compute_backend="cpu")
    )
    gpu_result = run_handover_monte_carlo(
        replace(parameters, compute_backend="gpu")
    )

    tolerance = 3.0 * (
        cpu_result.transfer_standard_error + gpu_result.transfer_standard_error
    )
    assert abs(
        cpu_result.transfer_efficiency - gpu_result.transfer_efficiency
    ) <= (tolerance + 1e-12)
    relative_temperature_difference = abs(
        cpu_result.final_temperature_uK - gpu_result.final_temperature_uK
    ) / cpu_result.final_temperature_uK
    assert relative_temperature_difference < 0.02
    # 平均散射事件数（固定事件槽实现）与逐事件 CPU 口径统计一致。
    assert (
        abs(
            gpu_result.mean_scattering_events
            - cpu_result.mean_scattering_events
        )
        / cpu_result.mean_scattering_events
        < 0.10
    )


def _batch_test_points():
    """三个仅逐点物理量不同的 GPU handover 参数点。"""
    base = replace(
        paper_handover_parameters(),
        particle_count=1_000,
        time_step_us=0.5,
        compute_backend="gpu",
    )
    return [
        replace(base, temperature_uK=temperature, depth2_uK=depth)
        for temperature, depth in (
            (30.8, 500.0),
            (25.0, 450.0),
            (35.0, 550.0),
        )
    ]


@GPU_REQUIRED
def test_handover_batch_matches_pointwise_gpu():
    """批量与逐点 GPU 调用统计一致（效率 ≤3σ、末温 <10%）。"""
    from continuous_loading.handover_batch import run_handover_monte_carlo_batch

    points = _batch_test_points()
    batch_results = run_handover_monte_carlo_batch(points, backend="gpu")
    solo_results = [run_handover_monte_carlo(point) for point in points]

    assert len(batch_results) == len(points)
    for batch, solo in zip(batch_results, solo_results):
        tolerance = 3.0 * (
            batch.transfer_standard_error + solo.transfer_standard_error
        )
        assert abs(
            batch.transfer_efficiency - solo.transfer_efficiency
        ) <= (tolerance + 1e-12)
        assert (
            abs(batch.final_temperature_uK - solo.final_temperature_uK)
            / solo.final_temperature_uK
            < 0.10
        )
        # 批量 trace 只保留两个端点。
        assert len(batch.trace.time_ms) == 2
        assert batch.trace.time_ms[0] == 0.0


@GPU_REQUIRED
def test_handover_batch_matches_pointwise_cpu():
    """批量 GPU 与逐点 CPU 调用统计一致（效率 ≤3σ、末温 <10%）。"""
    from continuous_loading.handover_batch import run_handover_monte_carlo_batch

    points = _batch_test_points()
    batch_results = run_handover_monte_carlo_batch(points, backend="gpu")
    cpu_results = [
        run_handover_monte_carlo(replace(point, compute_backend="cpu"))
        for point in points
    ]

    for batch, cpu in zip(batch_results, cpu_results):
        tolerance = 3.0 * (
            batch.transfer_standard_error + cpu.transfer_standard_error
        )
        assert abs(
            batch.transfer_efficiency - cpu.transfer_efficiency
        ) <= (tolerance + 1e-12)
        assert (
            abs(batch.final_temperature_uK - cpu.final_temperature_uK)
            / cpu.final_temperature_uK
            < 0.10
        )


def test_handover_batch_rejects_inconsistent():
    """批量一致性约束不满足时抛 ValueError（在校验阶段、无需 GPU）。"""
    from continuous_loading.handover_batch import run_handover_monte_carlo_batch

    points = _batch_test_points()
    inconsistent = replace(points[1], particle_count=500)
    with pytest.raises(ValueError, match="全批一致"):
        run_handover_monte_carlo_batch(
            [points[0], inconsistent], backend="gpu"
        )
    with pytest.raises(ValueError, match="不能为空"):
        run_handover_monte_carlo_batch([], backend="gpu")


def test_handover_batch_cpu_backend_bitwise():
    """backend="cpu" 的批量退化为逐点调用，结果逐位一致。"""
    from continuous_loading.handover_batch import run_handover_monte_carlo_batch

    points = [
        replace(point, particle_count=200, time_step_us=1.0, compute_backend="cpu")
        for point in _batch_test_points()
    ]
    batch_results = run_handover_monte_carlo_batch(points, backend="cpu")
    solo_results = [run_handover_monte_carlo(point) for point in points]
    for batch, solo in zip(batch_results, solo_results):
        assert batch.transfer_efficiency == solo.transfer_efficiency
        assert batch.final_temperature_uK == solo.final_temperature_uK
        assert batch.trace == solo.trace


def test_handover_batch_cpu_carries_supplied_phase_space():
    import numpy as np

    from continuous_loading.handover_batch import run_handover_monte_carlo_batch
    from continuous_loading.phase_space import ParticleEnsemble

    point = replace(
        _batch_test_points()[0],
        particle_count=16,
        duration_ms=0.01,
        time_step_us=1.0,
        include_scattering=False,
        compute_backend='cpu',
    )
    ensemble = ParticleEnsemble(
        positions_m=np.zeros((16, 3)),
        velocities_m_s=np.zeros((16, 3)),
        frame='handover_l1_local',
    )
    results, captured = run_handover_monte_carlo_batch(
        [point],
        backend='cpu',
        initial_ensembles=[ensemble],
        return_captured_ensembles=True,
    )
    assert len(results) == len(captured) == 1
    assert results[0].captured_count == (
        0 if captured[0] is None else captured[0].particle_count
    )


@GPU_REQUIRED
def test_handover_batch_chunking_deterministic(monkeypatch):
    """显存保护分块不改变结果（无散射时逐块与整批逐等）。"""
    import continuous_loading.handover_batch as handover_batch

    points = [
        replace(
            point,
            include_scattering=False,
            randomize_relative_phase=False,
        )
        for point in _batch_test_points()
    ]
    reference = handover_batch.run_handover_monte_carlo_batch(
        points, backend="gpu"
    )
    # 阈值设为单点粒子数 → 强制每块只有 1 个点。
    monkeypatch.setattr(
        handover_batch,
        "_MAX_BATCH_PARTICLES",
        points[0].particle_count,
    )
    chunked = handover_batch.run_handover_monte_carlo_batch(
        points, backend="gpu"
    )
    for chunked_result, reference_result in zip(chunked, reference):
        assert (
            chunked_result.transfer_efficiency
            == reference_result.transfer_efficiency
        )
        assert (
            chunked_result.final_temperature_uK
            == reference_result.final_temperature_uK
        )


@GPU_REQUIRED
def test_l1_handover_scan_gpu_real_small_grid():
    """2×2 小网格实测：gpu 后端走批量路径且结果矩阵有效。"""
    from continuous_loading.l1_handover import analyze_l1_handover_scan

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
    inputs = L1HandoverInputs(
        transport=transport,
        particle_count=200,
        time_step_us=1.0,
        trace_points=2,
        compute_backend="gpu",
    )
    messages: list[str] = []
    result = analyze_l1_handover_scan(inputs, progress=messages.append)

    assert result.evaluated_points > 0
    assert any("GPU 批量" in message for message in messages)
    efficiency = [
        value
        for row in result.handover_transfer_efficiency
        for value in row
        if value is not None
    ]
    assert efficiency
    assert all(0.0 <= value <= 1.0 for value in efficiency)


def _leg_batch_test_inputs(**overrides):
    """小距离 MC 运输腿输入（GPU）。"""
    base = replace(
        l1_transport_inputs_for_species("Rb-87"),
        distance_m=0.005,
        minimum_waist_um=None,
        minimum_waist_position_m=None,
        acceleration_m_s2=4000.0,
        maximum_velocity_m_s=4.0,
        time_points=11,
        transport_time_step_us=0.5,
        mc_particle_count=500,
        mc_compute_backend="gpu",
        **overrides,
    )
    return base


@GPU_REQUIRED
def test_leg_batch_matches_pointwise_gpu():
    """批量运输腿 vs 逐点 GPU：留存率 ≤3σ、末温相对差 <15%。"""
    from continuous_loading.transport_batch import run_leg_monte_carlo_batch
    from continuous_loading.transport_mc import simulate_leg_monte_carlo

    inputs = _leg_batch_test_inputs()
    tasks = [
        ((0, point), inputs, detuning, 1.0)
        for point, detuning in enumerate((250.0, 300.0, 350.0))
    ]
    batch = run_leg_monte_carlo_batch(tasks, backend="gpu")
    solo = [
        simulate_leg_monte_carlo(inputs, detuning, 1.0)
        for _, _, detuning, _ in tasks
    ]

    assert len(batch) == len(tasks)
    for batch_trace, solo_trace in zip(batch, solo):
        tolerance = 3.0 * (
            batch_trace.retention_standard_error
            + solo_trace.retention_standard_error
        )
        assert abs(
            batch_trace.point.final_retention_fraction
            - solo_trace.point.final_retention_fraction
        ) <= (tolerance + 1e-12)
        assert (
            abs(
                batch_trace.point.final_temperature_uK
                - solo_trace.point.final_temperature_uK
            )
            / solo_trace.point.final_temperature_uK
            < 0.15
        )
        # 快照网格与逐点一致；t=0 初态温度同 seed 近逐位（归约顺序
        # 不同，允许 1e-9 级浮点差）。
        assert len(batch_trace.time_ms) == len(solo_trace.time_ms)
        assert math.isclose(
            batch_trace.temperature_uK[0],
            solo_trace.temperature_uK[0],
            rel_tol=1e-9,
        )


def test_leg_batch_rejects_inconsistent():
    """inputs 不全等或 conveyor 几何时抛 ValueError（无需 GPU）。"""
    from continuous_loading.transport_batch import run_leg_monte_carlo_batch

    inputs = _leg_batch_test_inputs()
    task = ((0, 0), inputs, 300.0, 1.0)
    other = ((0, 1), replace(inputs, distance_m=0.006), 300.0, 1.0)
    with pytest.raises(ValueError, match="全等"):
        run_leg_monte_carlo_batch([task, other], backend="gpu")
    conveyor = ((0, 0), replace(inputs, conveyor_enabled=True), 300.0, 1.0)
    with pytest.raises(ValueError, match="conveyor"):
        run_leg_monte_carlo_batch([conveyor], backend="gpu")
    with pytest.raises(ValueError, match="不能为空"):
        run_leg_monte_carlo_batch([], backend="gpu")


def _leg_batch_per_point_initial_state_variants(**overrides):
    """三个仅初温/初原子数/MOT 原子数不同的运输腿输入（白名单字段）。"""
    base = _leg_batch_test_inputs(**overrides)
    return [
        replace(
            base,
            initial_temperature_uK=25.0,
            initial_atom_number=1.0e6,
            mot_atom_number=1.0e7,
        ),
        replace(
            base,
            initial_temperature_uK=30.0,
            initial_atom_number=5.0e5,
            mot_atom_number=5.0e6,
        ),
        replace(
            base,
            initial_temperature_uK=35.0,
            initial_atom_number=2.0e6,
            mot_atom_number=2.0e7,
        ),
    ]


def test_leg_batch_allows_per_point_initial_state():
    """逐点初态字段（初温/原子数）不同的任务可同批，CPU 下逐位一致。"""
    from continuous_loading.transport_batch import run_leg_monte_carlo_batch
    from continuous_loading.transport_mc import simulate_leg_monte_carlo

    variants = [
        replace(variant, mc_compute_backend="cpu")
        for variant in _leg_batch_per_point_initial_state_variants()
    ]
    tasks = [
        ((0, number), variant, 300.0, 1.0)
        for number, variant in enumerate(variants)
    ]
    batch = run_leg_monte_carlo_batch(tasks, backend="cpu")
    solo = [
        simulate_leg_monte_carlo(variant, 300.0, 1.0) for variant in variants
    ]
    assert len(batch) == len(tasks)
    for batch_trace, solo_trace, variant in zip(batch, solo, variants):
        assert batch_trace.point == solo_trace.point
        assert batch_trace.temperature_uK == solo_trace.temperature_uK
        # 末端原子数按各点自身初态原子数缩放。
        assert batch_trace.point.final_atom_number == (
            variant.initial_atom_number
            * batch_trace.point.final_retention_fraction
        )


@GPU_REQUIRED
def test_leg_batch_per_point_initial_state_matches_pointwise_gpu():
    """逐点初态不同的批量运输腿与逐点 GPU 统计一致（≤3σ、末温 <15%）。"""
    from continuous_loading.transport_batch import run_leg_monte_carlo_batch
    from continuous_loading.transport_mc import simulate_leg_monte_carlo

    variants = _leg_batch_per_point_initial_state_variants()
    tasks = [
        ((0, number), variant, 300.0, 1.0)
        for number, variant in enumerate(variants)
    ]
    batch = run_leg_monte_carlo_batch(tasks, backend="gpu")
    solo = [
        simulate_leg_monte_carlo(variant, 300.0, 1.0) for variant in variants
    ]

    assert len(batch) == len(tasks)
    for batch_trace, solo_trace, variant in zip(batch, solo, variants):
        tolerance = 3.0 * (
            batch_trace.retention_standard_error
            + solo_trace.retention_standard_error
        )
        assert abs(
            batch_trace.point.final_retention_fraction
            - solo_trace.point.final_retention_fraction
        ) <= (tolerance + 1e-12)
        assert (
            abs(
                batch_trace.point.final_temperature_uK
                - solo_trace.point.final_temperature_uK
            )
            / solo_trace.point.final_temperature_uK
            < 0.15
        )
        assert batch_trace.point.final_atom_number == (
            variant.initial_atom_number
            * batch_trace.point.final_retention_fraction
        )


def test_leg_batch_cpu_backend_bitwise():
    """backend="cpu" 的批量运输腿退化为逐点调用，结果逐位一致。"""
    from continuous_loading.transport_batch import run_leg_monte_carlo_batch
    from continuous_loading.transport_mc import simulate_leg_monte_carlo

    inputs = _leg_batch_test_inputs()
    tasks = [((0, 0), inputs, 300.0, 1.0), ((0, 1), inputs, 350.0, 0.8)]
    batch = run_leg_monte_carlo_batch(tasks, backend="cpu")
    solo = [
        simulate_leg_monte_carlo(inputs, detuning, power)
        for _, _, detuning, power in tasks
    ]
    for batch_trace, solo_trace in zip(batch, solo):
        assert batch_trace.point == solo_trace.point
        assert batch_trace.temperature_uK == solo_trace.temperature_uK


@GPU_REQUIRED
def test_scan_gpu_progress_messages():
    """GPU 扫描全程有进度反馈：L1 腿、批量 handover、步进与收尾。

    各阶段消息都含 n/total；同一阶段内比例单调不减；收尾消息达到
    total/total（阶段间比例回退由 UI 进度条 max 逻辑吸收）。
    """
    import re

    from continuous_loading.l1_handover import analyze_l1_handover_scan

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
    inputs = L1HandoverInputs(
        transport=transport,
        particle_count=200,
        time_step_us=1.0,
        trace_points=2,
        compute_backend="gpu",
    )
    messages: list[str] = []
    result = analyze_l1_handover_scan(inputs, progress=messages.append)
    assert result.evaluated_points > 0

    pattern = re.compile(r"(\d+)\s*/\s*(\d+)")
    phases: dict[str, list[float]] = {}
    for message in messages:
        match = pattern.search(message)
        if match and int(match.group(2)) > 0:
            key = message[: match.start()]
            phases.setdefault(key, []).append(
                int(match.group(1)) / int(match.group(2))
            )
    # 至少三个阶段有 n/total 进度：L1 腿、批量积分步进、逐点收尾。
    assert any("L1 运输腿" in key for key in phases)
    assert any("批量 handover 积分" in key for key in phases)
    assert any("批量运行" in message for message in messages)
    for ratios in phases.values():
        assert all(
            later >= earlier - 1e-12
            for earlier, later in zip(ratios, ratios[1:])
        ), phases
    # 收尾阶段达到 100%。
    final_key = f"{transport.atom_label}: "
    assert phases[final_key][-1] == 1.0


@GPU_REQUIRED
def test_scan_gpu_mc_transport_batch_legs():
    """GPU + MC 运输 2×2 小网格：批量腿路径完成且结果矩阵有效。"""
    from continuous_loading.l1_handover import analyze_l1_handover_scan

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
    inputs = L1HandoverInputs(
        transport=transport,
        particle_count=300,
        time_step_us=1.0,
        trace_points=2,
        compute_backend="gpu",
    )
    messages: list[str] = []
    result = analyze_l1_handover_scan(inputs, progress=messages.append)

    assert result.evaluated_points > 0
    assert any("批量运行" in message and "L1 运输" in message for message in messages)
    assert any("批量运输腿积分" in message for message in messages)
    efficiency = [
        value
        for row in result.handover_transfer_efficiency
        for value in row
        if value is not None
    ]
    assert efficiency
    assert all(0.0 <= value <= 1.0 for value in efficiency)


@GPU_REQUIRED
def test_device_loop_handover_matches_fused_fallback(monkeypatch):
    """设备端循环 kernel 与逐步融合 kernel（强制回退）统计一致。"""
    import continuous_loading.handover_batch as handover_batch

    points = _batch_test_points()
    device = handover_batch.run_handover_monte_carlo_batch(points, backend="gpu")
    monkeypatch.setattr(
        handover_batch, "get_handover_loop_kernels", lambda: None
    )
    fused = handover_batch.run_handover_monte_carlo_batch(points, backend="gpu")
    for device_result, fused_result in zip(device, fused):
        tolerance = 3.0 * (
            device_result.transfer_standard_error
            + fused_result.transfer_standard_error
        )
        assert abs(
            device_result.transfer_efficiency
            - fused_result.transfer_efficiency
        ) <= (tolerance + 1e-12)
        assert (
            abs(
                device_result.final_temperature_uK
                - fused_result.final_temperature_uK
            )
            / fused_result.final_temperature_uK
            < 0.10
        )


@GPU_REQUIRED
def test_device_loop_leg_matches_fused_fallback_and_deterministic(monkeypatch):
    """设备端循环腿 kernel：与逐步融合路径统计一致，且同 seed 逐位确定。"""
    import continuous_loading.transport_batch as transport_batch

    inputs = _leg_batch_test_inputs()
    tasks = [
        ((0, point), inputs, detuning, 1.0)
        for point, detuning in enumerate((250.0, 300.0, 350.0))
    ]
    first = transport_batch.run_leg_monte_carlo_batch(tasks, backend="gpu")
    second = transport_batch.run_leg_monte_carlo_batch(tasks, backend="gpu")
    # 同 seed 两次设备端循环运行结果逐等（cuRAND 序列确定）。
    for first_trace, second_trace in zip(first, second):
        assert (
            first_trace.point.final_retention_fraction
            == second_trace.point.final_retention_fraction
        )
        assert (
            first_trace.point.final_temperature_uK
            == second_trace.point.final_temperature_uK
        )

    monkeypatch.setattr(
        transport_batch, "get_leg_loop_kernels", lambda: None
    )
    fused = transport_batch.run_leg_monte_carlo_batch(tasks, backend="gpu")
    for device_trace, fused_trace in zip(first, fused):
        tolerance = 3.0 * (
            device_trace.retention_standard_error
            + fused_trace.retention_standard_error
        )
        assert abs(
            device_trace.point.final_retention_fraction
            - fused_trace.point.final_retention_fraction
        ) <= (tolerance + 1e-12)
        assert (
            abs(
                device_trace.point.final_temperature_uK
                - fused_trace.point.final_temperature_uK
            )
            / fused_trace.point.final_temperature_uK
            < 0.15
        )
