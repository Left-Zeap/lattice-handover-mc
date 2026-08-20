"""cloud_sigma_scan.py 云宽一维扫描的 smoke、失败隔离与可重复性测试。

小成本口径：Rb-87、(N,T) 约化接口（解析 L1/L2 腿，无运输 MC）、
100 条 handover 轨迹、1 µs 步长、500 µs 交接，3 个 σ 点（含 σ=0，
即全部原子位于中心格点链）。每个 σ 点只有一次小规模 handover
Monte Carlo，全部测试数秒内完成。
"""

from dataclasses import replace

import pytest

import continuous_loading.cloud_sigma_scan as cloud_sigma_scan
from continuous_loading.cloud_sigma_scan import (
    CloudSigmaScanInputs,
    analyze_cloud_sigma_scan,
)
from continuous_loading.full_chain import FullChainInputs
from continuous_loading.l1_handover import L1HandoverInputs
from continuous_loading.l1_transport import l1_transport_inputs_for_species
from continuous_loading.l2_transport import L2TransportInputs


_DETUNING_GHZ = 300.0
_SOURCE_POWER_W = 1.0


def _small_inputs(**overrides) -> CloudSigmaScanInputs:
    transport = replace(
        l1_transport_inputs_for_species("Rb-87"),
        time_points=11,
        initial_temperature_uK=30.0,
    )
    handover = L1HandoverInputs(
        transport=transport,
        particle_count=100,
        duration_us=500.0,
        time_step_us=1.0,
        trace_points=2,
        parallel_backend="serial",
        worker_count=1,
    )
    chain = FullChainInputs(
        handover=handover,
        l2=L2TransportInputs(),
        # 本文件走 (N,T) 约化接口（解析腿，成本最低）；连续相空间
        # 路径复用同一个 simulate_full_chain_point 单点入口。
        phase_space_continuity=False,
    )
    base = dict(
        chain=chain,
        detuning_ghz=_DETUNING_GHZ,
        source_power_w=_SOURCE_POWER_W,
        sigma_min_mm=0.0,
        sigma_max_mm=0.4,
        points=3,
    )
    base.update(overrides)
    return CloudSigmaScanInputs(**base)


def test_inputs_validation():
    with pytest.raises(ValueError):
        _small_inputs(sigma_min_mm=-0.1)
    with pytest.raises(ValueError):
        _small_inputs(sigma_min_mm=0.5, sigma_max_mm=0.5)
    with pytest.raises(ValueError):
        _small_inputs(sigma_min_mm=0.6, sigma_max_mm=0.5)
    with pytest.raises(ValueError):
        _small_inputs(sigma_max_mm=float("nan"))
    with pytest.raises(ValueError):
        _small_inputs(points=1)
    with pytest.raises(ValueError):
        _small_inputs(detuning_ghz=0.0)
    with pytest.raises(ValueError):
        _small_inputs(source_power_w=0.0)


def test_sigma_written_to_both_fields():
    """两个云宽字段（约化 handover 自采样 / 连续初态采样）同步替换。"""
    chain = cloud_sigma_scan._chain_with_sigma(_small_inputs().chain, 0.3)
    assert chain.handover.cloud_axial_sigma_mm == pytest.approx(0.3)
    assert chain.handover.transport.mc_cloud_axial_sigma_mm == pytest.approx(0.3)
    # frozen dataclass：原输入不被修改。
    original = _small_inputs().chain
    replaced = cloud_sigma_scan._chain_with_sigma(original, 0.3)
    assert original.handover.cloud_axial_sigma_mm != pytest.approx(0.3)
    assert replaced is not original


def test_scan_structure_progress_and_sigma_over_waist():
    inputs = _small_inputs()
    messages: list[str] = []
    result = analyze_cloud_sigma_scan(inputs, progress=messages.append)

    assert result.inputs is inputs
    waist_mm = inputs.chain.handover.transport.handover_waist_um * 1e-3
    assert result.waist_mm == pytest.approx(waist_mm)
    assert len(result.points) == 3
    assert [p.sigma_mm for p in result.points] == pytest.approx([0.0, 0.2, 0.4])
    for point in result.points:
        assert point.sigma_over_waist == pytest.approx(point.sigma_mm / waist_mm)
        import math

        sin_angle = math.sin(
            math.radians(inputs.chain.handover.crossing_angle_deg)
        )
        assert point.chi == pytest.approx(
            point.sigma_mm * sin_angle / waist_mm
        )
        assert point.error is None
        assert 0.0 <= point.handover_efficiency <= 1.0
        assert 0.0 < point.final_retention_from_mot <= 1.0
        assert point.handover_temperature_uK > 0.0
        assert point.final_temperature_uK > 0.0
    # 进度逐点计数 "x/N"；无失败点时没有附加的失败汇总消息。
    assert messages == [
        "Rb-87: 云宽扫描 1/3",
        "Rb-87: 云宽扫描 2/3",
        "Rb-87: 云宽扫描 3/3",
    ]


def test_same_seed_reproducible():
    first = analyze_cloud_sigma_scan(_small_inputs())
    second = analyze_cloud_sigma_scan(_small_inputs())
    assert first.points == second.points


def test_point_failure_isolated_and_reported(monkeypatch):
    """单点异常只影响该点：指标 None、error 记录原因，其余点正常。"""
    real = cloud_sigma_scan.simulate_full_chain_point

    def flaky(chain, detuning_ghz, source_power_w, *, trace_points=None):
        if chain.handover.cloud_axial_sigma_mm == pytest.approx(0.2):
            raise RuntimeError("boom")
        return real(
            chain, detuning_ghz, source_power_w, trace_points=trace_points
        )

    monkeypatch.setattr(cloud_sigma_scan, "simulate_full_chain_point", flaky)
    messages: list[str] = []
    result = analyze_cloud_sigma_scan(_small_inputs(), progress=messages.append)

    assert [point.error for point in result.points] == [
        None,
        "RuntimeError: boom",
        None,
    ]
    bad = result.points[1]
    assert bad.sigma_mm == pytest.approx(0.2)
    assert bad.handover_temperature_uK is None
    assert bad.final_temperature_uK is None
    assert bad.handover_efficiency is None
    assert bad.final_retention_from_mot is None
    # 失败原因经 progress 汇总（与二维扫描相同的 Top-3 口径）。
    assert any(
        "1 个网格点无有效结果" in message and "RuntimeError: boom" in message
        for message in messages
    )
