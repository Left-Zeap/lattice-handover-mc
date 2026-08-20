"""offset-waist 双束 conveyor 几何模块与 L1 接入的测试。

公式锚点见 ``reports/offset_waist双束conveyor几何理论框架.md`` §3；
关闭回归保证 ``conveyor_enabled=False`` 时既有行为不变。
"""

import os

# 必须在 import PySide6 之前设置离屏平台（UI 表单测试用）。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import math

import numpy as np
import pytest

from continuous_loading.atomic import RB87
from continuous_loading.conveyor_geometry import (
    beam_radius_um,
    conveyor_point,
    conveyor_profile,
)
from continuous_loading.full_chain import simulate_full_chain_point
from continuous_loading.l1_handover import L1HandoverInputs
from continuous_loading.l1_transport import (
    l1_transport_inputs_for_species,
    simulate_l1_transport,
)
from continuous_loading.l2_transport import L2TransportInputs
from continuous_loading.full_chain import FullChainInputs
from continuous_loading.lattice import evaluate_lattice


_WAVELENGTH_NM = RB87.laser_wavelength_red_of_d1_nm(300.0)
_DISTANCE_M = 0.39


def test_beam_radius_at_focus_and_rayleigh():
    """焦点处半径等于束腰；距焦点一个 Rayleigh 长度处为 w0*sqrt(2)。"""
    waist_um = 250.0
    focus_m = 0.1
    assert (
        beam_radius_um(waist_um, _WAVELENGTH_NM, focus_m, focus_m)
        == pytest.approx(waist_um)
    )
    wavelength_m = _WAVELENGTH_NM * 1e-9
    rayleigh_m = math.pi * (waist_um * 1e-6) ** 2 / wavelength_m
    assert (
        beam_radius_um(waist_um, _WAVELENGTH_NM, focus_m + rayleigh_m, focus_m)
        == pytest.approx(waist_um * math.sqrt(2.0))
    )


def test_zero_separation_matches_evaluate_lattice():
    """sep=0 时 conveyor_point 的阱深必须锚定 evaluate_lattice（1%）。"""
    forward_power_w = 1.0
    retro_power_ratio = 0.6
    position_m = 0.1
    point = conveyor_point(
        RB87,
        _WAVELENGTH_NM,
        forward_power_w,
        waist_um=250.0,
        separation_cm=0.0,
        distance_m=_DISTANCE_M,
        position_m=position_m,
        retro_power_ratio=retro_power_ratio,
    )
    local_waist_um = beam_radius_um(
        250.0, _WAVELENGTH_NM, position_m, _DISTANCE_M / 2.0
    )
    lattice = evaluate_lattice(
        RB87,
        _WAVELENGTH_NM,
        forward_power_w,
        waist_um=local_waist_um,
        retro_power_ratio=retro_power_ratio,
    )
    assert point.depth_uK == pytest.approx(lattice.depth_uK, rel=0.01)


def test_visibility_unity_for_matched_beams_and_drops_when_offset():
    """w1=w2 且 R=1 时 V=1；错腰使 w1≠w2 处 V<1。"""
    matched = conveyor_point(
        RB87,
        _WAVELENGTH_NM,
        1.0,
        waist_um=250.0,
        separation_cm=0.0,
        distance_m=_DISTANCE_M,
        position_m=0.05,
        retro_power_ratio=1.0,
    )
    assert matched.visibility == pytest.approx(1.0)
    offset = conveyor_point(
        RB87,
        _WAVELENGTH_NM,
        1.0,
        waist_um=250.0,
        separation_cm=19.5,
        distance_m=_DISTANCE_M,
        position_m=0.0,
        retro_power_ratio=1.0,
    )
    assert offset.forward_radius_um != pytest.approx(offset.retro_radius_um)
    assert 0.0 < offset.visibility < 1.0


def test_offset_waist_flattens_antinode_depth_profile():
    """Rb、300 GHz、P_f=1 W、L=0.39 m：sep=19.5 的 U_anti 起伏小于 sep=0。"""
    positions = np.linspace(0.0, _DISTANCE_M, 401)

    def _ripple(separation_cm: float) -> float:
        profile = conveyor_profile(
            RB87,
            _WAVELENGTH_NM,
            1.0,
            waist_um=250.0,
            separation_cm=separation_cm,
            distance_m=_DISTANCE_M,
            positions=positions,
            retro_power_ratio=1.0,
        )
        return profile.maximum_depth_uK / profile.minimum_depth_uK

    assert _ripple(19.5) < _ripple(0.0)


def test_disabled_by_default_matches_existing_behavior():
    """默认 conveyor_enabled=False，simulate_l1_transport 正常跑通。"""
    inputs = l1_transport_inputs_for_species("Rb-87")
    assert inputs.conveyor_enabled is False
    trace = simulate_l1_transport(inputs, 300.0, 1.0)
    assert trace.point.depth_uK == pytest.approx(510.0, rel=0.05)
    assert trace.point.start_source_power_w == pytest.approx(
        1.0 * (inputs.start_waist_um / inputs.handover_waist_um) ** 2
    )


def _enabled_inputs() -> object:
    return replace(
        l1_transport_inputs_for_species("Rb-87"),
        conveyor_enabled=True,
        time_points=21,
    )


def test_enabled_l1_single_point():
    """启用后 L1 单点：跑通、末温>0、留存率∈(0,1]、末点阱深锚定剖面。"""
    inputs = _enabled_inputs()
    trace = simulate_l1_transport(inputs, 300.0, 1.0)
    point = trace.point
    assert point.final_temperature_uK > 0.0
    assert 0.0 < point.final_retention_fraction <= 1.0
    # 恒功率策略：起点源端功率等于 handover 源端功率，沿程功率恒定。
    assert point.start_source_power_w == pytest.approx(1.0)
    assert all(power == pytest.approx(1.0) for power in trace.source_power_w)
    end_point = conveyor_point(
        RB87,
        _WAVELENGTH_NM,
        1.0 * inputs.delivery_efficiency,
        inputs.conveyor_waist_um,
        inputs.conveyor_waist_separation_cm,
        inputs.distance_m,
        inputs.distance_m,
        inputs.retro_power_ratio,
    )
    assert point.depth_uK == pytest.approx(end_point.depth_uK)
    assert point.scattering_rate_s == pytest.approx(end_point.scattering_rate_s)


def test_full_chain_smoke_with_conveyor_enabled():
    """小参数全链路在 conveyor_enabled=True 下跑通，三相齐全。"""
    transport = _enabled_inputs()
    handover = L1HandoverInputs(
        transport=transport,
        particle_count=200,
        time_step_us=0.5,
        trace_points=5,
        parallel_backend="serial",
        worker_count=1,
    )
    # 连续相空间路径要求轨迹级 MC 腿；本 smoke 走 (N,T) 约化接口。
    inputs = FullChainInputs(
        handover=handover,
        l2=L2TransportInputs(),
        phase_space_continuity=False,
    )
    simulation = simulate_full_chain_point(inputs, 300.0, 1.0, trace_points=5)
    assert set(simulation.combined_trace.phase) == {
        "L1 transport",
        "handover",
        "L2 transport",
    }
    assert 0.0 < simulation.point.final_retention_from_mot <= 1.0


def test_form_contains_conveyor_group_and_roundtrip():
    """ChainParameterForm 含三个 conveyor 键且 params() 往返一致。"""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ui.widgets.forms import ChainParameterForm

    app = QApplication.instance() or QApplication([])
    form = ChainParameterForm()
    params = form.params()
    for key in (
        "conveyor_enabled",
        "conveyor_waist_um",
        "conveyor_waist_separation_cm",
    ):
        assert key in params
    assert params["conveyor_enabled"] is False
    form.set_params(
        {
            "conveyor_enabled": True,
            "conveyor_waist_um": 300.0,
            "conveyor_waist_separation_cm": 12.5,
        }
    )
    updated = form.params()
    assert updated["conveyor_enabled"] is True
    assert updated["conveyor_waist_um"] == pytest.approx(300.0)
    assert updated["conveyor_waist_separation_cm"] == pytest.approx(12.5)
    form.deleteLater()
    app.processEvents()
