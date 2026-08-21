"""运输腿轨迹级 Monte Carlo（transport_mc.py）的物理与数值回归测试。

运行时间控制：全部用 ``replace`` 构造小距离 L1 输入（5 mm、
4000 m/s²、4 m/s，总时长约 2.25 ms、4500 步），Monte Carlo 数值
参数经 ``L1TransportInputs`` 的 ``mc_*`` 字段传入（200–500），串行。
"""

import math
import os
from dataclasses import replace

# 必须在 import PySide6 之前设置离屏平台。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from continuous_loading import transport_mc
from continuous_loading.atomic import RB87
from continuous_loading.conveyor_geometry import conveyor_point
from continuous_loading.dipole import scalar_potential_and_scattering
from continuous_loading.full_chain import (
    FullChainInputs,
    analyze_full_chain_scan,
    simulate_full_chain_point,
)
from continuous_loading.l1_handover import L1HandoverInputs, analyze_l1_handover_scan
from continuous_loading.l1_transport import (
    L1TransportInputs,
    L1TransportTrace,
    L1_TRANSPORT_CONFIGURATION,
    l1_transport_inputs_for_species,
    simulate_l1_transport,
)
from continuous_loading.l2_transport import L2TransportInputs
from continuous_loading.lattice import evaluate_lattice


@pytest.fixture(scope="session")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _mc_overrides(
    *,
    particle_count=300,
    include_scattering=True,
    seed=20250902,
    cloud_axial_sigma_mm=0.5,
):
    """运输 MC 数值参数的 inputs 覆盖字典（与 handover 合并调用）。"""
    return dict(
        mc_particle_count=particle_count,
        mc_include_scattering=include_scattering,
        mc_seed=seed,
        mc_cloud_axial_sigma_mm=cloud_axial_sigma_mm,
    )


def _small_leg_inputs(**overrides):
    """小距离运输腿：总时长约 2.25 ms，0.5 µs 步长下 4500 步。"""
    base = dict(
        distance_m=0.005,
        acceleration_m_s2=4000.0,
        maximum_velocity_m_s=4.0,
        time_points=21,
        transport_method="monte_carlo",
        transport_time_step_us=0.5,
        # 小距离数值核验沿用旧端点剖面；论文尺度的标定高斯几何另测。
        minimum_waist_um=None,
        minimum_waist_position_m=None,
        **_mc_overrides(),
    )
    base.update(overrides)
    return replace(l1_transport_inputs_for_species("Rb-87"), **base)


def test_default_configuration_keeps_monte_carlo_disabled():
    """默认关闭：transport_method 为 analytic，解析腿无标准误字段值。"""
    group = L1_TRANSPORT_CONFIGURATION["transport_monte_carlo"]
    assert bool(group["enabled"]) is False
    assert float(group["time_step_us"]) == 0.5
    inputs = L1TransportInputs()
    assert inputs.transport_method == "analytic"
    assert inputs.transport_time_step_us == 0.5
    trace = simulate_l1_transport(
        replace(inputs, time_points=21),
        300.0,
        1.0,
    )
    assert trace.retention_standard_error is None


def test_transport_method_and_time_step_validation():
    """非法运输方法或非正步长必须在输入校验阶段抛错。"""
    with pytest.raises(ValueError):
        replace(L1TransportInputs(), transport_method="bogus")
    for bad_step in (0.0, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            replace(L1TransportInputs(), transport_time_step_us=bad_step)


def test_double_beam_force_matches_potential_gradient():
    """双束解析梯度力必须与势的数值梯度一致（含错腰与非零相位）。"""
    wavelength_nm = RB87.laser_wavelength_red_of_d1_nm(300.0)
    c_u = abs(scalar_potential_and_scattering(RB87, wavelength_nm, 1.0).potential_j)
    kwargs = dict(
        intensity1_w_m2=3.2e7,
        intensity2_w_m2=1.9e7,
        waist1_m=260e-6,
        waist2_m=310e-6,
        wave_number_m=2.0 * math.pi / (wavelength_nm * 1e-9),
        lattice_position_m=0.013,
        phase_rad=0.7,
        potential_per_intensity_j=c_u,
    )
    rng = np.random.default_rng(1234)
    positions = np.empty((64, 3))
    positions[:, :2] = rng.normal(scale=50e-6, size=(64, 2))
    # 轴向采样覆盖格点附近的节点与波腹。
    positions[:, 2] = 0.013 + rng.normal(scale=150e-9, size=64)
    _, force, _, _ = transport_mc._double_beam_potential_and_force(
        positions,
        **kwargs,
    )

    def _numerical_force(axis: int, h: float) -> np.ndarray:
        plus = positions.copy()
        plus[:, axis] += h
        minus = positions.copy()
        minus[:, axis] -= h
        v_plus = transport_mc._double_beam_potential_and_force(plus, **kwargs)[0]
        v_minus = transport_mc._double_beam_potential_and_force(minus, **kwargs)[0]
        return -(v_plus - v_minus) / (2.0 * h)

    numerical = np.empty_like(force)
    numerical[:, 0] = _numerical_force(0, 1e-8)
    numerical[:, 1] = _numerical_force(1, 1e-8)
    numerical[:, 2] = _numerical_force(2, 1e-9)
    relative_error = np.max(np.abs(force - numerical)) / np.max(np.abs(force))
    assert relative_error < 5e-5


def test_non_conveyor_fixed_power_antinode_follows_gaussian_waist():
    """关闭 conveyor 时源功率恒定，波腹强度按 1/w² 沿程变化。"""
    inputs = _small_leg_inputs()
    detuning_ghz = 300.0
    wavelength_nm = RB87.laser_wavelength_red_of_d1_nm(detuning_ghz)
    source_power_w = 1.0
    profile = transport_mc._leg_optics_profile(
        inputs,
        wavelength_nm,
        source_power_w,
    )
    c_u = abs(scalar_potential_and_scattering(RB87, wavelength_nm, 1.0).potential_j)
    antinode_j = (
        c_u
        * float(profile.intensity1_w_m2[0])
        * (1.0 + math.sqrt(inputs.retro_power_ratio)) ** 2
    )
    lattice = evaluate_lattice(
        RB87,
        wavelength_nm,
        forward_power_w=source_power_w * inputs.delivery_efficiency,
        waist_um=inputs.start_waist_um,
        retro_power_ratio=inputs.retro_power_ratio,
    )
    assert math.isclose(
        antinode_j,
        abs(lattice.dipole.potential_j),
        rel_tol=1e-9,
    )
    assert np.allclose(profile.source_power_w, source_power_w, rtol=1e-12)
    assert profile.intensity1_w_m2[-1] / profile.intensity1_w_m2[0] == pytest.approx(
        (inputs.start_waist_um / inputs.handover_waist_um) ** 2,
        rel=1e-12,
    )


def test_static_deep_well_retains_all_and_temperature():
    """近似静态深阱（0.2 mm、无散射、1 mK 级阱深）：无人逃逸，温度不变。

    步长取 0.25 µs，使 2 W 阱深处 ω_z·Δt≈0.87 位于步长收敛区内
    （理论文档 §3.3 认可 ω_z·Δt≈1.26 以下）。
    """
    inputs = _small_leg_inputs(
        distance_m=0.0002,
        acceleration_m_s2=100.0,
        maximum_velocity_m_s=0.14,
        start_waist_um=250.0,
        handover_waist_um=250.0,
        transport_time_step_us=0.25,
        **_mc_overrides(particle_count=300, include_scattering=False),
    )
    trace = simulate_l1_transport(inputs, 300.0, 2.0)

    assert trace.point.final_retention_fraction == 1.0
    ratio = trace.point.final_temperature_uK / trace.temperature_uK[0]
    assert 0.8 <= ratio <= 1.25
    assert math.isfinite(trace.point.final_temperature_uK)
    assert trace.retention_standard_error is not None


def test_seed_reproducibility():
    """同参数同种子两次运行结果逐等。"""
    inputs = _small_leg_inputs()
    first = simulate_l1_transport(inputs, 300.0, 1.0)
    second = simulate_l1_transport(inputs, 300.0, 1.0)

    assert first.point.final_temperature_uK == second.point.final_temperature_uK
    assert first.point.final_retention_fraction == (
        second.point.final_retention_fraction
    )
    assert first.point.cumulative_scattering_events == (
        second.point.cumulative_scattering_events
    )


def test_shallow_well_evaporative_selection():
    """浅阱（0.15 W、阱深 ~77 µK）：留存<1 且幸存者动能温度低于初始。"""
    inputs = _small_leg_inputs(
        start_waist_um=250.0,
        handover_waist_um=250.0,
        **_mc_overrides(particle_count=500, include_scattering=False),
    )
    trace = simulate_l1_transport(inputs, 300.0, 0.15)

    assert trace.point.final_retention_fraction < 1.0
    assert trace.point.final_retention_fraction > 0.3
    assert trace.point.final_temperature_uK < trace.temperature_uK[0]


def test_monte_carlo_matches_analytic_magnitude():
    """Rb 300 GHz/1 W 小距离版：MC 末温与解析末温同一量级。"""
    mc_inputs = _small_leg_inputs()
    analytic_inputs = replace(mc_inputs, transport_method="analytic")
    mc_trace = simulate_l1_transport(mc_inputs, 300.0, 1.0)
    analytic_trace = simulate_l1_transport(analytic_inputs, 300.0, 1.0)

    ratio = (
        mc_trace.point.final_temperature_uK
        / analytic_trace.point.final_temperature_uK
    )
    assert 0.5 <= ratio <= 2.0
    assert 0.0 < mc_trace.point.final_retention_fraction <= 1.0
    assert mc_trace.point.cumulative_scattering_events > 0.0


def test_time_step_halving_converges():
    """步长减半（0.5 vs 0.25 µs）末温相对差小于 15%（无散射）。"""
    coarse = simulate_l1_transport(
        _small_leg_inputs(
            transport_time_step_us=0.5,
            **_mc_overrides(include_scattering=False),
        ),
        300.0,
        1.0,
    )
    fine = simulate_l1_transport(
        _small_leg_inputs(
            transport_time_step_us=0.25,
            **_mc_overrides(include_scattering=False),
        ),
        300.0,
        1.0,
    )

    relative_difference = abs(
        fine.point.final_temperature_uK - coarse.point.final_temperature_uK
    ) / coarse.point.final_temperature_uK
    assert relative_difference < 0.15


def test_dispatcher_returns_same_type_with_standard_error():
    """transport_method='monte_carlo' 时 simulate_l1_transport 返回同型
    L1TransportTrace，且留存率带 Jeffreys 标准误。"""
    trace = simulate_l1_transport(
        _small_leg_inputs(**_mc_overrides(particle_count=200)),
        300.0,
        1.0,
    )

    assert isinstance(trace, L1TransportTrace)
    assert trace.retention_standard_error is not None
    assert trace.retention_standard_error >= 0.0
    assert len(trace.time_ms) == len(trace.retention_fraction)
    assert trace.stage[0] == "acceleration"
    assert trace.stage[-1] == "arrived"


def test_conveyor_enabled_leg_smoke():
    """conveyor 开启时 MC 腿走同一套双束力，端点阱深与几何一致。"""
    inputs = _small_leg_inputs(
        conveyor_enabled=True,
        **_mc_overrides(particle_count=200, include_scattering=False),
    )
    detuning_ghz = 300.0
    trace = simulate_l1_transport(inputs, detuning_ghz, 1.0)

    assert 0.0 < trace.point.final_retention_fraction <= 1.0
    assert math.isfinite(trace.point.final_temperature_uK)
    wavelength_nm = RB87.laser_wavelength_red_of_d1_nm(detuning_ghz)
    expected = conveyor_point(
        RB87,
        wavelength_nm,
        1.0 * inputs.delivery_efficiency,
        inputs.conveyor_waist_um,
        inputs.conveyor_waist_separation_cm,
        inputs.distance_m,
        inputs.distance_m,
        inputs.retro_power_ratio,
    )
    assert math.isclose(
        trace.point.depth_uK,
        expected.depth_uK,
        rel_tol=1e-12,
    )
    # conveyor 模式下源端功率全程恒定。
    assert np.allclose(trace.source_power_w, 1.0)


def test_full_chain_smoke_monte_carlo():
    """MC 模式全链路冒烟：小距离 L1/L2 + handover 200 轨迹，三相齐全。"""
    transport = _small_leg_inputs(**_mc_overrides(particle_count=200))
    handover = L1HandoverInputs(
        transport=transport,
        particle_count=200,
        time_step_us=0.5,
        trace_points=5,
        parallel_backend="serial",
        worker_count=1,
    )
    l2 = L2TransportInputs(
        distance_m=0.003,
        acceleration_m_s2=3000.0,
        maximum_velocity_m_s=2.9,
        time_points=21,
    )
    simulation = simulate_full_chain_point(
        FullChainInputs(
            handover=handover, l2=l2, phase_space_continuity=False
        ),
        300.0,
        1.0,
        trace_points=5,
    )

    assert set(simulation.combined_trace.phase) == {
        "L1 transport",
        "handover",
        "L2 transport",
    }
    assert simulation.l2_result is not None
    assert math.isfinite(simulation.l2_result.final_temperature_uK)
    assert 0.0 <= simulation.point.final_retention_from_mot <= 1.0


def test_ui_form_transport_mc_fields_roundtrip(qapp):
    """表单"运输动力学"与"运输 MC 步长"往返：默认值、修改与参数传递。"""
    from ui import controllers
    from ui.widgets.forms import ChainParameterForm

    form = ChainParameterForm()
    params = form.params()
    assert params["transport_method"] == "analytic"
    assert params["transport_time_step_us"] == 0.5

    method_combo = form._widgets["transport_method"]
    method_combo.setCurrentIndex(method_combo.findData("monte_carlo"))
    form._widgets["transport_time_step_us"].setValue(0.25)
    updated = form.params()
    assert updated["transport_method"] == "monte_carlo"
    assert updated["transport_time_step_us"] == 0.25

    inputs = controllers.build_full_chain_inputs(updated)
    assert inputs.handover.transport.transport_method == "monte_carlo"
    assert inputs.handover.transport.transport_time_step_us == 0.25
    form.close()
    form.deleteLater()


def test_leg_honors_inputs_mc_parameters_over_config():
    """运输 MC 数值参数必须来自 inputs 而非全局配置（参数合并修复）。

    inputs 的 mc_particle_count=80 时，Jeffreys 标准误必须按 N=80
    计算（若误读全局配置的 1000，标准误会对不上）；不同 mc_seed
    必须给出不同结果。
    """
    inputs = _small_leg_inputs(**_mc_overrides(particle_count=80, seed=7))
    trace = simulate_l1_transport(inputs, 300.0, 1.0)

    n = 80
    survivors = round(trace.point.final_retention_fraction * n)
    alpha = survivors + 0.5
    beta = n - survivors + 0.5
    expected_se = math.sqrt(alpha * beta / ((n + 1.0) ** 2 * (n + 2.0)))
    assert math.isclose(
        trace.retention_standard_error, expected_se, rel_tol=1e-9
    )
    other = simulate_l1_transport(
        _small_leg_inputs(**_mc_overrides(particle_count=80, seed=8)),
        300.0,
        1.0,
    )
    assert trace.point.final_temperature_uK != other.point.final_temperature_uK


def test_unsamplable_point_returns_zero_retention():
    """浅阱到几乎无束缚初态的点（0.01 W ≈ 5 µK ≪ 20 µK 初温）：
    返回零留存 trace（温度 NaN）而不是抛错中断扫描。"""
    trace = simulate_l1_transport(_small_leg_inputs(), 300.0, 0.01)

    assert trace.point.final_retention_fraction == 0.0
    assert math.isnan(trace.point.final_temperature_uK)
    assert trace.point.final_atom_number == 0.0
    assert trace.retention_standard_error is not None


def test_scan_starts_with_mc_transport_and_isolates_failures(monkeypatch):
    """二维扫描在 MC 运输下能正常启动完成（原 bug 的回归）。

    网格含一个无法采样的 0.01 W 浅阱点：该点必须被隔离为无效点
    （NaN），不拖垮整个扫描；可行性预检不得调用 MC 腿（否则每点
    预检就是一次全距离轨迹模拟，扫描看似无法启动）。
    """
    from types import SimpleNamespace

    import continuous_loading.l1_handover as integrated

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
        integrated, "run_handover_monte_carlo", fake_handover
    )
    leg_calls = []
    original_leg = transport_mc.simulate_leg_monte_carlo

    def counting_leg(inputs, detuning_ghz, handover_source_power_w):
        leg_calls.append((detuning_ghz, handover_source_power_w))
        return original_leg(inputs, detuning_ghz, handover_source_power_w)

    monkeypatch.setattr(
        transport_mc, "simulate_leg_monte_carlo", counting_leg
    )

    transport = _small_leg_inputs(
        detuning_min_ghz=250.0,
        detuning_max_ghz=350.0,
        detuning_points=2,
        handover_source_power_min_w=0.01,
        handover_source_power_max_w=1.0,
        power_points=2,
        **_mc_overrides(particle_count=100),
    )
    handover = L1HandoverInputs(
        transport=transport,
        particle_count=50,
        time_step_us=1.0,
        trace_points=3,
        parallel_backend="serial",
        worker_count=1,
    )
    l2 = L2TransportInputs(
        distance_m=0.003,
        acceleration_m_s2=3000.0,
        maximum_velocity_m_s=2.9,
        time_points=21,
    )

    result = analyze_full_chain_scan(
        FullChainInputs(
            handover=handover, l2=l2, phase_space_continuity=False
        )
    )

    # 扫描正常完成：两个 1.0 W 好点有效，两个 0.01 W 浅阱点为 NaN。
    assert result.evaluated_points == 2
    heating = np.asarray(result.science_total_temperature_rise_uK, dtype=float)
    assert np.isnan(heating[0]).all()  # 0.01 W 行
    assert np.isfinite(heating[1]).all()  # 1.0 W 行
    assert result.optimal.science_atom_number > 0.0
    # MC 腿只被真实任务调用：4 个 L1 腿（网格任务）+ 2 个（l1_handover
    # 最优/较差重跑）+ 2 个 L2 腿（好点）+ 4 个（full_chain 最优/较差
    # 重跑各 2 腿）；可行性预检不得调用 MC 腿。
    l1_leg_calls = [c for c in leg_calls]
    assert len(l1_leg_calls) == 4 + 2 + 2 + 4


def test_point_grid_carries_l1_joint_results(monkeypatch):
    """point_grid 携带逐点完整联合结果，供全链路复用（不再重算）。"""
    from types import SimpleNamespace

    import continuous_loading.l1_handover as integrated

    def fake_handover(parameters):
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

    monkeypatch.setattr(
        integrated, "run_handover_monte_carlo", fake_handover
    )
    transport = _small_leg_inputs(
        detuning_min_ghz=250.0,
        detuning_max_ghz=350.0,
        detuning_points=2,
        handover_source_power_min_w=0.5,
        handover_source_power_max_w=1.0,
        power_points=2,
        **_mc_overrides(particle_count=100),
    )
    handover = L1HandoverInputs(
        transport=transport,
        particle_count=50,
        time_step_us=1.0,
        trace_points=3,
        parallel_backend="serial",
        worker_count=1,
    )
    result = analyze_l1_handover_scan(handover)

    assert result.point_grid is not None
    assert len(result.point_grid) == 2
    entry = result.point_grid[1][0]
    assert entry is not None
    # point_grid 中的 L1 末态必须是 MC 腿的结果（带留存标准误语义：
    # 解析腿 final_retention 与 MC 不同，这里验证温度是有限正值）。
    assert entry.transport.final_temperature_uK > 0.0
    assert entry.handover_transfer_efficiency == 0.9
