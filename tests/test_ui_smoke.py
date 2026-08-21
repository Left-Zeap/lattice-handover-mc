"""ui 包的冒烟测试。

测试 1 创建主窗口（离屏平台，不需要显示器）；测试 2、3 只走纯 Python
控制层 ``ui.controllers``，不实例化任何 Qt 对象。
"""

import os

# 必须在 import PySide6 之前设置离屏平台。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_has_six_pages(qapp):
    from ui.app import create_main_window

    window = create_main_window()
    assert window.windowTitle() == "连续装载双光晶格计算平台"
    assert window.page_count() == 6
    nav_items = [
        window._nav.item(row).text() for row in range(window._nav.count())
    ]
    assert nav_items == ["概览", "单点计算", "时序可视化", "二维扫描", "云宽扫描", "结果导出"]
    window.close()
    window.deleteLater()


def test_lattice_quick_metrics_paper_depth():
    """Rb 300 GHz / 1 W / 250 µm 的阱深必须复现论文工作点的 ~510 µK。"""
    from ui import controllers

    metrics = controllers.lattice_quick_metrics(
        "Rb-87",
        300.0,
        1.0,
        1.0,
        250.0,
        0.59969536,
    )
    assert metrics["depth_uK"] == pytest.approx(510.0, rel=0.05)


def test_run_single_point_small_form():
    """小参数全链路：三相拼接齐全，MOT 总留存在物理合理区间。"""
    from ui import controllers

    simulation = controllers.run_single_point(_small_form_params())
    assert set(simulation.combined_trace.phase) == {
        "L1 transport",
        "handover",
        "L2 transport",
    }
    assert 0.0 < simulation.point.final_retention_from_mot <= 0.4


def _small_form_params() -> dict:
    from ui import controllers

    params = controllers.default_form_params("Rb-87")
    params.update(
        {
            "particle_count": 200,
            "time_step_us": 0.5,
            "l1_time_points": 21,
            "trace_points": 5,
            "parallel_backend": "serial",
            "worker_count": 1,
        }
    )
    return params


def test_legacy_l1_waist_dict_interface_still_builds():
    """旧前端只传起点/交接半径时继续使用线性兼容剖面。"""
    from ui import controllers

    params = controllers.default_form_params("Rb-87")
    for key in (
        "l1_start_beam_diameter_um",
        "l1_minimum_waist_um",
        "l1_minimum_waist_position_m",
    ):
        params.pop(key)
    params["l1_start_waist_um"] = 330.0
    params["handover_waist_um"] = 250.0

    transport = controllers.build_full_chain_inputs(params).handover.transport
    assert not transport.calibrated_gaussian_geometry
    assert transport.start_waist_um == pytest.approx(330.0)
    assert transport.handover_waist_um == pytest.approx(250.0)


def test_timeline_build_and_sample():
    """时间轴单调、三相齐全、末端到科学区，采样插值与夹取正确。"""
    import numpy as np

    from ui import controllers
    from ui.timeline import build_timeline, sample_timeline

    simulation = controllers.run_single_point(_small_form_params())
    timeline = build_timeline(simulation)

    time_ms = timeline["time_ms"]
    assert np.all(np.diff(time_ms) >= 0.0)
    assert set(timeline["phase"]) == {"L1 transport", "handover", "L2 transport"}
    # L1 0.39 m + L2 0.17 m，时间轴末端必须在科学区。
    assert timeline["position_m"][-1] == pytest.approx(0.56)

    handover_mid_ms = 0.5 * (
        timeline["handover_start_ms"] + timeline["handover_end_ms"]
    )
    assert sample_timeline(timeline, handover_mid_ms)["phase"] == "handover"

    # 越界采样夹到端点。
    early = sample_timeline(timeline, -1.0)
    assert early["time_ms"] == pytest.approx(time_ms[0])
    assert early["phase"] == "L1 transport"
    late = sample_timeline(timeline, 1e6)
    assert late["time_ms"] == pytest.approx(time_ms[-1])
    assert late["phase"] == "L2 transport"


def _fake_scan_result():
    from types import SimpleNamespace

    # 矩阵按 [功率行][失谐列] 排列；(1.5 W, 200 GHz) 点交接率为 None，
    # 用于验证 None 一律视为不符合。
    return SimpleNamespace(
        detuning_ghz=(100.0, 200.0),
        source_power_w=(1.0, 1.5),
        handover_transfer_efficiency=((0.9, 0.8), (0.7, None)),
        science_total_temperature_rise_uK=((10.0, 50.0), (20.0, 30.0)),
        final_retention_from_mot=((0.5, 0.2), (0.4, 0.1)),
        science_peak_density_m3=((1e18, 2e18), (3e18, 4e18)),
    )


_BASE_CONDITIONS = {
    "power_enabled": False,
    "power_max_w": 0.0,
    "retention_enabled": False,
    "retention_min": 0.0,
    "heating_enabled": False,
    "heating_max_uK": 0.0,
    "mode": "and",
    "expression": "",
}


def test_scan_condition_mask():
    from ui import controllers

    result = _fake_scan_result()

    # 未勾选任何条件：全部有效点符合，None 点不符合。
    mask = controllers.scan_condition_mask(result, dict(_BASE_CONDITIONS))
    assert mask.tolist() == [[True, True], [True, False]]

    # AND：P<=1.2 且 ret>=0.3，只剩左上角一点。
    mask = controllers.scan_condition_mask(
        result,
        {
            **_BASE_CONDITIONS,
            "power_enabled": True,
            "power_max_w": 1.2,
            "retention_enabled": True,
            "retention_min": 0.3,
        },
    )
    assert mask.tolist() == [[True, False], [False, False]]

    # OR：P<=1.2 或 heat<=15。
    mask = controllers.scan_condition_mask(
        result,
        {
            **_BASE_CONDITIONS,
            "mode": "or",
            "power_enabled": True,
            "power_max_w": 1.2,
            "heating_enabled": True,
            "heating_max_uK": 15.0,
        },
    )
    assert mask.tolist() == [[True, True], [False, False]]

    # 表达式模式优先于勾选条件（勾选项与之冲突时以表达式为准）。
    mask = controllers.scan_condition_mask(
        result,
        {
            **_BASE_CONDITIONS,
            "power_enabled": True,
            "power_max_w": 0.1,
            "expression": "(P<=1.2)&(ret>=0.2)",
        },
    )
    assert mask.tolist() == [[True, True], [False, False]]

    # 表达式中 eff 为 None 的点必须视为不符合。
    mask = controllers.scan_condition_mask(
        result, {**_BASE_CONDITIONS, "expression": "eff>=0.75"}
    )
    assert mask.tolist() == [[True, True], [False, False]]

    # 非法表达式：语法错误、未授权节点、未知变量都必须抛 ValueError。
    for bad in ("import os", "__import__('os')", "foo>=1", "P<=1.2 & ret"):
        with pytest.raises(ValueError):
            controllers.scan_condition_mask(
                result, {**_BASE_CONDITIONS, "expression": bad}
            )


def test_no_wheel_spinbox_ignores_wheel(qapp):
    """滚轮事件必须被 ignore（数值不变，事件未接受，可传播给滚动区）。"""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    from ui.widgets.forms import ChainParameterForm, NoWheelDoubleSpinBox

    form = ChainParameterForm()
    spin = form._widgets["detuning_ghz"]
    assert isinstance(spin, NoWheelDoubleSpinBox)
    before = spin.value()
    event = QWheelEvent(
        QPointF(5.0, 5.0),
        QPointF(5.0, 5.0),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    spin.wheelEvent(event)
    assert spin.value() == before
    assert not event.isAccepted()


def test_phase_space_and_waveform_form_logic_matches_compute_modes(qapp):
    """高保真模式锁定 MC；扫描禁用；实测文件接管理想时序字段。"""
    from ui import controllers
    from ui.widgets.forms import ChainParameterForm

    single = ChainParameterForm()
    assert "target_depth_uK" not in single._widgets
    assert "target_depth_uK" not in single.params()
    assert "handover_waist_um" not in single._widgets
    assert "l1_start_waist_um" not in single._widgets
    assert single._specs["l1_start_beam_diameter_um"].label == (
        "L1 起点光束直径 2w (µm)"
    )
    assert single._specs["l1_minimum_waist_um"].label == (
        "L1 最小束腰半径 w₀ (µm)"
    )
    assert single._specs["source_power_w"].label == (
        "L1/L2 固定源端功率/分支 (W)"
    )
    built = controllers.build_full_chain_inputs(single.params())
    assert built.handover.transport.target_depth_uK == pytest.approx(
        controllers.default_form_params()["target_depth_uK"]
    )
    geometry = built.handover.transport
    assert geometry.start_beam_diameter_um == pytest.approx(660.0)
    assert geometry.minimum_waist_um == pytest.approx(250.0)
    assert geometry.minimum_waist_position_m == pytest.approx(0.2)
    assert geometry.handover_waist_um == pytest.approx(323.0727472)
    continuity = single._widgets["phase_space_continuity"]
    continuity.setChecked(True)
    assert single._widgets["transport_method"].currentData() == "monte_carlo"
    assert single._widgets["l1_kinematic_profile"].currentData() == "minimum_jerk"
    assert single._widgets["l2_kinematic_profile"].currentData() == "minimum_jerk"
    assert not single._widgets["transport_method"].isEnabled()
    assert not single._widgets["l1_kinematic_profile"].isEnabled()
    assert not single._widgets["l2_kinematic_profile"].isEnabled()

    # LGM 装载模块已移除：表单不再有任何 loading_* 控件。
    assert not any(key.startswith("loading") for key in single._widgets)

    single._widgets["l1_control_waveform_path"].setText("measured_l1.csv")
    assert not single._widgets["l1_acceleration_m_s2"].isEnabled()
    assert not single._widgets["l1_maximum_velocity_m_s"].isEnabled()
    assert not single._widgets["conveyor_enabled"].isEnabled()
    single._widgets["handover_control_waveform_path"].setText("handover.csv")
    assert not single._widgets["duration_us"].isEnabled()

    scan = ChainParameterForm(scan_preset=True)
    scan_continuity = scan._widgets["phase_space_continuity"]
    assert not scan_continuity.isChecked()
    assert scan_continuity.isEnabled()
    single.deleteLater()
    scan.deleteLater()


def test_cs_initial_temperature_is_wired_and_continuous_mode_is_smooth():
    """表单初温进入 L1 输入（默认 20 µK）；连续相空间强制平滑加速。"""
    from ui.controllers import build_full_chain_inputs, default_form_params

    # 静止 L1 晶格热平衡初态的默认温度。
    assert default_form_params("Rb-87")["initial_temperature_uK"] == pytest.approx(20.0)

    params = default_form_params("Cs-133")
    params.update(
        initial_temperature_uK=30.0,
        transport_method="monte_carlo",
        phase_space_continuity=True,
        l1_kinematic_profile="trapezoid",
        l2_kinematic_profile="trapezoid",
    )
    inputs = build_full_chain_inputs(params)
    transport = inputs.handover.transport
    assert transport.initial_temperature_uK == pytest.approx(30.0)
    assert transport.kinematic_profile == "minimum_jerk"
    assert inputs.l2.kinematic_profile == "minimum_jerk"


def test_handover_phase_mode_wiring(qapp):
    """交接相位口径控件：随机为默认（相位值禁用）；固定相位换算成弧度
    并透传到 L1HandoverInputs（单点与扫描共用 build_full_chain_inputs）。"""
    import math

    from ui.controllers import build_full_chain_inputs, default_form_params
    from ui.widgets.forms import ChainParameterForm

    form = ChainParameterForm()
    # 默认随机口径，相位值输入框禁用。
    assert form._widgets["phase_mode"].currentData() == "random"
    assert not form._widgets["relative_phase_deg"].isEnabled()
    # 切到固定口径后解锁。
    fixed_index = form._widgets["phase_mode"].findData("fixed")
    form._widgets["phase_mode"].setCurrentIndex(fixed_index)
    assert form._widgets["relative_phase_deg"].isEnabled()
    form.deleteLater()

    params = default_form_params("Rb-87")
    assert params["phase_mode"] == "random"
    inputs = build_full_chain_inputs(params)
    assert inputs.handover.randomize_relative_phase is True
    assert inputs.handover.relative_phase_rad == pytest.approx(0.0)

    params.update(phase_mode="fixed", relative_phase_deg=45.0)
    inputs = build_full_chain_inputs(params)
    assert inputs.handover.randomize_relative_phase is False
    assert inputs.handover.relative_phase_rad == pytest.approx(
        math.radians(45.0)
    )


def test_build_full_series_covers_l1_and_l2():
    """全程序列覆盖到 L2 末端；handover 段 NaN；L2 从 L1 交接半径压缩至 150 µm。"""
    import numpy as np

    from ui import controllers
    from ui.timeline import build_full_series, build_timeline

    simulation = controllers.run_single_point(_small_form_params())
    series = build_full_series(simulation)
    timeline = build_timeline(simulation)

    time_ms = series["time_ms"]
    assert time_ms[-1] == pytest.approx(timeline["l2_end_ms"])
    phase = np.asarray(series["phase"])
    handover = phase == "handover"
    l2 = phase == "L2 transport"
    assert handover.any() and l2.any()
    # handover 段运动学/光路量无定义，必须全是 NaN。
    for key in (
        "velocity_m_s",
        "acceleration_m_s2",
        "aom_frequency_difference_mhz",
        "waist_um",
        "source_power_w",
    ):
        assert np.all(np.isnan(series[key][handover]))
        assert np.all(np.isfinite(series[key][l2]))
    # L2 从 L1 自动计算的 handover 半径降到末端 150 µm。
    l1_handover_radius = (
        simulation.l1_handover_simulation.transport_trace.waist_um[-1]
    )
    assert series["waist_um"][l2][0] == pytest.approx(l1_handover_radius)
    assert series["waist_um"][l2][-1] == pytest.approx(150.0)
    assert np.allclose(series["beam_diameter_um"], 2.0 * series["waist_um"], equal_nan=True)


def test_export_live_figure_lookup(qapp):
    """当前显示结果返回功能页注册的 figure，其他历史条目返回 None。"""
    from matplotlib.figure import Figure

    from ui.pages.export_page import ExportPage
    from ui.state import AppState

    state = AppState()
    export_page = ExportPage(state)
    live_figure = Figure()
    state.register_figure("single_point", live_figure)

    current_payload = object()
    state.set_single_point(current_payload)
    state.add_history(
        "单点",
        "当前结果",
        "完成",
        payload=current_payload,
        elapsed_seconds=1.25,
    )
    state.add_history("单点", "更早的历史", "完成", payload=object())

    current_entry, older_entry = state.history
    assert export_page._live_figure_for(current_entry) is live_figure
    assert export_page._live_figure_for(older_entry) is None
    export_page.refresh()
    assert export_page.table.item(1, 4).text() == "1.250 s"
    export_page.deleteLater()


def test_failed_scan_still_registers_a_diagnostic_figure(qapp):
    from ui.controllers import default_form_params
    from ui.pages.scan_page import ScanPage
    from ui.state import AppState

    state = AppState()
    page = ScanPage(state)
    params = default_form_params("Cs-133", scan_preset=True)
    page._show_failed_plot(
        params,
        "所有动态可行点均为零捕获",
    )
    figure = state.figure_for("scan")
    assert figure is page.heatmap_canvas.figure
    assert len(figure.axes) == 4
    assert all(axis.texts for axis in figure.axes)
    assert "无可绘有效点" in figure.axes[0].texts[0].get_text()
    page.deleteLater()


def test_calc_worker_records_elapsed_time_on_success(qapp):
    from ui.workers import CalcWorker

    values = []
    worker = CalcWorker(lambda progress: 42)
    worker.finished.connect(values.append)
    worker.run()
    assert values == [42]
    assert worker.elapsed_seconds is not None
    assert worker.elapsed_seconds >= 0.0
    worker.deleteLater()


def test_export_files_include_runtime_metadata(qapp, tmp_path):
    import csv
    import json

    from continuous_loading.full_chain import FullChainInputs, FullChainScanResult
    from ui.pages.export_page import ExportPage
    from ui.state import AppState

    result = FullChainScanResult(
        inputs=FullChainInputs(),
        detuning_ghz=(300.0,),
        source_power_w=(1.0,),
        transport_feasible=((True,),),
        handover_transfer_efficiency=((0.9,),),
        science_final_temperature_uK=((20.0,),),
        science_total_temperature_rise_uK=((2.0,),),
        final_retention_from_mot=((0.3,),),
        science_peak_density_m3=((1e16,),),
        evaluated_points=1,
        optimal=None,
        comparison=None,
        optimal_simulation=None,
        comparison_simulation=None,
    )
    page = ExportPage(AppState())
    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "result.csv"
    page._export_json(result, json_path, 2.5)
    page._export_csv(result, csv_path, 2.5)
    assert json.loads(json_path.read_text(encoding="utf-8"))["runtime_seconds"] == 2.5
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["runtime_seconds", "2.5"]
    assert rows[1] == ["runtime_display", "2.500 s"]
    page.deleteLater()


def test_cloud_sigma_page_constructs(qapp):
    """云宽扫描页可构造；控制组默认值与参数收集键符合 controllers 约定。"""
    from ui.pages.cloud_sigma_page import CloudSigmaPage
    from ui.state import AppState

    page = CloudSigmaPage(AppState())
    assert page.form is not None
    assert page.sigma_min_spin.value() == 0.0
    assert page.sigma_max_spin.value() == 5.0
    assert page.points_spin.value() == 10
    page.sigma_min_spin.setValue(0.1)
    page.sigma_max_spin.setValue(1.5)
    page.points_spin.setValue(7)
    params = page._gather_params()
    assert params["cloud_sigma_min_mm"] == pytest.approx(0.1)
    assert params["cloud_sigma_max_mm"] == pytest.approx(1.5)
    assert params["cloud_sigma_points"] == 7
    # 共享表单键齐全（工作点与计算设备由 ChainParameterForm 提供）。
    assert "detuning_ghz" in params and "compute_backend" in params
    page.deleteLater()


def test_cloud_sigma_controllers_wiring(monkeypatch):
    """run_cloud_sigma_scan 把表单键正确组装进 CloudSigmaScanInputs。"""
    from continuous_loading.cloud_sigma_scan import CloudSigmaScanInputs
    from ui import controllers

    params = controllers.default_form_params("Rb-87")
    params.update(
        detuning_ghz=250.0,
        source_power_w=1.2,
        cloud_sigma_min_mm=0.0,
        cloud_sigma_max_mm=1.5,
        cloud_sigma_points=7,
    )
    captured = {}

    def fake_analyze(inputs, *, progress=None):
        captured["inputs"] = inputs
        return "sentinel"

    monkeypatch.setattr(controllers, "analyze_cloud_sigma_scan", fake_analyze)
    assert controllers.run_cloud_sigma_scan(params) == "sentinel"
    inputs = captured["inputs"]
    assert isinstance(inputs, CloudSigmaScanInputs)
    assert inputs.detuning_ghz == pytest.approx(250.0)
    assert inputs.source_power_w == pytest.approx(1.2)
    assert inputs.sigma_min_mm == pytest.approx(0.0)
    assert inputs.sigma_max_mm == pytest.approx(1.5)
    assert inputs.points == 7
    assert inputs.chain.handover.transport.atom_label == "Rb-87"
    # 历史摘要包含云宽范围与取点数。
    summary = controllers.summarize_cloud_sigma_params(params)
    assert "云宽 0-1.5 mm 7 点" in summary
