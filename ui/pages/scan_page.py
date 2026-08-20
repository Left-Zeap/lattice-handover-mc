"""二维扫描页：失谐--功率网格扫描、2x2 热图与条件框选。

扫描完成后不再自动选最优/较差点，而是基于已算矩阵做条件筛选
（``controllers.scan_condition_mask``，不触发重新计算），并在四张
热图上叠加符合点的散点和掩膜轮廓线。
"""

from __future__ import annotations

import re

import matplotlib
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from continuous_loading.full_chain import FullChainScanResult

from .. import controllers
from ..state import AppState, format_runtime
from ..widgets.forms import (
    ChainParameterForm,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)
from ..widgets.plot_canvas import PlotCanvas
from ..widgets.result_cards import format_value
from ..workers import CalcWorker


_HEATMAP_PANELS = (
    ("science_total_temperature_rise_uK", "科学区总升温 (µK)", "inferno"),
    ("final_retention_from_mot", "相对 MOT 总留存", "viridis"),
    ("handover_transfer_efficiency", "handover 交接率", "viridis"),
    ("science_peak_density_m3", "科学区峰值密度 (m$^{-3}$)", "magma"),
)

_PROGRESS_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)")


class ScanPage(QWidget):
    """左侧扫描参数表单，右侧进度条、热图、点详情和条件筛选面板。"""

    def __init__(
        self,
        state: AppState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._worker: CalcWorker | None = None
        self._result: FullChainScanResult | None = None
        self._mask: np.ndarray | None = None
        self._heatmap_axes = None

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.form = ChainParameterForm(scan_preset=True)
        left_layout.addWidget(self.form)
        left_layout.addWidget(self._build_scan_grid_group())
        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setWidget(left)
        splitter.addWidget(form_scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        button_row = QHBoxLayout()
        self.run_button = QPushButton("开始扫描")
        self.run_button.clicked.connect(self._run_scan)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setProperty("danger", True)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.cancel_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        button_row.addWidget(self.progress_bar, 1)
        right_layout.addLayout(button_row)

        self.progress_label = QLabel(
            "就绪（默认 9x9 网格、每点 500 轨迹、串行后端；"
            "MC 运输每点数十秒级，请用小网格；"
            "GPU 模式不用进程池，全部网格点自动合并为单次批量 GPU 调用）"
        )
        self.progress_label.setObjectName("hint")
        right_layout.addWidget(self.progress_label)

        self._heat_stack = QStackedLayout()
        heat_hint = QLabel("尚无扫描结果：设置网格后点击「开始扫描」")
        heat_hint.setObjectName("hint")
        heat_hint.setAlignment(Qt.AlignCenter)
        self._heat_stack.addWidget(heat_hint)
        self.heatmap_canvas = PlotCanvas(rows=2, cols=2)
        self.heatmap_canvas.canvas.mpl_connect(
            "button_press_event", self._on_heatmap_click
        )
        self._heat_stack.addWidget(self.heatmap_canvas)
        right_layout.addLayout(self._heat_stack, 1)

        self.detail_label = QLabel("点击热图查看任意网格点的详细指标")
        self.detail_label.setObjectName("hint")
        self.detail_label.setWordWrap(True)
        right_layout.addWidget(self.detail_label)

        right_layout.addWidget(self._build_condition_panel())

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setSizes((430, 850))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._state.scan_updated.connect(self._show_last_scan)

    def _build_scan_grid_group(self) -> QGroupBox:
        """本页专属的扫描网格参数（从共享表单独立出来）。"""
        group = QGroupBox("扫描网格（本页参数）")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        defaults = controllers.default_form_params(
            str(self.form._widgets["atom_label"].currentData()),
            scan_preset=True,
        )

        self.scan_detuning_min = NoWheelDoubleSpinBox()
        self.scan_detuning_min.setRange(1.0, 5000.0)
        self.scan_detuning_min.setSingleStep(10.0)
        self.scan_detuning_min.setValue(
            float(defaults["scan_detuning_min_ghz"])
        )
        form.addRow("失谐下限 (GHz)", self.scan_detuning_min)

        self.scan_detuning_max = NoWheelDoubleSpinBox()
        self.scan_detuning_max.setRange(1.0, 5000.0)
        self.scan_detuning_max.setSingleStep(10.0)
        self.scan_detuning_max.setValue(
            float(defaults["scan_detuning_max_ghz"])
        )
        form.addRow("失谐上限 (GHz)", self.scan_detuning_max)

        self.scan_detuning_points = NoWheelSpinBox()
        self.scan_detuning_points.setRange(2, 201)
        self.scan_detuning_points.setValue(
            int(defaults["scan_detuning_points"])
        )
        form.addRow("失谐点数", self.scan_detuning_points)

        self.scan_power_min = NoWheelDoubleSpinBox()
        self.scan_power_min.setRange(0.0, 50.0)
        self.scan_power_min.setDecimals(2)
        self.scan_power_min.setSingleStep(0.1)
        self.scan_power_min.setValue(float(defaults["scan_power_min_w"]))
        form.addRow("功率下限 (W)", self.scan_power_min)

        self.scan_power_max = NoWheelDoubleSpinBox()
        self.scan_power_max.setRange(0.01, 50.0)
        self.scan_power_max.setDecimals(2)
        self.scan_power_max.setSingleStep(0.1)
        self.scan_power_max.setValue(float(defaults["scan_power_max_w"]))
        form.addRow("功率上限 (W)", self.scan_power_max)

        self.scan_power_points = NoWheelSpinBox()
        self.scan_power_points.setRange(2, 201)
        self.scan_power_points.setValue(int(defaults["scan_power_points"]))
        form.addRow("功率点数", self.scan_power_points)
        return group

    def _gather_grid_params(self) -> dict[str, object]:
        """把本页扫描网格控件的值并入参数字典。"""
        return {
            "scan_detuning_min_ghz": self.scan_detuning_min.value(),
            "scan_detuning_max_ghz": self.scan_detuning_max.value(),
            "scan_detuning_points": self.scan_detuning_points.value(),
            "scan_power_min_w": self.scan_power_min.value(),
            "scan_power_max_w": self.scan_power_max.value(),
            "scan_power_points": self.scan_power_points.value(),
        }

    def _build_condition_panel(self) -> QGroupBox:
        group = QGroupBox("条件筛选（基于已算数据框选，不重新计算）")
        grid = QGridLayout(group)

        self.power_check = QCheckBox("功率上限 P ≤")
        self.power_check.setChecked(True)
        self.power_spin = NoWheelDoubleSpinBox()
        self.power_spin.setRange(0.0, 50.0)
        self.power_spin.setDecimals(2)
        self.power_spin.setSingleStep(0.1)
        self.power_spin.setValue(1.2)
        self.power_spin.setSuffix(" W")
        grid.addWidget(self.power_check, 0, 0)
        grid.addWidget(self.power_spin, 0, 1)

        self.retention_check = QCheckBox("留存率下限 ret ≥")
        self.retention_check.setChecked(True)
        self.retention_spin = NoWheelDoubleSpinBox()
        self.retention_spin.setRange(0.0, 1.0)
        self.retention_spin.setDecimals(3)
        self.retention_spin.setSingleStep(0.05)
        self.retention_spin.setValue(0.35)
        grid.addWidget(self.retention_check, 1, 0)
        grid.addWidget(self.retention_spin, 1, 1)

        self.heating_check = QCheckBox("升温上限 heat ≤")
        self.heating_check.setChecked(False)
        self.heating_spin = NoWheelDoubleSpinBox()
        self.heating_spin.setRange(0.0, 100000.0)
        self.heating_spin.setDecimals(1)
        self.heating_spin.setSingleStep(5.0)
        self.heating_spin.setValue(40.0)
        self.heating_spin.setSuffix(" µK")
        grid.addWidget(self.heating_check, 2, 0)
        grid.addWidget(self.heating_spin, 2, 1)

        grid.addWidget(QLabel("组合方式"), 0, 2)
        self.and_radio = QRadioButton("全部满足 (AND)")
        self.and_radio.setChecked(True)
        self.or_radio = QRadioButton("任一满足 (OR)")
        grid.addWidget(self.and_radio, 0, 3)
        grid.addWidget(self.or_radio, 1, 3)

        grid.addWidget(QLabel("自定义表达式"), 2, 2)
        self.expression_edit = QLineEdit()
        self.expression_edit.setPlaceholderText("(P<=1.2)&(ret>=0.35)&(heat<=40)")
        self.expression_edit.setToolTip(
            "可用变量：P（源端功率 W）、ret（相对 MOT 总留存）、\n"
            "heat（科学区总升温 µK）、eff（handover 交接率）、dens（峰值密度 m⁻³）；\n"
            "支持比较、and/or/&amp;|、括号和一元负号。留空则用左侧勾选条件。"
        )
        grid.addWidget(self.expression_edit, 2, 3)

        self.apply_button = QPushButton("应用条件")
        self.apply_button.clicked.connect(self._apply_conditions)
        self.clear_button = QPushButton("清除条件")
        self.clear_button.setProperty("secondary", True)
        self.clear_button.clicked.connect(self._clear_conditions)
        button_box = QHBoxLayout()
        button_box.addWidget(self.apply_button)
        button_box.addWidget(self.clear_button)
        self.condition_label = QLabel("尚无扫描结果")
        self.condition_label.setObjectName("hint")
        button_box.addWidget(self.condition_label, 1)
        grid.addLayout(button_box, 3, 0, 1, 4)
        return group

    def _gather_conditions(self) -> dict[str, object]:
        return {
            "power_enabled": self.power_check.isChecked(),
            "power_max_w": self.power_spin.value(),
            "retention_enabled": self.retention_check.isChecked(),
            "retention_min": self.retention_spin.value(),
            "heating_enabled": self.heating_check.isChecked(),
            "heating_max_uK": self.heating_spin.value(),
            "mode": "and" if self.and_radio.isChecked() else "or",
            "expression": self.expression_edit.text().strip(),
        }

    def _set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def _run_scan(self) -> None:
        if self._worker is not None:
            return
        params = self.form.params()
        params.update(self._gather_grid_params())
        try:
            controllers.build_full_chain_inputs(params)
        except ValueError as exc:
            QMessageBox.warning(self, "参数无效", str(exc))
            return
        self._set_busy(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("扫描启动中…")
        self._worker = CalcWorker(
            lambda progress: controllers.run_scan(params, progress), self
        )
        self._worker.progressed.connect(self._on_progress)
        self._worker.finished.connect(lambda result: self._on_finished(result, params))
        self._worker.failed.connect(lambda message: self._on_failed(message, params))
        self._worker.cancelled.connect(lambda: self._on_cancelled(params))
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.progress_label.setText(
                "正在取消…（串行模式在当前批次结束后生效）"
            )

    def _teardown_worker(self) -> float | None:
        worker = self._worker
        elapsed = None if worker is None else worker.elapsed_seconds
        self._worker = None
        self._set_busy(False)
        if worker is not None and worker.wait(30000):
            worker.deleteLater()
        # wait 超时（进程池退出慢）时保留对象，由父对象托管到线程结束。
        return elapsed

    def _on_progress(self, message: str) -> None:
        self.progress_label.setText(message)
        match = _PROGRESS_PATTERN.search(message)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            if total > 0:
                # 多阶段（L1 腿/handover/L2 腿）各自从 0 计数，进度条
                # 只增不减，避免阶段切换时回退造成"倒退"错觉。
                self.progress_bar.setValue(
                    max(
                        self.progress_bar.value(),
                        int(100 * done / total),
                    )
                )

    def _on_finished(self, result: FullChainScanResult, params: dict) -> None:
        elapsed = self._teardown_worker()
        # 新扫描完成后自动清除旧条件叠加。
        self._mask = None
        self.condition_label.setText("未应用条件")
        self._state.set_scan(result)
        self._state.add_history(
            "扫描",
            controllers.summarize_scan_params(params),
            "完成",
            payload=result,
            elapsed_seconds=elapsed,
        )
        self.progress_bar.setValue(100)
        self.progress_label.setText(
            f"扫描完成：{result.evaluated_points} 个可行点；"
            f"运行时间 {format_runtime(elapsed)}"
        )

    def _on_failed(self, message: str, params: dict) -> None:
        elapsed = self._teardown_worker()
        self._state.add_history(
            "扫描",
            controllers.summarize_scan_params(params),
            f"失败：{message}",
            elapsed_seconds=elapsed,
        )
        self.progress_label.setText(
            f"扫描失败：{message}；运行时间 {format_runtime(elapsed)}"
        )
        # 即使全网格均为零捕获或出现不可恢复异常，也不要把旧图留在
        # 页面上造成“本次扫描没有执行”的误解；绘制带实际坐标范围的
        # 空诊断图。正常的局部失败点仍由结果矩阵中的 NaN 灰格表示。
        self._show_failed_plot(params, message)
        QMessageBox.warning(self, "扫描失败", message)

    def _on_cancelled(self, params: dict) -> None:
        elapsed = self._teardown_worker()
        self._state.add_history(
            "扫描",
            controllers.summarize_scan_params(params),
            "已取消",
            elapsed_seconds=elapsed,
        )
        self.progress_label.setText(
            f"扫描已取消；运行时间 {format_runtime(elapsed)}"
        )

    def _show_last_scan(self) -> None:
        if self._state.last_scan is not None:
            self._show_result(self._state.last_scan)

    def _show_result(self, result: FullChainScanResult) -> None:
        self._result = result
        self._heat_stack.setCurrentIndex(1)
        detunings = np.asarray(result.detuning_ghz)
        powers = np.asarray(result.source_power_w)
        axes = self.heatmap_canvas.make_axes().ravel()
        self._heatmap_axes = axes
        for axis, (field, title, cmap_name) in zip(axes, _HEATMAP_PANELS):
            matrix = np.asarray(getattr(result, field), dtype=float)
            cmap = matplotlib.colormaps[cmap_name].with_extremes(bad="#e5e7eb")
            mesh = axis.pcolormesh(
                detunings,
                powers,
                matrix,
                shading="nearest",
                cmap=cmap,
            )
            axis.set_xlabel("D1 红失谐 (GHz)")
            axis.set_ylabel("handover 端源端功率 (W)")
            axis.set_title(title)
            self.heatmap_canvas.figure.colorbar(mesh, ax=axis)
        if self._mask is not None:
            self._overlay_mask(axes, detunings, powers, self._mask)
        self.heatmap_canvas.redraw()
        self._state.register_figure("scan", self.heatmap_canvas.figure)

    def _show_failed_plot(self, params: dict, message: str) -> None:
        """Draw an honest empty scan canvas when no result object exists."""
        self._result = None
        self._mask = None
        # 当前画布已经不是上一份成功结果，清除 live-result 关联，避免
        # 导出页把这张失败诊断图误配给历史扫描数据。
        self._state.last_scan = None
        self._heat_stack.setCurrentIndex(1)
        axes = self.heatmap_canvas.make_axes().ravel()
        self._heatmap_axes = axes
        detuning_min = float(params["scan_detuning_min_ghz"])
        detuning_max = float(params["scan_detuning_max_ghz"])
        power_min = float(params["scan_power_min_w"])
        power_max = float(params["scan_power_max_w"])
        short_message = (
            message if len(message) <= 90 else message[:87] + "..."
        )
        for axis, (_, title, _) in zip(axes, _HEATMAP_PANELS):
            axis.set_facecolor("#e5e7eb")
            axis.set_xlim(detuning_min, detuning_max)
            axis.set_ylim(power_min, power_max)
            axis.set_xlabel("D1 红失谐 (GHz)")
            axis.set_ylabel("handover 端源端功率 (W)")
            axis.set_title(title)
            axis.text(
                0.5,
                0.5,
                "本次扫描无可绘有效点\n" + short_message,
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#374151",
                fontsize=9,
                wrap=True,
            )
        self.detail_label.setText(
            "本次扫描没有有效网格结果；坐标图仍已生成，"
            "灰色区域表示无数据。请检查上方失败原因。"
        )
        self.condition_label.setText("无有效扫描结果")
        self.heatmap_canvas.redraw()
        self._state.register_figure("scan", self.heatmap_canvas.figure)

    def _overlay_mask(self, axes, detunings, powers, mask: np.ndarray) -> None:
        """在四张热图上叠加符合点散点和掩膜轮廓线。"""
        power_indices, detuning_indices = np.nonzero(mask)
        has_boundary = bool(mask.any()) and bool((~mask).any())
        for axis in axes:
            if len(detuning_indices):
                axis.scatter(
                    detunings[detuning_indices],
                    powers[power_indices],
                    facecolors="none",
                    edgecolors="white",
                    s=42,
                    linewidths=1.2,
                    zorder=5,
                )
            if has_boundary and mask.shape[0] >= 2 and mask.shape[1] >= 2:
                axis.contour(
                    detunings,
                    powers,
                    mask.astype(float),
                    levels=[0.5],
                    colors="#2563eb",
                    linewidths=1.5,
                )

    def _apply_conditions(self) -> None:
        if self._result is None:
            QMessageBox.information(self, "无数据", "请先运行一次扫描")
            return
        conditions = self._gather_conditions()
        try:
            mask = controllers.scan_condition_mask(self._result, conditions)
        except ValueError as exc:
            QMessageBox.warning(self, "条件无效", str(exc))
            return
        self._mask = mask
        self.condition_label.setText(
            f"符合条件 {int(mask.sum())} / {mask.size} 点"
        )
        self._show_result(self._result)

    def _clear_conditions(self) -> None:
        self._mask = None
        self.condition_label.setText(
            "未应用条件" if self._result is not None else "尚无扫描结果"
        )
        if self._result is not None:
            self._show_result(self._result)

    def _on_heatmap_click(self, event) -> None:
        if self._result is None or event.xdata is None or event.ydata is None:
            return
        if self._heatmap_axes is None or event.inaxes not in self._heatmap_axes:
            return
        result = self._result
        detunings = np.asarray(result.detuning_ghz)
        powers = np.asarray(result.source_power_w)
        detuning_index = int(np.argmin(np.abs(detunings - event.xdata)))
        power_index = int(np.argmin(np.abs(powers - event.ydata)))
        efficiency = result.handover_transfer_efficiency[power_index][detuning_index]
        heating = result.science_total_temperature_rise_uK[power_index][detuning_index]
        retention = result.final_retention_from_mot[power_index][detuning_index]
        density = result.science_peak_density_m3[power_index][detuning_index]
        feasible = result.transport_feasible[power_index][detuning_index]
        selected = (
            None if self._mask is None else bool(self._mask[power_index][detuning_index])
        )
        selected_text = (
            "" if selected is None else f" | 当前条件 {'符合' if selected else '不符合'}"
        )
        self.detail_label.setText(
            f"网格点：失谐 {detunings[detuning_index]:g} GHz，"
            f"功率 {powers[power_index]:g} W | "
            f"L1 可行 {'是' if feasible else '否'} | "
            f"交接率 {format_value(efficiency)} | "
            f"科学区总升温 {format_value(heating)} µK | "
            f"总留存 {format_value(retention)} | "
            f"峰值密度 {format_value(density)} m⁻³{selected_text}"
        )
