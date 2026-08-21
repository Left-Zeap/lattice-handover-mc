"""云宽扫描页：固定工作点上的原子云轴向宽度一维扫描。

左侧为共享物理参数表单（``ChainParameterForm``，含计算设备选择——
GPU 由它控制）与 "云宽扫描" 控制组（σ 范围、取点数）；右上为
开始/取消按钮与进度条（与二维扫描页同位置），下方为两张折线图
（温度、交接率，横轴均为无量纲 χ = σ_c·sinθ/w）。逐点失败由后端
隔离为无效点，绘图时跳过；全部无效时显示提示。
"""

from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from continuous_loading.cloud_sigma_scan import CloudSigmaScanResult

from .. import controllers
from ..state import AppState, format_runtime
from ..widgets.forms import (
    ChainParameterForm,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)
from ..widgets.plot_canvas import PlotCanvas
from ..workers import CalcWorker


_PROGRESS_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)")


class CloudSigmaPage(QWidget):
    """左侧参数表单与扫描控制，右侧进度条和温度/交接率折线图。"""

    def __init__(
        self,
        state: AppState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._worker: CalcWorker | None = None
        self._result: CloudSigmaScanResult | None = None

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.form = ChainParameterForm()
        left_layout.addWidget(self.form)
        left_layout.addWidget(self._build_control_group())
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
            "就绪（固定失谐/固定源功率工作点，扫描原子云轴向宽度 σ；"
            "σ=0 表示全部原子位于中心格点链；"
            "连续相空间或 Monte Carlo 运输模式下每个 σ 点数十秒级）"
        )
        self.progress_label.setObjectName("hint")
        self.progress_label.setWordWrap(True)
        right_layout.addWidget(self.progress_label)

        self._plot_stack = QStackedLayout()
        self._hint_label = QLabel("尚无扫描结果：设置 σ 范围后点击「开始扫描」")
        self._hint_label.setObjectName("hint")
        self._hint_label.setAlignment(Qt.AlignCenter)
        self._plot_stack.addWidget(self._hint_label)
        plots = QWidget()
        plots_layout = QVBoxLayout(plots)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(8)
        self.temperature_canvas = PlotCanvas()
        self.efficiency_canvas = PlotCanvas()
        plots_layout.addWidget(self.temperature_canvas, 1)
        plots_layout.addWidget(self.efficiency_canvas, 1)
        self._plot_stack.addWidget(plots)
        right_layout.addLayout(self._plot_stack, 1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setSizes((430, 850))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def _build_control_group(self) -> QGroupBox:
        group = QGroupBox("云宽扫描")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.sigma_min_spin = NoWheelDoubleSpinBox()
        self.sigma_min_spin.setRange(0.0, 100.0)
        self.sigma_min_spin.setDecimals(2)
        self.sigma_min_spin.setSingleStep(0.1)
        self.sigma_min_spin.setValue(0.0)
        form.addRow("σ 下限 (mm)", self.sigma_min_spin)

        self.sigma_max_spin = NoWheelDoubleSpinBox()
        self.sigma_max_spin.setRange(0.01, 100.0)
        self.sigma_max_spin.setDecimals(2)
        self.sigma_max_spin.setSingleStep(0.1)
        self.sigma_max_spin.setValue(5.0)
        form.addRow("σ 上限 (mm)", self.sigma_max_spin)

        self.points_spin = NoWheelSpinBox()
        self.points_spin.setRange(2, 1000)
        self.points_spin.setValue(10)
        form.addRow("取点数", self.points_spin)
        return group

    def _gather_params(self) -> dict[str, object]:
        params = self.form.params()
        params["cloud_sigma_min_mm"] = self.sigma_min_spin.value()
        params["cloud_sigma_max_mm"] = self.sigma_max_spin.value()
        params["cloud_sigma_points"] = self.points_spin.value()
        return params

    def _set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def _run_scan(self) -> None:
        if self._worker is not None:
            return
        params = self._gather_params()
        try:
            controllers.build_full_chain_inputs(params)
        except ValueError as exc:
            QMessageBox.warning(self, "参数无效", str(exc))
            return
        self._set_busy(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("扫描启动中…")
        self._worker = CalcWorker(
            lambda progress: controllers.run_cloud_sigma_scan(params, progress),
            self,
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
                "正在取消…（串行模式在当前点结束后生效）"
            )

    def _teardown_worker(self) -> float | None:
        worker = self._worker
        elapsed = None if worker is None else worker.elapsed_seconds
        self._worker = None
        self._set_busy(False)
        if worker is not None and worker.wait(30000):
            worker.deleteLater()
        return elapsed

    def _on_progress(self, message: str) -> None:
        self.progress_label.setText(message)
        match = _PROGRESS_PATTERN.search(message)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            if total > 0:
                # 进度条只增不减，避免失败汇总等附加消息造成回退。
                self.progress_bar.setValue(
                    max(
                        self.progress_bar.value(),
                        int(100 * done / total),
                    )
                )

    def _on_finished(self, result: CloudSigmaScanResult, params: dict) -> None:
        elapsed = self._teardown_worker()
        self._result = result
        self._state.add_history(
            "云宽扫描",
            controllers.summarize_cloud_sigma_params(params),
            "完成",
            payload=result,
            elapsed_seconds=elapsed,
        )
        self.progress_bar.setValue(100)
        valid = sum(1 for point in result.points if point.error is None)
        self.progress_label.setText(
            f"扫描完成：{valid}/{len(result.points)} 个有效点；"
            f"运行时间 {format_runtime(elapsed)}"
        )
        self._show_result(result)

    def _on_failed(self, message: str, params: dict) -> None:
        elapsed = self._teardown_worker()
        self._state.add_history(
            "云宽扫描",
            controllers.summarize_cloud_sigma_params(params),
            f"失败：{message}",
            elapsed_seconds=elapsed,
        )
        self.progress_label.setText(
            f"扫描失败：{message}；运行时间 {format_runtime(elapsed)}"
        )
        self._show_hint(f"本次扫描失败：{message}")
        QMessageBox.warning(self, "扫描失败", message)

    def _on_cancelled(self, params: dict) -> None:
        elapsed = self._teardown_worker()
        self._state.add_history(
            "云宽扫描",
            controllers.summarize_cloud_sigma_params(params),
            "已取消",
            elapsed_seconds=elapsed,
        )
        self.progress_label.setText(
            f"扫描已取消；运行时间 {format_runtime(elapsed)}"
        )

    def _show_hint(self, text: str) -> None:
        self._hint_label.setText(text)
        self._plot_stack.setCurrentIndex(0)

    def _show_result(self, result: CloudSigmaScanResult) -> None:
        valid = [point for point in result.points if point.error is None]
        if not valid:
            self._show_hint(
                "本次扫描无有效点（全部失败），请检查工作点参数"
            )
            return
        x = [point.chi for point in valid]
        self._plot_lines(
            self.temperature_canvas.make_axes(),
            x,
            (
                ("handover 末温", [p.handover_temperature_uK for p in valid]),
                ("链末总温", [p.final_temperature_uK for p in valid]),
            ),
            ylabel="温度 (µK)",
            title="云宽对温度的影响",
        )
        self.temperature_canvas.redraw()
        self._plot_lines(
            self.efficiency_canvas.make_axes(),
            x,
            (
                ("handover 交接率", [p.handover_efficiency for p in valid]),
                ("相对 MOT 总留存", [p.final_retention_from_mot for p in valid]),
            ),
            ylabel="比例",
            title="云宽对交接效率的影响",
        )
        self.efficiency_canvas.redraw()
        self._plot_stack.setCurrentIndex(1)
        self._state.register_figure(
            "cloud_sigma_temperature", self.temperature_canvas.figure
        )
        self._state.register_figure(
            "cloud_sigma_efficiency", self.efficiency_canvas.figure
        )

    @staticmethod
    def _plot_lines(axis, x, series, *, ylabel: str, title: str) -> None:
        """绘制多条折线；逐条跳过 None（如 L2 零捕获）指标。"""
        for label, values in series:
            xs = [xi for xi, yi in zip(x, values) if yi is not None]
            ys = [yi for yi in values if yi is not None]
            if xs:
                axis.plot(xs, ys, "o-", markersize=4, label=label)
        axis.set_xlabel("χ = σ_c·sinθ/w（云宽/重合区尺度）")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        if axis.lines:
            axis.legend()
