"""时序可视化页：时间轴拖动 + 装置示意图 + 2x2 时序曲线光标。

数据来自 ``AppState.last_single_point``：时间轴由 ``ui.timeline`` 组装；
拖动时间滑块时只更新光标线、标记点、示意图红点和读出条
（``set_data``/``set_xdata`` + ``draw_idle``），不整图重画。
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Rectangle
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from ..state import AppState
from ..timeline import build_full_series, build_timeline, sample_timeline
from ..widgets.plot_canvas import PlotCanvas


_STAGE_LABELS = {
    "L1 transport": "L1 运输",
    "handover": "交接 handover",
    "L2 transport": "L2 运输",
}
_STAGE_COLORS = {
    "L1 transport": "#2563eb",
    "handover": "#d97706",
    "L2 transport": "#059669",
}
_SLIDER_STEPS_PER_MS = 100  # 滑块整数刻度，0.01 ms 分辨率


class TimingPage(QWidget):
    """顶部控制条 + 装置示意图 + 状态读出条 + 2x2 时序图。"""

    def __init__(
        self,
        state: AppState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._timeline: dict | None = None
        self._cursors: list = []
        self._markers: list = []
        self._schematic_dot = None
        self._schematic_stage_texts: dict = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        top_row = QHBoxLayout()
        title = QLabel("时序可视化（最近一次单点全链路结果）")
        title.setStyleSheet("font-weight: bold;")
        refresh = QPushButton("刷新图表")
        refresh.setProperty("secondary", True)
        refresh.clicked.connect(self.refresh)
        top_row.addWidget(title)
        top_row.addStretch(1)
        top_row.addWidget(refresh)
        layout.addLayout(top_row)

        self._stack = QStackedLayout()
        hint = QLabel("请先在「单点计算」页运行一次单点全链路计算")
        hint.setObjectName("hint")
        hint.setAlignment(Qt.AlignCenter)
        self._stack.addWidget(hint)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("时间"))
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.valueChanged.connect(self._on_slider_changed)
        control_row.addWidget(self.time_slider, 1)
        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(0.0, 0.0)
        self.time_spin.setDecimals(2)
        self.time_spin.setSingleStep(0.1)
        self.time_spin.setSuffix(" ms")
        self.time_spin.valueChanged.connect(self._on_spin_changed)
        control_row.addWidget(self.time_spin)
        content_layout.addLayout(control_row)

        self.schematic_canvas = PlotCanvas(rows=1, cols=1)
        self.schematic_canvas.setFixedHeight(220)
        content_layout.addWidget(self.schematic_canvas)

        self.readout_label = QLabel("—")
        self.readout_label.setObjectName("hint")
        content_layout.addWidget(self.readout_label)

        self.canvas = PlotCanvas(rows=2, cols=2)
        content_layout.addWidget(self.canvas, 1)

        self._stack.addWidget(content)
        layout.addLayout(self._stack, 1)

        self._state.single_point_updated.connect(self.refresh)
        self.refresh()

    # ---- 数据与整图重建 ----

    def refresh(self) -> None:
        simulation = self._state.last_single_point
        if simulation is None:
            self._timeline = None
            self._stack.setCurrentIndex(0)
            return
        self._timeline = build_timeline(simulation)
        time_ms = self._timeline["time_ms"]
        t_end = float(time_ms[-1])
        for widget in (self.time_slider, self.time_spin):
            widget.blockSignals(True)
        self.time_slider.setRange(0, int(round(t_end * _SLIDER_STEPS_PER_MS)))
        self.time_slider.setValue(0)
        self.time_spin.setRange(0.0, t_end)
        self.time_spin.setValue(0.0)
        for widget in (self.time_slider, self.time_spin):
            widget.blockSignals(False)
        self._plot(simulation)
        self._stack.setCurrentIndex(1)
        self._update_cursor(0.0)

    def _plot(self, simulation) -> None:
        self._cursors = []
        self._markers = []
        timeline = self._timeline
        series = build_full_series(simulation)
        axes = self.canvas.make_axes().ravel()

        # 全程时间轴：handover 段为 NaN 断开，由色带解释。
        time_ms = series["time_ms"]
        spans = [
            (series["handover_start_ms"], series["handover_end_ms"], "#d97706"),
            (series["l2_start_ms"], series["l2_end_ms"], "#059669"),
        ]
        ax = axes[0]
        ax.plot(time_ms, series["velocity_m_s"], color="#2563eb", label="速度 (m/s)")
        ax.plot(
            time_ms,
            series["acceleration_m_s2"] / 1000.0,
            color="#d97706",
            linestyle="--",
            label="加速度 ×10$^{-3}$ (m/s²)",
        )
        ax_right = ax.twinx()
        ax_right.plot(
            time_ms,
            series["aom_frequency_difference_mhz"],
            color="#059669",
            linestyle=":",
            label="AOM 频差 (MHz)",
        )
        ax.set_xlabel("时间 (ms)")
        ax.set_ylabel("速度 / 加速度")
        ax_right.set_ylabel("AOM 频差 (MHz)")
        lines = ax.get_lines() + ax_right.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="best")
        ax.set_title("运动学时序（L1+L2）")
        ax.grid(alpha=0.2)
        self._register_cursor(ax, time_ms)
        self._register_marker(ax, time_ms, series["velocity_m_s"], "#2563eb")
        self._register_marker(
            ax_right, time_ms, series["aom_frequency_difference_mhz"], "#059669"
        )

        ax = axes[1]
        ax.plot(time_ms, series["waist_um"], color="#2563eb", label="束腰 (µm)")
        ax_right = ax.twinx()
        ax_right.plot(
            time_ms,
            series["source_power_w"],
            color="#dc2626",
            linestyle="--",
            label="源端功率/分支 (W)",
        )
        ax.set_xlabel("时间 (ms)")
        ax.set_ylabel("束腰 (µm)")
        ax_right.set_ylabel("源端功率 (W)")
        lines = ax.get_lines() + ax_right.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="best")
        ax.set_title("光路时序（L1+L2）")
        ax.grid(alpha=0.2)
        self._register_cursor(ax, time_ms)
        self._register_marker(ax, time_ms, series["waist_um"], "#2563eb")
        self._register_marker(
            ax_right, time_ms, series["source_power_w"], "#dc2626"
        )

        ax = axes[2]
        ax.plot(time_ms, timeline["temperature_uK"], color="#dc2626")
        ax.set_xlabel("时间 (ms)")
        ax.set_ylabel("温度 (µK)")
        ax.set_title("全链路温度（handover 段为动能温度口径）")
        ax.grid(alpha=0.2)
        self._register_cursor(ax, time_ms)
        self._register_marker(ax, time_ms, timeline["temperature_uK"], "#dc2626")

        ax = axes[3]
        ax.plot(time_ms, timeline["retention_from_mot"], color="#2563eb")
        ax.set_xlabel("时间 (ms)")
        ax.set_ylabel("相对 MOT 留存率")
        ax.set_title("全链路留存率")
        ax.grid(alpha=0.2)
        self._register_cursor(ax, time_ms)
        self._register_marker(
            ax, time_ms, timeline["retention_from_mot"], "#2563eb"
        )

        # handover（橙）/L2（绿）色带：四张图统一标注。
        for axis in axes:
            for start, end, color in spans:
                axis.axvspan(start, end, alpha=0.15, color=color)

        self.canvas.redraw()
        self._state.register_figure("single_point", self.canvas.figure)
        self._draw_schematic()

    def _register_cursor(self, axis, time_ms) -> None:
        cursor = axis.axvline(
            float(time_ms[0]), color="#6b7280", linewidth=1.0, linestyle="-", zorder=4
        )
        self._cursors.append(cursor)

    def _register_marker(self, axis, time_ms, values, color) -> None:
        (marker,) = axis.plot(
            [float(time_ms[0])],
            [float(values[0])],
            marker="o",
            color=color,
            markeredgecolor="white",
            markersize=7,
            linestyle="none",
            zorder=5,
        )
        self._markers.append((marker, np.asarray(time_ms), np.asarray(values)))

    # ---- 装置示意图 ----

    def _draw_schematic(self) -> None:
        timeline = self._timeline
        phases = list(timeline["phase"])
        # L1 末端位置（handover 点）取相位切换处的位置，避免硬编码距离。
        handover_index = (
            phases.index("handover") if "handover" in phases else len(phases) - 1
        )
        handover_x = float(timeline["position_m"][handover_index])
        total_x = float(timeline["position_m"][-1])
        y = 0.5
        ax = self.schematic_canvas.make_axes()
        ax.set_xlim(-0.05 * total_x, 1.08 * total_x)
        ax.set_ylim(0.05, 0.95)
        ax.set_axis_off()

        # 程序从 L1 起点（静止晶格热平衡系综）传播；MOT/compress/idle
        # 只通过边界分布和前级存活率进入，不显式传播其动力学。
        mot_width = 0.05 * total_x
        ax.add_patch(
            Rectangle(
                (-mot_width / 2, y - 0.12),
                mot_width,
                0.24,
                facecolor="#eff6ff",
                edgecolor="#2563eb",
            )
        )
        ax.text(
            0.0,
            y - 0.22,
            "L1 起点（静止晶格热平衡）\n(MOT/compress/idle 为边界输入)",
            ha="center",
            fontsize=8,
            color="#374151",
        )
        # L1 光路
        ax.plot([0.0, handover_x], [y, y], color="#2563eb", linewidth=3)
        ax.text(
            handover_x / 2, y + 0.13, "L1 光路", ha="center", fontsize=9,
            color="#2563eb",
        )
        # handover 交叉示意（真实交叉角约 4°，图中夸大以便辨认）
        cross = 0.03 * total_x
        ax.plot(
            [handover_x - cross, handover_x + cross],
            [y - 0.09, y + 0.09],
            color="#d97706",
            linewidth=2,
        )
        ax.plot(
            [handover_x - cross, handover_x + cross],
            [y + 0.09, y - 0.09],
            color="#d97706",
            linewidth=2,
            linestyle="--",
        )
        ax.text(
            handover_x, y + 0.2, "handover\n(L1/L2 交叉 ≈4°，示意)",
            ha="center", fontsize=8, color="#d97706",
        )
        # L2 光路
        ax.plot([handover_x, total_x], [y, y], color="#059669", linewidth=3)
        ax.text(
            (handover_x + total_x) / 2, y + 0.13, "L2 光路", ha="center",
            fontsize=9, color="#059669",
        )
        # 科学区（矩形 + 物镜）
        science_width = 0.05 * total_x
        ax.add_patch(
            Rectangle(
                (total_x - science_width / 2, y - 0.12),
                science_width,
                0.24,
                facecolor="#ecfdf5",
                edgecolor="#059669",
            )
        )
        ax.add_patch(
            Circle(
                (total_x, y + 0.24), 0.035 * total_x,
                facecolor="none", edgecolor="#059669", linestyle="--",
            )
        )
        ax.plot([total_x, total_x], [y + 0.12, y + 0.205], color="#059669", linewidth=1)
        ax.text(
            total_x, y - 0.22, "科学区（物镜示意）", ha="center", fontsize=9,
            color="#374151",
        )
        # 阶段标注（随时间拖动高亮当前阶段）
        self._schematic_stage_texts = {}
        for phase, x_text in (
            ("L1 transport", handover_x / 2),
            ("handover", handover_x),
            ("L2 transport", (handover_x + total_x) / 2),
        ):
            if phase not in phases:
                continue
            text = ax.text(
                x_text, y - 0.32, _STAGE_LABELS[phase], ha="center", fontsize=10,
                color="#9ca3af", fontweight="normal",
            )
            self._schematic_stage_texts[phase] = text
        # 原子云红点
        (self._schematic_dot,) = ax.plot(
            [0.0], [y], marker="o", color="#dc2626", markeredgecolor="white",
            markersize=11, linestyle="none", zorder=10,
        )
        self.schematic_canvas.redraw()

    # ---- 拖动时的轻量更新 ----

    def _on_slider_changed(self, value: int) -> None:
        t_ms = value / _SLIDER_STEPS_PER_MS
        self.time_spin.blockSignals(True)
        self.time_spin.setValue(t_ms)
        self.time_spin.blockSignals(False)
        self._update_cursor(t_ms)

    def _on_spin_changed(self, value: float) -> None:
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(int(round(value * _SLIDER_STEPS_PER_MS)))
        self.time_slider.blockSignals(False)
        self._update_cursor(value)

    def _update_cursor(self, t_ms: float) -> None:
        if self._timeline is None:
            return
        sample = sample_timeline(self._timeline, t_ms)
        t = float(sample["time_ms"])
        for cursor in self._cursors:
            cursor.set_xdata([t, t])
        for marker, time_ms, values in self._markers:
            marker.set_data([t], [float(np.interp(t, time_ms, values))])
        self.canvas.redraw()

        if self._schematic_dot is not None:
            self._schematic_dot.set_data([float(sample["position_m"])], [0.5])
            phase = str(sample["phase"])
            for name, text in self._schematic_stage_texts.items():
                active = name == phase
                text.set_fontweight("bold" if active else "normal")
                text.set_color(_STAGE_COLORS[name] if active else "#9ca3af")
            self.schematic_canvas.redraw()

        self.readout_label.setText(
            f"当前时间 {t:.2f} ms | 阶段 {_STAGE_LABELS.get(str(sample['phase']), sample['phase'])} | "
            f"位置 {sample['position_m']:.3f} m | 速度 {sample['velocity_m_s']:.2f} m/s | "
            f"温度 {sample['temperature_uK']:.1f} µK | "
            f"相对 MOT 留存率 {sample['retention_from_mot']:.4f}"
        )
