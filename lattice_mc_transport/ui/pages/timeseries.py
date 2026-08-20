"""时序可视化页:温度/存活率折线 + 可拖动时间光标 + 阶段概览条。"""
from __future__ import annotations
import numpy as np

from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from .. import theme
from ..widgets import Card, ChartCanvas, StageBar


class TimeseriesPage(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._data = None      # {"t_s","T_uK","S","cfg"}
        self._cursor_t = 0.0
        self._lines = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        # 阶段条 + 光标读数
        top_card = Card("时序阶段")
        self.stage_bar = StageBar()
        top_card.layout.addWidget(self.stage_bar)
        readout = QHBoxLayout()
        self.lbl_t = QLabel("t = —")
        self.lbl_T = QLabel("T = —")
        self.lbl_S = QLabel("存活率 = —")
        self.lbl_stage = QLabel("阶段:—")
        for lb in (self.lbl_t, self.lbl_T, self.lbl_S, self.lbl_stage):
            lb.setObjectName("cardTitle")
            readout.addWidget(lb)
        readout.addStretch(1)
        self.btn_load = QPushButton("加载 timeseries.npz…")
        self.btn_load.setObjectName("ghostBtn")
        readout.addWidget(self.btn_load)
        top_card.layout.addLayout(readout)
        root.addWidget(top_card)

        # 两张折线图
        self.chart_T = ChartCanvas(height_in=2.6)
        self.chart_S = ChartCanvas(height_in=2.6)
        card_t = Card("平均温度随时间")
        card_t.layout.addWidget(self.chart_T)
        card_s = Card("存活率随时间")
        card_s.layout.addWidget(self.chart_S)
        root.addWidget(card_t)
        root.addWidget(card_s)

        for cv in (self.chart_T, self.chart_S):
            cv.canvas.mpl_connect("button_press_event", self._on_mouse)
            cv.canvas.mpl_connect("motion_notify_event", self._on_mouse)
        self.stage_bar.cursor_changed.connect(self.set_cursor)

        self.btn_load.clicked.connect(self._load_npz)
        state.single_finished.connect(self._on_result)
        self._draw_empty()

    # ---- data ----
    def _on_result(self, payload):
        h = payload["history"]
        self.set_data({
            "t_s": np.asarray(h["time_s"]),
            "T_uK": np.asarray(h["T_K"]) * 1e6,
            "S": np.asarray(h["survival"]),
            "cfg": payload["cfg"],
        })

    def _load_npz(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载时间序列", "output", "npz (*.npz)")
        if not path:
            return
        d = np.load(path)
        self.set_data({
            "t_s": d["time_s"],
            "T_uK": d["T_K"] * 1e6,
            "S": d["survival"],
            "cfg": self.state.cfg,
        })

    def set_data(self, data):
        self._data = data
        self.stage_bar.set_stages(data["cfg"])
        self.set_cursor(0.0)
        self._draw()

    # ---- plotting ----
    def _draw_empty(self):
        for cv, ylabel in ((self.chart_T, "T (µK)"), (self.chart_S, "存活率")):
            fig = cv.figure
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_xlabel("t (ms)")
            ax.set_ylabel(ylabel)
            ax.text(0.5, 0.5, "先在「单点计算」页运行,或加载 timeseries.npz",
                    ha="center", va="center", color=theme.TEXT_DIM,
                    transform=ax.transAxes)
            cv.redraw()

    def _draw(self):
        d = self._data
        if d is None:
            return
        t_ms = d["t_s"] * 1e3
        self._lines = []
        for cv, y, ylabel, color in (
            (self.chart_T, d["T_uK"], "T (µK)", theme.ACCENT),
            (self.chart_S, d["S"], "存活率", theme.OK),
        ):
            fig = cv.figure
            fig.clear()
            ax = fig.add_subplot(111)
            ax.plot(t_ms, y, color=color, lw=1.6)
            ax.set_xlabel("t (ms)")
            ax.set_ylabel(ylabel)
            if cv is self.chart_S:
                ax.set_ylim(-0.03, 1.03)
            line = ax.axvline(self._cursor_t * 1e3, color=theme.DANGER, lw=1.2,
                              ls="--", alpha=0.85)
            self._lines.append(line)
            cv.redraw()
        self._update_readout()

    def set_cursor(self, t_s: float):
        self._cursor_t = float(t_s)
        self.stage_bar.set_cursor(self._cursor_t)
        for line in self._lines:
            line.set_xdata([self._cursor_t * 1e3])
        self.chart_T.canvas.draw_idle()
        self.chart_S.canvas.draw_idle()
        self._update_readout()

    def _on_mouse(self, event):
        if self._data is None or event.xdata is None:
            return
        if event.name == "button_press_event" or (
            event.name == "motion_notify_event" and event.button == 1
        ):
            self.set_cursor(event.xdata / 1e3)

    def _update_readout(self):
        d = self._data
        if d is None:
            return
        t = np.clip(self._cursor_t, d["t_s"][0], d["t_s"][-1])
        T = float(np.interp(t, d["t_s"], d["T_uK"], left=np.nan, right=np.nan))
        S = float(np.interp(t, d["t_s"], d["S"]))
        g = d["cfg"]["geometry"]
        t1 = float(g["l1"]["duration_s"])
        t2 = t1 + float(g["handover"]["duration_s"])
        stage = "L1 输运" if t < t1 else ("交接" if t < t2 else "L2 输运")
        self.lbl_t.setText(f"t = {t * 1e3:.2f} ms")
        self.lbl_T.setText(f"T = {T:.1f} µK" if T == T else "T = —")
        self.lbl_S.setText(f"存活率 = {100 * S:.1f} %")
        self.lbl_stage.setText(f"阶段:{stage}")
