"""二维扫描页:扫描参数编辑、运行、存活率/温度热力图。"""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from .. import theme
from ..workers import ScanWorker
from ..widgets import Card, ChartCanvas, ParamForm

SCAN_SCHEMA = [
    ("扫描网格", [
        ("detuning_GHz.start", "失谐起点", "float", "GHz"),
        ("detuning_GHz.stop", "失谐终点", "float", "GHz"),
        ("detuning_GHz.num", "失谐点数", "int", ""),
        ("power_w.start", "功率起点", "float", "W"),
        ("power_w.stop", "功率终点", "float", "W"),
        ("power_w.num", "功率点数", "int", ""),
        ("n_atoms_override", "每点原子数", "int", ""),
    ]),
]

DEFAULT_SCAN_CFG = {
    "detuning_GHz": {"start": 200.0, "stop": 700.0, "num": 11},
    "power_w": {"start": 0.8, "stop": 3.5, "num": 10},
    "n_atoms_override": 1500,
}


class ScanPage(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._worker: ScanWorker | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        # 左:参数 + 控制
        left_card = Card("扫描参数")
        self.form = ParamForm(SCAN_SCHEMA, DEFAULT_SCAN_CFG)
        left_card.layout.addWidget(self.form)
        hint = QLabel("GPU 后端将把整张网格合并为一次批量模拟;"
                      "基础物理参数取「单点计算」页当前配置。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        left_card.layout.addWidget(hint)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("后端"))
        self.backend_box = QComboBox()
        self.backend_box.addItems(["auto", "gpu", "cpu"])
        ctrl.addWidget(self.backend_box)
        ctrl.addStretch(1)
        left_card.layout.addLayout(ctrl)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        left_card.layout.addWidget(self.progress)
        self.btn_run = QPushButton("开始扫描")
        self.btn_run.setObjectName("primaryBtn")
        left_card.layout.addWidget(self.btn_run)
        self.status_label = QLabel("尚未运行。")
        self.status_label.setObjectName("hint")
        left_card.layout.addWidget(self.status_label)
        left_card.layout.addStretch(1)
        root.addWidget(left_card, 2)

        # 右:两张热力图
        right = QVBoxLayout()
        right.setSpacing(12)
        card_T = Card("最终温度热力图 (µK)")
        self.chart_T = ChartCanvas(height_in=3.2)
        card_T.layout.addWidget(self.chart_T)
        card_S = Card("最终存活率热力图 (%)")
        self.chart_S = ChartCanvas(height_in=3.2)
        card_S.layout.addWidget(self.chart_S)
        right.addWidget(card_T)
        right.addWidget(card_S)
        right_widget = QWidget()
        right_widget.setLayout(right)
        root.addWidget(right_widget, 3)

        self.btn_run.clicked.connect(self._run)
        self.state.scan_finished.connect(self._on_result)
        self._draw_placeholder()

    def _draw_placeholder(self):
        for cv in (self.chart_T, self.chart_S):
            fig = cv.figure
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "扫描完成后显示热力图", ha="center", va="center",
                    color=theme.TEXT_DIM, transform=ax.transAxes)
            cv.redraw()

    def draw_heatmaps(self, dets, pows, T, S):
        self._draw_one(self.chart_T, dets, pows, T, "最终温度 (µK)",
                       "viridis", "{:.0f}")
        self._draw_one(self.chart_S, dets, pows, S * 100, "最终存活率 (%)",
                       "viridis", "{:.0f}", vmin=0, vmax=100)

    @staticmethod
    def _draw_one(canvas, dets, pows, Z, title, cmap, fmt, vmin=None, vmax=None):
        fig = canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        im = ax.imshow(
            Z, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[pows[0], pows[-1], dets[0], dets[-1]],
        )
        # 网格均匀时把像素对齐到格点中心
        if len(pows) > 1 and len(dets) > 1:
            dp = (pows[-1] - pows[0]) / (len(pows) - 1)
            dd = (dets[-1] - dets[0]) / (len(dets) - 1)
            im.set_extent([pows[0] - dp / 2, pows[-1] + dp / 2,
                           dets[0] - dd / 2, dets[-1] + dd / 2])
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                v = Z[i, j]
                if v == v:  # not NaN
                    ax.text(pows[j], dets[i], fmt.format(v), ha="center",
                            va="center", fontsize=7,
                            color="white" if v < (np.nanmax(Z) * 0.6) else "black")
        ax.set_xlabel("功率 (W)")
        ax.set_ylabel("D1 红失谐 (GHz)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.9)
        canvas.redraw()

    def _run(self):
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            scan_cfg = self.form.cfg()
        except ValueError as e:
            QMessageBox.warning(self, "参数格式错误", str(e))
            return
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("扫描中…")
        self._worker = ScanWorker(self.state.cfg, scan_cfg,
                                  self.backend_box.currentText(), self)
        self._worker.progress.connect(
            lambda d, t: self.progress.setValue(int(100 * d / max(t, 1))))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, payload):
        self.btn_run.setEnabled(True)
        self.progress.setValue(100)
        self.state.set_scan(payload)
        n = payload["T"].size
        self.status_label.setText(
            f'完成:{n} 个点,后端 {payload["backend"]}。'
            "「结果导出」页已更新。")

    def _on_result(self, payload):
        self.draw_heatmaps(payload["detunings"], payload["powers"],
                           payload["T"], payload["S"])

    def _on_fail(self, msg):
        self.btn_run.setEnabled(True)
        self.status_label.setText(f"扫描失败:{msg}")
        QMessageBox.critical(self, "扫描失败", msg)
