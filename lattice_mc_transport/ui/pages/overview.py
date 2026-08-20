"""概览页:输运流程图 + 当前参数总览 + 运行环境。"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)
from matplotlib import patches

from .. import theme
from ..widgets import Card, ChartCanvas


class OverviewPage(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        flow_card = Card("输运流程")
        self.flow_chart = ChartCanvas(height_in=2.0)
        flow_card.layout.addWidget(self.flow_chart)
        root.addWidget(flow_card)

        row = QHBoxLayout()
        row.setSpacing(12)
        param_card = Card("当前参数")
        self.param_grid = QGridLayout()
        self.param_grid.setHorizontalSpacing(24)
        self.param_grid.setVerticalSpacing(6)
        param_card.layout.addLayout(self.param_grid)
        row.addWidget(param_card, 3)

        env_card = Card("运行环境")
        self.env_box = QVBoxLayout()
        self.env_box.setSpacing(6)
        env_card.layout.addLayout(self.env_box)
        env_card.layout.addStretch(1)
        row.addWidget(env_card, 2)
        root.addLayout(row)

        self._fill_env()
        self.refresh()
        state.cfg_changed.connect(self.refresh)

    # ---- pipeline diagram ----
    def _draw_flow(self, cfg):
        fig = self.flow_chart.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.axis("off")

        g = cfg["geometry"]
        stages = [
            ("L1 输运", f'{g["l1"]["distance_m"]*100:g} cm / {g["l1"]["duration_s"]*1e3:g} ms',
             f'a = {g["l1"]["acceleration_m_s2"]:g} m/s²', theme.STAGE_COLORS["L1"]),
            ("交接 handover", f'{g["handover"]["duration_s"]*1e3:g} ms',
             f'夹角 {g["handover_angle_deg"]:g}°', theme.STAGE_COLORS["handover"]),
            ("L2 输运", f'{g["l2"]["distance_m"]*100:g} cm / {g["l2"]["duration_s"]*1e3:g} ms',
             f'a = {g["l2"]["acceleration_m_s2"]:g} m/s²', theme.STAGE_COLORS["L2"]),
        ]
        n = len(stages)
        for i, (name, line1, line2, color) in enumerate(stages):
            x0 = 0.04 + i * (0.94 / n)
            w = 0.94 / n - 0.06
            ax.add_patch(patches.FancyBboxPatch(
                (x0, 0.35), w, 0.42, boxstyle="round,pad=0.015",
                facecolor=color, edgecolor="none"))
            ax.text(x0 + w / 2, 0.64, name, ha="center", va="center",
                    color="white", fontsize=12, fontweight="bold")
            ax.text(x0 + w / 2, 0.50, line1, ha="center", va="center",
                    color="white", fontsize=9)
            ax.text(x0 + w / 2, 0.41, line2, ha="center", va="center",
                    color="white", fontsize=9, alpha=0.85)
            if i < n - 1:
                ax.annotate("", xy=(x0 + w + 0.055, 0.56), xytext=(x0 + w + 0.005, 0.56),
                            arrowprops=dict(arrowstyle="-|>", color=theme.TEXT_DIM, lw=2))
        ax.text(0.5, 0.12,
                "6D 单原子轨迹 Monte Carlo:偶极力 + 重力 + 随机反冲,"
                "统计存活率 / 温度 / 散射",
                ha="center", color=theme.TEXT_DIM, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        self.flow_chart.redraw()

    def _fill_params(self, cfg):
        while self.param_grid.count():
            item = self.param_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        rows = [
            ("原子种类", cfg["species"]),
            ("原子数", f'{cfg["initial"]["n_atoms"]:,}'),
            ("初始温度", f'{cfg["initial"]["temperature_uK"]:g} µK'),
            ("D1 红失谐", f'{cfg["laser"]["d1_red_detuning_GHz"]:g} GHz'),
            ("L1 / L2 功率", f'{cfg["geometry"]["l1"]["power_w"]:g} W / '
                             f'{cfg["geometry"]["l2"]["power_w"]:g} W'),
            ("retro 功率比", f'{cfg["laser"]["retro_power_ratio"]:g}'),
            ("交接夹角", f'{cfg["geometry"]["handover_angle_deg"]:g}°'),
            ("积分步长 dt", f'{cfg["simulation"]["dt_s"]:g} s'),
            ("散射模型", "开启" if cfg["simulation"]["enable_scattering"] else "关闭"),
        ]
        for i, (k, v) in enumerate(rows):
            kl = QLabel(k)
            kl.setObjectName("hint")
            vl = QLabel(str(v))
            self.param_grid.addWidget(kl, i % 5, (i // 5) * 2)
            self.param_grid.addWidget(vl, i % 5, (i // 5) * 2 + 1)

    def _fill_env(self):
        try:
            import cupy as cp
            name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            backend = f"GPU 可用:{name}"
        except Exception:
            backend = "GPU 不可用,将使用 CPU (NumPy)"
        for text in [backend, "CPU 后端:NumPy", "GPU 后端:CuPy(融合 CUDA kernel)",
                     "扫描模式:GPU 批量并行整张网格"]:
            lb = QLabel(text)
            lb.setObjectName("hint")
            self.env_box.addWidget(lb)

    def refresh(self):
        self._draw_flow(self.state.cfg)
        self._fill_params(self.state.cfg)
