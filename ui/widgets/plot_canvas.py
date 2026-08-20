"""内嵌 matplotlib 画布：FigureCanvasQTAgg + 导航工具栏。

全局设置中文字体（Microsoft YaHei / SimHei）和负号显示，保证图中
中文与负数正常渲染。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")

from matplotlib import pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402


class PlotCanvas(QWidget):
    """带工具栏（平移/缩放/保存）的 matplotlib 画布容器。"""

    def __init__(
        self,
        rows: int = 1,
        cols: int = 1,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self._rows = rows
        self._cols = cols
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def make_axes(self):
        """按构造时的行列数重建子图并返回 axes 数组。"""
        self.figure.clear()
        return self.figure.subplots(self._rows, self._cols)

    def clear(self) -> None:
        self.figure.clear()
        self.canvas.draw_idle()

    def redraw(self) -> None:
        self.canvas.draw_idle()
