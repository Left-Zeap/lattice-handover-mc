"""共享 UI 组件:卡片、图表画布、schema 驱动参数表单、时序阶段条。"""
from __future__ import annotations
from copy import deepcopy

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QFont
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QSizePolicy, QVBoxLayout, QWidget,
)

from . import theme


def get_nested(cfg: dict, path: str):
    node = cfg
    for key in path.split("."):
        node = node[key]
    return node

def set_nested(cfg: dict, path: str, value):
    keys = path.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


class Card(QFrame):
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 14)
        self.layout.setSpacing(10)
        if title:
            t = QLabel(title)
            t.setObjectName("cardTitle")
            self.layout.addWidget(t)


class MetricCard(QFrame):
    """大号数值展示卡(名称 + 数值 + 单位)。"""
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        self.value_label = QLabel("—")
        self.value_label.setObjectName("metricValue")
        name_label = QLabel(name)
        name_label.setObjectName("metricName")
        lay.addWidget(self.value_label)
        lay.addWidget(name_label)

    def set_value(self, text: str):
        self.value_label.setText(text)


class ChartCanvas(QWidget):
    """内嵌 matplotlib 画布。"""
    def __init__(self, parent=None, height_in=3.2):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(5, height_in))
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def redraw(self):
        self.figure.tight_layout()
        self.canvas.draw_idle()


class ParamForm(QWidget):
    """
    schema: [(分组名, [(path, 标签, 类型, 单位或选项), ...]), ...]
    类型: float / int / bool / text(逗号分隔列表) / choice
    """
    def __init__(self, schema, cfg: dict, parent=None):
        super().__init__(parent)
        self._schema = schema
        self._editors = {}
        self.bind_base(cfg)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)
        for group_name, fields in schema:
            card = Card(group_name)
            grid = QGridLayout()
            grid.setColumnStretch(1, 1)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(8)
            for row, (path, label, ftype, extra) in enumerate(fields):
                grid.addWidget(QLabel(label), row, 0)
                editor = self._make_editor(ftype, extra)
                self._editors[path] = (editor, ftype)
                cell = QHBoxLayout()
                cell.setContentsMargins(0, 0, 0, 0)
                cell.addWidget(editor, 1)
                if ftype in ("float", "int") and extra:
                    unit = QLabel(extra)
                    unit.setObjectName("hint")
                    cell.addWidget(unit)
                grid.addLayout(cell, row, 1)
            card.layout.addLayout(grid)
            outer.addWidget(card)
        outer.addStretch(1)
        self.set_cfg(cfg)

    def _make_editor(self, ftype, extra):
        if ftype == "bool":
            return QCheckBox()
        if ftype == "choice":
            cb = QComboBox()
            cb.addItems(extra)
            return cb
        return QLineEdit()

    def set_cfg(self, cfg: dict):
        for path, (editor, ftype) in self._editors.items():
            value = get_nested(cfg, path)
            if ftype == "bool":
                editor.setChecked(bool(value))
            elif ftype == "choice":
                editor.setCurrentText(str(value))
            elif ftype == "text":
                editor.setText(", ".join(f"{v:g}" for v in value))
            else:
                editor.setText(f"{value:g}" if isinstance(value, float) else str(value))

    def cfg(self) -> dict:
        out = deepcopy(self._base_cfg)
        for path, (editor, ftype) in self._editors.items():
            if ftype == "bool":
                set_nested(out, path, editor.isChecked())
            elif ftype == "choice":
                set_nested(out, path, editor.currentText())
            elif ftype == "int":
                set_nested(out, path, int(float(editor.text())))
            elif ftype == "float":
                set_nested(out, path, float(editor.text()))
            elif ftype == "text":
                set_nested(out, path, [float(x) for x in editor.text().split(",") if x.strip()])
        return out

    def bind_base(self, cfg: dict):
        """记录深拷贝基准,供 cfg() 合成完整配置。"""
        self._base_cfg = deepcopy(cfg)


class StageBar(QWidget):
    """时序阶段条:L1 → handover → L2 三段 + 可拖动光标。"""
    cursor_changed = Signal(float)  # t in seconds

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(64)
        self._stages = []          # [(name, t0, t1)]
        self._cursor = 0.0
        self._dragging = False

    def set_stages(self, cfg: dict):
        g = cfg["geometry"]
        t1 = float(g["l1"]["duration_s"])
        t2 = t1 + float(g["handover"]["duration_s"])
        t3 = t2 + float(g["l2"]["duration_s"])
        self._stages = [("L1 输运", 0.0, t1), ("交接", t1, t2), ("L2 输运", t2, t3)]
        self.update()

    def set_cursor(self, t: float):
        self._cursor = t
        self.update()

    def _t_total(self):
        return self._stages[-1][2] if self._stages else 1.0

    def _x_of(self, t):
        pad = 8
        w = self.width() - 2 * pad
        return pad + w * t / self._t_total()

    def _t_of(self, x):
        pad = 8
        w = self.width() - 2 * pad
        return max(0.0, min(1.0, (x - pad) / w)) * self._t_total()

    def mousePressEvent(self, e):
        self._dragging = True
        self.cursor_changed.emit(self._t_of(e.position().x()))

    def mouseMoveEvent(self, e):
        if self._dragging:
            self.cursor_changed.emit(self._t_of(e.position().x()))

    def mouseReleaseEvent(self, e):
        self._dragging = False

    def paintEvent(self, e):
        if not self._stages:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        y0, h = 14, 26
        total = self._t_total()
        color_map = {"L1 输运": theme.STAGE_COLORS["L1"],
                     "交接": theme.STAGE_COLORS["handover"],
                     "L2 输运": theme.STAGE_COLORS["L2"]}
        for name, t0, t1 in self._stages:
            x0 = self._x_of(t0)
            x1 = max(self._x_of(t1), x0 + 4)  # handover 至少可见
            width = x1 - x0
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(color_map[name]))
            p.drawRoundedRect(int(x0), y0, int(width) - 2, h, 4, 4)
            f = QFont(p.font())
            f.setPointSize(9)
            p.setFont(f)
            if width >= 70:
                p.setPen(QColor("white"))
                p.drawText(int(x0) + 6, y0 + 17, name)
                p.setPen(QColor(theme.TEXT_DIM))
                p.drawText(int(x0) + 6, y0 + h + 16,
                           f"{(t1 - t0) * 1e3:g} ms")
            else:
                # 窄段(如交接):段内只留色块,名称+时长右对齐放在边界左侧
                p.setPen(QColor(theme.TEXT_DIM))
                p.drawText(int(x0) - 110, y0 + h + 16, 106, 14,
                           int(Qt.AlignRight),
                           f"{name} {(t1 - t0) * 1e3:g} ms")
        # 光标
        cx = self._x_of(self._cursor)
        p.setPen(QPen(QColor(theme.TEXT), 2))
        p.drawLine(int(cx), y0 - 6, int(cx), y0 + h + 20)
        p.end()
