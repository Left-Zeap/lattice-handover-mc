"""指标卡片网格与分阶段结果表格。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def format_value(value: float | None, digits: int = 4) -> str:
    """把数值格式化成合适有效数字的字符串；``None`` 显示为横线。"""
    if value is None:
        return "—"
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e5 or magnitude < 1e-2:
        return f"{value:.{max(digits - 1, 1)}e}"
    return f"{value:.{digits}g}"


class MetricCardGrid(QWidget):
    """白底圆角指标卡片网格，每张卡片为 标题/数值/单位。"""

    def __init__(self, columns: int = 4, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._columns = columns
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(8)
        self._layout.setVerticalSpacing(8)

    def set_metrics(self, metrics: list[tuple[str, str, str]]) -> None:
        """重建卡片；每项为 ``(标题, 数值文本, 单位)``。"""
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, (title, value_text, unit) in enumerate(metrics):
            card = QFrame()
            card.setObjectName("card")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 8, 10, 8)
            card_layout.setSpacing(2)
            title_label = QLabel(title)
            title_label.setObjectName("cardTitle")
            value_label = QLabel(
                f"{value_text} <small>{unit}</small>" if unit else value_text
            )
            value_label.setObjectName("cardValue")
            card_layout.addWidget(title_label)
            card_layout.addWidget(value_label)
            row, column = divmod(index, self._columns)
            self._layout.addWidget(card, row, column)


class StageResultTable(QTableWidget):
    """分阶段结果表：阶段 / 温度 µK / 相对 MOT 留存 / 原子数。"""

    HEADERS = ("阶段", "温度 (µK)", "相对 MOT 留存", "原子数")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionMode(QTableWidget.NoSelection)
        self.setMaximumHeight(190)

    def set_rows(self, rows: list[tuple[str, str, str, str]]) -> None:
        self.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column, text in enumerate(row):
                item = QTableWidgetItem(text)
                if column > 0:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(row_index, column, item)
