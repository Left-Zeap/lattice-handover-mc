"""界面浅色主题：QSS 样式表与全局字体。

配色约定：背景 ``#f5f6f8``、卡片白底 6 px 圆角、主色 ``#2563eb``、
成功 ``#059669``、警告 ``#d97706``。
"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication


PRIMARY_COLOR = "#2563eb"
SUCCESS_COLOR = "#059669"
WARNING_COLOR = "#d97706"
BACKGROUND_COLOR = "#f5f6f8"

QSS = f"""
* {{
    font-family: "Microsoft YaHei", "SimHei", sans-serif;
    font-size: 13px;
    color: #1f2937;
}}
QMainWindow, QDialog {{
    background: {BACKGROUND_COLOR};
}}
QWidget#central {{
    background: {BACKGROUND_COLOR};
}}
QListWidget#nav {{
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 6px;
    outline: none;
}}
QListWidget#nav::item {{
    padding: 10px 12px;
    border-radius: 6px;
    margin: 2px 0;
}}
QListWidget#nav::item:selected {{
    background: {PRIMARY_COLOR};
    color: #ffffff;
    font-weight: bold;
}}
QListWidget#nav::item:hover:!selected {{
    background: #eff6ff;
}}
QGroupBox {{
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: #374151;
}}
QFrame#card {{
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
}}
QLabel#cardTitle {{
    color: #6b7280;
    font-size: 12px;
}}
QLabel#cardValue {{
    color: #111827;
    font-size: 17px;
    font-weight: bold;
}}
QLabel#hint {{
    color: #6b7280;
    font-size: 14px;
}}
QLabel#pageTitle {{
    font-size: 18px;
    font-weight: bold;
    color: #111827;
}}
QLabel#successText {{
    color: {SUCCESS_COLOR};
}}
QLabel#warningText {{
    color: {WARNING_COLOR};
}}
QPushButton {{
    background: {PRIMARY_COLOR};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 7px 16px;
    font-weight: bold;
}}
QPushButton:hover {{
    background: #1d4ed8;
}}
QPushButton:pressed {{
    background: #1e40af;
}}
QPushButton:disabled {{
    background: #bfdbfe;
    color: #eff6ff;
}}
QPushButton[secondary="true"] {{
    background: #ffffff;
    color: {PRIMARY_COLOR};
    border: 1px solid {PRIMARY_COLOR};
}}
QPushButton[secondary="true"]:hover {{
    background: #eff6ff;
}}
QPushButton[secondary="true"]:disabled {{
    color: #93c5fd;
    border-color: #bfdbfe;
}}
QPushButton[danger="true"] {{
    background: {WARNING_COLOR};
}}
QPushButton[danger="true"]:hover {{
    background: #b45309;
}}
QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {{
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    padding: 4px 6px;
    min-height: 20px;
}}
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
    border-color: {PRIMARY_COLOR};
}}
QTableWidget {{
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    gridline-color: #f3f4f6;
}}
QHeaderView::section {{
    background: #f9fafb;
    border: none;
    border-bottom: 1px solid #e5e7eb;
    padding: 6px;
    font-weight: bold;
}}
QProgressBar {{
    background: #e5e7eb;
    border: none;
    border-radius: 4px;
    height: 10px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {PRIMARY_COLOR};
    border-radius: 4px;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QToolTip {{
    background: #ffffff;
    border: 1px solid #d1d5db;
    color: #1f2937;
}}
QStatusBar {{
    background: #ffffff;
    border-top: 1px solid #e5e7eb;
}}
"""


def apply_theme(app: QApplication) -> None:
    """应用浅色主题和中文字体。"""
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei", 10))
    app.setStyleSheet(QSS)
