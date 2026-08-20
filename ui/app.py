"""QApplication 与主窗口：左侧导航 + 右侧页面堆栈。"""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QListWidget,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from .pages.cloud_sigma_page import CloudSigmaPage
from .pages.export_page import ExportPage
from .pages.overview_page import OverviewPage
from .pages.scan_page import ScanPage
from .pages.single_point_page import SinglePointPage
from .pages.timing_page import TimingPage
from .state import AppState
from .theme import apply_theme


_NAV_ITEMS = ("概览", "单点计算", "时序可视化", "二维扫描", "云宽扫描", "结果导出")


class MainWindow(QMainWindow):
    """主窗口：左侧 QListWidget 导航，右侧 QStackedWidget 五个页面。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("连续装载双光晶格计算平台")
        self.resize(1280, 800)

        self.state = AppState(self)
        central = QWidget()
        central.setObjectName("central")
        layout = QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self._nav = QListWidget()
        self._nav.setObjectName("nav")
        self._nav.addItems(_NAV_ITEMS)
        self._nav.setFixedWidth(170)
        layout.addWidget(self._nav)

        self._stack = QStackedWidget()
        self._stack.addWidget(OverviewPage(self.state, self.go_to))
        self._stack.addWidget(SinglePointPage(self.state))
        self._stack.addWidget(TimingPage(self.state))
        self._stack.addWidget(ScanPage(self.state))
        self._stack.addWidget(CloudSigmaPage(self.state))
        self._stack.addWidget(ExportPage(self.state))
        layout.addWidget(self._stack, 1)

        self.setCentralWidget(central)
        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._nav.setCurrentRow(0)
        self.statusBar().showMessage("就绪")

    def go_to(self, index: int) -> None:
        """切换到指定页面（供概览页跳页按钮使用）。"""
        self._nav.setCurrentRow(index)

    def page_count(self) -> int:
        return self._stack.count()


def create_main_window() -> MainWindow:
    """创建主窗口（调用方需已创建 QApplication）。"""
    return MainWindow()


def run_application(argv: list[str] | None = None) -> int:
    """创建 QApplication、应用主题并进入事件循环。

    设置环境变量 ``UI_AUTO_QUIT_MS`` 时按给定毫秒数自动退出，用于
    无显示环境下的冒烟验证。
    """
    app = QApplication(argv if argv is not None else sys.argv)
    apply_theme(app)
    window = create_main_window()
    window.show()
    auto_quit_ms = os.environ.get("UI_AUTO_QUIT_MS")
    if auto_quit_ms:
        QTimer.singleShot(int(auto_quit_ms), window.close)
    return app.exec()
