"""主窗口:侧边导航 + 页面堆栈。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from .state import AppState
from .pages.overview import OverviewPage
from .pages.single import SinglePage
from .pages.timeseries import TimeseriesPage
from .pages.scan import ScanPage
from .pages.export import ExportPage

PAGES = [
    ("概览", "输运流程与参数总览", OverviewPage),
    ("单点计算", "修改参数,运行一次完整 L1→交接→L2 模拟", SinglePage),
    ("时序可视化", "温度/存活率随时间变化,拖动光标查看任意时刻", TimeseriesPage),
    ("二维扫描", "失谐 × 功率网格扫描,输出热力图", ScanPage),
    ("结果导出", "带数据标签命名,批量导出图表与数据", ExportPage),
]


class MainWindow(QMainWindow):
    def __init__(self, state: AppState):
        super().__init__()
        self.setWindowTitle("光晶格连续输运 Monte Carlo")
        self.resize(1280, 820)

        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- sidebar ----
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(190)
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(0, 0, 0, 12)
        sb.setSpacing(2)
        title = QLabel("lattice-mc")
        title.setObjectName("appTitle")
        subtitle = QLabel("双光晶格连续输运\n6D Monte Carlo")
        subtitle.setObjectName("appSubtitle")
        sb.addWidget(title)
        sb.addWidget(subtitle)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.stack = QStackedWidget()
        self.title_label = QLabel()
        self.title_label.setObjectName("pageTitle")
        self.subtitle_label = QLabel()
        self.subtitle_label.setObjectName("pageSubtitle")

        for i, (name, sub, cls) in enumerate(PAGES):
            btn = QPushButton(name)
            btn.setObjectName("navBtn")
            btn.setCheckable(True)
            self.nav_group.addButton(btn, i)
            sb.addWidget(btn)
            page = cls(state)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            content = QWidget()
            content.setObjectName("contentRoot")
            cl = QVBoxLayout(content)
            cl.setContentsMargins(20, 18, 20, 18)
            cl.setSpacing(12)
            cl.addWidget(page)
            scroll.setWidget(content)
            self.stack.addWidget(scroll)
        sb.addStretch(1)
        self.nav_group.idClicked.connect(self._switch)

        # ---- content ----
        right = QVBoxLayout()
        right.setContentsMargins(20, 16, 20, 0)
        right.setSpacing(4)
        right.addWidget(self.title_label)
        right.addWidget(self.subtitle_label)
        right.addWidget(self.stack, 1)

        container = QWidget()
        root.addWidget(sidebar)
        root.addLayout(right, 1)
        container.setLayout(root)
        self.setCentralWidget(container)

        self.nav_group.button(0).setChecked(True)
        self._switch(0)

    def _switch(self, idx: int):
        self.stack.setCurrentIndex(idx)
        name, sub, _ = PAGES[idx]
        self.title_label.setText(name)
        self.subtitle_label.setText(sub)
