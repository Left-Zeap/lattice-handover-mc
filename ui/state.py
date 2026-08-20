"""跨页面共享的应用状态。

保存最近一次单点全链路结果、最近一次二维扫描结果和计算历史列表，
页面通过 Qt 信号订阅刷新。状态对象本身不做任何计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from PySide6.QtCore import QObject, Signal

from continuous_loading.full_chain import (
    FullChainPointSimulation,
    FullChainScanResult,
)


@dataclass
class HistoryEntry:
    """一条计算历史记录，``payload`` 保留完整结果对象用于导出。"""

    time_text: str
    kind: str
    summary: str
    status: str
    payload: object | None = None
    elapsed_seconds: float | None = None

    @staticmethod
    def now_text() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class AppState(QObject):
    """应用级共享状态；结果更新时发射对应信号。"""

    single_point_updated = Signal()
    scan_updated = Signal()
    history_updated = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.last_single_point: FullChainPointSimulation | None = None
        self.last_scan: FullChainScanResult | None = None
        self.history: list[HistoryEntry] = []
        self._figures: dict[str, object] = {}

    def register_figure(self, kind: str, figure: object) -> None:
        """登记功能页面当前显示的 figure（``"single_point"``/``"scan"``）。"""
        self._figures[kind] = figure

    def figure_for(self, kind: str) -> object | None:
        return self._figures.get(kind)

    def set_single_point(self, simulation: FullChainPointSimulation) -> None:
        self.last_single_point = simulation
        self.single_point_updated.emit()

    def set_scan(self, result: FullChainScanResult) -> None:
        self.last_scan = result
        self.scan_updated.emit()

    def add_history(
        self,
        kind: str,
        summary: str,
        status: str,
        payload: object | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        self.history.append(
            HistoryEntry(
                time_text=HistoryEntry.now_text(),
                kind=kind,
                summary=summary,
                status=status,
                payload=payload,
                elapsed_seconds=elapsed_seconds,
            )
        )
        self.history_updated.emit()


def format_runtime(seconds: float | None) -> str:
    """面向 UI/导出的统一墙钟时间格式。"""
    if seconds is None:
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.3f} s"
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)} min {remainder:.1f} s"
