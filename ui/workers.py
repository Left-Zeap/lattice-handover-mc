"""后台计算线程。

``CalcWorker`` 把耗时计算（单点全链路、二维扫描）放到 ``QThread`` 中，
通过信号向界面报告进度、成功或失败。取消是协作式的：计算库的
``progress`` 回调会周期性调用，取消请求在回调里抛 ``CancelledError``
中止计算；若计算结束时尚未回调（例如单点计算没有 progress 参数），
则在返回结果时按取消处理。
"""

from __future__ import annotations

import time
from typing import Callable

from PySide6.QtCore import QThread, Signal


class CancelledError(Exception):
    """用户请求取消时从 progress 回调中抛出。"""


class CalcWorker(QThread):
    """在后台线程执行 ``callable(progress) -> object``。"""

    progressed = Signal(str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        fn: Callable[[Callable[[str], None]], object],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._fn = fn
        self._cancel_requested = False
        self.elapsed_seconds: float | None = None

    def cancel(self) -> None:
        """请求取消；在下一次 progress 回调或计算返回时生效。"""
        self._cancel_requested = True

    def _progress(self, message: str) -> None:
        if self._cancel_requested:
            raise CancelledError()
        self.progressed.emit(message)

    def run(self) -> None:
        started = time.perf_counter()
        try:
            result = self._fn(self._progress)
        except CancelledError:
            self.elapsed_seconds = time.perf_counter() - started
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001 - 把任何计算错误转给界面
            self.elapsed_seconds = time.perf_counter() - started
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.elapsed_seconds = time.perf_counter() - started
        if self._cancel_requested:
            self.cancelled.emit()
            return
        self.finished.emit(result)
