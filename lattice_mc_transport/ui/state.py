"""共享应用状态:当前配置、最近一次单点/扫描结果。"""
from __future__ import annotations
from PySide6.QtCore import QObject, Signal

class AppState(QObject):
    cfg_changed = Signal()
    single_finished = Signal(dict)   # {"summary":..., "history":..., "cfg":...}
    scan_finished = Signal(dict)     # {"detunings":..., "powers":..., "T":..., "S":..., "meta":...}

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg                 # 当前编辑的配置(JSON dict)
        self.last_single: dict | None = None
        self.last_scan: dict | None = None

    def set_cfg(self, cfg: dict):
        self.cfg = cfg
        self.cfg_changed.emit()

    def set_single(self, result: dict):
        self.last_single = result
        self.single_finished.emit(result)

    def set_scan(self, result: dict):
        self.last_scan = result
        self.scan_finished.emit(result)
