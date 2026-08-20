"""后台计算线程:单点与扫描,带进度/完成/出错信号。"""
from __future__ import annotations
import numpy as np
from PySide6.QtCore import QThread, Signal

class SingleWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, cfg: dict, backend: str, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._backend = backend

    def run(self):
        try:
            from lattice_mc.simulation.runner import run_single, history_arrays

            def on_progress(done, total):
                self.progress.emit(done, total)

            result = run_single(self._cfg, backend_name=self._backend,
                                progress=on_progress)
            arr = history_arrays(result["history"])
            last = result["history"][-1]
            cfg = self._cfg
            summary = {
                "backend": result["backend"],
                "species": cfg["species"],
                "d1_red_detuning_GHz": cfg["laser"]["d1_red_detuning_GHz"],
                "laser_wavelength_nm": result["response"].wavelength_m * 1e9,
                "n_atoms": cfg["initial"]["n_atoms"],
                "final_survival": last["survival"],
                "final_temperature_uK": last["T_K"] * 1e6,
                "final_Txyz_uK": [last["Tx_K"] * 1e6, last["Ty_K"] * 1e6,
                                  last["Tz_K"] * 1e6],
                "total_scatter_events": last["scatter_events_total"],
            }
            self.finished_ok.emit({
                "summary": summary, "history": arr, "cfg": cfg,
            })
        except Exception as e:  # noqa: BLE001 - 把错误带到 UI 层
            import traceback
            self.failed.emit("".join(traceback.format_exception_only(type(e), e)).strip())

class ScanWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, cfg: dict, scan_cfg: dict, backend: str, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._scan_cfg = scan_cfg
        self._backend = backend

    def run(self):
        try:
            from lattice_mc.backend import get_backend

            def on_progress(done, total):
                self.progress.emit(done, total)

            xp, resolved = get_backend(self._backend)
            if resolved == "gpu":
                from lattice_mc.scan import run_detuning_power_scan_batched
                res = run_detuning_power_scan_batched(
                    self._cfg, self._scan_cfg, xp, progress=on_progress)
            else:
                from lattice_mc.scan import run_detuning_power_scan
                res = run_detuning_power_scan(
                    self._cfg, self._scan_cfg, backend_name=self._backend)
            self.finished_ok.emit({
                "detunings": np.asarray(res["detunings_GHz"]),
                "powers": np.asarray(res["powers_W"]),
                "T": np.asarray(res["final_T_uK"]),
                "S": np.asarray(res["final_survival"]),
                "backend": resolved,
                "scan_cfg": self._scan_cfg,
                "cfg": self._cfg,
            })
        except Exception as e:  # noqa: BLE001
            import traceback
            self.failed.emit("".join(traceback.format_exception_only(type(e), e)).strip())
