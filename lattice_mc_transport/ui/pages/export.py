"""结果导出页:勾选图表/数据,以带数据标签的文件名批量保存。"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QVBoxLayout, QWidget, QListWidget, QListWidgetItem,
)
from matplotlib.figure import Figure

from .. import theme
from ..widgets import Card

OUT_DEFAULT = "output/exports"


def _fmt_num(x: float) -> str:
    """91.8 -> '91p8'; 300 -> '300'。文件名安全。"""
    s = f"{x:g}".replace(".", "p").replace("-", "m")
    return s


def _single_tag(result: dict) -> str:
    s, cfg = result["summary"], result["cfg"]
    det = cfg["laser"]["d1_red_detuning_GHz"]
    p = cfg["geometry"]["l1"]["power_w"]
    T = s["final_temperature_uK"]
    S = s["final_survival"] * 100
    tag = (f'{s["species"]}_det{_fmt_num(det)}GHz_P{_fmt_num(p)}W'
           f'_S{_fmt_num(round(S, 1))}pct')
    if T == T:  # not NaN
        tag += f'_T{_fmt_num(round(T, 1))}uK'
    return tag


def _scan_tag(result: dict) -> str:
    d, p = result["detunings"], result["powers"]
    return (f"det{_fmt_num(d[0])}-{_fmt_num(d[-1])}GHz"
            f"_P{_fmt_num(p[0])}-{_fmt_num(p[-1])}W")


class ExportPage(QWidget):
    """每一项 = (标题, 是否可用(state), 导出函数(state, outdir) -> [保存的文件])"""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        left = Card("可导出内容")
        self.list = QListWidget()
        left.layout.addWidget(self.list)
        btn_row = QHBoxLayout()
        self.btn_all = QPushButton("全选")
        self.btn_all.setObjectName("ghostBtn")
        self.btn_none = QPushButton("全不选")
        self.btn_none.setObjectName("ghostBtn")
        btn_row.addWidget(self.btn_all)
        btn_row.addWidget(self.btn_none)
        btn_row.addStretch(1)
        left.layout.addLayout(btn_row)
        root.addWidget(left, 2)

        right = Card("导出")
        self.dir_label = QLabel(str(Path(OUT_DEFAULT).resolve()))
        self.dir_label.setObjectName("hint")
        self.dir_label.setWordWrap(True)
        btn_dir = QPushButton("选择目录…")
        btn_dir.setObjectName("ghostBtn")
        btn_dir.clicked.connect(self._choose_dir)
        self.btn_export = QPushButton("导出所选")
        self.btn_export.setObjectName("primaryBtn")
        self.btn_export.clicked.connect(self._export)
        self.status = QLabel("文件名将自动附带参数与结果标签。")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        right.layout.addWidget(QLabel("保存到"))
        right.layout.addWidget(self.dir_label)
        right.layout.addWidget(btn_dir)
        right.layout.addWidget(self.btn_export)
        right.layout.addWidget(self.status)
        right.layout.addStretch(1)
        root.addWidget(right, 1)

        self._dir = Path(OUT_DEFAULT)
        self._items = []  # (title, checker, exporter, QListWidgetItem)
        self._build_items()
        self.btn_all.clicked.connect(lambda: self._set_all(Qt.Checked))
        self.btn_none.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        state.single_finished.connect(lambda *_: self.refresh())
        state.scan_finished.connect(lambda *_: self.refresh())
        self.refresh()

    # ---- items ----
    def _build_items(self):
        self._items = [
            ("时序组合图 (温度+存活率, PNG)",
             lambda st: st.last_single is not None, self._export_timeseries),
            ("温度时序图 (PNG)",
             lambda st: st.last_single is not None, self._export_temp_ts),
            ("存活率时序图 (PNG)",
             lambda st: st.last_single is not None, self._export_survival_ts),
            ("单点 summary.json",
             lambda st: st.last_single is not None, self._export_summary),
            ("时间序列数据 timeseries.npz",
             lambda st: st.last_single is not None, self._export_npz),
            ("扫描温度热力图 (PNG)",
             lambda st: st.last_scan is not None, self._export_scan_T),
            ("扫描存活率热力图 (PNG)",
             lambda st: st.last_scan is not None, self._export_scan_S),
            ("扫描 scan_summary.json / npz",
             lambda st: st.last_scan is not None, self._export_scan_data),
        ]
        for title, _, _ in self._items:
            QListWidgetItem(title, self.list)

    def refresh(self):
        for row, (title, available, _) in enumerate(self._items):
            ok = available(self.state)
            # 重建带复选的 item
            new_item = QListWidgetItem(("✓ " if ok else "✗ ") + title)
            flags = new_item.flags() | Qt.ItemIsUserCheckable
            if not ok:
                flags &= ~Qt.ItemIsEnabled
            new_item.setFlags(flags)
            new_item.setCheckState(Qt.Checked if ok else Qt.Unchecked)
            self.list.takeItem(row)
            self.list.insertItem(row, new_item)

    def _set_all(self, checkstate):
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item.flags() & Qt.ItemIsEnabled:
                item.setCheckState(checkstate)

    def _choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择导出目录", str(self._dir))
        if d:
            self._dir = Path(d)
            self.dir_label.setText(str(self._dir.resolve()))

    # ---- export ----
    def _export(self):
        saved = []
        for row, (_, available, exporter) in enumerate(self._items):
            item = self.list.item(row)
            if (item.flags() & Qt.ItemIsEnabled
                    and item.checkState() == Qt.Checked):
                saved += exporter(self._dir)
        if saved:
            self.status.setText("已导出:\n" + "\n".join(p.name for p in saved))
        else:
            self.status.setText("没有可导出的内容(先运行计算或扫描)。")

    # ---- figure builders (与展示页解耦,导出时重建干净图) ----
    def _ts_figure(self, mode="both") -> Figure:
        d = self.state.last_single
        h, cfg = d["history"], d["cfg"]
        t_ms = np.asarray(h["time_s"]) * 1e3
        T = np.asarray(h["T_K"]) * 1e6
        S = np.asarray(h["survival"])
        s = d["summary"]
        fig = Figure(figsize=(7, 5.4 if mode == "both" else 3.2))
        if mode == "both":
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax1)
        else:
            ax1 = fig.add_subplot(111)
            ax2 = None
        if mode in ("both", "temp"):
            ax1.plot(t_ms, T, color=theme.ACCENT, lw=1.5)
            ax1.set_ylabel("T (µK)")
            ax1.set_title(
                f'{s["species"]}  det={cfg["laser"]["d1_red_detuning_GHz"]:g} GHz  '
                f'P={cfg["geometry"]["l1"]["power_w"]:g} W  '
                f'最终 S={100 * s["final_survival"]:.1f}%  '
                f'T={s["final_temperature_uK"]:.1f} µK')
            ax1.grid(True, color="#eceef2")
        else:
            ax1.plot(t_ms, S, color=theme.OK, lw=1.5)
            ax1.set_ylabel("存活率")
            ax1.set_ylim(-0.03, 1.03)
            ax1.set_xlabel("t (ms)")
            ax1.grid(True, color="#eceef2")
        if ax2 is not None:
            ax2.plot(t_ms, S, color=theme.OK, lw=1.5)
            ax2.set_ylabel("存活率")
            ax2.set_ylim(-0.03, 1.03)
            ax2.set_xlabel("t (ms)")
            ax2.grid(True, color="#eceef2")
        fig.tight_layout()
        return fig

    def _scan_figure(self, kind: str) -> Figure:
        d = self.state.last_scan
        dets, pows = d["detunings"], d["powers"]
        Z, title, unit = ((d["T"], "最终温度", "µK") if kind == "T"
                          else (d["S"] * 100, "最终存活率", "%"))
        fig = Figure(figsize=(7, 5.4))
        ax = fig.add_subplot(111)
        im = ax.imshow(Z, aspect="auto", origin="lower", cmap="viridis",
                       extent=[pows[0], pows[-1], dets[0], dets[-1]],
                       vmin=0 if kind == "S" else None,
                       vmax=100 if kind == "S" else None)
        fig.colorbar(im, ax=ax, label=unit)
        ax.set_xlabel("功率 (W)")
        ax.set_ylabel("D1 红失谐 (GHz)")
        ax.set_title(f"{title}  det {dets[0]:g}-{dets[-1]:g} GHz,"
                     f" P {pows[0]:g}-{pows[-1]:g} W")
        fig.tight_layout()
        return fig

    # ---- exporters ----
    def _export_timeseries(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"single_{_single_tag(self.state.last_single)}_timeseries.png"
        self._ts_figure("both").savefig(p)
        return [p]

    def _export_temp_ts(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"single_{_single_tag(self.state.last_single)}_temperature.png"
        self._ts_figure("temp").savefig(p)
        return [p]

    def _export_survival_ts(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"single_{_single_tag(self.state.last_single)}_survival.png"
        self._ts_figure("survival").savefig(p)
        return [p]

    def _export_summary(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"single_{_single_tag(self.state.last_single)}_summary.json"
        p.write_text(json.dumps(self.state.last_single["summary"],
                                ensure_ascii=False, indent=2), encoding="utf-8")
        return [p]

    def _export_npz(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"single_{_single_tag(self.state.last_single)}_timeseries.npz"
        np.savez_compressed(p, **self.state.last_single["history"])
        return [p]

    def _export_scan_T(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"scan_{_scan_tag(self.state.last_scan)}_temperature_heatmap.png"
        self._scan_figure("T").savefig(p)
        return [p]

    def _export_scan_S(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"scan_{_scan_tag(self.state.last_scan)}_survival_heatmap.png"
        self._scan_figure("S").savefig(p)
        return [p]

    def _export_scan_data(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        d = self.state.last_scan
        tag = _scan_tag(d)
        pj = outdir / f"scan_{tag}_summary.json"
        pj.write_text(json.dumps({
            "detunings_GHz": d["detunings"].tolist(),
            "powers_W": d["powers"].tolist(),
            "final_temperature_uK": d["T"].tolist(),
            "final_survival": d["S"].tolist(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        pn = outdir / f"scan_{tag}_results.npz"
        np.savez_compressed(pn, detunings_GHz=d["detunings"], powers_W=d["powers"],
                            final_T_uK=d["T"], final_survival=d["S"])
        return [pj, pn]
