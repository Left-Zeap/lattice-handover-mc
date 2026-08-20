"""结果导出页：计算历史表格 + JSON / CSV / PNG 导出。

历史条目保留完整结果对象：单点结果导出拼接时间轨迹 CSV，扫描结果
导出网格 CSV；JSON 用 ``dataclasses.asdict`` 全量导出；PNG 保存当前
预览图。默认文件名 ``output/ui_export_<时间戳>.*``。
"""

from __future__ import annotations

import csv
import dataclasses
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from continuous_loading.full_chain import FullChainPointSimulation, FullChainScanResult

from ..state import AppState, format_runtime
from ..widgets.plot_canvas import PlotCanvas


_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


def _json_default(obj: object) -> object:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class ExportPage(QWidget):
    """计算历史（时间/类型/参数摘要/状态）与三种格式导出。"""

    def __init__(
        self,
        state: AppState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        title = QLabel("计算历史与结果导出")
        title.setStyleSheet("font-weight: bold;")
        top_row.addWidget(title)
        top_row.addStretch(1)
        self.export_json_button = QPushButton("导出 JSON")
        self.export_csv_button = QPushButton("导出轨迹/网格 CSV")
        self.export_png_button = QPushButton("导出功能图 PNG")
        for button in (
            self.export_json_button,
            self.export_csv_button,
            self.export_png_button,
        ):
            button.setProperty("secondary", True)
            button.setEnabled(False)
            top_row.addWidget(button)
        layout.addLayout(top_row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ("时间", "类型", "参数摘要", "状态", "运行时间")
        )
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table, 2)

        preview_title = QLabel("选中结果预览")
        preview_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(preview_title)
        self.preview = PlotCanvas(rows=1, cols=2)
        layout.addWidget(self.preview, 3)

        self.export_json_button.clicked.connect(lambda: self._export("json"))
        self.export_csv_button.clicked.connect(lambda: self._export("csv"))
        self.export_png_button.clicked.connect(lambda: self._export("png"))

        self._state.history_updated.connect(self.refresh)

    def refresh(self) -> None:
        history = self._state.history
        self.table.setRowCount(len(history))
        for row, entry in enumerate(reversed(history)):
            for column, text in enumerate(
                (
                    entry.time_text,
                    entry.kind,
                    entry.summary,
                    entry.status,
                    format_runtime(entry.elapsed_seconds),
                )
            ):
                self.table.setItem(row, column, QTableWidgetItem(text))

    def _selected_entry(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        history = self._state.history
        index = len(history) - 1 - row
        if not 0 <= index < len(history):
            return None
        return history[index]

    def _on_selection_changed(self) -> None:
        entry = self._selected_entry()
        exportable = entry is not None and entry.payload is not None
        for button in (
            self.export_json_button,
            self.export_csv_button,
            self.export_png_button,
        ):
            button.setEnabled(exportable)
        if exportable:
            self._plot_preview(entry.payload)

    def _plot_preview(self, payload: object) -> None:
        axes = self.preview.make_axes()
        if isinstance(payload, FullChainPointSimulation):
            combined = payload.combined_trace
            axes[0].plot(combined.time_ms, combined.temperature_uK, color="#dc2626")
            axes[0].set_xlabel("时间 (ms)")
            axes[0].set_ylabel("温度 (µK)")
            axes[0].set_title("全链路温度")
            axes[0].grid(alpha=0.2)
            axes[1].plot(
                combined.time_ms, combined.retention_from_mot, color="#2563eb"
            )
            axes[1].set_xlabel("时间 (ms)")
            axes[1].set_ylabel("相对 MOT 留存率")
            axes[1].set_title("全链路留存率")
            axes[1].grid(alpha=0.2)
        elif isinstance(payload, FullChainScanResult):
            detunings = np.asarray(payload.detuning_ghz)
            powers = np.asarray(payload.source_power_w)
            for axis, field, title in (
                (axes[0], "science_total_temperature_rise_uK", "科学区总升温 (µK)"),
                (axes[1], "final_retention_from_mot", "相对 MOT 总留存"),
            ):
                matrix = np.asarray(getattr(payload, field), dtype=float)
                mesh = axis.pcolormesh(
                    detunings, powers, matrix, shading="nearest", cmap="viridis"
                )
                axis.set_xlabel("D1 红失谐 (GHz)")
                axis.set_ylabel("源端功率 (W)")
                axis.set_title(title)
                self.preview.figure.colorbar(mesh, ax=axis)
        self.preview.redraw()

    def _default_path(self, extension: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return _OUTPUT_DIR / f"ui_export_{stamp}.{extension}"

    def _ask_path(self, extension: str, filter_text: str) -> Path | None:
        default = self._default_path(extension)
        chosen, _ = QFileDialog.getSaveFileName(
            self, "导出结果", str(default), filter_text
        )
        return Path(chosen) if chosen else None

    def _live_figure_for(self, entry):
        """返回功能页面当时显示的 figure；非当前显示结果时返回 None。"""
        if entry is None or entry.payload is None:
            return None
        if entry.payload is self._state.last_single_point:
            return self._state.figure_for("single_point")
        if entry.payload is self._state.last_scan:
            return self._state.figure_for("scan")
        return None

    def _export(self, kind: str) -> None:
        entry = self._selected_entry()
        if entry is None or entry.payload is None:
            return
        note = ""
        try:
            if kind == "json":
                path = self._ask_path("json", "JSON 文件 (*.json)")
                if path is not None:
                    self._export_json(
                        entry.payload, path, entry.elapsed_seconds
                    )
            elif kind == "csv":
                path = self._ask_path("csv", "CSV 文件 (*.csv)")
                if path is not None:
                    self._export_csv(
                        entry.payload, path, entry.elapsed_seconds
                    )
            else:
                path = self._ask_path("png", "PNG 图片 (*.png)")
                if path is not None:
                    figure = self._live_figure_for(entry)
                    if figure is None:
                        # 更早的历史条目：回退到预览重画。
                        self._plot_preview(entry.payload)
                        figure = self.preview.figure
                        note = "\n（该条目非当前显示结果，PNG 为重绘预览图）"
                    figure.savefig(
                        path,
                        dpi=180,
                        facecolor="white",
                        metadata={
                            "RuntimeSeconds": (
                                ""
                                if entry.elapsed_seconds is None
                                else f"{entry.elapsed_seconds:.9g}"
                            )
                        },
                    )
            if path is not None:
                QMessageBox.information(
                    self, "导出完成", f"已保存到\n{path}{note}"
                )
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def _export_json(
        self,
        payload: object,
        path: Path,
        elapsed_seconds: float | None = None,
    ) -> None:
        document = {
            "type": (
                "single_point"
                if isinstance(payload, FullChainPointSimulation)
                else "scan"
            ),
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "runtime_seconds": elapsed_seconds,
            "runtime_display": format_runtime(elapsed_seconds),
            "result": dataclasses.asdict(payload),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def _export_csv(
        self,
        payload: object,
        path: Path,
        elapsed_seconds: float | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(("runtime_seconds", elapsed_seconds))
            writer.writerow(("runtime_display", format_runtime(elapsed_seconds)))
            writer.writerow(())
            if isinstance(payload, FullChainPointSimulation):
                combined = payload.combined_trace
                writer.writerow(
                    ("time_ms", "phase", "temperature_uK", "retention_from_mot")
                )
                for time_ms, phase, temperature, retention in zip(
                    combined.time_ms,
                    combined.phase,
                    combined.temperature_uK,
                    combined.retention_from_mot,
                ):
                    writer.writerow((time_ms, phase, temperature, retention))
            elif isinstance(payload, FullChainScanResult):
                writer.writerow(
                    (
                        "detuning_ghz",
                        "source_power_w",
                        "transport_feasible",
                        "handover_transfer_efficiency",
                        "science_total_temperature_rise_uK",
                        "final_retention_from_mot",
                        "science_peak_density_m3",
                    )
                )
                for power_index, power in enumerate(payload.source_power_w):
                    for detuning_index, detuning in enumerate(payload.detuning_ghz):
                        writer.writerow(
                            (
                                detuning,
                                power,
                                payload.transport_feasible[power_index][detuning_index],
                                payload.handover_transfer_efficiency[power_index][detuning_index],
                                payload.science_total_temperature_rise_uK[power_index][detuning_index],
                                payload.final_retention_from_mot[power_index][detuning_index],
                                payload.science_peak_density_m3[power_index][detuning_index],
                            )
                        )
            else:
                raise TypeError("该历史条目没有可导出的轨迹或网格数据")
