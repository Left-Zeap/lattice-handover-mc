"""单点计算页：参数表单 + 晶格指标速算 + 单点全链路后台计算。"""

from __future__ import annotations

import time

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from continuous_loading.full_chain import FullChainPointSimulation

from .. import controllers
from ..state import AppState, format_runtime
from ..widgets.forms import ChainParameterForm
from ..widgets.result_cards import (
    MetricCardGrid,
    StageResultTable,
    format_value,
)
from ..workers import CalcWorker


class SinglePointPage(QWidget):
    """左侧参数表单，右侧操作按钮与结果卡片/表格。"""

    def __init__(
        self,
        state: AppState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._worker: CalcWorker | None = None

        splitter = QSplitter(Qt.Horizontal)
        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        self.form = ChainParameterForm()
        form_scroll.setWidget(self.form)
        splitter.addWidget(form_scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(10)

        button_row = QHBoxLayout()
        self.quick_button = QPushButton("晶格指标速算")
        self.quick_button.setProperty("secondary", True)
        self.quick_button.clicked.connect(self._run_quick_metrics)
        self.run_button = QPushButton("运行单点全链路")
        self.run_button.clicked.connect(self._run_full_chain)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setProperty("danger", True)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        button_row.addWidget(self.quick_button)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.cancel_button)
        right_layout.addLayout(button_row)

        self.progress_label = QLabel("就绪")
        self.progress_label.setObjectName("hint")
        right_layout.addWidget(self.progress_label)

        lattice_title = QLabel("晶格指标（速算）")
        lattice_title.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(lattice_title)
        self.lattice_cards = MetricCardGrid(columns=4)
        right_layout.addWidget(self.lattice_cards)

        chain_title = QLabel(
            "链路结果（L1 运输→handover→L2→科学区；初态为静止 L1 晶格热平衡系综）"
        )
        chain_title.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(chain_title)
        self.chain_cards = MetricCardGrid(columns=4)
        right_layout.addWidget(self.chain_cards)
        self.stage_table = StageResultTable()
        right_layout.addWidget(self.stage_table)
        right_layout.addStretch(1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setSizes((430, 850))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._state.single_point_updated.connect(self._show_last_result)
        if self._state.last_single_point is not None:
            self._show_last_result()

    def _set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)
        self.quick_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def _run_quick_metrics(self) -> None:
        params = self.form.params()
        started = time.perf_counter()
        try:
            metrics = controllers.lattice_quick_metrics(
                str(params["atom_label"]),
                float(params["detuning_ghz"]),
                float(params["source_power_w"]),
                float(params["delivery_efficiency"]),
                float(params["handover_waist_um"]),
                float(params["retro_power_ratio"]),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "参数无效", str(exc))
            return
        self.lattice_cards.set_metrics(
            [
                ("阱深", format_value(float(metrics["depth_uK"])), "µK"),
                ("散射率", format_value(float(metrics["scattering_rate_s"])), "s^-1"),
                ("径向阱频", format_value(float(metrics["radial_frequency_hz"]) / 1e3), "kHz"),
                ("轴向阱频", format_value(float(metrics["axial_frequency_hz"]) / 1e3), "kHz"),
                ("临界加速度", format_value(float(metrics["critical_axial_acceleration_m_s2"])), "m/s²"),
                ("激光波长", format_value(float(metrics["laser_wavelength_nm"]), 6), "nm"),
                ("波腹强度", format_value(float(metrics["antinode_intensity_w_m2"])), "W/m²"),
                ("原子处前向功率", format_value(float(metrics["forward_power_w"])), "W"),
            ]
        )
        self.progress_label.setText(
            f"晶格指标速算完成；运行时间 {format_runtime(time.perf_counter() - started)}"
        )

    def _run_full_chain(self) -> None:
        if self._worker is not None:
            return
        params = self.form.params()
        try:
            controllers.build_full_chain_inputs(params)
        except ValueError as exc:
            QMessageBox.warning(self, "参数无效", str(exc))
            return
        self._set_busy(True)
        self.progress_label.setText("正在计算单点全链路（Monte Carlo 中无进度回调，请稍候）…")
        self._worker = CalcWorker(lambda progress: controllers.run_single_point(params), self)
        self._worker.finished.connect(lambda result: self._on_finished(result, params))
        self._worker.failed.connect(lambda message: self._on_failed(message, params))
        self._worker.cancelled.connect(lambda: self._on_cancelled(params))
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.progress_label.setText("正在取消…")

    def _teardown_worker(self) -> float | None:
        worker = self._worker
        elapsed = None if worker is None else worker.elapsed_seconds
        self._worker = None
        self._set_busy(False)
        if worker is not None and worker.wait(5000):
            worker.deleteLater()
        return elapsed

    def _on_finished(self, result: FullChainPointSimulation, params: dict) -> None:
        elapsed = self._teardown_worker()
        self._state.set_single_point(result)
        self._state.add_history(
            "单点",
            controllers.summarize_single_point_params(params),
            "完成",
            payload=result,
            elapsed_seconds=elapsed,
        )
        self.progress_label.setText(
            f"单点全链路计算完成；运行时间 {format_runtime(elapsed)}"
        )

    def _on_failed(self, message: str, params: dict) -> None:
        elapsed = self._teardown_worker()
        self._state.add_history(
            "单点",
            controllers.summarize_single_point_params(params),
            f"失败：{message}",
            elapsed_seconds=elapsed,
        )
        self.progress_label.setText(
            f"计算失败：{message}；运行时间 {format_runtime(elapsed)}"
        )
        QMessageBox.warning(self, "计算失败", message)

    def _on_cancelled(self, params: dict) -> None:
        elapsed = self._teardown_worker()
        self._state.add_history(
            "单点",
            controllers.summarize_single_point_params(params),
            "已取消",
            elapsed_seconds=elapsed,
        )
        self.progress_label.setText(
            f"已取消；运行时间 {format_runtime(elapsed)}"
        )

    def _show_last_result(self) -> None:
        simulation = self._state.last_single_point
        if simulation is None:
            return
        point = simulation.point
        handover = simulation.l1_handover_simulation.handover_result
        transport_point = point.l1_handover.transport
        self.chain_cards.set_metrics(
            [
                ("handover 交接率", f"{handover.transfer_efficiency:.3f} ± {handover.transfer_standard_error:.3f}", ""),
                ("L1 末温", format_value(transport_point.final_temperature_uK), "µK"),
                ("科学区末温", format_value(point.l2_final_temperature_uK), "µK"),
                ("科学区总升温", format_value(point.science_total_temperature_rise_uK), "µK"),
                ("相对 MOT 总留存", format_value(point.final_retention_from_mot), ""),
                ("科学区原子数", format_value(point.science_atom_number), ""),
                ("峰值密度", format_value(point.science_peak_density_m3), "m^-3"),
                ("每格点原子数", format_value(point.science_atoms_per_site), ""),
                (
                    "阶段接口",
                    (
                        "相空间连续"
                        if simulation.interface_mode == "phase_space_continuous"
                        else "N,T 约化"
                    ),
                    "",
                ),
            ]
        )
        transport_trace = simulation.l1_handover_simulation.transport_trace
        pre_survival = transport_trace.pre_ramp_survival_fraction
        # 初态为静止 L1 晶格热平衡系综；MOT/compress/idle 只通过边界
        # 存活率进入，未显式传播。
        rows: list[tuple[str, str, str, str]] = [
            (
                "MOT/compress/idle→L1 起点（边界输入，未显式传播）",
                "—",
                format_value(pre_survival),
                "—",
            ),
            (
                "L1 初始（静止晶格热平衡系综）",
                format_value(transport_point.initial_temperature_uK),
                "—",
                format_value(transport_point.initial_atom_number),
            ),
        ]
        self.stage_table.set_rows(
            rows
            + [
                (
                    "L1 运输末",
                    format_value(transport_point.final_temperature_uK),
                    format_value(transport_point.total_retention_from_mot_fraction),
                    format_value(transport_point.final_atom_number),
                ),
                (
                    "handover 捕获",
                    format_value(point.l1_handover.final_temperature_uK),
                    format_value(point.l1_handover.final_retention_from_mot),
                    format_value(point.l1_handover.final_atom_number),
                ),
                (
                    "L2 末 / 科学区",
                    format_value(point.l2_final_temperature_uK),
                    format_value(point.final_retention_from_mot),
                    format_value(point.science_atom_number),
                ),
            ]
        )
