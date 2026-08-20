"""单点计算页:参数编辑、运行、数值结果展示。"""
from __future__ import annotations
import json

from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QVBoxLayout, QWidget, QGridLayout,
)

from lattice_mc.config import load_config, save_json

from ..workers import SingleWorker
from ..widgets import Card, MetricCard, ParamForm

TEMPLATE = "configs/paper_rb87.json"

SCHEMA = [
    ("基础", [
        ("species", "原子种类", "choice", ["rb87", "cs133"]),
        ("initial.n_atoms", "原子数", "int", ""),
        ("initial.temperature_uK", "初始温度", "float", "µK"),
        ("initial.seed", "随机种子", "int", ""),
    ]),
    ("激光", [
        ("laser.d1_red_detuning_GHz", "D1 红失谐", "float", "GHz"),
        ("laser.retro_power_ratio", "retro 功率比", "float", ""),
        ("laser.phase_offset_l1_rad", "L1 相位偏移", "float", "rad"),
        ("laser.phase_offset_l2_rad", "L2 相位偏移", "float", "rad"),
    ]),
    ("几何 / 时序", [
        ("geometry.handover_angle_deg", "交接夹角", "float", "deg"),
        ("geometry.l1.distance_m", "L1 输运距离", "float", "m"),
        ("geometry.l1.duration_s", "L1 时长", "float", "s"),
        ("geometry.l1.acceleration_m_s2", "L1 加速度", "float", "m/s²"),
        ("geometry.l1.power_w", "L1 功率", "float", "W"),
        ("geometry.l1.waist_um", "L1 waist(起点,终点)", "text", "µm"),
        ("geometry.handover.duration_s", "交接时长", "float", "s"),
        ("geometry.l2.distance_m", "L2 输运距离", "float", "m"),
        ("geometry.l2.duration_s", "L2 时长", "float", "s"),
        ("geometry.l2.acceleration_m_s2", "L2 加速度", "float", "m/s²"),
        ("geometry.l2.power_w", "L2 功率", "float", "W"),
        ("geometry.l2.waist_um", "L2 waist(起点,终点)", "text", "µm"),
    ]),
    ("数值设置", [
        ("simulation.dt_s", "积分步长 dt", "float", "s"),
        ("simulation.record_interval_s", "记录间隔", "float", "s"),
        ("simulation.enable_scattering", "启用散射", "bool", ""),
        ("simulation.survival.energy_factor", "存活能垒因子", "float", ""),
        ("simulation.survival.loss_grace_s", "丢失宽限", "float", "s"),
        ("simulation.survival.check_every_steps", "存活检查间隔", "int", "步"),
    ]),
]


class SinglePage(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._worker: SingleWorker | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        # 左:参数表单
        left_card = Card("参数")
        left = QVBoxLayout()
        self.form = ParamForm(SCHEMA, state.cfg)
        self.form.bind_base(state.cfg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.form)
        left.addWidget(scroll)

        ctrl = QHBoxLayout()
        self.backend_box = QComboBox()
        self.backend_box.addItems(["auto", "gpu", "cpu"])
        ctrl.addWidget(QLabel("后端"))
        ctrl.addWidget(self.backend_box)
        self.btn_load = QPushButton("载入模板")
        self.btn_load.setObjectName("ghostBtn")
        self.btn_save = QPushButton("保存配置…")
        self.btn_save.setObjectName("ghostBtn")
        ctrl.addWidget(self.btn_load)
        ctrl.addWidget(self.btn_save)
        left.addLayout(ctrl)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        left.addWidget(self.progress)
        self.btn_run = QPushButton("开始计算")
        self.btn_run.setObjectName("primaryBtn")
        left.addWidget(self.btn_run)
        left_card.layout.addLayout(left)
        root.addWidget(left_card, 3)

        # 右:结果
        right_card = Card("数值结果")
        grid = QGridLayout()
        grid.setSpacing(10)
        self.m_survival = MetricCard("最终存活率")
        self.m_temp = MetricCard("最终温度")
        self.m_tx = MetricCard("Tx")
        self.m_ty = MetricCard("Ty")
        self.m_tz = MetricCard("Tz")
        self.m_scatter = MetricCard("散射事件总数")
        grid.addWidget(self.m_survival, 0, 0)
        grid.addWidget(self.m_temp, 0, 1)
        grid.addWidget(self.m_tx, 1, 0)
        grid.addWidget(self.m_ty, 1, 1)
        grid.addWidget(self.m_tz, 1, 2)
        grid.addWidget(self.m_scatter, 0, 2)
        right_card.layout.addLayout(grid)
        self.status_label = QLabel("尚未运行。修改参数后点击「开始计算」。")
        self.status_label.setObjectName("hint")
        self.status_label.setWordWrap(True)
        right_card.layout.addWidget(self.status_label)
        right_card.layout.addStretch(1)
        root.addWidget(right_card, 2)

        self.btn_run.clicked.connect(self._run)
        self.btn_load.clicked.connect(self._load_template)
        self.btn_save.clicked.connect(self._save_cfg)
        state.cfg_changed.connect(self._on_cfg_changed)

    def _on_cfg_changed(self):
        self.form.set_cfg(self.state.cfg)
        self.form.bind_base(self.state.cfg)

    def _load_template(self):
        try:
            self.state.set_cfg(load_config(TEMPLATE))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "载入失败", str(e))

    def _save_cfg(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存配置", "configs/custom.json",
                                              "JSON (*.json)")
        if not path:
            return
        try:
            save_json(self.form.cfg(), path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "保存失败", str(e))

    def _run(self):
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            cfg = self.form.cfg()
        except ValueError as e:
            QMessageBox.warning(self, "参数格式错误", f"请检查输入:{e}")
            return
        self.state.set_cfg(cfg)
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("计算中…")
        self._worker = SingleWorker(cfg, self.backend_box.currentText(), self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_progress(self, done, total):
        self.progress.setValue(int(100 * done / max(total, 1)))

    def _on_done(self, payload):
        self.btn_run.setEnabled(True)
        self.progress.setValue(100)
        self.state.set_single(payload)
        s = payload["summary"]
        self.m_survival.set_value(f'{100 * s["final_survival"]:.2f} %')
        t = s["final_temperature_uK"]
        self.m_temp.set_value(f"{t:.1f} µK" if t == t else "—")
        tx, ty, tz = s["final_Txyz_uK"]
        self.m_tx.set_value(f"{tx:.1f} µK" if tx == tx else "—")
        self.m_ty.set_value(f"{ty:.1f} µK" if ty == ty else "—")
        self.m_tz.set_value(f"{tz:.1f} µK" if tz == tz else "—")
        self.m_scatter.set_value(f'{s["total_scatter_events"]:,}')
        self.status_label.setText(
            f'完成。后端 {s["backend"]},{s["species"]},'
            f'{s["n_atoms"]:,} 原子,波长 {s["laser_wavelength_nm"]:.1f} nm。'
            "「时序可视化」页已更新。")

    def _on_fail(self, msg):
        self.btn_run.setEnabled(True)
        self.status_label.setText(f"计算失败:{msg}")
        QMessageBox.critical(self, "计算失败", msg)
