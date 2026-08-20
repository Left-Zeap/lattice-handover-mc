"""全链路参数表单。

按 原子与光路 / L1 时序 / conveyor 几何（可选）/ handover Monte Carlo
/ L2 段分组（扫描网格为二维扫描页专属参数，不在本表单）。默认值来自
``controllers.default_form_params``（即计算库
``data/l1_transport_defaults.json`` 的口径）；切换原子种类时自动替换
该物种的光路传输效率（Rb 1.0 / Cs 0.7）和参考失谐量。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import controllers


class NoWheelSpinBox(QSpinBox):
    """忽略滚轮的整数输入框：滚轮事件交还父级滚动区，避免误改数值。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # StrongFocus 保留键盘上下键调整，但不因滚轮经过而抢焦点。
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """忽略滚轮的浮点输入框：滚轮事件交还父级滚动区，避免误改数值。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelComboBox(QComboBox):
    """忽略滚轮的下拉框：滚轮事件交还父级滚动区，避免误改选择。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event) -> None:
        event.ignore()


class FilePathEdit(QWidget):
    """带浏览按钮的 CSV 路径输入框。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("留空则使用理想波形")
        button = QPushButton("浏览…")
        button.setProperty("secondary", True)
        button.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(button)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择实测控制波形", "", "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if path:
            self.edit.setText(path)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:
        self.edit.setText(value)


@dataclass(frozen=True)
class FieldSpec:
    """一个表单字段的控件配置。"""

    key: str
    label: str
    kind: str = "double"  # "double" | "int" | "combo" | "check" | "file"
    minimum: float = 0.0
    maximum: float = 1e9
    decimals: int = 2
    step: float = 1.0
    options: tuple[tuple[str, str], ...] = field(default_factory=tuple)


_GROUP_SPECS: tuple[tuple[str, tuple[FieldSpec, ...]], ...] = (
    (
        "原子与光路",
        (
            FieldSpec("atom_label", "原子种类", "combo",
                      options=(("Rb-87", "Rb-87"), ("Cs-133", "Cs-133"))),
            FieldSpec("detuning_ghz", "D1 红失谐 (GHz)", "double",
                      1.0, 5000.0, 1, 10.0),
            FieldSpec("source_power_w", "handover 端源端功率/分支 (W)",
                      "double", 0.0, 50.0, 3, 0.1),
            FieldSpec("delivery_efficiency", "源端到原子传输效率", "double",
                      0.01, 1.0, 3, 0.05),
            FieldSpec("retro_power_ratio", "回程功率比", "double",
                      0.0, 1.0, 8, 0.01),
            FieldSpec("target_depth_uK", "目标阱深 (µK)", "double",
                      1.0, 5000.0, 1, 10.0),
            FieldSpec("handover_waist_um", "handover 束腰 (µm)", "double",
                      10.0, 2000.0, 1, 10.0),
            FieldSpec("mot_atom_number", "MOT 原子数", "double",
                      1.0, 1e12, 0, 1e6),
            FieldSpec("pre_ramp_survival_fraction",
                      "MOT/compress/idle→L1 起点存活率", "double",
                      0.000001, 1.0, 4, 0.01),
            FieldSpec("initial_atom_number", "L1 初始原子数", "double",
                      1.0, 1e12, 0, 1e6),
            FieldSpec("initial_temperature_uK",
                      "L1 初始温度 (µK，静止晶格热平衡)", "double",
                      0.1, 10000.0, 1, 5.0),
            FieldSpec("occupied_lattice_sites", "占据格点数", "double",
                      1.0, 1e8, 0, 100.0),
            FieldSpec("include_gravity", "全时序重力（沿 -y）", "check"),
        ),
    ),
    (
        "L1 时序",
        (
            FieldSpec("l1_distance_m", "运输距离 (m)", "double",
                      0.01, 5.0, 3, 0.01),
            FieldSpec("l1_kinematic_profile", "理想速度轨迹", "combo",
                      options=(
                          ("minimum_jerk", "最小冲击 S 曲线（推荐）"),
                          ("trapezoid", "梯形速度（加速度阶跃）"),
                      )),
            FieldSpec("l1_acceleration_m_s2", "加速度 (m/s²)", "double",
                      1.0, 50000.0, 0, 100.0),
            FieldSpec("l1_maximum_velocity_m_s", "最大速度 (m/s)", "double",
                      0.1, 50.0, 2, 0.1),
            FieldSpec("l1_start_waist_um", "起点束腰 (µm)", "double",
                      10.0, 2000.0, 1, 10.0),
            FieldSpec("l1_time_points", "时间点数", "int", 4, 5001, 0, 50),
        ),
    ),
    (
        "实测控制波形 CSV（可选）",
        (
            FieldSpec("l1_control_waveform_path", "L1 运输波形", "file"),
            FieldSpec("handover_control_waveform_path", "handover 波形", "file"),
            FieldSpec("l2_control_waveform_path", "L2 运输波形", "file"),
        ),
    ),
    (
        "conveyor 几何（可选）",
        (
            FieldSpec("conveyor_enabled",
                      "启用 offset-waist 双束几何", "check"),
            FieldSpec("conveyor_waist_um", "conveyor 单束腰 (µm)", "double",
                      10.0, 2000.0, 1, 10.0),
            FieldSpec("conveyor_waist_separation_cm", "束腰间距 s (cm)",
                      "double", 0.0, 500.0, 1, 1.0),
        ),
    ),
    (
        "handover Monte Carlo",
        (
            FieldSpec("duration_us", "交接时长 (µs)", "double",
                      1.0, 100000.0, 0, 100.0),
            FieldSpec("particle_count", "轨迹数 N", "int",
                      1, 1000000, 0, 100),
            FieldSpec("time_step_us", "时间步长 (µs)", "double",
                      0.01, 100.0, 2, 0.05),
            FieldSpec("trace_points", "轨迹采样点数", "int", 2, 1001, 0, 2),
            FieldSpec("crossing_angle_deg", "L1/L2 交叉角 (°)", "double",
                      0.0, 90.0, 1, 0.5),
            FieldSpec("phase_mode", "交接相对相位口径", "combo",
                      options=(
                          ("random", "随机相位（多发次系综平均）"),
                          ("fixed", "固定相位（单发次）"),
                      )),
            FieldSpec("relative_phase_deg", "固定相对相位 (°，周期 180)",
                      "double", 0.0, 180.0, 1, 5.0),
            FieldSpec("cloud_axial_sigma_mm", "原子云轴向宽度 (mm)", "double",
                      0.01, 100.0, 2, 0.1),
            FieldSpec("seed", "随机种子", "int", 0, 2**31 - 1, 0, 1),
            FieldSpec("include_scattering", "包含散射反冲", "check"),
            FieldSpec("phase_space_continuity",
                      "阶段间连续传递相空间（单点/二维扫描）", "check"),
            FieldSpec("transport_method", "运输动力学", "combo",
                      options=(
                          ("analytic", "解析预算"),
                          ("monte_carlo", "Monte Carlo（轨迹级，慢）"),
                      )),
            FieldSpec("transport_time_step_us", "运输 MC 步长 (µs)",
                      "double", 0.01, 100.0, 2, 0.05),
            FieldSpec("compute_backend", "计算设备", "combo",
                      options=(
                          ("cpu", "CPU（默认）"),
                          ("gpu", "GPU（CuPy/CUDA，需安装）"),
                      )),
            FieldSpec("parallel_backend", "计算后端", "combo",
                      options=(
                          ("serial", "串行"),
                          ("process", "进程池（大网格推荐）"),
                      )),
            FieldSpec("worker_count", "进程数", "int", 1, 128, 0, 1),
        ),
    ),
    (
        "L2 段",
        (
            FieldSpec("l2_distance_m", "L2 运输距离 (m)", "double",
                      0.01, 5.0, 3, 0.01),
            FieldSpec("l2_kinematic_profile", "L2 理想速度轨迹", "combo",
                      options=(
                          ("minimum_jerk", "最小冲击 S 曲线（推荐）"),
                          ("trapezoid", "梯形速度（加速度阶跃）"),
                      )),
            FieldSpec("l2_acceleration_m_s2", "L2 加速度 (m/s²)", "double",
                      1.0, 50000.0, 0, 100.0),
            FieldSpec("l2_maximum_velocity_m_s", "L2 最大速度 (m/s)",
                      "double", 0.1, 50.0, 2, 0.1),
            FieldSpec("l2_end_waist_um", "L2 末端束腰 (µm)", "double",
                      10.0, 2000.0, 1, 10.0),
            FieldSpec("l2_time_points", "L2 时间点数", "int", 4, 5001, 0, 50),
        ),
    ),
)
# 注：扫描网格（失谐/功率范围与点数）是二维扫描页专属参数，
# 由 ``ui/pages/scan_page.py`` 自建控件组提供，不在共享表单内。


class ChainParameterForm(QWidget):
    """分组参数表单；``params()`` 返回控制层使用的扁平参数字典。"""

    def __init__(
        self,
        *,
        scan_preset: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._scan_preset = scan_preset
        self._widgets: dict[str, QWidget] = {}
        self._specs: dict[str, FieldSpec] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for group_title, specs in _GROUP_SPECS:
            group = QGroupBox(group_title)
            form = QFormLayout(group)
            form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            for spec in specs:
                widget = self._build_widget(spec)
                self._widgets[spec.key] = widget
                self._specs[spec.key] = spec
                form.addRow(spec.label, widget)
            layout.addWidget(group)
        reset_button = QPushButton("恢复默认参数")
        reset_button.setProperty("secondary", True)
        reset_button.clicked.connect(self.reset_defaults)
        layout.addWidget(reset_button)
        layout.addStretch(1)

        atom_combo = self._widgets["atom_label"]
        atom_combo.currentIndexChanged.connect(self._on_atom_changed)
        self._widgets["phase_space_continuity"].toggled.connect(
            self._on_phase_space_changed
        )
        self._widgets["phase_mode"].currentIndexChanged.connect(
            self._on_phase_mode_changed
        )
        for key in (
            "l1_control_waveform_path",
            "handover_control_waveform_path",
            "l2_control_waveform_path",
        ):
            self._widgets[key].edit.textChanged.connect(self._on_waveform_changed)
        self._apply_defaults()
        self._on_phase_space_changed()
        self._on_waveform_changed()
        self._on_phase_mode_changed()

    def _build_widget(self, spec: FieldSpec) -> QWidget:
        if spec.kind == "combo":
            combo = NoWheelComboBox()
            for value, text in spec.options:
                combo.addItem(text, value)
            return combo
        if spec.kind == "check":
            return QCheckBox()
        if spec.kind == "file":
            return FilePathEdit()
        if spec.kind == "int":
            spin = NoWheelSpinBox()
            spin.setRange(int(spec.minimum), int(spec.maximum))
            spin.setSingleStep(int(spec.step))
            return spin
        spin = NoWheelDoubleSpinBox()
        spin.setRange(spec.minimum, spec.maximum)
        spin.setDecimals(spec.decimals)
        spin.setSingleStep(spec.step)
        return spin

    def _apply_defaults(self) -> None:
        defaults = controllers.default_form_params(
            str(self._widgets["atom_label"].currentData()),
            scan_preset=self._scan_preset,
        )
        self.set_params(defaults)

    def set_params(self, params: dict[str, object]) -> None:
        for key, widget in self._widgets.items():
            if key not in params:
                continue
            value = params[key]
            if isinstance(widget, QComboBox):
                index = widget.findData(value)
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, FilePathEdit):
                widget.setText(str(value))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(type(widget.value())(value))

    def params(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, widget in self._widgets.items():
            if isinstance(widget, QComboBox):
                result[key] = widget.currentData()
            elif isinstance(widget, QCheckBox):
                result[key] = widget.isChecked()
            elif isinstance(widget, FilePathEdit):
                result[key] = widget.text()
            else:
                result[key] = widget.value()
        return result

    def reset_defaults(self) -> None:
        self._apply_defaults()

    def _on_phase_space_changed(self) -> None:
        checkbox = self._widgets["phase_space_continuity"]
        enabled = checkbox.isChecked()
        if enabled:
            combo = self._widgets["transport_method"]
            index = combo.findData("monte_carlo")
            combo.setCurrentIndex(index)
            combo.setEnabled(False)
            profile = self._widgets["l1_kinematic_profile"]
            profile.setCurrentIndex(profile.findData("minimum_jerk"))
            profile.setEnabled(False)
            l2_profile = self._widgets["l2_kinematic_profile"]
            l2_profile.setCurrentIndex(l2_profile.findData("minimum_jerk"))
            l2_profile.setEnabled(False)
        else:
            self._widgets["transport_method"].setEnabled(True)
            self._widgets["l1_kinematic_profile"].setEnabled(True)
            self._widgets["l2_kinematic_profile"].setEnabled(True)
        self._on_waveform_changed()

    def _on_phase_mode_changed(self) -> None:
        """固定相位口径下才允许编辑相位值。"""
        fixed = self._widgets["phase_mode"].currentData() == "fixed"
        self._widgets["relative_phase_deg"].setEnabled(fixed)

    def _on_waveform_changed(self) -> None:
        l1_measured = bool(self._widgets["l1_control_waveform_path"].text())
        handover_measured = bool(
            self._widgets["handover_control_waveform_path"].text()
        )
        l2_measured = bool(self._widgets["l2_control_waveform_path"].text())
        phase_continuous = self._widgets["phase_space_continuity"].isChecked()
        for key in (
            "l1_kinematic_profile",
            "l1_acceleration_m_s2",
            "l1_maximum_velocity_m_s",
        ):
            self._widgets[key].setEnabled(
                not l1_measured
                and not (phase_continuous and key == "l1_kinematic_profile")
            )
        self._widgets["duration_us"].setEnabled(not handover_measured)
        for key in ("l2_acceleration_m_s2", "l2_maximum_velocity_m_s"):
            self._widgets[key].setEnabled(not l2_measured)
        self._widgets["l2_kinematic_profile"].setEnabled(
            not l2_measured and not phase_continuous
        )
        conveyor = self._widgets["conveyor_enabled"]
        if l1_measured:
            conveyor.setChecked(False)
        conveyor.setEnabled(not l1_measured)

    def _on_atom_changed(self) -> None:
        """切换物种时更新传输效率和参考失谐量。"""
        atom_label = str(self._widgets["atom_label"].currentData())
        defaults = controllers.default_form_params(atom_label)
        self._widgets["delivery_efficiency"].setValue(
            float(defaults["delivery_efficiency"])
        )
        self._widgets["detuning_ghz"].setValue(
            float(defaults["detuning_ghz"])
        )
