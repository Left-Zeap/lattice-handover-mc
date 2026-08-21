"""概览页：链路流程图、默认配置摘要和页面跳转。"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import controllers
from ..state import AppState
from ..widgets.result_cards import format_value


_CHAIN_STAGES = (
    ("MOT/compress/idle", "边界分布与前级存活率"),
    ("静止 L1 初态", "晶格热平衡系综\n默认 20 µK"),
    ("L1 运输", "39 cm / 45 ms\\n直径 660→500→约646 µm"),
    ("handover", "1 ms 三维轨迹\nMonte Carlo"),
    ("L2 运输", "17 cm / 21 ms\\n直径 约646→300 µm"),
    ("科学区", "原子库密度\n总升温与留存"),
)


class OverviewPage(QWidget):
    """链路流程卡片 + 默认配置摘要 + 跳页按钮。"""

    def __init__(
        self,
        state: AppState,
        navigate: Callable[[int], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._navigate = navigate

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("连续装载双光晶格计算平台")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "静止 L1 晶格热平衡初态→L1运输→handover→L2→科学区定量计算："
            "参数设置 → 后台计算 → 时序/热图可视化 → 结果导出。"
        )
        subtitle.setObjectName("hint")
        layout.addWidget(subtitle)

        flow = QHBoxLayout()
        flow.setSpacing(6)
        for index, (stage, detail) in enumerate(_CHAIN_STAGES):
            card = QFrame()
            card.setObjectName("card")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            name = QLabel(stage)
            name.setAlignment(Qt.AlignCenter)
            name.setStyleSheet("font-weight: bold; font-size: 15px;")
            desc = QLabel(detail)
            desc.setAlignment(Qt.AlignCenter)
            desc.setObjectName("cardTitle")
            card_layout.addWidget(name)
            card_layout.addWidget(desc)
            flow.addWidget(card, 1)
            if index < len(_CHAIN_STAGES) - 1:
                arrow = QLabel("→")
                arrow.setStyleSheet("color: #2563eb; font-size: 18px;")
                flow.addWidget(arrow)
        layout.addLayout(flow)

        summary = QFrame()
        summary.setObjectName("card")
        summary_layout = QVBoxLayout(summary)
        summary_title = QLabel("默认配置摘要（Rb-87，取自 data/l1_transport_defaults.json）")
        summary_title.setStyleSheet("font-weight: bold;")
        summary_layout.addWidget(summary_title)
        defaults = controllers.default_form_params("Rb-87")
        lines = (
            f"工作点：失谐 {defaults['detuning_ghz']:g} GHz，"
            f"L1/L2 固定源功率/分支 {defaults['source_power_w']:g} W，"
            f"handover 半径 {defaults['handover_waist_um']:g} µm，"
            f"回程功率比 {defaults['retro_power_ratio']:g}",
            f"L1：距离 {defaults['l1_distance_m']:g} m，"
            f"起点直径 {defaults['l1_start_beam_diameter_um']:g} µm，"
            f"最小半径 {defaults['l1_minimum_waist_um']:g} µm @ "
            f"{defaults['l1_minimum_waist_position_m']:g} m，"
            f"加速度 {defaults['l1_acceleration_m_s2']:g} m/s²，"
            f"最大速度 {defaults['l1_maximum_velocity_m_s']:g} m/s，"
            f"初温 {defaults['initial_temperature_uK']:g} µK",
            f"计算边界：L1 起点（静止晶格热平衡系综，无装载阶段）；"
            f"MOT/compress/idle→L1 起点存活率 "
            f"{defaults['pre_ramp_survival_fraction']:.3f}",
            f"handover MC：N={int(defaults['particle_count'])}，"
            f"时长 {defaults['duration_us']:g} µs，"
            f"步长 {defaults['time_step_us']:g} µs，"
            f"交叉角 {defaults['crossing_angle_deg']:g}°",
            f"L2：距离 {defaults['l2_distance_m']:g} m，"
            f"最大速度 {defaults['l2_maximum_velocity_m_s']:g} m/s，"
            f"末端束腰 {defaults['l2_end_waist_um']:g} µm，"
            f"格点数 {format_value(float(defaults['occupied_lattice_sites']))}",
        )
        for line in lines:
            text = QLabel(line)
            text.setObjectName("hint")
            summary_layout.addWidget(text)
        layout.addWidget(summary)

        buttons = QHBoxLayout()
        entries = (
            ("单点计算", 1),
            ("时序可视化", 2),
            ("二维扫描", 3),
            ("云宽扫描", 4),
            ("结果导出", 5),
        )
        for text, page_index in entries:
            button = QPushButton(text)
            button.setProperty("secondary", True)
            button.clicked.connect(
                lambda checked=False, target=page_index: self._navigate(target)
            )
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        layout.addStretch(1)
