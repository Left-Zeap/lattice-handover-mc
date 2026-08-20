"""统一视觉主题:调色板、Qt 样式表、matplotlib 风格。"""
from __future__ import annotations

# ---- palette ----
BG = "#f4f5f7"          # 内容区底色
CARD = "#ffffff"        # 卡片
BORDER = "#e2e5ea"
TEXT = "#1f2937"
TEXT_DIM = "#6b7280"
ACCENT = "#2563eb"      # 主色(蓝)
ACCENT_DARK = "#1d4ed8"
SIDEBAR_BG = "#17202c"
SIDEBAR_FG = "#cbd5e1"
SIDEBAR_ACTIVE = "#2563eb"
DANGER = "#dc2626"
OK = "#059669"

# 时序阶段配色(L1 / handover / L2)
STAGE_COLORS = {"L1": "#2563eb", "handover": "#f59e0b", "handover_end": "#f59e0b", "L2": "#10b981"}

APP_STYLESHEET = f"""
QWidget {{
    font-family: "Microsoft YaHei", "Microsoft YaHei UI", "DengXian", "SimHei", sans-serif;
    font-size: 13px;
    color: {TEXT};
}}
QMainWindow, QWidget#contentRoot {{ background: {BG}; }}
QFrame#sidebar {{ background: {SIDEBAR_BG}; border: none; }}
QLabel#appTitle {{
    color: white; font-size: 15px; font-weight: 600; padding: 18px 16px 6px 16px;
}}
QLabel#appSubtitle {{
    color: {SIDEBAR_FG}; font-size: 11px; padding: 0 16px 16px 16px;
}}
QPushButton#navBtn {{
    background: transparent; color: {SIDEBAR_FG}; border: none;
    text-align: left; padding: 10px 18px; font-size: 13px;
    border-left: 3px solid transparent;
}}
QPushButton#navBtn:hover {{ background: #223047; color: white; }}
QPushButton#navBtn:checked {{
    background: #223047; color: white; border-left: 3px solid {SIDEBAR_ACTIVE};
    font-weight: 600;
}}
QLabel#pageTitle {{ font-size: 20px; font-weight: 600; }}
QLabel#pageSubtitle {{ color: {TEXT_DIM}; font-size: 12px; }}
QFrame#card {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
}}
QLabel#cardTitle {{ font-size: 14px; font-weight: 600; }}
QLabel#metricValue {{ font-size: 22px; font-weight: 700; color: {ACCENT_DARK}; }}
QLabel#metricName {{ color: {TEXT_DIM}; font-size: 12px; }}
QLabel#hint {{ color: {TEXT_DIM}; font-size: 12px; }}
QPushButton#primaryBtn {{
    background: {ACCENT}; color: white; border: none; border-radius: 6px;
    padding: 8px 18px; font-weight: 600;
}}
QPushButton#primaryBtn:hover {{ background: {ACCENT_DARK}; }}
QPushButton#primaryBtn:disabled {{ background: #9db8e8; }}
QPushButton#ghostBtn {{
    background: transparent; color: {ACCENT}; border: 1px solid {ACCENT};
    border-radius: 6px; padding: 7px 14px;
}}
QPushButton#ghostBtn:hover {{ background: #eaf1fe; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: white; border: 1px solid {BORDER}; border-radius: 5px;
    padding: 5px 8px; selection-background-color: {ACCENT};
}}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid {ACCENT}; }}
QProgressBar {{
    background: #e8eaee; border: none; border-radius: 4px; height: 8px;
    text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QTableWidget {{
    background: white; border: 1px solid {BORDER}; gridline-color: #eef0f3;
    border-radius: 6px;
}}
QHeaderView::section {{
    background: #f0f2f5; border: none; border-bottom: 1px solid {BORDER};
    padding: 6px; font-weight: 600;
}}
QListWidget {{ background: white; border: 1px solid {BORDER}; border-radius: 6px; }}
QCheckBox {{ spacing: 6px; }}
QScrollArea {{ border: none; }}
"""

def apply_mpl_theme():
    """全局 matplotlib 风格:与 Qt 主题一致的浅色卡片风。"""
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": ["Microsoft YaHei", "SimHei", "DengXian", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.facecolor": CARD,
        "axes.facecolor": CARD,
        "axes.edgecolor": BORDER,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#eceef2",
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.facecolor": CARD,
    })
