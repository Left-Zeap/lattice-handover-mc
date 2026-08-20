"""`python -m ui` 启动图形界面。"""
from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_CFG = "configs/paper_rb87.json"


def main():
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication

    from lattice_mc.config import load_config
    from . import theme
    from .app import MainWindow
    from .state import AppState

    theme.apply_mpl_theme()
    app = QApplication(sys.argv)
    # 选中系统中实际存在的第一个中文字体,避免界面出现方框
    families = set(QFontDatabase.families())
    for want in ("Microsoft YaHei UI", "Microsoft YaHei", "DengXian", "SimHei"):
        if want in families:
            app.setFont(QFont(want))
            break
    app.setStyleSheet(theme.APP_STYLESHEET)

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CFG
    try:
        cfg = load_config(cfg_path)
    except Exception:
        cfg = None
    if cfg is None:
        # 模板缺失时给出一个最小可用默认配置
        cfg = load_config(str(Path(__file__).resolve().parent.parent
                              / "configs" / "paper_rb87.json"))

    win = MainWindow(AppState(cfg))
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
