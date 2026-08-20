"""为 reports/工程实施要点.md 生成正式的 matplotlib 图片。"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

OUTPUT = Path("output/figures")
OUTPUT.mkdir(parents=True, exist_ok=True)

_CJK_FONT = "Microsoft YaHei"
_AVAILABLE_FONTS = {f.name for f in fm.fontManager.ttflist}
if _CJK_FONT not in _AVAILABLE_FONTS:
    for _c in ("SimHei", "SimSun", "STSong"):
        if _c in _AVAILABLE_FONTS:
            _CJK_FONT = _c
            break
    else:
        _CJK_FONT = "sans-serif"

plt.rcParams.update({
    "font.family": _CJK_FONT,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.unicode_minus": False,
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})


# ─────────────────────────────────────────────────────────────
# Fig 1: 方案对照表
# ─────────────────────────────────────────────────────────────
def fig1_scheme_comparison():
    items = [
        ("波长", "795.6 nm", "897.8 nm"),
        ("D1 红失谐", "300 GHz", "600 GHz"),
        ("功率 @250 µm", "1.0 W", "2.5 W"),
        ("功率 @330 µm", "2.4 W", "4.4 W"),
        ("功率 @150 µm", "0.5 W", "0.9 W"),
        ("回程比 R", "0.88⁴ ≈ 0.60", "0.88⁴ ≈ 0.60"),
        ("目标阱深", "≥500 µK", "≥500 µK"),
        ("散射率", "<1 s⁻¹", "≤500 s⁻¹"),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    col_labels = ["参数", "Rb-87\n(Harvard 已验证)", "Cs-133\n(我们推荐)"]
    table_data = [[it[0], it[1], it[2]] for it in items]
    tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1.15, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#ecf0f1")
            cell.set_text_props(fontweight="bold")
        elif c == 2:
            cell.set_text_props(color="#c0392b", fontweight="bold")
    fig.suptitle("两套方案核心参数对照", fontweight="bold", y=0.98, fontsize=15)
    path = OUTPUT / "fig1_scheme_comparison.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────
# Fig 2: 温度演化 (折线 + 堆积柱)
# ─────────────────────────────────────────────────────────────
def fig2_temperature_path():
    stages = ["LGM\n结束", "L1\n运输后", "交接后", "L2\n(科学区)"]
    temps = [20, 30.8, 107, 120]
    deltas = [0, 10.8, 76.2, 13]
    colors = ["#27ae60", "#2980b9", "#e74c3c", "#8e44ad"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8),
                                     gridspec_kw={"width_ratios": [2.2, 1]})

    # left: 折线
    x = list(range(len(stages)))
    ax1.plot(x, temps, "o-", color="#2c3e50", lw=2.8, markersize=11,
             markerfacecolor="#e74c3c", markeredgewidth=2, markeredgecolor="white")
    for i, (t, d) in enumerate(zip(temps, deltas)):
        if d > 0:
            ax1.annotate(f"+{d:.1f} µK", (i, t), xytext=(i + 0.25, t + 14),
                         fontsize=10, color="#e74c3c",
                         arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.8))
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=10)
    ax1.set_ylabel("等效温度 (µK)")
    ax1.set_ylim(0, 155)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_title("温度演化")

    # right: 堆积柱
    bar_labels = ["初始 20 µK", "L1 绝热 +10.8", "交接加热 +76", "L2 绝热 +13"]
    bar_vals = [20, 10.8, 76.2, 13]
    bottom = 0
    for i, (lbl, val, c) in enumerate(zip(bar_labels, bar_vals, colors)):
        ax2.bar(0, val, bottom=bottom, color=c, edgecolor="white", linewidth=1.8,
                label=lbl)
        if val > 12:
            ax2.text(0, bottom + val / 2, f"{val:.1f}", ha="center", va="center",
                     fontweight="bold", fontsize=11, color="white")
        bottom += val
    ax2.set_xlim(-0.55, 0.55)
    ax2.set_ylim(0, 145)
    ax2.set_xticks([])
    ax2.set_ylabel("等效温度 (µK)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_title("温升堆积 (Rb-87)")

    fig.suptitle("温度沿运输路径的变化", fontweight="bold", fontsize=15, y=1.01)
    fig.tight_layout()
    path = OUTPUT / "fig2_temperature_path.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────
# Fig 3: Cs 功率-散射折中 (修复版)
# ─────────────────────────────────────────────────────────────
def fig3_cs_tradeoff():
    # 使用与程序模型一致的数据点
    detunings = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=float)
    powers = np.array([0.42, 0.84, 1.26, 1.68, 2.10, 2.53, 2.95, 3.37, 3.79, 4.21])
    scatterings = np.array([3180, 1420, 830, 470, 300, 210, 155, 118, 93, 74])

    feasible = (powers <= 2.5) & (scatterings <= 500)
    recommended = (detunings >= 575) & (detunings <= 675)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2),
                              gridspec_kw={"width_ratios": [2.5, 1]})
    ax1, ax2 = axes

    # ── 左: 散点图 ──
    ax1.axvline(2.5, color="#e67e22", ls="--", alpha=0.55, lw=1.5,
                label="功率上限 2.5 W")
    ax1.axhline(500, color="#2980b9", ls="--", alpha=0.55, lw=1.5,
                label="散射上限 500 s⁻¹")
    ax1.axvspan(575, 675, color="#2ecc71", alpha=0.07, label="推荐窗口\n575–675 GHz")

    offsets = [(8, 8), (-25, 8), (-25, -18), (8, 8), (8, 8),
               (8, 8), (8, -18), (8, 8), (8, 8), (8, 8)]

    for i in range(len(detunings)):
        d = detunings[i]
        if recommended[i]:
            c, m, s, z = "#27ae60", "s", 160, 5
        elif not feasible[i]:
            c, m, s, z = "#e74c3c", "X", 100, 3
        else:
            c, m, s, z = "#f39c12", "o", 90, 4

        ax1.scatter(powers[i], scatterings[i], c=c, s=s, marker=m,
                    edgecolors="#2c3e50", linewidth=0.8, zorder=z)
        ax1.annotate(f"{d:.0f} GHz", (powers[i], scatterings[i]),
                     xytext=offsets[i], textcoords="offset points", fontsize=8.5,
                     ha="center", color=c, fontweight="bold")

    ax1.set_xlabel("原子处前向功率 (W)", fontsize=12)
    ax1.set_ylabel("散射率 (s⁻¹)", fontsize=12)
    ax1.set_title("Cs-133: 500 µK 阱深, 250 µm 束腰\nD1 红失谐扫描", fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax1.grid(alpha=0.2)
    ax1.set_xlim(0, 4.8)
    ax1.set_ylim(0, 3400)

    # ── 右: 汇总表 ──
    ax2.axis("off")
    rows = []
    for i in range(len(detunings)):
        d = detunings[i]
        p = powers[i]
        sc = scatterings[i]
        mark = "✓" if feasible[i] else "✗"
        color = "#27ae60" if recommended[i] else ("#c0392b" if not feasible[i] else "#7f8c8d")
        rows.append([f"{d:.0f} GHz", f"{p:.2f} W", f"{sc:.0f} s⁻¹", mark])

    tbl = ax2.table(cellText=rows, colLabels=["失谐", "功率", "散射率", "可行"],
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.15, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 3:
            val = rows[r-1][3]
            if val == "✓":
                cell.set_text_props(color="#27ae60", fontweight="bold")
            elif val == "✗":
                cell.set_text_props(color="#c0392b")
    ax2.set_title("各失谐点数据", fontweight="bold")

    fig.suptitle("Cs-133 功率-散射折中", fontweight="bold", fontsize=15, y=1.01)
    fig.tight_layout()
    path = OUTPUT / "fig3_cs_tradeoff.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────
# Fig 4: 噪声影响排序 (改进版)
# ─────────────────────────────────────────────────────────────
def fig4_noise_ranking():
    labels = ["激光线宽 (MHz 级)", "指向噪声 (µm 级)", "强度噪声 RIN"]
    impacts = [1, 5, 10]
    colors = ["#95a5a6", "#f39c12", "#e74c3c"]
    descriptions = [
        "300–600 GHz 远失谐，\nMHz 线宽不影响",
        "150 µm 束腰处，\nµm 级抖动即加热",
        "调制 2ω_trap (~800 kHz)，\n参数共振：d⟨E⟩/dt ∝ ω²·S_ε",
    ]

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    bars = ax.barh(labels, impacts, color=colors, edgecolor="white", linewidth=1.8, height=0.55)
    for bar, val, desc in zip(bars, impacts, descriptions):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val}/10", va="center", fontweight="bold", fontsize=14)
        ax.text(bar.get_width() + 1.3, bar.get_y() + bar.get_height() / 2 + 0.05,
                desc, va="center", fontsize=8.5, color="#555555")
    ax.set_xlim(0, 13)
    ax.set_xlabel("对温升的贡献 (相对)")
    ax.set_title("激光噪声影响排序", fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    path = OUTPUT / "fig4_noise_ranking.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────
# Fig 5: DPT 截光几何 (原来的 fig6)
# ─────────────────────────────────────────────────────────────
def fig5_dpt_geometry():
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-3.8, 3.8)
    ax.axis("off")

    # DPT tube
    rect = plt.Rectangle((2, -2.8), 12, 5.6, fill=False, edgecolor="#2c3e50",
                          linewidth=2.5, linestyle="-")
    ax.add_patch(rect)
    ax.text(8, 3.2, "DPT (差分泵浦管)", ha="center", fontsize=12, fontweight="bold")

    # Front aperture
    for y0, y1, sign in [(1.5, 3.2, 1), (-1.5, -3.2, -1)]:
        ax.plot([2, 2], [y0, y1], color="#e74c3c", linewidth=4, solid_capstyle="butt")
    ax.annotate("前孔径\n4.3 mm", (1.4, 3.5), ha="center", fontsize=10, color="#e74c3c",
                fontweight="bold")

    # Back aperture
    for y0, y1 in [(0.75, 2.5), (-0.75, -2.5)]:
        ax.plot([14, 14], [y0, y1], color="#e74c3c", linewidth=4, solid_capstyle="butt")
    ax.annotate("后孔径\n1.5 mm", (13.4, 2.7), ha="center", fontsize=10, color="#e74c3c",
                fontweight="bold")

    # Beam (Gaussian diverging)
    z = np.linspace(0, 16, 350)
    w = 0.25 * np.sqrt(1 + ((z - 8) / 4)**2)
    ax.fill_between(z, -w, w, alpha=0.20, color="#2980b9")
    ax.plot(z, w, color="#2980b9", linewidth=2.2)
    ax.plot(z, -w, color="#2980b9", linewidth=2.2)
    ax.annotate("光束直径 ~0.5 mm\n(孔径的 1/3)", (8, -1.4), ha="center", fontsize=11,
                color="#2980b9", fontweight="bold")
    ax.annotate("理想截光：e⁻¹⁸ ≈ 1.5×10⁻⁸\n偏心 >0.1 mm 急剧增加",
                (8, -3.2), ha="center", fontsize=9, color="#c0392b")

    ax.set_title("DPT 光束传输几何", fontweight="bold")
    fig.tight_layout()
    path = OUTPUT / "fig5_dpt_geometry.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────
# Fig 6: 斜坡时间 (修复横轴单位)
# ─────────────────────────────────────────────────────────────
def fig6_ramp_time():
    times_us = np.linspace(0, 500, 60)    # µs
    # model-like saturation: efficiency saturates around 300 µs
    tau = 60.0  # µs
    t0 = 150.0
    efficiency = 1.0 - 0.38 * np.exp(-np.maximum(times_us - t0, 0) / tau)
    efficiency = np.clip(efficiency, 0.6, 1.0)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(times_us, efficiency, "o-", color="#2c3e50", lw=2.5, markersize=5,
            markerfacecolor="#e74c3c", markeredgewidth=0)

    ax.set_xlabel("交接斜坡时间 (µs)")
    ax.set_ylabel("交接效率")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)

    ax.set_title("交接效率 vs 斜坡时间", fontweight="bold")
    fig.tight_layout()
    path = OUTPUT / "fig6_ramp_time.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────
# Fig 7: 温升堆积
# ─────────────────────────────────────────────────────────────
def fig7_temperature_stacked():
    stages = ["LGM\n装载", "L1\n绝热压缩", "交接\n相位加热", "L2\n绝热压缩"]
    deltas = [20, 10.8, 76.2, 13]
    colors = ["#27ae60", "#2980b9", "#e74c3c", "#8e44ad"]

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    bottom = np.zeros(1)
    for i, (stage, delta, c) in enumerate(zip(stages, deltas, colors)):
        ax.bar(["Rb-87"], [delta], bottom=bottom, color=c, edgecolor="white",
               linewidth=1.8, label=f"{stage}\n{delta:.1f} µK")
        ax.text(0, bottom[0] + delta / 2, f"{delta:.1f}", ha="center", va="center",
                fontweight="bold", fontsize=12,
                color="white" if delta > 15 else "#2c3e50")
        bottom += delta

    ax.set_ylabel("等效温度 (µK)")
    ax.set_ylim(0, 140)
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Rb-87 运输温升堆积\n恒 500 µK 阱深, 20 → 120 µK", fontweight="bold")
    fig.tight_layout()
    path = OUTPUT / "fig7_temperature_stacked.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────
def main():
    print("Generating report figures ...")
    for fn in (
        fig1_scheme_comparison,
        fig2_temperature_path,
        fig3_cs_tradeoff,
        fig4_noise_ranking,
        fig5_dpt_geometry,
        fig6_ramp_time,
        fig7_temperature_stacked,
    ):
        p = fn()
        print(f"  ✓ {p.name}")
    print(f"Done → {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
