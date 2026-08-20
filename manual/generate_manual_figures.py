"""生成说明手册与汇报 PPT 共用的示意图和数据图（manual/OUTLINE.md §4b）。

运行方式（系统 Python，需已装 matplotlib/numpy）::

    C:\\Python314\\python.exe manual/generate_manual_figures.py

输出 12 张图到 ``manual/figures/``。所有上下标、希腊字母和负号一律用
matplotlib mathtext，避免 SimHei 缺字出现 □。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 让 ``continuous_loading`` 可导入（脚本位于 manual/ 子目录）。
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from continuous_loading.atomic import CS133, RB87  # noqa: E402
from continuous_loading.constants import BOLTZMANN  # noqa: E402
from continuous_loading.lattice import evaluate_lattice  # noqa: E402
from continuous_loading.linear_design import (  # noqa: E402
    LinearDesignInputs,
    analyze_detuning_power_lp,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

# 统一配色与字体口径（OUTLINE §6/§7）。
DEEP_BLUE = "#1F3B5C"
ACCENT_RED = "#C0392B"
NEUTRAL_GRAY = "#7F8C8D"
LIGHT_BLUE = "#D6E4F0"
LIGHT_RED = "#F5D5D0"

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": DEEP_BLUE,
        "axes.labelcolor": "black",
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "savefig.dpi": 150,
        "savefig.facecolor": "white",
    }
)


def _save(figure: plt.Figure, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"[ok] {path}")
    return path


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = LIGHT_BLUE,
    edgecolor: str = DEEP_BLUE,
    fontsize: float = 13,
    text_color: str = "black",
    bold: bool = False,
    linewidth: float = 1.6,
) -> FancyBboxPatch:
    """绘制圆角文本框，``xy`` 为左下角。"""
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.15,rounding_size=0.6",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        mutation_scale=1.0,
    )
    axis.add_patch(box)
    axis.text(
        xy[0] + width / 2.0,
        xy[1] + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold" if bold else "normal",
        linespacing=1.35,
    )
    return box


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = DEEP_BLUE,
    linewidth: float = 2.0,
    style: str = "-|>",
    connectionstyle: str = "arc3,rad=0.0",
    linestyle: str = "-",
) -> FancyArrowPatch:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=18,
        color=color,
        linewidth=linewidth,
        connectionstyle=connectionstyle,
        linestyle=linestyle,
        shrinkA=2,
        shrinkB=2,
    )
    axis.add_patch(arrow)
    return arrow


# ---------------------------------------------------------------------------
# 1. arch_schematic.png 系统架构示意
# ---------------------------------------------------------------------------
def fig_arch_schematic() -> None:
    figure, axis = plt.subplots(figsize=(13.5, 6.2))
    axis.set_xlim(0, 140)
    axis.set_ylim(0, 62)
    axis.axis("off")

    # MOT / 装载腔
    _box(
        axis,
        (3, 26),
        24,
        20,
        "MOT / 装载腔\nMOT→压缩→LGM\n$4\\times10^{6}$ 原子 @ 20 $\\mu$K",
        fontsize=12.5,
        bold=False,
    )
    axis.text(15, 51.5, "99 ms 准备", ha="center", fontsize=11.5, color=NEUTRAL_GRAY)

    # 科学腔（右端，含三个功能区）
    _box(axis, (112, 12), 25, 38, "", facecolor="#F4F7FA", linewidth=1.8)
    axis.text(
        124.5, 45.5, "科学腔", ha="center", fontsize=14, fontweight="bold", color=DEEP_BLUE
    )
    _box(axis, (114.5, 33), 20, 8, "reservoir 原子库", fontsize=11)
    _box(axis, (114.5, 22.5), 20, 8, "制备区", fontsize=11)
    _box(axis, (114.5, 12.5), 20, 8, "3240 位存储阵列", fontsize=11)

    # L1 晶格光束（约 4° 倾角，穿过 DPT）
    x0, y0 = 27, 36
    x1 = 101  # 交接点 x
    angle = math.radians(4.0)
    y1 = y0 + (x1 - x0) * math.tan(angle)
    # DPT 差分泵浦管（中段细长管）
    dpt_x0, dpt_x1 = 55, 78
    dpt_y0 = y0 + (dpt_x0 - x0) * math.tan(angle)
    dpt_y1 = y0 + (dpt_x1 - x0) * math.tan(angle)
    tube = patches.Polygon(
        [
            (dpt_x0, dpt_y0 - 3.2),
            (dpt_x1, dpt_y1 - 3.2),
            (dpt_x1, dpt_y1 + 3.2),
            (dpt_x0, dpt_y0 + 3.2),
        ],
        closed=True,
        facecolor="#EDF1F5",
        edgecolor=NEUTRAL_GRAY,
        linewidth=1.4,
        zorder=1,
    )
    axis.add_patch(tube)
    axis.text(
        (dpt_x0 + dpt_x1) / 2,
        (dpt_y0 + dpt_y1) / 2 - 7.2,
        "DPT 差分泵浦管",
        ha="center",
        fontsize=11.5,
        color=NEUTRAL_GRAY,
    )

    # L1 光束
    axis.plot([x0, x1], [y0, y1], color=DEEP_BLUE, linewidth=4.5, solid_capstyle="round", zorder=2)
    axis.text(
        41,
        y0 + (41 - x0) * math.tan(angle) + 4.6,
        "L1 晶格：39 cm / 50 ms\nwaist 330→250 $\\mu$m，$a \\approx 4000$ m/s$^{2}$",
        ha="center",
        fontsize=12,
        color=DEEP_BLUE,
        linespacing=1.4,
    )
    # 4° 倾角标注
    axis.plot([x0, x0 + 16], [y0, y0], color=NEUTRAL_GRAY, linewidth=1.2, linestyle="--")
    arc = patches.Arc((x0, y0), 22, 22, angle=0, theta1=0, theta2=4, color=ACCENT_RED, linewidth=1.6)
    axis.add_patch(arc)
    axis.text(x0 + 12.6, y0 - 3.0, "$\\approx 4^{\\circ}$", fontsize=12, color=ACCENT_RED)

    # 交接点
    axis.plot([x1], [y1], marker="o", markersize=11, color=ACCENT_RED, zorder=3)
    axis.annotate(
        "交接点：1 ms\n强度反向线性 ramp",
        xy=(x1, y1),
        xytext=(x1 - 8, y1 + 9.5),
        fontsize=12,
        color=ACCENT_RED,
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color=ACCENT_RED, lw=1.6),
    )

    # L2 光束
    x2, y2 = 112, 33.5
    axis.plot([x1, x2], [y1, y2], color=ACCENT_RED, linewidth=4.5, solid_capstyle="round", zorder=2)
    axis.text(
        (x1 + x2) / 2 + 1,
        (y1 + y2) / 2 - 8.6,
        "L2 晶格：17 cm / 21 ms\nwaist 250→150 $\\mu$m",
        ha="center",
        fontsize=12,
        color=ACCENT_RED,
        linespacing=1.4,
    )

    # 交付指标底注
    axis.text(
        70,
        5.5,
        "稳态交付：每 150 ms 一团 $2.5\\times10^{6}$ 原子 @ $\\approx$120 $\\mu$K，双晶格效率 $\\approx$60%",
        ha="center",
        fontsize=12.5,
        color="black",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F7F9FB", edgecolor=NEUTRAL_GRAY, lw=1.0),
    )
    axis.set_title("连续装载双光晶格输运系统架构（示意，未按比例）", fontsize=18, pad=10)
    figure.tight_layout()
    _save(figure, "arch_schematic.png")


# ---------------------------------------------------------------------------
# 2. timing_sequence.png 时序图
# ---------------------------------------------------------------------------
def fig_timing_sequence() -> None:
    # 分段（名称, 起点 ms, 终点 ms, 颜色）
    segments = [
        ("MOT", 0, 80, LIGHT_BLUE),
        ("压缩", 80, 87, "#B9D3E8"),
        ("idle", 87, 88, "#E5E5E5"),
        ("LGM\n装载", 88, 99, "#9FC2DE"),
        ("L1 运输", 99, 149, DEEP_BLUE),
        ("交接", 149, 150, ACCENT_RED),
        ("L2 运输", 150, 171, "#D98880"),
    ]
    durations = {"MOT": "80", "压缩": "7", "idle": "1", "LGM\n装载": "11", "L1 运输": "50", "交接": "1", "L2 运输": "21"}

    figure, axes = plt.subplots(
        3, 1, figsize=(13.5, 8.6), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.5, 1.5]}
    )

    # --- 分段条 ---
    ax = axes[0]
    for name, start, end, color in segments:
        width = end - start
        text_color = "white" if color in (DEEP_BLUE, ACCENT_RED) else "black"
        ax.barh(0, width, left=start, height=0.62, color=color, edgecolor=DEEP_BLUE, linewidth=1.0)
        if width >= 6:
            label = name if "\n" not in name else name.replace("\n", " ")
            ax.text(
                (start + end) / 2, 0,
                f"{label}\n{durations[name]} ms",
                ha="center", va="center", fontsize=11.5, color=text_color, linespacing=1.25,
            )
    # 窄分段在条外标注
    for name, start, end, dy in (("idle", 87, 88, 0.62), ("交接", 149, 150, 0.62)):
        ax.annotate(
            f"{name} {durations.get(name, '1')} ms",
            xy=((start + end) / 2, 0.31),
            xytext=((start + end) / 2 - 6, dy),
            fontsize=11,
            color=ACCENT_RED if name == "交接" else NEUTRAL_GRAY,
            arrowprops=dict(arrowstyle="-|>", color=NEUTRAL_GRAY, lw=1.1),
        )
    ax.set_ylim(-0.55, 1.15)
    ax.set_yticks([])
    ax.set_xlim(-2, 173)
    ax.set_title("单周期时序（总周期 171 ms，稳态节拍 150 ms）", fontsize=17, pad=8)
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)

    # --- v(t) 速度曲线 ---
    ax = axes[1]
    # L1: 99→101.03 ms 线性加速到 8.1 m/s 后巡航；交接期间两晶格同速；
    # L2: 150→150.6 ms 微调至 9.1 m/s，巡航后 168.7→171 ms 减速到 0。
    t = [0, 99, 101.03, 150, 150.6, 168.7, 171]
    v = [0, 0, 8.1, 8.1, 9.1, 9.1, 0]
    ax.plot(t, v, color=DEEP_BLUE, linewidth=2.6)
    ax.fill_between(t, v, color=DEEP_BLUE, alpha=0.10)
    ax.axvspan(149, 150, color=ACCENT_RED, alpha=0.18)
    ax.annotate(
        "L1 巡航 8.1 m/s\n($a \\approx 4000$ m/s$^{2}$)",
        xy=(125, 8.1), xytext=(118, 4.2), fontsize=12, color=DEEP_BLUE,
        arrowprops=dict(arrowstyle="-|>", color=DEEP_BLUE, lw=1.3),
    )
    ax.annotate(
        "交接：两晶格同速",
        xy=(149.5, 8.15), xytext=(137, 10.6), fontsize=11.5, color=ACCENT_RED,
        arrowprops=dict(arrowstyle="-|>", color=ACCENT_RED, lw=1.3),
    )
    ax.annotate(
        "L2 峰值 9.1 m/s\n减速进科学腔",
        xy=(169.8, 4.6), xytext=(152, 2.0), fontsize=12, color=ACCENT_RED,
        arrowprops=dict(arrowstyle="-|>", color=ACCENT_RED, lw=1.3),
    )
    ax.text(
        40, 8.6,
        "$v = \\lambda\\,\\Delta\\nu / 2$（AOM 频差线性扫描）\n10 m/s $\\leftrightarrow$ $\\Delta\\nu$ = 25.14 MHz",
        fontsize=12.5, color="black", ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F7F9FB", edgecolor=NEUTRAL_GRAY, lw=0.9),
    )
    ax.set_ylabel("$v(t)$（m/s）")
    ax.set_ylim(0, 12.5)
    ax.grid(alpha=0.25)

    # --- P(t) 功率跟随（归一化） ---
    ax = axes[2]
    # L1 支路：LGM 末升至 P0，L1 恒阱深 P∝w²（330→250 µm），交接线性降到 0
    t_l1 = [88, 99, 149, 150]
    p_l1 = [0.0, 1.0, (250.0 / 330.0) ** 2, 0.0]
    ax.plot(t_l1, p_l1, color=DEEP_BLUE, linewidth=2.6, label="L1 支路（源端，归一化）")
    # L2 支路：交接线性升到 P_HO，随后 P∝w²（250→150 µm），末端 0.36·P_HO
    p_ho = 0.55  # 归一化交接功率（示意）
    t_l2 = [149, 150, 171]
    p_l2 = [0.0, p_ho, 0.36 * p_ho]
    ax.plot(t_l2, p_l2, color=ACCENT_RED, linewidth=2.6, label="L2 支路（源端，归一化）")
    ax.axvspan(149, 150, color=ACCENT_RED, alpha=0.18)
    ax.annotate(
        "恒阱深功率跟随：$P \\propto w^2$",
        xy=(124, 0.78), xytext=(103, 1.18), fontsize=12.5, color=DEEP_BLUE,
        arrowprops=dict(arrowstyle="-|>", color=DEEP_BLUE, lw=1.3),
    )
    ax.annotate(
        "交接：强度反向线性 ramp（无冷却）",
        xy=(149.5, 0.30), xytext=(108, 0.42), fontsize=12, color=ACCENT_RED,
        arrowprops=dict(arrowstyle="-|>", color=ACCENT_RED, lw=1.3),
    )
    ax.annotate(
        "$P_{\\mathrm{L2,end}} = 0.36\\,P_{\\mathrm{HO}}$",
        xy=(171, 0.36 * p_ho), xytext=(154, 0.75), fontsize=12.5, color=ACCENT_RED,
        arrowprops=dict(arrowstyle="-|>", color=ACCENT_RED, lw=1.3),
    )
    ax.set_ylabel("$P(t)$（归一化）")
    ax.set_xlabel("时间（ms）")
    ax.set_ylim(0, 1.55)
    ax.legend(loc="upper right", fontsize=11.5, framealpha=0.95)
    ax.grid(alpha=0.25)

    figure.tight_layout(h_pad=2.2)
    _save(figure, "timing_sequence.png")


# ---------------------------------------------------------------------------
# 3. dipole_curves.png 真实计算 A(δ)、S(δ)
# ---------------------------------------------------------------------------
def fig_dipole_curves() -> None:
    """A(δ)=U0/P_src（µK/W）与 S(δ)=Γsc/P_src（s⁻¹/W）随 D1 红失谐变化。

    通过 ``evaluate_lattice`` 在前向功率 = 输运效率（0.7 W）处取值，
    直接得到“每源端瓦特”的阱深与散射率（工程线性化系数）。
    """
    inputs = LinearDesignInputs()  # waist 250 µm, 输运效率 0.70, 回程比 0.88^4
    detunings = np.linspace(100.0, 1000.0, 181)
    species = ((RB87, DEEP_BLUE, "Rb-87"), (CS133, ACCENT_RED, "Cs-133"))

    data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for atom, _, label in species:
        depths, scatterings = [], []
        for detuning in detunings:
            wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning)
            metrics = evaluate_lattice(
                atom,
                wavelength_nm,
                forward_power_w=inputs.delivery_efficiency,  # 每 1 W 源端功率
                waist_um=inputs.waist_um,
                retro_power_ratio=inputs.retro_power_ratio,
            )
            depths.append(metrics.depth_uK)
            scatterings.append(metrics.scattering_rate_s)
        data[label] = (np.array(depths), np.array(scatterings))

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    ax = axes[0]
    for _, color, label in species:
        ax.plot(detunings, data[label][0], color=color, linewidth=2.6, label=label)
    ax.set_xlabel("D1 红失谐 $\\delta$（GHz）")
    ax.set_ylabel("$A(\\delta) = U_0 / P_{\\mathrm{src}}$（$\\mu$K/W）")
    ax.set_title("阱深系数（waist 250 $\\mu$m，$R=0.88^4$）")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    ax = axes[1]
    for _, color, label in species:
        ax.plot(detunings, data[label][1], color=color, linewidth=2.6, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("D1 红失谐 $\\delta$（GHz）")
    ax.set_ylabel("$S(\\delta) = \\Gamma_{\\mathrm{sc}} / P_{\\mathrm{src}}$（s$^{-1}$/W）")
    ax.set_title("散射率系数（对数纵轴）")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right")

    # 工作点标注
    for atom, color, label, det0 in ((RB87, DEEP_BLUE, "Rb-87", 300.0), (CS133, ACCENT_RED, "Cs-133", 600.0)):
        wavelength_nm = atom.laser_wavelength_red_of_d1_nm(det0)
        metrics = evaluate_lattice(
            atom, wavelength_nm,
            forward_power_w=inputs.delivery_efficiency,
            waist_um=inputs.waist_um,
            retro_power_ratio=inputs.retro_power_ratio,
        )
        for axis, value, unit in (
            (axes[0], metrics.depth_uK, "$\\mu$K/W"),
            (axes[1], metrics.scattering_rate_s, "s$^{-1}$/W"),
        ):
            axis.plot([det0], [value], marker="o", markersize=9, color=color, zorder=5)
        axes[0].annotate(
            f"{label} 工作点\n$\\delta$={det0:.0f} GHz\n{metrics.depth_uK:.0f} $\\mu$K/W",
            xy=(det0, metrics.depth_uK),
            xytext=(det0 + 90, metrics.depth_uK * (1.35 if label == "Cs-133" else 0.72)),
            fontsize=11.5, color=color,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3),
        )
        axes[1].annotate(
            f"{label}\n{metrics.scattering_rate_s:.0f} s$^{{-1}}$/W",
            xy=(det0, metrics.scattering_rate_s),
            xytext=(det0 + 110, metrics.scattering_rate_s * (4.5 if label == "Cs-133" else 0.35)),
            fontsize=11.5, color=color,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3),
        )

    figure.suptitle(
        "标量偶极势与散射率系数（Grimm 远失谐，D1:D2 = 1:2，含反旋项）",
        fontsize=17, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    _save(figure, "dipole_curves.png")


# ---------------------------------------------------------------------------
# 4. conveyor_principle.png 传送带原理
# ---------------------------------------------------------------------------
def fig_conveyor_principle() -> None:
    figure, axis = plt.subplots(figsize=(12.5, 6.8))
    z = np.linspace(-1.6, 1.6, 800)  # 单位：λ/2
    shifts = (0.0, 1.0 / 6.0, 1.0 / 3.0)
    labels = ("$t_0$", "$t_1 > t_0$", "$t_2 > t_1$")
    offsets = (0.0, 1.35, 2.7)

    for shift, label, offset in zip(shifts, labels, offsets):
        potential = -(np.cos(np.pi * (z - shift)) ** 2)
        axis.plot(z, potential + offset, color=DEEP_BLUE, linewidth=2.4)
        axis.text(
            -1.62, offset + 0.32, label,
            fontsize=14, color=DEEP_BLUE, ha="right", va="center",
        )
        # 跟踪同一个势阱极小（z = shift）展示平移
        axis.plot(
            [shift], [offset - 1.0], marker="o", markersize=9, color=ACCENT_RED, zorder=5
        )
    # 极小点轨迹虚线
    axis.plot(
        [shifts[0], shifts[-1]], [offsets[0] - 1.0, offsets[-1] - 1.0],
        color=ACCENT_RED, linewidth=1.6, linestyle="--",
    )
    axis.annotate(
        "同一势阱随驻波平移",
        xy=(0.28, 1.65), xytext=(0.55, 1.45),
        fontsize=12.5, color=ACCENT_RED,
        arrowprops=dict(arrowstyle="-|>", color=ACCENT_RED, lw=1.4),
    )

    # λ/2 间距标注（底部曲线）
    axis.annotate(
        "", xy=(1.0, -1.28), xytext=(0.0, -1.28),
        arrowprops=dict(arrowstyle="<|-|>", color=NEUTRAL_GRAY, lw=1.5),
    )
    axis.text(0.5, -1.52, "$\\lambda/2$", ha="center", fontsize=13, color=NEUTRAL_GRAY)

    # 运动方向箭头
    axis.annotate(
        "", xy=(1.45, 3.55), xytext=(0.55, 3.55),
        arrowprops=dict(arrowstyle="-|>", color=ACCENT_RED, lw=2.6, mutation_scale=24),
    )
    axis.text(1.0, 3.72, "运动方向", ha="center", fontsize=13, color=ACCENT_RED)

    axis.text(
        0.0, 4.35,
        "$U(z,t) = -U_0 \\cos^2(kz - \\pi\\,\\Delta\\nu\\,t)$"
        "    $\\Rightarrow$    $v = \\lambda\\,\\Delta\\nu/2$",
        ha="center", fontsize=16, color="black",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F7F9FB", edgecolor=NEUTRAL_GRAY, lw=1.0),
    )
    axis.text(
        0.0, -2.05,
        "逆向两束频差 $\\Delta\\nu$ 由 AOM 线性扫频产生；"
        "$\\Delta\\nu$ 匀变 $\\Rightarrow$ 匀加速传送带",
        ha="center", fontsize=12.5, color=NEUTRAL_GRAY,
    )

    axis.set_xlim(-2.1, 2.1)
    axis.set_ylim(-2.4, 4.9)
    axis.axis("off")
    axis.set_title("光晶格传送带原理：移动驻波（F4）", fontsize=18, pad=6)
    figure.tight_layout()
    _save(figure, "conveyor_principle.png")


# ---------------------------------------------------------------------------
# 5. tilted_barrier.png 加速倾斜势垒 F5
# ---------------------------------------------------------------------------
def fig_tilted_barrier() -> None:
    beta = np.linspace(0.0, 0.999, 500)
    fraction = np.sqrt(1.0 - beta**2) - beta * np.arccos(beta)

    # Rb-87 工作点：λ=795.6 nm，U=500 µK，a≈4000 m/s²
    wavelength_m = 795.6e-9
    depth_j = 500e-6 * BOLTZMANN
    wave_number = 2.0 * math.pi / wavelength_m
    a_critical = depth_j * wave_number / RB87.mass_kg
    a_work = 4000.0
    beta_work = a_work / a_critical
    fraction_work = math.sqrt(1.0 - beta_work**2) - beta_work * math.acos(beta_work)

    figure, axis = plt.subplots(figsize=(10.5, 6.4))
    axis.plot(beta, fraction, color=DEEP_BLUE, linewidth=2.8)
    axis.fill_between(beta, fraction, color=DEEP_BLUE, alpha=0.08)

    axis.plot([beta_work], [fraction_work], marker="o", markersize=11, color=ACCENT_RED, zorder=5)
    axis.annotate(
        f"工作点：$a \\approx 4000$ m/s$^{{2}}$\n"
        f"$a_c = U k / m \\approx {a_critical:.0f}$ m/s$^{{2}}$\n"
        f"$\\beta = a/a_c \\approx {beta_work:.3f}$\n"
        f"$U_{{\\mathrm{{eff}}}}/U_0 \\approx {fraction_work:.2f}$",
        xy=(beta_work, fraction_work),
        xytext=(0.30, 0.62),
        fontsize=13, color=ACCENT_RED, linespacing=1.5,
        arrowprops=dict(arrowstyle="-|>", color=ACCENT_RED, lw=1.6),
    )
    axis.annotate(
        "$\\beta \\to 1$：局域极小消失\n（原子全部溢出）",
        xy=(0.985, 0.012), xytext=(0.70, 0.22),
        fontsize=12.5, color=NEUTRAL_GRAY,
        arrowprops=dict(arrowstyle="-|>", color=NEUTRAL_GRAY, lw=1.3),
    )

    axis.set_xlabel("$\\beta = a / a_c$（加速度 / 临界加速度）")
    axis.set_ylabel("$U_{\\mathrm{eff}} / U_0$")
    axis.set_title(
        "加速倾斜势垒（F5）：$U_{\\mathrm{eff}}/U_0 = \\sqrt{1-\\beta^2} - \\beta\\,\\arccos\\beta$"
        "\n（Rb-87，$\\lambda$=795.6 nm，$U_0$=500 $\\mu$K）",
        fontsize=15.5,
    )
    axis.set_xlim(0, 1.02)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    _save(figure, "tilted_barrier.png")


# ---------------------------------------------------------------------------
# 6. bound_fraction.png 热束缚比例 F6
# ---------------------------------------------------------------------------
def fig_bound_fraction() -> None:
    eta = np.linspace(0.0, 12.0, 600)
    bound = 1.0 - np.exp(-eta) * (1.0 + eta + eta**2 / 2.0)

    def f3(value: float) -> float:
        return 1.0 - math.exp(-value) * (1.0 + value + value**2 / 2.0)

    figure, axis = plt.subplots(figsize=(10.5, 6.4))
    axis.plot(eta, bound, color=DEEP_BLUE, linewidth=2.8)
    axis.fill_between(eta, bound, color=DEEP_BLUE, alpha=0.08)

    for eta0, color, note in (
        (5.0, ACCENT_RED, "工程口径下限\n$\\eta=5 \\Rightarrow 87.5\\%$"),
        (10.0, DEEP_BLUE, "$\\eta=10 \\Rightarrow 99.95\\%$"),
    ):
        axis.plot([eta0], [f3(eta0)], marker="o", markersize=11, color=color, zorder=5)
        axis.plot([eta0, eta0], [0, f3(eta0)], color=color, linewidth=1.2, linestyle=":")
        axis.plot([0, eta0], [f3(eta0), f3(eta0)], color=color, linewidth=1.2, linestyle=":")
    axis.annotate(
        "工程口径下限\n$\\eta=5 \\Rightarrow F_3 = 87.5\\%$",
        xy=(5.0, f3(5.0)), xytext=(5.6, 0.55),
        fontsize=13, color=ACCENT_RED, linespacing=1.5,
        arrowprops=dict(arrowstyle="-|>", color=ACCENT_RED, lw=1.6),
    )
    axis.annotate(
        "$\\eta=10 \\Rightarrow F_3 \\approx 99.95\\%$",
        xy=(10.0, f3(10.0)), xytext=(7.6, 0.30),
        fontsize=13, color=DEEP_BLUE,
        arrowprops=dict(arrowstyle="-|>", color=DEEP_BLUE, lw=1.6),
    )
    axis.text(
        5.5, 0.10,
        "注意：$U/(k_B T) \\geq \\alpha$ 是势深裕量判据，\n不是交接率（交接率须由轨迹 MC 判定，F9）",
        fontsize=12, color=NEUTRAL_GRAY,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F7F9FB", edgecolor=NEUTRAL_GRAY, lw=0.9),
    )

    axis.set_xlabel("$\\eta = U / (k_B T)$（阱深 / 热能）")
    axis.set_ylabel("$F_3(\\eta)$（束缚原子比例）")
    axis.set_title(
        "三维谐振子热束缚比例（F6）：$F_3(\\eta) = 1 - e^{-\\eta}\\,(1 + \\eta + \\eta^2/2)$",
        fontsize=15.5,
    )
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 1.03)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    _save(figure, "bound_fraction.png")


# ---------------------------------------------------------------------------
# 7. handover_potential.png 交接瞬态双晶格势
# ---------------------------------------------------------------------------
def fig_handover_potential() -> None:
    z = np.linspace(-2.2, 2.2, 1200)  # 单位：λ/2
    taus = (0.0, 0.5, 1.0)
    phases = ((0.0, "相位对齐 $\\varphi=0$"), (math.pi, "相位反相 $\\varphi=\\pi$（错开 $\\lambda/4$）"))

    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), sharex=True, sharey=True)

    for row, (phase, phase_label) in enumerate(phases):
        for col, tau in enumerate(taus):
            ax = axes[row][col]
            weight1 = 1.0 - tau
            weight2 = tau
            v1 = -weight1 * 0.5 * (1.0 + np.cos(2.0 * np.pi * z))
            v2 = -weight2 * 0.5 * (1.0 + np.cos(2.0 * np.pi * z + phase))
            total = v1 + v2
            if weight1 > 1e-9:
                ax.plot(z, v1, color=DEEP_BLUE, linewidth=1.6, linestyle="--", alpha=0.75,
                        label="L1 势" if (row == 0 and col == 1) else None)
            if weight2 > 1e-9:
                ax.plot(z, v2, color=ACCENT_RED, linewidth=1.6, linestyle="--", alpha=0.75,
                        label="L2 势" if (row == 0 and col == 1) else None)
            ax.plot(z, total, color="black", linewidth=2.4,
                    label="合成势" if (row == 0 and col == 1) else None)
            ax.axhline(0.0, color=NEUTRAL_GRAY, linewidth=0.8, linestyle=":")
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-1.18, 0.30)
            ax.grid(alpha=0.22)
            if row == 0:
                ax.set_title(f"$t = {tau:g}\\,\\tau_{{\\mathrm{{HO}}}}$（$\\tau_{{\\mathrm{{HO}}}}$=1 ms）", fontsize=14)
            if col == 0:
                ax.set_ylabel(f"{phase_label}\n$V(z)$（归一化）", fontsize=12.5)
            if row == 1:
                ax.set_xlabel("轴向位置 $z$（$\\lambda/2$）")

            # 关键物理标注
            if row == 1 and col == 1:
                ax.annotate(
                    "相位失配：中点合成势瞬间变平\n→ 能量再分配 / 相位加热",
                    xy=(0.35, -0.50), xytext=(-0.9, 0.16),
                    fontsize=11.5, color=ACCENT_RED, ha="center",
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT_RED, lw=1.3),
                )
            if row == 0 and col == 1:
                ax.legend(loc="lower center", fontsize=11, ncol=3, framealpha=0.95)
            if row == 0 and col == 0:
                ax.text(-2.05, 0.10, "L1 满载，L2=0", fontsize=11.5, color=DEEP_BLUE)
            if row == 0 and col == 2:
                ax.text(-2.05, 0.10, "L1=0，L2 满载", fontsize=11.5, color=ACCENT_RED)

    figure.suptitle(
        "交接瞬态：$V(z,t) = -U_1(1-\\bar{t})\\cos^2(kz) - U_2\\,\\bar{t}\\cos^2(kz+\\varphi/2)$，"
        "$\\bar{t}=t/\\tau_{\\mathrm{HO}}$",
        fontsize=15.5, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.945))
    _save(figure, "handover_potential.png")


# ---------------------------------------------------------------------------
# 8. pipeline_flowchart.png 程序数据流图
# ---------------------------------------------------------------------------
def fig_pipeline_flowchart() -> None:
    figure, axis = plt.subplots(figsize=(14.5, 8.6))
    axis.set_xlim(0, 148)
    axis.set_ylim(0, 88)
    axis.axis("off")

    # 主链五个阶段框
    stages = [
        (4, "装载\nloading_ramp.py\n(LGM+静止晶格)"),
        (34, "L1 运输\nl1_transport.py\n(transport.py)"),
        (64, "交接\nhandover.py"),
        (94, "L2 运输\nl2_transport.py"),
        (120, "科学区汇总\nfull_chain.py"),
    ]
    box_w, box_h, box_y = 24, 14, 52
    centers = []
    for x, text in stages:
        _box(axis, (x, box_y), box_w, box_h, text, fontsize=11.5)
        centers.append((x + box_w, box_y + box_h / 2, x, box_y + box_h / 2))

    transfers = [
        "捕获 $(N, T)$\n或 ParticleEnsemble",
        "$(N, T)$ + 相位空间\n（双轨一致）",
        "交接率 $\\eta_{\\mathrm{HO}}$\n+ 末温",
        "$(N, T)$ 全程\n留存 $S_{\\mathrm{total}}$",
    ]
    for index, ((x_right, y, _, _), label) in enumerate(zip(centers[:-1], transfers)):
        next_left = centers[index + 1][2]
        _arrow(axis, (x_right + 0.6, y), (next_left - 0.6, y))
        axis.text(
            (x_right + next_left) / 2, box_y + box_h + 3.0, label,
            ha="center", va="bottom", fontsize=10.5, color=NEUTRAL_GRAY, linespacing=1.3,
        )

    # 双轨制：解析腿 / 轨迹 MC（下方两条平行带）
    _box(
        axis, (8, 34), 84, 8.5,
        "解析腿（快，用于扫描）：transport.py / l1_transport.py 解析温升预算 F7 + 统计留存",
        facecolor="#EAF0F6", fontsize=11.5,
    )
    _box(
        axis, (8, 22), 84, 8.5,
        "轨迹 Monte Carlo（准，用于核验）：loading_ramp_mc.py / transport_mc.py / handover.py（F9 逐粒子判定）",
        facecolor="#F9E9E6", fontsize=11.5,
    )
    _arrow(axis, (46, box_y - 0.6), (46, 43.3), color=NEUTRAL_GRAY, linewidth=1.5)
    _arrow(axis, (46, 21.4), (46, 12.5), color=NEUTRAL_GRAY, linewidth=1.5, style="-")
    _box(
        axis, (8, 4), 84, 8.5,
        "双轨对照：解析腿给出趋势与上界，MC 给出工作点定量值（如交接净升温 5.8 $\\mu$K $\\ll$ 解析上界 76.2 $\\mu$K）",
        facecolor="#F4F7FA", fontsize=11.5,
    )

    # LP 初筛支路（右侧）
    _box(
        axis, (100, 34), 44, 11,
        "LP 可行域初筛\nlinear_design.py（F11 五约束）",
        facecolor="#EAF0F6", fontsize=11.5,
    )
    _box(
        axis, (100, 18), 44, 11,
        "交接率热力图 MC 复核\nhandover_map.py",
        facecolor="#F9E9E6", fontsize=11.5,
    )
    _arrow(axis, (122, 45.6), (122, box_y - 0.6), connectionstyle="arc3,rad=0.0")
    axis.text(126, 48.6, "推荐点\n(δ, P)", fontsize=10.5, color=NEUTRAL_GRAY, ha="left")
    _arrow(axis, (122, 33.4), (122, 29.6))
    _arrow(axis, (122, 17.4), (90, 12.5), connectionstyle="arc3,rad=-0.15", linestyle="--", color=NEUTRAL_GRAY, linewidth=1.4)
    axis.text(108, 14.6, "复核结论反馈设计", fontsize=10.5, color=NEUTRAL_GRAY, ha="center")

    # 入口与出口标注
    axis.text(16, 82.5, "配置 data/*.json → dataclass 默认 → CLI 覆盖", fontsize=11.5, color=NEUTRAL_GRAY)
    _arrow(axis, (16, 81.5), (16, 72.5), color=NEUTRAL_GRAY, linewidth=1.4)
    axis.text(132, 82.5, "输出：科学区 $(N, T)$、$S_{\\mathrm{total}}$、峰值密度", fontsize=11.5, color=NEUTRAL_GRAY)
    _arrow(axis, (132, 72.5), (132, 81.5), color=NEUTRAL_GRAY, linewidth=1.4)

    axis.set_title("程序数据流：装载 → L1 → 交接 → L2 → 科学区（双轨制 + LP 初筛支路）", fontsize=17, pad=8)
    figure.tight_layout()
    _save(figure, "pipeline_flowchart.png")


# ---------------------------------------------------------------------------
# 9. module_layers.png 模块分层架构图
# ---------------------------------------------------------------------------
def fig_module_layers() -> None:
    figure, axis = plt.subplots(figsize=(14.0, 8.8))
    axis.set_xlim(0, 140)
    axis.set_ylim(0, 90)
    axis.axis("off")

    layers = [
        ("CLI 入口层", 74, ["cli.py（13 个子命令）", "__main__.py"], "#DDE8F3"),
        (
            "编排层",
            56,
            ["full_chain.py", "l1_handover.py", "scenarios.py", "linear_design.py", "design_optimization.py", "*_scan.py / *_plots.py"],
            "#EAF0F6",
        ),
        (
            "阶段层",
            38,
            ["loading_ramp.py", "l1_transport.py", "transport.py", "handover.py", "l2_transport.py", "collisions.py"],
            "#EAF0F6",
        ),
        (
            "波形与接口层",
            20,
            ["control_waveforms.py", "phase_space.py", "conveyor_geometry.py"],
            "#F0F4F8",
        ),
        (
            "基础物理层",
            4,
            ["constants.py", "atomic.py", "dipole.py", "lattice.py"],
            "#F0F4F8",
        ),
    ]
    layer_x, layer_w, layer_h = 4, 100, 13
    for name, y, modules, color in layers:
        band = FancyBboxPatch(
            (layer_x, y), layer_w, layer_h,
            boxstyle="round,pad=0.2,rounding_size=0.8",
            facecolor=color, edgecolor=DEEP_BLUE, linewidth=1.5,
        )
        axis.add_patch(band)
        axis.text(
            layer_x + 1.8, y + layer_h / 2, name,
            ha="left", va="center", fontsize=13.5, fontweight="bold", color=DEEP_BLUE,
            rotation=0,
        )
        # 模块芯片
        chip_x = 22
        for module in modules:
            # CJK 字符按两个 ASCII 宽度估算芯片宽度。
            text_width = sum(2 if ord(char) > 127 else 1 for char in module)
            chip_w = max(text_width * 1.02, 10)
            if chip_x + chip_w > layer_x + layer_w - 2:
                break
            chip = FancyBboxPatch(
                (chip_x, y + 2.6), chip_w, layer_h - 5.2,
                boxstyle="round,pad=0.12,rounding_size=0.5",
                facecolor="white", edgecolor=NEUTRAL_GRAY, linewidth=1.0,
            )
            axis.add_patch(chip)
            # 等宽字体（DejaVu Sans Mono）不含中文字形，含中文的芯片用默认字体。
            is_ascii = all(ord(char) < 128 for char in module)
            axis.text(
                chip_x + chip_w / 2, y + layer_h / 2, module,
                ha="center", va="center", fontsize=10,
                family="monospace" if is_ascii else "sans-serif",
                color="black",
            )
            chip_x += chip_w + 1.6

    # 层间依赖箭头（自上而下调用）
    for y_top in (74, 56, 38, 20):
        _arrow(axis, (60, y_top - 0.8), (60, y_top - 3.6), color=NEUTRAL_GRAY, linewidth=1.6)
    axis.text(62.0, 71.6, "调用 / import", fontsize=10.5, color=NEUTRAL_GRAY)

    # GPU 支路（右侧竖条）
    _box(
        axis, (112, 20), 25, 49,
        "GPU 支路（可选）\n\ngpu_backend.py\ndevice_loop.py\nhandover_batch.py\ntransport_batch.py\nloading_ramp_batch.py\n\nCuPy/CUDA，\n默认回退 CPU",
        facecolor="#F9E9E6", edgecolor=ACCENT_RED, fontsize=10.5,
    )
    _arrow(
        axis, (111.4, 42), (104.6, 42),
        color=ACCENT_RED, linewidth=1.6, style="<|-|>",
    )
    axis.text(108, 45.2, "加速 MC", fontsize=10.5, color=ACCENT_RED, ha="center")

    axis.set_title("continuous_loading 包分层架构（下层不依赖上层）", fontsize=17, pad=8)
    figure.tight_layout()
    _save(figure, "module_layers.png")


# ---------------------------------------------------------------------------
# 10. lp_schematic.png LP 五约束可行域（真实 Cs 参数，单面板中文）
# ---------------------------------------------------------------------------
def fig_lp_schematic() -> None:
    inputs = LinearDesignInputs()  # Cs-133 默认口径
    result = analyze_detuning_power_lp(inputs, handover_times_ms=(1.0,))
    handover = result.handover_results[0]

    figure, axis = plt.subplots(figsize=(11.5, 7.0))
    chinese_labels = {
        "目标阱深下限": f"阱深下限 $U_0 \\geq$ {inputs.target_depth_uK:.0f} $\\mu$K",
        "最大散射率": f"散射率上界 $\\Gamma_{{\\mathrm{{sc}}}} \\leq$ {inputs.max_scattering_rate_s:.0f} s$^{{-1}}$",
        "最大源端功率": f"硬件功率上界 $P \\leq$ {inputs.max_source_power_w:.0f} W",
    }
    colors = ("#2563eb", "#7c3aed", "#d97706", "#dc2626", "#374151")

    for boundary, color in zip(handover.boundaries, colors):
        label = chinese_labels.get(boundary.label, boundary.label)
        axis.plot(
            boundary.detuning_ghz, boundary.source_power_w,
            color=color, linewidth=2.2, label=label,
        )

    for cell_index, cell in enumerate(handover.feasible_cells):
        xs = [point[0] for point in cell.polygon]
        ys = [point[1] for point in cell.polygon]
        axis.fill(
            xs, ys, color="#2E8B57", alpha=0.25, linewidth=0.0,
            label="LP 可行域（凸多边形）" if cell_index == 0 else None,
        )

    if handover.recommended is not None:
        point = handover.recommended
        axis.scatter(
            [point.detuning_ghz], [point.source_power_w],
            marker="*", s=340, color=ACCENT_RED, edgecolor="white",
            linewidth=1.0, zorder=6, label="LP 推荐点",
        )
        axis.annotate(
            f"推荐点：$\\delta$={point.detuning_ghz:.0f} GHz，"
            f"$P$={point.source_power_w:.2f} W\n"
            f"（代回原模型：$U_0$={point.depth_uK:.0f} $\\mu$K，"
            f"$\\Gamma_{{\\mathrm{{sc}}}}$={point.scattering_rate_s:.0f} s$^{{-1}}$）",
            xy=(point.detuning_ghz, point.source_power_w),
            xytext=(point.detuning_ghz + 60, point.source_power_w + 0.9),
            fontsize=12, color=ACCENT_RED, linespacing=1.45,
            arrowprops=dict(arrowstyle="-|>", color=ACCENT_RED, lw=1.4),
        )

    axis.set_xlabel("D1 红失谐 $\\delta$（GHz）")
    axis.set_ylabel("单支路源端功率 $P_{\\mathrm{src}}$（W）")
    axis.set_title(
        "失谐–功率 LP 可行域示意（F11）：Cs-133，$\\tau_{\\mathrm{HO}}$=1 ms，"
        "waist 250 $\\mu$m\n五约束半平面相交 → 凸多边形；目标 $J = P/P_{\\max} + 0.05\\,(\\delta-\\delta_{\\min})/(\\delta_{\\max}-\\delta_{\\min})$",
        fontsize=14.5,
    )
    axis.set_xlim(inputs.detuning_min_ghz, inputs.detuning_max_ghz)
    axis.set_ylim(0, inputs.max_source_power_w * 1.12)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=11, loc="upper left", framealpha=0.95)
    figure.tight_layout()
    _save(figure, "lp_schematic.png")


# ---------------------------------------------------------------------------
# 11. fig1_scheme_comparison_fixed.png 修复版方案对照表
# ---------------------------------------------------------------------------
def fig1_scheme_comparison_fixed() -> None:
    columns = ("参数", "Rb-87\n（Harvard 已验证）", "Cs-133\n（我们推荐）")
    rows = [
        ("波长", "795.6 nm", "897.8 nm"),
        ("D1 红失谐", "300 GHz", "600 GHz"),
        ("功率 @250 $\\mu$m", "1.0 W", "2.5 W"),
        ("功率 @330 $\\mu$m", "2.4 W", "4.4 W"),
        ("功率 @150 $\\mu$m", "0.5 W", "0.9 W"),
        ("回程比 $R$", "$0.88^{4} \\approx 0.60$", "$0.88^{4} \\approx 0.60$"),
        ("目标阱深", "$\\geq$500 $\\mu$K", "$\\geq$500 $\\mu$K"),
        ("散射率", "$< 1$ s$^{-1}$", "$\\leq 500$ s$^{-1}$"),
    ]

    figure, axis = plt.subplots(figsize=(12.5, 6.8))
    axis.axis("off")

    table = axis.table(
        cellText=[list(row) for row in rows],
        colLabels=list(columns),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.0, 2.15)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(NEUTRAL_GRAY)
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor(DEEP_BLUE)
            cell.set_text_props(color="white", fontweight="bold", fontsize=14.5)
        else:
            cell.set_facecolor("#F4F7FA" if row % 2 == 1 else "white")
            if col == 2:
                cell.set_text_props(color=ACCENT_RED, fontweight="bold")
            elif col == 0:
                cell.set_text_props(fontweight="bold")

    axis.set_title("两套方案核心参数对照", fontsize=18, fontweight="bold", pad=18)
    figure.tight_layout()
    _save(figure, "fig1_scheme_comparison_fixed.png")


# ---------------------------------------------------------------------------
# 12. fig5_dpt_geometry_fixed.png 修复版 DPT 几何
# ---------------------------------------------------------------------------
def fig5_dpt_geometry_fixed() -> None:
    figure, axis = plt.subplots(figsize=(13.5, 5.4))
    axis.set_xlim(0, 100)
    axis.set_ylim(-30, 30)
    axis.axis("off")

    # 纵向尺度：1 单位 = 0.1 mm（夸张显示）。孔径半径：前 2.15 mm，后 0.75 mm；
    # 光束 1/e² 半径 w = 0.25 mm（直径 ~0.5 mm）。
    scale = 10.0  # y 单位 / mm
    front_x, rear_x = 18.0, 82.0
    front_half = 2.15 * scale
    rear_half = 0.75 * scale

    # 管壁
    axis.plot([front_x, rear_x], [front_half, front_half], color=DEEP_BLUE, linewidth=2.2)
    axis.plot([front_x, rear_x], [-front_half, -front_half], color=DEEP_BLUE, linewidth=2.2)
    axis.plot([rear_x, 92], [front_half, front_half], color=DEEP_BLUE, linewidth=2.2, alpha=0.35)
    axis.plot([rear_x, 92], [-front_half, -front_half], color=DEEP_BLUE, linewidth=2.2, alpha=0.35)
    axis.plot([8, front_x], [front_half, front_half], color=DEEP_BLUE, linewidth=2.2, alpha=0.35)
    axis.plot([8, front_x], [-front_half, -front_half], color=DEEP_BLUE, linewidth=2.2, alpha=0.35)

    # 前后孔径光阑（红色挡板，中间开孔）
    for x, half in ((front_x, front_half), (rear_x, rear_half)):
        plate_top = 26.0
        axis.add_patch(patches.Rectangle((x - 0.9, half), 1.8, plate_top - half, color=ACCENT_RED, zorder=3))
        axis.add_patch(patches.Rectangle((x - 0.9, -plate_top), 1.8, plate_top - half, color=ACCENT_RED, zorder=3))

    # 高斯光束包络（焦点在管中央附近，w0=0.25mm，夸张瑞利长度）
    z = np.linspace(6, 94, 400)
    w0 = 0.25 * scale
    z0 = 50.0
    zr = 26.0  # 夸张的“瑞利长度”，仅为视觉
    envelope = w0 * np.sqrt(1.0 + ((z - z0) / zr) ** 2)
    axis.fill_between(z, -envelope, envelope, color=LIGHT_BLUE, alpha=0.85, zorder=1)
    axis.plot(z, envelope, color="#2E6DA4", linewidth=2.0, zorder=2)
    axis.plot(z, -envelope, color="#2E6DA4", linewidth=2.0, zorder=2)

    # 标注
    axis.annotate(
        "前孔径 4.3 mm", xy=(front_x, front_half + 0.4), xytext=(front_x - 3, 27.5),
        fontsize=13, color=ACCENT_RED, ha="center",
        arrowprops=dict(arrowstyle="-|>", color=ACCENT_RED, lw=1.4),
    )
    axis.annotate(
        "后孔径 1.5 mm", xy=(rear_x, rear_half + 0.4), xytext=(rear_x + 2, 27.5),
        fontsize=13, color=ACCENT_RED, ha="center",
        arrowprops=dict(arrowstyle="-|>", color=ACCENT_RED, lw=1.4),
    )
    axis.text(
        50, 13.5,
        "光束直径 $\\sim$0.5 mm（$w \\approx 0.25$ mm，孔径的 1/3）",
        ha="center", fontsize=13.5, color="#2E6DA4",
    )
    axis.text(
        50, -19.5,
        "理想截光（F13）：$L_{\\mathrm{clip}} = e^{-2a^{2}/w^{2}} \\approx e^{-18} \\approx 1.5\\times 10^{-8}$",
        ha="center", fontsize=14, color=ACCENT_RED,
    )
    axis.text(
        50, -26.5,
        "偏心 $> 0.1$ mm 时截光急剧增加",
        ha="center", fontsize=12.5, color=NEUTRAL_GRAY,
    )

    axis.set_title("DPT（差分泵浦管）光束传输几何", fontsize=18, pad=8)
    figure.tight_layout()
    _save(figure, "fig5_dpt_geometry_fixed.png")


# ---------------------------------------------------------------------------
def main() -> None:
    figures = (
        fig_arch_schematic,
        fig_timing_sequence,
        fig_dipole_curves,
        fig_conveyor_principle,
        fig_tilted_barrier,
        fig_bound_fraction,
        fig_handover_potential,
        fig_pipeline_flowchart,
        fig_module_layers,
        fig_lp_schematic,
        fig1_scheme_comparison_fixed,
        fig5_dpt_geometry_fixed,
    )
    for generator in figures:
        generator()
    print(f"全部完成，输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
