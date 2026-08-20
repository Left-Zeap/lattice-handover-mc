"""
激光参数计算工具的 Tkinter 图形界面。

程序提供两个输入面板：

1. “孔径 → 波长范围”：调用理想高斯光束聚焦公式；
2. “光阱最小功率”：调用简化二能级光学偶极阱模型。

界面只负责收集输入、调用 ``laser_formulas`` 中的计算函数并显示
结果。具体物理假设和模型局限记录在 ``laser_formulas.py`` 中。

运行方式
--------
在当前目录执行 ``python main.py``。

当前文件沿用原始程序结构，会在被导入时立即创建窗口并进入 Tk
事件循环。后续若要编写单元测试或让其他程序复用界面，建议将窗口
创建过程封装为 ``main()``，并增加 ``if __name__ == "__main__"``。
"""

import tkinter as tk
from tkinter import messagebox

from laser_formulas import wavelength_range_from_apertures, min_laser_power_for_trap


def calc_wavelength_range():
    """
    读取第一个面板的输入，计算并显示两个边界对应的波长。

    Tkinter ``Entry.get()`` 返回字符串，因此这里先用 ``float`` 转换。
    如果字符串不是合法数字，``float`` 会抛出 ``ValueError``，随后
    通过消息框提示用户。

    注意：底层函数目前不会自动排序两个结果，所以界面中显示的第一
    个数不一定小于第二个数；也尚未检查焦距为零等物理非法输入。
    """
    try:
        # 两个孔径输入均为直径，单位 mm。
        mot = float(entry_mot.get())
        dpt = float(entry_dpt.get())

        # 焦距与透镜处入射光束半径也使用 mm，保证公式内部量纲一致。
        f = float(entry_focal.get())
        wl = float(entry_waist_in.get())

        # 计算函数返回的两个数均以 nm 为单位。
        lower, upper = wavelength_range_from_apertures(mot, dpt, f, wl)

        # 保留两位小数以便界面阅读。这里的“范围”按原始输入顺序显示。
        label_range_result.config(text=f"波长范围：{lower:.2f} ~ {upper:.2f} nm")

    except ValueError:
        # 目前主要捕获字符串无法转换为浮点数的情况。
        # 焦距为零产生的 ZeroDivisionError 尚不在此捕获范围内。
        messagebox.showerror("输入错误", "请输入有效的数值")


def calc_min_power():
    """
    读取第二个面板的原子和光束参数，估算并显示最小激光功率。

    输入采用实验中常用的 THz、MHz、µm、µK 和 nm，底层函数负责
    转换为 SI 单位。若底层函数判断参数无效，会抛出 ``ValueError``，
    其错误信息会显示在消息框中。
    """
    try:
        # 原子跃迁参数：普通频率 f0 和自然线宽 Gamma/(2*pi)。
        f0_THz = float(entry_f0_trap.get())
        gamma_MHz = float(entry_gamma_trap.get())

        # eta 表示从输入功率到原子处有效功率的总效率。
        eta = float(entry_eta_trap.get())

        # 光束、原子温度和陷阱激光参数。
        waist_um = float(entry_waist_trap.get())
        temp_uK = float(entry_temp_trap.get())
        lam_nm = float(entry_lambda_trap.get())

        # 返回值单位为 W。物理模型的假设和限制见 laser_formulas.py。
        P_min = min_laser_power_for_trap(f0_THz, gamma_MHz, eta, waist_um, temp_uK, lam_nm)

        # 固定显示四位小数沿用原始界面行为。若结果远小于 0.1 mW，
        # 后续可改为根据数量级自动选择 W、mW 或 µW。
        label_power_result.config(text=f"所需最小激光功率：{P_min:.4f} W")

    except ValueError as e:
        # 同时处理输入字符串转换错误和底层函数主动报告的参数错误。
        messagebox.showerror("输入错误", f"请检查输入值\n{str(e)}")


# =============================================================================
# 主窗口
# =============================================================================
# Tk() 创建应用程序的顶层窗口。当前代码在模块顶层执行，因此运行
# main.py 或导入 main 都会立即创建窗口。
root = tk.Tk()
root.title("激光器参数计算工具")

# 固定初始窗口尺寸，并禁止用户缩放，避免简单 grid 布局被拉伸。
root.geometry("480x650")
root.resizable(False, False)


# =============================================================================
# 面板 1：孔径 → 波长范围
# =============================================================================
# LabelFrame 同时承担分组容器和标题显示。标题中直接列出当前使用的
# 理想高斯光束聚焦公式，便于用户了解计算依据。
frame1 = tk.LabelFrame(root, text="1. 孔径 → 波长范围 (w0 = λf / πwl)", padx=10, pady=10)
frame1.pack(padx=15, pady=10, fill="x")

# 每一行由一个文本标签和一个 Entry 输入框组成。
# sticky="w" 使标签左对齐，pady 增加各行的垂直间距。
tk.Label(frame1, text="MOT 孔径 (直径 mm)").grid(row=0, column=0, sticky="w", pady=2)
entry_mot = tk.Entry(frame1, width=12)
entry_mot.grid(row=0, column=1, pady=2)

tk.Label(frame1, text="DPT 孔径 (直径 mm)").grid(row=1, column=0, sticky="w", pady=2)
entry_dpt = tk.Entry(frame1, width=12)
entry_dpt.grid(row=1, column=1, pady=2)

tk.Label(frame1, text="焦距 f (mm)").grid(row=2, column=0, sticky="w", pady=2)
entry_focal = tk.Entry(frame1, width=12)
entry_focal.grid(row=2, column=1, pady=2)

tk.Label(frame1, text="入射束腰 wl (半径 mm)").grid(row=3, column=0, sticky="w", pady=2)
entry_waist_in = tk.Entry(frame1, width=12)
entry_waist_in.grid(row=3, column=1, pady=2)

# 点击按钮后，Tkinter 会调用 calc_wavelength_range；command 参数传入
# 函数对象本身，不能在这里写成 calc_wavelength_range()。
btn_range = tk.Button(frame1, text="计算波长范围", command=calc_wavelength_range, bg="#4CAF50", fg="white")
btn_range.grid(row=4, columnspan=2, pady=8)

# 结果标签创建时只有提示文字，计算成功后通过 config(text=...) 更新。
label_range_result = tk.Label(frame1, text="波长范围：", fg="blue", font=("Arial", 10, "bold"))
label_range_result.grid(row=5, columnspan=2, pady=2)


# =============================================================================
# 面板 2：光阱最小功率
# =============================================================================
frame2 = tk.LabelFrame(root, text="2. 光阱最小功率 (V0 > 1.5 kBT)", padx=10, pady=10)
frame2.pack(padx=15, pady=10, fill="x")

# 使用数据表描述六个输入项，避免为每一行重复编写几乎相同的控件
# 创建代码。每个元组包含“界面标签”和“保存 Entry 的全局变量名”。
trap_params = [
    ("原子共振频率 (THz)", "entry_f0_trap"),
    ("自然线宽 (MHz)", "entry_gamma_trap"),
    ("激光效率 η", "entry_eta_trap"),
    ("束腰半径 w0 (μm)", "entry_waist_trap"),
    ("原子温度 (μK)", "entry_temp_trap"),
    ("激光波长 (nm)", "entry_lambda_trap"),
]

for i, (label_text, var_name) in enumerate(trap_params):
    # i 同时作为 grid 的行号，使控件从上到下依次排列。
    tk.Label(frame2, text=label_text).grid(row=i, column=0, sticky="w", pady=2)
    entry = tk.Entry(frame2, width=16)
    entry.grid(row=i, column=1, pady=2)

    # 沿用原始实现：根据字符串动态创建全局变量，使回调函数可以通过
    # entry_f0_trap 等名称访问输入框。更易维护的方案是把 Entry 保存
    # 在字典或应用类的实例属性中。
    globals()[var_name] = entry

# len(trap_params) 恰好是输入项之后的下一行，因此新增/删除参数时，
# 按钮和结果标签会自动移动，不需要手动修改固定行号。
btn_power = tk.Button(frame2, text="计算最小功率", command=calc_min_power, bg="#FF9800", fg="white")
btn_power.grid(row=len(trap_params), columnspan=2, pady=8)

label_power_result = tk.Label(frame2, text="所需最小激光功率：", fg="red", font=("Arial", 10, "bold"))
label_power_result.grid(row=len(trap_params)+1, columnspan=2, pady=2)

# 底部提示说明当前函数如何处理失谐。这里的“阱深为正”应理解为
# 显示势能深度的绝对值，而不是说红、蓝失谐产生相同的陷阱几何。
tk.Label(root, text="提示：Δ = ω_laser - ω0，取绝对值保证阱深为正。",
         fg="gray", font=("Arial", 8)).pack(pady=5)

# 启动 Tk 事件循环。此调用会持续处理按钮点击、窗口重绘等事件，
# 直到用户关闭窗口；因此它也是一个阻塞调用。
root.mainloop()
