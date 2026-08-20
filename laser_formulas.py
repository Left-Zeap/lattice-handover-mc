"""
激光参数计算模块。

本模块目前包含两个彼此独立的简化计算：

1. 使用理想高斯光束的薄透镜聚焦公式，根据两个给定半径反推波长；
2. 使用理想二能级原子的远失谐偶极势近似，估算达到指定热运动
   能标所需的激光功率。

重要说明
--------
这些函数是早期的公式计算原型，并不是对论文中完整连续装载装置的
数值模拟。特别是：

* ``wavelength_range_from_apertures`` 没有计算光束沿差分泵浦管的传播，
  也没有考虑瑞利长度、束腰位置、光束质量 M² 或孔径截断；
* ``min_laser_power_for_trap`` 只使用单条原子跃迁的二能级近似，没有
  同时考虑碱金属 D1/D2 线、动态极化率、偏振、超精细结构和散射率；
* 功率公式的数值系数沿用当前项目的原始实现。它需要在明确光强、
  束腰、线宽和效率的定义后，再与选定的理论公式逐项核对。

因此，本模块的输出适合进行数量级探索，不应在未经进一步校准和
实验验证的情况下直接用于器件选型或实验参数定型。
"""

import math


def wavelength_range_from_apertures(mot_diameter_mm, dpt_diameter_mm, focal_length_mm, input_waist_mm):
    """
    用理想高斯光束聚焦公式反推两个边界半径对应的波长。

    当前函数使用的关系为 ``w0 = wavelength * f / (pi * wl)``，其中
    ``w0`` 是理想聚焦束腰半径，``wl`` 是透镜处入射光束半径，
    ``f`` 是焦距。将两个输入直径的一半分别代入 ``w0``，即可得到
    两个对应波长。

    参数
    ----
    mot_diameter_mm : float
        第一个边界的直径，单位 mm。名称沿用原始 GUI 的“MOT 孔径”；
        它在数学上只是第一个直径，并不自动代表真实 MOT 光学孔径。
    dpt_diameter_mm : float
        第二个边界的直径，单位 mm。名称沿用原始 GUI 的“DPT 孔径”。
    focal_length_mm : float
        聚焦透镜焦距，单位 mm。
    input_waist_mm : float
        透镜处入射光束的半径，单位 mm。这里默认使用与高斯光束公式
        一致的束腰/光斑半径定义，通常是 1/e² 强度半径。

    返回
    ----
    tuple[float, float]
        两个边界半径对应的波长，单位 nm。当前实现按输入顺序返回，
        不会自动排序，因此第一个值不一定小于第二个值。

    局限
    ----
    论文中的 DPT 前、后孔径位于不同轴向位置，严格计算应传播高斯
    光束并分别检查两个位置的光斑，而不能简单把两个孔径都当成焦点
    束腰。本函数暂时保留原始算法，仅用于说明该公式的数值结果。
    """
    # GUI 输入的是直径，而聚焦公式使用半径，因此先除以 2。
    r_mot = mot_diameter_mm / 2.0
    r_dpt = dpt_diameter_mm / 2.0

    # 由 w0 = wavelength * f / (pi * wl) 移项得到
    # wavelength = pi * wl * w0 / f。
    # 这里所有长度均使用 mm，因此算出的波长也首先是 mm。
    lam_min_mm = (math.pi * input_waist_mm * r_mot) / focal_length_mm
    lam_max_mm = (math.pi * input_waist_mm * r_dpt) / focal_length_mm

    # 1 mm = 10^6 nm。
    lam_min_nm = lam_min_mm * 1e6
    lam_max_nm = lam_max_mm * 1e6

    return lam_min_nm, lam_max_nm


def min_laser_power_for_trap(resonance_freq_THz, linewidth_MHz, efficiency,
                             waist_um, temperature_uK, laser_wavelength_nm):
    """
    估算光阱深度达到 ``(3/2) k_B T`` 时所需的激光功率。

    当前实现把原子近似为只有一条跃迁的理想二能级系统，并使用
    ``Delta = omega_laser - omega0`` 表示角频率失谐。计算中对失谐
    取绝对值，因此返回的是势阱深度大小对应的功率，而不是带符号
    的光学偶极势。

    参数
    ----
    resonance_freq_THz : float
        所选原子跃迁的普通频率 ``f0``，单位 THz。
    linewidth_MHz : float
        所选跃迁的自然线宽 ``Gamma / (2*pi)``，单位 MHz。函数内部会
        乘以 ``2*pi``，转换为角频率线宽 ``Gamma``。
    efficiency : float
        从输入功率到原子位置有效功率的总效率，无量纲。当前代码只
        禁止它等于零；物理上通常还应要求 ``0 < efficiency <= 1``。
    waist_um : float
        原子位置处高斯光束的束腰半径，单位 µm。应明确使用 1/e²
        强度半径，并与推导光强公式时的半径定义保持一致。
    temperature_uK : float
        原子温度，单位 µK。
    laser_wavelength_nm : float
        陷阱激光的真空波长，单位 nm。

    返回
    ----
    float
        当前近似模型得到的最小输入激光功率，单位 W。

    注意
    ----
    * 对 Rb、Cs 等碱金属的远失谐光阱，D1、D2 两条线通常都会贡献
      偶极势；只输入一条跃迁不能得到高精度结果。
    * 蓝失谐光与红失谐光形成的陷阱几何不同。这里取绝对值只是在
      计算能量尺度，不能证明普通高斯光束中心对蓝失谐原子是稳定阱。
    * 靠近共振时，远失谐近似和“只比较阱深”的判据都会失效，必须
      同时计算光子散射率。
    * 本函数保留项目原始功率公式，尚未修正此前审查发现的前因子
      约定问题，以避免本次“加注释”任务无意改变现有计算结果。
    """
    # SI 定义值。使用显式常数可避免为两个常数额外引入依赖。
    c = 299792458.0       # 真空光速，单位 m/s。
    k_B = 1.380649e-23    # 玻尔兹曼常数，单位 J/K。

    # 将用户友好的 THz、MHz、µm、µK、nm 全部转换为 SI 单位。
    # 先统一单位，再进行角频率和功率计算，可减少量纲混用。
    f0_Hz      = resonance_freq_THz * 1e12
    Gamma_Hz   = linewidth_MHz * 1e6
    w0_m       = waist_um * 1e-6
    T_K        = temperature_uK * 1e-6
    lambda_m   = laser_wavelength_nm * 1e-9

    # 普通频率 f 与角频率 omega 的关系是 omega = 2*pi*f。
    # 输入线宽按常见原子数据表约定解释为 Gamma/(2*pi)，所以也乘 2*pi。
    omega0 = 2 * math.pi * f0_Hz
    Gamma  = 2 * math.pi * Gamma_Hz

    # 由 f_laser = c/lambda 得到 omega_laser = 2*pi*c/lambda。
    omega_laser = 2 * math.pi * c / lambda_m

    # 定义失谐 Delta = omega_laser - omega0。
    # 红失谐对应 Delta < 0，蓝失谐对应 Delta > 0。
    # 当前函数只计算势能深度的大小，因此沿用原始实现取绝对值；
    # 若要判断具体陷阱几何和势能方向，必须保留 Delta 的符号。
    Delta = omega_laser - omega0
    Delta_abs = abs(Delta)

    # 防止功率公式分母为零。更完整的实现还应检查所有输入均为
    # 有限正数，并要求效率位于 (0, 1]。
    if efficiency == 0:
        raise ValueError("激光效率不能为零")

    # 项目原始的最小功率公式：
    #
    #   P_min = 2 k_B T omega0^3 w0^2 |Delta|
    #           / (c^2 Gamma efficiency)
    #
    # 它表达了以下定性标度：
    #   * 温度越高，需要的阱深和功率越高；
    #   * 束腰越大，中心光强越低，功率按 w0^2 增长；
    #   * 失谐越大，同一功率产生的偶极势越弱，需要更多功率；
    #   * 线宽或有效传输效率越大，模型给出的所需功率越低。
    #
    # 注意：该前因子与 docstring 中 1.5 k_B T 判据的标准二能级推导
    # 仍需统一。本次仅补充说明，不改变原始数值行为。
    P_min = (2 * k_B * T_K * omega0**3 * w0_m**2 * Delta_abs) / (c**2 * Gamma * efficiency)

    return P_min
