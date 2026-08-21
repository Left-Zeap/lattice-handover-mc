"""L1/L2 运输腿的轨迹级 Monte Carlo 模型（可选，默认关闭）。

与 ``handover.py`` 同型的三维经典轨迹模拟，但光场采用底层双束干涉
形式。全部公式与
``reports/运输蒙特卡洛与双束底层光场理论框架.md`` 一致：

- 势与力（§2、§3.1）：
  ``V = -|C_U|·[I₁e^(−2ρ²/w₁²) + I₂e^(−2ρ²/w₂²)
        + 2√(I₁I₂)e^(−ρ²(1/w₁²+1/w₂²))·cos(2k(z−z_L(t))−φ)]``，
  轴向力取交叉项的 sin 梯度，径向力取三包络项梯度之和；几何 z
  梯度（∂I_i/∂z、∂w_i/∂z）按 §2.3 忽略，I_i、w_i 取 z_L(t) 处的
  剖面值。``|C_U|`` 由 ``dipole.scalar_potential_and_scattering``
  的单位强度系数给出。``conveyor_enabled`` 时 I_i、w_i 沿程取
  conveyor 几何剖面（源端功率恒定）；关闭时 w₁=w₂=线性插值束腰、
  源端功率同样恒定（波腹强度按 1/w² 变化），势退化为含节点
  基底的单包络驻波，与 ``evaluate_lattice`` 的 (1+√R)² 波腹一致。
  两种模式共用同一套双束力代码。
- 运动（§3.2）：z_L(t) 复用 ``l1_transport.l1_timing`` 与
  ``_kinematics`` 的梯形轨迹，驻波相位吸进运动坐标，与 handover
  行波相位 k(q−vt) 同一处理；积分在实验室系进行。相对相位 φ 固定
  为 0：它只对应驻波图样的整体平移，且与 t=0 格点吸附约定
  （格点位于 n·λ/2）一致。
- 重力：全局开关启用时在 CPU/GPU 融合力核中统一加入
  ``F_y=-mg``；非连续初态从重力下垂平衡点采样，逃逸同时检查轴向
  加速势垒和径向高斯重力下坡势垒。
- 初态（§3.4）：谐振提议（ω_r²=(4|C_U|/m)(I₁/w₁²+I₂/w₂²)、
  ω_z²=2|C_U|ΔI_ax·k²/m，ΔI_ax=4√(I₁I₂)，均取 z=0 值）+
  ``cloud_axial_sigma_mm`` 轴向云吸附到最近格点 + 完整双束势拒绝
  （相对共动系总激发能 ε₁<U_ax(z=0)，拒绝超上限抛错）。
- 散射（§3.5，可选）：Poisson(Γ·dt)，局域率按非相干强度和缩放
  Γ=s_I·[I₁e^(−2ρ²/w₁²)+I₂e^(−2ρ²/w₂²)]；吸收沿 z 轴（前向概率
  1/(1+R_loc)，R_loc 为局域有效回程比），自发辐射各向同性；
  kick=(ħk/m)(s·ẑ−n̂)。
- 逃逸（§3.6）：每 200 步（默认 0.5 µs 步长下为 0.1 ms）按瞬时
  倾斜势垒 U_eff=|C_U|ΔI_ax·F(a(t)/a_c(z)) 剔除（F 用
  ``lattice.tilted_lattice_barrier_fraction``，a(t) 为当前瞬时
  加速度）；启用重力时再以 ``V+mgy`` 检查径向下坡鞍点势垒，两者
  任一失效即剔除。末步再做一次终态剔除；全灭则提前结束。
- 积分（§3.3）：velocity-Verlet（与 handover 同一辛积分器），
  请求步长取 ``inputs.transport_time_step_us``（默认来自配置
  ``transport_monte_carlo.time_step_us``，0.5 µs），再经
  ``_stable_leg_step_s`` 精度守卫钳制（ω_z·dt ≤ 1，与 handover
  同一判据，ω_z 取沿程最大轴向调制深度对应的阱频），整数步精确
  落在终点，实际步长记录在 ``L1DesignPoint.actual_time_step_us``。
  守卫的必要性：ω_z·dt 接近稳定界 2 时深阱热尾会被 Verlet 相位
  振荡瞬时推过势垒，逃逸后的快粒子又被驻波力的欠采样系统加速，
  产生非物理损耗（旧默认 0.5 µs 在 L1 典型阱频下 ω_z·dt ≈ 1.2，
  多微秒步长直接发散、L1 末端全灭）。定量工作仍建议做步长减半
  收敛检查。

输出与解析腿完全同型（``L1TransportTrace``），快照复用
``_time_grid`` 网格：温度为幸存者共动系动能温度（去质心，同
handover 的捕获子样本口径），留存率为幸存比例并附 Jeffreys
Beta(½,½) 后验标准误（``retention_standard_error``），
``bound_fraction`` 为幸存者中 ε<U_eff 的比例，
``cumulative_scattering_events`` 为每初始粒子的平均散射计数，
``instantaneous_loss_rate_s`` 填相邻快照间逃逸率的估算（解析腿的
速率方程口径在 MC 中无定义）。``L1DesignPoint`` 的
``depth_uK``/``scattering_rate_s`` 取 z=L 值（与解析腿同一调用），
``feasible_hardware_point`` 沿用同一检查。粒子全灭后温度记 NaN，
扫描层按既有 NaN 口径视为无效点。

参数合并：几何、时序、初态、效率、回程比和 conveyor 参数来自
``inputs``（与解析腿共用）；``mc_particle_count``、``mc_seed``、
``mc_include_scattering``、``mc_cloud_axial_sigma_mm`` 同样取自
``inputs``（``L1TransportInputs`` 的字段，默认与
``handover_monte_carlo`` 配置组同值，UI/CLI 的每调用设置随之对
运输腿生效）；本模块只新增 ``transport_monte_carlo`` 的
enabled（经 ``L1TransportInputs.transport_method``）与
time_step_us 两个参数。浅阱/高温到无法采样束缚初态的参数点返回
零留存 trace（温度 NaN、留存 0），不再抛错中断扫描。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .constants import BOLTZMANN, GRAVITY, HBAR
from .conveyor_geometry import conveyor_point, conveyor_profile
from .dipole import scalar_potential_and_scattering
from .gpu_backend import (
    module_of as _module_of,
    resolve_backend as _resolve_backend,
    rng_standard_normal as _rng_standard_normal,
    scatter_add as _scatter_add,
    scattering_kicks_gpu as _scattering_kicks_gpu,
    scattering_rng_gpu as _scattering_rng_gpu,
)
from .handover import _kinetic_temperature_uK
from .lattice import (
    evaluate_lattice,
    gaussian_gravity_trap,
    tilted_lattice_barrier_fraction,
)
from .l1_transport import (
    L1DesignPoint,
    L1TransportInputs,
    L1TransportTrace,
    _atom_from_label,
    _kinematics,
    _time_grid,
    l1_timing,
)
from .phase_space import ParticleEnsemble


_Z_AXIS = np.array((0.0, 0.0, 1.0))
# 已越过倾斜势垒的轨迹必须在绘图采样前及时移除，否则快逃逸原子会
# 制造“先升到数千 µK、下一帧又下降”的伪温度尖刺。默认 0.5 µs
# 步长下每 0.1 ms 判定一次；相对力积分成本仍很小。
_ESCAPE_CHECK_INTERVAL_STEPS = 200
_PROFILE_GRID_POINTS = 801


@dataclass(frozen=True)
class _LegOpticsProfile:
    """沿运输轴预计算的双束参数剖面（SI），逐步按 z_L 线性插值。"""

    position_m: np.ndarray
    intensity1_w_m2: np.ndarray
    intensity2_w_m2: np.ndarray
    waist1_m: np.ndarray
    waist2_m: np.ndarray
    effective_waist_um: np.ndarray
    source_power_w: np.ndarray


def _leg_optics_profile(
    inputs: L1TransportInputs,
    wavelength_nm: float,
    handover_source_power_w: float,
) -> _LegOpticsProfile:
    """预计算沿程 I₁、I₂、w₁、w₂ 及记录用的束腰/源端功率剖面。

    conveyor 开启：错腰双束半径（束腰位于 (L∓s)/2）+ 恒定源端功率，
    原子处前向功率 = 源端×传输效率，与 ``conveyor_point`` 同一口径；
    关闭：w₁=w₂=L1 标定高斯包络（旧输入可回退线性插值）、源端功率
    恒定（与解析腿同一口径），前向/反射光强按 1/w² 沿程变化并形成
    驻波。
    """
    distance = inputs.distance_m
    grid = np.linspace(0.0, distance, _PROFILE_GRID_POINTS)
    forward_at_atoms_w = handover_source_power_w * inputs.delivery_efficiency
    if inputs.conveyor_enabled:
        separation_m = inputs.conveyor_waist_separation_cm * 1e-2
        wavelength_m = wavelength_nm * 1e-9
        waist_m = inputs.conveyor_waist_um * 1e-6
        rayleigh_m = math.pi * waist_m**2 / wavelength_m
        focus_forward = 0.5 * (distance - separation_m)
        focus_retro = 0.5 * (distance + separation_m)
        w1 = waist_m * np.sqrt(1.0 + ((grid - focus_forward) / rayleigh_m) ** 2)
        w2 = waist_m * np.sqrt(1.0 + ((grid - focus_retro) / rayleigh_m) ** 2)
        intensity1 = 2.0 * forward_at_atoms_w / (math.pi * w1**2)
        intensity2 = (
            inputs.retro_power_ratio
            * 2.0
            * forward_at_atoms_w
            / (math.pi * w2**2)
        )
        effective_waist_um = (
            np.sqrt(
                (intensity1 + intensity2)
                / (intensity1 / w1**2 + intensity2 / w2**2)
            )
            * 1e6
        )
        source_power = np.full_like(grid, handover_source_power_w)
    else:
        waist_um = np.asarray(
            [inputs.beam_radius_um_at(float(position)) for position in grid],
            dtype=float,
        )
        w1 = waist_um * 1e-6
        w2 = w1.copy()
        source_power = np.full_like(grid, handover_source_power_w)
        intensity1 = (
            2.0
            * source_power
            * inputs.delivery_efficiency
            / (math.pi * w1**2)
        )
        intensity2 = inputs.retro_power_ratio * intensity1
        effective_waist_um = waist_um
    return _LegOpticsProfile(
        position_m=grid,
        intensity1_w_m2=intensity1,
        intensity2_w_m2=intensity2,
        waist1_m=w1,
        waist2_m=w2,
        effective_waist_um=np.asarray(effective_waist_um, dtype=float),
        source_power_w=np.asarray(source_power, dtype=float),
    )


def _leg_optics_at(
    inputs: L1TransportInputs,
    profile: _LegOpticsProfile,
    handover_source_power_w: float,
    position_m: float,
    time_s: float,
) -> tuple[float, float, float, float, float, float]:
    """返回瞬时 ``I1,I2,w1,w2,effective_waist_um,source_power_w``。

    无实测光学通道时完全沿用原按位置插值的剖面；有通道时只覆盖
    CSV 明确给出的量。该函数仅在 host 预计算系数，GPU 内核结构和
    每步设备端运算量不变。
    """
    grid = profile.position_m
    i1 = float(np.interp(position_m, grid, profile.intensity1_w_m2))
    i2 = float(np.interp(position_m, grid, profile.intensity2_w_m2))
    w1 = float(np.interp(position_m, grid, profile.waist1_m))
    w2 = float(np.interp(position_m, grid, profile.waist2_m))
    effective_waist = float(
        np.interp(position_m, grid, profile.effective_waist_um)
    )
    source_power = float(np.interp(position_m, grid, profile.source_power_w))
    if inputs.control_waveform is None:
        return i1, i2, w1, w2, effective_waist, source_power
    control = inputs.control_waveform.sample(time_s)
    if control["waist_um"] is not None:
        effective_waist = float(control["waist_um"])
        w1 = effective_waist * 1e-6
        w2 = w1
    if control["source_power_scale"] is not None:
        source_power = handover_source_power_w * float(
            control["source_power_scale"]
        )
    delivery_scale = (
        1.0
        if control["delivery_efficiency_scale"] is None
        else float(control["delivery_efficiency_scale"])
    )
    if any(
        control[name] is not None
        for name in (
            "waist_um",
            "source_power_scale",
            "delivery_efficiency_scale",
        )
    ):
        i1 = (
            2.0
            * source_power
            * inputs.delivery_efficiency
            * delivery_scale
            / (math.pi * w1**2)
        )
        i2 = inputs.retro_power_ratio * i1
    return i1, i2, w1, w2, effective_waist, source_power


def _stable_leg_step_s(
    inputs: L1TransportInputs,
    atom,
    wavelength_nm: float,
    profile: _LegOpticsProfile,
) -> float:
    """Velocity-Verlet 精度守卫：沿程最快轴向阱频 ω_z·dt ≤ 1。

    与 ``handover._stable_handover_step_s`` 同一判据：数学稳定界为
    ω_z·dt < 2，但接近该值会出现严重长期能量漂移——深阱热尾被
    Verlet 相位振荡瞬时推过势垒，逃逸后的快粒子又被欠采样的驻波力
    系统性加速，产生非物理损耗（旧默认 0.5 µs 在 L1 典型阱频下
    ω_z·dt ≈ 1.2，已接近失稳区；用户选取的多微秒步长更是直接
    发散）。ω_z 取沿程最大轴向调制深度
    U_ax = 4·|C|·√(I₁I₂)（波腹→波节）对应的 k√(2U_ax/m)；实测波形
    通道若短时超过剖面强度，本守卫按位置剖面口径估计，不逐时追踪。
    """
    potential_per_intensity = abs(
        scalar_potential_and_scattering(atom, wavelength_nm, 1.0).potential_j
    )
    wave_number = 2.0 * math.pi / (wavelength_nm * 1e-9)
    # 轴向调制深度（波腹→波节）= 2 × 交叉项幅值 = 4·|C|·√(I₁I₂)。
    modulation_j = (
        4.0
        * potential_per_intensity
        * np.sqrt(profile.intensity1_w_m2 * profile.intensity2_w_m2)
    )
    max_modulation = float(np.max(modulation_j)) if modulation_j.size else 0.0
    if max_modulation <= 0.0:
        return inputs.transport_time_step_us * 1e-6
    axial_omega = wave_number * math.sqrt(2.0 * max_modulation / atom.mass_kg)
    return 1.0 / axial_omega


def _leg_integration_schedule(
    inputs: L1TransportInputs,
    atom,
    wavelength_nm: float,
    profile: _LegOpticsProfile,
    total_time_s: float,
) -> tuple[int, float]:
    """（步数, 步长）：请求步长经 ``_stable_leg_step_s`` 钳制后均分总时长。

    CPU 逐步路径、GPU 批量路径与 ``light_field`` 时序表必须共用本函数，
    保证三套实现步长逐位一致。
    """
    requested = min(
        inputs.transport_time_step_us * 1e-6,
        _stable_leg_step_s(inputs, atom, wavelength_nm, profile),
    )
    steps = max(1, math.ceil(total_time_s / requested))
    return steps, total_time_s / steps


def _double_beam_step_coefficients(
    *,
    intensity1_w_m2: float,
    intensity2_w_m2: float,
    waist1_m: float,
    waist2_m: float,
    wave_number_m: float,
    lattice_position_m: float,
    phase_rad: float,
    potential_per_intensity_j: float,
) -> tuple[float, ...]:
    """双束融合 kernel 的 host 预计算标量系数（与 kernel 参数同序）。

    所有标量系数在 host 预计算（规避 CuPy 14 + sm_120 对标量-
    标量子表达式的融合代码生成 bug，见 handover.py 同名注释）。
    """
    geometric_mean = math.sqrt(intensity1_w_m2 * intensity2_w_m2)
    return (
        -2.0 / waist1_m**2,
        -2.0 / waist2_m**2,
        -(1.0 / waist1_m**2 + 1.0 / waist2_m**2),
        intensity1_w_m2,
        intensity2_w_m2,
        potential_per_intensity_j * intensity1_w_m2,
        potential_per_intensity_j * intensity2_w_m2,
        potential_per_intensity_j * 2.0 * geometric_mean,
        potential_per_intensity_j
        * 2.0
        * geometric_mean
        * 2.0
        * wave_number_m,
        potential_per_intensity_j
        * 4.0
        * intensity1_w_m2
        / waist1_m**2,
        potential_per_intensity_j
        * 4.0
        * intensity2_w_m2
        / waist2_m**2,
        potential_per_intensity_j
        * 4.0
        * geometric_mean
        * (1.0 / waist1_m**2 + 1.0 / waist2_m**2),
        2.0 * wave_number_m,
        lattice_position_m,
        phase_rad,
    )


def _double_beam_potential_and_force(
    positions_m: np.ndarray,
    *,
    intensity1_w_m2: float,
    intensity2_w_m2: float,
    waist1_m: float,
    waist2_m: float,
    wave_number_m: float,
    lattice_position_m: float,
    phase_rad: float,
    potential_per_intensity_j: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """双束干涉势、解析梯度力和局域非相干强度（理论文档 §2、§3.1）。

    运输轴沿 z，ρ 为横向两轴；几何 z 梯度忽略，I_i、w_i 取调用时
    z_L(t) 处的值。返回 ``(V, F, 局域前向强度, 局域非相干强度和)``；
    后两者用于散射反冲的局域率和前向概率 1/(1+R_loc)。数组运算
    使用输入数组所属的后端（NumPy 或 CuPy）。
    """
    xp = _module_of(positions_m)
    if xp is not np:
        kernel = _get_fused_double_beam_kernel()
        potential, radial_coefficient, axial_force, local_forward, local_incoherent = (
            kernel(
                positions_m,
                *_double_beam_step_coefficients(
                    intensity1_w_m2=intensity1_w_m2,
                    intensity2_w_m2=intensity2_w_m2,
                    waist1_m=waist1_m,
                    waist2_m=waist2_m,
                    wave_number_m=wave_number_m,
                    lattice_position_m=lattice_position_m,
                    phase_rad=phase_rad,
                    potential_per_intensity_j=potential_per_intensity_j,
                ),
            )
        )
        force = positions_m * radial_coefficient[:, None]
        force[:, 2] = axial_force
        return potential, force, local_forward, local_incoherent
    rho2 = (positions_m[:, :2] * positions_m[:, :2]).sum(axis=1)
    zeta = positions_m[:, 2] - lattice_position_m
    envelope1 = xp.exp(-2.0 * rho2 / waist1_m**2)
    envelope2 = xp.exp(-2.0 * rho2 / waist2_m**2)
    cross_envelope = xp.exp(
        -rho2 * (1.0 / waist1_m**2 + 1.0 / waist2_m**2)
    )
    theta = 2.0 * wave_number_m * zeta - phase_rad
    geometric_mean = math.sqrt(intensity1_w_m2 * intensity2_w_m2)
    cosine = xp.cos(theta)

    potential = -potential_per_intensity_j * (
        intensity1_w_m2 * envelope1
        + intensity2_w_m2 * envelope2
        + 2.0 * geometric_mean * cross_envelope * cosine
    )
    axial_force = (
        -potential_per_intensity_j
        * (2.0 * geometric_mean * cross_envelope)
        * (2.0 * wave_number_m)
        * xp.sin(theta)
    )
    radial_coefficient = -potential_per_intensity_j * (
        4.0 * intensity1_w_m2 * envelope1 / waist1_m**2
        + 4.0 * intensity2_w_m2 * envelope2 / waist2_m**2
        + 2.0
        * geometric_mean
        * cross_envelope
        * 2.0
        * cosine
        * (1.0 / waist1_m**2 + 1.0 / waist2_m**2)
    )
    force = xp.empty_like(positions_m)
    force[:, 0] = radial_coefficient * positions_m[:, 0]
    force[:, 1] = radial_coefficient * positions_m[:, 1]
    force[:, 2] = axial_force
    local_forward = intensity1_w_m2 * envelope1
    local_incoherent = local_forward + intensity2_w_m2 * envelope2
    return potential, force, local_forward, local_incoherent


_FUSED_DOUBLE_BEAM_KERNEL = None


def _get_fused_double_beam_kernel():
    """惰性创建双束势-力融合的 CuPy kernel（与 NumPy 路径逐式同构）。

    返回 ``(V, 径向系数, 轴向力, 局域前向强度, 局域非相干强度和)``；
    力数组由调用方用两个 kernel 组装，避免在融合函数内部显式分配。
    """
    global _FUSED_DOUBLE_BEAM_KERNEL
    if _FUSED_DOUBLE_BEAM_KERNEL is None:
        import cupy as cp

        @cp.fuse()
        def kernel(
            positions,
            envelope_c1,
            envelope_c2,
            envelope_cc,
            intensity1,
            intensity2,
            potential_c1,
            potential_c2,
            potential_cc,
            axial_c,
            radial_c1,
            radial_c2,
            radial_cc,
            two_wave_number,
            lattice_position,
            phase_rad,
        ):
            rho2 = (
                positions[:, 0] * positions[:, 0]
                + positions[:, 1] * positions[:, 1]
            )
            zeta = positions[:, 2] - lattice_position
            envelope1 = cp.exp(rho2 * envelope_c1)
            envelope2 = cp.exp(rho2 * envelope_c2)
            cross_envelope = cp.exp(rho2 * envelope_cc)
            theta = two_wave_number * zeta - phase_rad
            cosine = cp.cos(theta)
            potential = -(
                potential_c1 * envelope1
                + potential_c2 * envelope2
                + potential_cc * cross_envelope * cosine
            )
            axial_force = -(axial_c * cross_envelope * cp.sin(theta))
            radial_coefficient = -(
                radial_c1 * envelope1
                + radial_c2 * envelope2
                + radial_cc * cross_envelope * cosine
            )
            local_forward = intensity1 * envelope1
            local_incoherent = local_forward + intensity2 * envelope2
            return (
                potential,
                radial_coefficient,
                axial_force,
                local_forward,
                local_incoherent,
            )

        _FUSED_DOUBLE_BEAM_KERNEL = kernel
    return _FUSED_DOUBLE_BEAM_KERNEL


_FUSED_LEG_STEP_KERNEL = None


def _get_fused_leg_step_kernel():
    """惰性创建运输腿整步 velocity-Verlet 融合的 CuPy kernel（mega-step）。

    一次 kernel 完成：半步速度 → 整步位置 → 新位置/新时刻的双束势与
    合力 → 半步速度，就地更新 ``positions``/``velocities``/``force``
    （以列视图 ``p0..f2`` 传入 (M,3) 数组的三列，cupy.fuse 只支持
    ``[...]`` 整体赋值）。返回新位置处的 ``(势, 局域前向强度, 局域
    非相干强度和)``，供逃逸剔除与散射反冲使用。与 CPU 路径逐式同构；
    几何插值系数（I_i、w_i 及全部组合系数）逐步在 host 预计算后经
    ``_double_beam_step_coefficients`` 传入。
    """
    global _FUSED_LEG_STEP_KERNEL
    if _FUSED_LEG_STEP_KERNEL is None:
        import cupy as cp

        @cp.fuse()
        def kernel(
            p0,
            p1,
            p2,
            v0,
            v1,
            v2,
            f0,
            f1,
            f2,
            envelope_c1,
            envelope_c2,
            envelope_cc,
            intensity1,
            intensity2,
            potential_c1,
            potential_c2,
            potential_cc,
            axial_c,
            radial_c1,
            radial_c2,
            radial_cc,
            two_wave_number,
            lattice_position,
            phase_rad,
            gravity_force_y,
            half_dt_over_mass,
            time_step,
        ):
            # 半步速度 + 整步位置。
            nv0 = v0 + f0 * half_dt_over_mass
            nv1 = v1 + f1 * half_dt_over_mass
            nv2 = v2 + f2 * half_dt_over_mass
            np0 = p0 + nv0 * time_step
            np1 = p1 + nv1 * time_step
            np2 = p2 + nv2 * time_step
            # 新位置、新时刻的双束势与力。
            rho2 = np0 * np0 + np1 * np1
            zeta = np2 - lattice_position
            envelope1 = cp.exp(rho2 * envelope_c1)
            envelope2 = cp.exp(rho2 * envelope_c2)
            cross_envelope = cp.exp(rho2 * envelope_cc)
            theta = two_wave_number * zeta - phase_rad
            cosine = cp.cos(theta)
            potential = -(
                potential_c1 * envelope1
                + potential_c2 * envelope2
                + potential_cc * cross_envelope * cosine
            )
            axial_force = -(axial_c * cross_envelope * cp.sin(theta))
            radial_coefficient = -(
                radial_c1 * envelope1
                + radial_c2 * envelope2
                + radial_cc * cross_envelope * cosine
            )
            g0 = radial_coefficient * np0
            g1 = radial_coefficient * np1 + gravity_force_y
            g2 = axial_force
            # 半步速度（新力），全部状态就地写回。
            p0[...] = np0
            p1[...] = np1
            p2[...] = np2
            v0[...] = nv0 + g0 * half_dt_over_mass
            v1[...] = nv1 + g1 * half_dt_over_mass
            v2[...] = nv2 + g2 * half_dt_over_mass
            f0[...] = g0
            f1[...] = g1
            f2[...] = g2
            local_forward = intensity1 * envelope1
            local_incoherent = local_forward + intensity2 * envelope2
            return potential, local_forward, local_incoherent

        _FUSED_LEG_STEP_KERNEL = kernel
    return _FUSED_LEG_STEP_KERNEL


def _sample_initial_ensemble(
    *,
    particle_count: int,
    atom_mass_kg: float,
    temperature_uK: float,
    intensity1_w_m2: float,
    intensity2_w_m2: float,
    waist1_m: float,
    waist2_m: float,
    axial_modulation_j: float,
    wave_number_m: float,
    potential_per_intensity_j: float,
    cloud_axial_sigma_mm: float,
    include_gravity: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """谐振提议 + 轴向云吸附格点 + 完整双束势拒绝（理论文档 §3.4）。

    束缚判据为相对共动系总激发能 ε₁=½m|v|²+V+U_ax<U_ax（z_L(0)=0、
    φ=0，格点位于 n·λ/2）；拒绝超上限说明当前温度/阱深下几乎无束缚
    初态，抛 ``ValueError``。
    """
    mass = atom_mass_kg
    temperature_k = temperature_uK * 1e-6
    omega_radial = math.sqrt(
        4.0
        * potential_per_intensity_j
        / mass
        * (
            intensity1_w_m2 / waist1_m**2
            + intensity2_w_m2 / waist2_m**2
        )
    )
    omega_axial = math.sqrt(
        2.0 * axial_modulation_j * wave_number_m**2 / mass
    )
    sigma_radial = math.sqrt(BOLTZMANN * temperature_k / mass) / omega_radial
    sigma_axial = math.sqrt(BOLTZMANN * temperature_k / mass) / omega_axial
    sigma_velocity = math.sqrt(BOLTZMANN * temperature_k / mass)

    radial_depth_j = potential_per_intensity_j * (
        intensity1_w_m2
        + intensity2_w_m2
        + 2.0 * math.sqrt(intensity1_w_m2 * intensity2_w_m2)
    )
    radial_curvature_weight = potential_per_intensity_j * (
        intensity1_w_m2 / waist1_m**2
        + intensity2_w_m2 / waist2_m**2
        + math.sqrt(intensity1_w_m2 * intensity2_w_m2)
        * (1.0 / waist1_m**2 + 1.0 / waist2_m**2)
    )
    effective_waist_m = math.sqrt(radial_depth_j / radial_curvature_weight)
    gravity_barrier_j = radial_depth_j
    gravity_minimum_j = -radial_depth_j
    gravity_sag_m = 0.0
    if include_gravity:
        gravity_barrier_j, gravity_minimum_j, gravity_sag_m = (
            gaussian_gravity_trap(radial_depth_j, effective_waist_m, mass)
        )

    lattice_spacing = math.pi / wave_number_m
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    accepted = 0
    total_drawn = 0
    while accepted < particle_count:
        batch = max(256, 2 * (particle_count - accepted))
        trial_positions = rng.normal(size=(batch, 3))
        trial_positions[:, 0] *= sigma_radial
        trial_positions[:, 1] *= sigma_radial
        trial_positions[:, 1] += gravity_sag_m
        trial_positions[:, 2] *= sigma_axial
        if cloud_axial_sigma_mm > 0.0:
            site_coordinate = rng.normal(
                scale=cloud_axial_sigma_mm * 1e-3,
                size=batch,
            )
            trial_positions[:, 2] += (
                np.rint(site_coordinate / lattice_spacing) * lattice_spacing
            )
        trial_velocities = rng.normal(scale=sigma_velocity, size=(batch, 3))
        potential, _, _, _ = _double_beam_potential_and_force(
            trial_positions,
            intensity1_w_m2=intensity1_w_m2,
            intensity2_w_m2=intensity2_w_m2,
            waist1_m=waist1_m,
            waist2_m=waist2_m,
            wave_number_m=wave_number_m,
            lattice_position_m=0.0,
            phase_rad=0.0,
            potential_per_intensity_j=potential_per_intensity_j,
        )
        kinetic = 0.5 * mass * np.einsum(
            "ij,ij->i",
            trial_velocities,
            trial_velocities,
        )
        excitation = kinetic + potential + axial_modulation_j
        bound = excitation < axial_modulation_j
        if include_gravity:
            radial_excitation = (
                kinetic
                + potential
                + mass * GRAVITY * trial_positions[:, 1]
                - gravity_minimum_j
            )
            bound &= (
                gravity_barrier_j > 0.0
            ) & (radial_excitation < gravity_barrier_j)
        if np.any(bound):
            positions.append(trial_positions[bound])
            velocities.append(trial_velocities[bound])
            accepted += int(np.count_nonzero(bound))
        total_drawn += batch
        if total_drawn > 1_000 * particle_count:
            raise ValueError(
                "当前温度/阱深下的初始束缚比例过低，无法稳定采样"
            )
    return (
        np.concatenate(positions, axis=0)[:particle_count].copy(),
        np.concatenate(velocities, axis=0)[:particle_count].copy(),
    )


def _apply_scattering_kicks(
    velocities_m_s: np.ndarray,
    *,
    local_forward_w_m2: np.ndarray,
    local_incoherent_w_m2: np.ndarray,
    rate_per_intensity_s: float,
    time_step_s: float,
    wave_number_m: float,
    atom_mass_kg: float,
    rng: np.random.Generator,
) -> int:
    """按局域非相干强度施加 Poisson 吸收和各向同性自发辐射反冲。"""
    xp = _module_of(velocities_m_s)
    rates_s = rate_per_intensity_s * local_incoherent_w_m2
    counts = rng.poisson(rates_s * time_step_s)
    atom_indices = xp.repeat(xp.flatnonzero(counts), counts[counts > 0])
    event_count = int(atom_indices.size)
    if event_count == 0:
        return 0
    forward_probability = (
        local_forward_w_m2[atom_indices]
        / local_incoherent_w_m2[atom_indices]
    )
    absorption_sign = xp.where(
        rng.random(event_count) < forward_probability,
        1.0,
        -1.0,
    )
    emission_direction = _rng_standard_normal(rng, size=(event_count, 3))
    emission_direction /= xp.linalg.norm(
        emission_direction,
        axis=1,
    )[:, None]
    recoil_velocity = HBAR * wave_number_m / atom_mass_kg
    kicks = recoil_velocity * (
        absorption_sign[:, None] * xp.asarray(_Z_AXIS) - emission_direction
    )
    _scatter_add(xp, velocities_m_s, atom_indices, kicks)
    return event_count


def _zero_retention_trace(
    inputs: L1TransportInputs,
    timing,
    profile: _LegOpticsProfile,
    potential_per_intensity: float,
    detuning_ghz: float,
    handover_source_power_w: float,
    wavelength_nm: float,
    start_source_power: float,
    handover_depth_uK: float,
    handover_scattering_rate_s: float,
    feasible: bool,
    particle_count: int,
) -> L1TransportTrace:
    """构造零留存 trace：用于浅阱/高温到几乎无束缚初态的参数点。

    物理含义是该工作点无法装载/保住原子（留存率≈0），而不是程序
    错误；温度类字段填 NaN，由扫描层按既有 NaN 口径视为无效点。
    """
    time_out: list[float] = []
    stage_out: list[str] = []
    position_out: list[float] = []
    velocity_out: list[float] = []
    acceleration_out: list[float] = []
    frequency_out: list[float] = []
    waist_out: list[float] = []
    power_out: list[float] = []
    barrier_out: list[float] = []
    nan = float("nan")
    for grid_time_s in _time_grid(inputs, timing):
        position, velocity, acceleration, stage = _kinematics(
            min(float(grid_time_s), timing.total_time_s),
            inputs,
            timing,
        )
        i1 = float(np.interp(position, profile.position_m, profile.intensity1_w_m2))
        i2 = float(np.interp(position, profile.position_m, profile.intensity2_w_m2))
        axial_modulation = potential_per_intensity * 4.0 * math.sqrt(i1 * i2)
        critical = axial_modulation * (
            2.0 * math.pi / (wavelength_nm * 1e-9)
        ) / _atom_from_label(inputs.atom_label).mass_kg
        barrier = axial_modulation * tilted_lattice_barrier_fraction(
            acceleration, critical
        )
        time_out.append(float(grid_time_s) * 1e3)
        stage_out.append(stage)
        position_out.append(position)
        velocity_out.append(velocity)
        acceleration_out.append(acceleration)
        frequency_out.append(2.0 * velocity / (wavelength_nm * 1e-9) * 1e-6)
        waist_out.append(
            float(np.interp(position, profile.position_m, profile.effective_waist_um))
        )
        power_out.append(
            float(np.interp(position, profile.position_m, profile.source_power_w))
        )
        barrier_out.append(barrier / BOLTZMANN * 1e6)
    count = len(time_out)
    # Jeffreys Beta(1/2, 1/2) 后验：k=0 时仍有有限标准误。
    standard_error = math.sqrt(
        (0.5 * (particle_count + 0.5))
        / ((particle_count + 1.0) ** 2 * (particle_count + 2.0))
    )
    point = L1DesignPoint(
        detuning_ghz=detuning_ghz,
        handover_source_power_w=handover_source_power_w,
        start_source_power_w=start_source_power,
        wavelength_nm=wavelength_nm,
        depth_uK=handover_depth_uK,
        scattering_rate_s=handover_scattering_rate_s,
        final_temperature_uK=nan,
        final_temperature_rise_uK=nan,
        final_retention_fraction=0.0,
        total_retention_from_mot_fraction=0.0,
        final_atom_number=0.0,
        cumulative_scattering_events=0.0,
        maximum_loss_rate_s=0.0,
        feasible_hardware_point=feasible,
        initial_temperature_uK=inputs.initial_temperature_uK,
        initial_atom_number=inputs.initial_atom_number,
    )
    return L1TransportTrace(
        point=point,
        time_ms=tuple(time_out),
        stage=tuple(stage_out),
        position_m=tuple(position_out),
        velocity_m_s=tuple(velocity_out),
        acceleration_m_s2=tuple(acceleration_out),
        aom_frequency_difference_mhz=tuple(frequency_out),
        waist_um=tuple(waist_out),
        source_power_w=tuple(power_out),
        effective_barrier_uK=tuple(barrier_out),
        temperature_uK=(nan,) * count,
        temperature_rise_uK=(nan,) * count,
        retention_fraction=(0.0,) * count,
        bound_fraction=(0.0,) * count,
        cumulative_scattering_events=(0.0,) * count,
        instantaneous_loss_rate_s=(0.0,) * count,
        retention_standard_error=standard_error,
    )


def simulate_leg_monte_carlo(
    inputs: L1TransportInputs,
    detuning_ghz: float,
    handover_source_power_w: float,
    *,
    initial_ensemble: ParticleEnsemble | None = None,
    return_final_ensemble: bool = False,
    escape_check_interval_steps: int | None = None,
    escape_lenient: bool = False,
) -> L1TransportTrace | tuple[L1TransportTrace, ParticleEnsemble | None]:
    """用轨迹级 Monte Carlo 积分一条 L1/L2 运输腿，输出同型 trace。

    ``handover_source_power_w`` 是 handover 束腰处每条晶格分支的源端
    功率（与解析腿同一口径）；几何、时序、初态、效率、回程比和
    conveyor 参数来自 ``inputs``，Monte Carlo 数值参数取自
    ``handover_monte_carlo`` 配置组。

    逃逸剔除口径（默认值与历史行为逐位一致）：

    - ``escape_check_interval_steps``：两次逃逸判定之间的动力学步数；
      ``None`` 保持默认 ``_ESCAPE_CHECK_INTERVAL_STEPS``（200），
      传 1 表示每个动力学步都判定（链式 MC 模式，见 ``chain_mc``）。
    - ``escape_lenient``：``False`` 时判据为共动系轴向激发能
      ε < U_ax·F(a/a_c)（倾斜势垒缩减）；``True`` 时不乘缩减因子，
      直接用未缩减全深 ε < U_ax（"宽容"判据；重力径向下坡鞍点判据
      保持不变）。

    两个参数**仅在 CPU 逐步路径生效**；GPU 批量（``transport_batch``）
    与逐步融合 kernel 路径忽略它们，仍用原"200 步 + 缩减势垒"判据
    ——GPU 是性能层，kernel 公式不动，判据口径差异的统计影响记录在
    ``chain_mc`` 模块 docstring。非法取值（非正间隔）在所有后端下
    都报错。
    """
    if escape_check_interval_steps is None:
        escape_interval_steps = _ESCAPE_CHECK_INTERVAL_STEPS
    else:
        escape_interval_steps = int(escape_check_interval_steps)
        if escape_interval_steps <= 0:
            raise ValueError("逃逸判定间隔必须是正整数")
    atom = _atom_from_label(inputs.atom_label)
    timing = l1_timing(inputs)
    wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning_ghz)
    wave_number = 2.0 * math.pi / (wavelength_nm * 1e-9)
    mass = atom.mass_kg
    forward_power_w = handover_source_power_w * inputs.delivery_efficiency

    unit_dipole = scalar_potential_and_scattering(atom, wavelength_nm, 1.0)
    potential_per_intensity = abs(unit_dipole.potential_j)
    scattering_per_intensity = unit_dipole.scattering_rate_s

    profile = _leg_optics_profile(
        inputs,
        wavelength_nm,
        handover_source_power_w,
    )

    # 端点量与可行性检查沿用解析腿同一口径（也在此拦截非正功率）。
    if inputs.conveyor_enabled:
        end_point = conveyor_point(
            atom,
            wavelength_nm,
            forward_power_w,
            inputs.conveyor_waist_um,
            inputs.conveyor_waist_separation_cm,
            inputs.distance_m,
            inputs.distance_m,
            inputs.retro_power_ratio,
        )
        geometry_profile = conveyor_profile(
            atom,
            wavelength_nm,
            forward_power_w,
            inputs.conveyor_waist_um,
            inputs.conveyor_waist_separation_cm,
            inputs.distance_m,
            np.linspace(0.0, inputs.distance_m, 401),
            inputs.retro_power_ratio,
        )
        start_source_power = handover_source_power_w
        handover_depth_uK = end_point.depth_uK
        handover_scattering_rate_s = end_point.scattering_rate_s
        minimum_critical_acceleration = (
            geometry_profile.minimum_critical_acceleration_m_s2
        )
    else:
        end_optics = _leg_optics_at(
            inputs,
            profile,
            handover_source_power_w,
            inputs.distance_m,
            timing.total_time_s,
        )
        end_i1, end_i2, _, _, end_waist_um, _ = end_optics
        lattice = evaluate_lattice(
            atom,
            wavelength_nm,
            forward_power_w=(
                math.pi * (end_waist_um * 1e-6) ** 2 * end_i1 / 2.0
            ),
            waist_um=end_waist_um,
            retro_power_ratio=inputs.retro_power_ratio,
        )
        start_source_power = _leg_optics_at(
            inputs, profile, handover_source_power_w, 0.0, 0.0
        )[-1]
        handover_depth_uK = lattice.depth_uK
        handover_scattering_rate_s = lattice.scattering_rate_s
        minimum_critical_acceleration = (
            lattice.critical_axial_acceleration_m_s2
        )
    feasible = (
        (
            not inputs.require_minimum_depth
            or handover_depth_uK >= inputs.target_depth_uK
        )
        and (
            not inputs.require_maximum_start_power
            or start_source_power <= inputs.maximum_l1_source_power_w
        )
        and (
            not inputs.require_critical_acceleration
            or minimum_critical_acceleration > inputs.acceleration_m_s2
        )
    )

    # Monte Carlo 数值参数（与 handover 合并调用：默认与
    # handover_monte_carlo 配置组同值，可由调用方经 inputs 覆盖）。
    particle_count = inputs.mc_particle_count
    seed = inputs.mc_seed
    include_scattering = inputs.mc_include_scattering
    cloud_axial_sigma_mm = inputs.mc_cloud_axial_sigma_mm
    backend = _resolve_backend(inputs.mc_compute_backend)
    if backend == "gpu":
        # 单点 GPU 走批量（P=1）设备端时间循环 kernel：逐步融合路径
        # 每步有 ~1 ms 固定开销（Python 调度/kernel 启动/散射同步），
        # 对 ~1e5 步的运输腿是主要墙钟成本；批量未覆盖的情形
        # （conveyor 几何抛 ValueError）回退本模块的逐步 GPU 路径。
        from .transport_batch import run_leg_monte_carlo_batch

        try:
            batched = run_leg_monte_carlo_batch(
                [((0, 0), inputs, detuning_ghz, handover_source_power_w)],
                backend="gpu",
                initial_ensembles=(
                    None
                    if initial_ensemble is None
                    else {(0, 0): initial_ensemble}
                ),
                return_final_ensembles=return_final_ensemble,
            )
            if return_final_ensemble:
                traces, ensembles = batched
                return traces[0], ensembles[0]
            return batched[0]
        except ValueError:
            pass
    if backend == "gpu":
        import cupy as cp

        xp = cp
    else:
        xp = np
    # 初态采样始终在 CPU 上用 NumPy RNG（与 CPU 后端逐位一致）；
    # 散射反冲的 RNG 按后端创建（GPU 序列与 CPU 仅统计一致）。
    rng = np.random.default_rng(seed)
    kick_rng = rng if xp is np else _scattering_rng_gpu(seed)

    total_time = timing.total_time_s
    # 步长经 ω_z·dt ≤ 1 精度守卫钳制（_stable_leg_step_s；三套实现共用
    # _leg_integration_schedule，保证逐位一致）。
    integration_steps, time_step_s = _leg_integration_schedule(
        inputs, atom, wavelength_nm, profile, total_time
    )

    # z=0 初态采样（z_L(0)=0，φ=0 时格点位于 n·λ/2）。
    (
        intensity1_0,
        intensity2_0,
        waist1_0,
        waist2_0,
        _,
        _,
    ) = _leg_optics_at(
        inputs, profile, handover_source_power_w, 0.0, 0.0
    )
    axial_modulation_0 = (
        potential_per_intensity * 4.0 * math.sqrt(intensity1_0 * intensity2_0)
    )
    try:
        if initial_ensemble is None:
            positions, velocities = _sample_initial_ensemble(
                particle_count=particle_count,
                atom_mass_kg=mass,
                temperature_uK=inputs.initial_temperature_uK,
                intensity1_w_m2=intensity1_0,
                intensity2_w_m2=intensity2_0,
                waist1_m=waist1_0,
                waist2_m=waist2_0,
                axial_modulation_j=axial_modulation_0,
                wave_number_m=wave_number,
                potential_per_intensity_j=potential_per_intensity,
                cloud_axial_sigma_mm=cloud_axial_sigma_mm,
                include_gravity=inputs.include_gravity,
                rng=rng,
            )
        else:
            propagated = initial_ensemble.resampled(particle_count, seed)
            positions, velocities, _ = propagated.host_arrays()
    except ValueError:
        # 几乎不存在束缚初态（浅阱/高温点）：物理上留存率≈0，返回
        # 零留存 trace 而不是让整个扫描因单点失败崩溃。
        zero_trace = _zero_retention_trace(
            inputs,
            timing,
            profile,
            potential_per_intensity,
            detuning_ghz,
            handover_source_power_w,
            wavelength_nm,
            start_source_power,
            handover_depth_uK,
            handover_scattering_rate_s,
            feasible,
            particle_count,
        )
        return (zero_trace, None) if return_final_ensemble else zero_trace

    # GPU 后端：采样完成后把粒子状态一次性传入 GPU。
    if xp is not np:
        positions = xp.asarray(positions)
        velocities = xp.asarray(velocities)
    z_axis = xp.asarray(_Z_AXIS)

    def _optics_at(
        position_m: float, time_s: float
    ) -> tuple[float, float, float, float, float, float]:
        return _leg_optics_at(
            inputs,
            profile,
            handover_source_power_w,
            position_m,
            time_s,
        )

    time_out: list[float] = []
    stage_out: list[str] = []
    position_out: list[float] = []
    velocity_out: list[float] = []
    acceleration_out: list[float] = []
    frequency_out: list[float] = []
    waist_out: list[float] = []
    power_out: list[float] = []
    barrier_out: list[float] = []
    temperature_out: list[float] = []
    retention_out: list[float] = []
    bound_out: list[float] = []
    scattering_out: list[float] = []
    loss_rate_out: list[float] = []

    total_scattering_events = 0
    gravity_force_y = -mass * GRAVITY if inputs.include_gravity else 0.0
    previous_record_alive = particle_count
    previous_record_time = 0.0

    def _record(grid_time_s: float, state_potential: np.ndarray) -> None:
        """在 ``_time_grid`` 网格时刻记录幸存者快照。"""
        nonlocal previous_record_alive, previous_record_time
        position_g, velocity_g, acceleration_g, stage_g = _kinematics(
            min(grid_time_s, total_time),
            inputs,
            timing,
        )
        i1_g, i2_g, _, _, waist_g, source_power_g = _optics_at(
            position_g, grid_time_s
        )
        axial_modulation_g = (
            potential_per_intensity * 4.0 * math.sqrt(i1_g * i2_g)
        )
        critical_g = axial_modulation_g * wave_number / mass
        barrier_fraction_g = tilted_lattice_barrier_fraction(
            acceleration_g,
            critical_g,
        )
        effective_barrier_j = axial_modulation_g * barrier_fraction_g
        radial_depth_g = potential_per_intensity * (
            i1_g + i2_g + 2.0 * math.sqrt(i1_g * i2_g)
        )
        gravity_barrier_g = radial_depth_g
        gravity_minimum_g = -radial_depth_g
        if inputs.include_gravity:
            gravity_barrier_g, gravity_minimum_g, _ = gaussian_gravity_trap(
                radial_depth_g, waist_g * 1e-6, mass
            )
            effective_barrier_j = min(
                effective_barrier_j, gravity_barrier_g
            )
        alive_now = int(positions.shape[0])
        if alive_now:
            relative = velocities - velocity_g * z_axis
            temperature = _kinetic_temperature_uK(relative, mass)
            kinetic = 0.5 * mass * (relative * relative).sum(axis=1)
            axial_excitation = kinetic + state_potential + axial_modulation_g
            bound_mask = axial_excitation < (
                axial_modulation_g * barrier_fraction_g
            )
            if inputs.include_gravity:
                radial_excitation = (
                    kinetic
                    + state_potential
                    + mass * GRAVITY * positions[:, 1]
                    - gravity_minimum_g
                )
                bound_mask &= radial_excitation < gravity_barrier_g
                bound_mask &= gravity_barrier_g > 0.0
            bound = float(xp.mean(bound_mask))
        else:
            temperature = float("nan")
            bound = 0.0
        interval = grid_time_s - previous_record_time
        lost = previous_record_alive - alive_now
        loss_rate = (
            lost / previous_record_alive / interval
            if lost > 0 and previous_record_alive > 0 and interval > 0.0
            else 0.0
        )
        time_out.append(grid_time_s * 1e3)
        stage_out.append(stage_g)
        position_out.append(position_g)
        velocity_out.append(velocity_g)
        acceleration_out.append(acceleration_g)
        frequency_out.append(
            2.0 * velocity_g / (wavelength_nm * 1e-9) * 1e-6
            if inputs.control_waveform is None
            else float(
                inputs.control_waveform.sample(grid_time_s)[
                    "aom_frequency_difference_mhz"
                ]
            )
        )
        waist_out.append(waist_g)
        power_out.append(source_power_g)
        barrier_out.append(effective_barrier_j / BOLTZMANN * 1e6)
        temperature_out.append(temperature)
        retention_out.append(alive_now / particle_count)
        bound_out.append(bound)
        scattering_out.append(total_scattering_events / particle_count)
        loss_rate_out.append(loss_rate)
        previous_record_alive = alive_now
        previous_record_time = grid_time_s

    # t=0 的初始力与首条快照。
    i1, i2, w1, w2, _, _ = _optics_at(0.0, 0.0)
    potential, force, _, _ = _double_beam_potential_and_force(
        positions,
        intensity1_w_m2=i1,
        intensity2_w_m2=i2,
        waist1_m=w1,
        waist2_m=w2,
        wave_number_m=wave_number,
        lattice_position_m=0.0,
        phase_rad=0.0,
        potential_per_intensity_j=potential_per_intensity,
    )
    force[:, 1] += gravity_force_y
    snapshot_times = _time_grid(inputs, timing)
    _record(0.0, potential)
    snapshot_index = 1

    # Velocity-Verlet：半步速度、整步位置、新力、半步速度；散射反冲在
    # 每个完整动力学步之后按局域 Poisson 事件加入；默认每 200 步（以及
    # 末步）按瞬时倾斜势垒剔除逃逸轨迹（链式 MC 模式经
    # escape_check_interval_steps/escape_lenient 改为每步 + 未缩减全深）。
    if xp is np:
        for step in range(1, integration_steps + 1):
            velocities += 0.5 * time_step_s * force / mass
            positions += time_step_s * velocities
            time_s = step * time_step_s
            lattice_position, lattice_velocity, acceleration, _ = _kinematics(
                min(time_s, total_time),
                inputs,
                timing,
            )
            i1, i2, w1, w2, _, _ = _optics_at(
                lattice_position, time_s
            )
            potential, force, local_forward, local_incoherent = (
                _double_beam_potential_and_force(
                    positions,
                    intensity1_w_m2=i1,
                    intensity2_w_m2=i2,
                    waist1_m=w1,
                    waist2_m=w2,
                    wave_number_m=wave_number,
                    lattice_position_m=lattice_position,
                    phase_rad=0.0,
                    potential_per_intensity_j=potential_per_intensity,
                )
            )
            force[:, 1] += gravity_force_y
            velocities += 0.5 * time_step_s * force / mass

            if include_scattering:
                total_scattering_events += _apply_scattering_kicks(
                    velocities,
                    local_forward_w_m2=local_forward,
                    local_incoherent_w_m2=local_incoherent,
                    rate_per_intensity_s=scattering_per_intensity,
                    time_step_s=time_step_s,
                    wave_number_m=wave_number,
                    atom_mass_kg=mass,
                    rng=kick_rng,
                )

            if step % escape_interval_steps == 0 or step == integration_steps:
                axial_modulation = (
                    potential_per_intensity * 4.0 * math.sqrt(i1 * i2)
                )
                if escape_lenient:
                    # 宽容判据：未缩减全深（不乘 F(a/a_c) 倾斜势垒缩减
                    # 因子）；下方重力径向下坡鞍点判据保持不变。
                    effective_barrier = axial_modulation
                else:
                    critical = axial_modulation * wave_number / mass
                    effective_barrier = axial_modulation * (
                        tilted_lattice_barrier_fraction(acceleration, critical)
                    )
                relative = velocities - lattice_velocity * z_axis
                kinetic = 0.5 * mass * (relative * relative).sum(axis=1)
                axial_excitation = kinetic + potential + axial_modulation
                alive = axial_excitation < effective_barrier
                if inputs.include_gravity:
                    radial_depth = potential_per_intensity * (
                        i1 + i2 + 2.0 * math.sqrt(i1 * i2)
                    )
                    gravity_barrier, gravity_minimum, _ = gaussian_gravity_trap(
                        radial_depth, _optics_at(lattice_position, time_s)[4] * 1e-6, mass
                    )
                    radial_excitation = (
                        kinetic
                        + potential
                        + mass * GRAVITY * positions[:, 1]
                        - gravity_minimum
                    )
                    alive &= radial_excitation < gravity_barrier
                    alive &= gravity_barrier > 0.0
                if not bool(xp.all(alive)):
                    positions = positions[alive]
                    velocities = velocities[alive]
                    potential = potential[alive]
                    force = force[alive]
                    if not positions.shape[0]:
                        break

            while (
                snapshot_index < len(snapshot_times)
                and snapshot_times[snapshot_index] <= time_s + 1e-12
            ):
                _record(float(snapshot_times[snapshot_index]), potential)
                snapshot_index += 1
    else:
        # GPU：整个 velocity-Verlet 步融合为单个 mega-step kernel
        # （就地更新粒子数组，每步一次 kernel 启动）；几何插值系数
        # 逐步在 host 预计算。逃逸剔除后粒子数变化，需重建列视图。
        step_kernel = _get_fused_leg_step_kernel()
        half_dt_over_mass = 0.5 * time_step_s / mass
        scatter_counts = xp.zeros(positions.shape[0], dtype=xp.int64)
        scattered_base = 0
        p0 = positions[:, 0]
        p1 = positions[:, 1]
        p2 = positions[:, 2]
        v0 = velocities[:, 0]
        v1 = velocities[:, 1]
        v2 = velocities[:, 2]
        f0 = force[:, 0]
        f1 = force[:, 1]
        f2 = force[:, 2]
        for step in range(1, integration_steps + 1):
            time_s = step * time_step_s
            lattice_position, lattice_velocity, acceleration, _ = _kinematics(
                min(time_s, total_time),
                inputs,
                timing,
            )
            i1, i2, w1, w2, _, _ = _optics_at(
                lattice_position, time_s
            )
            potential, local_forward, local_incoherent = step_kernel(
                p0,
                p1,
                p2,
                v0,
                v1,
                v2,
                f0,
                f1,
                f2,
                *_double_beam_step_coefficients(
                    intensity1_w_m2=i1,
                    intensity2_w_m2=i2,
                    waist1_m=w1,
                    waist2_m=w2,
                    wave_number_m=wave_number,
                    lattice_position_m=lattice_position,
                    phase_rad=0.0,
                    potential_per_intensity_j=potential_per_intensity,
                ),
                gravity_force_y,
                half_dt_over_mass,
                time_step_s,
            )

            if include_scattering:
                # 前向吸收概率逐粒子（局域前向/非相干强度比）；固定事件
                # 槽融合实现，避免逐事件设备同步。
                _scattering_kicks_gpu(
                    velocities,
                    shape1=local_incoherent,
                    coefficient1_s=scattering_per_intensity,
                    time_step_s=time_step_s,
                    axis2_0=0.0,
                    axis2_1=0.0,
                    axis2_2=1.0,
                    forward_probability=(
                        local_forward / local_incoherent
                    ),
                    recoil_m_s=HBAR * wave_number / mass,
                    rng=kick_rng,
                    accumulate_counts=scatter_counts,
                )

            if (
                step % _ESCAPE_CHECK_INTERVAL_STEPS == 0
                or step == integration_steps
            ):
                axial_modulation = (
                    potential_per_intensity * 4.0 * math.sqrt(i1 * i2)
                )
                critical = axial_modulation * wave_number / mass
                effective_barrier = axial_modulation * (
                    tilted_lattice_barrier_fraction(acceleration, critical)
                )
                relative = velocities - lattice_velocity * z_axis
                kinetic = 0.5 * mass * (relative * relative).sum(axis=1)
                axial_excitation = kinetic + potential + axial_modulation
                alive = axial_excitation < effective_barrier
                if inputs.include_gravity:
                    radial_depth = potential_per_intensity * (
                        i1 + i2 + 2.0 * math.sqrt(i1 * i2)
                    )
                    gravity_barrier, gravity_minimum, _ = gaussian_gravity_trap(
                        radial_depth, _optics_at(lattice_position, time_s)[4] * 1e-6, mass
                    )
                    radial_excitation = (
                        kinetic
                        + potential
                        + mass * GRAVITY * positions[:, 1]
                        - gravity_minimum
                    )
                    alive &= radial_excitation < gravity_barrier
                    alive &= gravity_barrier > 0.0
                if not bool(xp.all(alive)):
                    # 被剔除粒子的散射计数并入 host 基数，保持与 CPU
                    # 路径一致的累计口径。
                    scattered_base += int(scatter_counts[~alive].sum())
                    scatter_counts = scatter_counts[alive]
                    positions = positions[alive]
                    velocities = velocities[alive]
                    potential = potential[alive]
                    force = force[alive]
                    if not positions.shape[0]:
                        break
                    p0 = positions[:, 0]
                    p1 = positions[:, 1]
                    p2 = positions[:, 2]
                    v0 = velocities[:, 0]
                    v1 = velocities[:, 1]
                    v2 = velocities[:, 2]
                    f0 = force[:, 0]
                    f1 = force[:, 1]
                    f2 = force[:, 2]

            # 散射计数驻留 GPU，仅在需要记录快照时同步一次。
            if (
                snapshot_index < len(snapshot_times)
                and snapshot_times[snapshot_index] <= time_s + 1e-12
            ):
                total_scattering_events = scattered_base + int(
                    scatter_counts.sum()
                )
            while (
                snapshot_index < len(snapshot_times)
                and snapshot_times[snapshot_index] <= time_s + 1e-12
            ):
                _record(float(snapshot_times[snapshot_index]), potential)
                snapshot_index += 1

    # 提前全灭（或末步浮点舍入）时补齐剩余快照。
    if xp is not np:
        total_scattering_events = scattered_base + int(scatter_counts.sum())
    while snapshot_index < len(snapshot_times):
        _record(float(snapshot_times[snapshot_index]), potential)
        snapshot_index += 1

    survivor_count = int(positions.shape[0])
    final_retention = survivor_count / particle_count
    # Jeffreys Beta(1/2, 1/2) 后验标准差在 k=0 或 k=N 时仍为有限值。
    posterior_alpha = survivor_count + 0.5
    posterior_beta = particle_count - survivor_count + 0.5
    posterior_sum = posterior_alpha + posterior_beta
    standard_error = math.sqrt(
        posterior_alpha
        * posterior_beta
        / (posterior_sum**2 * (posterior_sum + 1.0))
    )

    final_temperature = temperature_out[-1]
    point = L1DesignPoint(
        detuning_ghz=detuning_ghz,
        handover_source_power_w=handover_source_power_w,
        start_source_power_w=start_source_power,
        wavelength_nm=wavelength_nm,
        depth_uK=handover_depth_uK,
        scattering_rate_s=handover_scattering_rate_s,
        final_temperature_uK=final_temperature,
        final_temperature_rise_uK=(
            final_temperature - inputs.initial_temperature_uK
        ),
        final_retention_fraction=final_retention,
        total_retention_from_mot_fraction=(
            inputs.loading_efficiency * final_retention
        ),
        final_atom_number=inputs.initial_atom_number * final_retention,
        cumulative_scattering_events=scattering_out[-1],
        maximum_loss_rate_s=max(loss_rate_out),
        feasible_hardware_point=feasible,
        initial_temperature_uK=inputs.initial_temperature_uK,
        initial_atom_number=inputs.initial_atom_number,
        actual_time_step_us=time_step_s * 1e6,
    )
    trace = L1TransportTrace(
        point=point,
        time_ms=tuple(time_out),
        stage=tuple(stage_out),
        position_m=tuple(position_out),
        velocity_m_s=tuple(velocity_out),
        acceleration_m_s2=tuple(acceleration_out),
        aom_frequency_difference_mhz=tuple(frequency_out),
        waist_um=tuple(waist_out),
        source_power_w=tuple(power_out),
        effective_barrier_uK=tuple(barrier_out),
        temperature_uK=tuple(temperature_out),
        temperature_rise_uK=tuple(
            value - inputs.initial_temperature_uK for value in temperature_out
        ),
        retention_fraction=tuple(retention_out),
        bound_fraction=tuple(bound_out),
        cumulative_scattering_events=tuple(scattering_out),
        instantaneous_loss_rate_s=tuple(loss_rate_out),
        retention_standard_error=standard_error,
    )
    if not return_final_ensemble:
        return trace
    if survivor_count == 0:
        return trace, None
    positions_h = positions if xp is np else positions.get()
    velocities_h = velocities if xp is np else velocities.get()
    return trace, ParticleEnsemble(
        positions_m=np.asarray(positions_h).copy(),
        velocities_m_s=np.asarray(velocities_h).copy(),
        frame="transport_lab",
    )
