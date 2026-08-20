"""可选的 GPU（CuPy）计算后端。

默认后端是 CPU（NumPy），程序行为与无 GPU 环境完全一致。把
``compute_backend`` 设为 ``"gpu"`` 后，Monte Carlo 的粒子数组在内层
积分循环中驻留 GPU（CuPy/CUDA），采样、几何剖面等轻量前处理仍在
CPU 完成后一次性传入。GPU 内层循环做了两级融合：整个
velocity-Verlet 步合成单个 mega-step kernel（每步一次启动、就地
更新粒子数组），散射反冲用固定事件槽的融合 kernel（每步仅一次
标量同步，统计上与逐事件实现等价）；扫描场景还可经
``handover_batch.run_handover_monte_carlo_batch`` 把多个网格点摊平
成一次批量 GPU 调用。

注意：CPU 与 GPU 使用不同的随机数生成器，同 seed 的结果只在统计
意义上一致，不保证逐位一致；测试和结果引用都应区分后端。
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np


def cupy_available() -> bool:
    """检测 CuPy 与 CUDA 设备是否真正可用（不仅是已安装）。"""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 - CUDA 驱动缺失等任何初始化失败
        return False


def resolve_backend(requested: str) -> str:
    """校验计算后端请求；请求 gpu 但环境不可用时给出明确错误。"""
    if requested not in {"cpu", "gpu"}:
        raise ValueError("计算后端必须是 cpu 或 gpu")
    if requested == "gpu" and not cupy_available():
        raise ValueError(
            "请求了 GPU 计算后端，但未检测到可用的 CuPy/CUDA 环境。"
            "请安装 cupy-cuda13x（或与本机 CUDA 驱动匹配的 CuPy），"
            "或改用 cpu 后端"
        )
    return requested


def namespace(use_gpu: bool):
    """按后端返回数值模块（numpy 或 cupy）。"""
    if use_gpu:
        import cupy as cp

        return cp
    return np


def module_of(*arrays):
    """返回给定数组所属的数值模块（供后端无关的力/运动学函数使用）。

    先按类型判断，避免在纯 CPU 环境（尤其是进程池 worker）中 eager
    import cupy——多进程同时加载 CUDA DLL 可能耗尽 Windows 页面文件。
    """
    if isinstance(arrays[0], np.ndarray):
        return np
    import cupy as cp

    return cp.get_array_module(*arrays)


def scatter_add(xp, target, indices, values) -> None:
    """后端无关的 ``np.add.at`` 等价物（CuPy 用 cupyx.scatter_add）。"""
    if xp is np:
        np.add.at(target, indices, values)
    else:
        from cupyx import scatter_add as _scatter_add

        _scatter_add(target, indices, values)


def default_rng(xp, seed: int):
    """按后端创建随机数生成器（CPU/GPU 序列不同，仅统计一致）。"""
    return xp.random.default_rng(seed)


def scattering_rng_gpu(seed: int):
    """创建 GPU 散射反冲专用的随机数生成器（cupy RandomState）。

    不用 ``default_rng`` 的 Generator：实测本环境 ``Generator.poisson``
    每步约 0.3 ms 而 ``RandomState.poisson`` 约 0.05 ms，散射反冲在内
    层循环每步调用，分发开销是主要瓶颈之一。RandomState 序列与
    Generator 不同，同 seed 结果仅要求确定性与统计一致。
    """
    import cupy as cp

    return cp.random.RandomState(seed % 2**32)


_FUSED_KICK_RATES_KERNEL = None
_FUSED_KICK_RATES_SINGLE_KERNEL = None
_KICK_KERNEL_CACHE: dict[int, object] = {}

_TWO_PI = 2.0 * math.pi


def _get_kick_rates_kernel():
    """惰性创建散射率融合 kernel：返回 ``(λ·dt, 第一轴占比)``。

    ``coefficient*_dt`` 是 峰值率×斜坡分数×dt，可以是标量或逐粒子
    数组（批量路径），cupy.fuse 按实参类型分别编译缓存。总率为零的
    粒子占比为 NaN，但其 Poisson 计数恒为零，槽位掩码保证无影响。
    """
    global _FUSED_KICK_RATES_KERNEL
    if _FUSED_KICK_RATES_KERNEL is None:
        import cupy as cp

        @cp.fuse()
        def kernel(shape1, shape2, coefficient1_dt, coefficient2_dt):
            lam1 = shape1 * coefficient1_dt
            lam2 = shape2 * coefficient2_dt
            total = lam1 + lam2
            return total, lam1 / total

        _FUSED_KICK_RATES_KERNEL = kernel
    return _FUSED_KICK_RATES_KERNEL


def _get_kick_rates_single_kernel():
    """单吸收轴版散射率 kernel（运输腿：仅非相干强度和一项）。"""
    global _FUSED_KICK_RATES_SINGLE_KERNEL
    if _FUSED_KICK_RATES_SINGLE_KERNEL is None:
        import cupy as cp

        @cp.fuse()
        def kernel(shape1, coefficient1_dt):
            lam = shape1 * coefficient1_dt
            return lam, lam / lam

        _FUSED_KICK_RATES_SINGLE_KERNEL = kernel
    return _FUSED_KICK_RATES_SINGLE_KERNEL


def _get_kick_kernel(slots: int):
    """按事件槽数创建反冲组合融合 kernel（槽数在追踪期定值展开）。

    每个槽位：吸收轴二选一（第一轴恒为 ẑ=(0,0,1)，第二轴分量由参
    数给出）、前向/回程符号、各向同性自发辐射方向。辐射方向用
    ``z=2u-1, φ=2πv`` 均匀球面采样（纯逐元素，无归一化归约，分布
    与归一化高斯严格相同）。计数不足的槽位由 ``counts > j`` 掩码
    置零（counts 以浮点副本传入，规避 fusion 内 int32 数组与 int
    标量比较的弱类型提升 bug）。返回已乘反冲速度的三个分量
    ``(kick_x, kick_y, kick_z)``，由调用方就地加到速度列——实测本
    环境"循环展开 + 多个 ``[...]`` 写回"组合会触发
    CUDA_ERROR_NO_BINARY_FOR_GPU，故写回不放在 kernel 内。
    """
    if slots not in _KICK_KERNEL_CACHE:
        import cupy as cp

        @cp.fuse()
        def kernel(
            counts_f,
            ratio1,
            u_axis,
            u_sign,
            u_z,
            u_phi,
            forward_probability,
            recoil_m_s,
            axis2_0,
            axis2_1,
            axis2_2,
        ):
            kick0 = counts_f * 0.0
            kick1 = counts_f * 0.0
            kick2 = counts_f * 0.0
            for slot in range(slots):
                active = counts_f > slot
                sign = cp.where(u_sign[:, slot] < forward_probability, 1.0, -1.0)
                choose1 = u_axis[:, slot] < ratio1
                z = 2.0 * u_z[:, slot] - 1.0
                radius = cp.sqrt(1.0 - z * z)
                phi = u_phi[:, slot] * _TWO_PI
                dir0 = radius * cp.cos(phi)
                dir1 = radius * cp.sin(phi)
                dir2 = z
                abs0 = cp.where(choose1, 0.0, axis2_0) * sign
                abs1 = cp.where(choose1, 0.0, axis2_1) * sign
                abs2 = cp.where(choose1, 1.0, axis2_2) * sign
                kick0 = kick0 + (abs0 - dir0) * active
                kick1 = kick1 + (abs1 - dir1) * active
                kick2 = kick2 + (abs2 - dir2) * active
            return (
                recoil_m_s * kick0,
                recoil_m_s * kick1,
                recoil_m_s * kick2,
            )

        _KICK_KERNEL_CACHE[slots] = kernel
    return _KICK_KERNEL_CACHE[slots]


def scattering_kicks_gpu(
    velocities_m_s,
    *,
    shape1,
    coefficient1_s: float,
    time_step_s: float,
    axis2_0: float,
    axis2_1: float,
    axis2_2: float,
    forward_probability,
    recoil_m_s,
    rng,
    accumulate_counts,
    shape2=None,
    coefficient2_s: float = 0.0,
) -> None:
    """GPU 散射反冲（固定事件槽，避免逐事件 device→host 同步）。

    逐事件实现（``handover._apply_scattering_kicks``，CPU 路径仍在用）
    每步都要 ``nonzero`` + 取事件数，强制设备同步；本实现改为：融合
    kernel 算出逐粒子 Poisson 强度 λ·dt → ``RandomState.poisson`` 抽
    逐粒子计数 → 每步仅一次标量同步取最大计数 → 按 ``(粒子数, 最大
    计数)`` 固定形状一次抽取全部事件随机量，槽位用掩码置零，反冲施
    加也融合为单个 kernel。统计上与逐事件实现严格等价（每个粒子的
    反冲仍是其 Poisson 计数个独立事件的矢量和；自发辐射方向改用均
    匀球面的直接采样，分布不变），但 RNG 消耗顺序不同，与同 seed
    的 CPU 结果仅统计一致。

    ``shape1``/``shape2`` 为逐粒子相对光强（双轴吸收）或局域非相干
    强度和（单轴，此时 ``shape2=None``）；``coefficient*_s`` 为 峰值
    散射率×斜坡分数，可以是标量或逐粒子数组（批量路径）。
    ``forward_probability``（前向吸收概率）与 ``recoil_m_s``（反冲
    速度 ħk/m）同样可以是标量或逐粒子数组。``accumulate_counts``
    为 (M,) int64 GPU 数组，累加逐粒子散射事件数（供事后统计，
    免去逐步 host 归约）。``rng`` 由 ``scattering_rng_gpu`` 创建。
    """
    coefficient1_dt = coefficient1_s * time_step_s
    if shape2 is None:
        lam, ratio1 = _get_kick_rates_single_kernel()(shape1, coefficient1_dt)
    else:
        lam, ratio1 = _get_kick_rates_kernel()(
            shape1,
            shape2,
            coefficient1_dt,
            coefficient2_s * time_step_s,
        )
    counts = rng.poisson(lam)
    max_count = int(counts.max())
    if max_count == 0:
        return
    accumulate_counts += counts
    # 规避 fusion 内 int32 数组与 int 标量比较的弱类型提升 bug
    # （numpy.int32 无 .kind），比较用浮点副本。
    counts_f = counts * 1.0
    particle_count = counts.shape[0]
    kick0, kick1, kick2 = _get_kick_kernel(max_count)(
        counts_f,
        ratio1,
        rng.random_sample((particle_count, max_count)),
        rng.random_sample((particle_count, max_count)),
        rng.random_sample((particle_count, max_count)),
        rng.random_sample((particle_count, max_count)),
        forward_probability,
        recoil_m_s,
        axis2_0,
        axis2_1,
        axis2_2,
    )
    velocities_m_s[:, 0] += kick0
    velocities_m_s[:, 1] += kick1
    velocities_m_s[:, 2] += kick2


def rng_standard_normal(rng, size):
    """后端无关的标准正态抽样。

    CuPy 的 ``Generator`` 没有 NumPy 的 ``normal`` 别名，两边统一用
    ``standard_normal``（NumPy Generator 同样支持）。
    """
    return rng.standard_normal(size=size)


def to_host(array):
    """把后端数组取回 CPU（已是 NumPy 数组时原样返回）。"""
    if isinstance(array, np.ndarray):
        return array
    return array.get()
