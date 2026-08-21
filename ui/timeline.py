"""全链路时间轴组装与采样（纯 Python，无 Qt，可单测）。

把 ``FullChainPointSimulation`` 的三段结果拼成一条统一时间轴：
L1 段取 ``transport_trace``，handover 段位置恒定、速度为零，L2 段取
``l2_result.leg_trace`` 并平移到 handover 之后。温度与相对 MOT 留存率
直接取 ``combined_trace``（拼接约定与 ``full_chain`` 模块一致，段交界
处允许重复时间戳）。
"""

from __future__ import annotations

import numpy as np

from continuous_loading.full_chain import FullChainPointSimulation


def build_timeline(simulation: FullChainPointSimulation) -> dict[str, object]:
    """从单点全链路结果组装全程时间轴。

    返回字典的数组字段长度一致：``time_ms``、``position_m``（沿光路
    累计位置，L2 段从 L1 末端继续）、``temperature_uK``、
    ``retention_from_mot``、``velocity_m_s``、``phase``；另附
    handover/L2 段的开始结束时间。时间轴从 L1 起点开始（静止晶格
    热平衡初态，无装载阶段）。
    """
    combined = simulation.combined_trace
    l1_trace = simulation.l1_handover_simulation.transport_trace
    l2_result = simulation.l2_result

    time_ms = [float(value) for value in combined.time_ms]
    phase = list(combined.phase)
    temperature_uK = [float(value) for value in combined.temperature_uK]
    retention_from_mot = [float(value) for value in combined.retention_from_mot]

    handover_position_m = float(l1_trace.position_m[-1])
    position_m = [float(value) for value in l1_trace.position_m]
    velocity_m_s = [float(value) for value in l1_trace.velocity_m_s]

    handover_count = phase.count("handover")
    position_m.extend([handover_position_m] * handover_count)
    velocity_m_s.extend([0.0] * handover_count)

    l2_count = phase.count("L2 transport")
    if l2_count:
        if l2_result is None:
            raise ValueError("拼接轨迹含 L2 段但缺少 l2_result")
        leg = l2_result.leg_trace
        if len(leg.time_ms) != l2_count:
            raise ValueError("L2 腿轨迹与拼接轨迹长度不一致")
        position_m.extend(handover_position_m + float(value) for value in leg.position_m)
        velocity_m_s.extend(float(value) for value in leg.velocity_m_s)

    total = len(time_ms)
    if not (len(position_m) == len(velocity_m_s) == total):
        raise ValueError("时间轴各字段长度不一致")
    return {
        "time_ms": np.asarray(time_ms),
        "position_m": np.asarray(position_m),
        "temperature_uK": np.asarray(temperature_uK),
        "retention_from_mot": np.asarray(retention_from_mot),
        "velocity_m_s": np.asarray(velocity_m_s),
        "phase": phase,
        "handover_start_ms": float(combined.handover_start_ms),
        "handover_end_ms": float(combined.handover_end_ms),
        "l2_start_ms": float(combined.l2_start_ms),
        "l2_end_ms": float(combined.l2_end_ms),
        "interface_mode": simulation.interface_mode,
    }


def sample_timeline(
    timeline: dict[str, object], t_ms: float
) -> dict[str, float | str]:
    """线性插值采样 ``t_ms`` 时刻的状态；越界时夹到端点。

    ``phase`` 取时间上最近样本的阶段（段交界处的重复时间戳取哪一段
    均可，物理上只相差一个瞬时切换）。
    """
    time_ms = np.asarray(timeline["time_ms"])
    t = float(min(max(t_ms, time_ms[0]), time_ms[-1]))
    sample: dict[str, float | str] = {"time_ms": t}
    for key in ("position_m", "temperature_uK", "retention_from_mot", "velocity_m_s"):
        sample[key] = float(np.interp(t, time_ms, np.asarray(timeline[key])))
    index = int(np.searchsorted(time_ms, t, side="left"))
    if index >= len(time_ms):
        index = len(time_ms) - 1
    elif index > 0 and abs(time_ms[index - 1] - t) <= abs(time_ms[index] - t):
        index -= 1
    sample["phase"] = str(timeline["phase"][index])
    return sample


# 全程运动学/光路字段：L1/L2 段取各自 transport trace，handover 段无定义。
_FULL_SERIES_FIELDS = (
    "velocity_m_s",
    "acceleration_m_s2",
    "aom_frequency_difference_mhz",
    "waist_um",
    "source_power_w",
)


def build_full_series(simulation: FullChainPointSimulation) -> dict[str, object]:
    """组装覆盖全程 0→t_end 的运动学与光路时间序列。

    时间轴与 ``build_timeline`` 一致；handover 段这些量没有定义，
    用 NaN 占位（画线时自动断开）。L2 段取 ``l2_result.leg_trace``
    的同名字段（其时间已含 l2_start 偏移，与拼接时间轴对齐）。
    """
    timeline = build_timeline(simulation)
    l1_trace = simulation.l1_handover_simulation.transport_trace
    l2_result = simulation.l2_result
    phase = list(timeline["phase"])
    handover_count = phase.count("handover")
    l2_count = phase.count("L2 transport")
    if l2_count and l2_result is None:
        raise ValueError("拼接轨迹含 L2 段但缺少 l2_result")

    series: dict[str, object] = {
        "time_ms": timeline["time_ms"],
        "phase": phase,
        "handover_start_ms": timeline["handover_start_ms"],
        "handover_end_ms": timeline["handover_end_ms"],
        "l2_start_ms": timeline["l2_start_ms"],
        "l2_end_ms": timeline["l2_end_ms"],
        "interface_mode": timeline["interface_mode"],
    }
    for field_name in _FULL_SERIES_FIELDS:
        values = [float(value) for value in getattr(l1_trace, field_name)]
        values.extend([float("nan")] * handover_count)
        if l2_count:
            leg = l2_result.leg_trace
            values.extend(float(value) for value in getattr(leg, field_name))
        if len(values) != len(phase):
            raise ValueError(f"全程序列 {field_name} 与时间轴长度不一致")
        series[field_name] = np.asarray(values)
    series["beam_diameter_um"] = 2.0 * np.asarray(series["waist_um"])
    return series
