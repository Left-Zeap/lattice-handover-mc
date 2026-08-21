"""光场计算层：L1 腿 / handover / L2 腿统一的"预计算时序 + 查询"接口。

全链路 Monte Carlo 要求光场 I(x,y,z,t) 在**传播开始前一次性算好**，
之后只按 ``(positions, step_index)`` 查询势/力/局域强度。本模块把三段
现有的逐步取光学量逻辑归一为三类时序表，**物理公式一行不改**：

- 腿段（L1/L2）：``LegFieldTimeline`` 逐步存放
  ``t, z_L, v_L, a, I1, I2, w1, w2``。生成逻辑复用
  ``l1_transport.l1_timing`` + ``l1_transport._kinematics``（梯形/
  minimum-jerk/实测波形时序）与 ``transport_mc._leg_optics_profile``
  + ``transport_mc._leg_optics_at``（conveyor、L1 标定高斯或兼容线性剖面；
  ``control_waveform`` 的 waist/source_power/delivery 覆盖逻辑原样
  保留）。数组长度 = 步数+1，索引即 ``simulate_leg_monte_carlo``
  CPU 主循环的步号：step 0 对应 t=0 初始力，step k 对应第 k 个
  velocity-Verlet 步末时刻 ``t=k·dt``，取值与该主循环逐步调用
  ``_kinematics(min(t, T))`` + ``_leg_optics_at(z_L, t)`` 的结果
  逐位一致。步长取 ``inputs.transport_time_step_us`` 经
  ``transport_mc._stable_leg_step_s`` 精度守卫（ω_z·dt ≤ 1）钳制
  后按总时长归整（与主循环同一 ``_leg_integration_schedule``）。
- handover 段：``HandoverFieldTimeline`` 逐步存放
  ``t, fraction1, fraction2, phase_control``（默认线性 ramp 或
  ``HandoverControlWaveform.sampled_arrays``，即 handover.py 主
  循环的 ramp 预计算），外加两晶格轴 e1/e2、束偏移、满深度、束腰、
  波数、晶格速度和静相位。步长取
  ``min(time_step_us, _stable_handover_step_s)`` 并按总时长归整
  （与 ``run_handover_monte_carlo`` 同一公式）。

查询接口：

- ``leg_potential_and_force`` → ``(V, F, local_forward,
  local_incoherent)``，内部走 ``transport_mc``
  的 ``_double_beam_potential_and_force``（驻波相位固定 φ=0，与
  主循环一致；散射反冲用的局域前向/非相干强度一并返回）。
- ``handover_potential_and_force`` → ``(V, F, shape1, shape2)``，
  内部两次调 ``handover._lattice_potential_force`` 求和，语义同
  ``run_handover_monte_carlo`` 的 ``combined_force``（深度按
  fraction 缩放、L2 相位叠加 phase_control）。

口径说明：光场层只含光势力——重力（腿段 ``F_y-=mg``、handover 段
同理）与光子散射反冲仍由积分器按现行为叠加，不进时序表；
handover 的逐粒子随机相对相位不属于光场时序，需要时用
``dataclasses.replace(timeline, phase2_rad=逐粒子数组)`` 替换。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .constants import BOLTZMANN
from .handover import (
    HandoverParameters,
    _lattice_potential_force,
    _stable_handover_step_s,
    _unit_axes,
)
from .l1_transport import (
    L1TransportInputs,
    _atom_from_label,
    _kinematics,
    l1_timing,
)
from .l2_transport import (
    L2TransportInputs,
    l2_end_source_power_w,
    l2_leg_inputs,
)
from .transport_mc import (
    _double_beam_potential_and_force,
    _leg_integration_schedule,
    _leg_optics_at,
    _leg_optics_profile,
)


@dataclass(frozen=True)
class LegFieldTimeline:
    """一条运输腿（L1 或 L2）的逐步光场时序表（SI）。

    数组长度 = integration_steps + 1，索引即主循环步号（约定见模块
    docstring）；``wave_number_m``/``time_step_s`` 为该腿常量。
    """

    step_times_s: np.ndarray
    z_lattice_m: np.ndarray
    lattice_velocity_m_s: np.ndarray
    acceleration_m_s2: np.ndarray
    intensity1_w_m2: np.ndarray
    intensity2_w_m2: np.ndarray
    waist1_m: np.ndarray
    waist2_m: np.ndarray
    wave_number_m: float
    time_step_s: float


@dataclass(frozen=True)
class HandoverFieldTimeline:
    """handover 段的逐步光场时序表（SI）。

    ``fraction1/2``、``phase_control_rad`` 长度 = 步数+1；深度字段为
    满深度（查询时乘 fraction）。``phase2_rad`` 为标量静相位或逐粒子
    (N,) 数组（随机相对相位，见模块 docstring）。
    """

    step_times_s: np.ndarray
    fraction1: np.ndarray
    fraction2: np.ndarray
    phase_control_rad: np.ndarray
    axis1: np.ndarray
    axis2: np.ndarray
    beam_offset1_m: np.ndarray
    beam_offset2_m: np.ndarray
    phase1_rad: float
    phase2_rad: float | np.ndarray
    depth1_j: float
    depth2_j: float
    waist1_m: float
    waist2_m: float
    wave_number_m: float
    lattice1_velocity_m_s: float
    lattice2_velocity_m_s: float
    time_step_s: float


@dataclass(frozen=True)
class ChainLightField:
    """L1→handover→L2 三段统一光场（传播开始前一次性预计算）。"""

    l1: LegFieldTimeline
    handover: HandoverFieldTimeline
    l2: LegFieldTimeline

    @classmethod
    def precompute(
        cls,
        transport_inputs: L1TransportInputs,
        detuning_ghz: float,
        handover_source_power_w: float,
        handover_parameters: HandoverParameters,
        l2_inputs: L2TransportInputs,
    ) -> "ChainLightField":
        """一次性算好三段光场时序表。

        L2 腿输入经 ``l2_transport.l2_leg_inputs`` 构造，固定源端功率
        经 ``l2_end_source_power_w`` 取得；L2 腿的捕获温度/原子数在
        预计算时刻尚未知，用 ``transport_inputs`` 的初值占位——它们只
        进入 trace 记账字段，不影响任何光场时序。
        """
        l1_timeline = build_leg_field_timeline(
            transport_inputs,
            detuning_ghz,
            handover_source_power_w,
        )
        handover_timeline = build_handover_field_timeline(handover_parameters)
        end_source_power_w = l2_end_source_power_w(
            transport_inputs,
            l2_inputs,
            handover_source_power_w,
        )
        l2_leg = l2_leg_inputs(
            transport_inputs,
            l2_inputs,
            transport_inputs.initial_temperature_uK,
            transport_inputs.initial_atom_number,
        )
        l2_timeline = build_leg_field_timeline(
            l2_leg,
            detuning_ghz,
            end_source_power_w,
        )
        return cls(l1=l1_timeline, handover=handover_timeline, l2=l2_timeline)


def build_leg_field_timeline(
    inputs: L1TransportInputs,
    detuning_ghz: float,
    handover_source_power_w: float,
) -> LegFieldTimeline:
    """预计算一条运输腿的逐步光场时序表（生成口径见模块 docstring）。"""
    atom = _atom_from_label(inputs.atom_label)
    wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning_ghz)
    wave_number_m = 2.0 * math.pi / (wavelength_nm * 1e-9)
    timing = l1_timing(inputs)
    total_time = timing.total_time_s
    profile = _leg_optics_profile(
        inputs,
        wavelength_nm,
        handover_source_power_w,
    )
    # 步数/步长与 transport_mc 主循环同一来源（含 ω_z·dt ≤ 1 精度守卫），
    # 保证时序表与传播积分逐位一致。
    integration_steps, time_step_s = _leg_integration_schedule(
        inputs, atom, wavelength_nm, profile, total_time
    )

    count = integration_steps + 1
    step_times_s = np.arange(count, dtype=float) * time_step_s
    z_lattice_m = np.empty(count, dtype=float)
    lattice_velocity_m_s = np.empty(count, dtype=float)
    acceleration_m_s2 = np.empty(count, dtype=float)
    intensity1_w_m2 = np.empty(count, dtype=float)
    intensity2_w_m2 = np.empty(count, dtype=float)
    waist1_m = np.empty(count, dtype=float)
    waist2_m = np.empty(count, dtype=float)
    # 与 CPU 主循环同一调用链：_kinematics(min(t, T)) 取运动学，
    # _leg_optics_at(z_L, t) 取光学量（t 不夹持，与主循环一致）。
    for step in range(count):
        time_s = float(step_times_s[step])
        position, velocity, acceleration, _ = _kinematics(
            min(time_s, total_time),
            inputs,
            timing,
        )
        i1, i2, w1, w2, _, _ = _leg_optics_at(
            inputs,
            profile,
            handover_source_power_w,
            position,
            time_s,
        )
        z_lattice_m[step] = position
        lattice_velocity_m_s[step] = velocity
        acceleration_m_s2[step] = acceleration
        intensity1_w_m2[step] = i1
        intensity2_w_m2[step] = i2
        waist1_m[step] = w1
        waist2_m[step] = w2
    return LegFieldTimeline(
        step_times_s=step_times_s,
        z_lattice_m=z_lattice_m,
        lattice_velocity_m_s=lattice_velocity_m_s,
        acceleration_m_s2=acceleration_m_s2,
        intensity1_w_m2=intensity1_w_m2,
        intensity2_w_m2=intensity2_w_m2,
        waist1_m=waist1_m,
        waist2_m=waist2_m,
        wave_number_m=wave_number_m,
        time_step_s=time_step_s,
    )


def build_handover_field_timeline(
    parameters: HandoverParameters,
) -> HandoverFieldTimeline:
    """预计算 handover 段的逐步光场时序表（生成口径见模块 docstring）。"""
    e1, e2, e_out = _unit_axes(parameters.crossing_angle_deg)
    wave_number_m = 2.0 * math.pi / (parameters.wavelength_nm * 1e-9)
    duration_s = parameters.duration_ms * 1e-3
    requested_step_s = min(
        parameters.time_step_us * 1e-6,
        _stable_handover_step_s(parameters),
    )
    integration_steps = max(1, math.ceil(duration_s / requested_step_s))
    time_step_s = duration_s / integration_steps
    step_times_s = np.arange(integration_steps + 1, dtype=float) * time_step_s
    if parameters.control_waveform is None:
        fraction2 = step_times_s / duration_s
        fraction1 = 1.0 - fraction2
        phase_control = np.zeros_like(step_times_s)
    else:
        fraction1, fraction2, phase_control = (
            parameters.control_waveform.sampled_arrays(step_times_s)
        )

    distance_offset_m = (
        parameters.lattice1_distance_cm - parameters.optimal_distance_cm
    ) * 1e-2
    cloud_center = distance_offset_m * e1
    phase1 = -wave_number_m * float(cloud_center @ e1)
    beam_offset2 = parameters.l2_transverse_offset_um * 1e-6 * e_out
    return HandoverFieldTimeline(
        step_times_s=step_times_s,
        fraction1=np.asarray(fraction1, dtype=float),
        fraction2=np.asarray(fraction2, dtype=float),
        phase_control_rad=np.asarray(phase_control, dtype=float),
        axis1=np.asarray(e1, dtype=float),
        axis2=np.asarray(e2, dtype=float),
        beam_offset1_m=np.zeros(3, dtype=float),
        beam_offset2_m=np.asarray(beam_offset2, dtype=float),
        phase1_rad=phase1,
        phase2_rad=parameters.relative_phase_rad,
        depth1_j=parameters.depth1_uK * 1e-6 * BOLTZMANN,
        depth2_j=parameters.depth2_uK * 1e-6 * BOLTZMANN,
        waist1_m=parameters.waist1_um * 1e-6,
        waist2_m=parameters.waist2_um * 1e-6,
        wave_number_m=wave_number_m,
        lattice1_velocity_m_s=parameters.lattice1_velocity_m_s,
        lattice2_velocity_m_s=parameters.lattice2_velocity_m_s,
        time_step_s=time_step_s,
    )


def _checked_step(step_times_s: np.ndarray, step: int) -> int:
    index = int(step)
    if index < 0 or index >= step_times_s.shape[0]:
        raise IndexError(
            f"步号 {index} 超出时序表范围 [0, {step_times_s.shape[0] - 1}]"
        )
    return index


def leg_potential_and_force(
    timeline: LegFieldTimeline,
    step: int,
    positions_m: np.ndarray,
    potential_per_intensity_j: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按步号查询腿段双束势/力和局域强度（语义同主循环逐步调用）。

    返回 ``(V, F, local_forward, local_incoherent)``；驻波相位固定
    φ=0，重力不在光场层内（由积分器叠加）。
    """
    index = _checked_step(timeline.step_times_s, step)
    return _double_beam_potential_and_force(
        positions_m,
        intensity1_w_m2=float(timeline.intensity1_w_m2[index]),
        intensity2_w_m2=float(timeline.intensity2_w_m2[index]),
        waist1_m=float(timeline.waist1_m[index]),
        waist2_m=float(timeline.waist2_m[index]),
        wave_number_m=timeline.wave_number_m,
        lattice_position_m=float(timeline.z_lattice_m[index]),
        phase_rad=0.0,
        potential_per_intensity_j=potential_per_intensity_j,
    )


def handover_potential_and_force(
    timeline: HandoverFieldTimeline,
    step: int,
    positions_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按步号查询 handover 双晶格合成势/力（语义同 combined_force）。

    返回 ``(V, F, shape1, shape2)``：两次
    ``handover._lattice_potential_force`` 之和；重力不在光场层内
    （由积分器叠加）。
    """
    index = _checked_step(timeline.step_times_s, step)
    time_s = float(timeline.step_times_s[index])
    fraction1 = float(timeline.fraction1[index])
    fraction2 = float(timeline.fraction2[index])
    phase_control = float(timeline.phase_control_rad[index])
    potential1, force1, shape1 = _lattice_potential_force(
        positions_m,
        axis=timeline.axis1,
        beam_offset_m=timeline.beam_offset1_m,
        phase_rad=timeline.phase1_rad,
        axial_velocity_m_s=timeline.lattice1_velocity_m_s,
        time_s=time_s,
        wave_number_m=timeline.wave_number_m,
        waist_m=timeline.waist1_m,
        depth_j=timeline.depth1_j * fraction1,
    )
    potential2, force2, shape2 = _lattice_potential_force(
        positions_m,
        axis=timeline.axis2,
        beam_offset_m=timeline.beam_offset2_m,
        phase_rad=timeline.phase2_rad + phase_control,
        axial_velocity_m_s=timeline.lattice2_velocity_m_s,
        time_s=time_s,
        wave_number_m=timeline.wave_number_m,
        waist_m=timeline.waist2_m,
        depth_j=timeline.depth2_j * fraction2,
    )
    return potential1 + potential2, force1 + force2, shape1, shape2
