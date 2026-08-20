"""纯 Python 控制层：把界面表单参数映射到计算库调用。

本模块不导入 Qt，便于在无显示环境下直接测试。表单参数是一个扁平的
``dict``，键名集中在 ``default_form_params`` 中定义；界面表单和测试
共用这一套键，单位后缀与计算库保持一致（``_uK``、``_ghz`` 等）。
"""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from typing import Callable

import numpy as np

from continuous_loading.atomic import CS133, RB87, AlkaliAtom
from continuous_loading.cloud_sigma_scan import (
    CloudSigmaScanInputs,
    CloudSigmaScanResult,
    analyze_cloud_sigma_scan,
)
from continuous_loading.control_waveforms import (
    HandoverControlWaveform,
    TransportControlWaveform,
)
from continuous_loading.full_chain import (
    FullChainInputs,
    FullChainPointSimulation,
    FullChainScanResult,
    analyze_full_chain_scan,
    simulate_full_chain_point,
)
from continuous_loading.l1_handover import L1HandoverInputs
from continuous_loading.l1_transport import (
    L1_TRANSPORT_CONFIGURATION,
    l1_transport_inputs_for_species,
)
from continuous_loading.l2_transport import L2TransportInputs
from continuous_loading.lattice import evaluate_lattice


# 扫描页默认网格：9x9、每点 500 轨迹，串行后端几分钟内可完成。
SCAN_PRESET_DETUNING_POINTS = 9
SCAN_PRESET_POWER_POINTS = 9
SCAN_PRESET_PARTICLE_COUNT = 500


def atom_from_label(atom_label: str) -> AlkaliAtom:
    normalized = atom_label.strip().lower().replace("-", "").replace("_", "")
    if normalized in {"rb", "rb87", "87rb"}:
        return RB87
    if normalized in {"cs", "cs133", "133cs"}:
        return CS133
    raise ValueError("原子必须是 Rb-87 或 Cs-133")


def default_form_params(
    atom_label: str = "Rb-87",
    *,
    scan_preset: bool = False,
) -> dict[str, object]:
    """由计算库默认值构造一份表单参数。

    ``scan_preset=True`` 时换成扫描页的轻量默认（9x9 网格、500 轨迹），
    其余物理参数保持 ``data/l1_transport_defaults.json`` 的口径。
    """
    transport = l1_transport_inputs_for_species(atom_label)
    monte_carlo = L1HandoverInputs(transport=transport)
    l2 = L2TransportInputs()
    conveyor = L1_TRANSPORT_CONFIGURATION["conveyor_geometry"]
    transport_mc = L1_TRANSPORT_CONFIGURATION["transport_monte_carlo"]
    params: dict[str, object] = {
        # 原子与光路
        "atom_label": transport.atom_label,
        "detuning_ghz": 300.0 if transport.atom_label == "Rb-87" else 600.0,
        "source_power_w": 1.0,
        "delivery_efficiency": transport.delivery_efficiency,
        "retro_power_ratio": transport.retro_power_ratio,
        "target_depth_uK": transport.target_depth_uK,
        "handover_waist_um": transport.handover_waist_um,
        "mot_atom_number": transport.mot_atom_number,
        "pre_ramp_survival_fraction": transport.pre_ramp_survival_fraction,
        "initial_atom_number": transport.initial_atom_number,
        "initial_temperature_uK": transport.initial_temperature_uK,
        "occupied_lattice_sites": transport.occupied_lattice_sites,
        "include_gravity": transport.include_gravity,
        # L1 时序
        "l1_distance_m": transport.distance_m,
        "l1_acceleration_m_s2": transport.acceleration_m_s2,
        "l1_maximum_velocity_m_s": transport.maximum_velocity_m_s,
        "l1_kinematic_profile": transport.kinematic_profile,
        "l1_start_waist_um": transport.start_waist_um,
        "l1_time_points": transport.time_points,
        "phase_space_continuity": False,
        "l1_control_waveform_path": "",
        "handover_control_waveform_path": "",
        "l2_control_waveform_path": "",
        # conveyor 几何（可选，默认关闭）
        "conveyor_enabled": bool(conveyor["enabled"]),
        "conveyor_waist_um": float(conveyor["waist_um"]),
        "conveyor_waist_separation_cm": float(
            conveyor["waist_separation_cm"]
        ),
        # handover Monte Carlo（默认串行；进程池留给大网格）
        "duration_us": monte_carlo.duration_us,
        "particle_count": monte_carlo.particle_count,
        "time_step_us": monte_carlo.time_step_us,
        "trace_points": monte_carlo.trace_points,
        "crossing_angle_deg": monte_carlo.crossing_angle_deg,
        # 交接相对相位口径：随机（多发次系综平均）/固定（单发次）
        "phase_mode": (
            "random" if monte_carlo.randomize_relative_phase else "fixed"
        ),
        "relative_phase_deg": math.degrees(monte_carlo.relative_phase_rad),
        "cloud_axial_sigma_mm": monte_carlo.cloud_axial_sigma_mm,
        "seed": monte_carlo.seed,
        "include_scattering": monte_carlo.include_scattering,
        "compute_backend": monte_carlo.compute_backend,
        # 运输腿动力学（默认解析预算；Monte Carlo 为轨迹级核验，慢）
        "transport_method": (
            "monte_carlo" if bool(transport_mc["enabled"]) else "analytic"
        ),
        "transport_time_step_us": float(transport_mc["time_step_us"]),
        "parallel_backend": "serial",
        "worker_count": monte_carlo.worker_count,
        # L2 段
        "l2_distance_m": l2.distance_m,
        "l2_acceleration_m_s2": l2.acceleration_m_s2,
        "l2_maximum_velocity_m_s": l2.maximum_velocity_m_s,
        "l2_kinematic_profile": l2.kinematic_profile,
        "l2_end_waist_um": l2.end_waist_um,
        "l2_time_points": l2.time_points,
        # 扫描网格
        "scan_detuning_min_ghz": transport.detuning_min_ghz,
        "scan_detuning_max_ghz": transport.detuning_max_ghz,
        "scan_detuning_points": transport.detuning_points,
        "scan_power_min_w": transport.handover_source_power_min_w,
        "scan_power_max_w": transport.handover_source_power_max_w,
        "scan_power_points": transport.power_points,
    }
    if scan_preset:
        params["scan_detuning_points"] = SCAN_PRESET_DETUNING_POINTS
        params["scan_power_points"] = SCAN_PRESET_POWER_POINTS
        params["particle_count"] = SCAN_PRESET_PARTICLE_COUNT
    return params


def lattice_quick_metrics(
    atom_label: str,
    detuning_ghz: float,
    source_power_w: float,
    delivery_efficiency: float,
    waist_um: float,
    retro_power_ratio: float,
) -> dict[str, float | str]:
    """静态晶格指标速算：功率口径与 L1 模型一致（源端×传输效率）。"""
    atom = atom_from_label(atom_label)
    wavelength_nm = atom.laser_wavelength_red_of_d1_nm(detuning_ghz)
    forward_power_w = source_power_w * delivery_efficiency
    metrics = evaluate_lattice(
        atom,
        wavelength_nm,
        forward_power_w,
        waist_um,
        retro_power_ratio=retro_power_ratio,
    )
    return {
        "atom_label": atom.label,
        "detuning_ghz": detuning_ghz,
        "laser_wavelength_nm": wavelength_nm,
        "source_power_w": source_power_w,
        "forward_power_w": forward_power_w,
        "waist_um": waist_um,
        "retro_power_ratio": retro_power_ratio,
        "antinode_intensity_w_m2": metrics.antinode_intensity_w_m2,
        "depth_uK": metrics.depth_uK,
        "scattering_rate_s": metrics.scattering_rate_s,
        "recoil_temperature_uK": metrics.recoil_temperature_uK,
        "depth_in_recoil": metrics.depth_in_recoil,
        "radial_frequency_hz": metrics.radial_frequency_hz,
        "axial_frequency_hz": metrics.axial_frequency_hz,
        "critical_axial_acceleration_m_s2": (
            metrics.critical_axial_acceleration_m_s2
        ),
    }


def build_full_chain_inputs(params: dict[str, object]) -> FullChainInputs:
    """把表单参数组装成 ``FullChainInputs``（非法值由 dataclass 校验抛出）。"""
    atom_label = str(params["atom_label"])
    phase_space_continuity = bool(
        params.get("phase_space_continuity", False)
    )
    atom = atom_from_label(atom_label)
    reference_wavelength_nm = atom.laser_wavelength_red_of_d1_nm(
        float(params["detuning_ghz"])
    )

    def transport_waveform(key: str) -> TransportControlWaveform | None:
        path = str(params.get(key, "")).strip()
        return (
            None
            if not path
            else TransportControlWaveform.from_csv(
                path, wavelength_nm=reference_wavelength_nm
            )
        )

    l1_waveform = transport_waveform("l1_control_waveform_path")
    l2_waveform = transport_waveform("l2_control_waveform_path")
    handover_path = str(params.get("handover_control_waveform_path", "")).strip()
    handover_waveform = (
        None
        if not handover_path
        else HandoverControlWaveform.from_csv(handover_path)
    )
    species_transport = l1_transport_inputs_for_species(atom_label)
    transport = replace(
        species_transport,
        # 扫描网格参数为二维扫描页专属；其他页面缺省时回落物种默认。
        detuning_min_ghz=float(
            params.get("scan_detuning_min_ghz", species_transport.detuning_min_ghz)
        ),
        detuning_max_ghz=float(
            params.get("scan_detuning_max_ghz", species_transport.detuning_max_ghz)
        ),
        detuning_points=int(
            params.get("scan_detuning_points", species_transport.detuning_points)
        ),
        handover_source_power_min_w=float(
            params.get(
                "scan_power_min_w",
                species_transport.handover_source_power_min_w,
            )
        ),
        handover_source_power_max_w=float(
            params.get(
                "scan_power_max_w",
                species_transport.handover_source_power_max_w,
            )
        ),
        power_points=int(
            params.get("scan_power_points", species_transport.power_points)
        ),
        mot_atom_number=float(params["mot_atom_number"]),
        pre_ramp_survival_fraction=float(
            params.get("pre_ramp_survival_fraction", 1.0)
        ),
        initial_atom_number=float(params["initial_atom_number"]),
        initial_temperature_uK=float(params["initial_temperature_uK"]),
        occupied_lattice_sites=float(params["occupied_lattice_sites"]),
        include_gravity=bool(params.get("include_gravity", True)),
        distance_m=float(params["l1_distance_m"]),
        acceleration_m_s2=float(params["l1_acceleration_m_s2"]),
        maximum_velocity_m_s=float(params["l1_maximum_velocity_m_s"]),
        kinematic_profile=(
            "minimum_jerk"
            if phase_space_continuity and l1_waveform is None
            else str(params.get("l1_kinematic_profile", "minimum_jerk"))
        ),
        start_waist_um=float(params["l1_start_waist_um"]),
        handover_waist_um=float(params["handover_waist_um"]),
        time_points=int(params["l1_time_points"]),
        delivery_efficiency=float(params["delivery_efficiency"]),
        retro_power_ratio=float(params["retro_power_ratio"]),
        target_depth_uK=float(params["target_depth_uK"]),
        conveyor_enabled=bool(params["conveyor_enabled"]),
        conveyor_waist_um=float(params["conveyor_waist_um"]),
        conveyor_waist_separation_cm=float(
            params["conveyor_waist_separation_cm"]
        ),
        transport_method=str(params["transport_method"]),
        transport_time_step_us=float(params["transport_time_step_us"]),
        # 运输 MC 与 handover MC 合并调用同一组数值参数。
        mc_particle_count=int(params["particle_count"]),
        mc_seed=int(params["seed"]),
        mc_include_scattering=bool(params["include_scattering"]),
        mc_cloud_axial_sigma_mm=float(params["cloud_axial_sigma_mm"]),
        mc_compute_backend=str(params["compute_backend"]),
        control_waveform=l1_waveform,
    )
    monte_carlo = L1HandoverInputs(
        transport=transport,
        duration_us=(
            float(params["duration_us"])
            if handover_waveform is None
            else handover_waveform.duration_ms * 1e3
        ),
        particle_count=int(params["particle_count"]),
        time_step_us=float(params["time_step_us"]),
        trace_points=int(params["trace_points"]),
        include_scattering=bool(params["include_scattering"]),
        seed=int(params["seed"]),
        compute_backend=str(params["compute_backend"]),
        parallel_backend=str(params["parallel_backend"]),
        worker_count=int(params["worker_count"]),
        crossing_angle_deg=float(params["crossing_angle_deg"]),
        randomize_relative_phase=(
            str(params.get("phase_mode", "random")) != "fixed"
        ),
        relative_phase_rad=math.radians(
            float(params.get("relative_phase_deg", 0.0))
        ),
        cloud_axial_sigma_mm=float(params["cloud_axial_sigma_mm"]),
        control_waveform=handover_waveform,
    )
    l2 = L2TransportInputs(
        distance_m=float(params["l2_distance_m"]),
        acceleration_m_s2=float(params["l2_acceleration_m_s2"]),
        maximum_velocity_m_s=float(params["l2_maximum_velocity_m_s"]),
        kinematic_profile=(
            "minimum_jerk"
            if phase_space_continuity and l2_waveform is None
            else str(params.get("l2_kinematic_profile", "trapezoid"))
        ),
        end_waist_um=float(params["l2_end_waist_um"]),
        time_points=int(params["l2_time_points"]),
        occupied_lattice_sites=float(params["occupied_lattice_sites"]),
        control_waveform=l2_waveform,
    )
    return FullChainInputs(
        handover=monte_carlo,
        l2=l2,
        phase_space_continuity=phase_space_continuity,
    )


def run_single_point(params: dict[str, object]) -> FullChainPointSimulation:
    """运行一个失谐--功率点的 MOT→L1→handover→L2→科学区 全链路。"""
    inputs = build_full_chain_inputs(params)
    return simulate_full_chain_point(
        inputs,
        float(params["detuning_ghz"]),
        float(params["source_power_w"]),
        trace_points=int(params["trace_points"]),
    )


def run_scan(
    params: dict[str, object],
    progress: Callable[[str], None] | None = None,
) -> FullChainScanResult:
    """在表单给定的失谐--功率网格上运行全链路二维扫描。"""
    inputs = build_full_chain_inputs(params)
    return analyze_full_chain_scan(inputs, progress=progress)


def run_cloud_sigma_scan(
    params: dict[str, object],
    progress: Callable[[str], None] | None = None,
) -> CloudSigmaScanResult:
    """在表单给定的工作点上运行原子云轴向宽度一维扫描。"""
    inputs = CloudSigmaScanInputs(
        chain=build_full_chain_inputs(params),
        detuning_ghz=float(params["detuning_ghz"]),
        source_power_w=float(params["source_power_w"]),
        sigma_min_mm=float(params.get("cloud_sigma_min_mm", 0.0)),
        sigma_max_mm=float(params.get("cloud_sigma_max_mm", 5.0)),
        points=int(params.get("cloud_sigma_points", 10)),
    )
    return analyze_cloud_sigma_scan(inputs, progress=progress)


def _phase_summary(params: dict[str, object]) -> str:
    """交接相位口径摘要（固定口径时给出角度）。"""
    if str(params.get("phase_mode", "random")) == "fixed":
        return f"，固定相位 {float(params.get('relative_phase_deg', 0.0)):g}°"
    return ""


def summarize_single_point_params(params: dict[str, object]) -> str:
    """历史记录用的单点参数摘要。"""
    return (
        f"{params['atom_label']} 失谐 {float(params['detuning_ghz']):g} GHz，"
        f"源端功率 {float(params['source_power_w']):g} W，"
        f"N={int(params['particle_count'])}"
        f"{_phase_summary(params)}"
    )


def summarize_scan_params(params: dict[str, object]) -> str:
    """历史记录用的扫描参数摘要。"""
    return (
        f"{params['atom_label']} "
        f"{int(params['scan_detuning_points'])}x{int(params['scan_power_points'])} 网格，"
        f"失谐 {float(params['scan_detuning_min_ghz']):g}"
        f"-{float(params['scan_detuning_max_ghz']):g} GHz，"
        f"功率 {float(params['scan_power_min_w']):g}"
        f"-{float(params['scan_power_max_w']):g} W，"
        f"N={int(params['particle_count'])}，"
        f"{'串行' if params['parallel_backend'] == 'serial' else '进程池'}"
        f"{_phase_summary(params)}"
    )


def summarize_cloud_sigma_params(params: dict[str, object]) -> str:
    """历史记录用的云宽扫描参数摘要。"""
    return (
        f"{params['atom_label']} 失谐 {float(params['detuning_ghz']):g} GHz，"
        f"源端功率 {float(params['source_power_w']):g} W，"
        f"云宽 {float(params.get('cloud_sigma_min_mm', 0.0)):g}"
        f"-{float(params.get('cloud_sigma_max_mm', 5.0)):g} mm "
        f"{int(params.get('cloud_sigma_points', 10))} 点，"
        f"N={int(params['particle_count'])}"
        f"{_phase_summary(params)}"
    )


# ---- 扫描结果条件筛选（不重算，只基于已有矩阵）----

# 表达式白名单：BoolOp/Compare/Name/Constant/括号/一元负号，以及连接
# 比较结果的 & 和 |（BinOp 仅允许这两种位运算），其余一律拒绝。
_ALLOWED_EXPRESSION_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.BinOp,
    ast.BitAnd,
    ast.BitOr,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.UnaryOp,
    ast.USub,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
)

_COMPARE_OPERATORS = {
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
}


def _eval_expression_node(
    node: ast.AST, variables: dict[str, np.ndarray]
) -> np.ndarray:
    if isinstance(node, ast.Expression):
        return _eval_expression_node(node.body, variables)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value, (int, float)
        ):
            raise ValueError("表达式常量必须是数字")
        return np.asarray(float(node.value))
    if isinstance(node, ast.Name):
        if node.id not in variables:
            allowed = "、".join(sorted(variables))
            raise ValueError(f"表达式含未知变量 {node.id!r}，可用变量：{allowed}")
        return variables[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_expression_node(node.operand, variables)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.BitAnd, ast.BitOr)
    ):
        left = _eval_expression_node(node.left, variables)
        right = _eval_expression_node(node.right, variables)
        if left.dtype != np.bool_ or right.dtype != np.bool_:
            raise ValueError(
                "&/| 只能连接括号括起的比较表达式，"
                "例如 (P<=1.2)&(ret>=0.35)"
            )
        if isinstance(node.op, ast.BitAnd):
            return np.logical_and(left, right)
        return np.logical_or(left, right)
    if isinstance(node, ast.BoolOp):
        reducer = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        values = [_eval_expression_node(value, variables) for value in node.values]
        return reducer.reduce(values)
    if isinstance(node, ast.Compare):
        left = _eval_expression_node(node.left, variables)
        result = None
        for operator, comparator in zip(node.ops, node.comparators):
            right = _eval_expression_node(comparator, variables)
            part = _COMPARE_OPERATORS[type(operator)](left, right)
            result = part if result is None else np.logical_and(result, part)
            left = right
        return result
    raise ValueError(f"表达式含不允许的语法：{type(node).__name__}")


def evaluate_scan_expression(
    expression: str, variables: dict[str, np.ndarray]
) -> np.ndarray:
    """用 AST 白名单解析并求值筛选表达式，非法输入抛 ``ValueError``。"""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误：{exc.msg}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_EXPRESSION_NODES):
            raise ValueError(
                f"表达式含不允许的语法：{type(node).__name__}"
                "（只允许比较、and/or、&/|、括号和一元负号）"
            )
    return np.asarray(_eval_expression_node(tree, variables), dtype=bool)


def scan_condition_mask(
    result: FullChainScanResult, conditions: dict[str, object]
) -> np.ndarray:
    """在已有扫描矩阵上计算条件掩膜（不触发任何重新计算）。

    ``conditions`` 键：``power_enabled``/``power_max_w``、
    ``retention_enabled``/``retention_min``、
    ``heating_enabled``/``heating_max_uK``、``mode``（``"and"``/``"or"``）、
    ``expression``（非空时优先于勾选条件，可用变量 P/ret/heat/eff/dens）。
    矩阵中的 None/NaN 一律视为不符合。返回形状与扫描矩阵一致的布尔数组。
    """
    detunings = np.asarray(result.detuning_ghz, dtype=float)
    powers = np.asarray(result.source_power_w, dtype=float)
    _, power_grid = np.meshgrid(detunings, powers)
    variables = {
        "P": power_grid,
        "ret": np.asarray(result.final_retention_from_mot, dtype=float),
        "heat": np.asarray(result.science_total_temperature_rise_uK, dtype=float),
        "eff": np.asarray(result.handover_transfer_efficiency, dtype=float),
        "dens": np.asarray(result.science_peak_density_m3, dtype=float),
    }
    valid = np.ones_like(power_grid, dtype=bool)
    for matrix in variables.values():
        valid &= np.isfinite(matrix)

    expression = str(conditions.get("expression", "")).strip()
    if expression:
        mask = evaluate_scan_expression(expression, variables)
        if mask.shape != valid.shape:
            raise ValueError("表达式结果形状与扫描网格不一致")
        return mask & valid

    parts: list[np.ndarray] = []
    if bool(conditions.get("power_enabled", False)):
        parts.append(power_grid <= float(conditions["power_max_w"]))
    if bool(conditions.get("retention_enabled", False)):
        parts.append(variables["ret"] >= float(conditions["retention_min"]))
    if bool(conditions.get("heating_enabled", False)):
        parts.append(variables["heat"] <= float(conditions["heating_max_uK"]))
    mode = str(conditions.get("mode", "and"))
    if not parts:
        # 未勾选任何条件：AND 语义下全部有效点视为符合。
        return valid.copy()
    if mode == "or":
        mask = np.logical_or.reduce(parts)
    elif mode == "and":
        mask = np.logical_and.reduce(parts)
    else:
        raise ValueError("组合方式必须是 and 或 or")
    return np.asarray(mask, dtype=bool) & valid
