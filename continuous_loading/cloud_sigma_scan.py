"""原子云轴向宽度对 handover 交接影响的一维扫描。

在固定工作点（D1 红失谐、源端功率）上扫描原子云轴向宽度 σ，输出
handover 末温、链末总温、handover 交接率与相对 MOT 总留存随无量纲
云宽 χ = σ·sinθ/w（σ 与重合区尺度 z_ov = w/sinθ 之比，交接响应的
真正控制参数，见 reports/handover云宽尺度猜想与验证.md）的变化，
用于评估 handover 交接对初始云宽的敏感性。每点同时给出
``sigma_over_waist``（σ/w）与 ``chi`` 两种归一化。

物理口径：

- 云宽 σ 在链路中由两个字段描述同一物理量，扫描时必须同步设置为
  同一值：``L1HandoverInputs.cloud_axial_sigma_mm`` 是 (N,T) 约化
  接口下 handover 自采样初态的轴向云宽；
  ``L1TransportInputs.mc_cloud_axial_sigma_mm`` 是连续相空间接口下
  L1 起点初始系综的采样宽度（该系综随后贯穿 L1→handover→L2）。
  每个 σ 点用 ``dataclasses.replace`` 嵌套替换这两个字段。
- 指标取自 ``simulate_full_chain_point(..., trace_points=2)`` 的
  ``FullChainPoint``：handover 末温
  ``l1_handover.final_temperature_uK``、交接率
  ``l1_handover.handover_transfer_efficiency``、链末总温
  ``l2_final_temperature_uK``、相对 MOT 总留存
  ``final_retention_from_mot``。
- (N,T) 约化与连续相空间两种接口模式都适用（由
  ``FullChainInputs.phase_space_continuity`` 决定）；计算后端由
  chain 内的 ``compute_backend`` 决定——GPU 时逐点走 GPU kernel
  （本扫描等价于在同一工作点上重复单点调用，不做跨点批量合并）。
- 逐点失败隔离：单点异常不中断扫描，该点指标记 None 并记录原因；
  结束后经 progress 汇总 Top-3 失败原因（与二维扫描同一口径）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
from typing import Callable

import numpy as np

from .full_chain import (
    FullChainInputs,
    _report_point_failures,
    simulate_full_chain_point,
)


@dataclass(frozen=True)
class CloudSigmaScanInputs:
    """固定工作点上的云宽一维扫描输入。"""

    chain: FullChainInputs
    detuning_ghz: float
    source_power_w: float
    sigma_min_mm: float = 0.0
    sigma_max_mm: float = 5.0
    points: int = 10

    def __post_init__(self) -> None:
        finite = {
            "D1 红失谐": self.detuning_ghz,
            "源端功率": self.source_power_w,
            "云宽下限": self.sigma_min_mm,
            "云宽上限": self.sigma_max_mm,
        }
        for name, value in finite.items():
            if not math.isfinite(value):
                raise ValueError(f"{name}必须是有限数")
        if self.detuning_ghz <= 0.0:
            raise ValueError("D1 红失谐必须为正")
        if self.source_power_w <= 0.0:
            raise ValueError("源端功率必须为正")
        if self.sigma_min_mm < 0.0:
            raise ValueError("云宽下限必须是非负数")
        if self.sigma_max_mm <= self.sigma_min_mm:
            raise ValueError("云宽上限必须大于下限")
        if self.points < 2:
            raise ValueError("云宽扫描点数至少为 2")


@dataclass(frozen=True)
class CloudSigmaScanPoint:
    """一个 σ 点的扫描结果；``error`` 非 None 时各指标为 None。"""

    sigma_mm: float
    sigma_over_waist: float  # σ / (handover_waist_um × 1e-3 mm)
    # 无量纲云宽 χ = σ·sinθ/w（σ 与重合区尺度 z_ov = w/sinθ 之比），
    # 是交接率/升温响应的真正控制参数（见
    # reports/handover云宽尺度猜想与验证.md）。
    chi: float
    handover_temperature_uK: float | None
    final_temperature_uK: float | None
    handover_efficiency: float | None
    final_retention_from_mot: float | None
    error: str | None


@dataclass(frozen=True)
class CloudSigmaScanResult:
    """云宽一维扫描结果（归一化横轴见 ``CloudSigmaScanPoint.chi``）。"""

    inputs: CloudSigmaScanInputs
    waist_mm: float
    points: tuple[CloudSigmaScanPoint, ...]


def _chain_with_sigma(chain: FullChainInputs, sigma_mm: float) -> FullChainInputs:
    """把云宽 σ 同步写入 handover 自采样字段与运输 MC 初态采样字段。"""
    handover = chain.handover
    transport = replace(handover.transport, mc_cloud_axial_sigma_mm=sigma_mm)
    return replace(
        chain,
        handover=replace(
            handover,
            transport=transport,
            cloud_axial_sigma_mm=sigma_mm,
        ),
    )


def analyze_cloud_sigma_scan(
    inputs: CloudSigmaScanInputs,
    *,
    progress: Callable[[str], None] | None = None,
) -> CloudSigmaScanResult:
    """在固定工作点上逐 σ 运行全链路，返回随 σ 变化的指标序列。"""
    transport = inputs.chain.handover.transport
    atom_label = transport.atom_label
    waist_mm = transport.handover_waist_um * 1e-3
    sin_angle = math.sin(
        math.radians(inputs.chain.handover.crossing_angle_deg)
    )
    sigmas = np.linspace(inputs.sigma_min_mm, inputs.sigma_max_mm, inputs.points)
    total = len(sigmas)
    points: list[CloudSigmaScanPoint] = []
    failures: Counter = Counter()
    for index, sigma in enumerate(sigmas, start=1):
        sigma_mm = float(sigma)
        chi = sigma_mm * sin_angle / waist_mm
        chain = _chain_with_sigma(inputs.chain, sigma_mm)
        try:
            simulation = simulate_full_chain_point(
                chain,
                inputs.detuning_ghz,
                inputs.source_power_w,
                trace_points=2,
            )
        except Exception as exc:  # noqa: BLE001 - 单点异常按该点无有效结果处理
            error = f"{type(exc).__name__}: {exc}"
            failures[error] += 1
            point = CloudSigmaScanPoint(
                sigma_mm=sigma_mm,
                sigma_over_waist=sigma_mm / waist_mm,
                chi=chi,
                handover_temperature_uK=None,
                final_temperature_uK=None,
                handover_efficiency=None,
                final_retention_from_mot=None,
                error=error,
            )
        else:
            summary = simulation.point
            point = CloudSigmaScanPoint(
                sigma_mm=sigma_mm,
                sigma_over_waist=sigma_mm / waist_mm,
                chi=chi,
                handover_temperature_uK=(
                    summary.l1_handover.final_temperature_uK
                ),
                final_temperature_uK=summary.l2_final_temperature_uK,
                handover_efficiency=(
                    summary.l1_handover.handover_transfer_efficiency
                ),
                final_retention_from_mot=summary.final_retention_from_mot,
                error=None,
            )
        points.append(point)
        if progress is not None:
            progress(f"{atom_label}: 云宽扫描 {index}/{total}")
    _report_point_failures(failures, progress, atom_label)
    return CloudSigmaScanResult(
        inputs=inputs,
        waist_mm=waist_mm,
        points=tuple(points),
    )
