"""chain_mc.py 链式 Monte Carlo 编排器的端到端 smoke 与口径回归测试。

工作点沿用 test_transport_mc.py 的小距离 Rb 案例（5 mm L1、
300 GHz/1.0 W、300 粒子），handover 参数经
``l1_handover._handover_parameters`` 由解析探针 trace 构造（与工
作点自洽），L2 为 minimum_jerk 小距离腿。transport_mc 新逃逸参数
的回归用固定 seed 硬编码基线（基线在参数化修改**之前**捕获）。
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from continuous_loading.atomic import RB87
from continuous_loading.chain_mc import ChainMcInputs, run_chain_monte_carlo
from continuous_loading.initial_state import ensemble_kinetic_temperature_uK
from continuous_loading.l1_handover import L1HandoverInputs, _handover_parameters
from continuous_loading.l1_transport import (
    l1_transport_inputs_for_species,
    simulate_l1_transport,
)
from continuous_loading.l2_transport import L2TransportInputs
from continuous_loading.transport_mc import simulate_leg_monte_carlo


_DETUNING_GHZ = 300.0
_SOURCE_POWER_W = 1.0
_PARTICLE_COUNT = 300
_SEED = 20250902


def _transport_inputs(**overrides):
    """小距离运输腿：总时长约 2.25 ms，0.5 µs 步长下 4500 步。"""
    base = dict(
        distance_m=0.005,
        acceleration_m_s2=4000.0,
        maximum_velocity_m_s=4.0,
        time_points=21,
        transport_method="monte_carlo",
        transport_time_step_us=0.5,
    )
    base.update(overrides)
    return replace(l1_transport_inputs_for_species("Rb-87"), **base)


def _chain_handover_parameters_for(transport):
    """由解析探针 trace 构造与工作点自洽的 handover 参数。

    L2 为 minimum_jerk，交接时刻瞬时加速度为 0（与 full_chain 连续
    相空间模式的 ``_l2_boundary_acceleration`` 口径一致）。
    """
    probe = simulate_l1_transport(
        replace(transport, transport_method="analytic"),
        _DETUNING_GHZ,
        _SOURCE_POWER_W,
    )
    handover_inputs = L1HandoverInputs(
        transport=transport,
        particle_count=_PARTICLE_COUNT,
        time_step_us=0.5,
        trace_points=5,
        parallel_backend="serial",
        worker_count=1,
    )
    return _handover_parameters(
        handover_inputs,
        probe,
        trace_points=5,
        post_handover_acceleration_m_s2=0.0,
    )


def _l2_inputs():
    return L2TransportInputs(
        distance_m=0.003,
        acceleration_m_s2=3000.0,
        maximum_velocity_m_s=2.0,
        kinematic_profile="minimum_jerk",
        end_waist_um=150.0,
        time_points=11,
    )


def _chain_inputs(**overrides):
    transport = _transport_inputs()
    base = dict(
        atom_label="Rb-87",
        transport=transport,
        handover=_chain_handover_parameters_for(transport),
        l2=_l2_inputs(),
        initial_temperature_uK=20.0,
        particle_count=_PARTICLE_COUNT,
        seed=_SEED,
        include_gravity=True,
        include_scattering=True,
    )
    base.update(overrides)
    return ChainMcInputs(**base)


@pytest.fixture(scope="module")
def chain_result():
    return run_chain_monte_carlo(_chain_inputs(), _DETUNING_GHZ, _SOURCE_POWER_W)


def test_chain_inputs_validation():
    with pytest.raises(ValueError):
        _chain_inputs(atom_label="H-1")
    with pytest.raises(ValueError):
        _chain_inputs(initial_temperature_uK=0.0)
    with pytest.raises(ValueError):
        _chain_inputs(initial_temperature_uK=float("nan"))
    with pytest.raises(ValueError):
        _chain_inputs(particle_count=0)


def test_chain_smoke_end_to_end(chain_result):
    """端到端 smoke：字段齐全、留存 ∈ [0,1]、留存链自洽。"""
    result = chain_result
    assert result.l2_trace is not None
    assert result.science is not None
    # 光场时序先行：三段逐步数组随结果返回且全部有限。
    for timeline in (result.light_field.l1, result.light_field.l2):
        assert timeline.step_times_s.ndim == 1
        assert timeline.step_times_s.shape[0] > 1
        assert np.all(np.isfinite(timeline.step_times_s))
    handover_timeline = result.light_field.handover
    assert handover_timeline.fraction1[0] == 1.0
    assert handover_timeline.fraction2[-1] == 1.0
    # 逐段留存与总留存位于 [0,1]，S_total = S_L1·η_HO·S_L2 自洽。
    for value in (
        result.l1_retention_fraction,
        result.handover_transfer_efficiency,
        result.l2_retention_fraction,
        result.total_retention_fraction,
    ):
        assert 0.0 <= value <= 1.0
    assert result.total_retention_fraction == (
        result.l1_retention_fraction
        * result.handover_transfer_efficiency
        * result.l2_retention_fraction
    )
    # 原子数从 L1TransportInputs.initial_atom_number 起算（无 η_load）。
    assert result.final_atom_number == (
        result.inputs.transport.initial_atom_number
        * result.total_retention_fraction
    )
    # Jeffreys 标准误非负（透传自各段 trace/result）。
    assert result.l1_retention_standard_error >= 0.0
    assert result.handover_transfer_standard_error >= 0.0
    assert result.l2_retention_standard_error >= 0.0
    # 温度口径：腿段共动去质心动能温度、handover 双口径、科学区汇总。
    assert result.l1_trace.point.final_temperature_uK > 0.0
    assert math.isfinite(result.l1_trace.point.final_temperature_uK)
    assert result.handover_result.final_temperature_uK is not None
    assert result.handover_result.final_kinetic_temperature_uK is not None
    assert math.isfinite(result.science.temperature_uK)
    assert result.science.temperature_uK > 0.0
    assert result.science.atom_number >= 0.0


def test_leg_removal_conservation_and_handover_constant_count(chain_result):
    """剔除只发生在腿段：腿段进入数=幸存数+剔除数；handover 段内
    粒子数恒定（进入数==模拟粒子数），捕获选择只发生在段末。"""
    result = chain_result
    n = result.inputs.particle_count
    # 腿段剔除守恒：trace 留存与末态系综粒子数是两套独立记账。
    l1_removed = n - result.l1_survivor_count
    assert l1_removed >= 0
    assert result.l1_survivor_count + l1_removed == n
    assert round(result.l1_retention_fraction * n) == result.l1_survivor_count
    l2_removed = n - result.l2_survivor_count
    assert l2_removed >= 0
    assert result.l2_survivor_count + l2_removed == n
    assert round(result.l2_retention_fraction * n) == result.l2_survivor_count
    # handover 段粒子数恒定：模拟粒子数 == 链式进入数（段内无剔除），
    # 粒子数只在段末捕获子样本选择时减少一次。
    assert result.handover_result.parameters.particle_count == n
    assert result.handover_result.captured_count == (
        result.handover_captured_count
    )
    assert result.handover_result.transfer_efficiency == (
        result.handover_captured_count / n
    )


def test_initial_ensemble_temperature_matches_setpoint(chain_result):
    """初始系综去质心动能温度 ≈ 设定 20 µK。

    N=300 时动能温度相对 MC 标准差 ≈ √(2/3N) ≈ 4.7%，取 15% 容差
    （约 3σ）。
    """
    result = chain_result
    assert result.initial_sampled_temperature_uK == pytest.approx(
        result.inputs.initial_temperature_uK, rel=0.15
    )
    ensemble_temperature = ensemble_kinetic_temperature_uK(
        result.initial_ensemble, RB87.mass_kg
    )
    assert ensemble_temperature == result.initial_sampled_temperature_uK
    assert result.initial_ensemble.frame == "l1_local"
    assert result.initial_ensemble.particle_count == (
        result.inputs.particle_count
    )


def test_same_seed_reproducible(chain_result):
    """同 seed 两次链式运行结果逐等。"""
    second = run_chain_monte_carlo(
        _chain_inputs(), _DETUNING_GHZ, _SOURCE_POWER_W
    )
    np.testing.assert_array_equal(
        second.initial_ensemble.positions_m,
        chain_result.initial_ensemble.positions_m,
    )
    np.testing.assert_array_equal(
        second.initial_ensemble.velocities_m_s,
        chain_result.initial_ensemble.velocities_m_s,
    )
    assert second.total_retention_fraction == (
        chain_result.total_retention_fraction
    )
    assert second.l1_trace.point.final_temperature_uK == (
        chain_result.l1_trace.point.final_temperature_uK
    )
    assert second.handover_result.transfer_efficiency == (
        chain_result.handover_result.transfer_efficiency
    )
    assert second.l2_trace.point.final_temperature_uK == (
        chain_result.l2_trace.point.final_temperature_uK
    )


# ---- transport_mc 逃逸判据参数化回归 ----

_BASELINE_PARTICLE_COUNT = 200
# 以下基线在 transport_mc.py 参数化修改 + 步长精度守卫（ω_z·dt ≤ 1，
# 实际步长 0.5 → 0.408 µs）之后用同一代码路径再捕获（小距离腿、
# 300 GHz/1.0 W、200 粒子、seed 20250902、含散射、初温 30 µK）。
_BASELINE_FINAL_RETENTION = 1.0
_BASELINE_FINAL_TEMPERATURE_UK = 35.36944549217387
_BASELINE_SCATTERING_EVENTS = 1.28
_BASELINE_RETENTION_STANDARD_ERROR = 0.0035048581724460605
_BASELINE_FINAL_ENSEMBLE_COUNT = 200
_BASELINE_TEMPERATURE_UK = (
    28.091035263730245,
    27.581718196087582,
    27.77278790923564,
    29.59053313200073,
    28.870310583210227,
    31.749523192070644,
    30.952621584572697,
    32.663341485464805,
    34.83554042449372,
    36.25524575353574,
    35.96627008345233,
    36.83335733441087,
    37.268640439384555,
    36.87069848651231,
    38.884863881110704,
    38.19708985887307,
    39.698756248496714,
    38.435518309423394,
    37.97206996786861,
    36.65517029868613,
    36.90854583961801,
    36.145282042066455,
    35.36944549217387,
)
_BASELINE_RETENTION_FRACTION = (
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
)
_BASELINE_BOUND_FRACTION = (
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
)


def _baseline_leg_inputs():
    # 硬编码基线在阶段 3 参数化修改之前按当时 30 µK 的全局默认初温
    # 捕获；阶段 4 已把全局默认初温改为 20 µK。这里显式钉住 30 µK，
    # 使基线继续校验 transport_mc 缺省逃逸口径的逐位一致性。
    return _transport_inputs(
        initial_temperature_uK=30.0,
        mc_particle_count=_BASELINE_PARTICLE_COUNT,
        mc_include_scattering=True,
        mc_seed=_SEED,
        mc_cloud_axial_sigma_mm=0.5,
    )


def test_default_escape_parameters_match_prechange_baseline():
    """新参数缺省时结果必须与硬编码基线逐位一致（含步长守卫口径）。"""
    trace, final = simulate_leg_monte_carlo(
        _baseline_leg_inputs(),
        _DETUNING_GHZ,
        _SOURCE_POWER_W,
        return_final_ensemble=True,
    )
    assert trace.temperature_uK == _BASELINE_TEMPERATURE_UK
    assert trace.retention_fraction == _BASELINE_RETENTION_FRACTION
    assert trace.bound_fraction == _BASELINE_BOUND_FRACTION
    assert trace.point.final_retention_fraction == _BASELINE_FINAL_RETENTION
    assert (
        trace.point.final_temperature_uK == _BASELINE_FINAL_TEMPERATURE_UK
    )
    assert (
        trace.point.cumulative_scattering_events
        == _BASELINE_SCATTERING_EVENTS
    )
    assert (
        trace.retention_standard_error
        == _BASELINE_RETENTION_STANDARD_ERROR
    )
    assert final is not None
    assert final.particle_count == _BASELINE_FINAL_ENSEMBLE_COUNT


def test_explicit_none_escape_parameters_equal_default():
    """显式传 escape_check_interval_steps=None/escape_lenient=False
    与缺省调用逐位一致。"""
    reference, reference_final = simulate_leg_monte_carlo(
        _baseline_leg_inputs(),
        _DETUNING_GHZ,
        _SOURCE_POWER_W,
        return_final_ensemble=True,
    )
    explicit, explicit_final = simulate_leg_monte_carlo(
        _baseline_leg_inputs(),
        _DETUNING_GHZ,
        _SOURCE_POWER_W,
        return_final_ensemble=True,
        escape_check_interval_steps=None,
        escape_lenient=False,
    )
    assert explicit.temperature_uK == reference.temperature_uK
    assert explicit.retention_fraction == reference.retention_fraction
    assert explicit.bound_fraction == reference.bound_fraction
    assert explicit.point == reference.point
    np.testing.assert_array_equal(
        explicit_final.positions_m, reference_final.positions_m
    )
    np.testing.assert_array_equal(
        explicit_final.velocities_m_s, reference_final.velocities_m_s
    )


def test_escape_check_interval_validation():
    for bad in (0, -1, -200):
        with pytest.raises(ValueError):
            simulate_leg_monte_carlo(
                _baseline_leg_inputs(),
                _DETUNING_GHZ,
                _SOURCE_POWER_W,
                escape_check_interval_steps=bad,
            )


def test_lenient_per_step_escape_smoke():
    """每步 + 宽容判据（链式 MC 同款参数）跑通且物理量有界。"""
    trace, final = simulate_leg_monte_carlo(
        _baseline_leg_inputs(),
        _DETUNING_GHZ,
        _SOURCE_POWER_W,
        return_final_ensemble=True,
        escape_check_interval_steps=1,
        escape_lenient=True,
    )
    assert 0.0 < trace.point.final_retention_fraction <= 1.0
    assert math.isfinite(trace.point.final_temperature_uK)
    survivors = round(
        trace.point.final_retention_fraction * _BASELINE_PARTICLE_COUNT
    )
    assert (0 if final is None else final.particle_count) == survivors
