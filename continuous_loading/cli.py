"""命令行入口：论文 Rb 复现与 Cs 失谐/功率扫描。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

from .atomic import CS133, RB87
from .design_optimization import (
    RobustDesignInputs,
    analyze_robust_design,
    plot_robust_design,
)
from .handover import (
    HandoverResult,
    HandoverScanPoint,
    run_handover_monte_carlo,
    scan_handover_parameter,
)
from .handover_map import (
    HandoverMapInputs,
    analyze_dual_species_handover_map,
    analyze_dual_species_l1_handover_map,
    plot_dual_species_handover_map,
)
from .handover_angle_scan import (
    HandoverAngleScanInputs,
    analyze_handover_angle_scan,
    plot_handover_angle_scan,
)
from .lattice import evaluate_lattice
from .linear_design import (
    DEFAULT_HANDOVER_TIMES_MS,
    LinearDesignInputs,
    analyze_detuning_power_lp,
    plot_detuning_power_lp,
)
from .l1_transport import (
    L1TransportInputs,
    analyze_l1_transport_scan,
    l1_transport_inputs_for_species,
    plot_l1_transport_scan,
)
from .l1_handover import L1HandoverInputs, analyze_l1_handover_scan
from .l1_handover_plots import plot_l1_handover_scan
from .full_chain import FullChainInputs, analyze_full_chain_scan
from .full_chain_plots import plot_full_chain_scan
from .l2_transport import L2TransportInputs
from .scenarios import (
    extended_figure2_scan_preset,
    paper_handover_parameters,
    predict_cs_transport,
    reproduce_paper_rb87,
    scan_cs_designs,
)


def _paper_command(args: argparse.Namespace) -> int:
    result = reproduce_paper_rb87()
    payload = asdict(result)
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("Rb-87 双晶格论文参数复现")
    print(f"晶格波长: {result.laser_wavelength_nm:.6f} nm")
    print(
        "250 µm, 1 W, 回程比 0.88^4 的阱深: "
        f"{result.lattice_at_250um.depth_uK:.2f} µK"
    )
    print(
        "同功率 330 µm 的阱深: "
        f"{result.lattice_at_330um_same_power.depth_uK:.2f} µK"
    )
    print(
        "330 µm 达到 500 µK 所需前向功率: "
        f"{result.required_power_330um_for_500uK:.3f} W"
    )
    print(
        "平均速度 L1/L2: "
        f"{result.lattice1_average_speed_m_s:.3f}/"
        f"{result.lattice2_average_speed_m_s:.3f} m/s"
    )
    print(
        "10 m/s 所需 AOM 双束频差: "
        f"{result.lattice1_frequency_shift_mhz_at_10m_s:.3f} MHz"
    )
    print(
        "复现 120 µK 终温所需等效随机相位比例: "
        f"{result.inferred_handover_fraction:.3f}"
    )
    print(
        "由峰值密度反推单格点/占据格点数: "
        f"{result.inferred_atoms_per_lattice_site:.1f}/"
        f"{result.inferred_occupied_lattice_sites:.0f}"
    )
    print(
        "nV 随机重叠原子数与碰撞密度: "
        f"{result.stochastic_overlap_atoms:.2f}, "
        f"{result.collision_density_m3_s:.2e} m^-3 s^-1"
    )
    for stage in result.transport_budget.stages:
        print(
            f"{stage.name}: {stage.input_temperature_uK:.2f} -> "
            f"{stage.output_temperature_uK:.2f} µK, "
            f"散射 {stage.scattering_events:.1f} 次, "
            f"有效势垒 {stage.effective_barrier_uK:.1f} µK, "
            f"热平衡束缚比例 {stage.thermal_bound_fraction:.3f}"
        )
    if args.json:
        print(f"JSON 已写入: {Path(args.json)}")
    return 0


def _cs_command(args: argparse.Namespace) -> int:
    candidates = scan_cs_designs(
        target_depth_uK=args.depth,
        waist_um=args.waist,
        detuning_min_ghz=args.detuning_min,
        detuning_max_ghz=args.detuning_max,
        detuning_step_ghz=args.detuning_step,
        retro_power_ratio=args.retro_ratio,
        delivery_efficiency=args.delivery_efficiency,
        max_source_power_w=args.max_power,
        max_scattering_rate_s=args.max_scattering,
    )

    feasible = [candidate for candidate in candidates if candidate.feasible]
    print(
        f"Cs-133 扫描: {len(candidates)} 个候选，"
        f"{len(feasible)} 个满足约束"
    )
    print(
        "detuning_GHz wavelength_nm source_W scatter_s^-1 "
        "heat_uK/s feasible"
    )
    for candidate in candidates:
        if args.only_feasible and not candidate.feasible:
            continue
        print(
            f"{candidate.d1_red_detuning_ghz:10.1f} "
            f"{candidate.wavelength_nm:13.6f} "
            f"{candidate.source_power_w:8.3f} "
            f"{candidate.scattering_rate_s:12.2f} "
            f"{candidate.recoil_heating_rate_uK_s:10.3f} "
            f"{str(candidate.feasible):>8}"
        )

    if args.csv:
        output = Path(args.csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(candidate) for candidate in candidates]
        with output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV 已写入: {output}")
    return 0


def _cs_transport_command(args: argparse.Namespace) -> int:
    prediction = predict_cs_transport(
        d1_red_detuning_ghz=args.detuning,
        target_depth_uK=args.depth,
        initial_temperature_uK=args.temperature,
        acceleration_m_s2=args.acceleration,
        handover_random_phase_fraction=args.handover_fraction,
        retro_power_ratio=args.retro_ratio,
        delivery_efficiency=args.delivery_efficiency,
    )
    candidate = prediction.candidate_at_250um
    print("Cs-133 同型双晶格运输预测（恒阱深功率斜坡）")
    print(
        f"波长 {candidate.wavelength_nm:.6f} nm，"
        f"D1 红失谐 {candidate.d1_red_detuning_ghz:.1f} GHz"
    )
    print(
        "源端前向功率 L1起点/L1终点/L2终点: "
        f"{prediction.lattice1_start_power_w:.3f}/"
        f"{prediction.lattice1_end_power_w:.3f}/"
        f"{prediction.lattice2_end_power_w:.3f} W"
    )
    for stage in prediction.transport_budget.stages:
        print(
            f"{stage.name}: {stage.input_temperature_uK:.2f} -> "
            f"{stage.output_temperature_uK:.2f} µK；"
            f"反冲 +{stage.recoil_heating_uK:.2f} µK；"
            f"散射 {stage.scattering_events:.1f} 次"
        )
    print(
        f"交接等效加热: {prediction.transport_budget.handover_heating_uK:.2f} µK"
    )
    print(
        f"预测终温: {prediction.transport_budget.final_temperature_uK:.2f} µK"
    )
    return 0


_HANDOVER_SCAN_ALIASES = {
    "duration": "duration_ms",
    "distance": "lattice1_distance_cm",
    "acceleration": "post_handover_acceleration_m_s2",
    "temperature": "temperature_uK",
    "depth1": "depth1_uK",
    "depth2": "depth2_uK",
    "angle": "crossing_angle_deg",
    "cloud-size": "cloud_axial_sigma_mm",
    "offset": "l2_transverse_offset_um",
    "velocity1": "lattice1_velocity_m_s",
    "velocity2": "lattice2_velocity_m_s",
    "phase": "relative_phase_rad",
}


def _handover_parameters_from_args(args: argparse.Namespace):
    base = paper_handover_parameters(
        d1_red_detuning_ghz=args.detuning,
        depth1_uK=args.depth1,
        depth2_uK=args.depth2,
        waist1_um=args.waist1,
        waist2_um=args.waist2,
        temperature_uK=args.temperature,
    )
    return replace(
        base,
        duration_ms=args.duration,
        crossing_angle_deg=args.angle,
        lattice1_distance_cm=args.distance,
        optimal_distance_cm=args.optimal_distance,
        cloud_axial_sigma_mm=args.cloud_size,
        l2_transverse_offset_um=args.offset,
        relative_phase_rad=args.phase,
        randomize_relative_phase=not args.fixed_phase,
        initial_atom_number=args.atom_number,
        lattice1_velocity_m_s=args.velocity1,
        lattice2_velocity_m_s=args.velocity2,
        post_handover_acceleration_m_s2=args.acceleration,
        include_scattering=not args.no_scattering,
        particle_count=args.particles,
        time_step_us=args.time_step,
        trace_points=args.trace_points,
        seed=args.seed,
    )


def _format_optional(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _print_handover_result(result: HandoverResult) -> None:
    print(
        f"交接率: {result.transfer_efficiency:.4f} ± "
        f"{result.transfer_standard_error:.4f} "
        f"({result.captured_count}/{result.parameters.particle_count})"
    )
    print(
        "按 Lattice-1 实际原子数折算的 Lattice-2 捕获数: "
        f"{result.estimated_captured_atom_number:.3g} ± "
        f"{result.estimated_captured_atom_number_standard_error:.2g}"
    )
    print(
        "捕获子样本等效温度: "
        f"{_format_optional(result.captured_initial_temperature_uK)} -> "
        f"{_format_optional(result.final_temperature_uK)} µK；"
        f"交接净升温 {_format_optional(result.handover_heating_uK)} µK"
    )
    print(
        "末态纯动能温度/平均散射次数/反冲温升预算: "
        f"{_format_optional(result.final_kinetic_temperature_uK)} µK / "
        f"{result.mean_scattering_events:.3f} / "
        f"{result.recoil_heating_estimate_uK:.3f} µK"
    )
    print(
        "交接后有效势垒: "
        f"{result.effective_barrier_uK:.2f} µK "
        f"(a/a_c={abs(result.parameters.post_handover_acceleration_m_s2) / result.critical_acceleration_m_s2:.4f})"
    )
    print(
        f"积分步数/实际步长: {result.integration_steps}/"
        f"{result.actual_time_step_us:.5f} µs"
    )


def _handover_scan_row(point: HandoverScanPoint) -> dict[str, object]:
    result = point.result
    return {
        "parameter": point.parameter_name,
        "value": point.parameter_value,
        "transfer_efficiency": result.transfer_efficiency,
        "transfer_standard_error": result.transfer_standard_error,
        "captured_count": result.captured_count,
        "particle_count": result.parameters.particle_count,
        "estimated_captured_atom_number": (
            result.estimated_captured_atom_number
        ),
        "estimated_captured_atom_number_standard_error": (
            result.estimated_captured_atom_number_standard_error
        ),
        "captured_initial_temperature_uK": (
            result.captured_initial_temperature_uK
        ),
        "final_temperature_uK": result.final_temperature_uK,
        "handover_heating_uK": result.handover_heating_uK,
        "mean_scattering_events": result.mean_scattering_events,
        "effective_barrier_uK": result.effective_barrier_uK,
    }


def _save_handover_plot(
    output_path: str,
    *,
    result: HandoverResult | None = None,
    scan: list[HandoverScanPoint] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if scan is not None:
        values = [point.parameter_value for point in scan]
        efficiencies = [
            point.result.transfer_efficiency for point in scan
        ]
        errors = [
            point.result.transfer_standard_error for point in scan
        ]
        temperatures = [
            float("nan")
            if point.result.final_temperature_uK is None
            else point.result.final_temperature_uK
            for point in scan
        ]
        fig, axis = plt.subplots(figsize=(7.2, 4.6))
        axis.errorbar(
            values,
            efficiencies,
            yerr=errors,
            marker="o",
            color="#2ca02c",
            label="MC transfer",
        )
        axis.set_xlabel(scan[0].parameter_name)
        axis.set_ylabel("Transfer efficiency")
        axis.set_ylim(0.0, 1.05)
        if scan[0].parameter_name == "post_handover_acceleration_m_s2":
            axis.set_xscale("log")
        temperature_axis = axis.twinx()
        temperature_axis.plot(
            values,
            temperatures,
            marker="s",
            color="#1f77b4",
            label="Final temperature",
        )
        temperature_axis.set_ylabel("Equivalent temperature (µK)")
        fig.tight_layout()
    elif result is not None:
        trace = result.trace
        fig, axis = plt.subplots(figsize=(7.2, 4.6))
        axis.plot(
            trace.time_ms,
            trace.kinetic_temperature_uK,
            color="#1f77b4",
            label="Kinetic temperature",
        )
        axis.set_xlabel("Handover time (ms)")
        axis.set_ylabel("Kinetic temperature (µK)")
        ramp_axis = axis.twinx()
        ramp_axis.plot(
            trace.time_ms,
            trace.lattice1_fraction,
            "--",
            color="#6baed6",
            label="Lattice-1",
        )
        ramp_axis.plot(
            trace.time_ms,
            trace.lattice2_fraction,
            "--",
            color="#31a354",
            label="Lattice-2",
        )
        ramp_axis.set_ylabel("Relative lattice depth")
        ramp_axis.set_ylim(0.0, 1.05)
        fig.tight_layout()
    else:
        raise ValueError("绘图必须提供单点结果或扫描结果")

    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _handover_command(args: argparse.Namespace) -> int:
    parameters = _handover_parameters_from_args(args)
    scan_parameter: str | None = None
    scan_values: list[float] | tuple[float, ...] | None = None

    if args.figure2_panel:
        preset = extended_figure2_scan_preset(args.figure2_panel)
        scan_parameter = preset.parameter_name
        scan_values = preset.values
        print(f"Extended Data Fig. 2{preset.panel}: {preset.description}")
    elif args.scan_parameter:
        if not args.scan_values:
            raise ValueError("自定义扫描必须提供 --scan-values")
        scan_parameter = _HANDOVER_SCAN_ALIASES[args.scan_parameter]
        scan_values = [
            float(value.strip())
            for value in args.scan_values.split(",")
            if value.strip()
        ]

    print("Rb-87 双晶格 handover 三维经典轨迹 Monte Carlo")
    print(
        f"输入 T={parameters.temperature_uK:.2f} µK，"
        f"U1/U2={parameters.depth1_uK:.1f}/{parameters.depth2_uK:.1f} µK，"
        f"夹角={parameters.crossing_angle_deg:.2f}°，"
        f"轴向云尺寸={parameters.cloud_axial_sigma_mm:.3f} mm"
    )

    if scan_parameter is None:
        result = run_handover_monte_carlo(parameters)
        _print_handover_result(result)
        payload: object = asdict(result)
        rows = [
            {
                **_handover_scan_row(
                    HandoverScanPoint(
                        parameter_name="single",
                        parameter_value=0.0,
                        result=result,
                    )
                )
            }
        ]
        if args.plot:
            saved = _save_handover_plot(args.plot, result=result)
            print(f"轨迹图已写入: {saved}")
    else:
        if scan_parameter == "relative_phase_rad":
            parameters = replace(parameters, randomize_relative_phase=False)
        points = scan_handover_parameter(
            parameters,
            scan_parameter,
            scan_values or (),
        )
        print(
            "value efficiency sem final_T_uK heating_uK scatters"
        )
        for point in points:
            result = point.result
            print(
                f"{point.parameter_value:12.6g} "
                f"{result.transfer_efficiency:10.4f} "
                f"{result.transfer_standard_error:8.4f} "
                f"{_format_optional(result.final_temperature_uK):>12} "
                f"{_format_optional(result.handover_heating_uK):>10} "
                f"{result.mean_scattering_events:8.3f}"
            )
        payload = [asdict(point) for point in points]
        rows = [_handover_scan_row(point) for point in points]
        if args.plot:
            saved = _save_handover_plot(args.plot, scan=points)
            print(f"扫描图已写入: {saved}")

    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON 已写入: {output}")
    if args.csv:
        output = Path(args.csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV 已写入: {output}")
    return 0


def _plots_command(args: argparse.Namespace) -> int:
    from .plots import generate_plots

    rb_path, cs_path = generate_plots(args.output_dir)
    print(f"已生成: {rb_path}")
    print(f"已生成: {cs_path}")
    return 0


def _lp_design_command(args: argparse.Namespace) -> int:
    times = tuple(
        float(value.strip())
        for value in args.handover_times.split(",")
        if value.strip()
    )
    inputs = LinearDesignInputs(
        atom_label=args.atom,
        detuning_min_ghz=args.detuning_min,
        detuning_max_ghz=args.detuning_max,
        segment_count=args.segments,
        waist_um=args.waist,
        target_depth_uK=args.depth,
        design_temperature_uK=args.temperature,
        target_bound_fraction=args.bound_fraction,
        acceleration_m_s2=args.acceleration,
        handover_min_axial_cycles=args.min_cycles,
        max_source_power_w=args.max_power,
        max_scattering_rate_s=args.max_scattering,
        delivery_efficiency=args.delivery_efficiency,
        retro_power_ratio=args.retro_ratio,
        detuning_objective_weight=args.detuning_weight,
    )
    result = analyze_detuning_power_lp(
        inputs,
        handover_times_ms=times,
    )
    print(f"{inputs.atom_label} 失谐量--源端功率分段线性规划")
    print(
        "约束: "
        f"U>={inputs.target_depth_uK:g} µK, "
        f"bound>={inputs.target_bound_fraction:.3f}, "
        f"scatter<={inputs.max_scattering_rate_s:g} s^-1, "
        f"P<={inputs.max_source_power_w:g} W"
    )
    for time_result in result.handover_results:
        point = time_result.recommended
        if point is None:
            print(
                f"handover {time_result.handover_time_ms:g} ms: "
                "无 LP 可行域"
            )
            continue
        active = ", ".join(point.active_constraints) or "无（内部点）"
        print(
            f"handover {time_result.handover_time_ms:g} ms: "
            f"Δ={point.detuning_ghz:.2f} GHz, "
            f"P_source={point.source_power_w:.3f} W, "
            f"U={point.depth_uK:.1f} µK, "
            f"scatter={point.scattering_rate_s:.1f} s^-1, "
            f"bound={point.bound_fraction:.3f}, "
            f"cycles={point.handover_axial_cycles:.1f}; "
            f"active={active}"
        )

    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON 已写入: {output}")
    if args.plot:
        output = plot_detuning_power_lp(result, args.plot)
        print(f"LP 几何图已写入: {output}")
    return 0


def _handover_map_command(args: argparse.Namespace) -> int:
    scan = HandoverMapInputs(
        detuning_min_ghz=args.detuning_min,
        detuning_max_ghz=args.detuning_max,
        source_power_min_w=args.power_min,
        source_power_max_w=args.power_max,
        detuning_points=args.detuning_points,
        power_points=args.power_points,
        particle_count=args.particles,
        time_step_us=args.time_step,
        include_scattering=args.include_scattering,
        seed=args.seed,
        parallel_backend=args.parallel_backend,
        worker_count=args.workers,
        require_minimum_depth=args.minimum_depth,
        require_thermal_bound_fraction=args.thermal_bound_fraction,
        require_minimum_axial_cycles=args.minimum_axial_cycles,
    )
    print(
        "Cs-133/Rb-87 失谐量--功率可行域 handover Monte Carlo；"
        f"交接时间为 {scan.handover_time_us:g} µs"
    )
    integrated_results = ()
    if scan.use_l1_transport:
        result, integrated_results = analyze_dual_species_l1_handover_map(
            scan,
            progress=lambda message: print(message, flush=True),
        )
    else:
        result = analyze_dual_species_handover_map(
            scan,
            progress=lambda message: print(message, flush=True),
        )
    for species in result.species:
        print(
            f"{species.atom_label}: "
            f"{species.evaluated_points}/"
            f"{scan.detuning_points * scan.power_points} 个网格点完成计算"
        )
    if scan.write_l1_outputs:
        for integrated in integrated_results:
            slug = integrated.inputs.transport.atom_label.lower().replace(
                "-",
                "",
            )
            integrated_json = Path(
                f"output/l1_handover_scan_{slug}.json"
            )
            integrated_json.write_text(
                json.dumps(
                    asdict(integrated),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            integrated_plot = plot_l1_handover_scan(
                integrated,
                f"output/l1_handover_scan_{slug}.png",
            )
            print(
                f"{integrated.inputs.transport.atom_label} 联合结果已写入: "
                f"{integrated_json}, {integrated_plot}"
            )
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON 已写入: {output}")
    if args.plot:
        output = plot_dual_species_handover_map(result, args.plot)
        print(f"双原子体系 handover 热力图已写入: {output}")
    return 0


def _handover_angle_scan_command(args: argparse.Namespace) -> int:
    inputs = HandoverAngleScanInputs(
        angle_min_deg=args.angle_min,
        angle_max_deg=args.angle_max,
        angle_step_deg=args.angle_step,
        particle_count=args.particles,
        time_step_us=args.time_step,
        include_scattering=args.include_scattering,
        parallel_backend=args.parallel_backend,
        worker_count=args.workers,
    )
    print(
        "Rb-87/Cs-133 handover 夹角扫描："
        f"{inputs.angle_min_deg:g}–{inputs.angle_max_deg:g}°，"
        f"步长 {inputs.angle_step_deg:g}°，每点 N={inputs.particle_count}"
    )
    result = analyze_handover_angle_scan(
        inputs,
        progress=lambda message: print(message, flush=True),
    )
    for species in result.species:
        print(
            f"{species.atom_label}: Δ={species.detuning_ghz:g} GHz，"
            f"固定阱深={inputs.target_depth_uK:g} µK，"
            f"源端功率={species.required_source_power_w:.3f} W"
        )
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"JSON 已写入: {json_path}")
    plot_path = plot_handover_angle_scan(result, args.plot)
    print(f"夹角扫描折线图已写入: {plot_path}")
    return 0


def _l1_transport_scan_command(args: argparse.Namespace) -> int:
    species_defaults = l1_transport_inputs_for_species(args.atom)
    inputs = replace(
        species_defaults,
        detuning_min_ghz=args.detuning_min,
        detuning_max_ghz=args.detuning_max,
        detuning_points=args.detuning_points,
        handover_source_power_min_w=args.power_min,
        handover_source_power_max_w=args.power_max,
        power_points=args.power_points,
        acceleration_m_s2=args.acceleration,
        maximum_velocity_m_s=args.velocity,
        time_points=args.time_points,
        initial_temperature_uK=(
            species_defaults.initial_temperature_uK
            if args.initial_temperature is None
            else args.initial_temperature
        ),
        delivery_efficiency=(
            species_defaults.delivery_efficiency
            if args.delivery_efficiency is None
            else args.delivery_efficiency
        ),
        background_loss_rate_s=args.background_loss_rate,
        internal_loss_probability_per_scatter=args.scatter_loss_probability,
        two_body_loss_coefficient_m3_s=args.two_body_loss,
        three_body_loss_coefficient_m6_s=args.three_body_loss,
        require_minimum_depth=args.minimum_depth,
        require_maximum_start_power=args.maximum_start_power,
        require_critical_acceleration=args.critical_acceleration,
        include_gravity=args.gravity,
        transport_method=(
            species_defaults.transport_method
            if args.transport_method is None
            else args.transport_method
        ),
        kinematic_profile=(
            species_defaults.kinematic_profile
            if args.kinematic_profile is None
            else args.kinematic_profile
        ),
    )
    print(
        f"{inputs.atom_label} L1 宏观运输扫描："
        f"a={inputs.acceleration_m_s2:g} m/s²，"
        f"v={inputs.maximum_velocity_m_s:g} m/s，"
        f"T0={inputs.initial_temperature_uK:g} µK，"
        f"光路效率={inputs.delivery_efficiency:g}"
    )
    result = analyze_l1_transport_scan(
        inputs,
        progress=lambda message: print(message, flush=True),
    )
    print(
        "时序："
        f"加速 {1e3 * result.timing.acceleration_time_s:.3f} ms，"
        f"匀速 {1e3 * result.timing.cruise_time_s:.3f} ms，"
        f"总计 {1e3 * result.timing.total_time_s:.3f} ms"
    )
    for label, point in (
        ("最优点", result.optimal),
        ("较差可行点", result.comparison),
    ):
        print(
            f"{label}: Δ={point.detuning_ghz:.2f} GHz, "
            f"P_h={point.handover_source_power_w:.3f} W/分支, "
            f"ΔT={point.final_temperature_rise_uK:.3f} µK, "
            f"留存率={point.final_retention_fraction:.6f}, "
            f"散射={point.cumulative_scattering_events:.2f} 次"
        )
    for reference in result.reference_points:
        point = reference.point
        print(
            f"参考点 {reference.label}: Δ={point.detuning_ghz:.2f} GHz, "
            f"P_h={point.handover_source_power_w:.3f} W/分支, "
            f"U={point.depth_uK:.2f} µK, "
            f"ΔT={point.final_temperature_rise_uK:.3f} µK, "
            f"留存率={point.final_retention_fraction:.6f}"
        )
        if reference.note:
            print(f"  功率定义说明：{reference.note}")
    if not any(
        (
            inputs.background_loss_rate_s,
            inputs.internal_loss_probability_per_scatter,
            inputs.two_body_loss_coefficient_m3_s,
            inputs.three_body_loss_coefficient_m6_s,
        )
    ):
        print("说明：速率损失系数均为零；留存率当前主要表示有限势垒热溢出。")
    species_slug = inputs.atom_label.lower().replace("-", "")
    json_path = args.json or f"output/l1_transport_scan_{species_slug}.json"
    plot_path = args.plot or f"output/l1_transport_scan_{species_slug}.png"
    if json_path:
        output = Path(json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON 已写入: {output}")
    if plot_path:
        output = plot_l1_transport_scan(result, plot_path)
        print(f"L1 二维扫描与时间曲线已写入: {output}")
    return 0


def _l1_handover_scan_command(args: argparse.Namespace) -> int:
    """运行共享网格上的 L1 transport→handover 联合扫描。"""
    species_defaults = l1_transport_inputs_for_species(args.atom)
    transport = replace(
        species_defaults,
        detuning_min_ghz=args.detuning_min,
        detuning_max_ghz=args.detuning_max,
        detuning_points=args.detuning_points,
        handover_source_power_min_w=args.power_min,
        handover_source_power_max_w=args.power_max,
        power_points=args.power_points,
        initial_temperature_uK=(
            species_defaults.initial_temperature_uK
            if args.initial_temperature is None
            else args.initial_temperature
        ),
        time_points=args.time_points,
        require_minimum_depth=args.minimum_depth,
        require_maximum_start_power=args.maximum_start_power,
        require_critical_acceleration=args.critical_acceleration,
        include_gravity=args.gravity,
        transport_method=(
            species_defaults.transport_method
            if args.transport_method is None
            else args.transport_method
        ),
        kinematic_profile=(
            species_defaults.kinematic_profile
            if args.kinematic_profile is None
            else args.kinematic_profile
        ),
        # 运输 MC 与 handover MC 合并调用同一组数值参数。
        mc_particle_count=args.particles,
        mc_include_scattering=args.include_scattering,
        mc_compute_backend=args.compute_backend,
    )
    defaults = L1HandoverInputs(transport=transport)
    inputs = replace(
        defaults,
        particle_count=args.particles,
        time_step_us=args.time_step,
        compute_backend=args.compute_backend,
        parallel_backend=args.parallel_backend,
        worker_count=args.workers,
        include_scattering=args.include_scattering,
    )
    print(
        f"{transport.atom_label} L1 transport→handover 联合扫描："
        f"Δ={transport.detuning_min_ghz:g}–{transport.detuning_max_ghz:g} GHz，"
        f"P={transport.handover_source_power_min_w:g}–"
        f"{transport.handover_source_power_max_w:g} W，"
        f"T_MOT={transport.initial_temperature_uK:g} µK，"
        f"每点 N={inputs.particle_count}"
    )
    result = analyze_l1_handover_scan(
        inputs,
        progress=lambda message: print(message, flush=True),
    )
    for label, point in (
        ("联合最优点", result.optimal),
        ("较差可行点", result.comparison),
    ):
        efficiency = point.handover_transfer_efficiency
        efficiency_text = (
            "无"
            if efficiency is None or math.isnan(float(efficiency))
            else f"{efficiency:.4f}"
        )
        rise = point.total_temperature_rise_uK
        rise_text = (
            "无"
            if rise is None or math.isnan(float(rise))
            else f"{rise:.3f}"
        )
        print(
            f"{label}: Δ={point.detuning_ghz:.2f} GHz, "
            f"P={point.source_power_w:.4f} W, "
            f"η_HO={efficiency_text}, "
            f"MOT→L2={point.final_retention_from_mot:.4f}, "
            f"ΔT_total={rise_text} µK"
        )
    species_slug = transport.atom_label.lower().replace("-", "")
    json_path = args.json or f"output/l1_handover_scan_{species_slug}.json"
    plot_path = args.plot or f"output/l1_handover_scan_{species_slug}.png"
    output = Path(json_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"JSON 已写入: {output}")
    plot_output = plot_l1_handover_scan(result, plot_path)
    print(f"全流程热力图与时间轨迹已写入: {plot_output}")
    return 0


def _full_chain_scan_command(args: argparse.Namespace) -> int:
    """运行共享网格上的 L1→handover→L2 全链路扫描。"""
    species_defaults = l1_transport_inputs_for_species(args.atom)
    # 连续相空间扫描要求轨迹级 MC 与平滑轨迹；未显式指定时自动切换，
    # 显式给出的 --transport-method/--kinematic-profile 永远优先。
    transport_method = args.transport_method
    kinematic_profile = args.kinematic_profile
    if args.phase_space_continuity and (
        transport_method is None or kinematic_profile is None
    ):
        auto = []
        if transport_method is None:
            transport_method = "monte_carlo"
            auto.append("transport_method=monte_carlo")
        if kinematic_profile is None:
            kinematic_profile = "minimum_jerk"
            auto.append("minimum_jerk 平滑轨迹（L1/L2）")
        print(
            "说明：相空间连续模式自动启用 " + " 与 ".join(auto) + "；可用 "
            "--transport-method/--kinematic-profile 显式覆盖，或用 "
            "--no-phase-space-continuity 走解析快扫"
        )
    transport = replace(
        species_defaults,
        detuning_min_ghz=args.detuning_min,
        detuning_max_ghz=args.detuning_max,
        detuning_points=args.detuning_points,
        handover_source_power_min_w=args.power_min,
        handover_source_power_max_w=args.power_max,
        power_points=args.power_points,
        initial_temperature_uK=(
            species_defaults.initial_temperature_uK
            if args.initial_temperature is None
            else args.initial_temperature
        ),
        time_points=args.time_points,
        require_minimum_depth=args.minimum_depth,
        require_maximum_start_power=args.maximum_start_power,
        require_critical_acceleration=args.critical_acceleration,
        include_gravity=args.gravity,
        transport_method=(
            species_defaults.transport_method
            if transport_method is None
            else transport_method
        ),
        kinematic_profile=(
            species_defaults.kinematic_profile
            if kinematic_profile is None
            else kinematic_profile
        ),
        # 运输 MC 与 handover MC 合并调用同一组数值参数。
        mc_particle_count=args.particles,
        mc_include_scattering=args.include_scattering,
        mc_compute_backend=args.compute_backend,
    )
    handover_defaults = L1HandoverInputs(transport=transport)
    handover = replace(
        handover_defaults,
        particle_count=args.particles,
        time_step_us=args.time_step,
        compute_backend=args.compute_backend,
        parallel_backend=args.parallel_backend,
        worker_count=args.workers,
        include_scattering=args.include_scattering,
    )
    # L2 腿经 l2_leg_inputs 的 replace 继承 L1 的 transport_method，
    # 但 kinematic_profile 取自 L2TransportInputs，需一并解析。
    l2_defaults = L2TransportInputs()
    l2 = replace(
        l2_defaults,
        time_points=args.l2_time_points,
        kinematic_profile=(
            l2_defaults.kinematic_profile
            if kinematic_profile is None
            else kinematic_profile
        ),
    )
    inputs = FullChainInputs(
        handover=handover,
        l2=l2,
        phase_space_continuity=args.phase_space_continuity,
    )
    print(
        f"{transport.atom_label} L1→handover→L2 全链路扫描："
        f"Δ={transport.detuning_min_ghz:g}–{transport.detuning_max_ghz:g} GHz，"
        f"P={transport.handover_source_power_min_w:g}–"
        f"{transport.handover_source_power_max_w:g} W，"
        f"T_MOT={transport.initial_temperature_uK:g} µK，"
        f"每点 N={handover.particle_count}"
    )
    result = analyze_full_chain_scan(
        inputs,
        progress=lambda message: print(message, flush=True),
    )
    for label, point in (
        ("全链路最优点", result.optimal),
        ("较差可行点", result.comparison),
    ):
        density = point.science_peak_density_m3
        density_text = "无" if density is None else f"{density:.3e} m^-3"
        efficiency = point.l1_handover.handover_transfer_efficiency
        efficiency_text = (
            "无"
            if efficiency is None or math.isnan(float(efficiency))
            else f"{efficiency:.4f}"
        )
        rise = point.science_total_temperature_rise_uK
        rise_text = (
            "无"
            if rise is None or math.isnan(float(rise))
            else f"{rise:.3f}"
        )
        print(
            f"{label}: Δ={point.detuning_ghz:.2f} GHz, "
            f"P={point.source_power_w:.4f} W, "
            f"η_HO={efficiency_text}, "
            f"MOT→科学区={point.final_retention_from_mot:.4f}, "
            f"ΔT_total={rise_text} µK, "
            f"科学区峰值密度={density_text}"
        )
    species_slug = transport.atom_label.lower().replace("-", "")
    json_path = args.json or f"output/full_chain_scan_{species_slug}.json"
    plot_path = args.plot or f"output/full_chain_scan_{species_slug}.png"
    output = Path(json_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"JSON 已写入: {output}")
    plot_output = plot_full_chain_scan(result, plot_path)
    print(f"全链路热力图与时间轨迹已写入: {plot_output}")
    return 0


def _optimize_design_command(args: argparse.Namespace) -> int:
    inputs = RobustDesignInputs(
        atom_label=args.atom,
        detuning_min_ghz=args.detuning_min,
        detuning_max_ghz=args.detuning_max,
        detuning_points=args.detuning_points,
        source_power_min_w=args.power_min,
        source_power_max_w=args.power_max,
        power_points=args.power_points,
        waist_min_um=args.waist_min,
        waist_max_um=args.waist_max,
        waist_points=args.waist_points,
        relative_tolerance=args.tolerance,
        variation_mode=args.variation_mode,
        monte_carlo_candidate_count=args.candidates,
        minimum_transfer_efficiency=args.min_efficiency,
        particle_count=args.particles,
        time_step_us=args.time_step,
        parallel_backend=args.parallel_backend,
        worker_count=args.workers,
        phase_points=args.phase_points,
    )
    print(
        f"{inputs.atom_label} 稳健设计优化：扫描失谐、功率、束腰；"
        f"三变量 ±{100 * inputs.relative_tolerance:g}% "
        f"{inputs.variation_mode} 稳健检查"
    )
    result = analyze_robust_design(
        inputs,
        progress=lambda message: print(message, flush=True),
    )
    print(f"稳健可行点: {result.robust_point_count}")
    if result.recommended is None:
        print("没有候选同时通过稳健约束和 Monte Carlo 交接率阈值")
    else:
        item = result.recommended
        point = item.design
        print(
            "推荐工作点: "
            f"Δ={point.detuning_ghz:.2f} GHz, "
            f"P={point.source_power_w:.3f} W/分支, "
            f"w={point.waist_um:.1f} µm, "
            f"η={item.transfer_efficiency:.4f}±"
            f"{item.transfer_standard_error:.4f}, "
            f"平台评分={point.plateau_score:.3f}, "
            f"最差约束裕量={point.worst_constraint_margin:.3f} "
            f"({point.worst_constraint})"
        )
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON 已写入: {output}")
    if args.plot:
        output = plot_robust_design(result, args.plot)
        print(f"稳健优化图已写入: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="连续装载论文 Rb 复现与 Cs 光晶格反推"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    paper = subparsers.add_parser("paper", help="复现论文 Rb 双晶格参数")
    paper.add_argument("--json", help="保存完整结果到 JSON")
    paper.set_defaults(handler=_paper_command)

    cs = subparsers.add_parser("cs-scan", help="扫描 Cs D1 红失谐候选")
    cs.add_argument("--depth", type=float, default=500.0, help="目标阱深 µK")
    cs.add_argument("--waist", type=float, default=250.0, help="束腰 µm")
    cs.add_argument("--detuning-min", type=float, default=100.0, help="最小红失谐 GHz")
    cs.add_argument("--detuning-max", type=float, default=2000.0, help="最大红失谐 GHz")
    cs.add_argument("--detuning-step", type=float, default=100.0, help="扫描步长 GHz")
    cs.add_argument(
        "--retro-ratio",
        type=float,
        default=0.88**4,
        help="原子处回程/前向功率比",
    )
    cs.add_argument(
        "--delivery-efficiency",
        type=float,
        default=1.0,
        help="源端到原子处前向功率效率",
    )
    cs.add_argument("--max-power", type=float, help="源端最大功率 W")
    cs.add_argument("--max-scattering", type=float, help="最大散射率 s^-1")
    cs.add_argument("--only-feasible", action="store_true", help="只打印可行候选")
    cs.add_argument("--csv", help="保存全部候选到 CSV")
    cs.set_defaults(handler=_cs_command)

    cs_transport = subparsers.add_parser(
        "cs-transport",
        help="把论文同型双晶格运输应用到 Cs",
    )
    cs_transport.add_argument("--detuning", type=float, default=600.0, help="D1 红失谐 GHz")
    cs_transport.add_argument("--depth", type=float, default=500.0, help="恒定目标阱深 µK")
    cs_transport.add_argument("--temperature", type=float, default=20.0, help="初始温度 µK")
    cs_transport.add_argument("--acceleration", type=float, default=4000.0, help="加速度 m/s²")
    cs_transport.add_argument(
        "--handover-fraction",
        type=float,
        default=0.63,
        help="交接等效随机相位比例 [0,1]",
    )
    cs_transport.add_argument(
        "--retro-ratio",
        type=float,
        default=0.88**4,
        help="原子处回程/前向功率比",
    )
    cs_transport.add_argument(
        "--delivery-efficiency",
        type=float,
        default=1.0,
        help="源端到原子处前向功率效率",
    )
    cs_transport.set_defaults(handler=_cs_transport_command)

    handover = subparsers.add_parser(
        "handover",
        help="经典轨迹 Monte Carlo 计算双晶格交接率和升温",
    )
    handover.add_argument(
        "--detuning",
        type=float,
        default=300.0,
        help="相对 Rb D1 的红失谐 GHz",
    )
    handover.add_argument(
        "--depth1",
        type=float,
        default=500.0,
        help="Lattice-1 满功率阱深 µK",
    )
    handover.add_argument(
        "--depth2",
        type=float,
        default=500.0,
        help="Lattice-2 满功率阱深 µK",
    )
    handover.add_argument(
        "--waist1",
        type=float,
        default=250.0,
        help="交接点 Lattice-1 束腰 µm",
    )
    handover.add_argument(
        "--waist2",
        type=float,
        default=250.0,
        help="交接点 Lattice-2 束腰 µm",
    )
    handover.add_argument(
        "--temperature",
        type=float,
        default=30.8,
        help="交接前 Lattice-1 温度 µK",
    )
    handover.add_argument(
        "--atom-number",
        type=float,
        default=4_000_000.0,
        help="进入 handover 的 Lattice-1 实际原子数",
    )
    handover.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="反向线性强度斜坡时间 ms（Methods 工作点 1 ms）",
    )
    handover.add_argument(
        "--distance",
        type=float,
        default=38.85,
        help="Lattice-1 到交接点的运输距离 cm",
    )
    handover.add_argument(
        "--optimal-distance",
        type=float,
        default=38.85,
        help="空间 mode-match 的最佳 Lattice-1 距离 cm",
    )
    handover.add_argument(
        "--acceleration",
        type=float,
        default=4_000.0,
        help="交接后 Lattice-2 加速度 m/s²",
    )
    handover.add_argument(
        "--angle",
        type=float,
        default=4.0,
        help="两晶格交叉角度 deg（论文未给精确值，默认用腔体倾角）",
    )
    handover.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Lattice-2 垂直于交叉平面的横向偏移 µm",
    )
    handover.add_argument(
        "--cloud-size",
        type=float,
        default=0.5,
        help="跨多个 Lattice-1 格点的原子云轴向 1σ 尺寸 mm",
    )
    handover.add_argument(
        "--phase",
        type=float,
        default=0.0,
        help="两驻波相对相位 rad",
    )
    handover.add_argument(
        "--fixed-phase",
        action="store_true",
        help="所有原子使用固定相位；默认对未锁定格点映射作均匀平均",
    )
    handover.add_argument(
        "--velocity1",
        type=float,
        default=0.0,
        help="handover 时 Lattice-1 速度 m/s",
    )
    handover.add_argument(
        "--velocity2",
        type=float,
        default=0.0,
        help="handover 时 Lattice-2 速度 m/s",
    )
    handover.add_argument(
        "--particles",
        type=int,
        default=2_000,
        help="Monte Carlo 轨迹数",
    )
    handover.add_argument(
        "--time-step",
        type=float,
        default=0.1,
        help="积分目标时间步长 µs",
    )
    handover.add_argument(
        "--trace-points",
        type=int,
        default=51,
        help="交接期间保存的诊断点数",
    )
    handover.add_argument("--seed", type=int, default=20_250_902)
    handover.add_argument(
        "--no-scattering",
        action="store_true",
        help="关闭自发散射 Monte Carlo 反冲",
    )
    handover.add_argument(
        "--figure2-panel",
        choices=("a", "b", "c"),
        help="使用 Extended Data Fig. 2 对应面板的扫描网格",
    )
    handover.add_argument(
        "--scan-parameter",
        choices=tuple(_HANDOVER_SCAN_ALIASES),
        help="自定义扫描参数",
    )
    handover.add_argument(
        "--scan-values",
        help="逗号分隔的自定义扫描值",
    )
    handover.add_argument("--json", help="保存完整结果到 JSON")
    handover.add_argument("--csv", help="保存汇总结果到 CSV")
    handover.add_argument("--plot", help="保存温度轨迹或扫描图")
    handover.set_defaults(handler=_handover_command)

    handover_map = subparsers.add_parser(
        "handover-map",
        help="在失谐--功率可行域内计算 Rb/Cs handover 交接率热力图",
    )
    handover_map_defaults = HandoverMapInputs()
    handover_map.add_argument(
        "--detuning-min",
        type=float,
        default=handover_map_defaults.detuning_min_ghz,
        help="最小 D1 红失谐 GHz",
    )
    handover_map.add_argument(
        "--detuning-max",
        type=float,
        default=handover_map_defaults.detuning_max_ghz,
        help="最大 D1 红失谐 GHz",
    )
    handover_map.add_argument(
        "--power-min",
        type=float,
        default=handover_map_defaults.source_power_min_w,
        help="最小源端功率 W/晶格分支",
    )
    handover_map.add_argument(
        "--power-max",
        type=float,
        default=handover_map_defaults.source_power_max_w,
        help="最大源端功率 W/晶格分支",
    )
    handover_map.add_argument(
        "--detuning-points",
        type=int,
        default=handover_map_defaults.detuning_points,
        help="失谐网格点数",
    )
    handover_map.add_argument(
        "--power-points",
        type=int,
        default=handover_map_defaults.power_points,
        help="功率网格点数",
    )
    handover_map.add_argument(
        "--particles",
        type=int,
        default=handover_map_defaults.particle_count,
        help="每个可行网格点的 Monte Carlo 轨迹数",
    )
    handover_map.add_argument(
        "--time-step",
        type=float,
        default=handover_map_defaults.time_step_us,
        help="轨迹积分时间步长 µs",
    )
    handover_map.add_argument(
        "--seed",
        type=int,
        default=handover_map_defaults.seed,
    )
    handover_map.add_argument(
        "--parallel-backend",
        choices=("serial", "process"),
        default=handover_map_defaults.parallel_backend,
        help="网格点计算后端；process 使用多 CPU 进程",
    )
    handover_map.add_argument(
        "--workers",
        type=int,
        default=handover_map_defaults.worker_count,
        help="CPU 工作进程数；设为 1 时串行",
    )
    handover_map.set_defaults(
        include_scattering=handover_map_defaults.include_scattering
    )
    handover_map.add_argument(
        "--no-scattering",
        dest="include_scattering",
        action="store_false",
        help="关闭 handover 期间的随机散射反冲",
    )
    handover_map.add_argument(
        "--minimum-depth",
        action=argparse.BooleanOptionalAction,
        default=handover_map_defaults.require_minimum_depth,
        help="是否启用最小阱深前置条件",
    )
    handover_map.add_argument(
        "--thermal-bound-fraction",
        action=argparse.BooleanOptionalAction,
        default=handover_map_defaults.require_thermal_bound_fraction,
        help="是否启用热束缚比例前置条件",
    )
    handover_map.add_argument(
        "--minimum-axial-cycles",
        action=argparse.BooleanOptionalAction,
        default=handover_map_defaults.require_minimum_axial_cycles,
        help="是否启用最少轴向周期前置条件",
    )
    handover_map.add_argument(
        "--plot",
        default="output/handover_efficiency_map.png",
        help="双面板热力图输出路径",
    )
    handover_map.add_argument("--json", help="保存完整网格数据到 JSON")
    handover_map.set_defaults(handler=_handover_map_command)

    angle_defaults = HandoverAngleScanInputs()
    angle_scan = subparsers.add_parser(
        "handover-angle-scan",
        help="扫描 L1/L2 夹角并比较 Rb/Cs 的交接率和升温",
    )
    angle_scan.add_argument(
        "--angle-min", type=float, default=angle_defaults.angle_min_deg
    )
    angle_scan.add_argument(
        "--angle-max", type=float, default=angle_defaults.angle_max_deg
    )
    angle_scan.add_argument(
        "--angle-step", type=float, default=angle_defaults.angle_step_deg
    )
    angle_scan.add_argument(
        "--particles", type=int, default=angle_defaults.particle_count
    )
    angle_scan.add_argument(
        "--time-step", type=float, default=angle_defaults.time_step_us
    )
    angle_scan.add_argument(
        "--parallel-backend",
        choices=("serial", "process"),
        default=angle_defaults.parallel_backend,
    )
    angle_scan.add_argument(
        "--workers", type=int, default=angle_defaults.worker_count
    )
    angle_scan.set_defaults(
        include_scattering=angle_defaults.include_scattering
    )
    angle_scan.add_argument(
        "--no-scattering",
        dest="include_scattering",
        action="store_false",
    )
    angle_scan.add_argument(
        "--plot",
        default="output/handover_angle_scan.png",
    )
    angle_scan.add_argument(
        "--json",
        default="output/handover_angle_scan.json",
    )
    angle_scan.set_defaults(handler=_handover_angle_scan_command)

    l1_defaults = L1TransportInputs()
    l1_scan = subparsers.add_parser(
        "l1-transport-scan",
        help="固定加速度和速度，扫描 L1 失谐--功率的升温与留存率",
    )
    l1_scan.add_argument(
        "--atom",
        choices=("Rb-87", "Cs-133"),
        default=l1_defaults.atom_label,
    )
    l1_scan.add_argument(
        "--detuning-min", type=float, default=l1_defaults.detuning_min_ghz
    )
    l1_scan.add_argument(
        "--detuning-max", type=float, default=l1_defaults.detuning_max_ghz
    )
    l1_scan.add_argument(
        "--detuning-points", type=int, default=l1_defaults.detuning_points
    )
    l1_scan.add_argument(
        "--power-min",
        type=float,
        default=l1_defaults.handover_source_power_min_w,
        help="handover 端每分支源端功率下限 W",
    )
    l1_scan.add_argument(
        "--power-max",
        type=float,
        default=l1_defaults.handover_source_power_max_w,
        help="handover 端每分支源端功率上限 W",
    )
    l1_scan.add_argument(
        "--power-points", type=int, default=l1_defaults.power_points
    )
    l1_scan.add_argument(
        "--acceleration",
        type=float,
        default=l1_defaults.acceleration_m_s2,
        help="固定加速度幅值 m/s²",
    )
    l1_scan.add_argument(
        "--velocity",
        type=float,
        default=l1_defaults.maximum_velocity_m_s,
        help="固定最大运输速度 m/s",
    )
    l1_scan.add_argument(
        "--time-points", type=int, default=l1_defaults.time_points
    )
    l1_scan.add_argument(
        "--transport-method",
        choices=("analytic", "monte_carlo"),
        help="L1 运输腿的计算方式；省略时读取全局 JSON 配置",
    )
    l1_scan.add_argument(
        "--kinematic-profile",
        choices=("trapezoid", "minimum_jerk"),
        help="L1 理想运输轨迹；省略时读取全局 JSON 配置",
    )
    l1_scan.add_argument(
        "--initial-temperature",
        type=float,
        help="L1 起点初始温度 µK；省略时使用物种配置",
    )
    l1_scan.add_argument(
        "--delivery-efficiency",
        type=float,
        help="源端到原子处的功率效率；省略时使用物种配置",
    )
    l1_scan.add_argument(
        "--background-loss-rate",
        type=float,
        default=l1_defaults.background_loss_rate_s,
        help="背景一体损失率 s^-1",
    )
    l1_scan.add_argument(
        "--scatter-loss-probability",
        type=float,
        default=l1_defaults.internal_loss_probability_per_scatter,
        help="每次散射进入非束缚内态的概率",
    )
    l1_scan.add_argument(
        "--two-body-loss",
        type=float,
        default=l1_defaults.two_body_loss_coefficient_m3_s,
        help="二体损失系数 m^3/s",
    )
    l1_scan.add_argument(
        "--three-body-loss",
        type=float,
        default=l1_defaults.three_body_loss_coefficient_m6_s,
        help="三体损失系数 m^6/s",
    )
    l1_scan.add_argument(
        "--minimum-depth",
        action=argparse.BooleanOptionalAction,
        default=l1_defaults.require_minimum_depth,
        help="是否启用最小阱深前置条件",
    )
    l1_scan.add_argument(
        "--maximum-start-power",
        action=argparse.BooleanOptionalAction,
        default=l1_defaults.require_maximum_start_power,
        help="是否启用 L1 起点最大功率前置条件",
    )
    l1_scan.add_argument(
        "--critical-acceleration",
        action=argparse.BooleanOptionalAction,
        default=l1_defaults.require_critical_acceleration,
        help="是否启用临界加速度前置条件",
    )
    l1_scan.add_argument(
        "--gravity",
        action=argparse.BooleanOptionalAction,
        default=l1_defaults.include_gravity,
        help="是否在 L1/L2 运输和 handover 中统一启用沿 -y 的重力",
    )
    l1_scan.add_argument(
        "--plot",
        help="四联图输出路径；默认按原子命名",
    )
    l1_scan.add_argument(
        "--json",
        help="完整网格和两组时间轨迹输出路径；默认按原子命名",
    )
    l1_scan.set_defaults(handler=_l1_transport_scan_command)

    l1_handover_defaults = L1HandoverInputs()
    l1_handover = subparsers.add_parser(
        "l1-handover-scan",
        help="在同一失谐--功率网格上连接 L1 运输与 handover Monte Carlo",
    )
    l1_handover.add_argument(
        "--atom",
        choices=("Rb-87", "Cs-133"),
        default=l1_handover_defaults.transport.atom_label,
    )
    l1_handover.add_argument(
        "--detuning-min",
        type=float,
        default=l1_handover_defaults.transport.detuning_min_ghz,
    )
    l1_handover.add_argument(
        "--detuning-max",
        type=float,
        default=l1_handover_defaults.transport.detuning_max_ghz,
    )
    l1_handover.add_argument(
        "--detuning-points",
        type=int,
        default=l1_handover_defaults.transport.detuning_points,
    )
    l1_handover.add_argument(
        "--power-min",
        type=float,
        default=l1_handover_defaults.transport.handover_source_power_min_w,
    )
    l1_handover.add_argument(
        "--power-max",
        type=float,
        default=l1_handover_defaults.transport.handover_source_power_max_w,
    )
    l1_handover.add_argument(
        "--power-points",
        type=int,
        default=l1_handover_defaults.transport.power_points,
    )
    l1_handover.add_argument(
        "--initial-temperature",
        type=float,
        help="MOT 出射并装入 L1 时的初温 µK；默认读取全局 JSON 的 20 µK",
    )
    l1_handover.add_argument(
        "--time-points",
        type=int,
        default=l1_handover_defaults.transport.time_points,
    )
    l1_handover.add_argument(
        "--transport-method",
        choices=("analytic", "monte_carlo"),
        help="L1 运输腿的计算方式；省略时读取全局 JSON 配置",
    )
    l1_handover.add_argument(
        "--kinematic-profile",
        choices=("trapezoid", "minimum_jerk"),
        help="L1 理想运输轨迹；省略时读取全局 JSON 配置",
    )
    l1_handover.add_argument(
        "--minimum-depth",
        action=argparse.BooleanOptionalAction,
        default=l1_handover_defaults.transport.require_minimum_depth,
        help="是否启用最小阱深前置条件",
    )
    l1_handover.add_argument(
        "--maximum-start-power",
        action=argparse.BooleanOptionalAction,
        default=l1_handover_defaults.transport.require_maximum_start_power,
        help="是否启用 L1 起点最大功率前置条件",
    )
    l1_handover.add_argument(
        "--critical-acceleration",
        action=argparse.BooleanOptionalAction,
        default=l1_handover_defaults.transport.require_critical_acceleration,
        help="是否启用临界加速度前置条件",
    )
    l1_handover.add_argument(
        "--gravity",
        action=argparse.BooleanOptionalAction,
        default=l1_handover_defaults.transport.include_gravity,
        help="是否在 L1 运输和 handover 中统一启用沿 -y 的重力",
    )
    l1_handover.add_argument(
        "--particles",
        type=int,
        default=l1_handover_defaults.particle_count,
        help="每个 L1 可行点的 handover Monte Carlo 轨迹数",
    )
    l1_handover.add_argument(
        "--time-step",
        type=float,
        default=l1_handover_defaults.time_step_us,
        help="handover 轨迹积分步长 µs",
    )
    l1_handover.add_argument(
        "--compute-backend",
        choices=("cpu", "gpu"),
        default=l1_handover_defaults.compute_backend,
        help="Monte Carlo 内层积分的计算设备；gpu 需要 CuPy/CUDA 环境",
    )
    l1_handover.add_argument(
        "--parallel-backend",
        choices=("serial", "process"),
        default=l1_handover_defaults.parallel_backend,
    )
    l1_handover.add_argument(
        "--workers",
        type=int,
        default=l1_handover_defaults.worker_count,
    )
    l1_handover.set_defaults(
        include_scattering=l1_handover_defaults.include_scattering
    )
    l1_handover.add_argument(
        "--no-scattering",
        dest="include_scattering",
        action="store_false",
    )
    l1_handover.add_argument("--plot", help="联合四联图输出路径")
    l1_handover.add_argument("--json", help="联合网格与时间轨迹 JSON 输出路径")
    l1_handover.set_defaults(handler=_l1_handover_scan_command)

    full_chain_defaults = FullChainInputs()
    full_chain = subparsers.add_parser(
        "full-chain-scan",
        help="在同一失谐--功率网格上完成 L1→handover→L2→科学区 全链路计算",
    )
    full_chain.add_argument(
        "--atom",
        choices=("Rb-87", "Cs-133"),
        default=full_chain_defaults.handover.transport.atom_label,
    )
    full_chain.add_argument(
        "--detuning-min",
        type=float,
        default=full_chain_defaults.handover.transport.detuning_min_ghz,
    )
    full_chain.add_argument(
        "--detuning-max",
        type=float,
        default=full_chain_defaults.handover.transport.detuning_max_ghz,
    )
    full_chain.add_argument(
        "--detuning-points",
        type=int,
        default=full_chain_defaults.handover.transport.detuning_points,
    )
    full_chain.add_argument(
        "--power-min",
        type=float,
        default=full_chain_defaults.handover.transport.handover_source_power_min_w,
    )
    full_chain.add_argument(
        "--power-max",
        type=float,
        default=full_chain_defaults.handover.transport.handover_source_power_max_w,
    )
    full_chain.add_argument(
        "--power-points",
        type=int,
        default=full_chain_defaults.handover.transport.power_points,
    )
    full_chain.add_argument(
        "--initial-temperature",
        type=float,
        help="MOT 出射并装入 L1 时的初温 µK；默认读取全局 JSON 的 20 µK",
    )
    full_chain.add_argument(
        "--time-points",
        type=int,
        default=full_chain_defaults.handover.transport.time_points,
    )
    full_chain.add_argument(
        "--transport-method",
        choices=("analytic", "monte_carlo"),
        help="L1/L2 运输腿的计算方式；省略时读取全局 JSON 配置"
        "（相空间连续模式下自动取 monte_carlo）",
    )
    full_chain.add_argument(
        "--kinematic-profile",
        choices=("trapezoid", "minimum_jerk"),
        help="L1/L2 理想运输轨迹；省略时读取全局 JSON 配置"
        "（相空间连续模式下自动取 minimum_jerk）",
    )
    full_chain.add_argument(
        "--l2-time-points",
        type=int,
        default=full_chain_defaults.l2.time_points,
        help="L2 段宏观运输的时间点数",
    )
    full_chain.add_argument(
        "--minimum-depth",
        action=argparse.BooleanOptionalAction,
        default=full_chain_defaults.handover.transport.require_minimum_depth,
        help="是否启用最小阱深前置条件",
    )
    full_chain.add_argument(
        "--maximum-start-power",
        action=argparse.BooleanOptionalAction,
        default=full_chain_defaults.handover.transport.require_maximum_start_power,
        help="是否启用 L1 起点最大功率前置条件",
    )
    full_chain.add_argument(
        "--critical-acceleration",
        action=argparse.BooleanOptionalAction,
        default=full_chain_defaults.handover.transport.require_critical_acceleration,
        help="是否启用临界加速度前置条件",
    )
    full_chain.add_argument(
        "--gravity",
        action=argparse.BooleanOptionalAction,
        default=full_chain_defaults.handover.transport.include_gravity,
        help="是否在 L1、handover、L2 全时序统一启用沿 -y 的重力",
    )
    full_chain.add_argument(
        "--phase-space-continuity",
        action=argparse.BooleanOptionalAction,
        default=full_chain_defaults.phase_space_continuity,
        help="是否以同一相空间系综贯穿 L1→handover→L2（默认开启；"
        "开启且未显式指定时自动启用运输 Monte Carlo 腿与 minimum_jerk "
        "平滑轨迹，可用 --no-phase-space-continuity 走解析快扫，回到 "
        "(N,T) 约化接口）",
    )
    full_chain.add_argument(
        "--particles",
        type=int,
        default=full_chain_defaults.handover.particle_count,
        help="每个 L1 可行点的 handover Monte Carlo 轨迹数",
    )
    full_chain.add_argument(
        "--time-step",
        type=float,
        default=full_chain_defaults.handover.time_step_us,
        help="handover 轨迹积分步长 µs",
    )
    full_chain.add_argument(
        "--compute-backend",
        choices=("cpu", "gpu"),
        default=full_chain_defaults.handover.compute_backend,
        help="Monte Carlo 内层积分的计算设备；gpu 需要 CuPy/CUDA 环境",
    )
    full_chain.add_argument(
        "--parallel-backend",
        choices=("serial", "process"),
        default=full_chain_defaults.handover.parallel_backend,
    )
    full_chain.add_argument(
        "--workers",
        type=int,
        default=full_chain_defaults.handover.worker_count,
    )
    full_chain.set_defaults(
        include_scattering=full_chain_defaults.handover.include_scattering
    )
    full_chain.add_argument(
        "--no-scattering",
        dest="include_scattering",
        action="store_false",
    )
    full_chain.add_argument("--plot", help="全链路四联图输出路径")
    full_chain.add_argument("--json", help="全链路网格与时间轨迹 JSON 输出路径")
    full_chain.set_defaults(handler=_full_chain_scan_command)

    optimize_design = subparsers.add_parser(
        "optimize-design",
        help="固定时序参数，稳健优化失谐、源端功率和束腰",
    )
    optimize_defaults = RobustDesignInputs()
    optimize_design.add_argument(
        "--atom",
        choices=("Cs-133", "Rb-87"),
        default=optimize_defaults.atom_label,
    )
    optimize_design.add_argument(
        "--detuning-min", type=float, default=optimize_defaults.detuning_min_ghz
    )
    optimize_design.add_argument(
        "--detuning-max", type=float, default=optimize_defaults.detuning_max_ghz
    )
    optimize_design.add_argument(
        "--detuning-points", type=int, default=optimize_defaults.detuning_points
    )
    optimize_design.add_argument(
        "--power-min", type=float, default=optimize_defaults.source_power_min_w
    )
    optimize_design.add_argument(
        "--power-max", type=float, default=optimize_defaults.source_power_max_w
    )
    optimize_design.add_argument(
        "--power-points", type=int, default=optimize_defaults.power_points
    )
    optimize_design.add_argument(
        "--waist-min", type=float, default=optimize_defaults.waist_min_um
    )
    optimize_design.add_argument(
        "--waist-max", type=float, default=optimize_defaults.waist_max_um
    )
    optimize_design.add_argument(
        "--waist-points", type=int, default=optimize_defaults.waist_points
    )
    optimize_design.add_argument(
        "--tolerance",
        type=float,
        default=optimize_defaults.relative_tolerance,
        help="失谐、功率、束腰共同的相对容差",
    )
    optimize_design.add_argument(
        "--variation-mode",
        choices=("one_at_a_time", "box_corners"),
        default=optimize_defaults.variation_mode,
        help="逐变量 ±容差，或检查三变量联合容差盒",
    )
    optimize_design.add_argument(
        "--candidates",
        type=int,
        default=optimize_defaults.monte_carlo_candidate_count,
        help="进入 Monte Carlo 的稳健候选数",
    )
    optimize_design.add_argument(
        "--min-efficiency",
        type=float,
        default=optimize_defaults.minimum_transfer_efficiency,
        help="推荐点的保守交接率下限",
    )
    optimize_design.add_argument(
        "--particles", type=int, default=optimize_defaults.particle_count
    )
    optimize_design.add_argument(
        "--time-step", type=float, default=optimize_defaults.time_step_us
    )
    optimize_design.add_argument(
        "--phase-points", type=int, default=optimize_defaults.phase_points
    )
    optimize_design.add_argument(
        "--parallel-backend",
        choices=("serial", "process"),
        default=optimize_defaults.parallel_backend,
    )
    optimize_design.add_argument(
        "--workers", type=int, default=optimize_defaults.worker_count
    )
    optimize_design.add_argument(
        "--plot",
        default="output/robust_design_optimization.png",
        help="三变量稳健优化与相位扫描图",
    )
    optimize_design.add_argument(
        "--json",
        default="output/robust_design_optimization.json",
        help="保存完整优化结果",
    )
    optimize_design.set_defaults(handler=_optimize_design_command)

    plots = subparsers.add_parser("plots", help="生成 Rb 复现和 Cs 扫描图")
    plots.add_argument("--output-dir", default="output", help="图片输出目录")
    plots.set_defaults(handler=_plots_command)

    lp_design = subparsers.add_parser(
        "lp-design",
        help="在失谐量--源端功率平面上求分段 LP 可行域",
    )
    lp_design.add_argument(
        "--atom",
        choices=("Cs-133", "Rb-87"),
        default="Cs-133",
        help="原子种类",
    )
    lp_design.add_argument(
        "--handover-times",
        default=",".join(str(value) for value in DEFAULT_HANDOVER_TIMES_MS),
        help="逗号分隔的 handover 时间 ms",
    )
    lp_design.add_argument("--detuning-min", type=float, default=300.0, help="最小 D1 红失谐 GHz")
    lp_design.add_argument("--detuning-max", type=float, default=1000.0, help="最大 D1 红失谐 GHz")
    lp_design.add_argument("--segments", type=int, default=28, help="分段线性 LP 区间数")
    lp_design.add_argument("--waist", type=float, default=250.0, help="束腰 µm")
    lp_design.add_argument("--depth", type=float, default=500.0, help="静态目标阱深 µK")
    lp_design.add_argument("--temperature", type=float, default=120.0, help="束缚比例设计温度 µK")
    lp_design.add_argument(
        "--bound-fraction",
        type=float,
        default=0.80,
        help="加速有效势垒下的目标热平衡束缚比例",
    )
    lp_design.add_argument("--acceleration", type=float, default=4000.0, help="handover 后加速度 m/s²")
    lp_design.add_argument(
        "--min-cycles",
        type=float,
        default=80.0,
        help="handover 内要求的最少轴向振荡周数",
    )
    lp_design.add_argument("--max-power", type=float, default=6.0, help="每条晶格分支最大源端功率 W")
    lp_design.add_argument("--max-scattering", type=float, default=600.0, help="最大波腹散射率 s^-1")
    lp_design.add_argument(
        "--delivery-efficiency",
        type=float,
        default=0.70,
        help="源端到原子处前向功率效率",
    )
    lp_design.add_argument(
        "--retro-ratio",
        type=float,
        default=0.88**4,
        help="原子处回程/前向功率比",
    )
    lp_design.add_argument(
        "--detuning-weight",
        type=float,
        default=0.05,
        help="LP 目标中相对失谐惩罚权重",
    )
    lp_design.add_argument("--json", help="保存完整 LP 结果")
    lp_design.add_argument("--plot", help="保存失谐--功率几何图")
    lp_design.set_defaults(handler=_lp_design_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
