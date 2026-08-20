"""设备端时间循环 Monte Carlo kernel（RawKernel + cuRAND）。

把批量 Monte Carlo 的内层积分循环整体搬进 CUDA kernel：每个线程
负责一条轨迹，在设备上逐步执行 velocity-Verlet 与局域 Poisson 散射
反冲，段与段之间才与 host 交互（进度报告、快照统计、逃逸剔除）。
与逐步融合 kernel 路径相比，消除了每步的 Python 调度、kernel 启动
和散射标量同步开销——这是长步数运输腿（~1e5 步）的主要墙钟成本。

动力学部分与 ``handover_batch._get_fused_batch_step_kernel`` /
``transport_batch._get_fused_batch_leg_step_kernel`` 逐式同构；散射
反冲改用 cuRAND 在设备端逐事件抽取（Poisson 计数 + 吸收轴 + 前向/
回程符号 + 均匀球面自发辐射方向），分布与既有 GPU 路径相同。同
seed 结果确定性；cuRAND 序列与 NumPy/CuPy RNG 不同，跨实现仅统计
一致（与既有"CPU/GPU 仅统计一致"口径相同）。

编译失败（如无 NVRTC 或旧 CUDA 头文件）时 ``get_*_loop_kernels``
返回 ``None``，调用方回退逐步融合 kernel 路径，行为不变。
"""

from __future__ import annotations

# cuRAND XORWOW 状态为 12 个 uint32（48 B）；按 16 个 uint32（64 B）
# 分配留足对齐余量。
_STATE_UINT32_PER_THREAD = 16

_BLOCK_THREADS = 256

# NVRTC 编译选项：cupy 捆绑的 libcu++ 头文件要求 C++17。
_COMPILE_OPTIONS = ("--std=c++17",)

_KERNEL_SOURCE = r"""
#include <curand_kernel.h>
#define DEVICE_LOOP_TWO_PI 6.283185307179586476
#define DEVICE_LOOP_HBAR 1.0545718176461565e-34

// cuRAND 状态按固定步长 16 个 uint32（64 B）逐线程存储：比
// sizeof(curandState_t) 大且 64 B 对齐，逃逸剔除按行压缩时字节
// 不错位（kernel 内以显式步长寻址，不依赖 sizeof 打包）。
#define DEVICE_LOOP_STATE_STRIDE 16

extern "C" {

__global__ void device_loop_init_rng(
    unsigned int* states, unsigned long long seed, long long total
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        curandState_t* st =
            (curandState_t*)(states + i * DEVICE_LOOP_STATE_STRIDE);
        curand_init(seed, (unsigned long long)i, 0ULL, st);
    }
}

// 批量 handover 的设备端时间循环：与 _get_fused_batch_step_kernel
// 逐式同构的 velocity-Verlet mega-step + 逐事件散射反冲。
__global__ void handover_steps(
    double* positions, double* velocities, double* forces,
    const double* wave_number, const double* depth1_j,
    const double* depth2_j,
    const double* neg2_w1, const double* four_w1,
    const double* neg2_w2, const double* four_w2,
    const double* velocity1, const double* velocity2,
    const double* offset2_1, const double* mass,
    const double* gravity_force_y,
    const double* phase1, const double* phase2,
    const double* scattering1, const double* scattering2,
    const double* fraction1_steps, const double* fraction2_steps,
    const double* phase_control_steps,
    long long* scatter_counts, unsigned int* rng_states,
    double e2_0, double e2_1, double e2_2,
    double time_step, double duration, double forward_probability,
    long long step_begin, long long step_end, long long total,
    int include_scattering
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long i3 = 3 * i;
    double px = positions[i3];
    double py = positions[i3 + 1];
    double pz = positions[i3 + 2];
    double vx = velocities[i3];
    double vy = velocities[i3 + 1];
    double vz = velocities[i3 + 2];
    double fx = forces[i3];
    double fy = forces[i3 + 1];
    double fz = forces[i3 + 2];
    const double m = mass[i];
    const double k = wave_number[i];
    const double hdm = 0.5 * time_step / m;
    const double recoil = DEVICE_LOOP_HBAR * k / m;
    const double w1n = neg2_w1[i], w1f = four_w1[i];
    const double w2n = neg2_w2[i], w2f = four_w2[i];
    const double v1 = velocity1[i], v2 = velocity2[i];
    const double off = offset2_1[i];
    const double d1j = depth1_j[i], d2j = depth2_j[i];
    const double ph1 = phase1[i], ph2 = phase2[i];
    const double sc1 = scattering1[i], sc2 = scattering2[i];
    const double ke2_0 = k * e2_0, ke2_1 = k * e2_1, ke2_2 = k * e2_2;
    long long counts = scatter_counts[i];
    curandState_t state =
        *((curandState_t*)(rng_states + i * DEVICE_LOOP_STATE_STRIDE));
    for (long long s = step_begin; s < step_end; s++) {
        const double t = (double)(s + 1) * time_step;
        const double fraction1 = fraction1_steps[s + 1];
        const double fraction2 = fraction2_steps[s + 1];
        const double phase_control = phase_control_steps[s + 1];
        // 半步速度 + 整步位置。
        const double nvx = vx + fx * hdm;
        const double nvy = vy + fy * hdm;
        const double nvz = vz + fz * hdm;
        px += nvx * time_step;
        py += nvy * time_step;
        pz += nvz * time_step;
        // Lattice-1：轴 (0,0,1)、束偏移为零。
        const double env1 = exp((px * px + py * py) * w1n);
        const double p1 = k * (pz - v1 * t) + ph1;
        double c1 = cos(p1);
        c1 *= c1;
        const double shape1 = env1 * c1;
        const double s1 = sin(2.0 * p1);
        const double dep1 = d1j * fraction1;
        // Lattice-2：轴 e2、束偏移 (0, off, 0)。
        const double d0 = px, d1 = py - off, d2 = pz;
        const double ax = d0 * e2_0 + d1 * e2_1 + d2 * e2_2;
        const double t0 = d0 - ax * e2_0;
        const double t1 = d1 - ax * e2_1;
        const double t2 = d2 - ax * e2_2;
        const double env2 = exp((t0 * t0 + t1 * t1 + t2 * t2) * w2n);
        const double p2 = k * (ax - v2 * t) + ph2 + phase_control;
        double c2 = cos(p2);
        c2 *= c2;
        const double shape2 = env2 * c2;
        const double s2 = sin(2.0 * p2);
        const double dep2 = d2j * fraction2;
        const double g0 = -(dep1 * env1 * (c1 * px * w1f))
            - (dep2 * env2 * (ke2_0 * s2 + c2 * t0 * w2f));
        const double g1 = -(dep1 * env1 * (c1 * py * w1f))
            - (dep2 * env2 * (ke2_1 * s2 + c2 * t1 * w2f))
            + gravity_force_y[i];
        const double g2 = -(dep1 * env1 * (k * s1))
            - (dep2 * env2 * (ke2_2 * s2 + c2 * t2 * w2f));
        vx = nvx + g0 * hdm;
        vy = nvy + g1 * hdm;
        vz = nvz + g2 * hdm;
        fx = g0;
        fy = g1;
        fz = g2;
        if (include_scattering) {
            const double lam1 = sc1 * fraction1 * shape1 * time_step;
            const double lam2 = sc2 * fraction2 * shape2 * time_step;
            const double lam = lam1 + lam2;
            const int event_count = curand_poisson(&state, lam);
            counts += event_count;
            if (event_count > 0) {
                const double ratio1 = lam1 / lam;
                for (int j = 0; j < event_count; j++) {
                    const bool choose1 =
                        curand_uniform_double(&state) < ratio1;
                    const double sign =
                        curand_uniform_double(&state) < forward_probability
                            ? 1.0
                            : -1.0;
                    const double z =
                        2.0 * curand_uniform_double(&state) - 1.0;
                    const double phi =
                        DEVICE_LOOP_TWO_PI * curand_uniform_double(&state);
                    const double radius = sqrt(fmax(0.0, 1.0 - z * z));
                    const double dir0 = radius * cos(phi);
                    const double dir1 = radius * sin(phi);
                    const double a0 = (choose1 ? 0.0 : e2_0) * sign;
                    const double a1 = (choose1 ? 0.0 : e2_1) * sign;
                    const double a2 = (choose1 ? 1.0 : e2_2) * sign;
                    vx += recoil * (a0 - dir0);
                    vy += recoil * (a1 - dir1);
                    vz += recoil * (a2 - z);
                }
            }
        }
    }
    positions[i3] = px;
    positions[i3 + 1] = py;
    positions[i3 + 2] = pz;
    velocities[i3] = vx;
    velocities[i3 + 1] = vy;
    velocities[i3 + 2] = vz;
    forces[i3] = fx;
    forces[i3 + 1] = fy;
    forces[i3 + 2] = fz;
    scatter_counts[i] = counts;
    *((curandState_t*)(rng_states + i * DEVICE_LOOP_STATE_STRIDE)) = state;
}

// 批量运输腿的设备端时间循环：与 _get_fused_batch_leg_step_kernel
// 逐式同构；13 个双束系数逐步从 (步, 13, 点) 表读取。
__global__ void leg_steps(
    double* positions, double* velocities, double* forces,
    double* potential,
    const double* table, const double* z_lattice,
    const int* point_index,
    const double* scattering_coeff, const double* recoil,
    long long* scatter_counts, unsigned int* rng_states,
    double time_step, double half_dt_over_mass, double gravity_force_y,
    long long point_count, long long step_begin, long long step_end,
    long long total, int include_scattering
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long i3 = 3 * i;
    double px = positions[i3];
    double py = positions[i3 + 1];
    double pz = positions[i3 + 2];
    double vx = velocities[i3];
    double vy = velocities[i3 + 1];
    double vz = velocities[i3 + 2];
    double fx = forces[i3];
    double fy = forces[i3 + 1];
    double fz = forces[i3 + 2];
    const int point = point_index[i];
    const double scoeff = scattering_coeff[i];
    const double rec = recoil[i];
    const long long points = point_count;
    long long counts = scatter_counts[i];
    curandState_t state =
        *((curandState_t*)(rng_states + i * DEVICE_LOOP_STATE_STRIDE));
    for (long long s = step_begin; s < step_end; s++) {
        // 半步速度 + 整步位置。
        const double nvx = vx + fx * half_dt_over_mass;
        const double nvy = vy + fy * half_dt_over_mass;
        const double nvz = vz + fz * half_dt_over_mass;
        px += nvx * time_step;
        py += nvy * time_step;
        pz += nvz * time_step;
        // 本步系数行 (13, P) 中该点的列。
        const double* row = table + ((size_t)s * 13) * points + point;
        const double ec1 = row[0 * points];
        const double ec2 = row[1 * points];
        const double ecc = row[2 * points];
        const double i1 = row[3 * points];
        const double i2 = row[4 * points];
        const double pc1 = row[5 * points];
        const double pc2 = row[6 * points];
        const double pcc = row[7 * points];
        const double axc = row[8 * points];
        const double rc1 = row[9 * points];
        const double rc2 = row[10 * points];
        const double rcc = row[11 * points];
        const double twok = row[12 * points];
        const double zeta = pz - z_lattice[s];
        const double rho2 = px * px + py * py;
        const double env1 = exp(rho2 * ec1);
        const double env2 = exp(rho2 * ec2);
        const double envc = exp(rho2 * ecc);
        const double theta = twok * zeta;
        const double cosine = cos(theta);
        const double pot =
            -(pc1 * env1 + pc2 * env2 + pcc * envc * cosine);
        const double g2 = -(axc * envc * sin(theta));
        const double radial =
            -(rc1 * env1 + rc2 * env2 + rcc * envc * cosine);
        const double g0 = radial * px;
        const double g1 = radial * py + gravity_force_y;
        vx = nvx + g0 * half_dt_over_mass;
        vy = nvy + g1 * half_dt_over_mass;
        vz = nvz + g2 * half_dt_over_mass;
        fx = g0;
        fy = g1;
        fz = g2;
        potential[i] = pot;
        if (include_scattering) {
            const double local_forward = i1 * env1;
            const double local_incoherent = local_forward + i2 * env2;
            const double lam = scoeff * local_incoherent * time_step;
            const int event_count = curand_poisson(&state, lam);
            counts += event_count;
            if (event_count > 0) {
                const double forward = local_forward / local_incoherent;
                for (int j = 0; j < event_count; j++) {
                    const double sign =
                        curand_uniform_double(&state) < forward
                            ? 1.0
                            : -1.0;
                    const double z =
                        2.0 * curand_uniform_double(&state) - 1.0;
                    const double phi =
                        DEVICE_LOOP_TWO_PI * curand_uniform_double(&state);
                    const double radius = sqrt(fmax(0.0, 1.0 - z * z));
                    const double dir0 = radius * cos(phi);
                    const double dir1 = radius * sin(phi);
                    // 吸收轴恒为 ẑ（前向/回程由 sign 选择）。
                    vx += rec * (0.0 - dir0);
                    vy += rec * (0.0 - dir1);
                    vz += rec * (sign - z);
                }
            }
        }
    }
    positions[i3] = px;
    positions[i3 + 1] = py;
    positions[i3 + 2] = pz;
    velocities[i3] = vx;
    velocities[i3 + 1] = vy;
    velocities[i3 + 2] = vz;
    forces[i3] = fx;
    forces[i3 + 1] = fy;
    forces[i3 + 2] = fz;
    scatter_counts[i] = counts;
    *((curandState_t*)(rng_states + i * DEVICE_LOOP_STATE_STRIDE)) = state;
}

}  // extern "C"
"""

_HANDOVER_KERNELS: tuple | None = None
_LEG_KERNELS: tuple | None = None


def _compile_kernel(function_name: str):
    """编译并返回指定 kernel；任何编译失败返回 ``None``（调用方回退）。"""
    try:
        import cupy as cp

        return cp.RawKernel(
            _KERNEL_SOURCE, function_name, options=_COMPILE_OPTIONS
        )
    except Exception:  # noqa: BLE001 - NVRTC/头文件/驱动问题统一回退
        return None


def get_handover_loop_kernels() -> tuple | None:
    """返回 ``(init_rng, handover_steps)``；编译失败返回 ``None``。"""
    global _HANDOVER_KERNELS
    if _HANDOVER_KERNELS is None:
        init = _compile_kernel("device_loop_init_rng")
        steps = _compile_kernel("handover_steps")
        if init is None or steps is None:
            _HANDOVER_KERNELS = None
            return None
        _HANDOVER_KERNELS = (init, steps)
    return _HANDOVER_KERNELS


def get_leg_loop_kernels() -> tuple | None:
    """返回 ``(init_rng, leg_steps)``；编译失败返回 ``None``。"""
    global _LEG_KERNELS
    if _LEG_KERNELS is None:
        init = _compile_kernel("device_loop_init_rng")
        steps = _compile_kernel("leg_steps")
        if init is None or steps is None:
            _LEG_KERNELS = None
            return None
        _LEG_KERNELS = (init, steps)
    return _LEG_KERNELS


def allocate_rng_states(xp, init_kernel, total_particles: int, seed: int):
    """分配并初始化 cuRAND 状态数组（(total, 16) uint32，确定性）。"""
    states = xp.empty(
        (total_particles, _STATE_UINT32_PER_THREAD), dtype=xp.uint32
    )
    grid = ((total_particles + _BLOCK_THREADS - 1) // _BLOCK_THREADS,)
    init_kernel(
        grid,
        (_BLOCK_THREADS,),
        (states, xp.uint64(seed % 2**64), xp.int64(total_particles)),
    )
    return states


def launch_config(total_particles: int) -> tuple[tuple, tuple]:
    """一维启动配置 ``(grid, block)``（块大小 256 线程）。"""
    grid = (total_particles + _BLOCK_THREADS - 1) // _BLOCK_THREADS
    return (grid,), (_BLOCK_THREADS,)
