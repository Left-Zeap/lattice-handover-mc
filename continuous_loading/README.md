# `continuous_loading`：正式计算包

这个目录放置当前项目真正使用的物理模型和命令行程序。第一次阅读时
不必从头读完所有代码，先理解模块之间的数据流：

```text
atomic.py
    ↓ 原子质量和 D1/D2 数据
dipole.py
    ↓ 势深与散射率
lattice.py
    ↓ 阱频、反冲、加速势垒
    ├── transport.py → 分段运输温升预算
    ├── l1_transport.py → L1 全程宏观升温、统计损失与二维扫描
    │       ↑ conveyor_geometry.py（可选错腰几何）/ transport_mc.py（可选轨迹级腿）
    ├── handover.py  → 三维轨迹、交接率和末温
    ├── linear_design.py → 失谐—功率分段 LP 可行域
    ├── design_optimization.py → 失谐—功率—束腰稳健平台与相位扫描
    └── scenarios.py → 论文 Rb 基准与 Cs 参数扫描
                         ↓
              l1_handover.py → L1→handover 联合扫描
                         ↓
              l2_transport.py → L2 宏观运输腿与科学区汇总
                         ↓
              full_chain.py → MOT→L1→handover→L2 全链路
                         ↓
                       cli.py

gpu_backend.py / handover_batch.py / transport_batch.py /
    device_loop.py：
    Monte Carlo 的可选 GPU（CuPy/CUDA）后端与扫描批量内核；
    device_loop.py 的设备端时间循环 RawKernel 把批量积分的整段
    时间循环搬进单个 kernel（散射反冲用 cuRAND 逐事件在设备端
    抽取），消除逐步 Python 调度、kernel 启动与散射同步开销，
    编译失败时自动回退逐步融合 kernel。默认 CPU，行为与无 GPU
    环境一致。
```

## 1. 最常用入口

从仓库根目录运行：

```powershell
python -m continuous_loading --help
```

可用子命令：

| 子命令 | 作用 |
|---|---|
| `paper` | 复现论文 Rb-87 运输参数 |
| `cs-scan` | 扫描 Cs D1 红失谐、功率和散射率 |
| `cs-transport` | 把论文同型运输预算应用到 Cs |
| `handover` | 运行 handover 经典轨迹 Monte Carlo |
| `lp-design` | 计算并绘制失谐量—源端功率 LP 可行域 |
| `handover-map` | 在可行域内绘制 Rb/Cs handover 交接率 |
| `handover-angle-scan` | 扫描 0–90° 晶格夹角，绘制 Rb/Cs 交接率和升温 |
| `optimize-design` | 固定时序，稳健优化失谐、功率、束腰并扫描相位 |
| `l1-transport-scan` | 固定加速度和速度，扫描 L1 升温与统计留存率 |
| `l1-handover-scan` | 同一网格上连接 L1 宏观运输与 handover Monte Carlo |
| `full-chain-scan` | MOT→L1→handover→L2 全链路，输出科学区末温、总留存率和密度 |
| `plots` | 生成基础 Rb/Cs 结果图 |

## 2. 文件说明

### `constants.py`

统一保存 \(c\)、\(k_B\)、\(h\)、\(\hbar\)、原子质量单位和标准重力。
其他模块应从这里导入常数，避免出现多套数值。

### `atomic.py`

定义 `DLine`、`AlkaliAtom`、`RB87` 和 `CS133`。主要负责：

- D1/D2 波长、线宽和线强权重；
- 原子质量；
- 从 D1 红失谐计算激光波长。

这里只放程序实际使用的碱金属参数。更广的原子背景资料在
[`../atom/`](../atom/)。

### `dipole.py`

计算 D1+D2 标量偶极势和总散射率：

```python
from continuous_loading.atomic import CS133
from continuous_loading.dipole import scalar_potential_and_scattering

result = scalar_potential_and_scattering(
    CS133,
    laser_wavelength_nm=896.2,
    intensity_w_m2=1.0e7,
)
print(result.depth_uK, result.scattering_rate_s)
```

该模型适合工程估算，不包含指定 \(F,m_F\) 态的矢量/张量光移。

### `lattice.py`

把局域光强转换为可直接使用的晶格量：

- 高斯束峰值强度；
- 不等回程功率下的驻波波腹强度；
- 势深、径向/轴向阱频和反冲温度；
- 达到目标势深所需的前向功率；
- 加速参考系中的有效势垒；
- 孔径截断和谐振近似密度。

常用函数是 `evaluate_lattice()` 和 `power_for_target_depth()`。

### `transport.py`

处理“分段运输预算”，包括：

- 绝热压缩温度变化；
- 散射反冲；
- 随机相位 handover 等效加热；
- 加速度突变加热；
- 三维谐振热分布的束缚比例；
- 各运输阶段的温升汇总。

它是解析工程模型，不逐原子传播轨迹。

### `l1_transport.py`

在 `transport.py` 的宏观方法上补充连续时序积分。扫描变量为 handover
端每分支源端功率和 D1 红失谐；加速度、最大速度与距离固定。模块输出
最终升温/留存率二维网格，以及最优点和较差可行点的完整时间轨迹。
L1 光束由起点直径 `start_beam_diameter_um`、最小半径
`minimum_waist_um` 和距起点焦点位置 `minimum_waist_position_m` 标定：
`z_R,eff=z0/sqrt[(w_start/w0)^2-1]`，随后逐点计算
`w(z)=w0 sqrt[1+((z-z0)/z_R,eff)^2]`；trace 同时提供半径 `waist_um`
和派生直径 `beam_diameter_um`。

公开接口为 `simulate_l1_transport()`（单参数点）和
`analyze_l1_transport_scan()`（二维扫描）。损失模型只使用统计速率方程
和热分布截断，不传播单原子；默认未知损失系数为零。

L1 初态为固定 (N, T)：物理图景是静止 L1 光晶格中束缚足够长时间的
热平衡系综（LGM 装载模块已移除），初始温度默认 20 µK，由
`data/l1_transport_defaults.json` 的 `initial_state` 分组给出；连续
相空间路径经 `initial_state.sample_static_lattice_thermal_ensemble()`
逐点采样束缚热平衡系综（晶格参数取 L1 起点光学量）。该点实际使用
的初态记录在 `L1DesignPoint.initial_temperature_uK /
initial_atom_number`。

计算动力学的明确起点是 L1 运输开始时刻（静止晶格 t=0），MOT 原子数
只作归一化计数基准。`pre_ramp_survival_fraction` 独立描述 MOT 计数
时刻到 L1 初态边界之间尚未显式传播的存活率（默认 1.0，保持旧结果）。

`control_waveform` 可选接收 `TransportControlWaveform`：位置、速度、
加速度和 AOM 频差来自实测表，功率比例、束腰、传输效率比例列按需
覆盖理想跟随。GPU 运输仍在 host 预计算 `(步,13,点)` 系数表并由同一
设备端时间循环读取，不增加逐步 kernel 启动；实测波形与 offset-waist
conveyor 都定义光学剖面，当前要求二选一。

### `conveyor_geometry.py`

可选的 offset-waist 双束 conveyor 几何（默认关闭）：两束腰沿运输轴
错开间距 s，逐点给出波腹阱深、轴/径向阱频、散射率、可见度和临界
加速度剖面。公式见
[`../reports/可选运输模型理论框架.md`](../reports/可选运输模型理论框架.md)
§3。`L1TransportInputs.conveyor_enabled=True` 时
`simulate_l1_transport()` 走逐点几何剖面 + 恒源端功率分支（L2 腿经
`replace` 自动继承）；关闭时使用 L1 标定高斯包络；旧输入仍可显式回退线性端点剖面。

### `transport_mc.py`

可选的 L1/L2 运输腿轨迹级 Monte Carlo（默认关闭）：与 `handover.py`
同型的三维经典轨迹模拟，光场采用底层双束干涉形式（保留 R<1 节点
基底），velocity-Verlet 传播 + 局域散射反冲 + 途中倾斜势垒逃逸
剔除，输出与解析腿完全同型的 `L1TransportTrace`（留存率附 Jeffreys
标准误）。`L1TransportInputs.transport_method="monte_carlo"` 时
`simulate_l1_transport()` 在顶部分支到本模块（L2 腿经 `replace`
自动继承）；粒子数、种子、散射开关和轴向云宽经 `L1TransportInputs`
的 `mc_*` 字段与 handover Monte Carlo 合并调用（默认与
`handover_monte_carlo` 配置组同值，UI/CLI 的每调用设置随之生效），
只新增 `transport_monte_carlo` 的 enabled 与 time_step_us 两个参数。
积分步长在请求值之上再经 `_stable_leg_step_s` 精度守卫自动钳制
（ω_z·dt ≤ 1，与 handover 同一判据，ω_z 取沿程最大轴向调制深度
对应的阱频）：旧默认 0.5 µs 在 L1 典型阱频下 ω_z·dt ≈ 1.2 已接近
失稳区，多微秒步长会让 Verlet 积分发散、原子在 L1 段全部伪逃逸
（扫描表现为大面积"L1 末端无存活原子"无效点）；CPU 单点、GPU
批量与 `light_field` 时序表共用 `_leg_integration_schedule`，实际
步长记录在 `L1DesignPoint.actual_time_step_us`。
无法采样束缚初态的浅阱点返回零留存 trace（温度 NaN）而非抛错；
联合扫描的可行性预检始终用解析腿，逐点失败被隔离为无效点，且
连续相空间扫描会在结束时汇总报告逐点失败原因 Top-3（此前静默
丢弃，无法定位）。handover 前置校验 `_validate_transport_trace`
同时拒绝 L1 末态温度退化（≤0，如仅剩 1 个幸存粒子、速度方差为
零）的点——此类退化点曾在 GPU 批量扫描中于 `HandoverParameters`
校验处整批崩溃。GPU 后端的单点调用会委托 `transport_batch.run_leg_monte_carlo_batch`
（P=1）走设备端时间循环 kernel（conveyor 几何不被批量覆盖时回退
本模块逐步 GPU 路径），消除长步数腿的逐步固定开销。公式见
[`../reports/可选运输模型理论框架.md`](../reports/可选运输模型理论框架.md)。

### `gpu_backend.py`

可选的 GPU（CuPy/CUDA）计算后端检测与适配层：CPU（NumPy）是默认
后端，行为与无 GPU 环境完全一致；`compute_backend="gpu"` 时
handover 与运输腿 Monte Carlo 的粒子数组在内层积分
循环中驻留 GPU。
初态采样始终在 CPU（NumPy RNG，保证初态与 CPU 逐位一致），散射
反冲的随机数按后端创建——CPU 与 GPU 结果只在统计意义上一致。
未安装 CuPy 时请求 GPU 会得到明确错误提示。

GPU 内层循环的两级融合优化：

- **mega-step kernel**（`handover._get_fused_verlet_step_kernel`、
  `transport_mc._get_fused_leg_step_kernel`）：整个 velocity-Verlet
  步（半步速度 → 整步位置 → 新力 → 半步速度）融合为单个
  `cupy.fuse` kernel，以列视图就地更新 (M,3) 粒子数组，每步仅一次
  kernel 启动；随时间变化的标量系数全部在 host 预计算（规避
  CuPy 14 + sm_120 的标量-标量子表达式融合 bug）。
- **固定事件槽散射反冲**（`scattering_kicks_gpu`）：逐粒子 Poisson
  计数 + 按最大计数固定形状批量抽取事件随机量 + 单个融合 kernel
  施加反冲，每步只有一次标量同步；统计上与 CPU 的逐事件实现严格
  等价（自发辐射方向改用均匀球面直接采样，分布不变）。

### `handover_batch.py`

扫描场景的批量 handover Monte Carlo：
`run_handover_monte_carlo_batch(parameters_list, backend="gpu")` 把
多个网格点的 `HandoverParameters` 摊平成单个 (P×N, 3) 粒子数组，
逐点参数 gather 成逐粒子数组。积分主循环默认走
`device_loop.handover_steps` 设备端时间循环 kernel（每个线程一条
轨迹、段间才与 host 交互，散射用 cuRAND 逐事件抽取）；RawKernel
编译失败时自动回退逐步融合的 mega-step kernel（两种实现逐式同构、
统计一致）。同批要求 `particle_count`、`duration_ms`、
`time_step_us`、`crossing_angle_deg`、`retro_power_ratio`、
`trace_points` 一致（扫描天然满足，不一致抛 `ValueError` 回退逐点
调用）；trace 只保留 t=0 与终点两个端点；P×N 超过 2×10⁶ 时自动
分块保护显存。`analyze_l1_handover_scan` 与
`analyze_species_handover_map` 在 `compute_backend="gpu"` 时自动走
批量路径（先逐点算 L1 腿/可行性，再一次批量完成全部 handover MC）。
`backend="cpu"` 时逐点调用 `simulate_leg_monte_carlo`，结果与直接
调用逐位一致。

### `transport_batch.py`

扫描场景的批量运输腿 Monte Carlo：
`run_leg_monte_carlo_batch(tasks, backend="gpu")` 把多个网格点的
L1/L2 运输腿粒子摊平后用批量 mega-step kernel 同时推进；与批量
handover 的差异是双束光学系数随 z_L(t) 逐步且逐点变化，因此全部
步的 13 个系数在 host 预计算成 `(步, 13, 点)` 表上传，每步取一行
按 `point_index` gather 成逐粒子数组。要求所有点的
`L1TransportInputs` 除三个初态字段（`initial_temperature_uK`、
`initial_atom_number`、`mot_atom_number`，只影响 host 端初态采样
与 trace 组装）外全等（扫描天然满足；全链路 L2 腿靠该白名单把各点
handover 捕获样本作为初态），conveyor 几何未覆盖
（抛 `ValueError` 回退逐点）；逃逸剔除、快照统计、Jeffreys 标准误
与逐点 `simulate_leg_monte_carlo` 同口径，采样失败的点返回零留存
trace。`analyze_l1_handover_scan` 在 GPU + `transport_method=
"monte_carlo"` 时自动走批量腿，`analyze_full_chain_scan` 的 L2 腿
在同条件下同样一次批量完成（逐点初温/原子数经白名单支持）。批量
积分主循环默认走 `device_loop.leg_steps` 设备端时间循环 kernel
（13 个系数逐步在设备上从系数表读取，快照/逃逸剔除在段间由 host
处理），编译失败时回退逐步融合 kernel。两个批量函数都接受 `progress`
回调，分块与积分过程周期性报告（消息含 `n/total` 供 UI 解析），
GPU 扫描全程有进度反馈。

### `handover.py`

在两个带夹角的三维高斯驻波势中传播经典轨迹。主要输出：

- `transfer_efficiency`：末态被 Lattice-2 捕获的比例；
- `final_temperature_uK`：捕获子样本的总激发能等效温度；
- `handover_heating_uK`：同一捕获子样本的净升温；
- `mean_scattering_events`：平均散射次数；
- `effective_barrier_uK`：考虑后续加速度的有效势垒。

程序使用 velocity-Verlet 积分。改变力模型或时间步长后，必须做
收敛检查。

若要按程序执行顺序理解初态采样、线性功率斜坡、三维势和力、散射
反冲、末态捕获判据及温度统计，阅读
[`../reports/handover交接的理论修正与程序逻辑.md`](../reports/handover交接的理论修正与程序逻辑.md)。

### `scenarios.py`

把底层函数组合成可运行的物理场景：

- 论文 Rb-87 参数复现；
- Extended Data Figure 2 的扫描预设；
- Cs 候选工作点；
- Cs 失谐扫描；
- Cs 双段运输预测。

论文没有公开的量会以可配置工程假设出现，不应误写成论文实测值。

### `collisions.py`

提供相同玻色子的 \(s\) 波散射截面、平均相对速度和二体碰撞密度
数量级。它主要服务于原子库密度与装载率估算。

### `handover_formula_validation.py`

独立复算原始 handover PDF 的式 (1)–(11)，并检查：

- 闭式公式与直接回代的残差；
- 单格点条件 \(k\theta\sigma_y\)；
- 突然切换条件 \(\omega\tau\)；
- 完整周期势给出的有界角度能量；
- 原始错误温升对功率结果的级联影响。

它用于理论审查，不代替 `handover.py` 的实际轨迹交接率。

### `linear_design.py`

把 D1 红失谐和每条晶格分支的源端功率作为二维设计变量，并处理五类
限制：

- 最小静态阱深；
- 给定 handover 时间内的最少轴向振荡周期数；
- 后续加速度下的目标热平衡束缚比例；
- 最大波腹散射率；
- 最大源端功率。

偶极势和散射率对失谐并非严格线性，因此模块把失谐区间分段，在每段
内构造保守仿射边界，再通过二维半平面相交求 LP 可行多边形。推荐点
会代回 `evaluate_lattice()` 和完整加速势垒公式核验。

默认 handover 时间为 0.2、0.3、0.4 和 1.0 ms。默认 80 个轴向周期
是用于比较斜坡快慢的工程判据，不等同于轨迹 Monte Carlo 的交接率。

### `handover_map.py`

把 L1 运输末态与 `handover.py` 的轨迹 Monte Carlo 连接起来：

1. 在 100–800 GHz、0–1.5 W 的 25×25 网格上计算晶格参数；
2. 每点先运行 L1 宏观运输，得到末温、末态原子数、阱深和散射率；
3. 把 L1 末态逐点传入 1000 µs handover；
4. 用交接率和捕获后升温着色，并在 2×2 图片中比较
   Cs-133 与 Rb-87。

当前假设 L1、L2 使用相同失谐和相同满功率。两种体系均从 JSON 中
可调的 30 µK MOT/L1 初温开始，handover 初温由各点 L1 运输结果决定。

所有全局默认值由
[`../data/handover_map_defaults.json`](../data/handover_map_defaults.json)
读取。输出图叠加最小阱深、handover 轴向周期、加速束缚比例、最大
散射率和最大源端功率五条边界。最大散射率仍作为工程参考线显示，
但不在 Monte Carlo 前删除网格点，因为随机散射反冲已在轨迹中显式
计算；功率上限由扫描范围直接给出。

各可行网格点的 Monte Carlo 由 `ProcessPoolExecutor` 分配到多个 CPU
进程；单个网格点内部仍使用原来的 NumPy 粒子向量化和随机数生成器。
`parallel.backend=serial` 或 `worker_count=1` 可切回原串行路径。

`data/handover_map_defaults.json` 的 `preconditions` 可以分别开关
最小阱深、热束缚比例和最少轴向周期；命令行可使用
`--no-minimum-depth`、`--no-thermal-bound-fraction` 和
`--no-minimum-axial-cycles` 临时关闭。当前 JSON 默认全部关闭；
`l1_transport_defaults.json/handover_preconditions` 中的三项 L1
前置验证也默认关闭。严格零功率没有束缚初态，不运行轨迹。

### `handover_angle_scan.py`

复用 `run_handover_monte_carlo()`，在固定阱深下只改变 L1/L2 夹角：

```powershell
python -m continuous_loading handover-angle-scan
```

默认扫描 0–90°、步长 1°，Rb 使用 300 GHz，Cs 使用 600 GHz，二者均
固定为 500 µK 阱深。输出交接率和 handover 升温随角度变化的双面板
折线图。升温面板中实线是最终被 L2 捕获子样本的前后升温，虚线是
不做捕获筛选的全部原子平均总激发能升温。扫描范围、粒子数、步长和
CPU 进程数集中保存在
`data/handover_map_defaults.json/angle_scan`。

### `l2_transport.py`

把 handover 捕获样本（末温、原子数）作为初态，复用
`simulate_l1_transport()` 的单段积分器计算 L2 段（0.17 m、21 ms、
束腰由 L1 计算的 handover 半径→150 µm、加速度 4000 m/s²）的宏观升温和统计留存，并汇总
科学区原子库的峰值密度和每格点原子数。L2 源端功率与 handover 端
保持相同，局部光强和阱深随束腰按高斯标度变化。固定参数集中在
`data/l1_transport_defaults.json` 的 `l2_transport` 分组。

注意口径假设：handover 的 `final_temperature_uK` 是捕获样本总激发能
等效温度，把它当作 L2 腿热平衡初温等价于假设 21 ms 内碰撞再热化，
引用结果时应显式说明。

### `full_chain.py`、`full_chain_plots.py`

把 L1 运输、handover Monte Carlo 和 L2 腿串成
L1→handover→L2→科学区链路（MOT 为计数基准，L1 初态为静止晶格
热平衡系综）：

1. `analyze_full_chain_scan()` 复用 `analyze_l1_handover_scan()` 的
   Monte Carlo 网格结果，逐点补算毫秒级的解析 L2 腿，不重复轨迹
   模拟（GPU + 轨迹级运输 MC 时 L2 腿改由 `transport_batch` 一次
   批量完成，批量未覆盖时回退逐点）；
2. 在科学区末态的总升温和总损失上重新选择最优/较差工作点；
3. 对选定点生成 L1、handover、L2 拼接轨迹；
4. `full_chain_plots.py` 绘制总升温、相对 MOT 留存热图和三相
   拼接轨迹。

公开接口为 `simulate_full_chain_point()`（单参数点）和
`analyze_full_chain_scan()`（二维扫描）。理论、口径假设和数值结果见
[`../reports/L1-L2运输与全链路计算框架.md`](../reports/L1-L2运输与全链路计算框架.md)。

`FullChainInputs.phase_space_continuity=True` 启用连续相空间路径：
静止晶格热平衡初态粒子集合依次传给 L1、handover、L2，不在阶段
边界重新生成热平衡系综；粒子数组只在进程内瞬态存在，输出仍为原
数据类。该模式要求运输腿为 Monte Carlo，且禁止梯形加速度阶跃——
无实测控制波形时 L1/L2 强制 minimum_jerk，梯形直接在库层抛错；
`TransportControlWaveform`/`HandoverControlWaveform` 可导入实测
波形。二维扫描同样支持该开关：CPU 按 serial/process 逐点运行，
GPU 按 L1/handover/L2 三段固定形状批量推进，粒子集合只在阶段
边界按网格索引传递。扫描进度口径：CPU 路径按"已完成点数/可行
任务数"逐点计数；GPU 路径的初态采样、L1、handover、L2 四个阶段
的 "done/total" 子计数统一折算为全局网格完成点数（阶段加权），
进度条不再在阶段切换时回退或提前到 100%；GPU 批量不可用回退
CPU 逐点时同样逐点汇报。

连续接口的动态温度统一为去除质心速度后的三维动能温度。L1 首点、
handover 首点以及 L2 首点使用实际跨阶段的同一集合口径；
不再把各向异性温度的几何平均、最终捕获子样本的回溯温度和当前全体
温度混画成边界尖刺。阶段间固定粒子数适配采用低方差系统重采样。

理想运输腿支持 `kinematic_profile="trapezoid"`（兼容旧接口）和
`"minimum_jerk"`。后者用五次 S 曲线令启停端速度、加速度与 jerk
连续，适合从静止 L1 晶格热平衡初态（端点速度与加速度为零）启动
L1；CPU 单点与 GPU 批量路径使用逐式同构的运动学数组。

handover 的 `control_waveform` 接受 `HandoverControlWaveform`；批量 GPU
把 L1/L2 深度分数和公共相位表传入同一设备端时间循环，未退化为逐步
host 调度。
handover 的 velocity-Verlet 步长还会按最快轴向阱频自动施加
`omega*dt <= 1` 的稳定性上限；默认 0.25 µs 通常不变，只有用户输入
过大步长时才自动细分，CPU/GPU 共用相同公式。

### `cloud_sigma_scan.py`

固定工作点（失谐、源端功率）上的原子云轴向宽度 σ 一维扫描，用于
评估手写 handover 交接对初始云宽的敏感性。每个 σ 点用
`dataclasses.replace` 把同一 σ 同步写入
`L1HandoverInputs.cloud_axial_sigma_mm`（(N,T) 约化接口的 handover
自采样云宽）与 `L1TransportInputs.mc_cloud_axial_sigma_mm`（连续
相空间接口的 L1 起点系综采样宽度）后调用
`simulate_full_chain_point(..., trace_points=2)`，输出 handover 末温、
链末总温、交接率与相对 MOT 总留存随 σ/w₀ 的序列；两种接口模式与
CPU/GPU 后端都适用（逐点等价于重复单点调用）。公开接口为
`analyze_cloud_sigma_scan()`；逐点失败隔离为无效点并在结束时汇总
Top-3 原因，口径与二维扫描一致。

### `device_loop.py`

批量 Monte Carlo 的设备端时间循环 kernel（RawKernel + cuRAND）：
每个线程负责一条轨迹，把整段 velocity-Verlet + 局域 Poisson 散射
反冲循环放在设备上执行，只在段边界与 host 交互（进度、快照、
逃逸剔除），消除了逐步 Python 调度、kernel 启动与散射标量同步的
固定开销。长步数运输腿（~10⁵ 步）实测提速约 20 倍，小粒子数
批量约 40 倍，大粒子数 handover 批量约 4 倍（本机 RTX 5070）。
动力学与逐步融合 kernel 逐式同构，散射分布一致（逐事件 Poisson +
均匀球面自发辐射），同 seed 结果确定性；与 CPU/逐步 GPU 路径仅
统计一致。``get_handover_loop_kernels()`` / ``get_leg_loop_kernels()``
编译失败返回 ``None``，两个批量模块据此自动回退逐步融合 kernel。
逃逸剔除压缩粒子数组时，cuRAND 状态按 64 B 固定步长寻址随动压缩。

### `plots.py`

生成两张基础图：

- Rb-87 分段温升路径；
- Cs-133 功率—散射折中图。

### `cli.py`、`__main__.py`

`cli.py` 定义命令行参数和输出格式；`__main__.py` 使
`python -m continuous_loading` 可以直接运行。

桌面图形界面在 [`../ui/`](../ui/)（`python -m ui`），它只是本包的
调用方，物理模型仍以这里为准。

### `__init__.py`

导出最常用的类和函数，便于其他 Python 程序调用。

## 3. 常用示例

### 在 Python 中计算一个 Cs 晶格

```python
from continuous_loading import CS133, evaluate_lattice

wavelength_nm = CS133.laser_wavelength_red_of_d1_nm(600.0)
metrics = evaluate_lattice(
    CS133,
    wavelength_nm,
    forward_power_w=2.0,
    waist_um=250.0,
    retro_power_ratio=0.88**4,
)

print(metrics.depth_uK)
print(metrics.scattering_rate_s)
```

### 扫描 handover 角度

```powershell
python -m continuous_loading handover `
  --scan-parameter angle `
  --scan-values 0,2,4,6,8 `
  --particles 1000 `
  --csv output\handover_angle.csv `
  --plot output\handover_angle.png
```

### 绘制失谐—功率 LP 几何图

```powershell
python -m continuous_loading lp-design `
  --atom Cs-133 `
  --handover-times 0.2,0.3,0.4,1.0 `
  --max-power 6 `
  --max-scattering 600 `
  --plot output\detuning_power_lp.png `
  --json output\detuning_power_lp.json
```

### 绘制 Rb/Cs handover 效率热力图

```powershell
python -m continuous_loading handover-map `
  --detuning-points 19 `
  --power-points 21 `
  --particles 128 `
  --time-step 0.25 `
  --workers 8 `
  --plot output\handover_efficiency_map.png
```

### 寻找可容忍 10% 调节的硬件工作平台

```powershell
python -m continuous_loading optimize-design
```

默认只对稳健性最好的少量候选运行 Monte Carlo，并在推荐点输出 AOM
相位—预计保留原子数曲线。集中参数位于
`data/design_optimization_defaults.json`。

### 扫描 L1 全程升温和留存率

```powershell
python -m continuous_loading l1-transport-scan `
  --atom Rb-87 `
  --acceleration 4000 `
  --velocity 8.13
```

集中参数位于 `data/l1_transport_defaults.json`；命令行可临时覆盖扫描
范围、时序、初温、光路效率和四类损失系数。按 `--atom Rb-87` 或
`--atom Cs-133` 分别生成 `l1_transport_scan_rb87.*` 和
`l1_transport_scan_cs133.*`。
参考点数值仍保存在 JSON 和终端摘要中，但热力图不再绘制范围外参考点；
两种原子的功率轴都严格使用配置的 0–1.5 W 扫描范围。

默认扫描范围统一为 100–800 GHz 和 0–1.5 W；MOT 出射初温从
`data/l1_transport_defaults.json` 的 `initial_state.temperature_uK`
读取，默认 20 µK。

`handover_preconditions` 可分别开关最小阱深、L1 起点最大功率和临界
加速度条件。CLI 对应支持 `--minimum-depth/--no-minimum-depth`、
`--maximum-start-power/--no-maximum-start-power` 和
`--critical-acceleration/--no-critical-acceleration`。

### 连接 L1 运输与 handover Monte Carlo

```powershell
python -m continuous_loading l1-handover-scan --atom Rb-87
python -m continuous_loading l1-handover-scan --atom Cs-133
```

该命令在与 transport scan 完全相同的失谐—功率网格上，先调用
`simulate_l1_transport()`，再把每点的末温和末态原子数传给现有
`run_handover_monte_carlo()`。当前关闭前置验证，默认每个正功率点
使用 1000 条轨迹。
输出四联图包含总升温、MOT→L2 装载率，以及最优/对比参数点的完整
L1+handover 温度和留存率时间曲线。模拟、扫描和绘图分别位于
`l1_handover.py` 与 `l1_handover_plots.py`。

可选的自适应粒子加密（两阶段扫描）：把
`data/l1_transport_defaults.json` 的
`handover_monte_carlo.adaptive_refinement.enabled` 置为 `true` 后，
第一遍仍按基础粒子数扫描，随后交接率 Jeffreys 标准误超过
`target_standard_error` 的点在同一 L1 末态上以
`particle_count×(SE/目标)²` 条轨迹（同 seed、上限
`max_particle_count`）自动复算一遍并更新网格——平滑区域不浪费
粒子，加密集中在交接边界附近。`l1-handover-scan`、
`full-chain-scan` 和 `handover-map` 的 CPU/GPU 路径均支持；
`L1HandoverPoint.handover_particle_count` 记录每点实际轨迹数，
`L1HandoverScanResult.refined_points` 记录被加密的点数。

### 运行到科学区的全链路扫描

```powershell
python -m continuous_loading full-chain-scan --atom Rb-87
python -m continuous_loading full-chain-scan --atom Cs-133
```

在与 `l1-handover-scan` 完全相同的网格和 Monte Carlo 配置上追加 L2
宏观运输腿，输出科学区末温、相对 MOT 总留存率、原子库峰值密度热图
和三相拼接时间曲线，按原子生成 `full_chain_scan_rb87.*` 和
`full_chain_scan_cs133.*`。L2 固定参数在
`data/l1_transport_defaults.json` 的 `l2_transport` 分组。

## 4. 修改模块时看哪些测试

| 修改内容 | 优先运行 |
|---|---|
| `atomic.py`、`dipole.py`、`lattice.py` | `tests/test_atomic_and_lattice.py` |
| `transport.py`、`scenarios.py` | `tests/test_transport_and_scenarios.py` |
| `handover.py` | `tests/test_handover.py` |
| `handover_formula_validation.py` | `tests/test_handover_formula_validation.py` |
| `linear_design.py` | `tests/test_linear_design.py` |
| `handover_map.py` | `tests/test_handover_map.py` |
| `handover_angle_scan.py` | `tests/test_handover_angle_scan.py` |
| `design_optimization.py` | `tests/test_design_optimization.py` |
| `l1_transport.py` | `tests/test_l1_transport.py` |
| `conveyor_geometry.py` | `tests/test_conveyor_geometry.py` |
| `transport_mc.py` | `tests/test_transport_mc.py` |
| `l1_handover.py`、`l1_handover_plots.py` | `tests/test_l1_handover.py` |
| `l2_transport.py` | `tests/test_l2_transport.py` |
| `full_chain.py`、`full_chain_plots.py` | `tests/test_full_chain.py` |
| `cloud_sigma_scan.py` | `tests/test_cloud_sigma_scan.py` |
| `gpu_backend.py`、`handover.py`/`transport_mc.py` 的后端分支、`handover_batch.py`、`transport_batch.py`、`device_loop.py` | `tests/test_gpu_backend.py` |
| `initial_state.py` | `tests/test_initial_state.py` |
| `light_field.py` | `tests/test_light_field.py` |
| `chain_mc.py` | `tests/test_chain_mc.py` |

完整测试：

```powershell
python -m pytest -q
```

## 5. 编码约定

- 内部计算优先使用 SI 单位；
- 对外 dataclass 和命令行参数用变量名后缀标明单位；
- 势深用正数表示，红失谐势能本身为负；
- 普通频率用 Hz，角频率用 rad/s，不混用；
- D1 红失谐参数取正数，内部实际 \(\Delta=\omega_L-\omega_0<0\)；
- Monte Carlo 随机计算必须保留可配置 `seed`。

新增物理模型时，先写极限情况测试，再接入 CLI。
