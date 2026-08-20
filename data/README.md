# `data`：机器可读输入数据

这个目录放置程序和报告共同使用的结构化输入。这里的数据是“计算的
起点”，不是程序运行后生成的结果。

## 当前文件

### `paper_rb87_transport.json`

它把论文中的 Rb-87 连续装载参数整理成 JSON，内容包括：

- MOT 装载、压缩和灰摩尔冷却；
- 两套移动晶格的束腰、距离、时间和加速度；
- handover 工作点与 Extended Figure 2 扫描范围；
- 科学区原子数、温度、密度和替换周期；
- 论文没有公开的参数及其解释边界。

每个主要分组中的 `source` 字段用于记录参数出处。
`model_*` 字段表示程序为了可计算而加入的假设，例如默认交叉角和
轴向云尺寸，不是论文明确给出的测量值。

### `handover_map_defaults.json`

它集中保存失谐—功率 handover 热力图的可调全局默认值：

- 横纵轴扫描范围和网格点数；
- 每个可行点的 Monte Carlo 粒子数、时间步长、随机种子和散射开关；
- CPU 并行后端和工作进程数；
- handover 时间；
- 束腰、阱深、束缚比例、加速度、轴向周期、散射率、光路效率；
- Cs-133、Rb-87 的初始温度和交接几何；
- 图片尺寸、颜色范围、色图和限制边界采样密度；
- handover 前置条件的独立布尔开关；
- 0–90° 双物种夹角扫描的范围、步长、固定阱深和数值精度。

运行 `python -m continuous_loading handover-map` 或
`python -m continuous_loading handover-angle-scan` 时会自动读取此文件。

## `design_optimization_defaults.json`

这是 `optimize-design` 的集中配置。它把参数分为：失谐/功率/束腰扫描、
固定 handover 时序、物理限制、10% 稳健性规则、Monte Carlo、相位扫描和
绘图。默认 `one_at_a_time` 表示分别改变一个变量；`box_corners` 表示
检查三个变量联合扰动，条件更严格。
命令行参数可以临时覆盖常用扫描和 Monte Carlo 参数；没有对应命令行
选项的参数直接在该 JSON 中修改。

## `l1_transport_defaults.json`

这是 `l1-transport-scan` 与 `l1-handover-scan` 的集中配置，保存失谐/功率网格、MOT 和 L1
初态、运输距离/加速度/速度、沿程束腰、光路效率、统计损失系数、技术
噪声接口、工作点权重、handover Monte Carlo 和画图参数。论文没有给出的损失系数默认置零，
需要用实验标定值替换后才能预测对应的真实速率损失。

当前统一扫描范围为 100–800 GHz、0–1.5 W；`initial_state.temperature_uK`
是 L1 起点的可调初温，默认 20 µK——物理图景为静止 L1 光晶格中束缚
足够长时间的热平衡系综（LGM 装载模块已移除，初始条件由
`continuous_loading/initial_state.py` 采样）。`species_defaults` 只
分别保存 Rb/Cs 光路效率和参考点。Rb 使用论文 300 GHz、典型入射功率
1 W；Cs 使用 70% 光路效率和由 600 GHz、500 µK 目标反解的范围外参考点。
`initial_state.pre_ramp_survival_fraction`（默认 1.0）是 MOT 计数时刻到
L1 初态边界的前级存活率记录；`loaded_l1_atom_number` 已是 L1 初态
原子数，不重复乘。

`handover_monte_carlo` 保存联合扫描参数，默认每个正功率点使用 1000
条轨迹、0.25 µs 步长、1000 µs handover 和 20 个 CPU 工作进程。
`compute_backend` 选择 Monte Carlo 内层积分的计算设备：`cpu`
（默认，NumPy）或 `gpu`（CuPy/CUDA，需自行安装；gpu 时外层扫描
不用进程池，全部网格点的 handover MC 自动摊平为一次批量 GPU
调用，见 `continuous_loading/handover_batch.py`）。其中的
`adaptive_refinement` 子分组（默认 `enabled: false`）开启两阶段
自适应粒子加密：第一遍按 `particle_count` 扫描，交接率 Jeffreys
标准误超过 `target_standard_error` 的点按
`particle_count×(SE/目标)²` 条轨迹（同 seed、上限
`max_particle_count`）自动复算一遍。
`handover_preconditions` 的三个布尔量分别控制最小阱深、L1 起点最大
功率和临界加速度；当前均为 `false`。handover-map 配置中的
`preconditions` 也全部关闭，`pipeline.use_l1_transport=true`，因此
双物种效率图与联合扫描使用同一批 L1→handover 数据。

`conveyor_geometry` 是可选的 offset-waist 双束 conveyor 几何分组：
`enabled`（默认 `false`，关闭时 L1/L2 行为与既有模型逐位一致）、
`waist_um`（单束腰，默认 250 µm）和 `waist_separation_cm`（束腰
间距 s，默认 19.5 cm = L1 距离的一半，即两腰位于 1/4、3/4 处）。
启用后 L1/L2 腿改用逐点几何剖面和恒源端功率策略，公式见
`reports/可选运输模型理论框架.md`。

`transport_monte_carlo` 是可选的运输腿轨迹级 Monte Carlo 分组：
`enabled`（默认 `false`，开启后 `simulate_l1_transport` 顶部分支到
`transport_mc.simulate_leg_monte_carlo`，L2 腿经 `replace` 自动继承）
和 `time_step_us`（velocity-Verlet 请求步长，默认 0.5 µs；实际步长
会按沿程最快轴向阱频自动钳制到 ω_z·dt ≤ 1——与 handover 同一判据，
实际值见 `L1DesignPoint.actual_time_step_us`——定量工作仍应做步长
减半收敛检查）。粒子数、随机种子、散射开关和轴向云宽与 handover
Monte Carlo 共用 `handover_monte_carlo` 分组，不重复设置；N=1000 时
L1 每网格点数十秒级。GPU 扫描时 L1 腿、handover 与全链路 L2 腿各自
合并为一次批量调用（L2 腿逐点初温/原子数经 `transport_batch` 初态
白名单支持），关闭 MC 或批量未覆盖（conveyor 几何）时回退逐点。
公式见
`reports/可选运输模型理论框架.md`。

`l2_transport` 分组保存 `full-chain-scan` 的 L2 段固定参数：运输距离
0.17 m、加速度 4000 m/s²、最大速度 9.07 m/s（由论文 17 cm / 21 ms
按对称梯形轨迹推导）、末端束腰 150 µm、时间点数和占据格点数。起点
束腰沿用 `transport.handover_waist_um`（250 µm），L2 初态由 handover
Monte Carlo 的捕获样本逐点给出，不在这个文件中固定。

`transport_monte_carlo` 分组控制可选的轨迹级运输 MC：`enabled`
（默认 false，开启后 L1/L2 腿用 Monte Carlo 替代解析预算）和
`time_step_us`（默认 0.5 µs，必须保证 ω_z·Δt<2）。运输 MC 的粒子数、
种子、散射开关和云尺寸与 `handover_monte_carlo` 组合并调用
（`L1TransportInputs` 的 `mc_*` 字段，默认同值，可被每次调用覆盖）。

## 怎样使用

普通用户通常不需要直接编辑此文件。先运行：

```powershell
python -m continuous_loading paper
```

需要核对某个输入时，再回到 JSON 查来源和单位。

## 修改规则

修改数据时请遵守：

1. 数值字段名必须带清楚单位，例如 `_ms`、`_um`、`_m_s2`；
2. 论文实测值和模型假设分开命名；
3. 同时更新 `source` 或说明字段；
4. 不把程序输出写回这个目录；
5. 修改后运行完整测试，并检查论文复现结果是否发生预期变化。

程序生成的 JSON 和 CSV 应放在 [`../output/`](../output/)。
