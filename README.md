# 连续装载双光晶格计算代码

这是一套面向冷原子连续装载问题的教学型定量计算仓库。它从论文参数
出发，计算移动光晶格的势深、散射、运输温升和 handover 交接率，并
把 Rb-87 论文方案迁移到 Cs-133 的激光器选型问题。

如果你是第一次接触本项目，先记住下面这条主线：

```text
论文和原子数据
    ↓
光偶极势、散射率和晶格阱频
    ↓
双晶格运输与 handover 经典轨迹
    ↓
交接率、捕获原子温度和有效势垒
    ↓
Cs 激光波长、源端功率和散射约束
```

## 1. 这个仓库可以做什么

目前主要支持以下任务：

1. 复现论文中的 Rb-87 双晶格运输参数和温升预算；
2. 扫描 Cs-133 D1 红失谐，反求达到目标势深所需的功率和散射率；
3. 用三维经典轨迹 Monte Carlo 计算 handover 交接率和升温；
4. 逐式检查原始 handover 理论，区分“代数复现正确”和“物理近似
   可用”；
5. 以 D1 红失谐和源端功率为二维自变量，将阱深、handover 时间、
   加速束缚、散射率和功率上限转成分段线性约束，求解并绘制 LP
   可行域；
6. 固定 L1 加速度和最大运输速度，扫描失谐—功率，计算全程宏观升温、
   统计留存率，并比较最优与较差可行工作点的时间轨迹。
7. 在同一失谐—功率网格上把 L1 末温和末态原子数传给 handover
   Monte Carlo，计算 MOT→L2 装载率、总升温和完整时间轨迹。
8. 固定晶格阱深，扫描 0–90° 的 L1/L2 夹角，比较 Rb/Cs 的 handover
   交接率和升温。
9. 在 L1→handover 之后接上 L2 宏观运输腿，完成 MOT→L1→handover→
   L2→科学区 全链路，输出科学区末温、总留存率和原子库密度。
10. 可选的 offset-waist 双束 conveyor 几何（错腰摊平沿程阱深，
    默认关闭，见 `data/l1_transport_defaults.json` 的
    `conveyor_geometry` 分组）。
11. 可选的轨迹级运输 Monte Carlo（L1/L2 腿从解析预算升级为
    双束光场轨迹模拟，默认关闭，见 `transport_monte_carlo` 分组）。
12. Monte Carlo 的 GPU 加速：内层积分可用 CuPy/CUDA，扫描时全部
    网格点合并为一次批量 GPU 调用（handover 与运输腿均支持，见
    `compute_backend` 参数）。
13. L1 初态为静止 L1 光晶格束缚热平衡系综（默认 20 µK，LGM 装载
    模块已移除）：宏观温度只用于初始采样，之后逐粒子传播（见
    `continuous_loading/initial_state.py`）。
14. 阶段间连续相空间传递（`FullChainInputs.phase_space_continuity`，
    默认开启）：粒子集合经 phase_space.py 在"L1→handover→L2"三段
    边界直通；该模式无实测波形时强制 minimum_jerk，禁止梯形
    加速度阶跃。

当前主程序是 [`continuous_loading/`](continuous_loading/)。根目录的
[`main.py`](main.py) 和 [`laser_formulas.py`](laser_formulas.py) 是早期
图形界面与二能级公式原型，只适合教学和数量级试算，不是当前正式
计算入口。

导师讨论后新增了面向实验搭建的三变量稳健优化：固定现有运输与 handover
时序，只扫描失谐、每分支源端功率和束腰；默认要求每个变量单独变化
±10% 时仍满足约束，再用 Monte Carlo 验证少量平台候选，并生成 AOM
相位—预计保留原子数曲线。运行：

```powershell
python -m continuous_loading optimize-design
```

理论和结果见
[`reports/导师讨论后的研究与计算路线.md`](reports/导师讨论后的研究与计算路线.md)。

L1 全流程扫描可直接运行：

```powershell
python -m continuous_loading l1-transport-scan
```

连接 L1 运输与 handover：

```powershell
python -m continuous_loading l1-handover-scan --atom Rb-87
python -m continuous_loading l1-handover-scan --atom Cs-133
```

扫描 L1/L2 夹角：

```powershell
python -m continuous_loading handover-angle-scan
```

对应理论见
[`reports/L1-L2运输与全链路计算框架.md`](reports/L1-L2运输与全链路计算框架.md)。

在 handover 之后继续接上 L2 运输，完成到科学区的全链路：

```powershell
python -m continuous_loading full-chain-scan --atom Rb-87
python -m continuous_loading full-chain-scan --atom Cs-133
```

对应理论见
[`reports/L1-L2运输与全链路计算框架.md`](reports/L1-L2运输与全链路计算框架.md)。

## 图形界面

仓库附带一个 PySide6 桌面图形界面（[`ui/`](ui/)），把 参数设置 →
后台计算 → 时序/热图可视化 → 结果导出 串在一个窗口里。启动：

```powershell
python -m ui
```

五个页面：

1. **概览**：MOT→L1→handover→L2→科学区 链路流程卡片、默认配置摘要
   和跳页入口；
2. **单点计算**：分组参数表单（默认值取自 `data/l1_transport_defaults.json`），
   「晶格指标速算」同步出阱深/散射率/阱频等，「运行单点全链路」在后台
   线程跑 Monte Carlo（可取消），结果显示为指标卡片和分阶段表格；
3. **时序可视化**：时间滑块拖动全程时间轴（0.01 ms 分辨率，与
   时间输入框双向同步），装置示意图实时显示原子云位置和当前阶段，
   2×2 时序图带光标线和当前值标记；运动学与光路图覆盖 L1+L2 全程
   （handover 段无定义以 NaN 断开，四张图统一色带标注）；
4. **二维扫描**：失谐—功率网格扫描（默认 9×9 网格、每点 500 轨迹、
   串行后端，进程池留给大网格），2×2 热图支持点击查询任意网格点指标；
   扫描后可用勾选阈值或自定义表达式（AST 白名单解析）在已有结果上
   条件框选，热图叠加符合点散点和掩膜轮廓线，不重新计算；
5. **结果导出**：计算历史列表，选中后可导出 JSON、时间轨迹/网格 CSV
   和功能图 PNG——PNG 优先保存时序/扫描页当时显示的图（含条件叠加
   状态），更早的历史条目回退为重绘预览图（默认 `output/ui_export_*.*`）。

所有耗时计算都在后台线程执行，界面保持响应。界面额外需要 PySide6
（本机环境已安装；为保持命令行依赖最小，未加入 `requirements.txt`）。
无显示环境下可用 `QT_QPA_PLATFORM=offscreen UI_AUTO_QUIT_MS=2000 python -m ui`
做冒烟启动。

## 2. 五分钟快速开始

### 2.1 安装环境

在本目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果已有包含 NumPy、Matplotlib 和 Pytest 的 Python 环境，可以直接
安装依赖或跳过虚拟环境创建。

### 2.2 先检查程序

```powershell
python -m pytest -q
```

测试通过后，再运行下面任意一个示例。

### 2.3 复现论文 Rb-87 参数

```powershell
python -m continuous_loading paper `
  --json output\rb87_paper_reproduction.json
```

它会输出论文工作点的晶格参数、运输速度、温升预算和原子库数量级。

### 2.4 扫描 Cs 激光器候选

```powershell
python -m continuous_loading cs-scan `
  --depth 614.258 `
  --waist 250 `
  --detuning-min 300 `
  --detuning-max 1000 `
  --detuning-step 100 `
  --retro-ratio 0.59969536 `
  --delivery-efficiency 0.7 `
  --max-power 4 `
  --max-scattering 600 `
  --csv output\cs133_handover_laser_scan_614uK.csv
```

输出表中的 `source_power_w` 是一套晶格分支的源端功率；
`forward_power_at_atoms_w` 是损耗之后到达原子处的前向功率。

### 2.5 计算 handover 交接率和升温

```powershell
python -m continuous_loading handover `
  --particles 2000 `
  --json output\handover_paper.json `
  --plot output\handover_temperature.png
```

程序会同时报告：

- 被 Lattice-2 捕获的轨迹比例；
- Monte Carlo 统计误差；
- 捕获子样本的初始、末态温度和净升温；
- 散射次数与反冲温升；
- 后续加速度造成的有效势垒降低。

完整计算链和每个输出字段的物理含义见
[`reports/handover交接的理论修正与程序逻辑.md`](reports/handover交接的理论修正与程序逻辑.md)。

### 2.6 检查原始 handover 公式

```powershell
python -m continuous_loading.handover_formula_validation `
  --depth 500 `
  --temperature 30.8 `
  --waist1 250 `
  --waist2 150 `
  --angle 4 `
  --duration 1 `
  --detuning 600 `
  --json output\handover_formula_validation.json `
  --csv output\handover_formula_trace.csv `
  --plot output\handover_formula_validation.png
```

看到闭式公式残差很小，只表示原公式的代数被正确复现。还必须检查
程序给出的 `k*theta*sigma_y`、`omega*tau` 和 `PASS/WARN`。

### 2.7 绘制失谐量—功率 LP 可行域

```powershell
python -m continuous_loading lp-design `
  --atom Cs-133 `
  --handover-times 0.2,0.3,0.4,1.0 `
  --depth 500 `
  --temperature 120 `
  --bound-fraction 0.8 `
  --max-power 6 `
  --max-scattering 600 `
  --plot output\detuning_power_lp.png `
  --json output\detuning_power_lp.json
```

图的横轴为相对 D1 的红失谐，纵轴为每条晶格分支的源端功率。曲线
分别表示目标阱深、handover 轴向周期数、加速后束缚比例、最大散射率
和最大功率；绿色区域是各分段二维 LP 的可行域，星号是线性目标函数
给出的推荐点。

底层物理约束本身不是严格线性的。程序在每个小失谐区间内构造保守
仿射边界，求解半平面交集，并把推荐点代回完整非线性晶格模型核验。
默认的“80 个轴向周期”是与论文 handover 时间平台同量级的工程判据，
不是论文直接测得的普适常数。

### 2.8 在可行域内计算 Rb/Cs handover 效率热力图

```powershell
python -m continuous_loading handover-map `
  --plot output\handover_efficiency_map.png `
  --json output\handover_efficiency_map.json
```

交接时间固定为 1000 µs。默认扫描 100–800 GHz、0–1.5 W，使用
25×25 网格和每点 10000 条轨迹。程序先用阱深、轴向周期数和加速后
热束缚比例筛选动力学可行点，再运行三维轨迹 Monte Carlo。输出为
2×2 图：Cs/Rb 各自的交接率和 handover 升温；空白表示动力学不可行。
Rb 默认初温为 30.8 µK，Cs 默认初温为 120 µK。

两体系共享的筛选默认值为：250 µm 束腰、500 µK 最小阱深、0.80
目标束缚比例、4000 m/s² 后续加速度、80 个最少轴向周期、600 s\(^{-1}\)
散射参考线、0.70 源端到原子效率和 \(0.88^4\) 回程功率比。散射率不再
提前删除网格点，因为随机散射反冲已经显式进入 Monte Carlo。

这些默认值统一保存在
[`data/handover_map_defaults.json`](data/handover_map_defaults.json)。
修改横纵轴范围、二维网格密度、Monte Carlo 粒子数和步长、初温、
CPU 工作进程数、物理限制或绘图色标时，只需修改该文件。热力图同时
叠加五条条件曲线、动力学可行域轮廓、图例、交接率和升温颜色标度尺。

可行网格点之间相互独立，默认使用 20 个 CPU 进程并行计算。设置
`--workers 1` 或 `--parallel-backend serial` 可恢复串行；相同随机
种子下串并行结果逐点一致。Monte Carlo 内层积分可用 GPU 加速：
安装 CuPy/CUDA 后把 `data/l1_transport_defaults.json` 的
`handover_monte_carlo.compute_backend` 改为 `"gpu"`（或 CLI 加
`--compute-backend gpu`、UI"计算设备"选 GPU）。CPU 与 GPU 随机数
序列不同，结果只在统计意义上一致。**GPU 模式下外层扫描不用进程
池**（多个进程共享单块 GPU 会竞争 CUDA 上下文和内核缓存），而是
把全部网格点的粒子摊平成一次批量 GPU 调用
（`handover_batch.run_handover_monte_carlo_batch`）：velocity-Verlet
整步已融合为单个 mega-step kernel，散射反冲用固定事件槽实现避免
逐事件设备同步，网格点越多批量优势越大；MC 运输腿同样可批量
（`transport_batch.run_leg_monte_carlo_batch`，conveyor 几何除外；
全链路 L2 腿的初温/原子数逐点不同，经逐点初态字段白名单同样一次
批量完成；连续相空间 GPU 扫描的 L1 初态为逐点 CPU 热平衡采样，
经 `initial_ensembles` 喂入同一批量 kernel），
且 GPU 扫描的 L1 腿、handover、L2 腿各阶段全程有进度反馈（详见
`continuous_loading/README.md` 的 `gpu_backend.py` 一节）。
批量积分的主循环默认走 `device_loop.py` 的设备端时间循环
RawKernel：每个线程一条轨迹、整段循环在设备上执行（散射用 cuRAND
逐事件抽取），只在段边界与 host 交互，消除逐步 Python 调度、
kernel 启动与散射同步的固定开销——长步数运输腿实测提速约 20 倍；
RawKernel 编译失败自动回退逐步融合 kernel（逐式同构、统计一致）。
另有两阶段自适应粒子加密可选开关
（`handover_monte_carlo.adaptive_refinement`，默认关闭）：第一遍
按基础粒子数扫描，交接率标准误超标的点按 1/√N 标定更多轨迹
自动复算，把 Monte Carlo 预算集中在交接边界附近。

当前 `output/handover_efficiency_map.*` 已按 10000 粒子正式重算。进一步
提高精度时仍应比较更密网格、多个随机种子以及 `--time-step 0.25` 与
`--time-step 0.1` 的收敛性。

前置筛选条件可以在两个全局 JSON 中分别开关：
`l1_transport_defaults.json/handover_preconditions` 控制 L1 的最小阱深、
起点最大功率和临界加速度；`handover_map_defaults.json/preconditions`
控制独立 handover 图的阱深、热束缚比例和最少轴向周期。对应 CLI 也支持
`--条件名` 与 `--no-条件名`。当前两套 JSON 默认全部关闭，且
`handover-map` 默认先运行 L1、再用逐点末态计算 handover。零功率点
始终跳过。

同属可选开关的还有 `l1_transport_defaults.json/conveyor_geometry`：
启用后 L1/L2 运输腿改用 offset-waist 双束几何（逐点阱深剖面 +
恒源端功率）；默认关闭，L1 关闭时采用起点直径、最小束腰和焦点位置
标定的高斯包络，详见
`reports/可选运输模型理论框架.md`。
`l1_transport_defaults.json/transport_monte_carlo` 启用后 L1/L2 运输
腿改用轨迹级 Monte Carlo（底层双束光场、逃逸剔除、Jeffreys 标准误，
默认关闭且关闭时行为不变），详见
`reports/可选运输模型理论框架.md`。

`handover-angle-scan` 默认在固定 500 µK 阱深下扫描 0–90°，步长 1°，
输出 Rb/Cs 交接率和升温双面板折线图
`output/handover_angle_scan.png`。

### 2.9 L1 初态与连续相空间全链路

LGM 静止装载模块已移除。L1 初态的物理图景：原子在 L1 运输起点附近
被静止 L1 光晶格束缚足够长时间，达到可调温度（默认 20 µK，见
`data/l1_transport_defaults.json` 的 `initial_state.temperature_uK`）
的热平衡；`continuous_loading/initial_state.py` 按该图景采样束缚
热平衡系综（谐振提议 + 重力下垂 + 轴向格点吸附 + 完整 cos² 势拒绝
+ Maxwell 速度），只把宏观温度用于初始采样，之后逐粒子传播。

全链路的连续相空间模式（`FullChainInputs.phase_space_continuity`，
默认开启；CLI 可加 `--no-phase-space-continuity` 回到 (N,T) 约化
接口）以同一经验粒子集合贯通 "L1→handover→L2" 三段（段边界做 z
平移与正交基旋转，不丢相位）；该模式要求运输腿走轨迹级 Monte
Carlo，且无实测波形时强制 minimum_jerk，禁止梯形加速度阶跃。
`continuous_loading/chain_mc.py` 提供同一图景的单系综三段编排器
（L1/L2 每步宽容捕获判定、handover 段不判定、粒子数恒定）。

## 3. 推荐学习顺序

如果目标是快速理解项目，建议按下面顺序阅读：

1. 本 README：先知道程序能做什么；
2. [`reports/handover交接的理论修正与程序逻辑.md`](reports/handover交接的理论修正与程序逻辑.md)：
   先理解最重要的理论错误和结果；
3. [`reports/连续装载双光晶格计算理论框架.md`](reports/连续装载双光晶格计算理论框架.md)：
   理解整个仓库的理论主线；
4. [`continuous_loading/README.md`](continuous_loading/README.md)：
   了解公式分别写在哪个模块；
5. [`tests/README.md`](tests/README.md)：通过测试理解程序应满足的物理
   极限。

## 4. 目录导航

| 目录 | 内容 | 适合什么时候看 |
|---|---|---|
| [`continuous_loading/`](continuous_loading/) | 当前正式 Python 计算包 | 修改模型、运行命令 |
| [`ui/`](ui/) | PySide6 桌面图形界面 | 交互式参数设置、可视化与导出 |
| [`atom/`](atom/) | Rb、Cs、Yb 原子知识文档 | 查原子数据和实验背景 |
| [`data/`](data/) | 论文参数的机器可读输入 | 查参数来源、修改基准输入 |
| [`reports/`](reports/) | 理论推导、审查和工程报告 | 理解物理、给同事交接 |
| [`tests/`](tests/) | 数值回归与物理极限测试 | 修改代码后验证 |
| [`output/`](output/) | 生成的 JSON、CSV 和 PNG | 查看计算结果 |
| [`output/figures/`](output/figures/) | 工程报告配图 | 重新生成报告图片 |
| [`.agents/`](.agents/) | 自动化协作说明的预留目录 | 一般用户无需使用 |

自动生成的 `.pytest_cache/`、`__pycache__/` 和版本控制目录 `.git/`
不属于项目内容，不需要阅读或手工修改。

## 5. 根目录文件

| 文件 | 作用 |
|---|---|
| [`requirements.txt`](requirements.txt) | Python 依赖 |
| [`Continuous operation of 3000qbit.pdf`](<Continuous operation of 3000qbit.pdf>) | Rb 连续运行论文原文 |
| [`handover升温分析.pdf`](handover升温分析.pdf) | 原始 handover 理论文件 |
| [`光晶格运输过程分析(1).pdf`](<光晶格运输过程分析(1).pdf>) | 原始晶格运输理论文件 |
| [`generate_report_figures.py`](generate_report_figures.py) | 生成工程报告的七张配图 |
| [`main.py`](main.py) | 早期 Tkinter 图形界面 |
| [`laser_formulas.py`](laser_formulas.py) | 早期二能级公式原型 |

运行旧图形界面：

```powershell
python main.py
```

使用前请先阅读 `laser_formulas.py` 顶部的局限说明。需要 D1+D2、
散射率、回程效率和 handover 动力学时，应使用
`python -m continuous_loading`。

## 6. 常用命令

查看所有入口：

```powershell
python -m continuous_loading --help
```

查看某个子命令：

```powershell
python -m continuous_loading handover --help
python -m continuous_loading cs-scan --help
```

生成基础结果图：

```powershell
python -m continuous_loading plots --output-dir output
```

生成工程报告配图：

```powershell
python generate_report_figures.py
```

扫描 Extended Data Figure 2：

```powershell
python -m continuous_loading handover `
  --figure2-panel a `
  --particles 1000 `
  --csv output\handover_fig2a.csv `
  --plot output\handover_fig2a.png
```

把 `a` 改为 `b` 或 `c`，可分别扫描交接距离、交接时间和后续加速度。

## 7. 单位和命名约定

命令行采用实验中常见单位：

- 波长：nm；
- 红失谐：GHz；
- 束腰：µm，指 \(1/e^2\) 强度半径；
- 势深和温度：µK；
- handover 时间：ms；
- 数值积分步长：µs；
- 功率：W；
- 散射率：s\(^{-1}\)；
- 加速度：m/s²。

程序内部统一转换为 SI 单位。变量名中的 `_uK`、`_um`、`_ghz`、
`_m_s2` 等后缀用于防止量纲混淆，新增变量时应继续遵守。

## 8. 怎样理解输出

三类输出回答不同问题：

| 输出 | 回答的问题 |
|---|---|
| `paper`、`cs-transport` | 解析运输预算和论文参数是否自洽 |
| `handover` | 在给定时变双晶格势中，有多少轨迹被捕获、捕获原子多热 |
| `cs-scan` | 达到指定静态势深时，需要什么波长、功率和散射率 |
| `lp-design` | 在失谐—功率平面上，各限制的几何交集和 LP 推荐点 |
| `handover_formula_validation` | 原始推导的代数和近似条件是否成立 |
| `full-chain-scan` | MOT→L1→handover→L2 全链路的科学区末温、总留存率和原子库密度 |

不要把 `U/(k_BT)` 直接称为 handover 交接率，也不要把突然切换解析
上界直接当作 1 ms 工作点的实际温升。

## 9. 模型边界

当前模型适合物理数量级计算、参数扫描和实验设计初筛，但不是完整
原子结构软件：

- 偶极势包含碱金属 D1、D2 标量贡献和反旋项；
- 未分辨指定 \(F,m_F\) 态的矢量与张量光移；
- 散射采用工程近似，未严格拆分 Raman 和 Rayleigh 通道；
- handover 使用经典轨迹，不描述接近基态时的量子波包和能带跃迁；
- 初始条件为静止 L1 晶格热平衡系综采样（经典轨迹 + 标量势），
  不描述装载前的 LGM 冷却、逐束散射力与内态泵浦过程；
- `lp-design` 是原非线性边界的分段保守线性化，不替代 handover
  Monte Carlo 或实验标定；
- 默认 AOM 斜坡、相位、交叉角和云尺寸中有论文未公开的工程假设；
- 实验最终选型还要加入实测光路效率、ASE、偏振和技术噪声。

因此，计算结果应与输入假设一起报告。

## 10. 修改代码后的最小检查

每次改动物理公式后至少执行：

```powershell
python -m pytest -q
python -m continuous_loading paper
python -m continuous_loading cs-scan --depth 500 --waist 250
```

若修改了 handover 力、积分器或捕获判据，还要比较两个时间步长：

```powershell
python -m continuous_loading handover --particles 1000 --time-step 0.1
python -m continuous_loading handover --particles 1000 --time-step 0.05
```

交接率和温升应在统计误差与预设数值容差内收敛。
