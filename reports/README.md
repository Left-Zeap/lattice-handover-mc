# `reports`：理论框架报告

这个目录保存面向同事阅读的理论推导、论文参数整理和计算路线说明。
代码告诉你"怎么算"，这里的文档解释"为什么这样算、哪些近似能用"。

## 现状注记（装载模块已移除）

L1 静止装载模块（`loading_ramp*` 全家）已自程序移除：初始条件改为
"L1 运输起点被静止 L1 光晶格束缚足够长时间的热平衡系综"（默认
20 µK，见 `data/l1_transport_defaults.json` 的 `initial_state` 组与
`continuous_loading/initial_state.py`）；全链留存链为
S_total = S_L1·η_HO·S_L2，η_load 不再进入。涉及装载模型的理论文档
（`LGM静止L1装载模型实现说明.md`、`L1静止装载温度口径与相空间尖刺分析.md`、
`双晶格连续装载时序与Loading_Ramp解读.md`、`程序计算流程与物理过程时序.md`
等）保留为理论历史记录，正文未随程序更新；引用其中以装载段为前提的
数值结论时请注意这一口径变化。

## 现状注记（运输腿 MC 步长精度守卫）

运输腿 Monte Carlo（`transport_mc`/`transport_batch`/`light_field`）
的积分步长现在统一经 `_stable_leg_step_s` 精度守卫钳制：ω_z·dt ≤ 1
（与 handover 既有守卫同一判据，ω_z 取沿程最大轴向调制深度对应的
阱频），实际步长记录在 `L1DesignPoint.actual_time_step_us`。此前运输
腿无守卫：请求步长偏大（含旧默认 0.5 µs 在 L1 典型阱频下
ω_z·dt ≈ 1.2 的临界情形）会让 velocity-Verlet 积分发散或产生非物理
加热，原子在 L1 段全部伪逃逸，二维连续相空间扫描因此大面积出现
"L1 末端无存活原子"无效点（21×21 网格曾仅剩约一成的有效点）。
`可选运输模型理论框架.md`、`L1-L2运输与全链路计算框架.md` 等文档中
涉及运输 MC 步长选择的论述（"保证 ω_z·Δt 不超过 ≈1.26"）以本守卫
为准；步长减半收敛检查的建议仍然有效。同期修复：连续相空间二维
扫描的进度计数统一折算为全局网格完成数，且逐点失败原因在扫描结束
时汇总报告（此前静默丢弃）。

## 推荐阅读顺序

### 第一次了解项目

1. [`handover交接的理论修正与程序逻辑.md`](handover交接的理论修正与程序逻辑.md)  
   上篇是最短理论入口：原始 handover 推导的关键错误和修正后的最重要
   结果；下篇是 `handover.py` 按代码执行顺序的完整计算逻辑。
2. [`连续装载双光晶格计算理论框架.md`](连续装载双光晶格计算理论框架.md)  
   从原子参数、晶格势到运输和 handover 的完整理论主线。
3. [`MOT到科学区参数提取.md`](MOT到科学区参数提取.md)  
   论文中各阶段的输入参数和信息边界。
4. [`激光空间分布与束腰功率表征.md`](激光空间分布与束腰功率表征.md)  
   L1/L2 晶格激光的空间形貌：束腰位置、横向高斯/轴向驻波分布、
   沿程功率（恒阱深跟随与 L2 的 0.36 缩放）及程序参数索引。
5. [`论文Rb87基准对比验证报告.md`](论文Rb87基准对比验证报告.md)  
   以论文参数原样运行程序（解析链 + 连续相空间 MC）与论文公开
   结果的逐项对比：效率/终温差距的归因分析与口径声明。
6. [`多体效应定量评估报告.md`](多体效应定量评估报告.md)  
   10⁶ 量级原子的全链路密度估算与多体效应盘点：弹性碰撞/再热化、
   蒸发、两体光辅助损失、平均场、辐射俘获的定量指标，及与程序
   已有升温/损失机制的对比。
7. [`handover云宽尺度猜想与验证.md`](handover云宽尺度猜想与验证.md)  
   原子云轴向宽度对 handover 的影响：重合区尺度 z_ov=w/sinθ 的
   捕获窗口模型与云宽一维扫描（`cloud_sigma_scan`）的对比验证，
   含升温/交接率两个通道敏感区间的区分。
5. [`程序计算流程与物理过程时序.md`](程序计算流程与物理过程时序.md)  
   当前程序的计算流程与实验物理时序的逐阶段对照：每个阶段由哪个
   模块计算、阶段间传递什么量、有哪些口径约定。

### 需要推导和接口细节

4. [`光晶格输运过程分析_修正版.md`](光晶格输运过程分析_修正版.md)  
   原始晶格运输理论文件的审查与修正：单晶格势、散射、输运和功率关系。
5. [`L1-L2运输与全链路计算框架.md`](L1-L2运输与全链路计算框架.md)  
   上篇：L1 固定加速度/速度下的时序、宏观升温、统计损失和失谐—功率
   扫描接口；下篇：L2 运输段与全链路汇总（科学区末温、总留存率、
   原子库密度），含 handover 末温作为 L2 初温的再热化口径假设。
6. [`失谐量-功率分段线性规划理论框架.md`](失谐量-功率分段线性规划理论框架.md)  
   失谐量—功率设计图中的五类限制、分段仿射化、二维半平面交集、
   目标函数和非线性回代验证。
7. [`可选运输模型理论框架.md`](可选运输模型理论框架.md)  
   默认关闭的两层可选运输模型：offset-waist 双束 conveyor 几何
   （逐点阱深/阱频/可见度剖面、恒功率策略）与 L1/L2 腿的轨迹级
   Monte Carlo（底层双束光场、拒绝采样、散射反冲、途中逃逸剔除）。
8. [`LGM静止L1装载模型实现说明.md`](LGM静止L1装载模型实现说明.md)  
   已移除的 loading 模型（MODEL_VERSION `lgm_static_l1_loading-2.0`，
   保留为理论历史记录，见顶部现状注记）：LGM 起点为动力学边界、
   L1 全程静止满功率、LGM 末端一次捕获判定、与 L1/全链路的拼接
   接口和待标定参数清单。
9. [`L1静止装载温度口径与相空间尖刺分析.md`](L1静止装载温度口径与相空间尖刺分析.md)  
   （历史记录）静止装载为何是 LGM 耗散冷却辅助的捕获；LGM 起点温度
   与 L1 运输起点温度两个口径；连续相空间加速尖刺的物理来源与数值
   伪峰修复（逃逸剔除 200 步、共动系去质心温度、强制 minimum_jerk）。
10. [`导师讨论后的研究与计算路线.md`](导师讨论后的研究与计算路线.md)  
    导师讨论后为何收缩优化变量、怎样定义 10% 工作平台，以及
    `optimize-design` 稳健优化程序的当前 Cs 结果。

## 文档之间的关系

```text
原始 PDF
  ├── 光晶格运输过程分析(1).pdf
  └── handover升温分析.pdf
          ↓ 审查与修正
reports 中的修正版和讲解版
          ↓ 对应程序
continuous_loading 中的数值模型
          ↓
失谐量—功率分段 LP 与非线性回代
          ↓
L1 统计运输与 handover 轨迹联合扫描
          ↓
L2 宏观运输与科学区汇总（全链路）
          ↓
output 中的 JSON、CSV 和图片
```

原始 PDF 位于仓库根目录。报告中的公式应尽量链接到程序变量或输出
文件，避免只给出无法复现的手算结果。

## 怎样复现报告结果

公式审查：

```powershell
python -m continuous_loading.handover_formula_validation `
  --json output\handover_formula_validation.json `
  --csv output\handover_formula_trace.csv `
  --plot output\handover_formula_validation.png
```

Handover 轨迹：

```powershell
python -m continuous_loading handover `
  --particles 2000 `
  --json output\handover_paper.json `
  --plot output\handover_temperature.png
```

失谐量—功率分段线性规划：

```powershell
python -m continuous_loading lp-design `
  --json output\detuning_power_lp.json `
  --plot output\detuning_power_lp.png
```

L1 运输与 handover 联合扫描：

```powershell
python -m continuous_loading l1-handover-scan --atom Rb-87
python -m continuous_loading l1-handover-scan --atom Cs-133
```

MOT→L1→handover→L2→科学区 全链路扫描：

```powershell
python -m continuous_loading full-chain-scan --atom Rb-87
python -m continuous_loading full-chain-scan --atom Cs-133
```

工程配图：

```powershell
python generate_report_figures.py
```

## 阅读报告时的三个注意点

1. "解析上界"不等于正式工作点的实际温升；
2. \(U/(k_BT)\) 是势深裕量，不是非平衡 handover 交接率；
3. 论文没有公开的交叉角、相位、AOM 波形和噪声谱必须标成模型假设。

## 新增报告的建议结构

为了便于同事复查，建议依次写：

1. 问题和输出量；
2. 输入参数、单位与来源；
3. 公式及适用条件；
4. 数值结果；
5. 与程序文件和输出文件的对应；
6. 尚未建模的物理；
7. 可复现命令。
