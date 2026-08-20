# `tests`：物理与数值回归测试

这个目录用来防止代码修改后悄悄破坏物理标度。测试不只是检查 Python
能否运行，还检查一些必须成立的物理极限。

## 运行方法

从仓库根目录运行全部测试：

```powershell
python -m pytest -q
```

运行单个文件：

```powershell
python -m pytest tests\test_handover.py -q
```

查看失败测试的详细输出：

```powershell
python -m pytest -vv
```

## 文件说明

| 文件 | 主要检查 |
|---|---|
| `test_atomic_and_lattice.py` | D1 红失谐波长、论文势深数量级、功率反解、孔径截断、加速势垒 |
| `test_transport_and_scenarios.py` | 论文速度和末温复现、Cs 功率—散射趋势、运输功率缩放、热束缚比例 |
| `test_handover.py` | 相同晶格恒等极限、空间错位损失、加速度势垒、随机种子复现、Figure 2 预设 |
| `test_handover_formula_validation.py` | 原始公式代数回代、近似失效告警、周期势有界性、Monte Carlo 平均 |
| `test_linear_design.py` | handover 时间标度、加速势垒深度、LP 可行性和非线性回代 |
| `test_handover_map.py` | 双原子默认参数、固定 1000 µs、只对可行点运行 Monte Carlo、串并行一致性、交接率/升温绘图 |
| `test_handover_angle_scan.py` | 0–90° 默认范围、双物种夹角扫描和交接率/升温折线图 |
| `test_design_optimization.py` | 失谐—功率—束腰稳健窗口、两种容差模式、候选 MC 和相位扫描 |
| `test_l1_transport.py` | L1 梯形时序、功率随束腰缩放、宏观损失积分、二维扫描与四联图 |
| `test_l1_handover.py` | transport 末温和原子数向 handover 的接口传递、统一网格与全流程四联图、自适应粒子加密第二遍复算 |
| `test_conveyor_geometry.py` | offset-waist 束腰半径、与 evaluate_lattice 的阱深锚定、可见度极限、错腰摊平效果、关闭回归 |
| `test_l2_transport.py` | L2 梯形时序复现论文 21 ms、恒阱深功率缩放、绝热压缩升温、非法输入 |
| `test_full_chain.py` | handover 捕获态向 L2 腿的传递、三段留存率相乘闭环、零捕获点排除、全链路四联图 |
| `test_transport_mc.py` | 运输 MC 力-势梯度一致、静态深阱守恒、蒸发选择、种子复现、步长收敛、解析量级一致、采样失败零留存、MC 模式扫描启动与逐点隔离 |
| `test_gpu_backend.py` | 后端校验、GPU 可用性守卫、CPU/GPU 统计一致（handover 与运输腿）、GPU 种子复现、固定相位口径 GPU 捕获系综返回、表单后端字段往返、设备端循环 kernel 与逐步融合路径统计一致、逐点初态白名单批量、全链路 L2 批量调度 |
| `test_transport_mc.py` | 双束力与势的数值梯度一致、(1+√R)² 波腹退化、深阱全留存、种子复现、浅阱蒸发选择、解析/MC 量级、步长收敛、分发器同型、全链路 MC 冒烟、UI 字段往返 |
| `test_ui_smoke.py` | UI 主窗口六页面与标题（离屏）、晶格速算阱深锚点、小表单单点全链路三相轨迹、云宽扫描页构造与 controllers 接线往返 |
| `test_cloud_sigma_scan.py` | 云宽一维扫描：结果结构与 σ/w₀ 归一化、双云宽字段同步替换、进度计数、同 seed 可重复、单点失败隔离与 Top-3 汇总、非法输入 |
| `test_initial_state.py` | 静止 L1 晶格热平衡系综采样：动能温度 ≈ 设定值、束缚域截断、种子复现、ParticleEnsemble 校验 |
| `test_light_field.py` | 三段光场"预计算时序 + (xyz,t) 查询"与现有逐点调用逐位一致（legs 对照 transport_mc、handover 对照 handover） |
| `test_chain_mc.py` | L1→handover→L2 单系综链式 MC 端到端冒烟、留存 ∈[0,1]、handover 段粒子数恒定、初温 ≈ 设定值 |
| `test_sequence_improvements.py` | 段边界晶格相位规范化、重力开关向 L1/L2 腿传播、实测波形驱动时序、minimum_jerk 端点加速度、连续相空间三段贯通与 2×2 扫描（CPU/GPU） |

## 最有教学意义的测试

### 相同晶格恒等极限

当两晶格完全相同、同相且 \(U_1(t)+U_2(t)\) 恒定时，handover 不应
产生额外升温，交接率应为 1。这个极限用于检查力和积分器。

### 功率反解闭环

先由目标势深反求功率，再把功率代回晶格模型，必须恢复原势深。这个
测试能发现强度、回程功率比和束腰定义中的系数错误。

### 周期势能量有界

原始角度公式会给出远大于势深的能量。测试要求完整周期势的相位错位
能量不能超过物理上界。

### 随机种子可复现

相同参数和种子必须产生相同 Monte Carlo 结果，便于定位回归问题。

### LP 推荐点必须通过原模型

分段直线只用于构造几何可行域。测试要求最终推荐点重新代入完整偶极
势、散射率和加速势垒模型后，仍满足全部原始非线性限制。

## 新增测试的原则

修改物理模型时，优先添加以下类型的测试：

1. 可手算的极限；
2. 单调性，例如失谐增大时固定势深所需功率上升、散射下降；
3. 正反计算闭环；
4. 时间步长收敛；
5. 单位和非法输入；
6. 固定种子的数值回归。

不要只把当前输出复制成断言；应说明为什么该结果在物理上必须成立。

## 测试未覆盖的内容

当前测试不能证明模型已经包含所有实验物理，例如：

- 实际 AOM 波形和相位瞬态；
- 多能级 Raman/Rayleigh 分解；
- 碰撞再热化；
- 量子能带和隧穿；
- 实验技术噪声。

测试通过表示“当前模型内部自洽”，不表示模型已经等同实验。
