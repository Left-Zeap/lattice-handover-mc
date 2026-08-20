# 交付物设计总纲（手册 + PPT 共用规范）

> 本文件是《连续装载双光晶格输运模拟》说明手册与汇报 PPT 的统一设计依据。
> 两份交付物共用同一套物理主线、数字口径、配色与图片素材。

## 0. 读者与目标

- 读者：同事、领导（冷原子/量子方向，但不一定熟悉本程序细节）。
- 目标：让读者理解 (1) 工作目标——为"连续装载双光晶格输运"实验方案建立数值模拟与参数设计工具；(2) 物理模型——每个阶段的物理过程与公式；(3) 计算方法——解析预算 + 轨迹 Monte Carlo 双轨制、LP 可行域筛选；(4) 程序架构与接口调用逻辑。
- 风格：简约、重点突出、少文字、多公式多图表；排版紧凑、留白少、字体大（PPT 标题 ≥32pt、正文 ≥18pt；手册正文小四/12pt，公式图表为主）。

## 1. 统一物理主线（一条线贯穿两份交付物）

```
实验目标(Nature 646,1075 连续3000量子比特)
  → 输运链路: MOT → 压缩 → LGM冷却装载 → L1运输(39cm/50ms) → 交接(1ms) → L2运输(17cm/21ms) → 科学区原子库
  → 核心问题: 原子数留存 × 温度控制 × 散射/相干性 → 需要全链路数值模拟 + 参数设计
  → 计算方法: 解析宏观腿(快,扫描) + 轨迹Monte Carlo(准,核验); 失谐-功率LP可行域初筛
  → 程序实现: continuous_loading 库(分层模块, 标量(N,T)/相空间连续两种接口)
  → 结果: Rb-87 复现论文; Cs-133 方案设计(614µK阱深, 600-700GHz失谐, 3-3.5W)
```

## 2. 关键数字口径（两份交付物必须一致）

### 论文（Rb-87, Nature 646, 1075, 2025）基准参数
- MOT: ~10⁷ 原子, 80 ms；压缩 7 ms；idle 1 ms；LGM 11 ms → 4×10⁶ 原子 @ ~20 µK
- 晶格激光 λ_L = 795.6118 nm，D1 红失谐 ~300 GHz；回程比 R = 0.88⁴ ≈ 0.60
- L1: 39 cm / 50 ms，waist 330→250 µm；交接 1 ms（强度反向线性 ramp，无冷却）
- L2: 17 cm / 21 ms，waist 250→150 µm；P_L2,end = 0.36·P_handover
- 最优 a_lat ≈ 4000 m/s²，v = 8–10 m/s（10 m/s ↔ Δν = 25.14 MHz，v = λΔν/2）
- 交付：每 150 ms 一团 2.5×10⁶ 原子 @ ~120 µK，双晶格效率 ~60%，温度 ×6
- DPT 截断 L_clip = e^(−2a²/w²) ≈ e^−18；阱深 U_lat > 500 µK（全程）
- 科学区后续（背景信息）：30 万原子/s 装镊，1.5–3 万量子比特/s，3240 位存储阵列维持 >2 h

### 程序计算结果（现有 output 图支撑）
- 解析温升路径（fig2/fig7 口径）：20 → L1 绝热 +10.8 → 交接相位加热(随机相位解析上界) +76.2 → L2 +13.0 ≈ 120 µK
- 交接 MC（正确工作点）：交接率 ≈ 1.000，净升温 ≈ 5.8 µK（远低于解析上界）
- 交接理论修正：原错误推导需 3238 W → 修正后 Cs 只需 3–3.5 W（阱深 614 µK，D1 红失谐 600–700 GHz，散射率 ~500 s⁻¹）
- 六大错误：条件极小值≠移动阱中心；小角度≠小相位(kθσ_y≈15.2≫1)；忽略周期性→能量无界(修正后 ΔE_site=U sin²(kδq)≤U)；自由度计数混用；1 ms 交接误用突然极限(ω∥τ≈1754)；U/(k_BT)≥α 是势深裕量非交接率(P_bound(5)=87.5%)

## 3. 核心公式清单（编号贯穿手册；PPT 选用加粗者）

- **(F1) 标量偶极势与散射率**（Grimm 远失谐，D1:D2 权重 1:2，可选反旋项）：
  U(r) = (πc²/2) Σ_j (g_j Γ_j/ω_j³) [1/Δ_j − 1/(ω_L+ω_j)] I(r)；Γ_sc = (Γ/ħω_L³)... 按 reports/连续装载双光晶格计算理论框架.md 与 dipole.py 口径
- **(F2) 驻波强度与阱深**：I_anti = 2P_f(1+√R)²/(πw²)；U = C·I_anti；工程线性化 U₀ = A(δ)·P_src，Γ_sc = S(δ)·P_src
- **(F3) 阱频与临界加速度**：ω_z = k√(2U/m)，ω_r = √(4U/(mw²))，a_c = U·k/m
- **(F4) 传送带速度**：v = λ·Δν/2（Δν 为逆向两束频差，AOM 线性扫频）
- **(F5) 加速倾斜势垒**：U_eff/U₀ = √(1−β²) − β·arccos β，β = a/a_c
- **(F6) 热束缚比例**：F₃(η) = 1 − e^−η(1+η+η²/2)，η = U/(k_BT)；η=5 → 87.5%
- **(F7) 分项温升**：绝热 T∝ω̄；反冲 ΔT = (2E_r/3k_B)·N_sc；加速度跳变 ΔT = m(Δa)²/(6k_Bω_z²)；参数噪声 Γ_para = ω̄²S_ε(2ω̄)/4
- **(F8) Langevin 装载**：m dv = −∇U_L1 dt − mγv dt + √(2mγk_BT_eq) dW + dp_rec；T(t) = T_eq + [T(0)−T_eq]e^−2γt
- **(F9) 交接捕获判据**：L2 共动系总激发能 E' < U₂,eff（轨迹 MC 逐粒子判定，不接受解析替代）
- **(F10) 科学区峰值密度**：n₀ = N·ω_r²·ω_z·(m/2πk_BT)^{3/2}；碰撞密度 γ = ½n²v_relσ，σ = 8πa²
- **(F11) LP 五约束**：P ≥ P_U(最小阱深)；P ≥ P_HO ∝ τ_HO^−2(交接轴向周期)；P ≥ P_bound(加速束缚, 二分法)；P ≤ P_sc(散射上界)；P ≤ P_max(硬件)；目标 J = P/P_max + 0.05(δ−δ_min)/(δ_max−δ_min)
- **(F12) 全链留存**：S_total = η_load · S_L1 · η_HO · S_L2，S_L1 = S_spill·S_rate
- **(F13) DPT 截断**：L_clip = exp[−2a²/w²(z_DPT)] ≈ e^−18

## 4. 图片素材清单

### 4a. 已有可直接复用（output/ 下，不得改动原图）
| 文件 | 内容 | 用途 |
|---|---|---|
| output/figures/fig2_temperature_path.png | 温度路径 + 温升堆积双面板 | 手册§温升预算 / PPT 温升页 |
| output/figures/fig7_temperature_stacked.png | Rb 温升分项堆积 | 同上（二选一或并用） |
| output/figures/fig3_cs_tradeoff.png | Cs 失谐功率散射折中 | Cs 方案页 |
| output/detuning_power_lp.png | LP 可行域四面板(Cs, τ_HO=0.2/0.3/0.4/1ms) | LP 设计页 |
| output/handover_efficiency_map.png | 失谐-功率交接率/升温热力图(Rb/Cs) | 交接 MC 页 |
| output/full_chain_scan_rb87.png | Rb 全链路扫描四面板 | 全链结果页 |
| output/full_chain_scan_cs133.png | Cs 全链路扫描 | Cs 结果页 |
| output/l1_handover_scan_rb87.png | L1+交接二维扫描 | L1/交接页备选 |
| output/loading_ramp_scan_rb87.png | 装载 LGM时间×功率扫描 | 装载页 |
| output/cs133_power_scattering_tradeoff.png | Cs 功率-散射折中 | Cs 页备选 |
| output/figures/fig5_dpt_geometry.png | DPT 几何（有 e□ 字形问题，需修复版） | 几何约束页 |

### 4b. 需新生成（放 manual/figures/，matplotlib，中文字体 SimHei/Microsoft YaHei，禁用 unicode 上下标一律用 mathtext）
1. `arch_schematic.png` 系统架构示意：MOT腔→DPT(倾角~4°)→交接点→L2→科学区(reservoir/制备区/存储区)，标注距离/时间/角度/waist
2. `timing_sequence.png` 时序图：MOT 80→压缩 7→idle 1→LGM 11→L1 50→HO 1→L2 21 ms 分段条 + v(t) 梯形速度曲线 + P(t) 功率跟随示意
3. `dipole_curves.png` 真实计算（import continuous_loading）：Rb87/Cs133 的 A(δ)、S(δ) 系数 vs D1 失谐 100–1000 GHz，双面板
4. `conveyor_principle.png` 传送带原理：t0<t1<t2 三时刻 cos² 驻波平移 + v=λΔν/2 标注
5. `tilted_barrier.png` F(β)=√(1−β²)−β·arccosβ 曲线(0≤β<1)，标注 a≈4000 m/s² 工作点与对应 a_c
6. `bound_fraction.png` F₃(η) 曲线，标注 η=5(87.5%)、η=10(≈99.95%)
7. `handover_potential.png` 交接瞬态 3 子图（t=0/0.5/1 ms）：L1 势渐降 L2 势渐升、相对相位 0 与 π 两行对比，展示相位失配→势垒/能量再分配
8. `pipeline_flowchart.png` 程序数据流图：装载→L1→HO→L2→科学区，各框标注模块文件名 + 传递量；双轨（解析腿/MC）与 LP 初筛支路
9. `module_layers.png` 模块分层架构图（基础物理层/波形接口层/阶段层/编排层/CLI入口 + GPU支路）
10. `lp_schematic.png` LP 五约束半平面→凸多边形示意（单面板中文，可用真实 Cs 参数重画）
11. `fig1_scheme_comparison_fixed.png` 修复版方案对照表（unicode ⁻¹ → mathtext）
12. `fig5_dpt_geometry_fixed.png` 修复版 DPT 几何（e□¹⁸ → mathtext e^{-18}）

### 4c. PPT 公式图片
用 matplotlib mathtext 渲染 F1–F12 中加粗公式为透明底 PNG（manual/figures/formula_*.png），字号大、深色字。

## 5. 说明手册结构（Markdown → pandoc → docx）

文件：`manual/连续装载双光晶格输运模拟_说明手册.md` → `.docx`

1. **引言：工作目标** — 论文背景一段话 + 我们的三个任务（复现 Rb / 设计 Cs / 建立全链模拟工具）+ 总体链路图(arch_schematic) + 时序图(timing_sequence)
2. **物理基础** — F1–F6：偶极势与散射率(dipole_curves)、光晶格与传送带(conveyor_principle)、加速势垒(tilted_barrier)、热束缚(bound_fraction)、DPT/重力几何约束
3. **分阶段物理过程与计算方法** —
   3.1 装载 LGM+静止晶格：F8 + Langevin MC + 捕获判据（loading_ramp_scan 图）
   3.2 L1 运输：F7 分项温升预算 + 恒阱深功率跟随 + 双轨(解析/MC)（fig2/fig7）
   3.3 交接 handover：双晶格时变势 + F9 + 六错误修正表 + MC 结果（handover_potential, handover_efficiency_map）
   3.4 L2 与科学区：F10 + 全链留存 F12（full_chain_scan_rb87）
4. **设计优化方法** — F11 LP 可行域（lp_schematic + detuning_power_lp）→ handover map MC 复核 → 稳健设计优化流程
5. **程序架构与接口** — 分层图(module_layers) + 数据流(pipeline_flowchart) + 两种接口约定（标量(N,T) vs 相空间连续 ParticleEnsemble）+ CLI 子命令表 + 配置流(data/*.json → dataclass 默认 → CLI 覆盖)
6. **典型结果** — Rb 复现对照论文表；Cs 方案（fig1_fixed, fig3_cs_tradeoff, full_chain_scan_cs133）；双轨结果对比与口径说明
7. **口径纪律与已知边界** — 三条纪律 + 再热化假设 + 论文未公开量=工程假设 + 损失系数默认全零
8. **附录** — 符号表；公式索引 F1–F13；CLI 命令速查；reports/ 文档索引

写作要求：每节"图+公式+少量解释"；公式用 LaTeX（pandoc 转 OMML）；数字一律采用 §2 口径；代码引用格式 `continuous_loading/xxx.py` 的 `func()`。

## 6. PPT 结构（16:9，~18 页，python-pptx，.venv 环境）

文件：`manual/连续装载双光晶格输运模拟_汇报.pptx`

每页 = 大标题 + 1–2 张图/公式 + ≤3 行短句。版式：标题栏 + 内容区，深蓝标题、红色强调、白底。

1. 封面：题目 + 一句话定位 + 日期
2. 工作目标：论文 3000 量子比特连续运行 → 我们的任务（3 条）→ 关键数字带（150ms/团、2.5×10⁶ 原子、120µK、~60%）
3. 总体架构：arch_schematic 全幅 + 链路旁注
4. 时序总览：timing_sequence 全幅 + 总周期 171ms/稳态 150ms
5. 物理基础①：F1+F2 公式图 + dipole_curves
6. 物理基础②：F4 conveyor_principle + F3/F5/F6 公式 + tilted_barrier
7. 计算方法总览：pipeline_flowchart（双轨制强调）+ 一句话口径
8. 装载阶段：F8 + loading_ramp_scan_rb87
9. L1 运输：F7 公式图 + fig2_temperature_path
10. 交接①物理：handover_potential + F9 判据
11. 交接②理论修正：六错误简表 + 3238W→3.5W 对比大字
12. 交接③MC 验证：handover_efficiency_map + 交接率1.000/净升温5.8µK
13. L2 与科学区：F10/F12 + full_chain_scan_rb87
14. 设计优化：F11 + detuning_power_lp（LP→MC 复核流程）
15. Cs-133 方案：fig1_fixed + full_chain_scan_cs133（或 fig3_cs_tradeoff）
16. 程序架构与接口：module_layers + 接口约定两条
17. 口径纪律：三条纪律 + 边界假设（大字少字）
18. 总结与下一步：3 条成果 + 3 条计划

配色：主色 #1F3B5C（深蓝），强调 #C0392B（红），辅助 #7F8C8D（灰），背景白；标题 Microsoft YaHei 32–36pt，正文 18–20pt。

## 7. 实现约束

- 图片统一 150–200 dpi，16:9 页面内图宽一般 ≥ 60% 页宽。
- 所有上标/下标/希腊字母用 mathtext（SimHei 缺字会显示 □，fig1/fig5 原图已有此问题）。
- 手册 docx 用 pandoc：`pandoc manual.md -o manual.docx --resource-path=...`（公式自动转 OMML，CJK 正常）。
- PPT 用 `.venv/Scripts/python`（已装 python-pptx 1.0.2）。
- 数字与 §2 不一致时以 §2 为准并在文中标注口径来源（论文/解析/MC）。
