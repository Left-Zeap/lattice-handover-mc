# lattice-mc-transport

一个面向双光晶格连续输运的 **6D 单原子 trajectory Monte Carlo** 程序库，模拟

\[
\mathrm{L1\ transport}\rightarrow\mathrm{handover}\rightarrow\mathrm{L2\ transport}
\]

并从大量单原子轨迹统计：

- 存活率 \(S(t)=N_{\rm alive}(t)/N_0\)
- 三轴动能温度 \(T_x,T_y,T_z\)
- 标量温度 \(T=(T_x+T_y+T_z)/3\)
- 最终 L2 捕获率
- 散射事件数量
- 二维 \(\{\mathrm{D1~red~detuning},P\}\) 扫描的最终温度/存活率

## 1. 建模原则

轨迹层只传播原子的真实状态

\[
X=(\mathbf r,\mathbf v),
\]

每一步只显式加入底层作用：

\[
m\dot{\mathbf v}
=
-\nabla[V_{L1}(\mathbf r,t)+V_{L2}(\mathbf r,t)]
+m\mathbf g
+\text{stochastic recoil kicks}.
\]

其中：

1. `-grad(V)`：远失谐光的 AC Stark 偶极力；
2. `m g`：重力；
3. `recoil kick`：由局域远失谐散射率抽样的 Poisson scattering event；
4. beam power、lattice phase、moving lattice position、L1/L2 handover ramp 均直接写进时变势。

**不会**在单原子轨迹上额外添加：

- `T_ad`
- `Delta T_handover`
- `F3(eta)`
- 宏观 recoil 温升

这些量应当从 ensemble 的轨迹统计中自然出现。

## 2. 默认时序

`configs/paper_rb87.json` 使用 Lukin/Chiu 论文中的主要输运时序：

- L1: 39 cm / 50 ms
- handover: 1 ms，L1 线性 ramp down、L2 线性 ramp up，无冷却光
- L2: 17 cm / 21 ms
- 典型加速度：4000 m/s^2
- 两晶格夹角：4 deg
- Rb-87 D1 红失谐：300 GHz
- retro power ratio: 0.60
- L1 waist endpoints: 330 -> 250 um
- L2 waist endpoints: 250 -> 150 um
- 初始温度：20 uK

注意：论文没有完整公开所有时域光强、相位噪声、精确 waist(z) 和原子态分辨散射参数。
因此这些未公开量均保留在 JSON 配置中，不假装是论文实测值。

## 3. 安装

```bash
cd lattice_mc_transport
pip install -e .
```

GPU（CUDA 12，若尚未安装 CuPy）：

```bash
pip install -e ".[gpu]"
```

程序不强制安装 GPU 依赖。`backend=auto` 会优先使用 CuPy，失败则退回 NumPy。

## 4. 单点模拟

```bash
python -m lattice_mc single \
  --config configs/paper_rb87.json \
  --backend auto \
  --out output/rb87_single
```

快速 CPU 调试可先把配置中的：

```json
"n_atoms": 1000,
"dt_s": 5e-7
```

正式计算建议减小 `dt_s`，尤其轴向阱频达到数百 kHz 时。

输出：

```text
summary.json
timeseries.npz
temperature_timeseries.png
survival_timeseries.png
```

## 5. 二维 detuning-power 扫描

```bash
python -m lattice_mc scan \
  --config configs/paper_rb87.json \
  --scan configs/scan_example.json \
  --backend gpu \
  --out output/rb87_scan
```

GPU 后端会把整张参数网格合并为一个批量 ensemble 一次传播（各扫描点的
失谐/功率以逐原子数组进入融合 kernel），总耗时与单点同量级；CPU 后端
仍逐点顺序运行。

输出：

```text
scan_results.npz
scan_summary.json
final_temperature_heatmap.png
final_survival_heatmap.png
```

扫描中的 `power_w` 会同时覆盖 L1/L2 的 forward power；如需独立功率扫描，可修改 `scan.py`。

## 5.1 图形界面

```bash
pip install -e ".[ui]"   # 或 pip install PySide6
python -m ui             # 默认载入 configs/paper_rb87.json
python -m ui configs/custom.json
```

界面包含五个页面：概览（流程与参数）、单点计算（参数编辑+运行+结果）、
时序可视化（温度/存活率折线 + 可拖动时间光标 + 阶段条）、二维扫描
（GPU 批量并行 + 热力图）、结果导出（文件名自动附带参数与结果标签）。
UI 代码独立在 `ui/` 目录，仅调用 `lattice_mc` 公共接口。

## 6. 模块结构

```text
lattice_mc/
├── backend.py
├── config.py
├── constants.py
├── physics/
│   ├── atom.py
│   ├── dipole.py
│   ├── lattice.py
│   └── waveforms.py
├── simulation/
│   ├── initializer.py
│   ├── propagator.py
│   └── runner.py
├── statistics/
│   ├── diagnostics.py
│   └── survival.py
├── visualization/
│   └── plots.py
├── scan.py
└── cli.py
```

## 7. 存活判据

运输段对每个原子计算共动系局域激发能

\[
E_{\rm exc}
=
\frac12m|\mathbf v-v_L\mathbf e_L|^2
+
V(\mathbf r)-V_{\min,\mathrm{local}}.
\]

轴向加速后的局域逃逸势垒使用

\[
U_{\rm eff}
=
U_{\rm ax}
\left[
\sqrt{1-\beta^2}
-\beta\arccos\beta
\right],
\qquad
\beta=\frac{|a|}{U_{\rm ax}k/m}.
\]

为了满足“尽量宽松”的要求，默认不是一超势垒就删除，而是要求：

\[
E_{\rm exc} >
f_E U_{\rm eff}
\]

连续维持 `loss_grace_s` 才标记 lost，默认 `f_E=1.20`。

handover 中不使用解析势垒删除粒子；完整传播 \(V_1+V_2\)，在 handover 结束后再按 L2 的局域机械能判断是否被 L2 捕获。

此外有一个非常宽松的 hard-domain 判据：离所有活动 beam 轴均超过若干个 waist 时才强制 lost。

## 8. 温度口径

单原子没有“温度”。温度只在统计层由仍存活的 ensemble 得出：

\[
T_i=
\frac{m}{k_B}
\mathrm{Var}(v_i),\qquad
T=\frac{T_x+T_y+T_z}{3}.
\]

这里自动减掉 ensemble 的质心速度，所以 L1 以 8-10 m/s 运输不会被误算成热运动。

## 9. GPU

所有大粒子数组都由 `backend.py` 提供：

- CPU: NumPy
- GPU: CuPy

传播器、势场、散射抽样和统计均使用相同的 array API。
只有在记录 diagnostics 时才把少量标量搬回 CPU。

## 10. 当前模型的边界

这是一个“物理干净的第一版”：

- 已做：完整 3D Gaussian-envelope standing-wave potential
- 已做：两晶格 4° 几何
- 已做：moving lattice trapezoidal waveform
- 已做：handover opposite linear power ramps
- 已做：D1/D2 标量 AC Stark 与总 scattering rate
- 已做：随机 absorption + isotropic spontaneous-emission recoil
- 已做：CPU/GPU
- 已做：单点和二维扫描

暂未做：

- Rayleigh/Raman branching
- \(F,m_F\) internal-state Monte Carlo
- scalar/vector/tensor polarizability 完整多能级模型
- measured RIN/phase/pointing noise PSD
- 原子-原子碰撞
- LGM / MOT

这些都已经通过模块边界留出了后续扩展空间。
