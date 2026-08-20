# `ui`：连续装载双光晶格计算平台（PySide6 桌面界面）

本包是 `continuous_loading` 计算库的界面封装，不包含任何物理模型。
启动方式：

```powershell
python -m ui
```

无显示环境冒烟启动：`QT_QPA_PLATFORM=offscreen UI_AUTO_QUIT_MS=2000 python -m ui`。

## 页面

| 页面 | 功能 |
|---|---|
| 概览 | 静止 L1 初态→L1→handover→L2→科学区流程（MOT 明确标为计数基准）、默认配置摘要、跳页入口 |
| 单点计算 | 分组参数表单（默认值取自 `data/l1_transport_defaults.json`，含可选的 "conveyor 几何（可选）" 分组：启用开关、单束腰、束腰间距 s；"handover Monte Carlo" 组内有交接 MC 数值参数、"阶段间连续传递相空间" 开关、"运输动力学"（解析预算 / Monte Carlo）和"运输 MC 步长 (µs)"，控制 L1/L2 运输腿是否走轨迹级 Monte Carlo；扫描页共用同一表单）、晶格指标速算（同步）、单点全链路 Monte Carlo（后台线程，可取消），指标卡片 + 分阶段表格（首两行显示 MOT→L1 边界存活率与静止晶格热平衡初态） |
| 时序可视化 | 时间滑块（0.01 ms 分辨率）+ 时间输入框双向同步；装置示意图（MOT 腔、L1/L2 光路、handover 交叉、科学区）用红点实时显示原子云位置并高亮当前阶段；2×2 时序图带光标线和当前值标记；运动学与光路图覆盖 L1+L2 全程（handover 段无定义，NaN 断开，四张图统一色带标注）；拖动只更新光标不重画全图 |
| 二维扫描 | 失谐—功率网格扫描（默认 9×9、每点 500 轨迹、串行后端；扫描网格参数为本页专属控件组，独立于共享表单）；2×2 热图点击查询网格点指标；扫描后可用勾选阈值（P/ret/heat）或自定义表达式（P、ret、heat、eff、dens 五个变量，AST 白名单解析）在已有结果上条件框选，热图叠加符合点散点和掩膜轮廓线，不重新计算 |
| 云宽扫描 | 固定失谐/功率工作点上扫描原子云轴向宽度 σ（默认 0–5 mm、10 点，本页专属控件组；开始/取消与进度条在右上区，与二维扫描同位置）；两张折线图分别给出 handover 末温/链末总温和 handover 交接率/相对 MOT 总留存随无量纲 χ = σ_c·sinθ/w 的变化；无效点跳过不画，全部无效时显示提示；计算设备（CPU/GPU）由共享表单的"计算设备"控制 |
| 结果导出 | 计算历史列表（含每次运行的墙钟时间），导出 JSON / 轨迹或网格 CSV / 功能图 PNG（默认 `output/ui_export_*.*`）；JSON/CSV 写入 `runtime_seconds`，PNG 写入运行时间元数据；PNG 优先保存时序/扫描页当时显示的 figure（含条件叠加状态），更早的历史条目回退为预览重绘并在完成提示中注明 |

## 结构

- `app.py`：QApplication、主窗口（左侧导航 + `QStackedWidget`）。
- `theme.py`：浅色 QSS 主题（主色 `#2563eb`）。
- `state.py`：`AppState`，跨页共享最近结果与计算历史，Qt 信号驱动刷新。
- `workers.py`：`CalcWorker(QThread)` 后台计算，协作式取消。
- `controllers.py`：纯 Python 控制层（无 Qt）：表单参数 ↔ 计算库输入的
  组装、单点/二维扫描/云宽扫描入口、扫描条件掩膜
  `scan_condition_mask`（含 AST 白名单表达式求值
  `evaluate_scan_expression`）。
- `timeline.py`：纯 Python 时间轴组装 `build_timeline` 与插值采样
  `sample_timeline`。
- `widgets/`：参数表单 `ChainParameterForm`、matplotlib 画布
  `PlotCanvas`（已设置中文字体与负号）、指标卡片与分阶段表格。
- `pages/`：六个页面，一页一个模块。

## 使用注意

- 数字输入框和下拉框已禁用滚轮（`NoWheelSpinBox` 系列），避免在
  滚动表单时误改参数；键盘上下键调整仍可用。
- 耗时计算都在 `CalcWorker` 后台线程执行，串行模式下取消在当前
  批次结束后生效；进程池模式需等已提交批次完成。成功、失败和取消
  都记录墙钟运行时间，单点/扫描页完成提示与结果导出页同步显示。
- "计算设备"选 GPU 时（需已安装 CuPy/CUDA），Monte Carlo 内层积分
  在 GPU 上进行；外层扫描不用进程池（多进程共享单 GPU 会挂起），
  全部网格点的 handover Monte Carlo 自动合并为一次批量 GPU 调用，
  MC 运输腿同样批量执行；L1 腿、批量积分、L2 腿各阶段都有进度
  反馈（进度条只增不减，阶段切换不回退）。CPU 与 GPU 结果只在
  统计意义上一致（随机数生成器不同）。
- 初始条件：原子系综在 L1 运输起点被静止 L1 光晶格束缚足够长时间
  达到热平衡（"L1 初始温度" 字段，默认 20 µK，取自
  `data/l1_transport_defaults.json` 的 initial_state 组）；LGM 装载
  模块已移除，链路无装载阶段。"L1 初始原子数" 与
  "MOT/compress/idle→L1 起点存活率" 分别给出边界原子数与尚未显式
  传播的前级损失。
- 单点页可选“阶段间连续传递相空间”；勾选后 UI 自动把运输动力学
  锁定为 Monte Carlo，并把 L1/L2 理想轨迹锁定为零端点加速度的
  最小冲击 S 曲线（无实测波形时禁止梯形加速度阶跃）。
  二维扫描同样支持该开关：CPU 逐点运行，GPU 按 L1/handover/L2
  三段固定形状批量推进。
- “交接相对相位口径”可选随机相位（多发次系综平均，默认）或固定
  相位（单发次口径，0–180° 可调，cos² 周期 180°）；固定口径下
  相位值进入 `L1HandoverInputs.relative_phase_rad`，单点与二维
  扫描经同一 `build_full_chain_inputs` 透传到 `HandoverParameters`。
- “实测控制波形 CSV”可分别指定 L1、handover、L2。选择文件后相应
  理想加速度/最大速度或 handover 时长会灰掉；handover 时长以文件为准。
  运输 CSV 必需 `time_ms`，并至少含 `position_m`、`velocity_m_s`、
  `aom_frequency_difference_mhz` 之一；可选 `acceleration_m_s2`、
  `source_power_scale`、`waist_um`、`delivery_efficiency_scale`。
  handover CSV 必需 `time_ms,lattice1_fraction,lattice2_fraction`，可选
  `relative_phase_rad`。
