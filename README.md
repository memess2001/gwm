# GWM：几何 / 信息几何世界模型的想法搜索

对"把某种几何结构放进世界模型某个部位，并给出一个可测不变量"这片想法空间做 MAP-Elites 式的搜索。
每个想法（格）按 **文献杀 → 思想实验杀 → 算力杀** 的顺序处决，只有幸存者才去碰外部 benchmark。
方法与约定见 `HANDOFF.md`。

**现状（2026-11-19）：** 格 0 阶段 1 与格 6 已跑完，格 0 的收敛复查（训练 5 倍长）分类全部不变，格 2 已判死；其余格子都做了预检（只核真值）。32 格的几何对象与同源量对照见 `correspondence_table.md`，需要 Z 决定的事见 `reports/2026-11-18_progress.md` 开头。

## 文件导航

| 路径 | 内容 |
|---|---|
| `HANDOFF.md` | 计划全文（v0.5）：生成器、31 格、动物园、主循环与过滤规则 R1–R5、出口、决定记录（§10） |
| `correspondence_table.md` | 32 格的几何对象、同源的非几何量、两者的关系类型与证据；"成立吗 / 新吗 / 比同源量好吗"三问 |
| `cells.yaml` | 生成器的活状态，**v0.5**（2026-11-04 JST，合并两份 v0.4 + 第二轮 16 格 + `rules` + `decisions`）。每格的 status / confidence / collision / nearest / must / same_source / notes；v0.4、v0.5 的改动分别以"v0.4（2026-11-03）""v0.5（2026-11-04 …）"开头写明理由 |
| `literature_audit.md` | 第一轮文献审计（2026-11-01），保持原样 |
| `literature_audit_round2.md` | 第二轮补查与引用核对（2026-11-03）；原始结构化输出在 `results/lit_round2_raw.json` |
| `cells/thought_experiments_2026-11-03.md` | 思想实验杀 / 修正：格 0、1、6、7、9、15，以及跨格模式 |
| `cells/<id>/prereg.md` | 预注册。格 0（v2.2）、格 6（v1.4）经 Z 批准并已跑完；格 15 只有草案（`cells/15/prereg_draft_2026-11-18.md`），未批准，不运行 |
| `cells/<id>/result.md` | 结果（格 0 阶段 1、格 6）；格 0 的收敛复查见 `cells/0/result_ext_convergence.md`（预注册 `cells/0/prereg_ext_convergence.md`） |
| `cells/<id>/precheck_*.md` | 写预注册之前的检查：只核真值、不训练模型，判断能否由思想实验决定；脚本在同格的 `exploratory/` |
| `cells/<id>/exploratory/` | 已披露的探索性数值检查（不是 kill test） |
| `cells/0/calibration.md` | 格 0 测量仪器（持续同调、能量平滑度）在真值嵌入上的校准报告 |
| `zoo/` | 世界动物园：每个世界都有 `sample()` 与 `ground_truth()`，答案是定理 |
| `baselines/` | 对照组世界模型：recon / jepa-sym / jepa-sg 三个变体（`jepa` 是 jepa-sym 的别名），4 个尺寸；`anchors="episode"` 用于整段开环训练（格 6）；另有训练脚本和 ledger 工具 |
| `analysis/` | 预注册测量用的仪器 |
| `results/ledger.md` | 实验流水账：只记预注册里列出的量 |
| `results/bench_cpu.md` | baseline 在 4 核 CPU 上的吞吐实测（安静机器），以及格 0、格 6 的预算表 |
| `reports/` | 进展报告（`2026-11-18_progress.md`，开头是需要 Z 决定的事）；外部评估（`2026-11-17_external_review.md`）；Wright–Fisher 的 RK4 分支离开单纯形的修复记录（`2026-11-17_wf_simplex_fix.md`）；给 Z 的第一周报告（`2026-11-04_week1.md`），以及写给 handoff 共同作者的详细结果报告（`2026-11-04_advisor_report.md`）；handoff 作者的回信（`2026-11-04_reply_from_handoff_author.md`） |

## 环境

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

CPU 版 PyTorch 可以改用 `--index-url https://download.pytorch.org/whl/cpu`。代码不区分设备，在 CPU、MPS（Mac）、CUDA 上都能跑。

## 测试

```bash
.venv/bin/python -m pytest
```

动物园的每条真值都由两套互相独立的测试核对：builder 写的，以及审稿人刻意换了方法写的 `tests/test_zoo_independent.py`，后者还做过变异测试。

## 动物园 API（`zoo/base.py`）

```python
import zoo
w = zoo.make("pendulum")                      # 另有 "toggle_grid"、"wright_fisher"
tr = w.sample(seed=0, horizon=64, n_traj=100, obs_mode="pixels64")   # 参数可以按次覆盖
gt = w.ground_truth(**tr.meta["params"])     # 真值依赖采样配置，例如 region_betti、region_b1_wrapping
```

- `Trajectories.state` 与 `.info` 只供分析使用，模型只能看到 `.obs` 和 `.actions`。
- 数据划分用互不相交的种子，见 `zoo.base.SPLIT_SEEDS`。

主要约定：
- 单摆：θ = 0 朝下，阻尼写成 θ'' = −ω0² sin θ − 2γθ'，所以散度为 −2γ；观测有 `angle_cs`、`angle_raw`（在 ±π 处有意撕开）和 `pixels64`。
- Wright–Fisher：采用 Amari 约定，α = +1 为 e、α = −1 为 m；Koide & van de Venn 用的是相反的号。
- toggle_grid：和乐等于开关进入次数的奇偶；`switches_enabled=False` 给出无开关对照，`switch_visible=False` 给出开关不可见的观测。

## 工作约定（摘自 HANDOFF §0、§5）

1. 每个 kill test 跑之前，先把预注册（预测、对照、判据、预算）给 Z 看，**Z 点头再跑**。例外（Z 2026-11-04）：格 0、格 6 的预注册逐条实现审稿清单、sha256 冻结、并经只核真值的审稿 agent 确认后，视为已批；清单外的新增或偏离仍须 Z 点头。
2. 结果记进 `results/ledger.md`，再更新 `cells.yaml` 的 status / confidence，并写明理由。
3. 每格结束写一段大白话结论。
