结论：确认

- 审稿：独立审稿 agent，2026-11-14 JST，按 `rules.R4` 只核真值。没有训练任何模型；`runs/` 下没有读任何东西，pilot 与正式 run 一律用伪造目录模拟；所有 Python 都是单线程（`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`）。没有改动任何已提交的文件。
- 对象：prereg v1.4。冻结 commit 是 69ebea5，账本记录在 31c3b27。对照的上一版是 v1.3（b30ff8f）。
- 改动的授权：`cells.yaml` 的 `decisions`（2026-11-14，`cell_6_after_failed_v1_3_pilots`）写明："M 尺寸跑一次 20k 步 pilot；过了就把主实验改成只用能学会的尺寸（v1.4，重新冻结、审稿）"。M pilot 通过的记录在 cb2521c。v1.4 的改动没有超出这条决定的范围。
- 脚本与输出：`/tmp/claude-0/-home-user-gwm/8bc92419-2f30-547d-8b18-8324b70c6f75/scratchpad/c6review/v14/`。
  - `fab.py`：伪造目录的生成器，自己写的，不用测试里的 helper。
  - `make_worlds.py`：生成 94 个伪造目录。
  - `runner.py`：在指定源码树上跑 `aggregate`。
  - `out_*.json`、`out_t5_compare.txt`：输出。
  - `tree_v13/`、`tree_v14/`：用 `git archive` 从 b30ff8f 与 69ebea5 导出的源码树。v1.3 与 v1.4 的代码各在自己的进程里跑。

## 核查

| 项 | 结果 | 证据 |
|---|---|---|
| 1. sha256 | 相符 | 六个文件在 69ebea5 与当前工作树上的 sha256，与 `results/ledger.md` 的 v1.4 一节逐一相同：<br>• `prereg.md` 6259766a…<br>• `holonomy.py` c19cacc5…<br>• `c6lib.py` c91b3543…<br>• `c6_train.py` 6b7ee961…<br>• `c6_score.py` 42c5bb26…<br>• `manipulation_check.json` 23ef3d07…<br>其中 `holonomy.py`、`c6_train.py` 与 `manipulation_check.json` 与 v1.3 相同：`git diff` 为 0 行。69ebea5 之后只改过 `results/ledger.md`。<br>`prereg_v1_3.md` 与 `git show b30ff8f:cells/6/prereg.md` 逐字节相同，sha256 为 1db5e5d6…，就是账本里 v1.3 的值。 |
| 2. §13 v1.4 表的落实 | 全部落实，没有表外的实质改动 | 逐行核对，见下文"第 2 项"。 |
| 3. 判定表（伪造目录） | 与 §7 一致 | 31 个伪造目录。<br>• M 的四种状态都按 §7 判定。每种状态用了 2–4 种不同的构造。<br>• 缺任何一个 run → INCOMPLETE。<br>• P6.6 只在 M 上判。<br>细节见下文"第 3 项"。 |
| 4. 步数规则（`protocol_check`） | 正确 | 24 个伪造目录，题目要求的 7 种情形都按 §3 的规则判定，另外还测了 10 种情形。细节见下文"第 4 项"。<br>另发现一个不影响现状的问题：同尺寸、同步数的 pilot 有多个时，结果取决于目录的遍历顺序（非阻塞第 3 条）。 |
| 5. §7 与 §13 的对照说法 | 在所述前提下成立 | v1.3 代码在伪造目录上跑出的 16 种 (S, M) 组合，与 `prereg_v1_3.md` 的表完全一致。换一套构造再跑 16 种，结果也一致。<br>S 为 NE 时，两版只在 M 为 PASS 时不同。<br>"判据变宽松"那一段对 P6.6 的描述，也由实跑结果证实。<br>前提不成立时的情况见非阻塞第 1 条。 |
| 6. 正文残留"两个尺寸" | 没有与代码矛盾的残留 | 起作用的各节已改为只用 M。剩下提到两个尺寸的三处，都明确标成了历史或对照（§7 末段、§11 第三条、附录"实现时做的选择"第 6 条）。注释与文字上的小遗留见非阻塞第 6 条。 |
| 7. 测试 | 全部通过 | 单线程运行 `pytest tests/test_c6_decision.py tests/test_holonomy.py`：43 passed（19 + 24），用时 12.3 s。 |

### 第 2 项：§13 v1.4 表逐行核对

| §13 表中的改动 | 代码 / 测试 | 核对 |
|---|---|---|
| 主实验只用 M；`c6lib.SIZES`；`c6_queue.run_list` | `c6lib.SIZES = ("M",)`；`run_list` 按 `c.SIZES` 循环 | `set(required_runs()) == set(run_list())`，都是 21 个 run，尺寸只有 M，条件为 5 个，n 取 {250, 500, 2000} |
| 全部 run 用 20k；`c6lib.MAIN_STEPS`；`protocol_check`；`test_s_pilot_does_not_set_the_step_count`、`test_runs_at_10k_steps_are_incomplete` | `MAIN_STEPS = 20000`；`protocol_check` 以 `"{size}_{steps}"` 为键，只认 H-act，只认 `SIZES` 里的尺寸 | 两个测试都在，都通过；第 4 项的伪造目录也逐一证实 |
| 完整性 21 个 run；读数层只看 M；P6.6 在 M 上任一路线胜出即可；`required_runs`、`aggregate`；`test_v14_needs_21_runs_and_no_s_runs`、`test_checked_failure_on_m_is_dead` | `required_runs` 返回 21 个；`aggregate` 的 `states`、`p66`、`geo` 都按 `c.SIZES` 循环；`geo = any(all(p66[s][m] for s in SIZES) for m in metrics)`，只有 M 时就是"M 上任一项胜出" | 两个测试都在，都通过；第 3 项证实 |
| §3、§6、§7、§9 的文字 | — | 已改。21 × 6638 s ≈ 38.7 单线程机时，4 路并行约 9.7 小时，与 §9 的"约 39 / 约 10 小时"相符 |

表外的改动（都与上表一致，不是实质性的新增）：
- 新增 `prereg_v1_3.md`。状态行里提到了，是 v1.3 的原文。
- §11 加了"只有一个尺寸"一条风险；附录"实现时做的选择"第 6 条加注"不再起作用"；P6.5 的"每个尺寸报告"改为"报告"。
- `c6_score`：P65 也按 `SIZES` 循环；`aggregate` 里 `elif "PASS" in states` 在只有一个尺寸时永远走不到，无害。
- `c6_queue`：文档字符串改了，删去"S before M"。
- 测试：
  - `_write` 的 pilot 目录名改为 `_pilot{steps}`；`_world` 增加了 `pilot_size`、`pilot_steps` 两个参数；
  - `test_size_disagreement_is_inconclusive` 删去，由 `test_checked_failure_on_m_is_dead` 取代。只剩一个尺寸后，这个测试已无对象；
  - 其余测试只是把 S 换成 M。
  - `test_holonomy.py` 没有改动。

### 第 3 项：判定表（v1.4 代码，伪造目录，M 的 20k pilot 通过，全部 run 用 20k 步）

| 伪造的 M 状态（构造方式） | 实得状态 | 判定 | status | §7 |
|---|---|---|---|---|
| PASS（三个种子全部成立） | PASS | READOUT_WORKS | merged_into:HCDP（与基线打平） | 读数有效 ✓ |
| FAIL_CHECKED：P6.1 在 3 个种子上不成立 | FAIL_CHECKED | READOUT_DEAD | dead | 读数死 ✓ |
| FAIL_CHECKED：P6.2 在 2 个种子上不成立 | FAIL_CHECKED | READOUT_DEAD | dead | ✓ |
| FAIL_CHECKED：本 run 零环路超阈，P6.3 不成立（2 个种子） | FAIL_CHECKED，`not_specific` | READOUT_DEAD | dead | ✓ |
| FAIL_CHECKED：对照学会了，但读数不特异 | FAIL_CHECKED，`not_specific` | READOUT_DEAD | dead | ✓ |
| FAIL_UNCHECKED：P6.1 不成立，但 H-act 模型检查 0.5 | FAIL_UNCHECKED | INCONCLUSIVE | unchanged | 不确定 ✓ |
| FAIL_UNCHECKED：一个种子已检查地不成立，另一个种子 gate 不过 | FAIL_UNCHECKED | INCONCLUSIVE | unchanged | ✓ |
| FAIL_UNCHECKED：对照不特异，但对照的模型检查 0.3 | FAIL_UNCHECKED | INCONCLUSIVE | unchanged | ✓ |
| NE：gate 为 [T, F, F] | NE | INCONCLUSIVE | unchanged | 不确定 ✓ |
| NE：gate 为 [F, F, F] | NE | INCONCLUSIVE | unchanged | ✓ |

- **INCOMPLETE。** 每次只拿掉 21 个 run 中的一个，分别试了：
  - H-act s0、C-open s1、C-vis s2、H-pc s0、H-pn s1、H-act n250 s2、H-act n500 s0；
  - 只删 `scores.json`（H-pn s2）；只删 `gate.json`（C-vis s0）。

  九次都判 INCOMPLETE，`missing_runs` 都恰好列出被拿掉的那一个。目录里没有任何 S 的 run 时判定照常完成，不会把 S 记为缺失。
- **P6.6 只在 M 上判。** 所有输出里 `per_size`、`P66`、`P65` 的键都只有 `["M"]`。M 为 PASS 时：
  - 与基线打平 → merged；
  - ΔAUC = 0.08 → alive；ΔAUC = 0.04 → merged（没达到 0.05 的幅度）；
  - Δprec = 0.25 → alive；
  - 探测数据效率胜出、定位数据效率胜出 → 各自都是 alive。四条路线每一条单独成立都判 alive；
  - M 为 FAIL_CHECKED 而 AUC 胜出时 → dead，`geo = None`。
- **目录里另有 S 的 run（20k 步）时，判定不受影响：**
  - S 为 PASS 且胜出、M 为 FAIL_CHECKED → READOUT_DEAD；
  - S 为 FAIL_CHECKED、M 为 PASS → READOUT_WORKS；
  - S 胜出、M 打平 → merged；
  - S 打平、M 胜出 → alive。

### 第 4 项：步数规则（v1.4 代码；正式 run 都伪造成 M 为 PASS）

| 情形 | pilots | steps_by_rule | steps_used | 判定 |
|---|---|---|---|---|
| a. M 的 20k pilot 通过 | {M_20000: T} | 20000 | [20000] | READOUT_WORKS ✓ |
| b. M 的 20k pilot 不通过 | {M_20000: F} | None | [20000] | INCOMPLETE ✓ |
| c. 只有 S 的 20k pilot，且通过 | {S_20000: T} | None | [20000] | INCOMPLETE ✓ |
| c'. 只有 S 的 10k pilot，且通过（正式 run 为 20k 或 10k） | {S_10000: T} | None | [20000] / [10000] | INCOMPLETE ✓ |
| d. S 的 20k pilot 通过、M 的不通过 | {S_20000: T, M_20000: F} | None | [20000] | INCOMPLETE ✓ |
| d'. S 的 10k 与 20k 都通过、M 的 20k 不通过 | 同上，另有 S_10000: T | None | [20000] | INCOMPLETE ✓ |
| e. 正式 run 全用 10k（M 的 20k pilot 通过） | {M_20000: T} | 20000 | [10000] | INCOMPLETE ✓ |
| e'. 另有一个通过的 M 10k pilot，正式 run 用 10k | {M_10000: T, M_20000: T} | 20000 | [10000] | INCOMPLETE ✓（v1.4 没有 10k 这条路） |
| f. 步数混用：5 个 run 用 10k | {M_20000: T} | 20000 | [10000, 20000] | INCOMPLETE ✓ |
| f'. 只有一个数据效率 run 用 10k；或只有一个对照 run 用 30k | {M_20000: T} | 20000 | [10000, 20000] / [20000, 30000] | INCOMPLETE ✓ |
| g. 只有 C-open 的 M 20k pilot，且通过；只有 H-pn 的，且通过 | {} | None | [20000] | INCOMPLETE ✓ |
| g'. H-act 通过 + C-open 不通过（同为 M、20k） | {M_20000: T} | 20000 | [20000] | READOUT_WORKS ✓（非 H-act 被跳过，不会覆盖） |
| g''. H-act 不通过 + C-vis 通过 | {M_20000: F} | None | [20000] | INCOMPLETE ✓ |
| 实际历史：S 10k 不过、S 20k 不过、M 20k 过 | 三项都在 | 20000 | [20000] | READOUT_WORKS ✓ |
| 只有 M 的 10k pilot，且通过 | {M_10000: T} | None | [20000] | INCOMPLETE ✓ |
| 没有 pilot；或 pilot 放在 `_archive_*` 子目录里 | {} | None | [20000] | INCOMPLETE ✓（glob 不递归） |
| 另有不在清单里的 S 正式 run，步数为 10k | {M_20000: T} | 20000 | [10000, 20000] | INCOMPLETE（保守，见非阻塞第 4 条） |
| 两个 H-act、M、20k 的 pilot（种子 98 与 99），一个过一个不过 | 取遍历顺序里最后那个 | 视顺序而定 | [20000] | 两种组合分别得到 INCOMPLETE 与 READOUT_WORKS（见非阻塞第 3 条） |

### 第 5 项：v1.3 与 v1.4 的对照

做法：每种 (S, M) 组合各造一个目录，包含 S 与 M 两套 run（都是 20k 步）和三个 pilot（S 10k 不过、S 20k 过、M 20k 过）。S 的 20k pilot 记为通过，只是为了让 v1.3 的步数规则放行，使 v1.3 的代码能走到判定表；v1.4 不看 S 的 pilot。同一个目录分别交给 `tree_v13`（b30ff8f 的 `c6_score.py`）与 `tree_v14` 的代码。

单元格写法是"v1.3 实跑 → v1.4 实跑"，加粗的是两版不同的格：

| S \ M | PASS | FAIL_CHECKED | FAIL_UNCHECKED | NE |
|---|---|---|---|---|
| **PASS** | 有效 → 有效 | 不确定 → **死** | 不确定 → 不确定 | 不确定 → 不确定 |
| **FAIL_CHECKED** | 不确定 → **有效** | 死 → 死 | 死 → **不确定** | 死 → **不确定** |
| **FAIL_UNCHECKED** | 不确定 → **有效** | 死 → 死 | 不确定 → 不确定 | 不确定 → 不确定 |
| **NE** | 不确定 → **有效** | 死 → 死 | 不确定 → 不确定 | 不确定 → 不确定 |

- **v1.3 代码与 v1.3 预注册一致。** v1.3 代码的 16 个输出与 `prereg_v1_3.md` §7 的 (S, M) 表逐格相同。换一套构造再跑一遍，16 格也逐格相同。第二套构造是：FAIL_CHECKED 用"对照学会了、读数不特异"，FAIL_UNCHECKED 用"一个种子已检查、一个种子 gate 不过"，NE 用三个种子的 gate 全不过。
- **S 为 NE 那一行。** 只有 M 为 PASS 那一格两版不同（不确定 → 有效），其余三格相同。这与 §7 末段、§13"读数层"一段的说法一致。在这个前提下，"v1.4 不会多判出死，只会多判出有效"成立。
- **几何层（P6.6），实跑结果：**
  - S、M 都为 PASS，S 只在 precision@4 与定位数据效率上胜出，M 只在 AUC 与探测数据效率上胜出：v1.3 判 merged（没有同一项指标在两个尺寸上都胜出），v1.4 判 alive。
  - 只有 M 胜出：v1.3 判 merged，v1.4 判 alive。
  - 只有 S 胜出：两版都判 merged。
  - 同一项指标在两个尺寸上都胜出：两版都判 alive。

  这证实了 §13 的说法："四选一 v1.3 就有，少的是跨尺寸的一致性要求"。
- **S 为 NE 时的最终 status。** S 为 NE、M 为 PASS 时，v1.3 判 `unchanged (inconclusive)`；v1.4 在 M 胜出时判 alive，在 M 与基线打平时判 merged_into:HCDP。
- **完全按 v1.3 的字面执行时**（S 的两次 pilot 都不过），v1.3 代码判 INCOMPLETE，也就是停下，不论 M 的结果如何。

## 剩下的问题

**阻塞：** 无。

**非阻塞：**

1. **§13"读数层"一段的前提。** 这一段的结论只在"S 的状态为 NE"这个前提下成立，§13 也写明了这个前提。不过这个前提本身是反事实的：照 v1.3 的字面，S 的 pilot 不过就停，根本不会有判定（实跑 INCOMPLETE）；而 S 的主种子会不会全是 NE，并没有人测过。去掉这个前提，16 格里有 7 格两版不同，而且方向不一：
   - (S PASS, M FAIL_CHECKED)：不确定 → 死；
   - (S FAIL_CHECKED, M FAIL_UNCHECKED 或 NE)：死 → 不确定；
   - S 不为 PASS、M 为 PASS 的三格：不确定 → 有效。

   S 两次 pilot 都没过，所以 S 为 PASS 的可能性很小，这不影响确认。

   另外，在最终 status 这一层，v1.4 新开放的不只是 alive，还有 merged_into:HCDP（几何主张被否）。所以 v1.4 不只是更宽松，也更容易下结论。建议在 `result.md` 或账本里补一句。
2. **授权的字面范围。** Z 的决定写的是"只用能学会的尺寸"。读数层的放宽、P6.6 不再要求跨尺寸一致，都是只剩一个尺寸的必然结果，§13 也如实写了，但决定原文没有逐项提到。建议 Z 认可时明确看过 §13"判据变宽松"那一段。这不影响本次确认。
3. **`protocol_check` 没有核对 pilot 的种子与 p_home。**
   - §3 写的是"种子 99"，但代码接受任何种子。
   - 同尺寸、同步数的 H-act pilot 有两个时，字典键冲突，结果取决于遍历顺序里最后那个：实测种子 98 与 99 一过一不过时，两种组合分别得到 INCOMPLETE 与 READOUT_WORKS。
   - p_home 不核对，这一点上一轮已经提过。

   按账本，现在只有一个 M 的 20k pilot，所以不影响现状。以后要改，可以只认种子 99，或者要求同键的 pilot 全部通过。
4. **`steps_used` 统计的是目录里所有已计分的非 pilot run，包括不在 21 个之内的 run。** 例如有 10k 步的 S 正式 run 时，就会判 INCOMPLETE。这个方向是保守的，不会错判，这里只记录行为。
5. **`c6_queue.py` 的 `--steps` 必须手填，没有默认取 `c6lib.MAIN_STEPS`，也不做核对。** 如果误填 10000，要白跑约 39 个单线程机时，`aggregate` 才会判 INCOMPLETE。`c6_queue.py` 不在冻结文件之列，可以改成默认 `MAIN_STEPS` 并断言相等，不算偏离。
6. **文字与注释的小遗留**（都不影响代码行为）：
   - `c6_train.py` 的文档示例仍是 `--size S --steps 10000`，`--size` 仍接受 S。这个文件冻结了、本次没动，`aggregate` 也不看 S。
   - `c6_score.py` 的模块文档仍写"v1"，示例是 `runs/c6/H-act_S_s0`。
   - prereg 的状态行与 §10 第 2 条仍把确认记录指向 `review_v1.md`；§10 第 3 条"审稿确认之后跑 pilot"不再适用于 v1.4，v1.4 的 pilot 在冻结之前就跑完了，这是 Z 的决定。
   - 日期行仍是 2026-11-12。
   - v1.3 审稿记下的勘误（§2 与 §13 的"约 7.7%"用的是种子 777；附录第 9 行"P6.5 可被证伪"）原样带进了 v1.4，账本里 v1.3 那条勘误仍然适用。
   - 测试没有覆盖三种情形：步数混用、非 H-act 的 pilot、S 过而 M 不过。本次已经用伪造目录逐一核过，以后可以补成测试。
