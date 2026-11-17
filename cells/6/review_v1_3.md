结论：确认

- 审稿：独立审稿 agent，2026-11-14 JST，按 `rules.R4` 只核真值。没有训练任何模型；`runs/` 下没有读任何东西，归档 pilot 的情形用伪造目录模拟；所有计算都是单线程（OMP_NUM_THREADS=1）。冻结文件没有改动。
- 对象：prereg v1.3。修订内容在 commit b30ff8f，账本记录在 3188dbb。
- 改动的授权：这次改动有 Z 的明确决定（`cells.yaml` 的 `decisions`，2026-11-14：p_home 改为 0.1，放弃覆盖匹配，P6.5 降为描述），不需要另行请示。
- 脚本与输出：`/tmp/claude-0/-home-user-gwm/8bc92419-2f30-547d-8b18-8324b70c6f75/scratchpad/c6review/v13/`。

## 核查

| 项 | 结果 | 证据 |
|---|---|---|
| sha256 | 相符 | 六个冻结文件的 sha256 与 `results/ledger.md` 中 v1.3 一节逐一相同。`c6lib.py`、`c6_train.py`、`holonomy.py` 与 v1.2 相同，所以查询集和 gate 集都不变。 |
| JSON 的数字可复现 | 是 | 用 `c6lib.sample_policy` 与 `coverage` 在训练种子上、按 JSON 里的 p_home（0.1）重算，`coverage_train_seed` 的 7 行（H-act 的 n = 2000 / 500 / 250，以及 H-pc、H-pn、C-open、C-vis）逐项相同。`p_home.previous` 保留了 v1.2 的匹配网格与结果 0.01。 |
| 训练实际用的 p_home | 正确 | `c6_train.p_home()` 读 JSON 得到 0.1。`main` 里的写法 `p_home() if policy == "active" else 0.0` 给出：H-act、C-open、C-vis 用 0.1，H-pc、H-pn 用 0.0。 |
| 判定表没有别的改动 | 是 | `c6_score.py` 只改了 P65 这一段，现在只输出描述性字段：`descriptive_only`、`act_minus_pc`、`pn_red_flag`。P65 本来就不进判定。重跑上一轮全部伪造 run 检查，结果与 v1.2 完全一致：16 种 (S, M) 组合与 §7 相符；缺 run → INCOMPLETE；对照没学会 → INCONCLUSIVE；对照学会了但读数有噪声 → READOUT_DEAD；τ = 1 → P6.3 不成立；偶数环路只报告；零环路参与判定；pilot 步数规则的六种情形都对。40 个测试全过。 |
| 归档的 pilot | 不会干扰 | `protocol_check` 的 `glob("*_pilot*")` 不递归进子目录，`_archive_v12` 这个名字本身也不匹配；`aggregate` 会跳过没有 `c6_run.json` 的目录。用伪造目录模拟（`_archive_v12/` 下放两个没过 gate 的 pilot）：<br>• 没有新 pilot → INCOMPLETE，`pilots = {}`；<br>• 新的 10k pilot 通过 → 按 10k 出判定，归档的 pilot 都看不到。 |
| 文字一致性 | 基本一致 | §2 的表、§6 的 P6.5、§11、§13 的 v1.3 表、实现选择清单都已更新，与代码和 JSON 一致。下面列出三处小的文字问题。 |

## 剩下的问题

**阻塞：** 无。

**非阻塞：**
1. "受门状态影响的帧" 前后两个数不是在同一个种子上算的。按"'门永远关着'的 rollout 与真值位置不同的帧"这个定义，我在训练种子、p_home 0.01 下复现了账本里的 2.0% 帧 / 5.75% episode。同一定义下，p_home 0.1 在训练种子上是 7.2%，在匹配种子 777 上是 7.7%。§2 与 §13 写的"约 7.7%"用的是 777，而 2.0% 用的是训练种子。建议改为 7.2%，或注明种子，并写明这个定义。
2. `protocol_check` 不核对 pilot 和正式 run 的 p_home（`c6_run.json` 里有这个字段）。现在靠把 v1.2 的 pilot 移进归档目录来隔离。以后再改 p_home 时，可以让 `aggregate` 顺带核对，但不是必须。
3. 文字小节：
   - 日期行仍是 2026-11-12；
   - 状态行仍把确认记录指向 `review_v1.md`（§13 已指向本文件）；
   - 文末对照表第 9 行仍写"P6.5 可被证伪"与 `choose_p_home`。这是 Z 已批准的偏离，建议在该行加注"v1.3 起为描述"。
