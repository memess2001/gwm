# 实验流水账（ledger）

规则：每次跑完一个配置追加一行；只记录预注册里列出的量；判定（verdict）只能是 prereg 里定义过的类别。
日期用实际运行日期。`params` 为实际参数量。

| date | cell | world | obs | variant | size | params | seed | metrics | verdict | notes |
|---|---|---|---|---|---|---|---|---|---|---|

## 预注册冻结记录

（用列表而不是表格：`baselines/ledger.py` 约定本文件只有一张表，即上面的实验流水账。）

- 2026-11-12 JST · 格 6 · prereg v1 · commit 8578e4b。审稿不确认（`cells/6/review_v1.md`），未运行任何东西，由 v1.1 取代。

完整的 sha256（格 6 v1，commit 8578e4b）：

```
0f3e892549348c597e6857bda18d6253950769420940686953a80bdbf43f4429  cells/6/prereg.md
690b707d44a1c0aa96cf01ef4f13471acc2a8580e692ac027e71a368b5f75106  analysis/holonomy.py
2728acffaa55d76748f1d0d9a22906383289826b3fa24c413fdecb1932e3cfc5  cells/6/c6lib.py
fec7f1f4e38e1c503460cb9a064da6d4968cd38329240c62d6bfa66632fa1943  cells/6/c6_train.py
79b95bfd0215223fd8c65bb8b906830d5b5258547ca1e22d7b29804cdae55e7e  cells/6/c6_score.py
f82c6b27fd335f1150b5424fe027afc3396f18c61383a77c9a4a13c8d8021e29  cells/6/manipulation_check.json
```

- 2026-11-12 JST · 格 6 · prereg v1.1 · commit 7899475。按 v1 审稿意见修订，修订记录见 `cells/6/prereg.md` §13。sha256：

```
bf3c76c8ad314e094a79cdfe86b5c04b4088ba6e4154423f7ceb97fe32b564bc  cells/6/prereg.md
c19cacc5c98a9654ddc8476a5ea2177caf8de3ab7bc1d002d45d53b062f2a970  analysis/holonomy.py
10a2655161e0371fd8a80557636570a9cca1b1c98547586ff014d68c634a5607  cells/6/c6lib.py
fec7f1f4e38e1c503460cb9a064da6d4968cd38329240c62d6bfa66632fa1943  cells/6/c6_train.py
713a13466937fd561ec28983263b49f04c75c4d4925ff449676f240ad2437c6c  cells/6/c6_score.py
f82c6b27fd335f1150b5424fe027afc3396f18c61383a77c9a4a13c8d8021e29  cells/6/manipulation_check.json
```

- 2026-11-12 JST · 格 6 · prereg v1.2 · commit 664b554。按第二轮审稿（`cells/6/review_v1_1.md`）修订，见 `cells/6/prereg.md` §13。v1.1 未运行任何东西。sha256：

```
be4a84c7a43defc310fdcce689c9261c9b28c3191b4c16f4d81038c577287db3  cells/6/prereg.md
c19cacc5c98a9654ddc8476a5ea2177caf8de3ab7bc1d002d45d53b062f2a970  analysis/holonomy.py
10a2655161e0371fd8a80557636570a9cca1b1c98547586ff014d68c634a5607  cells/6/c6lib.py
6b7ee961dad246774e314315d4580260ed47f7f1d3d2049c0e226413073a5663  cells/6/c6_train.py
7dc670cdd905ea54c0591da402ba841988159a2f9236928303bd3b54f9f1ff5c  cells/6/c6_score.py
f82c6b27fd335f1150b5424fe027afc3396f18c61383a77c9a4a13c8d8021e29  cells/6/manipulation_check.json
```

- 2026-11-12 JST · 格 6 · prereg v1.2 的审稿确认：`cells/6/review_v1_2.md`（结论：确认；预先授权第 3 条）。v1.1、v1.2 都是冻结后的修订，按 `decisions` 仍需 Z 认可后才能开跑。prereg 状态行里引用的确认记录应读作本文件（prereg 已冻结，不再改）。

## 格 6 pilot 记录（只有训练损失与 gate；预注册 §3）

- 2026-11-12 JST · H-act · S · 种子 99 · 10k 步：位置 gate 1.000；门 gate n = 0 / 1 / 2 / 3 为 1.0 / 0.0 / 1.0 / 0.0 → 不通过。val 损失从 6k 步起停在约 0.009。
- 2026-11-12 JST · H-act · S · 种子 99 · 20k 步：位置 gate 1.000；门 gate 1.0 / 0.0 / 1.0 / 0.0 → 不通过。val 损失 0.0088。
- 按 §3 的规则：**停**。模型学成了"门永远关着"。真值核算：H-act 训练数据里只有 2.0% 的帧（5.75% 的 episode）受门的状态影响。改数据或模型都属清单外偏离，交 Z 决定。没有计算任何读数。

- 2026-11-14 JST · 格 6 · prereg v1.3 · commit b30ff8f（Z 的决定：p_home 0.1；P6.5 降为描述）。sha256：

```
1db5e5d665c01bf4fc6904034e1afcc4e8748094ab0659f828b12de63f94f2d3  cells/6/prereg.md
c19cacc5c98a9654ddc8476a5ea2177caf8de3ab7bc1d002d45d53b062f2a970  analysis/holonomy.py
10a2655161e0371fd8a80557636570a9cca1b1c98547586ff014d68c634a5607  cells/6/c6lib.py
6b7ee961dad246774e314315d4580260ed47f7f1d3d2049c0e226413073a5663  cells/6/c6_train.py
2b44011f0acc8fde64ae1b9668f4ced1469daab556747582df82f31bfb71d117  cells/6/c6_score.py
23ef3d078d41d115dcdc7ba44ecf9cf13b4328783e493ffcf7263b8e6c211b26  cells/6/manipulation_check.json
```
- 2026-11-14 JST · 格 6 · prereg v1.3 的审稿确认：`cells/6/review_v1_3.md`（结论：确认）。三处非阻塞的文字问题不改冻结文件，记在这里：(1) "受门状态影响的帧"在训练种子上，p_home 0.01 为 2.0%，p_home 0.1 为 7.2%（§2 的 7.7% 来自种子 777）；定义是"门永远关着"的 rollout 与真值不符的帧；(2) 状态行应指向 `review_v1_3.md`；(3) 对照表第 9 行"P6.5 可被证伪"在 v1.3 下应读作"P6.5 为描述"。
- 2026-11-14 JST · 格 6 v1.3 pilot · H-act · S · 种子 99 · 10k 步（p_home 0.1）：位置 gate 1.000；门 gate n = 0 / 1 / 2 / 3 为 1.0 / 0.0 / 1.0 / 0.0 → 不通过。val 损失 0.025。按 §3 接着跑 20k 步的 pilot。
- 2026-11-14 JST · 格 6 v1.3 pilot · H-act · S · 种子 99 · 20k 步（p_home 0.1）：位置 gate 0.999；门 gate n = 0 / 1 / 2 / 3 为 1.0 / 0.035 / 0.96 / 0.03 → 不通过。val 损失 0.0206（10k 时 0.025）。按 §3：**停**，交 Z 决定。没有计算任何读数。
- 2026-11-14 JST · 格 6 · Z 决定先用 M 尺寸跑 20k 步 pilot（数据与其余设定不动）；结果只看训练损失与 gate。
- 2026-11-14 JST · 格 6 v1.3 设定下的 M 尺寸 pilot · H-act · M（50.0 万参数）· 种子 99 · 20k 步（p_home 0.1）：位置 gate 1.000（n = 24109）；门 gate n = 0 / 1 / 2 / 3 为 1.0 / 1.0 / 1.0 / 1.0（各 200 个事件）→ **通过**。val 损失 4k / 8k / 12k / 16k / 20k 步为 0.0244 / 0.0200 / 0.0016 / 0.00032 / 0.00021（S 在 20k 步停在 0.0206）。单线程、与格 0 的 3 个 job 同机，用时 6638 s。没有计算任何读数。按 Z 的决定，下一步是 v1.4：主实验只用 M。

- 2026-11-14 JST · 格 6 · prereg v1.4 · commit 69ebea5（Z 的决定：主实验只用尺寸 M，20k 步；21 个 run）。sha256：

```
6259766a9746437b10a0b9b9551086474f0bded56f86fd915c32761ebfb35fe7  cells/6/prereg.md
c19cacc5c98a9654ddc8476a5ea2177caf8de3ab7bc1d002d45d53b062f2a970  analysis/holonomy.py
c91b3543d0053e1b3d9330dd764539d3186cd333133c0d28326a282eff72960e  cells/6/c6lib.py
6b7ee961dad246774e314315d4580260ed47f7f1d3d2049c0e226413073a5663  cells/6/c6_train.py
42c5bb2610c103eaa6f1d3505f42be86d97d1253d3863d147e26a9a388784f05  cells/6/c6_score.py
23ef3d078d41d115dcdc7ba44ecf9cf13b4328783e493ffcf7263b8e6c211b26  cells/6/manipulation_check.json
```
- 2026-11-14 JST · 格 6 · prereg v1.4 的审稿确认：`cells/6/review_v1_4.md`（结论：确认，无阻塞）。非阻塞问题不改冻结文件，记在这里：
  - (1) §13 "v1.4 不会多判出死"以 S 为 NE 为前提，这个前提是反事实的（照 v1.3 字面，S 的 pilot 不过就停）。不带前提时，16 格里有 7 格判定不同，两个方向都有。v1.4 新开放的最终状态是 alive 与 merged_into:HCDP。
  - (2) Z 认可时应明确看过 §13 "判据变宽松了"一段。
  - (3) 同尺寸、同步数的 H-act pilot 若有两个，`protocol_check` 的结果取决于目录遍历顺序；现在只有一个 M 的 20k pilot（种子 99），不受影响。
  - (4) `steps_used` 把清单外的 run 也算进去，方向保守。
  - (5) `c6_queue --steps` 原来必须手填；已改为默认取 `MAIN_STEPS`，填别的值就拒绝。`c6_queue.py` 不在冻结之列。
  - (6) 文字遗留：`c6_train.py`、`c6_score.py` 的文档示例仍是 S、10k；状态行与 §10 仍指向 `review_v1.md`；日期行仍是 2026-11-12。
- 2026-11-14 JST · 格 6 · Z 认可 prereg v1.4（看过 §13 "判据变宽松了"一段），开跑 21 个 run。机器在 14:32 UTC 重启过一次，格 0 阶段 1 的队列随之中断（3 个跑到一半的 L run 会按队列规则重训），一并重启。
- 2026-11-15 JST · 格 6 · 结果（prereg v1.4，commit 69ebea5，只有尺寸 M）。计分前重新核对冻结的 6 个文件，sha256 全部一致。21 个 run 全部完成，步数 20k，完整性通过。H-act 种子 2 没过门 gate（门正确率 0.905 / 0.555 / 0.69）；P6.1、P6.2、P6.3 各 2/3 成立 → M 为 PASS → 读数有效。P6.6 没有胜出：AUC 差的中位数 0.012（门槛 0.05），precision@4 差 0（门槛 0.25），N_req 相同（探测 2000、定位 500）。按 §7，`cells.yaml` 格 6 status → merged_into:HCDP；格 12 notes 记"定位到来源而非后果"。限定：这个最终状态只在 v1.4 下可达（审稿非阻塞问题 (1)）；τ = 0（种子 0、1）；C-vis 种子 1、2 的模型检查 0.94。在云端跑，容器重启 3 次，中断的 run 从头重训。结果：`cells/6/result.md`、`cells/6/result_data.json`；逐 run 输出备份在分支 `runs-backup/2026-11`。
- 2026-11-16 JST · 格 2 · thought_killed（Z 的决定，没有跑算力）。写预注册前的思想实验（`cells/2/thought_experiment_2026-11-16.md`）：WF 噪声协方差是 Shahshahani 度量的逆，所以平稳、带突变时 Fisher-Hodge 残差 × 2N_e = 熵产生率（暂态时 = housekeeping 熵产生），按 R2 与同源熵产生打平；对欧氏 HHD 的胜利是定理（R1）。独立核对（`cells/2/review_thought_experiment.md`，只核真值、不训练）：有保留地确认，格点 Fokker–Planck 上两者相差约 0.01%；更正了一处出处（Bürger 的页码）与 SFI 的描述，补了恒等式的适用条件。唯一真实的分岔（非 Fisher 型噪声）由定理决定，不作为保留格 2 的理由。
