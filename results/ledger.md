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

## 预注册冻结记录（格 0）

- 2026-11-12 JST · 格 0 · prereg v2 · commit ca9c154。sha256：

```
5c8fdaa719569589ba7f3d95475c31bdc936ec2748845474f10f356f5e12361d  cells/0/prereg.md
fb94ae6cea460e0a2bc61f79c891572482d84811a1feea76947175a11607fbdd  analysis/seam.py
94af7c0276eee11ebda7a5ae46fb0dba23457029e3af7f93f6f2fa44090e8b63  cells/0/c0lib.py
53fff637cd01e651d376df0973427b4bfcd32927d6c08c5648d59edd54263c6b  cells/0/c0_score.py
18e27ac743b983e320f73aefe35504a748c7d7b7613271f51b5faf0d94842f85  cells/0/c0_aggregate.py
```

- 2026-11-12 JST · 格 0 · prereg v2 · commit ca9c154：审稿不确认（`cells/0/review_v2.md`），未运行任何东西，由 v2.1 取代。
- 2026-11-12 JST · 格 0 · prereg v2.1 · commit cdebf56。sha256（含所调用的仪器文件）：

```
532ec03b3f95d8951131e205a4ba305c63d07ac04475d81b038b3437da5e7616  cells/0/prereg.md
3836f67e14071db17fd0d0a1c03e0bf4df95192e76176d5283e469d97a5d179f  analysis/seam.py
22d4a6edd789521c0a448331b6eabb37a0131b73023dd1dd59d67859188ef433  cells/0/c0lib.py
43a4c7c2fce3db9c99e166a5828dedf7b92736d7c40d6ad1ac38def5b68a0784  cells/0/c0_score.py
be445c2eec335cda5334a37eef2141bf1335492d16f0bde3eec6e04ffad0ad70  cells/0/c0_aggregate.py
3645d838cbdd86341f6a7278495dc5e64573357996d529905c1bcd81cd8238cc  analysis/topology.py
3da571aa1c9428747f5ae42dd396b862468ce044d90ce9fac4920dc4f6ccc675  analysis/smoothness.py
2a29a0cfae9d96e264f3519d6201d668ed672ef9d790bb89065bf884e849fb17  analysis/gates.py
eb44263da6ab50ea985e11cfc58cbfaa7665f94cb22a0117e2e561b3ef4ab300  analysis/controls.py
```
- 2026-11-12 JST · 格 0 · prereg v2.1（cdebf56）：第二轮审稿不确认（`cells/0/review_v2_1.md`），未运行，由 v2.2 取代。
- 2026-11-12 JST · 格 0 · prereg v2.2 · commit 08702ee。sha256：

```
18dce12e6490a60f1394dac07856ebf85b096d4c1f2963eeffc939e293c4686c  cells/0/prereg.md
3836f67e14071db17fd0d0a1c03e0bf4df95192e76176d5283e469d97a5d179f  analysis/seam.py
22d4a6edd789521c0a448331b6eabb37a0131b73023dd1dd59d67859188ef433  cells/0/c0lib.py
edd3fc75cb76144962c4337f3b1c2a93fe6229dba6dfb4a3effa110308b831da  cells/0/c0_score.py
c45f95c7b686cb045967fbb97979779070bc02fc0ecf41f6c2d05e496df14eaa  cells/0/c0_aggregate.py
3645d838cbdd86341f6a7278495dc5e64573357996d529905c1bcd81cd8238cc  analysis/topology.py
3da571aa1c9428747f5ae42dd396b862468ce044d90ce9fac4920dc4f6ccc675  analysis/smoothness.py
2a29a0cfae9d96e264f3519d6201d668ed672ef9d790bb89065bf884e849fb17  analysis/gates.py
eb44263da6ab50ea985e11cfc58cbfaa7665f94cb22a0117e2e561b3ef4ab300  analysis/controls.py
```
- 2026-11-12 JST · 格 0 · prereg v2.2 的审稿确认：`cells/0/review_v2_2.md`（结论：确认；预先授权第 3 条）。v2.1、v2.2 都是冻结后的修订，按 Z 2026-11-12 的决定仍需 Z 逐次确认后才能开跑。
- 2026-11-14 JST · 格 0 · Z 接受 prereg v2.2（含两处点名的改动：像素仪器失败只停像素部分；收敛判据用 val 损失），允许开跑阶段 1，在云端跑。决定记录在 `cell6/v1` 分支 `cells.yaml` 的 `decisions`。
- 2026-11-15 JST · 格 0 · 阶段 1 结果（prereg v2.2，commit 08702ee）。计分前重新核对冻结的 9 个文件，sha256 全部一致。105 个训练 run 与 9 个零对照全部测量，无排除。判定 DECIDED：P0.1 / P0.2 pass；P0.3 = F；P0.4-sg = F（撕开率 1.0）；P0.4-sym = T；θ gate n = 2000 敏感性相同；P0.5 留到阶段 2；P0.6 我的预测 F，原版只在 raw-recon-mse 上成立。按 §7 改 `cells.yaml`：格 0 exists high → mid（status 不变），notes 记 sg "继承切口"、sym "粘合"；格 4 yields mid → lowmid；格 15 notes 记声明产出。偏离：设备对照没跑（云端无 MPS）；按 Z 2026-11-14 的决定在云端跑，容器重启 3 次，中断的 run 从头重训。结果：`cells/0/result.md`、`cells/0/result_data.json`；逐 run 输出备份在分支 `runs-backup/2026-11`。

## 格 0 延伸运行 E1：收敛复查

- 2026-11-17 JST · 格 0 · 延伸预注册 E1（`cells/0/prereg_ext_convergence.md`）· commit 3dab71f。原始角度的四个臂、S 与 M、种子 0 / 1 / 2，同一配方训练 100000 步（阶段 1 是 20000 步）。Z 2026-11-17 在会话里同意（对"格 0 的收敛复查：四个臂训练到 5 倍步数"回复"开始做吧"）；设计里超出这句话的选择写在预注册 §2。不改 v2.2 的冻结文件与阶段 1 的判定；开跑前核对 v2.2 的 9 个冻结文件，sha256 全部一致。数据门在本环境重跑，主世界与负对照都选中 index 0，H1 读数与阶段 1 的记录逐位相同。sha256：

```
d4d8f9e6b95f54d4214f0ecedf78cd0152c28975a4720e5a56d8d176c09e43df  cells/0/prereg_ext_convergence.md
f46e0d18f4f5fa9b605266300f4f0b18075ddb5e208289ef2ab23e87f3b19a0e  cells/0/c0_ext.py
eb4bef540a8eb12aedfb1b95667a9558739f544ec18e0d623b786a60f86ba171  cells/0/ext_convergence_baseline.json
```
- 2026-11-17 JST · 格 0 · 延伸 E1 的环境复现检查（`c0_ext.py --envcheck`，不计分）：`raw-jepa-sym_S_s0` 按阶段 1 的参数原样重跑 20000 步，分类"粘合"，接缝比 1.0515042212106465，val 损失 0.006504046119516715，三项都与备份里的阶段 1 结果逐位相同。本环境复现阶段 1。

## 动物园修复记录

- 2026-11-17 JST · `zoo/wright_fisher.py` · 单纯形漂移。确定性模式下 `mixed`、`mutation_selection`、`rps` 走 RK4，原来每步之后不投影回单纯形。记 s = Σx，这类向量场满足 d(s − 1)/dt = −r·(s − 1)（r = x·f + μ、x·f、x·Ax），r < 0 时舍入误差按指数放大。K = 3、600 步、400 条路径、种子 1：`mixed_only` 有 14 条偏离超过 1e-9（3 条变成非有限值），`mutation_selection` 有 52 条（9 条）。实测增长率与 −r 三位小数一致。修法：每个 RK4 子步之后截断并归一化；修后两个预设的最大偏离都是 2.2e-16。新增 5 个测试（`tests/test_wright_fisher.py` 的长视界不变性；`tests/test_zoo_independent.py` 用 log 坐标下的独立积分器对照 600 步），在修复前的代码上有 4 个失败，修复后全过；全套 310 个测试通过。
  - 影响：只在长视界的确定性多机制路径上出现（1e-16 放大到 1e-6 约需 25 个时间单位；默认 `seg_steps = 20`）。有限种群模式不受影响。格 0、格 6 不用这个世界，冻结文件不涉及。
  - 重跑 `cells/9/exploratory/c9_clock_metric.py`（600 步的 `mixed_only`），新输出存为同目录的 `c9_clock_metric_after_simplex_fix.out`，原输出保留。只有 mixed 两行变化：n 从 295 / 399 变为 298 / 400，CV 的变化不超过 0.006，最优时钟不变（euclid、logit）。格 9 notes v0.6 的结论（Fisher 在 8 个情形里只赢 1 个）不变。`precheck_c9_noise_clock.py` 用有限种群，输出与提交的逐字相同。
  - 发现经过与验证数字：`reports/2026-11-17_external_review.md` §8。
