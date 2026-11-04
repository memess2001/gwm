# 几何世界模型计划 — 文献审计（第一轮，2026-11-01；汇编于 2026-11-04）

本文件是五路并行审计的原始报告汇编（每路由一个独立的 Claude 子任务完成，约 200 次 WebSearch 与数百次 arXiv 页面核对）。结构化结论见 `cells.yaml`（`collision / nearest / must` 字段）与 `HANDOFF.md` §7。

阅读说明：
- 撞车等级：A = 直接撞车；B = 近邻（同一几何结构用于动力学 / RL / 表征，但主张不同或不是当世界模型用）；C = 远邻；带 +/− 表示偏向。
- 所有 URL 来自搜索或页面核对的真实返回；各路报告在不确定处均作了标注（"据记忆""未验证"）。引用均为转述。
- 会话级搜索预算在中途耗尽：格 10 零次搜索（靠已知 arXiv 编号逐篇核对）、格 7 两次、格 9 十一次；各路末尾列出了"未执行的搜索词"，第一周补查。

汇总表：

| 格 | 名字 | 等级 | 报告 |
|---|---|---|---|
| 1 | 曲率即规律（测地线） | B+ | G1 |
| 2 | 下降即规律（自然梯度流） | B+ | G1 |
| 3 | 搬运还是改权（WFR） | A（方法）/ B（主张） | G2 |
| 4 | 形状即规律（拓扑优先） | B+ | G3 |
| 5 | 可分辨性即规律（模型流形） | B | G4 |
| 6 | 联络即规律（和乐） | B | G3 |
| 7 | 两种直线（对偶平坦） | B− | G1 |
| 8 | 距离即代价（拟度量 / Finsler） | A− | G3 |
| 9 | 时间即弧长（信息时间） | B | G2 |
| 10 | 尺度即时间（RG） | B | G2 |
| 11 | 奇点即规律（分岔地图） | B | G4 |
| 12 | 离散曲率找瓶颈 | A− | G4 |
| 13 | 因果正交（IGCI） | B | G5 |
| 14 | Klein 层级 | B | G5 |
| 15 | Liouville 散度 | B | G5 |
| U | 伞状纲领 / benchmark 地形 / AWM 跟进 | — | G5 |

---


<!-- ===== 报告 G1 ===== -->

# 文献审计 G1：Cell 1 / Cell 2 / Cell 7（几何 / 信息几何世界模型）

审计日期：2026-11-01。方法：WebSearch（英文为主，优先 2024–2026，兼顾经典根源）+ WebFetch 读 arXiv 摘要页 / 正文确认。所有 URL 均来自搜索或 fetch 实际返回的结果；未能验证的内容明确标注"未验证"。

**重要限制**：本轮 WebSearch 配额（全部 agent 共享，200 次/轮）在 Cell 2 搜到一半时耗尽。Cell 1 完成了约 20 次搜索，Cell 2 约 12 次（另有 4 次被拒），Cell 7 只完成了 2 次专门搜索，其余靠已搜到的线索与交叉引用做 fetch 补充。因此 Cell 7 的覆盖信心偏低，见各节说明。

撞车等级定义：A = 直接撞车；B = 近邻（同一几何结构用于动力学 / RL / 表征学习，但主张不同或不是当世界模型用）；C = 远邻（几何工具存在，没找到动力学 / 世界模型方向的应用）。

---

## Cell 1：曲率即规律（测地线世界模型）

### 撞车等级：**B+**（技术机器已有人搭了大半，但"纯测地线 + 动作 = 测地线偏离 + 曲率作为不变量 + Fisher 拉回度量"的组合没人做过；同时有一股反方向潮流）

没有找到任何一篇把"学到的黎曼度量的测地线 = 无外力自由演化"明确当作 world model（像素观测、动作条件、多步 rollout）来做的工作。但以下三组工作在技术上离得很近：

### 最近邻列表

**第一组：潜空间 Lagrangian / 度量学习 + 动力学（最近）**

1. **A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics** — Katharina Friedl 等（Jaquier, Lundell, Asfour, Kragic），2024/2025，ICLR 2025。
   - arXiv: https://arxiv.org/abs/2410.18868 ；ICLR 页: https://proceedings.iclr.cc/paper_files/paper/2025/hash/913c7cbda1811776ff3db7316fd86808-Abstract-Conference.html
   - 做了什么：双正交约束的 autoencoder 学一个低维潜空间，潜空间上**学一个 SPD 质量矩阵 M̌(q̌)**（即学到的黎曼度量）和一个标量势 V̌，潜动力学是 Euler–Lagrange 方程 M̌ q̈ + č + ǧ = τ̌，外力 τ̌ 作为输入。在 16-DoF 摆、192-DoF 绳、600-DoF 布上做长程预测。
   - 接近程度：**非常近**。"潜空间上学度量 + 该度量定义的（受迫）测地线型动力学 + 控制输入为偏离项"这套机器与 Cell 1 几乎同构。差异：(a) 他们有势能项 V̌，不是纯测地线；(b) 输入是已知广义坐标，不是像素观测；(c) 度量是自由学的 SPD 网络，不是观测模型的 Fisher 拉回；(d) 完全不谈曲率，他们的"曲率"讨论只针对 SPD 流形本身（Cholesky 参数化忽略 SPD 曲率导致 swelling）；(e) 不叫 world model。
2. **Reduced-order Control and Geometric Structure of Learned Lagrangian Latent Dynamics** — Katharina Friedl 等，2026-02。
   - https://arxiv.org/abs/2602.08963
   - 上一篇的控制续作：在学到的潜 Lagrangian 系统上做跟踪控制，量化建模误差来源，扩展到欠驱动系统。
   - 接近程度：B。说明这个团队正沿"学到的潜几何 + 动力学 + 控制"推进，是 Cell 1 最需要盯住的团队。
3. **GeoHNNs: Geometric Hamiltonian Neural Networks** — Aboussalah & Ed-dib，2025-07。
   - https://arxiv.org/abs/2507.15678
   - 惯性矩阵参数化在 SPD 流形上 + 辛结构约束的 autoencoder；潜空间 Hamiltonian。
   - 接近程度：B−。学到的 SPD 度量进入动能，但以 Hamiltonian 守恒为主张，不谈测地线与曲率。
4. **LagNetViP: A Lagrangian Neural Network for Video Prediction** — Allen-Blanchette 等，2020。
   - https://arxiv.org/abs/2010.12932
   - 从像素学低维状态 + Lagrangian（动能、势能分开），用 Euler–Lagrange 做视频预测（Pendulum、Acrobot）。
   - 接近程度：B。这是"从像素学 Lagrangian 世界模型"的根源之一，但没把动能项解释为度量/测地线，没有曲率。

**第二组："障碍 = 度量/曲率，运动 = 测地线"（机器人运动生成，不是预测）**

5. **Learning Riemannian Manifolds for Geodesic Motion Skills** / **Reactive Motion Generation on Learned Riemannian Manifolds** — Hadi Beik-Mohammadi, Hauberg, Arvanitidis, Neumann, Rozo，2021 / 2022。
   - https://arxiv.org/abs/2106.04315 ； https://arxiv.org/abs/2203.07761
   - VAE 学一个黎曼度量，机器人技能 = 该度量下的测地线；**障碍物通过在线修改 ambient metric 让测地线绕开**（即"碰撞 = 空间弯曲"的工程版）。7-DoF 机械臂。
   - 接近程度：B+。这是"碰撞不是力而是度量"最直接的先例，但它是运动**生成**（policy），不是环境动力学**预测**；度量来自 Latent Space Oddity 的拉回，不是 Fisher。
6. **On the Design of Region-Avoiding Metrics for Collision-Safe Motion Generation on Riemannian Manifolds** — Holger Klein 等（Jaquier, Asfour），2023。
   - https://arxiv.org/abs/2307.15440
   - 用 barrier function 构造黎曼度量，使测地线严格避开禁区；人形机器人。
   - 接近程度：B。同上，是设计度量不是学度量。
7. **Space Is Intelligence: Neural Semigroup Superposition for Riemannian Metric Generation** — Chenghao Xu，2026-06。
   - https://arxiv.org/abs/2606.18828
   - 主张"智能在环境不在 agent"：场景诱导一个度量场，agent 的动作就是沿测地线走，无需规划器和碰撞检测；单场景训练零样本泛化到新障碍布局。
   - 接近程度：B+（哲学上最接近 Cell 1 的口号），但是规划/策略问题，摘要里没有"曲率"一词，没有预测与世界模型。
8. **Riemannian Motion Policies** — Ratliff 等，2018（根源）。
   - https://arxiv.org/abs/1801.02854
   - 二阶动力系统 + 黎曼度量配对，不同空间的策略几何一致地组合；碰撞回避是主要应用。

**第三组：World model 社区里的"几何"工作（名称撞车或反方向）**

9. **Temporal Straightening for Latent Planning** — Ying Wang 等（含 Balestriero, LeCun, Mengye Ren），2026-03。
   - https://arxiv.org/abs/2603.12231
   - JEPA 世界模型里加一个**曲率正则化**，把潜轨迹拉直，使 Euclidean 距离逼近 geodesic 距离，改善梯度规划。
   - 接近程度：B，但**方向相反**：他们把曲率当噪声要压掉；Cell 1 把曲率当规律要学出来。这是 Cell 1 最需要正面对比的基线。
10. **GeoWorld: Geometric World Models** — Zeyu Zhang, Li, Reid, Hartley，CVPR 2026。
    - https://arxiv.org/abs/2602.23058 ； https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_GeoWorld_Geometric_World_Models_CVPR_2026_paper.pdf
    - Hyperbolic JEPA（固定负曲率）+ 几何 RL 做多步视觉规划，比 V-JEPA 2 提升 2–3%。
    - 接近程度：C（固定曲率、不学度量、不做测地线动力学），但**名字已被占**："Geometric World Models" 这个标题不能再用。
11. **A Geometric Theory of Cognition for Machine Intelligence** — Laha Ale（单作者，西南交大），2025-12 / 2026-06 v3。
    - https://arxiv.org/abs/2512.12225
    - 学一个潜空间黎曼度量 G(η)，状态按 **Riemannian 梯度流 η̇ = −G⁻¹∇J** 演化（不是测地线），做 PPO 部分可观测 RL + 初步"world model"分析（Ant-v5 rollout）。
    - 接近程度：B（对 Cell 1）/ B+（对 Cell 2，见下）。正文有多处自相矛盾（度量参数化前后不一致、算法 3 用真值下一帧算梯度等），无 venue，MiniGrid 上没提升。质量低，但说明"学度量 + 几何流 + world model"这个标签已经有人贴了。
12. **latent-riemannian-world**（PyPI 包，作者 lajjadred，2026-05 v0.3.0）。
    - https://pypi.org/project/latent-riemannian-world/ ； https://github.com/lajjadred/latent-riemannian-world
    - 模块含 PullbackMetric / FisherMetric、GeodesicSolver、**RiemannianRSSM**，引用 Dream to Control。3 star、6 commits、无论文、无单位。
    - 接近程度：未知（README 没说明 RSSM 转移是否沿测地线）。**观察项**：名字和模块组合与 Cell 1 高度重合，值得定期回看。
13. **Latent geometry of world models** — Adam Černý，Charles University 本科论文，2026-06。
    - https://dspace.cuni.cz/handle/20.500.11956/210796
    - 分析 DreamerV1 潜空间的 Euclidean 距离与时间结构；无学到的度量、无测地线。C。

**第四组：度量学习 + 轨迹推断（群体数据，非 world model）**

14. **Riemannian Metric Learning via Optimal Transport** — Scarvelis & Solomon，2022/2023。
    - https://arxiv.org/abs/2205.09244
    - 从分布的横截面快照学一个空间变化的度量张量（神经网络），用其测地线做分布间插值与轨迹推断（scRNA、候鸟迁徙）。
    - 接近程度：B−。是"学度量使轨迹≈测地线"的代表，但对象是群体快照，没有动作与曲率。
15. **Metric Flow Matching for Smooth Interpolations on the Data Manifold** — Kapuśniak 等（Bronstein, Bose, Di Giovanni），NeurIPS 2024。
    - https://arxiv.org/abs/2405.14780 ； https://nips.cc/virtual/2024/poster/94221
    - 插值路径 = 数据诱导度量下的近似测地线（最小化动能），用于单细胞轨迹推断。C/B−。
16. **Follow the Energy, Find the Path: Riemannian Metrics from Energy-Based Models** — Béthune 等，NeurIPS 2025。
    - https://arxiv.org/abs/2505.18230 ； https://neurips.cc/virtual/2025/poster/119377
    - 从预训练 EBM 导出黎曼度量，测地线贴数据流形、曲率失真小；静态数据，无动力学。C。

**第五组：物理 / 理论根源**

17. **Learning Relativistic Geodesics and Chaotic Dynamics via Stabilized Lagrangian Neural Networks** — Hamzaogullari & Ozakin，2026-01。
    - https://arxiv.org/abs/2601.12519
    - 改进 LNN 训练稳定性（Hessian 正则、Lorentz 符号惩罚），从轨迹学 AdS₄ 的测地 Lagrangian；"从测地轨迹提取度量分量"列为未来工作。
    - 接近程度：B−。说明"从轨迹学测地 Lagrangian"可行但训练难。
18. **Stochastic Processes as Non-Metric Geodesics in Information Geometry** — T. Koide, A. van de Venn，2026-09。
    - https://arxiv.org/abs/2609.08870
    - Gaussian 统计流形上 α-联络的测地线 ↔ 常噪声强度的线性 SDE（Brownian bridge、OU 弛豫、自由扩散）。**关键一句：OU 的恢复力可以归因于标量曲率，也可以归因于非度量性（non-metricity），取决于联络的表示选择。**
    - 接近程度：B（理论）。对 Cell 1 的"碰撞不是力而是弯曲"有直接的概念警示（见负面结果）。
19. **The geometry of the Fisher selection dynamics** — Shapovalov & Evdokimov，1998。
    - http://arxiv.org/abs/physics/9805006
    - 把 Fisher 选择动力学改写为某个仿射联络（projectively Euclidean、equiaffine）下的测地运动 + 张量型"有效外场"；仿射曲率解释为等位基因耦合。
    - 接近程度：B（1998 的经典版本：动力学 = 测地线 + 偏离项，曲率 = 相互作用）。Cell 1 和 Cell 7 都该引。
20. **Pulling back information geometry** — Arvanitidis 等，AISTATS 2022。
    - https://proceedings.mlr.press/v151/arvanitidis22b.html ； https://arxiv.org/pdf/2106.05367
    - 把 decoder 输出分布的 Fisher–Rao 度量拉回潜空间，适用于任意 decoder 分布。无动力学。这是 Cell 1 变体（p(o|z) 的 Fisher 拉回）的直接根源。C（工具）。
21. **The Spacetime of Diffusion Models: An Information Geometry Perspective** — Karczewski, Heinonen, Pouplin, Hauberg, Garg，2025/2026。
    - https://arxiv.org/abs/2505.17517
    - 在扩散模型的 (x_t, t) "时空"上建 Fisher–Rao 几何（去噪分布是指数族），测地线用于编辑距离与分子过渡路径。C（信息几何 + 测地线，但不是环境动力学）。
22. **A Review on Riemannian Metric Learning: Closer to You than You Imagine** — 2025 综述。
    - https://arxiv.org/html/2503.05321v2
    - 覆盖 Hamiltonian learning（"测地线 = V≡0 的 Hamiltonian 轨迹"）、纵向轨迹建模（阿尔茨海默病随访轨迹建模为测地线）、分布轨迹推断（Scarvelis、Huguet、MFM）。**综述里没有 RL / world model 章节**——这本身是 Cell 1 空位的证据。
23. 其余远邻（C）：Tennenholtz & Mannor 2021 拉回度量做 offline RL 不确定性（https://arxiv.org/abs/2102.11327）；Testa 等 2025 Geometric Contact Flows（https://arxiv.org/abs/2506.17868）；Souveton 2026 Riemannian Neural Hamiltonian Flows，固定度量动能 + 学势（https://arxiv.org/abs/2609.21647）；Hartwig 等 2025 隐式潜流形上的测地线微积分（https://arxiv.org/abs/2510.09468）；Di Zhang 2025 用度量张量场优化建模、Einstein–Hilbert 类比（https://arxiv.org/abs/2510.26068）；Li/Qiu/Calinon 2024 神经 Riemannian eikonal 求解器（https://arxiv.org/abs/2412.05197）；Delliaux 等 2025 群结构潜空间世界模型（https://arxiv.org/abs/2506.01529）。

### 负面结果 / 反方向证据

- **反方向潮流**：Temporal Straightening（Wang 等 2026）在 JEPA 世界模型中发现潜轨迹的曲率**有害**于规划，主动把它压平。Cell 1 必须回答"为什么曲率在你这里是信号不是噪声"，并在同类任务上正面比较。
- **可辨识性问题（审计者的理论备注，非文献结论）**：经典的 Maupertuis / Jacobi 度量告诉我们，任何带势能的保守系统在固定能量面上都能改写为某个共形度量的测地线；Koide & van de Venn 2026 也明确指出同一恢复力可归因于曲率或非度量性，取决于联络表示。因此"碰撞是弯曲不是力"在**仅有轨迹数据**时不是可辨识的事实，而是表示选择。要把曲率说成不变量，必须额外固定度量的来源（这正是 Fisher 拉回变体的价值）或固定能量/速度归一化。
- **训练难度**：Hamzaogullari & Ozakin 2026 报告 LNN 学测地 Lagrangian 训练不稳定、需要 Hessian 正则；Friedl 等报告 SPD 层收益有限、Cholesky 参数化导致 swelling；综述指出测地线计算昂贵、度量学习非凸（常数度量除外）。
- **质量警示**：Ale 2025 的"几何认知理论"在 MiniGrid 上没有提升、多处自相矛盾，说明"学度量 + 几何流"如果没有干净的不变量与对照，很容易做成一篇没有说服力的论文。

### 还剩什么没做

- 没人做过：**纯测地线（无势能）潜动力学 + 动作 = 协变加速度（测地线偏离）+ 像素/高维观测 + 多步 rollout**，在含碰撞的标准环境（如 MiniGrid/PointMaze/物理 box 环境）上验证。
- 没人把 **p(o|z) 的 Fisher 拉回度量固定为世界模型的度量**（Arvanitidis 2022 的工具 + 动力学），从而把度量的自由度去掉，使曲率成为可辨识量。
- 没人报告过 **曲率场 vs. 障碍图的对应**、或用 Jacobi 场（geodesic deviation）预测邻近轨迹发散（碰撞敏感性）作为不变量。

### 可能的差异化点

- 主打"**曲率 = 障碍的地图**"：输出标量/截面曲率热图并与真实障碍布局做定量对齐（IoU / 相关），这是 Friedl、Beik-Mohammadi、Xu 都没做的可测不变量。
- 用 Fisher 拉回把度量**钉死**在观测模型上，只学 encoder/decoder 与动作偏离项，正面回应可辨识性问题；并与"自由学 SPD 度量"（Friedl 式）做消融。
- 以 Temporal Straightening 为对照基线：同一 JEPA 骨架，一边拉直一边允许弯曲，比较穿越障碍的长程 rollout。
- 命名避开 "Geometric World Models"（GeoWorld 已占）。

### 覆盖信心：**中高**
机器人测地线运动、潜 Lagrangian、度量学习综述、world model 社区三条线都搜到了代表作；可能遗漏 workshop 级别的 "geodesic JEPA" 类短文和 2026 下半年新作。

### 搜索词（Cell 1，约 20 次）
geodesic world model learned Riemannian metric latent dynamics / Riemannian latent dynamics geodesic prediction learned metric trajectory / "geodesic" latent dynamics "world model" Riemannian metric free motion action / neural geodesic flow learned metric tensor trajectory prediction "geodesic equation" / pullback metric Fisher information decoder geodesics dynamics latent state space / "metric learning" geodesics "collision" curvature latent space robot dynamics Riemannian motion policies / learning Riemannian metric from trajectories so that trajectories are geodesics neural ODE / "Riemannian" "world model" reinforcement learning latent metric geodesic planning 2025 / "metric tensor" learned dynamics "geodesic" curvature "force" general relativity machine learning / Jacobi metric Maupertuis principle neural network learn dynamics as geodesics / "Riemannian" latent "curvature" world model RSSM Dreamer geodesic / Latent Space Oddity follow-up geodesics dynamics time series latent ODE / Riemannian metric learning review "closer to you than you imagine" / "geodesic" "latent" video prediction "Christoffel symbols" / Hamiltonian neural network learned Riemannian metric mass matrix geodesic / Fisher information metric observation model pullback latent dynamics world model / "geodesic" "action-conditioned" metric learned "acceleration" deviation from geodesic / Riemannian flow matching learned metric trajectory inference "metric flow matching" / "geodesic" "neural" "Riemannian" "latent" dynamical systems curvature "potential" 2026 / "geodesic" JEPA "world model" latent "curvature" straightening / "geodesic neural" OR "neural geodesic" dynamics forecasting

---

## Cell 2：下降即规律（自然梯度流世界模型）

### 撞车等级：**B+**（三块零件——"replicator/Bayes = 自然梯度流"的理论、"学势函数 + 固定度量的梯度流拟合观测动力学"、"学到的向量场做 Helmholtz–Hodge 诊断"——各自都有成熟工作，但没有人把它们在 Fisher 度量下组合成 world model）

### 最近邻列表

**第一组：理论根源——世界/演化/贝叶斯 = Fisher 度量下的梯度流**

1. **Conjugate Natural Selection** — Luca de Alfaro, Reilly Raab, Yang Liu，2022/2023。
   - https://arxiv.org/abs/2208.13898
   - 证明 Fisher–Rao 自然梯度下降（FR-NGD）是连续 replicator 动力学在任意参数族上的最优（最小二乘）逼近；loss 取 surprisal 时同时最优逼近连续贝叶斯推断。定义了 **"natural deviation"**：真实动力学与 replicator 之间的 Fisher 加权误差，其 Hessian 恰是 Fisher，因而是凸的。
   - 接近程度：**B+**。这是 Cell 2 的核心等式（replicator = Bayes = 自然梯度流）的现代证明；而且"natural deviation"几乎就是 Cell 2 想要的诊断量。差异：他们的方向是"用 NGD 当算法模拟演化/贝叶斯"，不是"从观测到的世界动力学里学一个势 F"。
2. **Natural Gradient Ascent in Evolutionary Games** — Vladimir Jaćimović，2022。
   - https://arxiv.org/abs/2210.00573
   - 连续性状空间的 replicator 限制在多维 Gaussian 族上 = 平均适应度的自然梯度流；联系 Natural Evolution Strategies。纯理论，不学任何东西。B。
3. **Information Geometry and Evolutionary Game Theory** / **The Replicator Equation as an Inference Dynamic** — Marc Harper，2009。
   - https://arxiv.org/abs/0911.1383 ； https://arxiv.org/abs/0911.1763
   - Shahshahani 几何 = 单纯形的 Fisher 信息几何；对称适应度时 replicator 是平均适应度在 Shahshahani 度量下的梯度；replicator 的解可写成指数族；replicator 与贝叶斯更新形式相似。根源（已知）。
4. **Dynamical Systems on Statistical Models** — 藤原彰夫（Akio Fujiwara），1993，数理解析研究所講究録 822。
   - https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/0822-04.pdf
   - 对偶平坦流形上、由散度定义的势的梯度流沿**对偶联络的测地线**收敛到目标点；Legendre 结构下梯度流在对偶坐标里是线性的（η̇ = −η）；偶数维时完全可积（推广 Nakamura 的可积梯度系统）；指数族上向经验分布投影的梯度流沿 m-测地线收敛到 MLE。
   - 接近程度：B+（理论根源，同时是 Cell 7 的根源）。它说明"Fisher 梯度流 + 对偶平坦 ⇒ 平坦坐标里线性"是 1993 年就有的定理。

**第二组："世界是一个学习者"的物理/FEP 版本（概念撞车区）**

5. **A free energy principle for a particular physics** — Karl Friston，2019（专著式长文）。
   - https://arxiv.org/abs/1906.10184
   - 非平衡稳态的流写成 **f = (Q − Γ)∇ℑ**：Q 反对称（solenoidal、沿等概率面打转），Γ 梯度项（耗散、沿 surprisal 梯度下降），即 Helmholtz 分解；Fisher 度量定义在参数化密度的共轭变量空间上，给出 information length。
   - 接近程度：**B+**。FEP 把"世界 = 梯度下降 + 打转"当作公理摆出来了，Cell 2 的 Hodge 诊断在 FEP 里是出发点而不是结论。差异：FEP 是设定（posit），不是从数据学势 F 再检验。
6. **Stochastic Chaos and Markov Blankets** — Friston, Heins, Ueltzhöffer, Da Costa, Parr，Entropy 23(9):1220，2021。
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC8465859
   - 对 Lorenz 系统做广义 Helmholtz 分解（梯度 + solenoidal + 修正项），梯度部分 = 对 −log 稳态密度的 Riemannian 梯度下降；耦合两个 Lorenz 得到广义同步。
   - 接近程度：B+。这是 FEP 里"对具体混沌系统做梯度/旋转分解"的实例。
7. **Neural dynamics under active inference: plausibility and efficiency of information processing** — Da Costa, Parr, Sengupta, Friston，Entropy 23(4):454，2020/2021。
   - https://arxiv.org/abs/2001.08028
   - active inference 的信念更新动力学近似自然梯度下降（信息空间最速下降），用 information length 比较效率。
   - 接近程度：B。是 agent 内部推断的自然梯度流，不是环境的。
8. **Path integrals, particular kinds, and strange things** — Friston, Da Costa, Sakthivadivel, Heins, Pavliotis, Ramstead, Parr，2022/2023（Physics of Life Reviews）。
   - https://arxiv.org/abs/2210.12761
   - FEP 的路径积分版本，粒子分类为耗散/保守、惰性/主动、普通/奇异。摘要未提信息几何或 Helmholtz。C+。
9. **Bayesian mechanics of self-organising systems** — Takuya Isomura，2023/2024。
   - https://arxiv.org/abs/2311.10216
   - 通用动力系统的 Hamiltonian 构成一类生成模型，Helmholtz 自由能 ≡ 变分自由能。（注意："Helmholtz" 这里是热力学自由能，不是向量场分解。）C+。
10. **The world as a neural network** — Vitaly Vanchurin，2020（Entropy）；**Geometric Learning Dynamics** — Vanchurin，2025/2026。
    - https://arxiv.org/abs/2008.01540 ； https://arxiv.org/abs/2504.14728
    - 宇宙 = 神经网络的学习动力学；可训练变量的度量 g ∝ κ^α（κ 噪声协方差），α=1 量子、α=1/2 高效学习、α=0 经典演化。
    - 接近程度：B（概念）。"世界在学习"这个叙事已被 Vanchurin 占了大标题；但他没有从数据学势、没明说 Fisher。
11. **Entropic Geometry and Symmetry Breaking in Lie-Group Free-Energy Minimization** — Noboru Sagae，2025-11，preprints.org 未审稿。
    - https://www.preprints.org/manuscript/202511.0382
    - 自由能最小化 = Lie 群轨道上的自然梯度流，Fisher 度量；多数定理较薄。C。
12. **Verifying Good Regulator Conditions for Hypergraph Observers: Natural Gradient Learning from Causal Invariance** — Max Zhuravlev，2026-03。
    - https://arxiv.org/abs/2603.09067
    - Good Regulator 定理 ⇒ 观察者必须有内部模型；自然梯度是唯一可容许学习规则；接 Vanchurin 的 α 框架。概念性，无实验。C。

**第三组：学势函数 + 固定度量的梯度流拟合观测动力学（同一模板、不同度量）**

13. **Proximal Optimal Transport Modeling of Population Dynamics (JKOnet)** — Bunne, Meng-Papaxanthos, Krause, Cuturi，2021/2022。
    - https://arxiv.org/abs/2106.06345
    - 群体快照建模为 JKO 流：学一个参数化能量，使其 **Wasserstein 梯度流**拟合快照序列。
    - 接近程度：**B+**。这正是"学势 + 固定度量的梯度流 = 世界动力学"模板——只是度量是 Wasserstein 而不是 Fisher–Rao，对象是群体快照不是 agent 环境。
14. **Learning of Population Dynamics: Inverse Optimization Meets JKO Scheme (iJKOnet)** — Persiianov 等（Korotin），2025/2026。
    - https://arxiv.org/abs/2506.01502
    - 去掉 ICNN 限制的 JKO 学习，对抗式端到端。B。
15. **Wasserstein Residuals: Learning Gradient Flows from Population Dynamics** — Heinonen, Shenfeld, Baptista, Waxman, Batenkov, Cooijmans, Bingham，2026-07（搜索结果同时出现 ICML 2026 与 NeurIPS 2026 页面，具体 venue 未确认）。
    - https://arxiv.org/abs/2607.04738
    - 用连续性方程的非负残差损失（最小值即 WGF）替代 JKO，统一多种方法。**残差在原理上可当"是否梯度流"的诊断**，但摘要未这样用。B+。
16. **WFR-FM: Simulation-Free Dynamic Unbalanced Optimal Transport** — Qiangwei Peng 等（Peijie Zhou），ICLR 2026。
    - https://arxiv.org/abs/2601.06810 ； https://proceedings.iclr.cc/paper_files/paper/2026/hash/2030d35841f8617ab22cca913f55697e-Abstract-Conference.html
    - 同时学位移向量场（Wasserstein 部分）和标量增长率（Fisher–Rao 部分，对应出生/死亡），恢复 WFR 测地线。
    - 接近程度：B。Fisher–Rao 部分进入了学习，但作为增长项而非势的自然梯度。
17. **A Geometric Theory of Cognition for Machine Intelligence** — Laha Ale，2025/2026（详见 Cell 1 第 11 条）。
    - https://arxiv.org/abs/2512.12225
    - 潜状态 η̇ = −G(η)⁻¹∇J，G 学出来、J 是任务损失（RL 为负回报，world model 为预测重构误差）。
    - 接近程度：**B+**（结构上最像"Riemannian 梯度流世界模型"），但 G 不是 Fisher、J 不是从环境学的势、没有 Hodge 诊断、质量堪忧。

**第四组：学到的向量场的 Helmholtz–Hodge 分解（作为先验或诊断）**

18. **HHD-GP: Incorporating Helmholtz-Hodge Decomposition into Gaussian Processes for Learning Dynamical Systems** — Hao Xu, Jia Pan，NeurIPS 2024。
    - https://nips.cc/virtual/2024/poster/95052
    - 把 HHD 内建进 GP 先验，同时学 curl-free 与 divergence-free 分量；需要对称性先验才可辨识。
    - 接近程度：**B+**。这是"学动力学时显式分 gradient / curl"的最直接 ML 先例；但它是建模先验，不是对 world model 的事后诊断，也不在 Fisher 度量下。
19. **Dissipative Hamiltonian Neural Networks** — Sosanya & Greydanus，2022。
    - https://arxiv.org/abs/2201.10085
    - 同时参数化 Hamiltonian 与 Rayleigh 耗散函数，自称"隐式 Helmholtz 分解"；在海洋流场上得到"科学洞察"。B。
20. **The Mechanics of n-Player Differentiable Games** — Balduzzi 等，2018（ICML 2018，凭记忆，页面未确认 venue）。
    - https://arxiv.org/abs/1802.05642
    - 博弈动力学分解为 potential（梯度）与 Hamiltonian（守恒/旋转）两部分（广义 Helmholtz），由此提出 SGA。B（学习动力学诊断的经典）。
21. **Velde: constructing cell potential landscapes by RNA velocity vector field decomposition** — Junbo Jia, Luonan Chen，2023（搜索结果提示 Nature Communications 2025，未在页面确认）。
    - https://arxiv.org/abs/2311.10403
    - 对 RNA 速度场做 natural HHD：梯度部分 = 分化势景观，**旋转部分 = 细胞周期**。B+（"curl 大 = 世界在打转"的生物学实例）。
22. **Geometry-preserving vector field reconstruction of high-dimensional cell-state dynamics using ddHodge** — Maehara & Ohkawa，Nature Communications 16, 2025。
    - https://doaj.org/article/007796a287734316a66f86d09c360f65
    - Hodge 分解重建单细胞向量场的 gradient / curl / divergence，结论：小鼠胚胎发育基因表达基本是**势景观塑造的梯度系统**。B+。
23. **Curl Descent: Non-Gradient Learning Dynamics with Sign-Diverse Plasticity** — Ninou, Kadmon, Cayco-Gajic，2025/2026。
    - https://arxiv.org/abs/2510.02765
    - 学习动力学中的 curl 项：小则像梯度下降，大则破坏稳定甚至混沌，有时加速逃离鞍点。C+（学习动力学，不是世界动力学；但给"旋度大意味着什么"提供了理论参照）。
24. 其他工具类（C）：Espath 等 2023 物理信息谱学习求离散 HHD（https://arxiv.org/abs/2302.11061）；Richter-Powell, Lipman, Chen 2022 Neural Conservation Laws，无散度场参数化，含 Hodge 分解实验（https://arxiv.org/abs/2210.01741）；Casadei 2026 Graph Vector Field / Hodge 统一框架，提议作诊断但无实验（https://arxiv.org/abs/2605.18250）；Dus 2026 RL 的 Wasserstein 梯度流表述（https://arxiv.org/abs/2604.14765）。
25. **未检索验证、凭记忆提示**：Jin Wang 的 "landscape and flux" 理论（非平衡系统 = 势景观梯度 + 概率通量旋转，Advances in Physics 2015 左右）与 Cell 2 的诊断几乎同构，是生物物理界的标准工具；本轮配额用尽未能 fetch，建议补查。

### 负面结果 / 警示

- **纯下降的世界模型已知不够**：Friston 等 2021 在 Lorenz 系统上 solenoidal 部分占主导；HHD-GP 与 DHNN 的动机都是"只用梯度流学不了振荡/混沌系统"。所以 Cell 2 的"世界 = 下降者"若不带 Hodge 诊断，会在任何有周期的环境上失败——诊断不是附件而是必需品。
- **HHD 的可辨识性**：HHD-GP 指出必须加对称性先验才能辨识 gradient/curl 分量；有界域上 harmonic 分量不唯一。"旋度大 = 在打转"的量化需要先固定边界条件与度量，否则同一场的分解结果随坐标变。
- **Curl 可能有益**：Ninou 等 2025 显示适度 curl 加速学习；Velde 把 curl 解释为细胞周期（有意义的结构而非"没在下降"）。把 curl 单纯读成"不守规矩"是过度简化。
- **概念叙事拥挤**：FEP、Bayesian mechanics、Vanchurin 都在讲"世界/粒子在做梯度下降/推断"。Cell 2 若以哲学口号为主打，会被当成 FEP 的翻版；必须以可测的"学到的 F + 残差/旋度比"立身。

### 还剩什么没做

- 没人做过：**度量固定为 Fisher（来自学到的观测模型或潜空间的指数族参数化）+ 学标量势 F + 环境转移 = −G⁻¹∇F + 动作作为外加驱动**，在 RL / 视频 world model 基准上训练，并报告 Hodge 残差。JKOnet 系列只在 Wasserstein 度量、群体快照上做了同样的模板。
- 没人把 de Alfaro 等的 "natural deviation" 反过来用：作为**学 F 的训练损失**，其最优值作为"世界有多像一个 mirror-descent 学习者"的不变量。
- 没人在**合成可控环境**（已知是梯度流 / 已知是 replicator+周期 / 已知是 Lotka–Volterra 打转）上标定这个诊断的灵敏度与假阳性率。

### 可能的差异化点

- 做 **Fisher–Rao 版的 JKOnet**：对象从群体快照换成 agent 环境的状态分布 / 潜分布，度量换成 Fisher，势 F 可解释为"环境的隐含适应度 / 负自由能"。与 Wasserstein 版本并列比较，哪个度量下残差更小本身就是环境的几何指纹。
- **旋度比（curl ratio）作为不变量**：在 Fisher 度量下定义 Hodge 分解（而非 Euclidean），在 replicator（纯梯度）、带 mutation 的 replicator、rock–paper–scissors（纯旋转）等标定环境上先验证，再上真实环境。
- 以 CNS 的 natural deviation 为损失，做"世界的学习率 / 温度"估计——把"世界像学习者"从口号变成可估计的参数。
- 与 FEP 的关系写成"FEP 假设的分解，我们从数据估计并检验"，而不是再提一个原理。

### 覆盖信心：**中**
replicator/自然梯度理论、JKO/梯度流学习、HHD 工具三条线找得较全；FEP / Bayesian mechanics 线只覆盖了 Friston 系的几篇代表作，"Bayesian mechanics + 信息几何"的 2023–2026 综述（Ramstead 等）与 Jin Wang 的 landscape-flux 文献因配额用尽未能检索，是主要盲区。

### 搜索词（Cell 2，12 次完成 + 4 次被配额拒绝）
natural gradient flow world model learned potential Fisher metric dynamics / Helmholtz Hodge decomposition learned vector field neural dynamics curl gradient / natural gradient flow as model of evolution "world model" OR "environment dynamics" Fisher metric potential learned replicator neural / Helmholtz decomposition learned dynamics "gradient" "rotational" neural ODE "symplectic" "potential" GENERIC metriplectic / active inference free energy principle information geometry natural gradient flow 2024 2025 "Fisher" / Hodge decomposition diagnose learned dynamics "curl-free" "divergence-free" "gradient flow" / "natural gradient" "replicator" Bayesian inference equivalence "Fisher" Harper Shahshahani / learning potential function gradient flow from data "Wasserstein gradient flow" OR "Fisher-Rao gradient flow" "JKO" population / Friston "natural gradient" OR "information geometry" generalised synchrony Bayesian mechanics（被拒） / "mirror descent" dynamics "world" as learner Vanchurin / Helmholtz-Hodge decomposition single-cell RNA velocity "gradient" "curl" potential landscape diagnostic / Fisher-Rao gradient flow learned potential "unbalanced" Wasserstein-Fisher-Rao population dynamics neural 2025 / "Bayesian mechanics" Ramstead information geometry（被拒） / Vanchurin "evolution as multilevel learning"（被拒） / "natural gradient" world model latent dynamics "Fisher information" model-based RL（被拒）

---

## Cell 7：两种直线（对偶平坦世界模型）

### 撞车等级：**B−**（数学结构的动力学版本有 1993 年的理论根源和 2026 年的 Gaussian 特例，但"先判别每段轨迹是 e-直线还是 m-直线，再在对应平坦坐标里学线性动力学"这一学习/分类框架没有找到任何先例）

### 最近邻列表

1. **Dynamical Systems on Statistical Models** — 藤原彰夫，1993，RIMS 講究録 822。
   - https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/0822-04.pdf
   - 见 Cell 2 第 4 条。核心定理：对偶平坦结构下散度势的梯度流**沿对偶测地线**收敛；Legendre 诱导结构下流在对偶坐标里**线性**（η̇ = −η）；指数族上的 MLE 流沿 m-测地线。还展望了二阶（Lagrangian 型）类比——在对偶坐标里化为独立谐振子。
   - 接近程度：**B+**。"在平坦坐标里动力学是线性的"对一大类梯度流是定理，不是假设；Cell 7 的新意不能放在"线性"本身，要放在**判别 + 分段 + 从数据学**。
2. **Minimization of Functions on Dually Flat Spaces Using Geodesic Descent Based on Dual Connections** — Gaku Omiya, Fumiyasu Komaki，2025-12。
   - https://arxiv.org/abs/2512.09358
   - 对偶平坦空间上的测地下降：m-测地线更新理论上一步到达 MLE；e-测地线更新在测地完备时免约束。
   - 接近程度：B。明确对比了 e-步与 m-步，但是优化算法，不是对观测动力学的分类/建模。
3. **Stochastic Processes as Non-Metric Geodesics in Information Geometry** — T. Koide, A. van de Venn，2026-09。
   - https://arxiv.org/abs/2609.08870
   - Gaussian 流形上 α-联络测地线分三类：boundary-connecting（α > −1，对应常扩散的线性 SDE / 广义 Brownian bridge，极限为 OU 与自由扩散）、critical（α = −1，无 SDE 实现）、non-boundary-connecting（α < −1，无 SDE 实现）。**提出用 OU 极限的 (D, γ, R) 闭式反推观测过程的 α**（"operational method to determine the underlying connection parameter"）。注意：他们的约定里"期望坐标中的直线"对应 α = 1，与常见 m-联络 = α = −1 的记号相反，引用时要核对约定。
   - 接近程度：**B+ / 接近 A 的特例**。这是"看一段过程更像哪条 α-直线"最直接的先例，但：只有 Gaussian 一维、只有闭式反推、没有学习、没有 e=选择 / m=迁移的解释、不做序列建模。
4. **Notes on information geometry and evolutionary processes** — Marc Toussaint，2004。
   - https://arxiv.org/abs/nlin/0408040
   - 均匀交叉（crossover）= 沿 p 与其一阶约化 p^(1) 之间的 **m-测地线走一步 / 正交投影**，单调减少 θ 坐标中的相互作用项；选择只描述为降熵的随机映射，**没有**放到 e-测地线上。
   - 接近程度：B。支持"混合算子 = m-运动"这一半；"选择 = e-运动"那一半文献里没明说。
5. **Information Geometry of Polytopes**（Warwick WOGAS 讲义）— Nihat Ay，2010。
   - https://warwick.ac.uk/fac/sci/statistics/research/crism/workshops/wogas/programme/ay.pdf
   - replicator 向量场 = Fisher 度量下平均适应度的梯度；常适应度时解为 p e^{tf}/⟨e^{tf}⟩——这正是 e-测地线的形式（**审计者推断，讲义未明说**）；m-测地线定义为混合 (1−t)p + tq，但未与演化过程挂钩。根源。
6. **Information Geometry and Evolutionary Game Theory** — Marc Harper，2009。
   - https://arxiv.org/abs/0911.1383
   - replicator 的解可实现为指数族（x_i = exp(v_i − G)）；不讨论 m-测地线、混合/迁移或对偶平坦结构。根源（e-侧）。
7. **The geometry of the Fisher selection dynamics** — Shapovalov & Evdokimov，1998。
   - http://arxiv.org/abs/physics/9805006
   - 选择动力学 = 某 projectively Euclidean、equiaffine 联络的测地运动 + 有效外场；仿射曲率 = 等位基因耦合。B（"选择是某个平坦联络下的直线"的早期形式）。
8. **Projection based dimensionality reduction for measure valued evolution equations in statistical manifolds** — Damiano Brigo, Giovanni Pistone，2016。
   - https://arxiv.org/abs/1601.04189
   - 把 Fokker–Planck / 非线性滤波 SPDE 投影到有限维**指数族或混合族**上，投影与 Fisher–Rao 度量一致；充分统计量为 backward 算子特征函数时 Fisher–Rao 投影给出 MLE。
   - 接近程度：B。是"把一个演化方程分别投到 e-平坦与 m-平坦族"的先例，但目标是降维不是分类，也没有从数据学。
9. **WFR-FM** — Peng 等，ICLR 2026（见 Cell 2 第 16 条）。
   - https://arxiv.org/abs/2601.06810
   - Wasserstein（运输）+ Fisher–Rao（增长/消亡）两种运动的联合学习。B：这是 OT 世界里"两种运动分解"的对应物，但它的两种是"搬运 vs. 生灭"，与 Cell 7 的"混合（m）vs. 倍乘（e）"不同——m-混合是固定支撑点之间的质量再分配，Wasserstein 是支撑点本身移动，这其实是**三种**而不是两种直线。
10. **Information Geometry of Gradient Flows** — Shintaro Yoshizawa，2026-08。
    - https://arxiv.org/abs/2608.21152
    - 奇异对偶平坦几何（退化 Hessian、blow-up、Łojasiewicz 收敛）下的梯度流、Bures–Wasserstein 流、双括号流。纯理论，无学习。C+。
11. **An Extension of Least Angle Regression Based on the Information Geometry of Dually Flat Spaces** — Hirose & Komaki，2009。
    - https://www.keisu.t.u-tokyo.ac.jp/data/2009/METR09-09.pdf
    - e/m-测地线与 m-投影做变量选择路径；与动力学无关。C（但说明 Komaki 组长期在做 e/m 几何算法，Omiya & Komaki 2025 是其延续）。
12. **Information Geometry in Population Genetics** — Tat Dat Tran（与 Jost、Hofrichter），IGAIA 摘要 + MPI 讲座页。
    - https://igaia.utia.cz/abstracts/tran.pdf ； https://www.mis.mpg.de/events/event/an-introduction-to-information-geometry-and-mathematical-population-genetics
    - 用信息几何处理 Wright–Fisher 的漂变/选择/突变/重组；页面无细节。其 2017 年专著（Hofrichter–Jost–Tran, *Information Geometry and Population Genetics*）是 Cell 7 群体遗传根源的标准参考（凭记忆，本轮未 fetch）。C+。
13. **Optimizing information flow in Gene Regulatory Networks: a geometric perspective** — García-Ariza, Bravetti, Padilla, Romero-Arias，2025。
    - https://arxiv.org/abs/2509.08167
    - 基因调控网络的最优信息传递轨迹 = Gaussian Fisher 流形上 KL 的梯度流；只用度量测地线，不用 e/m。C。
14. Amari RIKEN 讲义（https://www2.riken.jp/lab/theory/colloquium/amari.pdf）：标准对偶平坦教材内容，e/m-投影用于 EM、belief propagation，无动力学。根源。

### 负面结果 / 警示

- **"二选一"可能太粗**：Koide & van de Venn 显示 Gaussian 上只有 α > −1 的一类测地线有 SDE 实现，另两类没有对应的随机过程——"每段轨迹是某条直线"的先验在很多 α 下根本没有物理实现。更实际的是用连续的 α（或 e/m 的混合权重）而不是硬分类。
- **真实群体动力学是 e 与 m 同时发生**：选择 + 迁移 + 突变同时作用时，轨迹既不是 e-直线也不是 m-直线（WFR-FM 也表明真实单细胞数据同时需要运输与增长两项）。分段判别只在"某时段某机制主导"时成立，需要先构造这种环境。
- **线性不是新意**：Fujiwara 1993 已证明一大类 Fisher 梯度流在对偶坐标里线性。审稿人会问"你比 1993 多了什么"——答案必须是判别器 + 从观测学坐标（encoder 到指数族参数）+ 不变量。
- **记号陷阱**：α = ±1 哪个是 e 哪个是 m 在不同文献（Amari vs. Koide & van de Venn）约定相反；"期望坐标里直线"也不等价于某些文献的 m-测地线定义。写论文时要固定一套记号并显式说明。

### 还剩什么没做

- 没有任何工作把观测序列 encode 进一个**对偶平坦潜流形**（指数族参数化），再**判别每段轨迹在 θ 坐标还是 η 坐标里更直**，并据此在对应坐标里学线性转移。
- 没有"α-计"（α-meter）：从数据估计每段轨迹最直的 α-联络，作为连续不变量；Koide & van de Venn 只给了 Gaussian OU 极限的闭式。
- 没有把 **e = 选择 / 贝叶斯更新、m = 迁移 / 扩散** 这个解释变成可检验的实验：在合成的 replicator+migration 环境里，看判别器能否正确分离两种机制的时段。

### 可能的差异化点

- 做一个**最小可验证实验**：单纯形上合成轨迹（纯选择段、纯迁移段、混合段），指标是"e-平坦残差 vs. m-平坦残差"随时间的曲线；再推广到 Gaussian（可对照 Koide & van de Venn 的闭式 α）。
- 把判别结果解释为**"谁在学习"**：e-直线段 = 世界在做贝叶斯更新/选择（与 Cell 2 的自然梯度流世界相接），m-直线段 = 世界在混合/扩散（没有信息增益）。这让 Cell 7 成为 Cell 2 的诊断前端，而不是孤立的几何练习。
- 明确区分三种"直线"：m（质量再分配）、e（倍乘重加权）、Wasserstein（支撑点搬运），与 WFR-FM 对照——这是 OT 社区还没讲清楚的区分。
- 引 Fujiwara 1993 与 Omiya–Komaki 2025 作为"坐标里线性"的理论背书，把贡献定位为"从数据判别并学习"。

### 覆盖信心：**低–中**
只完成 2 次专门搜索（配额在 Cell 2 阶段耗尽），其余靠交叉引用 fetch。未搜索的方向：Nakamura 的可积梯度系统（1993–1994）、Amari/Takeuchi 等关于时间序列/ARMA 谱的对偶平坦结构、信息几何在 Markov 链 / 隐 Markov 模型上的 e/m 结构、Hofrichter–Jost–Tran 专著正文。这些都可能含有"e/m 几何 + 时间序列"的工作，建议补查。

### 搜索词（Cell 7，2 次完成；其余为计划中未执行）
dually flat manifold dynamics e-geodesic m-geodesic time series modeling / information geometry evolutionary dynamics selection migration mixture geodesic exponential geodesic ／ 未执行：Amari dual coordinates latent dynamics；Nakamura completely integrable gradient system dually flat；information geometry time series e-flat m-flat Markov；mixture family vs exponential family trajectory classification

---

## 总结：三条里哪条最危险、哪条最空

**最危险（技术撞车）：Cell 1。** Friedl 等（ICLR 2025 + 2026 续作）已经在潜空间上学 SPD 度量 + Euler–Lagrange 动力学 + 外力输入，机器几乎同构；Beik-Mohammadi / Klein / Xu 已经把"障碍 = 度量，运动 = 测地线"做成机器人方法；LeCun 组的 Temporal Straightening 在 JEPA world model 里走了反方向（压平曲率）；"Geometric World Models" 的名字被 CVPR 2026 的 GeoWorld 占了；还有一个无名 PyPI 包叫 latent-riemannian-world 带 RiemannianRSSM。另有一个根本性的科学风险：仅凭轨迹，"曲率"与"力"不可辨识（Jacobi 度量；Koide & van de Venn 2026 明说同一恢复力可归为曲率或非度量性）。Cell 1 要活下来，必须把度量钉在观测模型的 Fisher 拉回上，并拿出"曲率图 = 障碍图"这种别人没报告过的不变量。

**最危险（叙事撞车）：Cell 2。** "世界在做自然梯度下降 / 贝叶斯更新"这一叙事已被 Harper、de Alfaro 等（Conjugate Natural Selection）、FEP（Friston 的 f = (Q−Γ)∇ℑ 分解）、Vanchurin 多方占据；"学势 + 固定度量梯度流"的模板在 Wasserstein 下已有 JKOnet 一族；HHD 诊断在生物学（Velde、ddHodge）与 ML（HHD-GP, NeurIPS 2024）都已成熟。但**Fisher 度量 + 学势 + Hodge 残差 + world model 基准**这个具体组合没人做，而且 CNS 的 "natural deviation" 几乎是现成的损失/不变量。技术空位清楚，风险在于被当作 FEP 翻版——必须以可估计量立身。

**最空：Cell 7。** 只有 Fujiwara 1993（理论）、Omiya–Komaki 2025（优化）、Koide & van de Venn 2026（Gaussian 特例、闭式 α 反推）、Toussaint 2004（crossover = m-步）这几块零散的砖，没有任何"判别 + 学习"框架。空的代价是两点：一是覆盖信心低（本轮只做了 2 次专门搜索），二是必须自己证明"硬分 e/m"在真实数据上有意义（真实过程常是两者叠加），建议改成连续 α 估计并把它做成 Cell 2 的诊断前端。

**建议优先级**：先把 Cell 7 的合成实验做出来（成本最低、最空、能给 Cell 2 提供"世界在哪些时段像学习者"的证据），Cell 2 以 Fisher 版 JKOnet + natural deviation 为主线，Cell 1 只有在能展示"曲率图 = 障碍图"并正面打败 Temporal Straightening 的前提下才值得投入。


<!-- ===== 报告 G2 ===== -->

# 文献审计 G2：Cell 3 / Cell 9 / Cell 10

审计日期：2026-11-01。方法：WebSearch（Cell 3 共 26 次、Cell 9 共 11 次；之后本轮所有 agent 共享的搜索预算耗尽，Cell 10 **零次搜索**，只能用 WebFetch 逐篇核对我已知 arXiv 编号的论文）+ WebFetch 读 arXiv 摘要页/HTML 全文核对。所有 URL 均为搜索或 fetch 实际返回的地址。会议/期刊只在页面确认（或 DOI 前缀明确指向）时标出；标"据记忆"者未经页面确认。引用均为转述。

---

## Cell 3：搬运还是改权（Wasserstein–Fisher–Rao 世界模型）

### 撞车等级：**A（方法层）/ B（主张与试验场层）**

"学一个速度场 v 和一个生长率场 r，按 WFR 最小作用量把分布变化分配给搬运与生灭"这件事，在单细胞轨迹推断社区 2023–2026 已被做成一条完整的方法链（TIGON → DeepRUOT → VGFM → UPFI → WFR-FM → WFR-MFM → USB），其中 WFR-FM 是 ICLR 2026、DeepRUOT 是 ICLR 2025 oral、UPFI 是 NeurIPS 2025。几何结构、损失、网络输出（v 与标量 r）都和你们的 Cell 3 相同。所以**作为"方法"它已经饱和**。但三件事没人做：(1) 把 (v, r) 分解当作过程的"指纹"（搬运/改权比）来刻画动力学类型；(2) 把它当作世界模型（有 action、多步 rollout、latent）；(3) 对任何元胞自动机 / Lenia 做 OT 或 WFR 分析。"Lenia 的 glider 有多少是搬运"这个问题目前没有任何文献触及。

### 最近邻工作

**A 层：同样几何 + 学 v 与 r（单细胞动力学）**

1. **WFR-FM: Simulation-Free Dynamic Unbalanced Optimal Transport** — Qiangwei Peng（Peijie Zhou、Tiejun Li、Qing Nie、Lei Zhang 等），ICLR 2026（arXiv 页面确认）— https://arxiv.org/abs/2601.06810 ；https://proceedings.iclr.cc/paper_files/paper/2026/hash/2030d35841f8617ab22cca913f55697e-Abstract-Conference.html
   做了什么：WFR flow matching，同时回归一个位移向量场和一个标量生长率场，声称其损失最小值精确恢复 WFR geodesic；输出 time-varying growth field。
   接近程度：**最高**。几何、学习目标、输出完全一致；差别只在：目标是 scRNA-seq 快照插值，不谈指纹/比值，不谈世界模型。其 related work 列表（Sha 2024、Zhang 2025、Sun 2025、Peng 2024 的 WFR-ODE 系；Huguet/Tong 的 neural-ODE 系；Eyring/Cao/Corso/Wang 的 unbalanced FM 系；Neklyudov 的 Action Matching / WLF）是你们写 related work 的现成骨架。

2. **WFR-MFM: One-Step Inference for Dynamic Unbalanced Optimal Transport** — Xinyu Wang（Peijie Zhou、Tiejun Li 等），arXiv 2026-01 — https://arxiv.org/abs/2601.20606
   做了什么：mean-flow 版 WFR，学"平均速度场 + 平均质量增长场"实现跨任意时间区间的一步生成。
   接近程度：高（同上组的后续）。

3. **Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport (DeepRUOT)** — Zhenyi Zhang（Tiejun Li、Peijie Zhou），ICLR 2025 oral（arXiv 页面确认）— https://arxiv.org/abs/2410.00844 ；https://proceedings.iclr.cc/paper_files/paper/2025/hash/32b8a612105de5c22db337b774ce7b61-Abstract-Conference.html
   做了什么：Regularized unbalanced OT（带扩散的 WFR），从快照推含生灭的随机动力学，并与 Schrödinger bridge 挂钩。
   接近程度：高。

4. **Reconstructing growth and dynamic trajectories from single-cell transcriptomics data (TIGON)** — Yutong Sha（Qing Nie 组），Nature Machine Intelligence 6:25–39（2024；NSF PAR 页面确认期刊与卷页）— https://par.nsf.gov/biblio/10503782 ；https://www.nature.com/articles/s42256-023-00763-w （nature 页面本次 fetch 失败）
   做了什么：明确基于 WFR 距离的无量纲形式，用神经网络同时重建轨迹与种群增长。
   接近程度：高。是这条线的起点。

5. **Inferring stochastic dynamics with growth from cross-sectional data (UPFI)** — Stephen Zhang（Maddu、Xiaojie Qiu、Chardès），NeurIPS 2025（arXiv 页面确认）— https://arxiv.org/abs/2505.13197
   做了什么：Lagrangian 形式 Fokker–Planck+growth，分两步学 score 与 (drift, growth)；**明确用 WFR energy 作为正则项以保证唯一性**。
   接近程度：高，且含本条最重要的负面结果（见下）。

6. **Joint Velocity-Growth Flow Matching (VGFM)** — Dongyi Wang（Peijie Zhou、Jian Sun 等），arXiv 2025-05（v2 2025-11）— https://arxiv.org/abs/2505.13413
   做了什么：从 static semi-relaxed OT 的两段视角导出理想动力学，联合学速度与质量增长。
   接近程度：高。

7. **Beyond Continuity: … Discrete Branching Dynamics from Single-cell Snapshots (USB)** — Junda Ying（Peijie Zhou、Lei Zhang），arXiv 2026-05 — https://arxiv.org/abs/2605.00545
   做了什么：把生灭做成离散跳跃的 unbalanced Schrödinger bridge，批评连续 unbalanced OT 把质量当流体。
   接近程度：中高（是对连续 WFR 的"改权必须离散"的反方案，Lenia 的逐点 growth 恰恰是连续的，反而对你们有利）。

8. **A Neural Network Framework for High-Dimensional Dynamic Unbalanced Optimal Transport** — Wei Wan（Chenglong Bao、Zuoqiang Shi 等），arXiv 2024-09（math.OC）— https://arxiv.org/abs/2409.13188
   做了什么：Lagrangian 离散 + Monte Carlo，网络同时表示速度场和 source 函数，GKL 松弛终端约束。
   接近程度：中高。

**B 层：同类几何用于动力学/生成，但主张不同**

9. **A Computational Framework for Solving Wasserstein Lagrangian Flows** — Kirill Neklyudov（Brekelmans、Tong、Makhzani 等），arXiv 2023-10（v3 2024-07；ICML 2024 据记忆）— https://arxiv.org/abs/2310.10649
   做了什么：把动能（几何）与势能（路径正则）参数化成 Lagrangian，统一 SB / unbalanced OT / 物理约束 OT，对偶形式无需模拟。
   接近程度：中高。你们"按 WFR 最小作用量分配"正是其一个 Lagrangian 实例。

10. **Action Matching** — Kirill Neklyudov（Makhzani），ICML 2023（PMLR 页面确认）— https://proceedings.mlr.press/v202/neklyudov23a.html
    做了什么：从独立快照学连续动力学，并扩展到有质量生灭的动力学。
    接近程度：中。

11. **Unbalanced Diffusion Schrödinger Bridge** — Matteo Pariset（Bunne、Krause、De Bortoli），arXiv 2023-06 — https://arxiv.org/abs/2306.09099
    做了什么：推导含 killing/birth 项 SDE 的时间反演，得 unbalanced 神经 SB。
    接近程度：中。

12. **Unbalancedness in Neural Monge Maps Improves Unpaired Domain Translation (UOT-FM)** — Luca Eyring（Theis 等），arXiv 2023-11（v2 2024-03）— https://arxiv.org/abs/2311.15100
    做了什么：把 unbalanced 加进任意 neural Monge map / OT flow matching。
    接近程度：中低（静态 unbalanced，不分解 v/r）。

13. **Unbalanced Regularized Optimal Mass Transport with Applications to Fluid Flows in the Brain (urOMT)** — 第一作者 Xinan Chen（页面未列作者；据 GitHub 线索与记忆，Tannenbaum 组），arXiv 2301.11228 — https://arxiv.org/abs/2301.11228
    做了什么：对 DCE-MRI 图像序列拟合 ∂ρ/∂t + ∇·(ρv) = σΔρ + χρr，数值上把每步拆成"质量增减 / 平流 / 扩散"三子步；文中用权重 α 平衡速度与 source，但**没有报告任何"搬运 vs 改权"的比值**。
    接近程度：中高——这是最接近"对图像序列做 WFR 分解"的工作，且是成像而非单细胞；它的 Gaussian 示例里有"ρ₀ = 搬运部分 + source 部分"的概念拆分，但未量化。

14. **An unbalanced Optimal Transport splitting scheme for general advection-reaction-diffusion problems** — Thomas Gallouët（Laborde、Monsaingeon），arXiv 2017-04（math.AP）— https://arxiv.org/abs/1704.04541
    做了什么：对 advection-reaction-diffusion 方程交替做 Wasserstein 极小化运动（搬运/扩散）与 Fisher–Rao 极小化运动（反应），证明弱解存在。
    接近程度：中。这是"Lenia 这类 reaction-diffusion 型系统天然可拆成 W 步 + FR 步"的数学依据；Kojima & Ikegami 已把 asymptotic Lenia 写成 reaction-diffusion（见 17）。

**根源（已核对）**：Chizat–Peyré–Schmitzer–Vialard《An Interpolating Distance between Optimal Transport and Fisher-Rao》(2015, https://arxiv.org/abs/1506.06430) 与《Unbalanced OT: Dynamic and Kantorovich Formulations》(2015/2018, https://arxiv.org/abs/1508.05216)；Kondratyev–Monsaingeon–Vorotnikov (2015, https://arxiv.org/abs/1505.07746)；Liero–Mielke–Savaré Hellinger–Kantorovich (2015, Invent. Math. DOI 页面确认, https://arxiv.org/abs/1508.07941)；Waddington-OT bioRxiv 版 (https://www.biorxiv.org/content/10.1101/191056v1)。

**试验场：Lenia / CA 侧**

15. **Flow-Lenia**（两版）— Erwan Plantec（Hamon、Etcheverry、Chan、Oudeyer、Moulin-Frier），arXiv 2022-12 (https://arxiv.org/abs/2212.07906)；期刊版 Artificial Life 31(2):228–248, 2025 (https://arxiv.org/abs/2506.08569)；模型页 https://sites.google.com/view/flowlenia/model
    做了什么：把 Lenia 的 growth/affinity 图 U 改读成流场 F = (1−α)∇U − α∇A_Σ，用 reintegration tracking 搬质量，实现质量守恒；明确对比了 Lenia 的 "A + dt·U 再 clip" 会创造/销毁质量。**不提 optimal transport**。
    接近程度：**对试验场而言最高**。Flow-Lenia 本身就是"同一 growth 场的纯搬运解读"，标准 Lenia 是"纯改权解读"，你们的 WFR 分解正好是二者之间的连续谱——这是现成的叙事。

16. **MaCE: General Mass Conserving Dynamics for Cellular Automata** — Vassilis Papadopoulos（Guichard），arXiv 2025-07（DOI 指向 ISAL 2025 proceedings）— https://arxiv.org/abs/2507.12306
    做了什么：可附加到任意 CA 的质量守恒规则，Lenia 上得到大量 soliton。不提 OT。
    接近程度：中（另一个"纯搬运"参照系）。

17. **Implementation of Lenia as a Reaction-Diffusion System** — Hiroki Kojima（Takashi Ikegami），arXiv 2023-05（DOI 指向 ALIFE 2023 proceedings）— https://arxiv.org/abs/2305.13784 ；同组 **The Glider Equation for Asymptotic Lenia**（Kojima、Yevenko、Ikegami，arXiv 2025-08，https://arxiv.org/abs/2508.04167）
    做了什么：asymptotic Lenia 写成 PDE = 扩散项 + 局部反应项；后一篇解析推导 glider 条件并用作 loss 优化 glider。
    接近程度：中。结构上已把 Lenia 拆成"扩散（搬运）+ 反应（改权）"，但没有量化比例，也没有 OT 视角。

18. **Analysis of dynamically stable patterns in a maze-like corridor using the Wasserstein metric** — Ryosuke Ishiwata（Kinukawa、Sugiyama），Scientific Reports 2018 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5913310/
    做了什么：对 2d-OV 粒子模型形成的集体图案用 Wasserstein 距离比较。
    接近程度：低（是我能找到的唯一"用 OT 度量分析自组织图案"的工作，且不是 CA）。

未找到：任何 "Lenia"+"optimal transport"、"cellular automata"+"unbalanced"/"Wasserstein" 的动力学分析论文（专门查了 4 次）。

### 负面结果 / 必须正视的坑

- **(v, r) 分解不可辨识** — UPFI（Zhang et al., NeurIPS 2025）Prop. 2.1 / Cor. 2.2：同一条密度序列可由不同的 drift+growth 组合生成（连 autonomous drift 也救不了；OU 情形可显式构造），所以他们加 WFR energy 正则才得到唯一解，并承认非线性情形的辨识性仍未解决。**含义：你们的"搬运/改权比"不是过程的内在量，而是"给定 WFR 尺度参数 δ 后的最小作用量选择"。**
- **WFR 存在截断距离（据记忆，页面未确认）**：静态 HK/WFR 代价在超过某个与尺度参数成正比的距离后变为 +∞（LMS 归一化下 π/2、Chizat 等 δ 参数化下 πδ），超过它就"销毁再创造"比搬运便宜。所以比值必然是 δ 的函数——这既是坑也是差异化点（见下）。
- USB (2026) 批评连续 unbalanced OT 把生灭当连续流体——对单细胞是缺点，对 Lenia（更新本来就是连续逐点 growth）反而是贴切的。

### 还剩什么没做

- 没人把 WFR 分解结果（v, r 或其能量占比）作为**动力学的描述量/指纹**；所有工作只把它当插值/生成工具。
- 没人在**有完整密度曲线**（每帧都观测到 p_t）的系统上做 WFR 分解——单细胞方法都在解决"只有稀疏快照"的问题，而 Lenia/视频给你全轨迹，问题从"推断"变成"归因"。
- 没有任何 CA/Lenia 的 OT 分析；Flow-Lenia 与 MaCE 只提供了"纯搬运"的对照系。

### 可能的差异化点

- 把比值做成 **δ-谱**：ratio(δ) 曲线而非单个数——随 δ 从 0（纯 Fisher–Rao）到 ∞（纯 Wasserstein）扫描，不同过程（glider 平移、生长、死亡、分裂）给出不同形状的曲线；这把不可辨识性变成了特征。
- 用 Lenia ↔ Flow-Lenia 作为两个极端的**有真值**对照：同一 growth 场、两种实现，WFR 分解应当对 Flow-Lenia 给 ≈1 的搬运占比、对标准 Lenia 的 glider 给中间值；这是审稿人能理解的 sanity check。
- 走"世界模型"方向：把 (v, r) 作为 latent 转移的两路参数化（搬运头 + 改权头），考察它对 OOD 预测/长程 rollout 的影响——单细胞社区没有做过多步 rollout 与 action 条件化。

### 覆盖信心：**高**
WFR-FM 的 related-work 列表与我检索到的集合高度重合，说明主干没有漏。盲区：2026 年下半年 unbalanced FM 新作；Lenia 社区的 workshop 论文（ALIFE 2026）。

### 搜索词（26 次）
Wasserstein Fisher-Rao dynamics learning neural trajectory inference unbalanced；unbalanced optimal transport neural trajectory inference single-cell growth 2025；optimal transport cellular automata Lenia Wasserstein；"Wasserstein-Fisher-Rao" generative model flow matching unbalanced 2024 2025 arXiv；DeepRUOT regularized unbalanced optimal transport stochastic dynamics snapshots ICLR 2025；action matching unbalanced Wasserstein Fisher-Rao Neklyudov Makhzani；Lenia glider mass transport velocity field "optimal transport" OR "Wasserstein"；transport versus growth decomposition "unbalanced optimal transport" reaction-diffusion dynamics learning；TIGON dynamical unbalanced optimal transport growth trajectories single-cell；"unbalanced" optimal transport "cellular automata" OR "Game of Life" Wasserstein；Wasserstein Fisher-Rao "world model" OR "video prediction" OR "spatiotemporal"；unbalanced optimal transport flow matching generative 2025 "mass" "teleportation" OR "birth-death"；Lenia "mass" "growth" glider motion "advection" OR "transport" Wasserstein gradient flow；"Wasserstein-Fisher-Rao" OR "Hellinger-Kantorovich" gradient flow reaction-diffusion "Lenia" OR "cellular automaton"；TIGON "unbalanced" "dynamical optimal transport" Sha；"optimal transport" neural cellular automata Wasserstein morphogenesis；"unbalanced optimal transport" video OR "image sequence" OR "fluid" "source term" optical flow；MIOFlow TrajectoryNet unbalanced extension growth；"Wasserstein Lagrangian Flows" Neklyudov unbalanced；"Fisher-Rao" "Wasserstein" decomposition "how much" mass transport versus reweighting fingerprint；unbalanced Schrödinger bridge diffusion 2024 2025 "birth" "death"；"Lenia" "optimal transport"；unbalanced optimal transport identifiability transport vs growth ambiguity "non-identifiable"；Lenia dynamics learned neural network "world model" OR "surrogate" OR "emulator"；"Hellinger-Kantorovich" OR "Wasserstein-Fisher-Rao" learning dynamics neural network 2025 2026 spatiotemporal images OR PDE；"cellular automata" "earth mover" OR "Wasserstein" glider spaceship motion transport。

---

## Cell 9：时间即弧长（信息时间世界模型）

### 撞车等级：**B**

"用 Fisher–Rao 弧长（information length）累积量当时间"这个**概念**已经在物理里被反复提出：Kim–Hollerbach 系列把 information length 定义为"以 PDF 变化特征时间 τ(t) 为单位度量的总时长"L = ∫dt/τ，并用它比较不同模型、诊断相变、预测突变；Tenkès–Hollerbach–Kim 2017 已明确说"以 information length 为单位看，logistic 与 Gompertz 模型更相似"——这就是你们的"重参数化后动力学更统一"的雏形；2026 年还有一篇物理预印本直接把"时钟时间 = 累积 Fisher 距离"当纲领。但**没有任何工作把它用于学习动力学 / 世界模型**：没有人先按信息弧长重参数化再学转移模型，也没有人检验"事件在信息时间里等间距/更可预测"。ML 侧最接近的是三类不同口味的"内在时间"：训练动力学按 Fisher–Rao geodesic 进度重索引（Mao et al. PNAS 2024）、surprise 触发的主观时间尺度世界模型（Zakharov 2020）、以及 2026 年的（欧氏）弧长时间重参数化 neural-ODE（TOTR）。

### 最近邻工作

**根源（已核对）**
- Wootters, *Statistical distance and Hilbert space*, Phys. Rev. D 23, 357 (1981) — https://link.aps.org/doi/10.1103/PhysRevD.23.357
- Crooks, *Measuring thermodynamic length*, PRL 99, 100602 (2007) — https://arxiv.org/abs/0706.0559 ；Sivak & Crooks, *Thermodynamic metrics and optimal paths*, PRL 2012 — https://arxiv.org/abs/1201.4166
- Ito, *Stochastic thermodynamic interpretation of information geometry*, PRL 2018 — https://arxiv.org/abs/1712.04311 ；Ito & Dechant, PRX 2020 — https://arxiv.org/abs/1810.06832 （temporal Fisher information 给出速度极限）
- Nicholson, Garcia-Pintos, del Campo, Green, *Time–information uncertainty relations in thermodynamics*, Nature Physics 2020 — https://arxiv.org/abs/2001.05418 （瞬态涨落的时间尺度由 Fisher information 普适地界定——可作"信息时间是内在时钟"的理论依据）

**B 层：information length 作为动力学诊断（物理，非学习）**

1. **Signature of nonlinear damping in geometric structure of a nonequilibrium process** — Eun-jin Kim & Rainer Hollerbach，Phys. Rev. E 95, 022137 (2017) — https://eprints.whiterose.ac.uk/id/eprint/112141/7/PhysRevE.95.022137.pdf
   做了什么：定义 τ(t)⁻² = ∫(∂ₜp)²/p，L(t) = ∫dt/τ，解释为"统计可分辨状态的总数"，证明 L 依赖路径；线性阻尼 L∝y₀，三次阻尼 L∝y₀^m。
   接近程度：高（就是你们的时钟定义，但只作为诊断量）。

2. **Time-dependent PDFs and information geometry in stochastic logistic and Gompertz models** — Lucille-Marie Tenkès（Hollerbach、Kim），J. Stat. Mech. 2017（DOI 页面给出）— https://arxiv.org/abs/1708.02789
   做了什么：**以 information length 为单位重新度量演化，两种增长模型变得更相似**。
   接近程度：**最高**——这是"信息时间下动力学更统一"命题的已有版本，只是没有学习、没有预测检验。

3. **Geometric structure and information change in phase transitions** — Kim & Hollerbach，PRE 95, 062107 (2017) — https://pureportal.coventry.ac.uk/en/publications/geometric-structure-and-information-change-in-phase-transitions/
   做了什么：有序–无序循环相变的 toy model，用 L 区分不稳定性驱动（扩散型）与峰移动驱动（平流型）两种路径，并发现一个"信息变化时间尺度与噪声无关"的 geodesic regime。
   接近程度：高（直接关联你们"相变在信息时间里是否等间距"的检验）。

4. **Information Geometric Theory in the Prediction of Abrupt Changes in System Dynamics** — A.-J. Guel-Cortez & Eun-jin Kim，Entropy 23(6):694 (2021) — https://pureportal.coventry.ac.uk/en/publications/information-geometric-theory-in-the-prediction-of-abrupt-changes-/
   做了什么：非自治 Kramers 方程突扰动，information length 比 information flow 更早预测突变发生。
   接近程度：高（"死亡/突变事件在信息时间里更可预测"的先例，但仍是诊断不是重参数化）。

5. **Information Geometry of Nonlinear Stochastic Systems** — Hollerbach, Dimanche, Kim，Entropy 20:550 (2018) — https://eprints.whiterose.ac.uk/134222/7/entropy-20-00550.pdf ；综述 Kim, *Information Geometry, Fluctuations, Non-Equilibrium Thermodynamics, and Geodesics in Complex Systems*, Entropy 23:1393 (2021) — https://www.mdpi.com/1099-4300/23/11/1393 ；poster Guel-Cortez 2022 — https://pureportal.coventry.ac.uk/en/publications/information-length-in-dynamical-systems/
   接近程度：中（给出 Γ = dL/dt 与熵产、热流的不等式；综述未提出把 L 当时间坐标）。

6. **Fisher-Informational Time: A Causal-Geometric Framework for Emergent Clock Time** — J. Sumaya-Martinez，arXiv 2026-05（quant-ph）— https://arxiv.org/abs/2605.03958
   做了什么：把时钟时间定义为沿因果可容许轨迹累积的 Fisher 几何距离 Λ_F，讨论 Schrödinger 动力学的重参数化，对比 Page–Wootters / thermal time。
   接近程度：概念上 **A-级重合**（"时间 = 累积 Fisher 距离"），但纯物理、无学习、无数据。写论文必须引。

7. **Unified speed limits … via temporal Fisher information** — Nishiyama & Hasegawa，arXiv 2025-04（APS DOI）— https://arxiv.org/abs/2504.04790
   接近程度：低–中（temporal Fisher information 上有熵产界、下有 Bhattacharyya 距离界——可用于论证信息时间的"速度"有物理上下界）。

**B 层：ML 里的"内在时间"/时间重参数化**

8. **The Training Process of Many Deep Networks Explores the Same Low-Dimensional Manifold** — Jialin Mao（Sethna、Transtrum、Chaudhari 等），PNAS 2024（DOI 页面给出）— https://arxiv.org/abs/2305.01604
   做了什么：把训练轨迹投影到"无知→真值"的 Fisher–Rao（球面大圆）geodesic 上得到 progress∈[0,1]，**按 progress 而非 epoch 重索引 checkpoint**，使不同速度的网络在同一"进度"比较。
   接近程度：**ML 侧最高**——完全是信息几何时间重参数化，只是对象是训练动力学而非环境动力学。

9. **Episodic Memory for Learning Subjective-Timescale Models** — Alexey Zakharov（Crosby、Fountas），arXiv 2020-10 — https://arxiv.org/abs/2010.01430
   做了什么：active-inference 世界模型，当转移模型的 free energy（KL，即 surprise）超过阈值时记一条 episodic memory，LSTM 转移模型只在"记忆序列"（主观时间步）上训练；定性显示 rollout 更常命中显著事件。
   接近程度：高。这是"剧变时钟快"的离散、surprise 门控版本；差别：阈值触发而非弧长积分、无 Fisher 度量、无可预测性量化。

10. **Trajectory-Optimized Time Reparameterization for … Stiff Dynamical Systems (TOTR)** — Joe Standridge（Livescu、Cizmas），arXiv 2026-03 — https://arxiv.org/abs/2603.16583
    做了什么：在弧长坐标下优化遍历速度剖面（惩罚加速度），拉伸快瞬态，使 neural-ODE 降阶模型可学，loss 降 1–2 个数量级；结论是"时间映射的正则性强烈影响可学性"。
    接近程度：高（弧长时间重参数化 + 学动力学，但弧长是欧氏/状态空间的，不是 Fisher–Rao 的）。

11. **Adaptive Skip Intervals** — Alexander Neitz（Parascandolo、Bauer、Schölkopf），arXiv 2018 — https://arxiv.org/abs/1808.04768
    做了什么：循环动力学模型自己选预测间隔，利用"某些间隔的转移特别好预测"。
    接近程度：中（"模型自选时钟以提高可预测性"的先例，无几何）。

12. **Can recurrent neural networks warp time?** — Corentin Tallec & Yann Ollivier，arXiv 2018（ICLR 2018 据记忆）— https://arxiv.org/abs/1804.11188
    做了什么：证明可学门控给 RNN 对一般时间变换的近似不变性，导出 chrono initialization。
    接近程度：中（时间重参数化不变性，是你们"先匀速再学"思路的反面：让模型自己吸收时间扭曲）。

13. **Time-Aware World Model** — Anh N. Nhu（Son、Lin），arXiv 2025-06 — https://arxiv.org/abs/2506.08441
    做了什么：Δt 作为条件输入、多 Δt 混合训练的世界模型。
    接近程度：低–中（时间步可变但外给，不是从数据内在定义）。

14. **Clockwork VAE** — Vaibhav Saxena（Ba、Hafner），arXiv 2021 — https://arxiv.org/abs/2102.09532 ；**TimewarpVAE** — Travers Rhodes & Daniel Lee，arXiv 2023/2024 — https://arxiv.org/abs/2310.16027 ；**RNNs perform task computations by dynamically warping neural representations** — Pellegrino & Chadwick，arXiv 2025-12 — https://arxiv.org/abs/2512.04310
    接近程度：低（多速率层级 / DTW 式时间对齐 / 表征扭曲，均非信息时间）。

15. **Geodesic learning** — Arnab Barua（Hatzikirou、Abe），Physica A 2025（DOI 页面给出）— https://arxiv.org/abs/2401.04559
    接近程度：低–中（把 Fisher–Rao geodesic 当"最优学习过程"参照，无时间重参数化）。

### 负面结果
- 未找到针对"information length 作时间坐标"的负面结果。
- 一个非同行评审但有信息量的反例：博客 *Training dynamics are not Fisher geodesics*（https://divitr.github.io/posts/fisher-tr-dyn）——作者自己推翻了"优化轨迹近似 Fisher geodesic"的理论（只在短时、well-specified、lazy regime 下成立）。对你们的提醒：不要把"匀速化后的轨迹"误当 geodesic；匀速是重参数化的结果，不是动力学性质。
- 实际障碍（据推理，非文献）：Γ² = ∫p(∂ₜ ln p)² 需要可微的密度模型；Kim 系列都是低维 Fokker–Planck 数值解。高维要么用 latent 高斯（RSSM 类世界模型的 Fisher 度量有闭式），要么用 score 估计。

### 还剩什么没做
- 没有人把 Fisher–Rao 弧长作为时间坐标**先重参数化再学转移模型**，并与墙钟时间下学到的模型比较预测误差/长程 rollout。
- 没有人在任何系统上量化检验"相变 / 死亡在信息时间里等间距"；Kim 组只在 toy SDE 上看 L 的标度律。
- 没有人把 Zakharov 式的 surprise 门控主观时间与连续的 information length 时钟对比（前者是后者的阈值离散化）。

### 可能的差异化点
- 用世界模型自身的 latent 后验 q(z_t) 的 Fisher–Rao 度量算 Γ(t)（RSSM 对角高斯有闭式），从而时钟是"模型内生"的——这是 Kim 系列（需要真 PDF）做不到的。
- 提出可证伪的三元对照：墙钟时间 / 欧氏弧长时间（TOTR 式）/ Fisher–Rao 弧长时间，同一 Lenia 死亡事件序列上比较事件间距的变异系数与 k 步预测误差。
- 把 Tenkès 2017 的观察升级为定理化主张："信息时间下同一普适类的动力学共享转移模型"，用多种 Lenia 规则做 transfer 测试。

### 覆盖信心：**中**
11 次搜索后预算耗尽，被挡掉的 3 条关键查询是：Ito/Nicholson 的 intrinsic-speed 口径、"information length"+machine learning、"Fisher-Rao arc length"+training dynamics。神经科学里的 time-warping（如 Williams 等 2020 的 neural time warping，据记忆）与 RL 的 temporal abstraction 文献也只触及表层。

### 搜索词（11 次）
information length Fisher-Rao arc length time reparameterization learning dynamics neural network；thermodynamic length "intrinsic time" OR "statistical clock" OR "proper time" dynamical systems learning reparameterize time Fisher information；information length Kim Hollerbach stochastic dynamical systems "information geometry" time-dependent PDF arc length 2024 2025；"information length" "time reparametrization" OR "reparameterization" OR "intrinsic clock" dynamics Fisher-Rao "statistical manifold" constant speed；"information length" phase transition OR "sudden event" OR "abrupt" detection "information rate" Kim；time warping neural dynamics learning "time-warp" "intrinsic time" recurrent 2024 2025；"thermodynamic length" machine learning "time" reparameterization OR "natural time" OR "optimal protocol" neural network；"arc-length" reparameterization of trajectories learning dynamical systems neural ODE constant speed "time rescaling"；"Fisher information" "intrinsic time" OR "informational time" OR "statistical time" stochastic process dynamics 2024 2025 arXiv learning；Crooks "measuring thermodynamic length" 2007 Wootters statistical distance；learned clock OR "event time" OR "subjective time" world model latent dynamics "variable time step" OR "adaptive time" reinforcement learning predictability。

---

## Cell 10：尺度即时间（重整化群世界模型）

### 撞车等级：**B**（但"尺度即时间"这句口号本身在生成模型里已被用滥）

分三块看：(a) **neural RG / RSMI** 线（Koch-Janusz–Ringel 2018 → Lenggenhager PRX 2020 → Gökmen PRL/PRE 2021 → Nat. Commun. 2024；Li–Wang 2018；Hu RG-Flow；Hou–You MLRG 2023；Chung–Kao 2021）全部是**平衡态统计物理**，RSMI-NE 作者在 PRE 2021 里明确写"远离平衡的情形需要时间 buffer，留作未来工作"——这是文献自己承认的空洞。(b) **"RG 流 = 沿尺度的动力学"** 线（Cotler–Rezchikov RG-as-OT PRD 2023 与 Renormalizing diffusion models 2023；Berman–Klinger Bayesian renormalization 2023 / ERG-as-inference 2024；Masuki–Ashida 2025；Sheshmani 2024；Marchand WC-RG PRX 2023；Guth WSGM 2022；VAR 2024 的 next-scale prediction；Di Sante PRL 2022 用 Neural ODE 学 fRG 沿尺度的流）——"把尺度当生成/扩散时间"已饱和，但它们都没有**物理时间**那一维，没有"每层各自的动力学 + 跨层一致性"。(c) **学习粗粒化+宏观动力学** 线（Israeli–Goldenfeld 2004/2006 对 ECA 的 RG 式粗粒化；NIS/NIS+ 用 effective information 联合学 encoder 与宏观动力学，SI 里含 Game of Life，有 Multistory 多尺度扩展；Barnett–Seth dynamical independence；Vlachas LED；Rosas 等 computational closure）——这是最接近"RG 世界模型"的，但判据是 EI / 动力学独立 / 预测误差，**没有一个用 mutual-information-preserving（RSMI）判据**，也没有一个把"规则随尺度漂移、不动点"当作输出。

### 最近邻工作

**(a) neural RG / RSMI（根源与近况）**

1. **Mutual information, neural networks and the renormalization group** — Maciej Koch-Janusz & Zohar Ringel，Nature Physics 14:578 (2018) — https://arxiv.org/abs/1704.06279 ；https://www.nature.com/articles/s41567-018-0081-4
   做了什么：用 real-space mutual information 作判据的神经网络 RG，自动找相关自由度并迭代 RG，抽出 Ising 临界指数。
   接近程度：中（你们的粗粒化算子来源；只做平衡态）。

2. **Optimal Renormalization Group Transformation from Information Theory** — Patrick Lenggenhager（Gökmen、Ringel、Huber、Koch-Janusz），PRX 10, 011037 (2020) — https://arxiv.org/abs/1809.09632
   做了什么：证明完美 RSMI 粗粒化不扩大相互作用范围、不在无序系统中制造虚假关联——"最优"的定义。
   接近程度：中（给你们"每层动力学保持局部"的性质一个理论锚）。

3. **Symmetries and phase diagrams with real-space mutual information neural estimation (RSMI-NE)** — Doruk Efe Gökmen（Ringel、Huber、Koch-Janusz），PRE 104, 064106 (2021) — https://arxiv.org/abs/2103.16887 ；同组 PRL 127, 240603 (2021) — https://arxiv.org/abs/2101.11633
   做了什么：可微粗粒化 + 神经 MI 下界；唯一的非平衡例子是 1D chipping–aggregation 模型，**只在空间上粗粒化**，作者明确说远离平衡需要 temporal buffer，留作未来工作。
   接近程度：中高——这是 Cell 10 的直接出发点，也是文献自认的空洞。

4. **Compression theory for inhomogeneous systems** — Gökmen 等（含 Koch-Janusz），Nature Communications 2024（DOI 页面给出）— https://arxiv.org/abs/2301.11934
   做了什么：把 RSMI/压缩方法推到无平移不变性的图（准晶、随机图）。
   接近程度：低–中（仍是平衡态；但说明该组最新方向是"任意几何"，不是动力学）。

5. **Relevance in the Renormalization Group and in Information Theory** — Amit Gordon（Banerjee、Koch-Janusz、Ringel），PRL 126, 240601 (2021) — https://arxiv.org/abs/2012.01447
   做了什么：Information Bottleneck 的"相关"= RG 的"相关"（最低 scaling dimension 算子）。
   接近程度：中（如果你们用 IB 式目标而不是 RSMI，这是等价性依据）。

6. **Neural Network Renormalization Group** — Shuo-Hui Li & Lei Wang，PRL 121, 260601 (2018) — https://arxiv.org/abs/1802.02840 ；**RG-Flow** — Hong-Ye Hu（Wu、You、Olshausen、Chen），MLST 2022（IOP DOI）— https://arxiv.org/abs/2010.00029 ；**Machine Learning Renormalization Group for Statistical Physics** — Wanda Hou & Yi-Zhuang You，arXiv 2023 — https://arxiv.org/abs/2306.11054 ；**Neural Monte Carlo Renormalization Group** — Jui-Hui Chung & Ying-Jer Kao，PRR 3, 023230 (2021) — https://arxiv.org/abs/2010.05703
   做了什么：normalizing-flow RG / 逐尺度 latent 的层级流模型 / 生成模型自动学 RG 方程与 RG monotone / RBM 学最大化 RSMI 的 MCRG。
   接近程度：中（均为平衡态 Ising 类；Hou–You 的"学 RG 方程并找 monotone"最接近"规律随尺度变化的轨迹"的表述）。

7. **An exact mapping between the Variational Renormalization Group and Deep Learning** — Pankaj Mehta & David Schwab，arXiv 2014 — https://arxiv.org/abs/1410.3831
   接近程度：低（根源；RBM 堆叠≈变分 RG）。

**(b) "尺度即时间"：RG 流作为沿尺度的动力学 / 扩散**

8. **Renormalization Group Flow as Optimal Transport** — Jordan Cotler & Semon Rezchikov，PRD 108, 025003 (2023) — https://arxiv.org/abs/2202.11737 ；**Renormalizing Diffusion Models** — 同作者，arXiv 2023-08 — https://arxiv.org/abs/2308.12355
   做了什么：Polchinski 精确 RG = 场论相对熵的 OT 梯度流；进而用扩散模型学逆 RG 流。
   接近程度：中高（把 RG 流严格当作"沿尺度的动力学"，且与 Cell 3 的 OT 几何相通——这是你们三条 cell 之间可以互引的桥）。

9. **Bayesian Renormalization** — David Berman（Klinger、Stapleton），MLST 2023（IOP DOI）— https://arxiv.org/abs/2305.10491 ；**The Inverse of Exact RG Flows as Statistical Inference** — Berman & Klinger，Entropy 2024 — https://arxiv.org/abs/2212.11379
   做了什么：用 Fisher 度量定义涌现的 RG 尺度（可分辨性 = 最大可观测数），把 ERG 等价为 Bayesian diffusion。
   接近程度：中（与 Cell 9 的 Fisher 时钟同源：RG 尺度 = 可分辨性尺度）。

10. **Generative diffusion model with inverse renormalization group flows** — Kanta Masuki & Yuto Ashida，arXiv 2025-01 — https://arxiv.org/abs/2501.09064 ；**RG flow, Optimal Transport and Diffusion-based Generative Model** — Artan Sheshmani（You 等），arXiv 2024-02 — https://arxiv.org/abs/2402.17090
    做了什么：逐尺度擦除细节的前向流 + 反向生成；Fourier 空间按色散关系定噪声表。
    接近程度：中（尺度=扩散时间，但无物理时间演化）。

11. **Wavelet Conditional Renormalization Group** — Tanguy Marchand（Ozawa、Biroli、Mallat），PRX 13, 041038 (2023) — https://arxiv.org/abs/2207.04941 ；**Wavelet Score-Based Generative Modeling** — Florentin Guth（Coste、De Bortoli、Mallat），arXiv 2022 — https://arxiv.org/abs/2208.05003
    做了什么：逐尺度条件概率 p(细|粗) 的能量/score 建模，避免临界慢化；非平衡只作为展望。
    接近程度：中。

12. **Visual Autoregressive Modeling (VAR): next-scale prediction** — Keyu Tian 等，arXiv 2024（NeurIPS 2024 best paper 据记忆）— https://arxiv.org/abs/2404.02905
    接近程度：低–中（"分辨率是自回归轴"已是主流生成范式——你们的 slogan 需要和它划清：你们的轴上流动的是**规律**，不是像素）。

13. **Deep Learning the Functional Renormalization Group** — Domenico Di Sante（Millis、Sengupta 等），PRL 129, 136402 (2022) — https://arxiv.org/abs/2202.13268
    做了什么：Neural ODE 在低维 latent 里学四点顶点函数沿尺度的 fRG 流，DMD 显示少数模态即可表示。
    接近程度：中高——字面上就是"用学到的动力学模型描述 RG 流"，只是对象是电子关联的顶点函数。

14. **Information loss under coarse-graining: a geometric approach** — Archishman Raju（Machta、Sethna），PRE 98, 052112 (2018) — https://arxiv.org/abs/1710.05787 ；**Parameter Space Compression Underlies Emergent Theories** — Machta 等，Science 2013（DOI）— https://arxiv.org/abs/1303.6738 ；**The renormalisation group via statistical inference** — Cédric Bény & Tobias Osborne，NJP 2015 — https://arxiv.org/abs/1402.4949
    做了什么：Fisher 度量在 RG 流下沿无关方向收缩、相关方向保持；用扩散方程涌现作例。
    接近程度：中（给你们"不动点 = 该尺度有效理论"一个可测版本：模型流形在尺度轴上的收缩谱）。

**(c) 学习粗粒化 + 宏观动力学（含 CA）**

15. **Coarse-graining of cellular automata, emergence, and the predictability of complex systems** — Navot Israeli & Nigel Goldenfeld，PRE 73, 026203 (2006) — https://arxiv.org/abs/nlin/0508033 ；PRL 92, 074105 (2004) — https://arxiv.org/abs/nlin/0309047
    做了什么：对全部四类 ECA 构造局部粗粒化（粗粒化与时间演化可交换），得到"规则→更简单规则"的 RG 式流与规则层级，粗粒越粗可粗粒化概率→1，宏观规则的 Kolmogorov 复杂度给出标度律。
    接近程度：**对"规律随尺度变化的轨迹就是 RG 流"而言最高**。差别：穷举构造而非学习；无 MI 判据；1D 二元 CA 而非 Lenia；不谈世界模型。

16. **Finding emergence in data by maximizing effective information (NIS+)** — Mingzhe Yang（Jiang Zhang 等），National Science Review 2024（DOI 页面给出）— https://arxiv.org/abs/2308.09952 ；前作 **Neural Information Squeezer** — Jiang Zhang & Kaiwei Liu，Entropy 2023（DOI）— https://arxiv.org/abs/2201.10154
    做了什么：联合学可逆/降维 encoder 与宏观动力学（含逆动力学），最大化 EI；测试 SIR、Boids、**Game of Life（SI）**、fMRI；"Multistory NIS+"在共享堆叠 encoder 上训练多尺度动力学。**不提 RG**。
    接近程度：高。这是你们最直接的竞争基线：同样"学粗粒化 + 每层动力学"，只是判据是 EI 而非 MI 保持，且没有把跨尺度的规律漂移当对象。

17. **Dynamical independence: discovering emergent macroscopic processes** — Lionel Barnett & Anil Seth，PRE 108, 014304 (2023) — https://arxiv.org/abs/2106.06511 ；**Software in the natural world** — Fernando Rosas（Geiger、Luppi、Seth、Polani、Gastpar、Mediano），arXiv 2024 — https://arxiv.org/abs/2402.09090
    做了什么：用信息论度量找"自成一体"的宏观过程，多尺度 emergence portrait；嵌套自封闭过程的层级。
    接近程度：中（第三种判据：动力学独立/计算封闭）。

18. **Multiscale Simulations of Complex Systems by Learning their Effective Dynamics (LED)** — Pantelis Vlachas（Arampatzis、Uhler、Koumoutsakos），arXiv 2020/2021（Nature Machine Intelligence 2022 据记忆）— https://arxiv.org/abs/2006.13431 ；**Learning data driven discretizations for PDEs** — Yohai Bar-Sinai（Hoyer、Hickey、Brenner），PNAS 2019 — https://arxiv.org/abs/1808.04930
    做了什么：autoencoder 粗粒化 + RNN 宏观演化、equation-free 微宏交替；学粗网格上的最优离散。
    接近程度：中（单一宏观层；判据是重建/预测误差）。

19. **Local Causal States and Discrete Coherent Structures** — Adam Rupe & James Crutchfield，Chaos 2018（AIP DOI）— https://arxiv.org/abs/1801.00515 ；DisCo（2019）— https://arxiv.org/abs/1909.11822
    做了什么：对 ECA 用局部因果态发现 domain/particle（glider）结构，行为驱动、无需方程。
    接近程度：中（是 CA 上"预测性等价类 = 粗粒化"的计算力学版本）。

20. **When the map is better than the territory** — Erik Hoel，Entropy 2017 — https://arxiv.org/abs/1612.09592 ；Comolatti & Hoel 2022 — https://arxiv.org/abs/2202.01854 ；**Optimal high-level descriptions of dynamical systems (SSC)** — David Wolpert（Grochow、Libby、DeDeo），arXiv 2014 — https://arxiv.org/abs/1409.7403
    接近程度：低–中（粗粒化判据的理论来源）。

21. 世界模型侧：**Director** — Danijar Hafner 等，arXiv 2022 — https://arxiv.org/abs/2206.04114 ；**Clockwork VAE** — https://arxiv.org/abs/2102.09532
    接近程度：低（时间层级而非空间尺度层级；无粗粒化算子）。

### 负面结果 / 警示
- **Scale-invariant Feature Extraction of Neural Network and RG Flow** — Satoshi Iso（Shiba、Yokoo），PRE 97, 053304 (2018) — https://arxiv.org/abs/1801.07172 ：迭代训练好的 RBM 产生的"流"把温度推向 T_c，**与真实 Ising RG 流方向相反**。警示：学到的粗粒化映射不自动是 RG，"不动点"可能是学习偏差的产物，必须用已知系统校准方向。
- RSMI-NE（上文 3）自述：无时间 buffer 时远离平衡系统的 MI 判据会被短时关联污染。
- Mehta–Schwab 映射只对特定 RBM 构造"精确"，且是静态的；不能直接把它当"RG 世界模型"的理论背书。

### 还剩什么没做
- RSMI 式 MI-保持粗粒化**从未被用于时间演化系统**（作者自己留白）；没有"空间粗粒化 ∘ 时间演化 = 宏观演化 ∘ 空间粗粒化"的可交换性损失与 MI 判据的结合。
- Israeli–Goldenfeld 的"规则流"没有被学习化、没有推到 Lenia/连续 CA，也没有人画出"学到的转移模型参数随尺度的轨迹"并找不动点。
- 三种宏观判据（RSMI 的 MI 保持 / NIS+ 的 EI / Barnett–Seth 的动力学独立）从未在同一 CA 上 head-to-head 比较。

### 可能的差异化点
- 二维流：(scale k, time t) 平面上的模型 F_k，用可交换性损失 P_k∘f^T ≈ F_k∘P_k 训练，输出"F_k 随 k 的漂移轨迹"与不动点——这是 Di Sante（沿尺度的 Neural ODE）与 Israeli–Goldenfeld（规则流）都没有的组合，也避开 VAR/diffusion 的"尺度=生成时间"口号。
- 可测不变量：每层的"信息损失谱"（Raju–Machta–Sethna 式 Fisher 度量收缩）或 Lenggenhager 式"相互作用范围不扩大"检验，而不是只报预测误差。
- 试验场选 Lenia 而非 ECA：连续状态让 MI 估计与可微粗粒化（RSMI-NE 的机制）直接可用；同时 Lenia 的 glider 给出"宏观层应保留的自由度"的直观真值。

### 覆盖信心：**低–中**
**本条零次 WebSearch**（预算耗尽），全部靠我已知的 arXiv 编号逐篇 fetch 核对，所以根源与 2018–2024 的主干可信，但 **2025–2026 的 neural RG 新作、"multiscale world model"/"hierarchical coarse-graining of dynamics" 关键词下的 ML 工作、以及 Koch-Janusz 组 2025 之后的动向完全没查**（作者页 fetch 被限流）。原计划但未能执行的查询见下。

### 搜索词
实际执行：0。原计划（未执行）：renormalization group world model；multiscale dynamics coarse graining learned；neural renormalization 2025；RSMI dynamics / real-space mutual information time series；coarse-graining cellular automata renormalization learning；Lenia renormalization；hierarchical coarse-graining of dynamics RG flow of learned dynamics；Koch-Janusz 2025 2026。已核对的论文列表即上文 21 条。

---

## 总结：三条里哪条最危险、哪条最空

- **最危险：Cell 3。** "WFR + 学 v 与 r"作为方法已被北大 Peijie Zhou / Tiejun Li 组（DeepRUOT ICLR'25 oral、VGFM、WFR-FM ICLR'26、WFR-MFM、USB）与 Nie 组（TIGON）、Theis 组（UOT-FM）、Chardès 组（UPFI NeurIPS'25）做透，并且 UPFI 已证明 (v, r) 分解不可辨识。若以"WFR 世界模型"投稿，审稿人会判为增量。活路只在：把比值做成 δ-谱指纹、用 Lenia↔Flow-Lenia 作有真值的两极对照、以及"完整密度曲线下的归因"这一与快照推断不同的问题设定。
- **最空：Cell 9。** 概念（information length、Fisher-informational time）在物理里成熟，但没有任何人把它用作学习动力学的时间坐标；ML 侧只有训练动力学的 geodesic 进度重索引（Mao PNAS'24）、surprise 门控主观时间（Zakharov'20）和欧氏弧长重参数化（TOTR'26）三个旁系。它的风险不是撞车而是**可行性**（高维 Γ 的估计）与"结论是否平凡"（匀速化后更好预测可能只是 stiffness 消除的副产品——TOTR 已显示欧氏弧长就能带来 1–2 个数量级的 loss 下降，必须和它做对照才能证明"Fisher"而非"弧长"在起作用）。
- **居中：Cell 10。** "尺度即时间"口号在生成模型里已饱和（RG-as-OT、RG diffusion、WC-RG、VAR），"学粗粒化 + 宏观动力学"也有 NIS+（含 Game of Life）与 LED，但"RSMI 判据 + 时间演化 + 规则沿尺度的流与不动点"这个交集是空的，且 RSMI-NE 作者自己留了白。注意本条覆盖信心最低，投稿前必须补做 2025–2026 的检索。

跨条备注：Cotler–Rezchikov（RG = OT 梯度流）与 Berman–Klinger（RG 尺度 = Fisher 可分辨性）把 Cell 3、9、10 的几何连成一体——如果要写一篇统一框架的 position paper，这两组是现成的理论桥。


<!-- ===== 报告 G3 ===== -->

# 文献审计 G3：Cell 4 / Cell 6 / Cell 8

审计日期：2026-11-01。方法：WebSearch（共 24 次有效查询，之后本轮共享搜索预算耗尽）+ WebFetch 逐篇核对 arXiv 摘要页。所有 URL 均为搜索或 fetch 实际返回的地址。会议/期刊标注仅在页面确认时给出；写"据记忆"者未经页面确认。引用均为转述。

---

## Cell 4：形状即规律（拓扑优先的世界模型）

### 撞车等级：**B+**（核心命题已被一篇"近 A"工作部分占据；"先发现再学"的流程与 Lenia 试验场仍空）

最接近的一篇（GD-VAE）已经做了"把 latent 放到 torus / Klein bottle 等流形上再学动力学，并论证拓扑不匹配会逼出不连续"，这与你们的主张几乎同构。但它是**预先指定**拓扑而非**发现**拓扑，动力学部分也是半手工给定的，并且没有"固定容量下拓扑匹配误差最低"这种受控命题。没有找到任何把 persistent homology → 选流形 → 学流 串成世界模型管线的工作。

### 最近邻工作

1. **GD-VAEs: Geometric Dynamic Variational Autoencoders for Learning Nonlinear Dynamics and Dimension Reductions** — Ryan Lopez (& Atzberger), arXiv 2022-06（v4 2025-03），venue 未标 — https://arxiv.org/abs/2206.05183
   做了什么：encoder 输出投影到预先指定的流形（circle、cylinder、torus/Clifford torus、Klein bottle，用最近点投影 + 隐函数定理求梯度），在其上做多步预测；附录 B 明确展示周期数据配 ℝ² latent 时 encoder 被迫近似一个不连续点。
   接近程度：**最高**。几何对象、"拓扑错了动力学撕裂"的论证、多步误差比较都重合。差别：拓扑是给定的，不是发现的；动力学（如 Brusselator 的旋转速度）部分由 PCA 预估后手工指定；表格结果混合（见负面结果）。

2. **Cloning Deterministic Worlds: The Critical Role of Latent Geometry in Long-Horizon World Models (GRWM)** — Zaishuo Xia, 2025-10, arXiv — https://arxiv.org/abs/2510.26782v1
   做了什么：诊断实验主张长程 rollout 精度的瓶颈是 latent 表征的几何结构而非 dynamics model，提出用 temporal contrastive 模块把 latent 形状对齐到物理状态流形。
   接近程度：中高。与你们"潜空间形状错了动力学会出问题"的叙事一致，但谈的是"几何"而非离散拓扑不变量，无 Betti 数、无流形先验。

3. **Learning Abstract World Models with a Group-Structured Latent Space** — Thomas Delliaux (van der Pol, Rachelson 等), 2025-06（v2 2026-05）, arXiv — https://arxiv.org/abs/2506.01529
   做了什么：给学到的 MDP transition model 的低维 latent 加上群作用先验（旋转/平移），允许非结构化信息并存，报告 transition 预测与下游 RL 改善。
   接近程度：中。属于 homomorphic world model 线的最新版本；拓扑是通过群结构间接进入的，没有把拓扑当作独立可测不变量。

4. **Constructing VAE Latent Spaces with Prescribed Topology** — Jilles S. van Hulst, 2026-06, arXiv（math.AT 标签）— https://arxiv.org/abs/2606.07058
   做了什么：对具有乘积覆盖空间的流形（circle、interval 的乘积，以及其有限群商：torus、cylinder、Möbius、Klein bottle、RP^n）构造可分解 KL 的 VAE 先验，实验显示拓扑匹配先验优于 Gaussian。
   接近程度：中。静态表征，无动力学；但它是"规定拓扑的 latent"最新工具箱，可直接作为你们 encoder 部件。

5. **Topological Obstructions to Autoencoding** — Joshua Batson, arXiv 2102.08380（2021）— https://arxiv.org/pdf/2102.08380
   做了什么：证明 latent 只是数据流形的一张 chart，S¹ 压到一维坐标必然出现"break point"；附录分析 SGD 训练动力学如何制造并维持该断裂。
   接近程度：中。为你们"拓扑错→撕裂"提供经典理论根据，但无动力学建模。

6. **Topological Obstructions and How to Avoid Them** — Babak Esmaeili (Walters, van de Meent), 2023-12, arXiv — https://arxiv.org/abs/2312.07529
   做了什么：指出把 encoder 训练进几何 latent 空间时，自交和错误 winding number 会制造局部最优；用多峰 normalizing flow 绕过，提高学到同胚 encoder 的概率。
   接近程度：中。直接关系到你们方法的可训练性（见负面结果）。

7. **Neural Persistence Dynamics** — Sebastian Zeng (Kwitt), NeurIPS 2024（页面确认）— https://arxiv.org/abs/2405.15732 ；https://proceedings.neurips.cc/paper_files/paper/2024/hash/3a509449a73fd0aab8c0cf5705827036-Abstract.html
   做了什么：对随时间演化的点云逐帧算 persistence diagram、向量化，再在其上学 latent ODE，用于反推控制方程参数。
   接近程度：中低。它把拓扑当作**观测特征序列**来学动力学，而你们是把拓扑当作**状态空间的形状**；但它是"persistent homology + latent dynamics"检索下唯一的主会论文，审稿人一定会问。

8. **Sensorimotor World Models: Perception for Action via Inverse Dynamics** — Petr Ivashkov (Balestriero, Schölkopf), 2026, arXiv — https://arxiv.org/pdf/2606.20104
   做了什么：JEPA 式世界模型加 inverse dynamics head；在 Reacher 上**观察到**学到的 latent 呈 flat 2-torus 结构，但未强制。
   接近程度：中低。可当作"拓扑会自发涌现"的对照组——你们要证明的是显式匹配比涌现更好。

**Lenia / CA 吸引子拓扑分析**

9. **Visualising the Attractor Landscape of Neural Cellular Automata** — James Stovold (Mordvintsev 等), 2026-04, arXiv — https://arxiv.org/html/2604.10639v1 （arXiv 页被限流，内容经 https://www.emergentmind.com/papers/2604.10639 确认）
   做了什么：对 NCA 用 PCA/autoencoder + persistent homology 画行为流形与吸引子景观，宏观层面发现连接两个吸引子的稳定环，纹理 NCA 检出 toroidal topology。
   接近程度：对试验场而言**最高**。几乎就是"CA 吸引子拓扑分类"，但对象是 NCA 而非 Lenia，且没有在拓扑上再学世界模型。

10. **Visualizing the Structure of Lenia Parameter Space** — Barbora Hudcová (Hongler 等), 2026-01, arXiv — https://arxiv.org/abs/2601.01932v1
    做了什么：自动把 Lenia 系统分成四个定性动力学类，检测移动 soliton，给交互网站。
    接近程度：中。是 Lenia 行为分类的最新基线，但不用 TDA、不谈吸引子拓扑。

11. **Performing TDA on Lenia**（Heidelberg 学生项目网页，无日期无作者）— https://pille.iwr.uni-heidelberg.de/~tda01/tda-lenia.html
    做了什么：对 Lenia 帧做 cubical persistent homology，逐帧 Betti 数，用 Otsu 阈值去噪，按平均 Betti 数给生物做"拓扑分类"。
    接近程度：中低。算的是**生物形态**的拓扑，不是**吸引子/状态空间**的拓扑；非正式出版物，但说明这个方向不是零。

12. **Agnosiophobia in a virtual agent: behavioral and dynamical architecture in Lenia** — Jesse Cool (Hartl, Levin, Petti), 2026-05, arXiv — https://arxiv.org/pdf/2605.30708
    做了什么：对 Lenia 生物做遮挡扰动，用 basin of attraction / cognitive basin 解释趋避行为。
    接近程度：中低。谈吸引子盆地几何但不算拓扑不变量，也无学习模型；是 Lenia 侧最近的"吸引子"叙事。

### 负面结果（单独标出）

- **Esmaeili et al. 2023**：把 encoder 训进几何 latent 空间会卡在自交/错误 winding number 的局部最优。意味着"先把拓扑放进去再学流"在优化上有实打实的障碍，需要 flow 式多峰后验或类似手段。
- **GD-VAE 自己的表格**：拓扑匹配模型稳定胜过无投影 baseline，但并不总胜过更高维的 Euclidean VAE 或带投影的 AE（例如 torus 情形 VAE-5D/10D 误差更低）。这直接削弱"固定容量下拓扑匹配误差最低"——除非容量比较设计得非常严格，这个命题可能被反例打掉。
- **Batson et al.**：断裂位置与大小依赖采样分布，训练时间只能对数级地缩小 gap。说明"撕裂"可能被大模型部分掩盖，必须控制容量才能看到效应。

### 还剩什么没做

- 没有人做"**先用 persistent homology 从轨迹数据发现状态空间拓扑（Betti 数/连通分支），据此自动选 latent 流形，再在其上学动力学**"的闭环世界模型；现有工作全是人工指定拓扑。
- 没有"固定参数量/latent 维度下，拓扑匹配 vs 不匹配 vs 自由 Euclidean 的多步预测误差"系统比较，更没有把它写成可证伪的不变量命题。
- Lenia 的**吸引子**拓扑（而非形态拓扑）与行为类别的对应，以及在该拓扑上学世界模型，完全没有。

### 可能的差异化点

- 把"拓扑发现"作为可测的离散输出（Betti 数预测对不对），与"动力学误差"做因果链：拓扑错 → 撕裂指标（encoder Lipschitz 爆点、rollout 不连续计数）→ 误差。GD-VAE 只有后半段。
- 用 NCA 吸引子景观（Stovold 2026）的方法迁移到 Lenia，但目标不是可视化而是"拓扑类 = 行为类"的分类命题，再证明在匹配拓扑上学的世界模型 rollout 更稳。
- 正面处理 Esmaeili 的优化障碍：用 van Hulst 2026 的乘积拓扑先验或 flow 式 encoder 作为部件，把"可训练性"也做成实验变量。

### 覆盖信心：**中高**
世界模型侧已多措辞交叉检索；Lenia/CA 侧覆盖中等（Lenia+TDA 的正式论文似乎确实不存在，但神经科学侧"grid cell torus / 神经流形拓扑发现"那条线因搜索预算耗尽未查，那边可能有"先发现拓扑再学动力学"的范式）。

### 搜索词
`topology-aware world model latent space Betti numbers dynamics learning` / `persistent homology latent dynamics reinforcement learning world model 2025` / `latent space topology mismatch dynamics discontinuity torus circle SO(3) world model representation learning` / `"topological" "latent dynamics" manifold "Betti" neural dynamics learning 2024 2025 arXiv` / `latent space topology learning circle torus SO(3) latent dynamics "world model" homeomorphic manifold 2025` / `neural ODE on manifold latent dynamics torus sphere topology prior continuity reinforcement learning encoder` / `topological regularization latent dynamics model "persistent homology" "dynamics model" prediction error topology mismatch` / `Lenia attractor topology persistent homology cellular automata behavior classification` / `topological data analysis cellular automata attractor classification Betti persistent homology Game of Life Lenia dynamics` / `Lenia creatures topological data analysis persistent homology soliton classification behaviour` / `arXiv 2604.10639 Lenia`。未能执行（预算耗尽）：`toroidal topology grid cells persistent homology latent dynamics discovery neural population manifold`。

---

## Cell 6：联络即规律（联络/和乐世界模型）

### 撞车等级：**B**（核心主张"holonomy ≠ 0 ⇔ 存在隐状态"已有一个离散/组合版本，2026-07；连续联络 + 学习型世界模型 + 曲率探测器的版本没人做）

### 最近邻工作

1. **Minimal Markovization via Stable Quotients in Holonomy-Cover Decision Processes** — Zuyuan Zhang (Chen, Imani, Lan), 2026-07-29, arXiv cs.LG, venue 未标 — https://arxiv.org/abs/2607.27132
   做了什么：定义 HCDP：可见动力学是 Markov 的，但隐藏模式沿每条可见转移被一个置换"搬运"；沿闭路累积的搬运就是 holonomy，构成一个有限置换群；证明"观测 + 稳定商类"是最小精确 Markov 状态，非交换 holonomy 时计数类记忆（abelian summary）必然失败；提出 HMRL 跟踪类坐标；并证明**仅凭被动日志无法辨识商结构，需要可重置的校准诊断**。
   接近程度：**最高，且是对核心主张的直接命中**。它说的正是"绕一圈回来状态变了 = 世界记得路径 = 需要记忆的隐状态"，而且明确用非交换 holonomy（LoopGuess 任务）。差别：离散覆盖空间/置换群语言，不是微分几何的联络与曲率；不是学习型 latent 世界模型，而是理论 + 表格 RL；没有"曲率场作为连续探测器"的概念。

2. **Tangential Action Spaces: Geometry, Memory and Cost in Holonomic and Nonholonomic Agents** — Marcel Blattner, 2025-09（v2 2025-09），arXiv eess.SY — https://arxiv.org/abs/2509.03399
   做了什么：把具身智能体建模为"物理态 → 认知表征 → 意图"的流形纤维化，用 Ehresmann connection 的曲率解释 holonomy = 路径依赖记忆；证明记忆需要高于 metric-lift 基线的能量，小环路下代价随 holonomy 平方增长；扩展到带可学习模型态的 rTAS。
   接近程度：高（概念上），低（方法上）。联络、曲率、holonomy、记忆全都在，但是解析理论 + 手工玩具系统，没有从数据学联络。

3. **US Patent 12651010 — holonomy-based cognitive state representation in persistent cognitive machines**（专利文本，题名/申请人页面未列）— https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12651010
   做了什么：权利要求把认知状态定义为"流形位置 + 活动 holonomy 描述子"，描述子记录路径依赖效应，由预测误差触发生成、沿路径顺序合成、衰减/修剪；描述里用 Christoffel 符号的测地方程与 Ricci 曲率。
   接近程度：中。作为 prior art 要知道它存在；但不是论文，没有实验或可检验命题，也没有"用 holonomy 检测部分可观测"。

4. **Learning Abstract World Models with a Group-Structured Latent Space** — Delliaux 2025（同 Cell 4 第 3 条）— https://arxiv.org/abs/2506.01529
   接近程度：中。homomorphic / group-structured 世界模型线的当前前沿，是你们"不要求动作成群"要对照的 baseline。

5. **Commutator-Induced Uncertainty in VAEs** — Tahereh Dehdarirad (Felsberg 等, Linköping), 2026, preprint — https://arxiv.org/html/2605.23449
   做了什么：在 Lie Group VAE 里把 latent 变换的非交换性当作可测量（有限 BCH 偏差、order-swap 测试），要求 decoder 的顺序敏感度不低于 latent 交换子的校准比例；指出对称 pullback metric 抓不到不对称的顺序效应，把交换子类比为 torsion。
   接近程度：中。"先左后上 ≠ 先上后左的差作为可测量"这件事已经有人在静态 VAE 里做了度量；但无动力学、无隐状态解释。

6. **Inter-environmental world modeling for continuous and compositional dynamics (WLA)** — Kohei Hayashi (Koyama, PFN), 2025-03, arXiv — https://arxiv.org/html/2503.09911v1
   做了什么：转移建模为 Lie 群在槽式 latent 上的作用，**明确假设群是 abelian、转移可交换**，并把非交换列为局限。
   接近程度：中。它留下的洞正是你们要填的。

7. **Structured Latent Dynamics in Wireless CSI via Homomorphic World Models** — Salmane Naoumi (Bennis, Chafii), 2026, preprint — https://arxiv.org/html/2603.20048
   做了什么：JEPA + Lie algebra 矩阵指数参数化的动作条件转移，用于无线信道状态轨迹。
   接近程度：中低。再次是群结构转移，非联络。

8. **Learning Disentangled Representations and Group Structure of Dynamical Environments** — Robin Quessard, 2020, arXiv — https://arxiv.org/pdf/2002.06991
   做了什么：每个动作学成 SO(n) 矩阵作用于单位球 latent，SO(3) 实验，提到解缠能保持"commutation relations"。
   接近程度：中低。经典根源之一；有非交换动作但不分析其几何含义。

9. **Sensorimotor World Models** — Ivashkov 2026（同 Cell 4 第 8 条）— https://arxiv.org/pdf/2606.20104
   接近程度：中低。推导出近似可加的 latent 动作模型，自己注明在接触动力学与非交换动作律下可能失效但未测试——又一个留下的洞。

10. **Gauge Fiber Bundle Geometry of Transformers** — Hong Wang & Kelly Wang, NeurIPS 2025 Workshop (Symmetry and Geometry in Neural Representations) poster — https://neurips.cc/virtual/2025/136821
    做了什么：参数空间按 head-wise 对称群纤维化，Euclidean 度量诱导 Ehresmann connection，报告曲率一般非零并给小环路 holonomy 估计器。
    接近程度：低（对象是参数空间，不是环境状态）。但"小环路 holonomy 估计器"这个工具可直接搬。

11. **The Curved Spacetime of Transformer Architectures** — Riccardo Di Sipio, 2025-11, arXiv — https://arxiv.org/abs/2511.03060v1
    做了什么：把 attention 当作在 token 间搬运 value 向量的离散联络，层间演化视为轨迹，做曲率可视化与偏转实验。
    接近程度：低。联络语言用在 LLM 内部表征，不是世界模型。

12. **Gauge Equivariant Deep Q-Learning on Discrete Manifolds** — Sourya Basu, ICLR 2022 Workshop (Geometrical and Topological Representation Learning) — https://www.iclr.cc/virtual/2022/8707
    做了什么：Q 值对 gauge 选择不变的 DQN，用于球面/山丘等嵌入环境。
    接近程度：低。gauge 词用了，但没有搬运动作或学联络。

13. **Wasserstein Parallel Transport for Predicting the Dynamics of Statistical Systems** — Tristan Luca Saidi (Wasserman 等), 2026-03, arXiv — https://arxiv.org/abs/2603.23736
    做了什么：沿 OT 测地线平行移动"趋势"来预测分布演化，用于因果推断/scRNA。
    接近程度：低。平行移动 + 动力学预测，但不是 latent 世界模型，也不谈 holonomy。

14. **latent-riemannian-world**（PyPI 包，作者 lajjadred，2026-05，无论文）— https://pypi.org/project/latent-riemannian-world/
    做了什么：扩散模型 latent 的 pullback/Fisher 度量、测地线求解、Schild's ladder/pole ladder 平行移动，以及 `RiemannianRSSM` 世界模型类。
    接近程度：低（无论文、无主张），但说明"Riemannian RSSM + parallel transport"已经有人在写代码。

### 负面结果（单独标出）

- 没有找到"联络/holonomy 做世界模型效果不好"的直接负面结果。
- **警示性结果（HCDP 2026）**：仅用被动轨迹日志无法一致地辨识 holonomy 商结构，需要可重置的主动诊断（走环路）。对你们意味着：曲率作为隐状态探测器很可能**需要主动探索闭环**，离线数据下不可辨识——这要在命题里写清。
- **Blattner 2025**：holonomy 记忆要花能量，代价随 holonomy 平方增长；不是否定，但暗示学到的联络会被能量/正则项压平，需要防止模型"把曲率学没了"。

### 还剩什么没做

- 从数据**学习**一个 latent 联络 Γ（动作 = 平行移动算子），用交换子/小环路 holonomy 作为连续量来**检测部分可观测性**，并在学习型世界模型（RSSM/JEPA 类）里实现——没有。
- "不要求动作可逆或成群"的联络式转移：现有 WLA、SMWM、Delliaux 全都假设群/可加/可交换，并有两篇自己写明非交换是未做的局限。
- 曲率场与 POMDP 信息量（如 belief 熵、所需记忆位数）的定量对照：HCDP 给了离散版上界（max_o |C_o| 个记忆符号），连续版没有。

### 可能的差异化点

- 把 HCDP 的离散定理当作理论锚点：你们的连续曲率探测器在离散极限应退化为它的非交换 holonomy 群；用它的 LoopGuess 类任务做首个基准，再上连续/像素环境。
- 主打"曲率 = 隐状态探测器"这一可证伪命题：在可控 POMDP 里系统调隐状态位数，预测学到的 holonomy 谱（或交换子范数）单调响应；对照组是 Delliaux/WLA 群结构模型必然把它压成零。
- 方法上借 Dehdarirad 2026 的交换子测量协议和 Wang & Wang 2025 的小环路 holonomy 估计器，避免重新发明度量工具。

### 覆盖信心：**中**
核心关键词（holonomy、parallel transport、gauge、fiber bundle、non-commutative actions）都查了，并捕到了 2026-07 的 HCDP，这是关键发现。但"parallel transport + world model / latent dynamics"的精确短语检索因预算耗尽未执行，arXiv 上 2026 年下半年可能还有我没看到的 Riemannian RSSM 类工作（PyPI 包就是信号）。

### 搜索词
`parallel transport latent dynamics world model connection curvature actions` / `holonomy partial observability hidden state detection learned dynamics gauge field reinforcement learning` / `fiber bundle reinforcement learning gauge equivariant world model action transition connection` / `Wasserstein Parallel Transport latent dynamics learned connection non-commutative actions latent curvature reinforcement learning` / `"Tangential Action Spaces" Blattner holonomy agents arXiv` / `"Holonomy-Cover Decision Processes" OR "holonomy memory reinforcement learning"` / `learned latent transition non-commuting actions curvature "world model" commutator hidden state detection` / `"world model" latent "Lie group" OR "Lie algebra" actions non-abelian commutator curvature learned transition 2025 2026` / `"gauge" equivariant latent dynamics agent actions "parallel transport" representation learning sensorimotor` / `Riemannian world model latent metric geodesic planning 2025 arXiv`。未能执行：`"parallel transport" "world model" OR "latent dynamics" learned connection ...`。

---

## Cell 8：距离即代价且不对称（拟度量/Finsler 世界模型）

### 撞车等级：**A−**（quasimetric 这个几何结构在 RL 里已是成熟且活跃的子领域，2025–2026 仍有顶会级工作在推；"quasimetric 进世界模型做规划"也已有先例；但"不对称场 = 时间箭头地图并与熵产生率对照"这一主张没人做，Finsler 形式化也没人做）

### 最近邻工作

**(a) Quasimetric RL 主线（根源与 2024–2026 进展）**

1. **Optimal Goal-Reaching RL via Quasimetric Learning (QRL)** — Tongzhou Wang (Torralba, Isola, Zhang), 2023, arXiv（据记忆为 ICML 2023，页面未确认）— https://arxiv.org/abs/2304.01203
   做了什么：用 quasimetric 模型直接学最优 goal-conditioned 值函数，带恢复保证。接近程度：根源，直接相关。

2. **Improved Representation of Asymmetrical Distances with Interval Quasimetric Embeddings (IQE)** — Tongzhou Wang & Isola, 2022（v2 2024-01）— https://arxiv.org/abs/2211.15120 ；**On the Learning and Learnability of Quasimetrics (PQE)** — Wang & Isola, ICLR 2022 — https://arxivlens.com/PaperView/Details/on-the-learning-and-learnability-of-quasimetrics-7563-1682036 ；**Metric Residual Networks** — Bo Liu (Stone), 2022 — https://arxiv.org/abs/2208.08133
   做了什么：可学习 quasimetric 头的三种参数化。接近程度：根源，你们的 d(s,s′) 必然用其中之一。

3. **Offline Goal-conditioned RL with Quasimetric Representations** — Vivek Myers (Eysenbach, Levine), NeurIPS 2025 poster（页面确认）— https://arxiv.org/abs/2509.20478v2 ；https://neurips.cc/virtual/2025/poster/118249
   做了什么：把 contrastive successor features 与 temporal distance 统一进带约束的 quasimetric 表征，处理次优数据与随机环境。接近程度：高（同一几何结构、同一 RL 用法），但不解释不对称的物理含义。

4. **Multistep Quasimetric Learning for Scalable Goal-conditioned RL** — Bill Chunyuan Zheng (Myers, Eysenbach, Levine), 2025-11（v3 2026-02）— https://arxiv.org/abs/2511.07730
   做了什么：多步 Monte Carlo 回报拟合 quasimetric，4000 步任务与真实机械臂。接近程度：高。说明这条线正被 Levine 组密集推进。

5. **Offline GCRL with Projective Quasimetric Planning (ProQ)** — Anthony Kobanda, 2025-06 — https://arxiv.org/abs/2506.18847v2
   做了什么：学不对称距离，既当 keypoint 排斥能量又当方向性代价做子目标规划。接近程度：高；"预测 = 最短路"的规划用法已在。

6. **QuasiNav** — Jumman Hossain, 2024-10 — https://www.emergentmind.com/papers/2410.16666
   做了什么：户外导航上下坡能耗不对称，constrained MDP + quasimetric embedding。接近程度：中高；是"不对称 = 物理代价"最直白的先例，但停在能耗层面。

7. **Quasimetric Value Functions with Dense Rewards** — Khadichabonu Valieva, 2024-09 — https://www.emergentmind.com/papers/2409.08724
   做了什么：证明一定条件下稠密奖励仍保持三角不等式。接近程度：中（理论补充）。

8. **Learning the Minimum Action Distance** — Lorenzo Steccanella (Jonsson), 2025-06（v4 2026-07）— https://arxiv.org/abs/2506.09276
   做了什么：仅从状态轨迹（无奖励无动作）学"最少动作数"距离，嵌入可对称或不对称。接近程度：中高；与你们"距离即最优代价"同义，但不追问不对称的含义。

**(b) 已进入世界模型的不对称距离**

9. **Temporal-Distance-JEPA: Plan-Aware Representation Learning for Latent World Model Predictive Control** — Jiaxin Bai & Jiaxuan Xiong (HKBU), 2026, preprint — https://arxiv.org/pdf/2607.25337
   做了什么：从演示日志挖有向时间代价，头部为"对称 Euclidean + 非负不对称残差"的 quasimetric 风格结构，在 JEPA 世界模型 rollout 终点上做 CEM 规划。
   接近程度：**最高**。已经把不对称距离放进 latent 世界模型当规划目标；但未证明 quasimetric 公理，也不做物理解释。

10. **GeoWorld: Geometric World Models** — Zeyu Zhang (Reid, Hartley), CVPR 2026 — https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_GeoWorld_Geometric_World_Models_CVPR_2026_paper.pdf ；https://arxiv.org/pdf/2602.23058
   做了什么：JEPA latent 映到 Poincaré 球，沿双曲测地线预测，三角不等式正则。接近程度：中；对称度量，可作"对称 vs 不对称"对照。

11. **Latent geometry of world models**（学士论文）— Adam Černý, Charles University, 2026 — https://dspace.cuni.cz/handle/20.500.11956/210796
    做了什么：DreamerV1 的 RSSM 在迷宫里 Euclidean latent 距离不是真距离的可靠代理，时间结构保留。接近程度：低，但是"为什么需要学距离而不是用 Euclidean"的现成证据。

**(c) 时间箭头 / 不可逆性 / 熵产生侧**

12. **A Path-Space Formulation of Prediction in World Models: From a Single Action to Prediction, Planning, and Irreversibility** — Gunn Kim, 2026-06, arXiv cs.LG + cond-mat.stat-mech — https://arxiv.org/abs/2606.28751
    做了什么：世界模型视为轨迹路径测度（Onsager–Machlup 作用量），把 latent 动力学分成可逆/不可逆部分，从 rollout 测熵产生；小 attention 模型里学到的 attention 不对称度与数据不可逆性成比例，对称化 attention 会降低熵产生并损害长程预测。
    接近程度：**对"不对称 ↔ 熵产生"主张最高**。但其不对称量是 attention 权重，不是 quasimetric，也没有 d(a,b)−d(b,a) 场。

13. **Temporal Reversal Asymmetry: A Physics-Inspired Metric for Evaluating World Models** — Kanpat Vesessook & Kevin Yang, ICLR 2026 tiny paper（workshop poster）— https://iclr.cc/virtual/2026/10018041
    做了什么：正向 vs 时间反转视频的预测损失差作为时间箭头探针，V-JEPA2 的 TRA 随耗散强度平滑变化。接近程度：中；评估量而非学到的几何。

14. **Learning the Arrow of Time** — Nasim Rahaman (Bengio), 2019, arXiv — https://arxiv.org/pdf/1907.01285
    做了什么：学标量势 h(s) 使其沿时间期望递增，定义有向可达性 η(s→s′)=h(s′)−h(s)，与 JKO 自由能泛函对照，用于安全探索与好奇心。
    接近程度：高（概念根源）。它的 η 是**势差**（反对称、可负），不是 quasimetric；你们的 d(a,b)−d(b,a) 若在势能型环境应退化为 η——这是一个可用的理论联系。

15. **Machine learning the thermodynamic arrow of time** — Alireza Seif (Hafezi, Jarzynski), 2019 arXiv（据记忆后发 Nature Physics 2021，未确认）— https://ar5iv.labs.arxiv.org/html/1909.12380
    做了什么：训练分类器判别轨迹正反向，网络自发学到耗散功 W−ΔF 作为判据。接近程度：中；物理侧根源，是分类器不是距离。

16. **There Is No Turning Back: Reversibility-Aware RL** — Nathan Grinsztajn (Geist), 2021 — https://arxiv.org/abs/2106.04480
    做了什么：用自监督的事件时间排序学可逆性估计。接近程度：中；不可逆性估计的 RL 根源。

**(d) Finsler 几何在学习里的应用**

17. **Finsler Multi-Dimensional Scaling** — Thomas Dagès (Kimmel 等), 2025-03 — https://arxiv.org/abs/2503.18010
    做了什么：MDS 推广到 Finsler 流形以嵌入不对称相异度，定义典范 Finsler 空间，用于有向图嵌入。接近程度：中；静态，无动力学。

18. **Learning Lineage-guided Geodesics with Finsler Geometry** — Aaron Zweig (Azizi), 2026-03 — https://arxiv.org/abs/2603.16708
    做了什么：轨迹推断中用 Finsler 度量把连续几何先验与离散有向（谱系树）先验合并，插值时间点间的分布演化。接近程度：中高；**Finsler + 有向时间演化**的最近先例，但是分布插值不是世界模型，也不谈熵产生。

19. **Finsler Geometry, Graph Neural Networks, and You** — T. Mitchell Roddenberry & Baraniuk, 2026-06 — https://arxiv.org/abs/2606.17185
    做了什么：Finsler Laplacian 的点云估计与收敛证明，作为各向异性 GNN 层。接近程度：低；工具。

### 负面结果（单独标出）

- **PQE (ICLR 2022)**：无约束 MLP 等通用学习器**可证明**无法可靠地学到与数据一致的 quasimetric；必须用结构化头（PQE/IQE/MRN）。你们的 d 不能随便用 MLP。
- **Myers et al. 2025** 的动机陈述：quasimetric 方法在噪声大、高维、随机环境下吃力（它们才做混合）。在 Lenia 这类连续随机场上可能复现这个弱点。
- **Rahaman et al. 2019**：η=0 不保证可逆，非遍历环境尤甚；方法要求环境耗散且数据策略近随机——你们的"时间箭头地图"同样会继承这些前提。
- **Černý 2026**：Dreamer 的 Euclidean latent 距离不可靠（支持你们的动机，不是反面）。
- QRL 原文据记忆假设确定性动力学（页面未确认）；若成立，你们在随机环境下需要 Myers 式修正。

### 还剩什么没做

- 没有人把学到的 quasimetric 的**不对称场 A(a,b)=d(a,b)−d(b,a)** 当作独立研究对象，画成"时间箭头地图"，并与熵产生率（或 Kim 2026 的 Onsager–Machlup 不可逆分量、Seif 的耗散功）做定量对照。
- 没有 Finsler 形式化的世界模型：现有 quasimetric RL 全是嵌入式参数化（区间/Poisson/残差），没人把它写成切空间上的 Finsler 范数 F(x,v) 并与局部动力学（漂移方向）联系。
- "预测 = 最短路"在 TD-JEPA、ProQ 里是规划用法，但没有人把它当**物理命题**检验（如：quasimetric 测地线是否与真实轨迹/最小作用路径一致）。

### 可能的差异化点

- 把三件事焊在一起：学 quasimetric（用 IQE/MRN 头）→ 不对称场 → 与熵产生率的局部对照（用 Kim 2026 的可逆/不可逆分解作为"物理真值"），给出可证伪命题"A(a,b) 与沿 a→b 的平均熵产生单调相关"。
- 在 Rahaman 的势能型极限做理论锚：当环境梯度流主导时，A 应退化为 η=h(s′)−h(s)；偏离该极限的残差就是"非势能"不可逆性——这是现有工作没有的分解。
- Finsler 化：把 d 写成 F(x,v) 的路径积分，直接学 F 的 Randers 型不对称项（β 一形式），β 就是时间箭头的局部向量场；对照 Zweig 2026 的 Finsler 轨迹推断说明你们是动力学模型而非插值。

### 覆盖信心：**高**（quasimetric 主线）/ **中**（熵产生对照与 Finsler 侧）
quasimetric RL 一侧通过 emergentmind 主题页与多次检索交叉覆盖，2024–2026 的主要进展应已抓全。"quasimetric + world model"精确短语与"reversibility-aware RL"后续的检索因预算耗尽未能执行，可能漏掉 2026 年把 quasimetric 放进 Dreamer/TD-MPC 类模型的工作。

### 搜索词
`quasimetric reinforcement learning 2025 asymmetric distance` / `arrow of time learned asymmetry irreversibility entropy production latent dynamics representation learning` / `Finsler metric learning neural network dynamics asymmetric geodesic 2024 2025` / `"A Path-Space Formulation of Prediction in World Models" irreversibility` / `quasimetric world model latent goal-conditioned temporal distance 2025 2026 "quasimetric" planning` / `Finsler geometry machine learning arXiv 2503.18010 OR 2603.16708 OR 2406.05400`；另经 https://www.emergentmind.com/topics/quasimetric-reinforcement-learning-qrl 主题页核对清单。未能执行：`reversibility-aware reinforcement learning Grinsztajn irreversibility estimator learned asymmetric transition`、`"quasimetric" "world model" OR "dynamics model" asymmetric latent distance model-based planning 2024 2025 2026`。

---

## 总结：三条里哪条最危险、哪条最空

**最危险：Cell 8（拟度量/Finsler）。** quasimetric 是 RL 里成熟且仍在加速的子领域（Wang/Isola 根源，Eysenbach/Levine 组 2025–2026 连发，NeurIPS 2025 已收），"不对称距离进 latent 世界模型做规划"已有 TD-JEPA 2026 先例，"世界模型里测熵产生/不可逆性"也有 Kim 2026 的路径空间工作。如果论文的主张停留在"学一个不对称距离做世界模型"，会被判为增量。唯一还空的是**不对称场 = 时间箭头地图 + 与熵产生率的定量对照 + Finsler 形式化**——必须把这三点做成论文的主命题，否则不要做。

**最空：Cell 6（联络/和乐）。** 没有任何学习型世界模型用联络 Γ 做动作转移或用曲率/holonomy 检测隐状态；现有群结构世界模型（WLA、Delliaux、SMWM）明确假设可交换/可加并把非交换列为未做局限。但要注意一个精确的理论近邻：**HCDP（2026-07）**已经用离散置换群的 holonomy 证明了"非交换 holonomy ⇔ 需要有序记忆的隐状态"，并给出"被动数据不可辨识"的警示。这篇不会撞掉你们的方法，但会撞掉"holonomy 探测隐状态"这个口号的首创性——应把它当作离散极限的理论锚点来引用，并把差异化放在**连续学习型联络 + 曲率作为连续探测器 + 主动闭环探索**上。

**居中：Cell 4（拓扑优先）。** GD-VAE（2022，2025 仍在修订）已把"流形拓扑 latent + 动力学 + 拓扑错则撕裂"做了一遍，是接近 A 的近邻；但它预先指定拓扑、动力学半手工、结果混合（拓扑匹配不总赢）。"先发现拓扑再学流"的闭环、固定容量下的可证伪命题、以及 Lenia **吸引子**拓扑（现有只有 NCA 吸引子景观和 Lenia 形态拓扑的学生项目）都是空的。最大的实际风险不是撞车而是**优化障碍**（Esmaeili 2023）和**命题被反例打掉**（GD-VAE 表格）——需要把"容量固定"和"可训练性"设计进实验。

**搜索覆盖说明**：本轮在 24 次 WebSearch 后触及共享预算上限，Cell 4 的神经流形拓扑发现线、Cell 6 的精确短语线、Cell 8 的 reversibility-aware RL 后续线未能检索；如需补查可在后续消息中继续。


<!-- ===== 报告 G4 ===== -->

# 文献审计 G4：Cell 5 / Cell 11 / Cell 12

审计日期：2026-11-01。方法：WebSearch（英文为主，约 45 次）+ WebFetch 读 arXiv / 会议页确认。URL 全部来自搜索或 fetch 的真实返回。所有引用为转述。注意：WebSearch 的会话级预算在 Cell 12 搜到一半时用尽（与其他 agent 共享），Cell 12 后半段只能靠已拿到的 URL 做 fetch 确认，覆盖信心相应下调。

---

## Cell 5：可分辨性即规律（模型流形世界模型）

### 撞车等级：B（强近邻；核心可证伪预测尚无人直接检验）

- 同样的几何对象（model manifold / FIM 谱 / hyper-ribbon）已被 Sethna–Transtrum–Chaudhari 一线系统地搬到深度网络上，而且他们**已经做过"不同架构落在同一条低维流形上"的实验**（Mao et al. 2023/2024）。这是最大的撞车风险点。
- 但他们的对象是分类器在训练过程中的预测轨迹，不是世界模型；比较的是"轨迹是否共享流形"，不是"FIM 硬方向是否跨架构重合 / 软方向是否各走各的"；也没有人用 MBAM 从学到的神经动力学模型里提取有效理论。因此定 B 而非 A。

### 最近邻列表

1. **The Training Process of Many Deep Networks Explores the Same Low-Dimensional Manifold** — Mao, Griniasty, Teoh, Ramesh, Yang, Transtrum, Sethna, Chaudhari；2023（arXiv）/ 2024（PNAS，DOI 前缀 10.1073）。https://arxiv.org/abs/2305.01604
   用信息几何（预测空间的 InPCA 式嵌入）分析训练轨迹，发现架构、宽度、优化器、正则、初始化都不同的网络的预测轨迹落在同一条有效低维流形上；不同架构走的路径可区分，但流形共享。
   **接近程度：最高。** 这几乎就是"两种架构学同一个世界，硬方向重合"的分类器版本——只是对象是训练轨迹而非 FIM 本征方向，任务是分类而非动力学。

2. **An Analytical Characterization of Sloppiness in Neural Networks: Insights from Linear Models** — Mao, Griniasty, Sun, Transtrum, Sethna, Chaudhari；2025（arXiv v2 2025-11；搜索结果显示有 PRE 版本 DOI 10.1103/fs4g-9nsz，未能打开确认）。https://arxiv.org/abs/2505.08915
   在线性网络里解析地推导 hyper-ribbon 何时出现：由输入相关矩阵的本征值衰减率、目标输出相对初始权重的尺度、梯度步数三者决定；推广到核机和 SGD。
   接近程度：高（同一几何语言直接用于 NN，并给出 ribbon 出现的相图）。

3. **Does the Data Induce Capacity Control in Deep Learning?** — Yang, Mao, Chaudhari；ICML 2022 spotlight。https://icml.cc/virtual/2022/spotlight/16832 （arXiv: https://export.arxiv.org/abs/2110.14163）
   指出典型数据集输入相关谱"先陡降、后拖出指数级宽的小本征值尾巴"，训练出的网络 Hessian/FIM 也呈同样 sloppy 谱；用 sloppy Hessian 推出非平凡 PAC-Bayes 界。
   接近程度：高。关键含义：**FIM 的 stiff/sloppy 结构很大程度来自数据谱而非"规律"**，这对 Cell 5 的"硬方向 = 规律"解释是一个需要正面回应的混淆因素。

4. **Does the Geometry of the Data Control the Geometry of Neural Predictions? (Student Abstract)** — Cowlagi, Chaudhari；AAAI 2022。https://liner.com/review/does-geometry-data-control-geometry-neural-predictions-student-abstract
   构造 NN 的 model manifold，比较 FIM 本征谱与数据自相关谱；提出把网络投影到 top FIM 本征向量上做模型约简（CIFAR-10 保留 top 5% 本征向量精度几乎不变，top 1% 则崩）。
   接近程度：高（这是"沿软方向压缩"的 NN 原型，但不是 MBAM 的边界极限，只是截断）。

5. **Sloppy model analysis identifies bifurcation parameters without normal form analysis** — Anderson, Transtrum；Phys. Rev. E 108, 064215（2023）。https://par.nsf.gov/servlets/purl/10598937
   提出 TWIG（time-widening information geometry）：在越来越长的观测时窗上算 FIM，看本征值如何随时窗变化，从而识别控制分岔的参数组合、分岔余维与分界面法向，无需化成 normal form。
   接近程度：中高。**它同时是 Cell 5 和 Cell 11 的桥**：FIM 谱（Cell 5 的工具）直接读出分岔结构（Cell 11 的目标）。但对象是已知方程的 ODE，不是学到的世界模型。

6. **Information geometry for multiparameter models: New perspectives on the origin of simplicity** — Quinn, Abbott, Transtrum, Machta, Sethna；2021/2022 综述。https://arxiv.org/abs/2111.07176
   model manifold、hyperribbon、测地线到边界（MBAM）、RG 流与 stiff/sloppy 的对应、InPCA 可视化的系统综述。
   接近程度：根源级（Cell 5 的理论框架全在这里）。

补充近邻（简要）：
- **Universal Statistics of Fisher Information in Deep Neural Networks** — Karakida, Akaho, Amari；AISTATS 2019。https://arxiv.org/abs/1806.01316 — 随机权重大宽度极限下 FIM 谱普适：绝大多数本征值≈0、最大本征值极大。含义：**sloppy 谱在随机初始化时就存在**，不是学到"规律"的标志。
- **Dissecting Hessian: Understanding Common Structure of Hessian in Neural Networks** — Wu, Zhu, Wu, Wang, Ge；2020。https://arxiv.org/abs/2010.04261 — Kronecker 分解猜想，推论之一是不同模型的 top Hessian 本征空间高度重叠。这是"硬方向跨模型一致"在参数空间的间接证据，但重叠来自结构（Kronecker），不来自"世界"。
- **Functional Equivalence and Geometric Diversity in Neural Network Approximations** — Anuragine S A, Jagadeesan；2026。https://arxiv.org/abs/2607.18930 — 用 Hessian 本征谱的 sloppiness 和 effective rank 刻画"功能等价但几何不同"的大等价类。
- **Identifiability-aware neural ODEs (iNODE)** — Campo-Manzanares, Balsa-Canto；2026。https://arxiv.org/abs/2608.13044 — 对 neural ODE 做 Fisher 信息置信区间与可辨识性驱动的结构选择；是"学到的动力学 + FIM"方向目前最近的工作，但没做谱分析和 MBAM。
- **Cortical state transitions and stimulus response evolve along stiff and sloppy parameter dimensions** — Ponce-Alvarez et al.；eLife 2020。https://pmc.ncbi.nlm.nih.gov/articles/PMC7108864 — 对 N=10 神经元的最大熵模型做 FIM 本征分解，发现状态转换沿 stiff 方向、刺激响应沿 sloppy 方向。是"stiff 方向携带动力学上有意义的东西"的实证先例（非 NN）。
- **Manifold boundaries give "gray-box" approximations of complex models** — Transtrum；2016。https://arxiv.org/abs/1605.08705 ；及 APS March Meeting 2024 摘要"Extending MBAM to large-scale, multi-parameter models"（Transtrum）https://meetings-archive.aps.org/mar/2024/n28/10 — MBAM 原文声称适用对象包括 Markov chains 和 artificial neural networks，2024 版把约简改写成一串凸优化以处理高维参数。**没有找到任何把 MBAM 用到学到的 NN 动力学模型上的实作。**
- **Dead Directions: Geometric Singular Learning** — Shirodkar；2026。https://arxiv.org/abs/2606.05957 — 把 Watanabe SLT 与 Fisher 退化方向联系起来；提醒：NN 的 FIM 不只是 sloppy，还是**奇异**的（有精确零方向，来自 gauge 对称）。

### 负面结果 / 风险文献

- **Sloppy models can be identifiable** — Chis, Banga, Balsa-Canto；2014。https://arxiv.org/abs/1403.1417 — 明确论证 sloppiness ≠ 不可辨识性，从 FIM 谱推参数结论会误导。对 Cell 5 的含义：软方向"走到天边预测不变"是局部（线性化）陈述，不能直接当全局结论。
- Karakida 2019（随机初始化已 sloppy）+ Yang/Mao/Chaudhari 2022（FIM 谱镜像输入数据谱）合起来构成对"硬方向 = 架构无关的规律"的最强替代解释：**硬方向可能只是数据协方差的 top 方向**。要区分"规律"与"数据谱"，必须设计让两者分离的实验（例如同一动力学、不同状态编码/不同观测分布）。
- Monsalve-Bravo et al. Sci. Adv. 2022（https://arxiv.org/abs/2203.15184 ）指出 stiff 组合可能被先验而非数据驱动——对带正则/先验的世界模型同样适用。
- 参数冗余是否淹没信号：Anuragine 2026 的"大等价类"和 Shirodkar 2026 的"dead directions"说明 NN 参数空间 FIM 的零/近零方向极多；但 Mao et al. 2023 的做法（在预测空间而非参数空间比较）本身就是绕开冗余的方案——这既是解决路径，也说明"跨架构比较参数空间硬方向"这条路别人已经判定不可行而改走预测空间。

### 还剩什么没做

- 没有人在**世界模型 / 学到的动力学模型**上做 FIM 谱 + 硬方向分析；现有 NN 工作全是分类器或线性网络。
- 没有人检验"两种架构学同一世界，硬方向（预测空间的 tangent 方向）重合、软方向各走各的"这一**可证伪预测**；Mao et al. 只比较了轨迹流形是否共享。
- 没有人用 MBAM（测地线到边界的极限，而非截断）从学到的神经动力学模型里抽出少参数有效理论；TWIG 做了"FIM 谱→分岔结构"，但对象是已知方程。

### 可能的差异化点

- 把"实验"定义为 (state, action) → next-state 预测分布，FIM 在预测空间定义，架构无关性由构造保证；然后比较不同架构的 top 本征方向在预测空间的重合度（subspace overlap / principal angles），并用 Karakida 式随机初始化 baseline 与 Yang 式"数据谱打乱"对照来排除两个替代解释。
- 用 TWIG 的时窗扩展思路：硬方向随预测时域变化的方式，可能是比单一 FIM 更稳的"规律"签名（并天然连接 Cell 11）。
- 把 MBAM 的"边界 = 可解释极限"作为主张：世界模型的有效理论应是模型流形的某个边界（如某些状态变量退化为常数或守恒量），而不是 top-k 截断。

### 覆盖信心：中高
Sethna/Transtrum/Chaudhari 一线的 NN 工作应已覆盖完整；"学到的动力学模型 + FIM"这一侧（neural ODE 可辨识性）只扫到 2 篇，可能漏掉控制/系统辨识社区的工作。

### 搜索词
sloppy models neural networks Fisher information spectrum sloppiness deep learning; model manifold deep neural network sloppy stiff directions parameter space Transtrum; manifold boundary approximation method neural network reduced model MBAM; hyper-ribbon training trajectories deep networks low-dimensional manifold Mao Chaudhari Sethna PNAS; "sloppy" neural network Fisher information eigenvalue spectrum parameter redundancy identifiability 2024; Fisher information matrix spectrum deep networks eigenvalues decay Karakida; MBAM neural dynamics effective theory reduction Hodgkin-Huxley; "sloppiness" recurrent neural network dynamical system fit stiff parameters cross-model comparison; "training process of many deep networks explores the same low-dimensional manifold"; Transtrum Quinn Sethna "model manifold" sloppy ribbons deep learning 2024 2025; neural ODE learned dynamics parameter sensitivity Fisher information sloppy effective model reduction; Fisher information "stiff directions" architecture independent same directions different networks; "model manifold" OR "sloppy" "world model" OR "dynamics model" reinforcement learning; InPCA Quinn Sethna; "manifold boundary approximation" 2024–2026 neural network; Fisher information singular degenerate neural network parameter symmetries; Hessian OR Fisher eigenvectors shared across architectures; Ponce-Alvarez stiff sloppy cortical state transitions; sloppiness model-based reinforcement learning learned dynamics model Fisher information.

---

## Cell 11：奇点即规律（突变/分岔分类世界模型）

### 撞车等级：B（学分岔结构的 ML 工作很多且活跃；但"分岔地图当世界模型/规划器"无人做；Thom 高余维突变分类在 ML 里几乎空白）

- "从数据学分岔图 / 预测分岔类型"已是一个成熟子领域（Bury–Bauch 的 EWS 深度学习系列、Kutz–Brunton 的 normal form autoencoder、Kevrekidis 的 equation-free + 神经算子、Räth 的 parameter-aware reservoir computing、2025–2026 的 neural ODE 跨分岔外推）。
- 但这些工作的目标是科学预测（tipping point 预警、分岔图重建），没有人把"奇点地图"作为 agent 的世界模型输出并用于规划；最接近"用学到的分岔结构做控制"的只有 Köglmayr–Haluszczynski–Räth 的 parameter-aware NGRC 控制。
- Thom 分类（fold/cusp/swallowtail…）在 ML 中仅见 Daw & He 2020 用 DNN 拟合 cusp 统计模型；没有人学"属于哪种 catastrophe"。

### 最近邻列表

1. **Deep learning for early warning signals of tipping points** — Bury, Sohrabi, Garnett, Bauch 等；PNAS 2021。https://www.pubmed.gov/34544867/
   用大量模拟的 normal form 数据训练 CNN-LSTM，在未见过的系统上给出 tipping 预警，并**预测即将到来的分岔的 normal form 类型**（如新状态是否振荡）；在 268 条经验/模型时间序列上优于方差/自相关等通用指标。
   接近程度：高（"学分岔类型"的 SOTA 路线；但是时间序列分类器，不是世界模型）。

2. **Predicting discrete-time bifurcations with deep learning** — Bury, Dylewsky, Bauch, Anand, Glass, Shrier, Bub；Nat. Commun. 2023（DOI 10.1038/s41467-023-42020-z）。https://arxiv.org/abs/2303.09669
   把分类器扩展到五种离散时间余维 1 分岔（period-doubling、Neimark-Sacker、fold 等），在鸡心细胞聚集体实验数据上验证。
   接近程度：高。

3. **Learning normal form autoencoders for data-driven discovery of universal, parameter-dependent governing equations** — Kalia, Brunton, Meijer, Brune, Kutz；2021。https://arxiv.org/abs/2106.05102
   自编码器把潜变量约束为给定 normal form（Hopf、pitchfork、transcritical、saddle-node），学出暴露分岔结构的坐标变换。
   接近程度：高（"把世界压成 normal form 坐标"的直接先例；但 normal form 要事先指定，不是学分类）。

4. **Let's do the time-warp-attend: Learning topological invariants of dynamical systems** — Moriel, Ricci, Nitzan；ICLR 2024。https://arxiv.org/abs/2312.09234 （会议页 https://iclr.cc/virtual/2024/poster/19053 ）
   用拓扑不变特征 + 时间扭曲增强训练卷积注意力网络，对未见系统分类动力学 regime 并定位分岔边界（聚焦超临界 Hopf），应用到单细胞基因表达数据。
   接近程度：中高（"学到 regime 边界地图"最接近"奇点地图"的表述）。

5. **Neural ODEs for Learning and Extrapolating System Dynamics Across Bifurcations** — van Tegelen, van Voorn, Athanasiadis, van Heijster；2025（arXiv；关联 DOI 10.1063/5.0288264，疑为 Chaos）。https://arxiv.org/abs/2507.19036
   把分岔参数作为 neural ODE 的输入，从时间序列学参数依赖的向量场，重建分岔图并外推到训练外参数（Lorenz、Rössler、带全局分岔的捕食模型）；指出外推成败取决于训练覆盖是否含鞍点。
   接近程度：高（"连续轨迹模型 → 事后读分岔图"路线的最新代表，正是 Cell 11 想绕开的"先学轨迹"做法）。

6. **Controlling Dynamical Systems into Unseen Target States Using Machine Learning** — Köglmayr, Haluszczynski, Räth；2024（v2 2026）。https://arxiv.org/abs/2412.10251 （同组前作：Extrapolating tipping points… NG-RC，Sci. Rep. 2024，https://www.nature.com/articles/s41598-023-50726-9 ）
   parameter-aware NGRC 在仅见过平稳数据的情况下外推 tipping，并用学到的参数依赖模型评分候选控制方案，把系统带到未见过的目标状态。
   接近程度：中高——**这是最接近"分岔结构当规划器"的工作**，但规划器仍是轨迹模拟 + 打分，不是显式奇点地图。

补充近邻（简要）：
- **Enabling local neural operators to perform equation-free system-level analysis** — Fabiani, Vandecasteele, Goswami, Siettos, Kevrekidis；Nat. Mach. Intell. 2026（arXiv 2025）。https://arxiv.org/abs/2505.02308 — 学短时神经算子 timestepper，再套 Newton-GMRES / Arnoldi / 伪弧长延拓算出稳态、不稳定分支和分岔图。思路是"学局部动力学，用数值分析读全局分岔"。
- **Machine Learning for Complex Systems Dynamics: Detecting Bifurcations with DNNs (EINNs)** — Pal, Melnik；2026。https://arxiv.org/abs/2603.04420 — 反向映射"平衡态 → 参数"，映射的可行性/连续性突变即临界阈值（saddle-node、多稳态）。
- **In-context learning of tipping points (TipPFN)** — Nathaniel et al.；NeurIPS 2026 气候 workshop。https://neurips.cc/virtual/2026/174667 — 基于 canonical bifurcation 场景合成数据训练 prior-data fitted network，零样本估计离临界转变的距离。
- **Ultra-Early Prediction of Tipping Points: Integrating Dynamical Measures with Reservoir Computing** — Li, Zhu, Zhao, Zhao, Zhang, Duan, Lin；2026。https://arxiv.org/abs/2603.14944 — RC 学局部动力学，再算 Jacobian 主本征值 / Floquet / Lyapunov 趋势外推 tipping 时间。这是"EWS + 学到的动力学"结合的直接例子。
- **Inferring bifurcation diagrams of two distinct chaotic systems by a single machine** — Guo, Du, Yu, Zou, Wang；2026。https://arxiv.org/abs/2604.26632 — 双通道 RC 从少量参数采样重建两个系统的分岔图。
- **Deep Neural Network in Cusp Catastrophe Model** — Daw, He；2020。https://arxiv.org/abs/2004.02359 — 用 NN 直接学 cusp 模型的输入-响应关系以回避非凸参数估计；是**唯一**找到的 Thom 突变 + NN 工作，且是统计建模而非动力学。
- **Sloppy model analysis identifies bifurcation parameters without normal form analysis (TWIG)** — Anderson, Transtrum；PRE 2023。见 Cell 5 第 5 条——FIM 谱随观测时窗的变化可读出分岔余维与控制参数组合。
- 世界模型侧的相关但不同：**World-Model Collapse as a Phase Transition**（Song, Cai 2026，https://arxiv.org/abs/2606.31399 ）讲的是 LLM agent 内部世界模型随任务复杂度的突变崩溃，是关于学习者而非世界的相变；**Changepoint-Aware World Models**（Yang 2026，https://arxiv.org/abs/2609.18950 ）用 CUSUM 检测动力学突变并清旧 replay，无分岔理论。
- Watanabe 的 singular learning theory：只需一句话区分——SLT 研究的是损失景观 / Fisher 退化的奇点（学习者的奇点，如 Shirodkar 2026 "Dead Directions"），Cell 11 研究的是环境稳态结构的奇点。两者唯一可能的交集是：世界模型在逼近有分岔的系统时，其参数空间是否也出现对应的奇异结构——这一点无人研究。

### 负面结果 / 风险文献

- **Can Transformers predict system collapse in dynamical systems?** — Zhai, Grebogi, Lai；2026。https://arxiv.org/abs/2605.04024 — 在正常 regime 训练、在未见 regime 测试：Transformer 在所有配置下都**未能**预测分岔阈值处的崩溃，而 RC 能；归因于注意力的置换不变性。对 Cell 11 的含义：若世界模型主干是 Transformer，"轨迹模型事后读分岔"路线有系统性失败证据，反而支持"直接学奇点地图"的动机。
- **Neural models for prediction of spatially patterned phase transitions: methods and challenges** — Dylewsky, Kéfi, Anand, Bauch；2025。https://arxiv.org/abs/2505.09718 — 对空间斑图相变，神经 EWS 的泛化远不如局部分岔情形，训练/测试数据源互换时性能剧变。
- Köglmayr & Räth 2024 自述：外推可靠性随 tipping 点远离训练样本而下降，缩放超参敏感，有案例预测轨迹发散而真实收敛。
- van Tegelen 2025 自述：训练数据不含鞍点时学不到全局分岔。
- 共同风险：几乎所有正面结果都来自低维、已知方程的玩具系统；高维/像素观测的世界模型里"控制参数"本身不可见，如何从潜变量中识别低维控制参数是无人解决的前置问题。

### 还剩什么没做

- 没有人把"分岔/突变地图"作为世界模型的输出对象并用于规划或探索（所有工作都是预测/预警/重建）。
- Thom 高余维突变（cusp 以上）在 ML 里无人学；现有分类器止于余维 1 normal form。
- 没有人在 RL 环境或高维观测上做：控制参数需要从数据中发现（normal form autoencoder 要求预先指定 normal form 和参数）。
- "EWS（critical slowing down）+ 学到的动力学"只在 RC 线上做过（Li 2026），未与基于模型的 RL 结合。

### 可能的差异化点

- 明确的可测不变量：学到的奇点地图应满足 Thom 分类的拓扑约束（如 cusp 的 3 支稳态与滞后环），且**跨架构一致**——把 Cell 5 的"硬方向跨架构重合"改写成"分岔集跨架构重合"，并用 TWIG 式 FIM 时窗分析作为无需 normal form 的检测器。
- 把奇点地图直接喂给规划器：规划 = 在控制参数空间里绕开/利用跳变（例如把 Köglmayr 的控制任务改成"规划一条不穿过 fold 的参数路径"），这是文献空白。
- 利用 Zhai 2026 的负面结果作为对照实验：同一环境下，轨迹型 Transformer 世界模型 vs 奇点地图型模型在未见 regime 的表现。

### 覆盖信心：中高
Bury/Bauch、Kutz/Brunton、Kevrekidis、Räth、Lai 几条主线都已触到；catastrophe theory + ML 的结果非常少，可能确实是空白，也可能用了不同术语（如"hysteresis learning"、"multistability map"）而漏检。

### 搜索词
learning bifurcation diagrams neural network catastrophe theory dynamics 2024; catastrophe theory machine learning cusp fold learned dynamics neural network 2025; neural network bifurcation discovery learned vector field "bifurcation diagram" data-driven; deep learning early warning signals critical slowing down tipping points Bury Bauch; reinforcement learning world model bifurcation OR "phase transition" OR "tipping point" planning; "bifurcation" "world model" OR "model-based reinforcement learning" latent dynamics saddle-node Hopf; Thom catastrophe "swallowtail" OR "cusp" neural network normal form classification; Bury PNAS 2021 CNN-LSTM bifurcation type classification; normal form identification from data SINDy Kutz Brunton; "learning bifurcation diagrams" reservoir computing parameter-aware extrapolation; data-driven bifurcation analysis "equation-free" Kevrekidis 2024 2025; catastrophe "cusp" deep learning regime switching 2025; reinforcement learning agent tipping point early warning learned dynamics control; "catastrophe theory" "reinforcement learning" OR "world model"; reservoir computing predicting bifurcations unseen parameters negative results; learned dynamical system "topological" OR "qualitative" regime classification world model; Bury Bauch review deep learning generalization failure limitations.

---

## Cell 12：图上的离散曲率找瓶颈（Ollivier–Ricci / Forman 曲率世界模型）

### 撞车等级：A−（几何结构 + RL 探索用途已有直接先例，但只是 workshop 级、tabular、未涉及世界模型与分层规划）

- **已有人在 RL 状态转移图上算 Ollivier–Ricci 曲率并用作探索奖励**：Saheb Pasand, Castro, Bashivan（RLC 2025 RLBrew workshop）。他们的解释（负曲率 = 分支点/房间之间的桥，正曲率 = 死胡同/走廊/角落）与 Cell 12 的"负曲率边 = 瓶颈"基本一致。这是撞车。
- 但该工作：(i) 只在 tabular 迷宫和预计算转移表的 Atari 上做；(ii) 曲率是从随机游走的 successor representation 估计，不是世界模型的输出；(iii) 没有子目标发现、分层规划；(iv) Atari 上增益很小，count-based 在某指标上反超；(v) 是 workshop 投稿（openreview 页面显示匿名、under review）。因此 A−：几何对象和用途撞了，"世界模型输出曲率地图 + 分层规划"这一层没撞。

### 最近邻列表

1. **A Geometric Lens on RL Environment Complexity Based on Ricci Curvature** — Saheb Pasand, Castro, Bashivan；RLC 2025 RLBrew workshop。https://openreview.net/pdf?id=9xPs0jKRZA （workshop 列表 https://rlbrew2-workshop.github.io/papers.html ；作者页 https://mila.quebec/fr/ali-saheb-pasand ）
   在观测转移构成的连通图上，用 SR 矩阵行作为每个状态的概率测度算 ORC；负曲率对应随机游走发散的分支/桥，正曲率对应收敛的死胡同；曲率统计与 Laidlaw 等人的环境复杂度指标相关；用 −ORC 作内在奖励，在无重启迷宫上覆盖速度优于 count-based。
   **接近程度：直接撞车（探索用途）。**

2. **Understanding over-squashing and bottlenecks on graphs via curvature** — Topping, Di Giovanni, Chamberlain, Dong, Bronstein；ICLR 2022（arXiv 2021）。https://arxiv.org/abs/2111.14522
   提出 Balanced Forman 曲率，证明负曲率边导致 over-squashing，并提出基于曲率的 rewiring。
   接近程度：根源级（"负曲率边 = 瓶颈"的严格版本）。

3. **Revisiting Over-smoothing and Over-squashing Using Ollivier-Ricci Curvature** — Nguyen, Nong, Nguyen, Ho, Osher, Nguyen；2022/2023（页面未列会议）。https://arxiv.org/abs/2211.15779
   用 ORC 统一解释 over-smoothing（正曲率）与 over-squashing（负曲率），提出 Batch Ollivier-Ricci Flow rewiring。
   接近程度：中（同一工具，GNN 语境）。

4. **Community Detection on Networks with Ricci Flow** — Ni, Lin, Luo, Gao；2019（Sci. Rep.）。https://arxiv.org/abs/1907.03993
   用离散 Ricci flow 把网络分解成社区；社区间的桥在 flow 下被拉长。
   接近程度：中（"曲率找社区边界"是 Cell 12 子目标发现的图论原型）。

5. **Comparative analysis of two discretizations of Ricci curvature for complex networks** — Samal, Sreejith, Gu, Liu, Saucan, Jost；Sci. Rep. 2018。https://ouci.dntb.gov.ua/works/4NgwDD6l
   在大量模型/真实网络上，Forman 与 Ollivier 曲率高度相关（augmented Forman 更高），建议粗分析时用快得多的 Forman 代替。
   接近程度：中（决定 Cell 12 工程上选 Forman 还是 Ollivier 的依据）。

6. **Curved Markov Chain Monte Carlo for Network Learning** — Sigbeku, Saucan, Monod；2021。https://arxiv.org/abs/2110.03413
   把 Forman 曲率显式写进 Markov 链转移概率和目标平稳分布，加速网络统计量收敛。
   接近程度：中（"曲率进入 Markov 链转移"的先例，但用于采样而非 MDP）。

补充近邻（简要）：
- **Analyzing Neural Network Robustness Using Graph Curvature** — Tan, Sia, Bogdan, Ivanov；2024（DOI 10.1109/ICAA64256.2024.00026）。https://arxiv.org/abs/2410.19607 — 在"神经数据图"上算 ORC，负曲率边 = 承载数据流的瓶颈边，与鲁棒性相关。
- **Discrete Curvature Graph Information Bottleneck (CurvGIB)** — Fu et al.；AAAI 2025。https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/AAAI/2025/discrete_curvature_graph_information_bottleneck__6e7febfb.txt — 用可微 ORC 近似 + Ricci flow 在信息瓶颈目标下学"信息传输结构"；负曲率边承载 backbone 路径。
- **Scalable Graph Curvature: Bounding Ollivier-Ricci via Local Combinatorics** — Micaletto, Nigrelli；NeurIPS 2026 OT workshop。https://neurips.cc/virtual/2026/159361 — 用 Balanced Forman 给 ORC 两侧分段仿射界，每边复杂度降到 O(max deg^1.5)。对 Cell 12 在大状态图上算曲率直接有用。
- **A Geometric Perspective on the Difficulties of Learning GNN-based SAT Solvers** — Skenderi；2025。https://arxiv.org/abs/2508.21513 — 随机 k-SAT 二部图负曲率随难度加深，曲率可预测泛化误差。"曲率 = 问题复杂度"与 Saheb Pasand 同调。
- 非曲率的瓶颈/子目标发现竞争者（必须作为 baseline）：**Q-Cut**（Menache, Mannor, Shimkin；ECML 2002；max-flow/min-cut 找瓶颈）https://webee.technion.ac.il/people/shimkin/PAPERS/QCut.pdf ；**Discovering Options for Exploration by Minimizing Cover Time**（Jinnai, Park, Abel, Konidaris；ICML 2019）https://proceedings.mlr.press/v97/jinnai19b.html ；**Bottleneck-Guided Spectral Subgoals for Offline GCRL**（Liang et al.；ICML 2026；Laplacian 谱聚类找瓶颈，AntMaze 96.5%）https://icml.cc/virtual/2026/poster/61973 ；**Learning Interpretable Options by Identifying Reward Diffusion Bottlenecks (VPS)**（Fei, Wang, Yan, Tang；ICML 2026；基于价值函数的瓶颈度量，明确批评"转移图拓扑分析在高维/连续域不可扩展"）https://icml.cc/virtual/2026/poster/64243 ；**Grounding subgoals in information transitions**（van Dijk, Polani；ADPRL 2011）https://uhra.herts.ac.uk/id/eprint/13234/ ；**Subgoal Discovery Using a Free Energy Paradigm**（Mesbah et al. 2024）https://arxiv.org/abs/2412.16687 。
- 世界模型 + 几何的近邻：**GeoWorld: Geometric World Models**（Zhang, Li, Reid, Hartley；CVPR 2026）https://arxiv.org/abs/2602.23058 — Hyperbolic JEPA 把潜表征放到双曲流形做多步规划；是"常负曲率潜空间"而非"离散曲率地图"。**World Model as a Graph (L3P)**（Zhang, Yang, Stadie；ICML 2021）https://proceedings.mlr.press/v139/zhang21x.html — 图结构世界模型（潜地标 + Q 函数可达性边），无曲率；是 Cell 12 最自然的宿主架构。
- 数学背景：Wiesler 2026 论文（https://arxiv.org/abs/2604.14211 ）把 ORC 推广到有向图——MDP 转移图是有向的，这是 Cell 12 必须处理的技术点。

### 负面结果 / 风险文献

- Saheb Pasand 等自述：tabular Atari 上曲率奖励增益很小，count-based 在熵指标上略胜；在线估计算法未评估；只与 random / count-based 比较。
- Fei et al. ICML 2026 的立论本身是对"转移图拓扑分析"路线的否定：高维/连续域建不出可靠转移图。Cell 12 若坚持"先建图再算曲率"，会被这类 value-based 方法在扩展性上压制。
- Topping 2022 与 Nguyen 2023 都把负曲率处理成"要修掉的问题"（rewiring）；Cell 12 反其道把负曲率当有用信号，需要论证为什么在 MDP 上瓶颈信息值得保留而非消除。

### 还剩什么没做

- 没有人让**学到的世界模型**输出曲率地图（现有工作从经验转移计数 / SR 估计曲率，不从模型泛化）。
- 没有人用离散曲率做**子目标发现或分层规划**（Saheb Pasand 止于探索奖励；瓶颈子目标工作全用 min-cut / Laplacian 谱 / 价值扩散）。
- 没有人比较 ORC vs Forman vs Laplacian 谱 vs VPS 在同一批 MDP 上找到的瓶颈集合是否一致（Samal 2018 只在社交/生物网络上比过 Forman vs Ollivier）。
- 有向 MDP 图上的曲率定义与动作依赖（每个动作一张图还是策略平均）无人系统处理。

### 可能的差异化点

- 可测不变量：世界模型预测的曲率地图应在不同探索策略、不同架构下给出同一组负曲率边（瓶颈集合的 Jaccard 一致性），并与真实图的 min-cut / Laplacian 谱瓶颈对齐——这是 Saheb Pasand 没做的"曲率 vs 其他瓶颈定义"对照。
- 把曲率嵌入 L3P 式图结构世界模型：地标之间的边曲率由模型预测的多步转移分布算出（ORC 天然基于转移核，Ollivier 2009 原始定义就是 Markov 链曲率），从而在连续域避开"建完整转移图"的扩展性批评。
- 用 Micaletto–Nigrelli 的 BF→ORC 界做廉价在线估计，与 Samal 的 Forman 替代一起回应计算成本问题。

### 覆盖信心：中
核心撞车（Saheb Pasand 2025）已确认并读了全文摘要。但 WebSearch 预算在本节用尽，以下计划中的检索**未能执行**：Forman curvature transition graph planning；curvature-based subgoal discovery hierarchical RL 的其他措辞；ORC + successor representation 的后续工作（Saheb Pasand 可能已有 2026 扩展版，未查到）；离散曲率在 2025–2026 的非 GNN 新应用（如 LLM 推理图、分子）。建议主会话补一轮。

### 搜索词
Ollivier-Ricci curvature reinforcement learning state transition graph bottleneck; Ricci curvature MDP state graph reinforcement learning exploration subgoal discovery bottleneck; "Forman" OR "Ollivier" curvature "reinforcement learning" OR "Markov decision process" transition graph; discrete curvature graph bottleneck community detection Ollivier-Ricci negative curvature bridges Ni Lin Gao "Ricci flow"; curvature-based subgoal discovery hierarchical reinforcement learning graph bottleneck states betweenness alternatives 2024 2025; "A Geometric Lens on RL Environment Complexity Based on Ricci Curvature"; Ollivier-Ricci curvature RL environment complexity successor representation exploration reward; Ollivier-Ricci curvature Markov chain transition probability "transition graph" random walk mixing bottleneck 2024 2025; curvature state space reinforcement learning "negative curvature" OR "hyperbolic" OR "Ricci" exploration goal-conditioned latent world model graph 2025 2026; RLC 2025 workshop Ricci curvature reinforcement learning environment complexity openreview。（未执行：Ollivier-Ricci "successor representation" OR "state-transition graph" 2025 curvature exploration bonus；graph curvature "subgoal" OR "option discovery" OR "bottleneck states" Ricci 2023 2024。）

---

## 总结：三条里哪条最危险、哪条最空

**最危险：Cell 12。** 同一几何对象（状态转移图上的 Ollivier–Ricci 曲率）、同一解释（负曲率 = 桥/分支，正曲率 = 死胡同）、同一用途之一（探索奖励）已在 RLC 2025 workshop 出现（Saheb Pasand, Castro, Bashivan），而且 Castro 组有能力很快推出正式版。Cell 12 要活下来，必须把重心放在他们没做的三件事上：世界模型预测曲率（而非经验计数）、曲率驱动的子目标/分层规划、以及与 min-cut / Laplacian 谱 / VPS 等非曲率瓶颈定义的对照。同时要正面回应 Fei et al. ICML 2026 对"转移图分析不可扩展"的批评。

**次危险：Cell 5。** Sethna–Transtrum–Chaudhari 一线已经把 model manifold / hyper-ribbon / FIM sloppiness 全部搬到深度网络，并做了"多架构共享同一预测流形"的实验（PNAS 2024）。但他们没碰世界模型，没检验"硬方向重合、软方向各走各的"这一具体预测，也没用 MBAM 提取有效理论。更大的科学风险不是撞车而是替代解释：Karakida 2019（随机初始化就 sloppy）和 Yang–Mao–Chaudhari 2022（FIM 谱镜像数据谱）意味着"硬方向 = 规律"很可能被"硬方向 = 数据协方差"解释掉，实验设计必须能区分二者。

**最空：Cell 11。** "学分岔"本身是热门子领域（Bury/Bauch、Kutz/Brunton、Kevrekidis、Räth、Lai 等），所以不是无人区；但"奇点地图作为世界模型输出并用于规划"没有任何先例，Thom 高余维突变分类在 ML 里几乎只有一篇 2020 年的 cusp 统计拟合，且 Zhai–Grebogi–Lai 2026 给出的"Transformer 预测不了分岔崩溃"负面结果反而为这条路提供了动机。空白的代价是前置问题也无人解决：高维观测下如何发现低维控制参数。Anderson–Transtrum 2023 的 TWIG（FIM 时窗谱读出分岔结构）是把 Cell 5 和 Cell 11 串成一条线的现成工具，值得作为两条想法的共同支点。


<!-- ===== 报告 G5 ===== -->

# 文献审计 G5：Cell 13 / Cell 14 / Cell 15 + 总体项 U

审计日期：2026-11-01。方法：WebSearch（英文为主，共约 50 次）+ WebFetch 读 arXiv / 会议页 / 项目页确认（约 70 次）+ Hugging Face papers 索引搜索（约 20 次，补 WebSearch 预算用尽后的缺口）。URL 全部来自搜索、fetch 或 HF 索引的真实返回；所有引用为转述，不抄原文。

**预算说明**：WebSearch 的会话级预算（与其他 agent 共享）在 Cell 15 搜到约三分之二时用尽。之后 Cell 15 的尾段和总体项 U 的 (a)(b)(c) 全部改用 HF papers 索引 + 直接 fetch arXiv 摘要页完成。HF 索引覆盖 arXiv daily papers 但不是全量 arXiv，所以 U(a) 的"伞状纲领是否已有人提出"和 U(c) 的"AWM 跟进"覆盖信心相应下调，文中已标注。

---

## Cell 13：因果正交（信息几何因果世界模型）

### 撞车等级：B（两半不对称："独立机制 = 模块化世界模型的组织原则"这一半接近 A，已被 Schölkopf–Posner 一线占住；"用 IGCI 式正交性给动力学变量定因果箭头"这一半是 B 偏 C）

- 根源层面，Janzing–Schölkopf 自己早就把"原因与机制独立"推广到了动力学：线性系统的 Spectral Independence Criterion（SIC，2015）和 CLeaR 2022 的理论基础；以及"初始条件与动力学律算法独立 → 时间箭头 + 因果不对称"（NJP 2016）。这意味着**Cell 13 的原理在动力学里有人做过，但只做到线性、双变量、非学习模型**。
- 世界模型层面，"independent mechanisms for modular world models"这句话已经是 COMET（Lei, Nolte, Schölkopf, Posner 2024）的标题；VCD（TMLR 2023）和 SPARTAN（2024）同一组人把 ICM / sparse mechanism shift 当作模块化世界模型的动机。但这三篇都**没有把独立性当作可计算的信息几何判据**（COMET 用 winner-takes-all 竞争，SPARTAN 用稀疏正则硬注意力，VCD 用跨环境不变性），也不输出因果方向的不变量。
- 没有找到任何一篇把 IGCI 的正交性（或其 Fisher/熵形式）用在学到的潜空间转移上、用于给潜变量之间定箭头方向的工作。这就是空位。

### 最近邻列表

1. **Cause-effect inference through spectral independence in linear dynamical systems: theoretical foundations** — Besserve, Shajarisales, Janzing, Schölkopf；CLeaR 2022（PMLR v177, pp. 110–143）。https://proceedings.mlr.press/v177/besserve22a.html （arXiv: https://arxiv.org/abs/2110.15595v1 ）
   把 ICM 原则落到时间序列：原因的功率谱与机制（滤波器）的平方频响不相关（SIC）；给出信息论解释、可识别性条件、对降采样的鲁棒性（Granger 在降采样下会坏），并用"不变性视角"指出 SIC 假设的极限与推广方向。
   **接近程度：最高（原理层）。** 这就是"因果正交用于动力学"的正统版本，但限于线性时不变系统、双变量、显式时间序列，没有学习的潜空间。前身：Shajarisales, Janzing, Schölkopf, Besserve 2015 "Telling cause from effect in deterministic linear dynamical systems"（https://emergentmind.com/papers/1503.01299 ）。

2. **Algorithmic independence of initial condition and dynamical law in thermodynamics and causal inference** — Janzing, Chaves, Schölkopf；New Journal of Physics 18, 093052（2016）。https://arxiv.org/abs/1512.02057
   提出"初始条件与动力学律典型地算法独立"原则，由此同时推出热力学时间箭头和 P(cause) 不含 P(effect|cause) 信息的因果不对称。
   接近程度：高（根源）。它给了 Cell 13 一个可测预测：正向学到的世界模型应比反向学到的更满足"状态分布 ⟂ 转移机制"。相关：Janzing 2009 "On the entropy production of time series with unidirectional linearity"（J. Stat. Phys.，https://arxiv.org/abs/0908.1861 ）。

3. **Compete and Compose: Learning Independent Mechanisms for Modular World Models (COMET)** — Lei, Nolte, Schölkopf, Posner；arXiv 2404.15109（2024）。https://arxiv.org/abs/2404.15109v1
   两阶段：竞争期用 winner-takes-all 梯度分配让各机制专门化，组合期重组机制去适应被干预的环境；在图像环境里无监督恢复出可辨认的机制，对物体数变化的适应比微调更省样本。
   **接近程度：高（世界模型层）。** 占住了"独立机制 → 模块化世界模型"的叙事，但独立性是程序性的（竞争），不是信息几何判据，不定因果方向。

4. **Variational Causal Dynamics: Discovering Modular World Models from Interventions (VCD)** — Lei, Schölkopf, Posner；TMLR 2023。https://mlanthology.org/tmlr/2023/lei2023tmlr-variational （arXiv: https://arxiv.org/pdf/2206.11131 ）
   把转移模型按因果图分解，联合变分推断学表示和结构；利用机制跨环境不变性，在新环境里只识别稀疏变化的机制。
   接近程度：高。SPARTAN（Lei, Schölkopf, Posner 2024，https://arxiv.org/html/2411.06890v3 ）是其 Transformer 版：Bernoulli 边 + Gumbel-softmax 硬注意力 + 拉格朗日约束的稀疏化，学状态依赖的局部因果图；摘要明确说可看作 ICM 的一个实例，但 ICM 不是训练判据。

5. **Information-Geometric Neural Granger Causality (IG-NGC)** — Bourigault, Mandic；ICML 2025 workshop（HiLD）。https://icml.cc/virtual/2025/47688
   把神经网络看成在学统计流形，用 Fisher 度量定义信息几何 Granger 因果量（IGGC），证明现有神经 Granger 方法是其近似（高斯下精确），给一致性与样本复杂度界。
   接近程度：中。它是"信息几何 + 时间序列因果"，但走的是 Granger 式"方向性信息流"，不是 IGCI 的"分布与机制正交"，也不是世界模型。关键提醒：标题里"information-geometric causality + time series"已被占用，命名要避开。

6. **Mechanistic Independence: A Principle for Identifiable Disentangled Representations** — Matthes, Han, Shen；ICLR 2026。https://arxiv.org/abs/2509.22196 （ICLR 页：https://iclr.cc/virtual/2026/poster/10011933 ）
   用"因子如何作用于观测"（Jacobian 的支撑/稀疏/高阶交叉导数结构）而非潜分布来定义独立，IMA（Gresele et al. 2021，Jacobian 列正交，https://arxiv.org/abs/2106.05200v1 ）是其特例；给局部→全局可辨识性。无动力学。
   接近程度：中（几何化的"机制独立"已有层级化定义，可直接借到转移 Jacobian 上）。

7. **A Unifying Perspective on Causal World Models: From Observations to Representations to Structure** — Kori, Russo；arXiv 2608.13456（2026-08，v2 09）。https://arxiv.org/abs/2608.13456
   给因果世界模型下形式定义，连接 CRL / 物体中心 / 因果发现 / SCM；可辨识性表指出潜变量间的因果图只能识别到 Markov 等价类，边的定向只能靠动作结构；不提 ICM、SMS 或任何信息几何判据，不推荐 benchmark（全文 fetch 确认）。
   接近程度：中（它恰好把"定向"留白，是 Cell 13 可以切入的点）。

补充近邻（简要）：
- Tagliapietra, Ensinger, Zimmer, Mian, **CADYT**（AAAI 2026，https://arxiv.org/html/2512.14361v1 ）：连续时间/不规则采样动力系统的因果结构学习，用 Algorithmic Markov Condition + MDL 做贪心搜索。算法独立性味道的判据用在动力学上，但 GP 模型、不是世界模型。
- Huang, Zhang 等 **CD-NOD**（https://arxiv.org/abs/1903.01672 ）：用非平稳数据里"机制独立变化"定向因果边。
- Zhao 等 "Curious Causality-Seeking Agents Learn Meta Causal World"（2025，https://arxiv.org/abs/2506.23068 ）：元因果图 + 好奇心干预；无 ICM/IG。
- Ding 等 "CausalVAE as a Plug-in for World Models"（2026，https://arxiv.org/pdf/2604.07712 ）：DAG 无环惩罚插件；无 ICM/IG。
- Hwang 等 "Quantized Local Independence Discovery for Fine-Grained Causal Dynamics Learning in RL"（NeurIPS 2023 GenPlan，https://neurips.cc/virtual/2023/79912 ）：事件依赖的局部独立。
- **命名撞车警告**：Cui, Heng, Ouyang "Orthogonal JEPA: Factorized Predictive States for Latent World Models"（2026-08/09，https://www.alphaxiv.org/abs/2608.20065 ，v2 已撤回并由 2609.20800 取代）——"orthogonal + world model"已有人用，但其正交是预测分量之间去冗余的几何正交，与因果无关。
- Surasinghe & Bollt "On Geometry of Information Flow for Causal Inference"（Entropy 2020，https://arxiv.org/abs/2002.02078v1 ）和 Choong, Kim, He "Causality Analysis with Information Geometry: A Comparison"（Entropy 2023，https://www.mdpi.com/1099-4300/25/5/806 ）：都叫"几何/信息几何因果"，前者用分形维数，后者用 information rate；都不是 IGCI 正交性。

### 负面结果

- **IGCI 本身的脆弱性**：Mooij 等 JMLR 2016 benchmark（https://www.jmlr.org/beta/papers/v17/14-518.html ）摘要只报告 ANM 最好（63±10%），IGCI 的具体数字在摘要里没有；据我所知（本次未在页面核实）IGCI 对参考测度（均匀/高斯）和估计器（slope/entropy）敏感，且假设近似确定性关系，噪声大时退化。Cell 13 在学到的潜空间里用 IGCI 必须正面处理噪声与参考测度问题。
- Besserve 等 2022 自己指出 SIC 假设在"不变性视角"下有明确极限。
- Kori & Russo 2026：仅观测数据下潜变量因果图只到 Markov 等价类——这既是负面结果也是机会：要定向就得加 ICM/IGCI 这类不对称假设。

### 还剩什么没做

- 非线性、学到的潜空间里，"潜状态分布 ⟂ 转移机制（Jacobian/条件分布）"的 IGCI 式正交性是否真能给潜变量定向，没人测过；SIC 只有线性情形。
- 没有任何世界模型把"每条边的正交性得分"当作输出的不变量；也没人检验 Janzing–Chaves–Schölkopf 的预测——正向学的世界模型比反向学的更满足正交。
- COMET/VCD/SPARTAN 的模块边界由竞争或稀疏决定，没人用正交性来决定模块切分。

### 可能的差异化点

- 把 IGCI 判据当**诊断量**（不当训练信号）：对学到的转移 f 计算每对潜变量的正交性残差，线性情形与 SIC 对照做 sanity check，并用 CausalDynamics 这类有真值图的 benchmark（见 U(b)）评估定向准确率。
- 用"正向/反向训练的世界模型正交性差"做时间箭头检验，把 Cell 13 和热力学不对称绑定，这是 COMET 一线没碰的角度。
- 用正交性替代 winner-takes-all 决定机制切分，与 COMET 直接对比。

### 覆盖信心：中高（ICM-世界模型一线很密、已基本穷尽；"IGCI 用于动力学"的统计学/物理学期刊侧可能有漏，例如 Janzing 近年的因果-热力学工作未逐篇查）

### 搜索词
WebSearch："information geometric causal inference time series dynamical systems"；"independent causal mechanisms world model reinforcement learning modular dynamics"；"IGCI "information geometric causal inference" dynamical systems OR "time series" 2024"；"sparse mechanism shift world model causal 2025"；"causal world model orthogonality mechanism independence latent dynamics arXiv 2025 2026"；""independent causal mechanisms" "time series" OR "dynamical" identifiability mechanism independence 2024 2025"；""Mechanistic Independence" identifiable disentangled representations arXiv 2509.22196"；"causal direction learned latent dynamics world model asymmetry "independence of cause and mechanism" reinforcement learning"；"Janzing Schölkopf "algorithmic independence" dynamics time series arrow of time causal direction learned model 2023 2024"；"causal world models survey 2025 "causal dynamics learning" modular factored transition model"；""independent mechanism analysis" Gresele orthogonal Jacobian extension dynamics time series sequential 2024 2025"；"IGCI limitations noise failure "information geometric causal inference" benchmark Tübingen pairs negative result"；"learning causal direction between latent state variables world model Fisher information orthogonal decomposition 2026"；""causal world model" OR "causal dynamics" "independence of mechanisms" OR "ICM principle" reinforcement learning transition model 2024 2025 arXiv"；""Recurrent Independent Mechanisms" Goyal world model dynamics modular successor 2024 2025"。
HF papers："independent causal mechanisms world model"；"independence of cause and mechanism causal direction time series"。

---

## Cell 14：Klein 层级（Erlangen 纲领世界模型）

### 撞车等级：B（对称性发现是成熟子领域；"发现的群 + 潜动力学联合学习"已有（NeurIPS 2026）；"比较不同 Klein 层级的世界模型几何"有一篇玩具级（2023）；但"自动判定世界保持到哪一级不变量、并据此决定世界模型几何 / 把层级当输出"没人做）

- 对称性发现（Yu–Walters–Dehmamy 一线：LieGAN → LaLiGAN → AtlasD → LieFlow；Brunton 一线：JMLR 2025 统一框架）已经能从数据里学出 Lie 代数基、非线性对称、局部对称、离散子群。
- 把"发现的生成元"和"潜 ODE"联合拟合的工作已经出现：LatentLieODE（NeurIPS 2026）从视频学 Lie 对称 + 第一积分；LieStoNet（ICML 2026）学 SDE 的 Lie 点对称。
- 从交互/动作学群结构并用于转移预测：Quessard 2020、Homomorphism AutoEncoder 2022、Barthélémy 等 2026（动作群的直积分解 + 可辨识性）。
- **但以上全部都是"找出一个群"而不是"判定在 Klein 层级里处于哪一级"**；也没有工作把"层级"作为可测不变量（交比残差 / 平行性残差 / 距离残差 / Betti 数）输出。离这个最近的是 Sergeant-Perthuis 等 2023 的玩具比较（欧氏 vs 射影，群是人选的）和 eLife 2025 的心理物理 + DNN 实验（Klein 层级决定迁移方向，不是世界模型）。

### 最近邻列表

1. **Data-driven Discovery of Lie Symmetries and First Integrals with Latent ODEs (LatentLieODE)** — Sampson, Melchior；NeurIPS 2026。https://neurips.cc/virtual/2026/157484
   自编码器到去相关潜空间，联合拟合潜 ODE f 和一组线性生成元 A_k（可微 Lie 括号残差把 Jacobian(f) 与 A 耦合），再训练网络给出第一积分；六个基准上不变量与真值相关 |ρ|≥0.978，在球面测地系统上恢复非阿贝尔 so(3) 并用 Killing 型负定来认证。
   **接近程度：最高。** 这是"从观测学世界的变换群并让它约束潜动力学"的直接实现；差别：只找线性 Lie 对称，不判层级，不把层级用来选几何。

2. **Latent Space Symmetry Discovery (LaLiGAN)** — Yang, Dehmamy, Walters, Yu；ICML 2024（PMLR 235, pp. 56047–56070）。https://proceedings.mlr.press/v235/yang24g.html （arXiv: https://arxiv.org/abs/2310.00105v3 ）
   学一个映射把非线性群作用变成潜空间里的线性作用，再在潜空间发现对称；高维动力系统实验，潜空间用于方程发现和长程预报。
   接近程度：高。前作 **LieGAN**（Yang, Walters, Dehmamy, Yu，ICML 2023，https://mlanthology.org/icml/2023/yang2023icml-generative ；arXiv https://arxiv.org/html/2302.00236v4 ）发现 SO(n)、SO(1,3)+ 的 Lie 代数基，用于轨迹预测与 top-quark tagging；后作 **AtlasD**（Bhat, Park, Yang, Dehmamy, Walters, Yu，ICML 2025，https://proceedings.mlr.press/v267/bhat25a.html ）发现只在局部邻域成立的对称（atlas equivariance）、多连通分支的群，PDE 实验；**LieFlow**（Park, Chen, Eijkelboom, van de Meent, Wong, Walters，2025，https://arxiv.org/abs/2512.20043 ）用 Lie 群上的 flow matching 学"假设群上的分布"，能找离散子群（C4、D4、四面体、八面体），但只在无噪声点云上，且承认把发现的对称接回下游等变网络仍是开放问题。

3. **A Unified Framework to Enforce, Discover, and Promote Symmetry in Machine Learning** — Otto, Zolman, Kutz, Brunton；JMLR 26(248):1–83, 2025。https://www.jmlr.org/papers/v26/24-1315.html
   用 Lie 导数统一三件事：强制已知对称、发现未知对称（与强制互为对偶的线性代数问题）、"促进"对称（核范数凸正则，除非数据明确破缺否则偏向对称）；适用于动力系统发现、神经算子。
   接近程度：高。"promote unless data breaks it"是 Klein 层级里"从最大群往下退"的现成优化语言。配套的"选多少对称"工作：van der Ouderaa, Immer, van der Wilk "Learning Layer-wise Equivariances Automatically using Gradients"（2023，https://arxiv.org/abs/2310.06131 ，用 Laplace 近似边际似然选每层等变程度）；Pertigkiozoglou, Petrache, Trivedi, Daniilidis **RECM**（2026，https://arxiv.org/abs/2602.02853 ，每层松弛水平收敛到该层的 symmetry gap）。两者都是图像分类，不是动力学。

4. **Disentangled Representation Learning through Unsupervised Symmetry Group Discovery** — Barthélémy, Annabi, Argentieri；arXiv 2603.11790（2026-03）。https://arxiv.org/abs/2603.11790
   具身智能体从无监督交互中发现动作群的直积分解 G = G₁×…×G_K（基于群论伪距离重组动作），证明分解的可辨识性，A-VAE 把动作当潜空间上的矩阵 ρ(g) 做转移预测；三个不同分解的环境（如 Colored Flatland：水平平移 × 垂直平移 × 颜色移位）。
   接近程度：高（"从交互发现群 + 转移模型 + 可辨识性"三件齐全），但发现的是直积结构，不是 Klein 层级。同线前作：Keurti, Pan, Besserve, Grewe, Schölkopf **Homomorphism AutoEncoder**（2022，https://arxiv.org/abs/2207.12067 ，无先验群、从转移学动作的群表示、改进多步动作效果预测）；Quessard, Barrett, Clements 2020（https://arxiv.org/abs/2002.06991 ）。

5. **Influence of the Geometry of the world model on Curiosity Based Exploration** — Sergeant-Perthuis 等；arXiv 2304.00188（2023）。https://arxiv.org/abs/2304.00188 （题名与作者来自 emergentmind 主题页 https://www.emergentmind.com/topics/group-structured-world-models ；arXiv 页正文已核对为同一内容，但该页因限速未取到元数据）
   主动推理玩具模型：内部世界模型取 R³ + 仿射变换（欧氏）或 P³(R) + PGL 射影变换；只用认知价值选动作时，欧氏智能体原地不动（不确定区域大小与位置无关），射影智能体会靠近物体（靠近放大不确定区域、观测更有信息量）。群由研究者选定，明确说表示学习留作未来工作。
   **接近程度：主题上最接近 Cell 14 的"哪一级几何决定世界模型行为"，但无学习、无判定。**

6. **The asymmetric transfers of visual perceptual learning determined by the stability of geometrical invariants** — Yang, Zhuo, Zuo, Zhuo, Chen；eLife reviewed preprint 93959 v2（2025-03）。https://elifesciences.org/reviewed-preprints/93959
   直接用 Klein 的 Erlangen 层级（射影 > 仿射 > 欧氏的稳定性）设计三类判别任务：人类训练低稳定不变量能迁移到高稳定不变量、反之不行；DNN 复现同样的不对称迁移，权重改变的层分布符合 reverse hierarchy theory。
   接近程度：中（不是世界模型，但是"Klein 层级作为实验变量"的唯一现成范式，可借其任务设计做 benchmark）。

补充近邻（简要）：
- **群给定的等变世界模型**：Park, Biza, Zhao, van de Meent, Walters **SEN**（2022，https://ar5iv.labs.arxiv.org/html/2204.11371 ，C₄×S₅ / D₄⋉R² / SO(3)，学输入侧作用）；Delliaux 等 "Learning Abstract World Models with a Group-Structured Latent Space"（2025/26，https://arxiv.org/abs/2506.01529 ）；Lillemark, Huang, Zhan, Du, Keller **Flow Equivariant World Models**（2026，https://arxiv.org/abs/2601.01075 ，单参数 Lie 群流做记忆）；Hongbo Wang "Exact equivariance, kept through training, buys zero-shot generalisation across the symmetry group"（2026，https://arxiv.org/abs/2606.03003 ）。
- Hayashi, Koyama, Guerreiro **WLA**（2025，https://arxiv.org/abs/2503.09911 ）：潜动作建模为 Lie 群（html 全文确认：假设连通紧阿贝尔群、块对角缩放+旋转、频率固定；群族是预设的，不是发现的；不测群不变量）。Koyama 等 **Neural Fourier Transform**（ICLR 2024，https://proceedings.iclr.cc/paper_files/paper/2024/hash/1aa1fde3661b23ba9b043082069fd144-Abstract-Conference.html ）：无显式群知识学潜线性群作用。
- Mitchel, Taylor, Sitzmann **Neural Isometries**（2024，https://arxiv.org/abs/2405.19296 ）：学一个潜空间让未知的世界变换变成等距，可从相邻视角回归相机位姿——这是"把更大的群压成等距群"的例子，但只到欧氏级。
- Li, Xiao, Guo, Weng **Latent MoS**（2025，https://arxiv.org/abs/2510.03578 ）：不假设单一全局群，用对称支配的潜因子混合学动力学。
- Hou, Li, You **MLSD**（2024/26，https://arxiv.org/abs/2412.14632 ）：从相空间轨迹学连续对称生成元及其 Lie 代数结构常数（so(4)、su(3)）；仅可积系统。
- Gabel 等（MLST 2024，https://doi.org/10.1088/2632-2153/ad2629 ）：MLP 从 PDE 时间序列识别 Lie 点对称生成元，明确不把它用来约束模型。Liu, Gupta, Sinha, Mahadevan **LieStoNet**（ICML 2026，https://icml.cc/virtual/2026/poster/62662 ）：SDE 的 Lie 点对称。
- Santos-Escriche, Lin, Jegelka **LieAugmenter**（2025/26，https://arxiv.org/abs/2506.03914 ）：可学习增广发现连续对称，N-body 动力学实验，学到的增广可作"对称缺失"的可解释信号。
- 根源：Bronstein, Bruna, Cohen, Veličković GDL（2021，https://arxiv.org/pdf/2104.13478 ）。

### 负面结果 / 警示

- Wang 2026（https://arxiv.org/abs/2606.03003 ）：精确等变只保证误差在群轨道上**平坦**，不保证**小**（3D 约 0.43 relMSE）；增广、尺度、软等变都恢复不了精确不变性。对 Cell 14 的含义：选对层级不等于预测变好，评价指标要分开"层级判定正确率"和"预测误差"。
- LieFlow：高阶群（二十面体）失败，只测过无噪点云；发现的对称接回下游仍开放。MLSD：只适用可积系统，混合/混沌相空间留作未来。
- Park 等 "Symmetry-Aware Steering of Equivariant Diffusion Policies: Benefits and Limits"（2025，https://arxiv.org/abs/2512.11345 ）：对称破缺时等变 RL 有实际极限。
- HF 索引中有 "On the hardness of learning under symmetries"（arXiv 2401.01869，未 fetch）称即使给了对称、用梯度下降学等变网络仍计算困难——若要引用需再核。

### 还剩什么没做

- 没有人把 G_Euclid ⊂ G_affine ⊂ G_projective ⊂ Homeo 当作嵌套假设做模型选择（Otto 的"promote"正则或 van der Ouderaa 的边际似然都是现成工具，但都没用在动力学层级上）。
- 现有发现方法几乎都找潜空间里的线性 Lie 群；射影群 PGL 在齐次坐标上也是线性的，技术上可接 LaLiGAN/LatentLieODE，但没人做。
- 没有"世界模型输出层级不变量"（交比、平行性、距离、拓扑不变量的残差谱），也没有带层级真值的 benchmark（PDI-Bench 只有射影残差，见 U(b)）。

### 可能的差异化点

- 建"同一场景四种观测映射（欧氏/仿射/射影/同胚）"的阶梯 benchmark，层级真值已知，评估"判定正确率"而非只看预测误差（回应 Wang 2026 的警示）。
- 输出不是群而是**层级残差谱**（cross-ratio / parallelism / distance / Betti），这与所有对称性发现工作的输出形态都不同。
- 用 Sergeant-Perthuis 的发现反向验证：学到的层级能否预测智能体的探索行为（欧氏→不动、射影→靠近）。

### 覆盖信心：高（对称性发现与群结构世界模型两条线）/ 中（Klein 层级专题：相关工作散在认知科学与视觉几何，只抓到 eLife 一篇和一篇玩具模型）

### 搜索词
WebSearch："symmetry discovery world model latent dynamics Lie group learned transformations 2025"；""generative adversarial symmetry discovery" LieGAN Yang 2023 Lie algebra discovery dynamics"；"Erlangen program world model group hierarchy affine projective invariance learned dynamics neural"；"automatic symmetry discovery Lie group neural networks 2025 2026 arXiv "symmetry discovery" latent dynamics"；""symmetry discovery" OR "learned symmetry" "world model" reinforcement learning discovered group equivariant transition 2025 2026"；"learning Lie group transformations from interaction agent actions latent dynamics "group structured" OR "homomorphism" world model 2025"；"projective geometry OR affine geometry neural world model invariance hierarchy "which geometry" learned representation camera"；"Dehmamy 2021 "Automatic symmetry discovery with Lie algebra convolutional network" L-conv follow-up"；""symmetry discovery" dynamics "Lie group" learned world model planning agent 2026 arXiv discovered symmetry used as inductive bias transition"；"Noether networks learning conserved quantities symmetry from video dynamics Alet 2021 successor 2024 2025"；""symmetry discovery" limitations "spurious symmetries" OR "approximate symmetry" OR "symmetry breaking" learned dynamics negative result 2024 2025"；""Klein" "Erlangen" invariants hierarchy machine learning "topological" "projective" "affine" "Euclidean" representation learning transformation group level"；"learning symmetry group of MDP from data discovered equivariance reinforcement learning "symmetry discovery" 2025 2026"；""Lie group" world model latent dynamics "learned generator" OR "learned Lie algebra" video prediction unsupervised 2025 2026"；""which symmetries" OR "how much symmetry" to impose selection group size learned equivariance relaxed approximate "symmetry level" model selection dynamics"；"Erlangen programme representation learning "geometric" latent "hierarchy of invariants" projective affine topological deep learning paper"；"Koyama "Neural Fourier Transform" group action learning unknown group latent world model "Lie action""；""symmetry discovery" "dynamical systems" review survey 2025 Lie point symmetries neural learned from data"；""world model" "equivariance" discovered OR learned "group" test which invariants environment preserves topological Euclidean 2026"。
HF papers："symmetry discovery world model"；"Lie group symmetry discovery dynamics latent"；"Erlangen program geometry hierarchy invariants learning"；"Erlangen symmetry level selection group hierarchy learned equivariance"；"Influence of the Geometry of the world model on Curiosity Based Exploration"。

---

## Cell 15：Liouville 相体积收缩率当耗散（辛/耗散世界模型）

### 撞车等级：B（"保守/耗散分解"在学习动力学里极密（2022–2026 每年多篇）；"从学到的模型算局部熵产生并画成图"已有（Zhu 等 2025）；"从学到的世界模型 Jacobian 算 Lyapunov 谱"已有（Wang 2026）；但"把潜流散度 div f（相体积变化率）当作世界模型的可解释输出/不变量，并给出'哪里在损失相体积'的地图"没人做，也没有带散度真值的世界模型 benchmark）

### 最近邻列表

1. **Identifiable learning of dissipative dynamics** — Zhu, Soh, Pavliotis, Li；arXiv 2510.24160（2025-10，v2 11 月）。https://arxiv.org/abs/2510.24160 （html: https://arxiv.org/html/2510.24160v2 ）
   学到的漂移 f = −(M+W)∇V + ∇·M + ∇·W（V = −log ρ 唯一到常数；M = σσᵀ/2 对称正定给耗散结构；W 反对称带状）；可逆部分 f_rev = −M∇V + ∇·M，不可逆部分 f_irr = −W∇V + ∇·W；**局部熵产生率 ṡ = f_irrᵀM⁻¹f_irr 作为状态的函数画成沿轨迹的空间图**（Fig. 4c–f），全局 EPR 为其对平稳密度的积分；方法里给出系统 EPR = ∇·f_irr = −∇·(W∇V)，并指出 f_irr 的流保持平稳密度。应用：拉伸流中的聚合物、SGLD。
   **接近程度：最高。** 它已经有"从学到的模型输出一张耗散地图"，而且用到了散度公式。差别：对象是随机 Langevin 型系统、量是热力学熵产生（相对于平稳密度 ρ 的不可逆流），不是确定性潜流的相体积散度；不是世界模型/RL；且需要 σ（扩散）来定义 M。

2. **Certified World Models: Predictability Across Configuration, Horizon, and Resolution** — Hongbo Wang；arXiv 2606.13092（2026-06，v3 07）。https://arxiv.org/abs/2606.13092
   对等变潜世界模型给可计算的"可预测性证书"：近似对称缺陷按系统的有限时 Lyapunov 谱增长；膨胀方向对数限制视界、中性方向线性累积、**收缩方向在有界非零误差底停住**；40 维学到的模型上对称约束网络恢复整个 Lyapunov 谱（R² 0.98–0.99）而稠密/循环基线失败；用模型 Jacobian 给出视界估计。
   接近程度：高。Lyapunov 谱之和就是平均散度（相体积收缩率），所以它离 Cell 15 一步之遥，但它没有把"和"单独拿出来当耗散量，也不做空间地图。同类诊断：Park, Yang, Chandramoorthy "When are dynamical systems learned from time series data statistically accurate?"（2024/25，https://arxiv.org/abs/2411.06311 ，遍历论意义的泛化，训练加 Jacobian 信息提高统计准确性）；Ko, Giraldo, Lee "Watch your neighbors"（2026，https://arxiv.org/abs/2605.14405 ，用相空间局部 patch 在动力学下的膨胀/收缩做 MMD 训练）；Wöhrer, Kuehn "Tracking FTLEs to Robustify Neural ODEs"（2026，https://arxiv.org/abs/2602.09613 ）。

3. **Dissipative Hamiltonian Neural Networks: Learning Dissipative and Conservative Dynamics Separately (D-HNN)** — Greydanus（页面只列此一位作者；arXiv 版为 Sosanya & Greydanus）；ICLR 2022 workshop "AI for Earth and Space Science"。https://www.iclr.cc/virtual/2022/7665 （arXiv: https://arxiv.org/abs/2201.10085 ；博客 https://greydanus.github.io/2022/01/25/dissipative-hnns ）
   同时参数化 Hamiltonian 和 Rayleigh 耗散函数，构成隐式 Helmholtz 分解，把摩擦等耗散效应与守恒动力学分开；阻尼弹簧 + 两个真实数据集（含大洋流数据）。
   接近程度：高（根源；Cell 15 给出的起点）。

4. **HHD-GP: Incorporating Helmholtz-Hodge Decomposition into Gaussian Processes for Learning Dynamical Systems** — Xu, Pan；NeurIPS 2024。https://nips.cc/virtual/2024/poster/95052
   GP 联合学向量场的无旋分量与**无散分量**，用对称先验提高预测并使分解可辨识、与系统的科学性质对应；噪声稀疏数据下得到物理可解释分解。
   接近程度：高。无散分量正是 Liouville 意义上"不改变相体积"的部分，所以 HHD 分解已经隐含了 Cell 15 的量，但它是 GP、非潜空间、没把散度当输出。近亲：Smith, Egeland（IFAC 2025，https://arxiv.org/abs/2410.18656 ，辛核 + 无旋核的 RKHS/RFF 版）。

5. **GENERIC-FNO: Embedding Energy Conservation and Entropy Production into Fourier Neural Operators** — Sulskis, Ravi；arXiv 2606.08343（2026-06，v8 10 月）。https://arxiv.org/abs/2606.08343 （html: https://arxiv.org/html/2606.08343v1 ）
   在函数空间里按结构强制 metriplectic（GENERIC）的两个退化条件（不罚、不投影），能量/熵作为神经算子学出；守恒与熵产生到 1e-13；**明确承认能量/熵分解不唯一（gauge），于是另给一个规范不变的耗散诊断 r_mech = −⟨u,∂ₜu⟩/(‖u‖‖∂ₜu‖)**（每个状态一个标量，不是空间图）；全文不含 Liouville / 相体积 / 散度。
   接近程度：中高。它的 gauge 问题是 Cell 15 必须回答的：**潜空间散度在潜空间重参数化下不是不变量**，除非指定体积形式。同线：Lee, Trask, Stinis "Machine learning structure preserving brackets for forecasting irreversible processes"（2021，https://arxiv.org/abs/2106.12619 ）；Baheri, Lindemann "Metriplectic Conditional Flow Matching for Dissipative Dynamics"（2025/26，https://arxiv.org/abs/2509.19526 ，保守/耗散分开进向量场与采样器，Strang-prox 保证离散能量单调）。

6. **世界模型侧的 Hamiltonian 结构（2026 密集出现，均未碰相体积）**：
   - Cui, Ma "Physically Native World Models: A Hamiltonian Perspective on Generative World Modeling"（2026-05，https://arxiv.org/abs/2605.00412 ）：立场文，潜相空间 + Hamiltonian 式演化 + 耗散项 + 残差项，无相体积/散度。
   - Tang 等 **HaM-World**（2026-05，v2 09，https://arxiv.org/abs/2605.05951 ）：潜态分成 (q,p) 子空间（Hamiltonian 式 + 残差 + 控制）和 context 子空间（语义与耗散因子），Mamba 选择性记忆；六个 DMC 任务；**自报：长开环视界落后，且记忆部件贡献大于 Hamiltonian 几何**。
   - Tang 等 **DreamSAC**（CVPR 2026，https://arxiv.org/abs/2603.07545 ；https://en.papernotes.org/CVPR2026/reinforcement_learning/dreamsac_learning_hamiltonian_world_models_via_symmetry_exploration/ ）：学 H_φ(z)（Lie Transformer 做 SE(3) 不变）+ 辛积分器 + 以 |ΔH| 为好奇心奖励；无耗散项，自报在高耗散系统上可能失效；不测相体积。
   - Xiao, Ghaffari **LaWM**（2026-05，https://arxiv.org/abs/2605.08279 ）：潜空间离散 Lagrangian + 变分积分器；无耗散、无散度。

补充近邻（简要）：
- 体积保持架构（按设计 div = 0，不做诊断）：Zhu, Zhu, Zhang, Tang, Liu **VPNets**（2022，https://arxiv.org/abs/2204.13843 ）；Bajārs "Locally-symplectic neural networks for learning volume-preserving dynamics"（2021，https://arxiv.org/abs/2109.09151v2 ，仅搜索结果题名，未 fetch）。
- 控制论意义的耗散性（storage/supply，不是相体积）：Okamoto, Kojima "Learning Deep Dissipative Dynamics"（AAAI 2025，https://arxiv.org/abs/2408.11479 ；https://ojs.aaai.org/index.php/AAAI/article/view/34175 ，非线性 KYP 引理的可微投影）；Xu, Sivaranjani（IEEE L-CSS，https://arxiv.org/abs/2309.16032 ）。
- port-Hamiltonian 2025–26：Cherifi 等 **ISO-pHNN**（https://arxiv.org/abs/2501.06118 ，J、R、H 分别学）；Di Persio, Ehrhardt, Outaleb 随机 pH-NN（2026，https://arxiv.org/abs/2603.10078 ，被动性 + 万有逼近）；Friedl, Jaquier, Liao, Kragic **RO-HNN**（2025/26，https://arxiv.org/abs/2509.24627 ，辛自编码器到低维辛子流形，无耗散）；Courtès 等非正则 Hamiltonian（2025/26，https://arxiv.org/abs/2510.01788 ）。
- Martini, Abdelaleem, Gulati, Nemenman **DySIB**（2026，https://arxiv.org/abs/2604.24662 ）：信息瓶颈从单摆实验视频学出与相空间维数、拓扑、几何一致的二维表示——不是耗散工作，但是"从视频学相空间"的可用测试床。
- 另注：连续归一化流（FFJORD 一类）的瞬时变量变换公式 d log p/dt = −div f 就是 Liouville 方程，Hutchinson 迹估计是现成的散度估计器；该机制一直用于密度而非用于解释动力学模型的耗散（这是我的背景知识，本次未专门检索）。

### 负面结果 / 警示

- **Gruver, Finzi, Stanton, Wilson "Deconstructing the Inductive Biases of Hamiltonian Neural Networks"**（ICLR 2022 spotlight，https://arxiv.org/abs/2202.04836v1 ；https://iclr.cc/virtual/2022/spotlight/6096 ）：HNN 的优势主要来自直接建模加速度（二阶结构）和坐标简化，**不是辛结构或能量守恒**；放松这些偏置后在非守恒/接触系统上大幅改进，并用于 MuJoCo 转移模型。对 Cell 15 的含义：若主张"辛/相体积结构带来世界模型收益"，必须控制二阶结构这一混淆变量。
- Shinde 等 **Deepmechanics**（2026，https://arxiv.org/abs/2602.18060 ）：HNN/LNN/SRNN 在六个系统（含弹跳球这种接触、非守恒系统）上都难以保持全轨迹稳定。
- HaM-World 自报：记忆比 Hamiltonian 几何更重要、长视界落后——"物理结构 ≠ 性能"的又一例。
- GENERIC-FNO 的 gauge 非唯一性：任何基于学到的能量/熵函数定义的"耗散地图"都依赖规范；Cell 15 的 div f 同样依赖潜空间的体积形式（潜微分同胚 φ 会改变 div），这是理论上必须先堵的洞。
- 原理层限制：相体积散度只刻画"体积"意义的耗散；Zhu 等 2025 的 EPR 是相对于平稳密度的不可逆流，两者在随机系统里不重合——需要说清楚 Cell 15 测的是哪一种"耗散"。

### 还剩什么没做

- 没有工作把学到的潜世界模型流的 div_μ f(z)（带明确体积形式 μ，如解码器拉回的数据空间体积或平稳密度）当作输出，也没有验证它对潜重参数化的不变性。
- 没有带散度真值的世界模型 benchmark：线性阻尼（每个阻尼自由度贡献 −γ）、Lorenz（常数 −(σ+1+b)）、Van der Pol（状态依赖 μ(1−x²)）、含接触的弹跳球——这些真值是解析的，却没人拿来考世界模型。
- Wang 2026 有 Lyapunov 谱但没检验"谱之和 = 散度"的一致性；也没人把耗散地图用于 RL（例如当作接触/摩擦区域探测或探索信号）。

### 可能的差异化点

- 输出标量场 div_μ f(z)，三种估计器（autodiff 迹、Hutchinson 迹、Lyapunov 谱之和）互相校验，并在已知常散度系统上对真值；同时报告对潜重参数化的鲁棒性（直接回应 GENERIC-FNO 的 gauge 警示）。
- 与 Zhu 等 2025 的 EPR 地图做对照实验（确定性散度 vs 随机熵产生），把两种"耗散"的关系说清楚，这是现有文献的空白。
- 用 Gruver 式消融（二阶结构 vs 辛结构 vs 散度正则）来隔离收益来源，避免 HNN 一线已被指出的归因错误。

### 覆盖信心：中高（保守/耗散学习线很密，已覆盖主要家族；但 WebSearch 预算在本节用尽，"entropy production + world model"、"divergence-free latent world model"两组词只在 HF 索引上搜过，可能漏掉统计物理侧 2025–26 的新作）

### 搜索词
WebSearch："phase space volume contraction learned dynamics dissipative Hamiltonian neural network 2025"；"Hamiltonian neural network survey 2025 dissipative port-Hamiltonian symplectic learning review"；""Liouville" divergence neural ODE interpretability phase space volume learned vector field dissipation"；"symplectic world model reinforcement learning Hamiltonian latent dynamics 2025 2026"；""dissipation" map learned latent dynamics Helmholtz decomposition conservative dissipative vector field neural network 2024 2025"；"Sosanya Greydanus "Dissipative Hamiltonian Neural Networks" follow-up citing 2024 2025 learning dissipation from data"；""volume-preserving" neural network dynamics learning "divergence-free" vector field conservative "phase space volume" interpretable 2024 2025"；"Lyapunov exponents learned latent dynamics world model Jacobian trace contraction rate estimation from neural model 2025"；""trace of the Jacobian" OR "divergence of the vector field" learned dynamics dissipation measure neural ODE "energy dissipation" physical interpretability 2024"；"Hamiltonian neural network negative result "does not" generalize dissipative failure "symplectic" world model limitations contact friction"；"GENERIC formalism neural network metriplectic GFINN GNODE learning dissipative dynamics entropy production 2024 2025"；"Helmholtz-Hodge decomposition neural vector field conservative dissipative "latent" dynamics learned decomposition interpretable 2024 2025 2026"；""phase space" "volume" contraction OR expansion latent dynamics "world model" OR "video prediction" dissipative learned divergence map 2026"；""port-Hamiltonian" neural network 2025 2026 learning from data robot dissipation structure identification"；"neural ODE "divergence" regularization "phase space" conservative dissipative classification learned dynamics "Liouville" 2023 2024 2025 arXiv"；""dissipative" "world model" learned latent dynamics energy decay friction contact reinforcement learning Hamiltonian 2025 2026 arXiv"；""Lyapunov" OR "contraction" diagnostic learned world model latent rollout stability Jacobian spectrum 2026 arXiv"；"Greydanus "dissipative" Hamiltonian "Rayleigh" dissipation function neural network learned damping term interpretable 2023 2024"。（以下两条因预算用尽未执行："divergence-free OR volume preserving latent dynamics neural ODE conservative world model video learned invariant volume 2025 2026"；""entropy production" learned latent dynamics neural network world model irreversibility map interpretable 2025 2026 arXiv"。）
HF papers："phase space volume Liouville neural dynamics dissipation"；"Liouville theorem volume preserving neural network learned dynamics"；"divergence vector field dissipative conservative decomposition learned latent dynamics"；"entropy production irreversibility learned dynamics neural network interpretable"；"Hamiltonian world model"。

---

## 总体项 U：伞状检查与 benchmark 地形

（WebSearch 预算用尽，本节全部靠 HF papers 索引 + arXiv 直接 fetch；HF 索引不是全量 arXiv，结论带此保留。）

### U(a) 有没有人提出"几何 / 信息几何世界模型"伞状纲领

**结论：没有找到任何伞状纲领或综述把"把某种几何结构放进世界模型某个部位并给一个可测不变量"当作系统性研究计划。但"geometric world model"这个名字已被占用，且含义与我们不同，命名有撞车风险。**

- **GeoWorld: Geometric World Models** — Zhang, Li, Reid, Hartley；arXiv 2602.23058（2026-02-26）。https://arxiv.org/abs/2602.23058
  "几何"= 把 V-JEPA 式潜表示映射到双曲流形（Hyperbolic JEPA）以保持层级结构，加 Geometric RL 做多步规划；CrossTask/COIN 上比 V-JEPA 2 高约 2–3% SR。不是纲领，没有不变量。**名字已被占。**
- "geometric world model"在 2025–26 的绝大多数用法指 **3D 几何一致性的视频世界模型**：Aether（2503.18945）、WorldPlay（2512.14614）、Geometry Forcing（2507.07982）、VerseCrafter（2601.05138）、Marionette（2608.14530）、GIM-World（2606.02436）等——与我们的"几何"无关，但搜索引擎会把我们混进去。
- **潜空间几何 → 规划**一线（最接近"几何结构放进世界模型某部位"的局部实践，但各自只做一种几何、无统一纲领）：AnisoWM（Kang 等 2026，https://arxiv.org/abs/2609.37441 ，可学习对角协方差 = 各向异性度量让潜规划代价对齐任务代价）；ATLAS（2609.36333，保持关系几何）；VIScore（2608.11174，规划相关质量诊断）；"Compact but Moving: Intervention-Relevant Geometry in Recurrent World Models"（Chen, Liu 2026，https://arxiv.org/abs/2609.21787 ，干预沿 Jacobian 链传输形成"移动的、状态依赖的局部几何"）；"Which Latent Coordinates Drive World-Model Dynamics?"（Subramanian，NeurIPS 2026，https://neurips.cc/virtual/2026/166118 ，共轭变换检验哪些坐标真被转移用到）；"Spectral Identifiability for World Models"（Mai 等，NeurIPS 2026，https://neurips.cc/virtual/2026/poster/149561 ，线性世界模型的 Krylov 瓶颈与谱记忆能量）；"Neural Fields as World Models"（Nunley 2026，https://arxiv.org/abs/2602.18690 ，保持感官拓扑的"同构世界模型"）；"When Does LeJEPA Learn a World Model?"（Klindt, LeCun, Balestriero 2026，https://arxiv.org/abs/2605.26379 ，线性/正交可辨识性）。
- **"information geometric world model" / "world model information geometry"：零命中**。信息几何在 HF 索引里只出现在扩散模型潜几何（"The Spacetime of Diffusion Models: An Information Geometry Perspective"，2505.17517）、Neural FIM（2306.06062）、IG-NGC（见 Cell 13）等，没有一篇把 Fisher 度量/IGCI 放进世界模型。
- **部分伞**：Cui & Ma 2026 的 Hamiltonian 世界模型立场文（物理结构伞）；Kori & Russo 2026 的因果世界模型统一视角（因果伞）；Thorpe 等 "Physically Viable World Models: A Case for Query-Conditioned Embodied AI"（2026，https://arxiv.org/abs/2605.30542 ，模块化、"最简单但保留查询相关区分的抽象"）。三者都没有"可测不变量"这一层。
- 综述/立场（均无几何纲领）："Understanding World or Predicting Future? A Comprehensive Survey of World Models"（2411.14499）、"Critiques of World Models"（2507.05169）、"A Mechanistic View on Video Generation as World Models: State and Dynamics"（2601.17067）、"Research on World Models Is Not Merely Injecting World Knowledge…"（2602.01630）、OpenWorldLib（2604.04707）、"From Generative Engines to Actionable Simulators"（2601.15533）。
- 早期孤例：Sergeant-Perthuis 等 2023（见 Cell 14）是唯一一篇直接问"世界模型的几何（群）是什么会改变什么"的工作，但是玩具主动推理模型。

**建议**：避免用"geometric world model"做项目名（GeoWorld + 3D 一线双重撞车）；"information-geometric world models"或"invariant-structured world models"目前都是空的。

### U(b) 带精确答案的诊断型动力学 benchmark

**结论：零散存在、各管一种不变量，没有一个同时提供"对称群层级 + 守恒/耗散真值（散度）+ 带机制独立性的因果图"的合成世界集合。这个空位本身可以是项目的第一个产出。**

- **Chen 等 "Automated discovery of fundamental variables hidden in experimental data"** — Chen, Huang, Raghupathi, Chandratreya, Du, Lipson；Nature Computational Science 2, 433–442（2022）。https://www.nature.com/articles/s43588-022-00281-6 ；代码与数据 https://github.com/BoyuanChen/neural-state-variables （Google Drive 分发）。README 列 9 个系统：circular motion、reaction diffusion、single pendulum、rigid double pendulum、elastic double pendulum、swing stick（non-magnetic）、air dancer、lava lamp、fire。**README 不给真值内在维数**（只给估计脚本）；前几个系统的维数可由物理直接写出（单摆 2、刚性双摆 4、弹性双摆 6 等），air dancer / lava lamp / fire 无真值只有论文估计。适合 Cell 14/15 做"相空间维数已知"的视频测试床，但没有对称层级或散度标注。
- **CausalDynamics** — Herdeanu, Nathaniel, Roesch, Buch, Ramien, Haux, Gentine；arXiv 2505.16620（2025，v2 10 月）。https://arxiv.org/abs/2505.16620 。数千个线性/非线性耦合 ODE/SDE + 两个理想化气候模型，**真值因果图来自方程**，含噪声、混杂、滞后；评估因果发现算法。直接可用于 Cell 13 的定向评估。
- **IRIS** — Khanbayov, Barhdadi, Serpedin, Kurban；arXiv 2603.16432（2026）。https://arxiv.org/abs/2603.16432 。240 段 4K/60fps 实验室实拍视频，单体与多体，**独立测量的物理参数（带不确定度）+ 配对的控制方程**；任务含可辨识性、外推、方程选择。
- **Morpheus / "Evaluating Newtonian Mechanics in Video Generative Models with Real Physical Systems"** — Tragoudaras 等；arXiv 2504.02918（2025，v3 2026-06）。https://arxiv.org/abs/2504.02918 。130 段真实物理视频，按每个场景已知的守恒律设计可解释指标。
- **WorldBench**（Upadhyay 等 2026，https://arxiv.org/abs/2601.21282 ，每个测试只隔离一条物理概念：物体恒存、尺度/透视、摩擦系数、粘度）；**World Models' Last Exam in Physics**（Gao 等 2026-10，https://arxiv.org/abs/2610.08791 ，40 个测量型任务、六个物理域、最好模型 57.76/100）；PhyWorldBench（2507.13428）、PhyGround（2605.10806）、WorldReasonBench（2605.10434）——都是视频生成模型的物理评测，答案是"是否符合物理"而非精确不变量。
- **PDI-Bench** — Wu 等 2026，https://arxiv.org/abs/2605.15185 ：用单目重建 + 点追踪算**射影几何残差**（尺度-深度对齐、3D 运动一致性、刚性）。是 Cell 14 里"射影层级残差"的现成雏形（但只有这一级）。
- **诊断协议类**：Subramanian 2026 "Counterfactual Action Evaluation, Observation Bottlenecks, and Representation Geometry in JEPA World Models"（https://arxiv.org/abs/2610.02860 ，精确模拟器分叉追踪同一干预；发现低误差 ≠ 动作感知、谱集中）；Sandilya, Liston, Grosenick 2026 "Identifiable World Models from Pretrained Diffusion Representations"（https://arxiv.org/abs/2610.07028 ，落体系统 + 双足机器人，真值滞后因果图）；Thorpe 等 2026（固定可见场景、变潜在物理的受控 benchmark）；stable-worldmodel（Maes 等 2026，https://arxiv.org/abs/2602.08968 ，可控视觉/物理因子的标准化环境）；Deepmechanics（Shinde 等 2026，六个经典力学系统含弹跳球）；DySIB 的单摆视频（2604.24662）。
- 守恒律发现：ConCerNet（2302.05783，HF 索引，未 fetch）。

**空位**：没有 benchmark 给每个合成世界同时标注 (i) 保持到哪一级 Klein 不变量、(ii) 解析散度场 / 守恒量、(iii) 带机制独立性结构的因果图。三条 Cell 各自需要的真值都能解析生成，合在一起就是"invariant-labelled synthetic worlds"。

### U(c) 清华 AWM 之后的跟进

- **Algebraic World Model: From Exploration to Proofs** — Pan, Tan, Yuan, Yao（Andrew Chi-Chih Yao）；arXiv 2610.40503，提交 2026-10-29（审计日前 3 天）。https://arxiv.org/abs/2610.40503
  内容（摘要页）：与陌生环境交互 → 写可执行 Python 假设并对观测测试修订 → 归成带类型的对象与共享规则 → 范畴论引导的化简 → Lean 4 形式化证明可达性/不可能性；ARC-AGI-3 全部 25 个公开游戏 183 关，Lean 可回放全部制胜轨迹并预测未见转移；两个不可能关卡上 6/6 给出核验证的 Lean 证明，基线（带代码工具的 Opus 5.5、GPT-6 Astra）12 次 0 次。**不涉及几何、信息几何、对称群**；辅助文件名里有 "ColorsConserved.lean"，暗示有守恒量式的不变量，但页面未说明。
- **跟进/评论：没有找到。** HF papers 索引尚未收录该文（cat 返回 not found），WebSearch 预算已尽无法做专门查询。3 天内有跟进的概率本来就低。
- **与本项目的关系**：正交而非撞车。AWM 是离散/符号世界的代数不变量（类型、规则、可证的不可达性），我们是连续动力学的几何不变量（群层级、散度、因果正交）。可借用的定位语：AWM 证明"世界模型可以输出可证的不变量（不可能性）"，我们的纲领是其连续几何对应物；AWM 的"ColorsConserved"类守恒量与 Cell 15/Noether 式守恒量是天然的对照点。

---

## 总结

| 条目 | 撞车等级 | 最近邻（一句话） | 空位 |
|---|---|---|---|
| Cell 13 因果正交 | **B**（"ICM→模块化世界模型"一半近 A：COMET/VCD/SPARTAN；"IGCI 定向"一半 B/C） | Besserve 等 CLeaR 2022 的 SIC 把"原因⟂机制"做到了线性时间序列；Janzing–Chaves–Schölkopf NJP 2016 给了时间箭头版本 | 非线性学到的潜空间里的正交性定向；每边正交性得分作为输出；正向/反向世界模型的正交性差 |
| Cell 14 Klein 层级 | **B** | LatentLieODE（NeurIPS 2026）联合学 Lie 对称与潜 ODE；Barthélémy 2026 从交互发现动作群分解；Sergeant-Perthuis 2023 比较欧氏 vs 射影世界模型（群人选） | 嵌套层级的模型选择；射影级发现；层级残差谱作为输出；带层级真值的阶梯 benchmark |
| Cell 15 Liouville 散度 | **B** | Zhu 等 2025 从学到的模型画局部熵产生地图；Wang 2026 从世界模型 Jacobian 恢复 Lyapunov 谱；HHD-GP/D-HNN/metriplectic 一线做保守/耗散分解 | 带体积形式的 div_μ f 作为世界模型输出 + 对潜重参数化的不变性；常散度系统真值 benchmark；散度 vs EPR 对照 |
| U(a) 伞状纲领 | 无人提出 | GeoWorld（双曲 JEPA）占了"geometric world model"名字；信息几何 + 世界模型零命中 | 命名避开 "geometric world model" |
| U(b) benchmark | 零散 | CausalDynamics（因果图真值）、Chen 2022 九系统视频、IRIS（参数+方程真值）、PDI-Bench（射影残差）、Morpheus（守恒律指标） | 同时标注层级 / 散度 / 因果结构的合成世界集 |
| U(c) AWM | 无跟进（文太新） | AWM = 离散世界的代数不变量 + Lean 证明 | 可作为"连续几何对应物"的定位参照 |

三条共同的最大风险不是"已有人做了"，而是**归因与不变性**：Gruver 2022（HNN 收益不来自辛结构）、Wang 2026（等变只保证平坦不保证低误差）、GENERIC-FNO（分解 gauge 不唯一）、HaM-World（记忆比 Hamiltonian 几何重要）都在说同一件事——物理/几何结构放进世界模型后，收益归因需要专门的消融，不变量必须对潜空间重参数化稳健。三条 Cell 若都以"可测不变量 + 解析真值 benchmark + 归因消融"为交付形式，与现有工作的差异化是清楚的。

未能读取的页面：arXiv 2304.00188 的元数据页（限速；正文已核对）；export.arxiv.org（robots 禁止）。未核实的引用已在文中逐条标注。

