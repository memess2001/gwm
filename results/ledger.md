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
