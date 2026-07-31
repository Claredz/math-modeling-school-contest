# Stage 7 评估与论文素材

统一验收包括 pytest、Ruff、场景 schema、场景生成、synthetic toy artifact 和 `git diff --check`。本轮 review 还增加 Q1 Sobol warm-start、Q2 区间分类/联合增益和 Q4 认证过滤回归测试。Q1–Q4 结果分别隔离在 `results/q1_rebuild/`、`results/q2_rebuild/`、`results/q3_rebuild/`、`results/q4_rebuild/`，图分别位于对应 `figures/` 子目录。

论文素材中已同步：Q1 算法比较表与固定候选失锁图、Q2 联合覆盖/裸露区间与基线增益、Q3 三机联合指标、Q4 资源工况/rolling–greedy–hindsight 图与 CSV 表，以及 `figures/model_workflow.svg` 总流程图。所有结果均注明当前候选预算、认证状态、未决区间和未宣称全局最优的边界；本轮只更新 Stage 7 素材，不开始 Stage 8 论文写作。
