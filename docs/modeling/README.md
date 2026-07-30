# B题模型文档状态索引

> 更新时间：2026-07-30
> 当前模型：一阶航向惯性纯追踪 + 舰载 UAV 起飞 + 多烟幕联合覆盖 + 连续时间三值认证

## 当前有效文档

按以下顺序阅读：

1. [题面事实清单](problem_facts.md)：只记录题面明确内容；
2. [模型假设登记表](assumption_register.md)：v0.3，A-001–A-022；
3. [统一模型契约](model_contract_v0.1.md)：正文 v0.2，定义 Q1–Q4 共享数学对象；
4. [Q1 解析可行性](q1_analytic_feasibility.md)：惯性模型下仍有效的证书；
5. [四问统一架构](revised_four_problem_architecture.md)：Q1 到 Q4 的求解递进；
6. [高级数学工具评估](advanced_math_options.md)：当前方法选择与后续扩展；
7. [当前修订设计](../plans/2026-07-30-inertial-pursuit-shipborne-launch-design.md)；
8. [当前唯一实施计划](../plans/2026-07-30-inertial-pursuit-shipborne-launch-implementation-plan.md)。

## 当前冻结决策

- 正式导弹模型是一阶航向惯性纯追踪；
- \(k\in\{0.5,1,2\}\rm\,s^{-1}\)；
- \(\omega_{\max}\in\{5^\circ,10^\circ,20^\circ\}/\rm s\)；
- 中值参考为 \(k=1\rm\,s^{-1}\)、
  \(\omega_{\max}=10^\circ/\rm s\)，但不是题面事实；
- 导弹初始航向等于出现时刻的视线角；
- 瞬时纯追踪只作消融；
- 所有 UAV 可在舰上等待，并从实际起飞时刻的舰船位置起飞；
- UAV 自由初态、初始航向和非零发射偏置均禁止；
- Q2–Q4 使用多烟幕空间联合覆盖；
- 只有连续时间验证器可以给出物理可行性三值状态。

## 历史归档

以下内容已过时，只能用于追溯模型演进：

- [旧模型审查报告](model_review_report.md)；
- [2026-07-29 建模设计](../plans/2026-07-29-b-problem-modeling-design.md)；
- [2026-07-29 执行计划](../plans/2026-07-29-b-problem-execution-plan.md)；
- [2026-07-30 旧实现架构](../plans/2026-07-30-b-problem-implementation-design.md)；
- [2026-07-30 旧实施计划](../plans/2026-07-30-b-problem-implementation-plan.md)。

历史文档中的“当前”“基准”“已冻结”均按其生成时语境理解，不具有现行效力。
