# 数模校赛

数学建模校赛项目工作区，已配置项目级 `mathmodel-skill` v6.0。

## 使用方法

在本目录中启动 Codex，然后直接说：

> 使用 `$mathmodel-skill` 开始建模

工作流支持 CUMCM 国赛、MCM/ICM 美赛和电工杯，包含选题、建模、求解、稳健性分析、论文写作与终稿审核。

## 校赛 B 题

- 题目：[B题：舰船烟幕遮蔽干扰优化.docx](B题：舰船烟幕遮蔽干扰优化.docx)
- 建模设计：[2026-07-29-b-problem-modeling-design.md](docs/plans/2026-07-29-b-problem-modeling-design.md)
- 冻结模型契约：[model_contract_v0.1.md](docs/modeling/model_contract_v0.1.md)
- 实现架构与场景契约：[2026-07-30-b-problem-implementation-design.md](docs/plans/2026-07-30-b-problem-implementation-design.md)
- 当前实施计划：[2026-07-30-b-problem-implementation-plan.md](docs/plans/2026-07-30-b-problem-implementation-plan.md)
- 历史执行计划（已被当前实施计划替代）：[2026-07-29-b-problem-execution-plan.md](docs/plans/2026-07-29-b-problem-execution-plan.md)

当前题面未给出各主体初始坐标和问题四的具体来袭批次，因此执行计划采用参数化场景，并禁止将基准假设误写成官方数据。

## 目录结构

- `.agents/skills/mathmodel-skill/`：项目级数学建模技能
- `state/`：工作流状态
- `results/`：模型结果
- `figures/`：论文图表
- `paper_workspace/`：论文工作区

技能来源：[handsomeZR-netizen/mathmodel-skill](https://github.com/handsomeZR-netizen/mathmodel-skill)，MIT License。
