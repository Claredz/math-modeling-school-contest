# 数模校赛

数学建模校赛项目工作区，已配置项目级 `mathmodel-skill` v6.0。

## 使用方法

在本目录中启动 Codex，然后直接说：

> 使用 `$mathmodel-skill` 开始建模

工作流支持 CUMCM 国赛、MCM/ICM 美赛和电工杯，包含选题、建模、求解、稳健性分析、论文写作与终稿审核。

## 校赛 B 题

- 题目：[B题：舰船烟幕遮蔽干扰优化.docx](B题：舰船烟幕遮蔽干扰优化.docx)
- 模型文档索引：[docs/modeling/README.md](docs/modeling/README.md)
- 当前假设：[assumption_register.md](docs/modeling/assumption_register.md)（v0.3，A-001–A-022）
- 当前模型契约：[model_contract_v0.1.md](docs/modeling/model_contract_v0.1.md)（正文 v0.2，文件名为兼容旧链接保留）
- 当前四问架构：[revised_four_problem_architecture.md](docs/modeling/revised_four_problem_architecture.md)
- 当前修订设计：[2026-07-30-inertial-pursuit-shipborne-launch-design.md](docs/plans/2026-07-30-inertial-pursuit-shipborne-launch-design.md)
- **当前唯一实施计划**：[2026-07-30-inertial-pursuit-shipborne-launch-implementation-plan.md](docs/plans/2026-07-30-inertial-pursuit-shipborne-launch-implementation-plan.md)

当前正式模型采用一阶航向惯性纯追踪，9 组
\((k,\omega_{\max})\) 参数扫描；所有 UAV 可在舰上等待，但只能从实际起飞时刻的舰船位置起飞。瞬时纯追踪只作消融对照。

下列计划均已过时并在文件头明确标注，禁止继续执行：

- `2026-07-29-b-problem-modeling-design.md`
- `2026-07-29-b-problem-execution-plan.md`
- `2026-07-30-b-problem-implementation-design.md`
- `2026-07-30-b-problem-implementation-plan.md`

当前仓库仍处于“模型与计划完成、求解代码尚未实现”阶段。题面未给出的初始坐标、Q4来袭批次和设备误差必须作为带来源的场景假设，禁止冒充官方数据。

## 目录结构

- `.agents/skills/mathmodel-skill/`：项目级数学建模技能
- `state/`：工作流状态
- `results/`：模型结果
- `figures/`：论文图表
- `paper_workspace/`：论文工作区

技能来源：[handsomeZR-netizen/mathmodel-skill](https://github.com/handsomeZR-netizen/mathmodel-skill)，MIT License。
