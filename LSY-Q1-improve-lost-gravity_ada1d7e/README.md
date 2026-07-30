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

`LSY-Q1-improve-lost&gravity` 分支新增 Q1 失锁耦合模型：烟幕完整遮蔽会切断实时目标信息，导弹按保留转弯惯性飞行，并通过连续可见条件判定重捕获。模型与可行示例见 [q1_lost_coupled_model.md](docs/modeling/q1_lost_coupled_model.md)，运行入口为 `scripts/run_q1_lost.py`。

当前工程状态：

- 公共物理内核已完成：题面常数与场景契约、导弹/舰船动力学、连续根事件与多分量探测集合、舰载 UAV 路径及约束证书、烟幕动力学、空间联合覆盖和连续时间三值验证器；
- Q1 求解、9 组惯性参数交叉验证、瞬时纯追踪消融和可追溯结果已完成；
- Q2–Q4 求解代码尚未实现，本轮验收不启动 Q2。

下列计划均已过时并在文件头明确标注，禁止继续执行：

- `2026-07-29-b-problem-modeling-design.md`
- `2026-07-29-b-problem-execution-plan.md`
- `2026-07-30-b-problem-implementation-design.md`
- `2026-07-30-b-problem-implementation-plan.md`

题面未给出的初始坐标、Q4 来袭批次和设备误差必须作为带来源的场景假设，禁止冒充官方数据。

## 目录结构

- `.agents/skills/mathmodel-skill/`：项目级数学建模技能
- `state/`：工作流状态
- `results/`：模型结果
- `figures/`：论文图表
- `paper_workspace/`：论文工作区

技能来源：[handsomeZR-netizen/mathmodel-skill](https://github.com/handsomeZR-netizen/mathmodel-skill)，MIT License。
