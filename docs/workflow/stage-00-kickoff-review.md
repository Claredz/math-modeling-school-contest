# Stage 0：项目重启与现状审计

## 1. 研究检查点

- 竞赛映射：按 CUMCM 三人制、中文、XeLaTeX 工作流执行，但题目是校赛 B 题，
  不能套用通用 CUMCM B 题“评价类”标签。
- 题目：`B题：舰船烟幕遮蔽干扰优化.docx`，共四问。
- 截止时间：2026-08-01 19:00（北京时间）。
- 本轮边界：完成 Stage 0–3 与 Gate B 后停止。
- 云端基线：`main@44fd43b83df2c91bf14f40f9a1cb73882694753f`。
- 研究分支：`agent/full-modeling-workflow-rebuild`，从上述云端基线直接创建。
- Draft PR：#6“按数学建模工作流重构 B 题模型与算法选型”。

## 2. 仓库与 PR 现状

### 2.1 已存在的工程资产

`main` 包含常数单一事实源、Pydantic 场景、场景哈希、动力学、事件时间线、
覆盖与连续验证、Q1 候选与结果、83 个基线测试、ruff 和场景生成检查。

这些只证明“已有实现内部一致并能运行”，不证明物理模型正确，也不证明 Q1
全局最优。

### 2.2 PR #5

PR #5 `agent/q2-multi-smoke` 已关闭且未合并。废弃原因已经写入 PR：

- 缺少前置文献检索和正式算法选型；
- Q2 候选空间过窄；
- 连续精修未真正实施；
- unresolved 区间的连续风险统计不完整；
- 原 Q2 数值结果与确定性收益表述不得继续使用。

该 PR 仅可用于根因审计，禁止 merge、rebase、cherry-pick 或复制正式结果。

## 3. 已知错误与待审查资产

### 3.1 高严重度

1. `state/decision_log.json` 原先不存在，旧文档中的“冻结模型”没有真实工作流状态支撑。
2. Q1 计划声称包含连续精修，但是否实际调用连续优化器尚未审计。
3. 通用 `topic_specs.json` 把 B 题映射为评价类，与本题优化、仿真、计算几何和
   调度性质不一致。
4. Q2 的旧候选枚举不代表连续原始空间。

### 3.2 中严重度

- 一阶惯性纯追踪及 9 组参数只有情景假设地位；
- 固定烟幕中心、UAV 瞬时转向和 12 km 解释需要重新选型；
- Q1 数值结果只能在完成模型、算法和最优性声明审计后决定保留或降级。

## 4. 团队角色

| 角色 | 主责 | 互备 | 最大顾虑 |
|---|---|---|---|
| A 建模 | 题意、假设、数学结构、结果验收 | 算法测试 | 代码结构反向固化模型 |
| B 论文 | 符号、图表、引用、论点证据 | 文献矩阵 | 把有限候选结果写成全局最优 |
| C 仿真 | toy demo、测试、复现、结果格式 | 模型反例 | 离散采样掩盖连续未决区间 |

## 5. 问题预扫

```json
{
  "problem_id": "2026-school-contest-B",
  "domain_keywords": [
    "几何覆盖",
    "混合事件系统",
    "连续时间认证",
    "非凸优化",
    "组合调度"
  ],
  "data_attachments": [
    "B题：舰船烟幕遮蔽干扰优化.docx"
  ],
  "subproblem_count": 4,
  "primary_problem_type": "optimization",
  "secondary_types": [
    "simulation",
    "computational_geometry",
    "scheduling"
  ],
  "estimated_difficulty": "high",
  "data_size_signal": "无数值附件，缺失初始场景"
}
```

## 6. 工具检查

Fresh 检查已确认：

- Python 3.11.9、NumPy 2.2.6、CVXPY 1.9.2；
- CVXPY solver：CLARABEL、SCS、GLPK、GLPK_MI、CVXOPT、SCIPY、
  HIGHS、OSQP；
- Git 2.51.2、GitHub CLI 2.88.1；
- XeTeX/TeX Live 2025；
- 核心建模、敏感性、文档和多目标依赖导入成功；
- `python -m pytest -q`：83 passed；
- ruff：passed；
- scenario schema：current；
- scenario matrix：144 formal + 16 ablation；
- `git diff --check`：通过。

环境仍存在两个与本项目运行路径无关的 `pip check` 冲突：
`doc2docx/pywin32` 和 `weasyprint/tinycss2`。本轮不调用这两个包，故不阻塞
Stage 0，但在任何 Word 转换或 WeasyPrint 渲染前必须单独修复并复测。

## 7. 时间盒

从 2026-07-31 00:21 估算剩余约 42.7 h：

| 工作 | 配额 |
|---|---:|
| Stage 0 | 1 h |
| Stage 1 | 1 h |
| Stage 2 | 4 h |
| 文献与证据映射 | 5 h |
| Q1 审计 | 4 h |
| Stage 3 与 toy demos | 14 h |
| Gate B review/PR | 4 h |
| buffer | 9 h |

## 8. 协作与版本约定

- Python 使用 `snake_case`，量纲进入字段名；文档使用 `kebab-case`。
- 每个 Stage 独立提交并更新 Draft PR。
- 多 Agent 只处理独立研究或审查，主 Agent 统一核验后写入状态。
- 高严重度问题必须停在当前 Gate，不用后续实现掩盖。
- toy demo 只能进入 `experiments/toy_demos/`，不得写入正式 `results/`。

## 9. Gate 设计

- Stage 0：工具、角色、题面预扫和协作协议真实完成。
- Stage 1：B 题适配和题型修正有证据。
- Stage 2：题面事实、歧义、原始空间和依赖不由代码反推。
- Gate A：Stage 0–2、文献、Q1 初步审计完整。
- Stage 3：每个关键模块至少三个候选，toy demo 真实运行。
- Gate B：Stage 3 L1/red-team 通过后立即停止。

## 10. Stage 0 当前 verdict

`pass_early`。五维评分均为 9：

| 维度 | 分数 | 证据 |
|---|---:|---|
| 角色分工 | 9 | 三主责、互备和最大顾虑明确 |
| 工具就绪 | 9 | 完整导入、solver 与 fresh 基线通过 |
| 时间规划 | 9 | Gate B 时间盒与 9 h buffer |
| 题目预扫 | 9 | 原题 P1–P22、四问和问题域已识别 |
| 协作协议 | 9 | 命名、提交、同步、升级和多 Agent 边界明确 |

`decision_log.current_stage` 已更新为 1，进入 Stage 1。
