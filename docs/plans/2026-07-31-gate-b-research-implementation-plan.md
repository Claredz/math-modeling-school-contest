# B 题 Gate B Research Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 完成 mathmodel-skill Stage 0–3、系统文献、Q1 审计和五类 toy demo，在 Gate B 停止。

**Architecture:** 题面、文献、现有资产和 toy demo 构成四条独立证据链，最终汇入 Stage 3 决策矩阵。研究原型与生产包物理隔离，不修改正式 Q1–Q4 实现。

**Tech Stack:** Python 3.11、NumPy/SciPy、CVXPY/PuLP、DEAP、PySwarms、pytest、Markdown/CSV/JSON。

---

### Task 1: 初始化状态与 Stage 0

**Files:**
- Create: `state/decision_log.json`
- Create: `docs/workflow/stage-00-kickoff-review.md`

**Step 1:** 从 skill v3.0 模板初始化状态，写入 cumcm、B 题、四问、三人角色、
截止时间、云端基线 SHA 和本轮 Gate B 边界。

**Step 2:** 运行 Python/依赖、solver、XeLaTeX、git、pytest、ruff、schema 和场景
生成检查，保存精确版本与失败项。

**Step 3:** 写 Stage 0 审计和五维 L1 评分；任何工具缺失先修复再评分。

**Step 4:** 更新 `current_stage=1` 并提交。

Run:
`python -m pytest -q && python -m ruff check . && python scripts/export_scenario_schema.py --check && python scripts/generate_scenarios.py --check`

Expected: 83 tests pass, lint/schema/scenarios exit 0。

### Task 2: Stage 1 选题回顾

**Files:**
- Create: `docs/workflow/stage-01-problem-selection-retrospective.md`
- Modify: `state/decision_log.json`

**Step 1:** 加载 Stage 1 reference、rubric 和 topic specs。

**Step 2:** 说明 B 题相对其他题目的适配、团队能力、失败风险；指出通用
`topic_specs.json` 将 B 映射为 evaluation 与本校题实际 optimization/simulation
不一致，记录项目级 task type 修正依据。

**Step 3:** 完成 L1 评分，更新 `current_stage=2` 并提交。

### Task 3: Stage 2 题面重解析

**Files:**
- Create: `docs/workflow/stage-02-problem-analysis.md`
- Draft modify: `docs/modeling/problem_facts.md`
- Draft modify: `docs/modeling/assumption_register.md`
- Draft modify: `docs/modeling/revised_four_problem_architecture.md`
- Modify: `state/decision_log.json`

**Step 1:** 从 DOCX XML 逐段提取 P1–P22，建立事实、歧义、解释、场景假设、
算法假设和鲁棒扩展七层表。

**Step 2:** 为 Q1–Q4 写状态、原始决策变量、离散变量、连续约束、目标和依赖。

**Step 3:** 明确 Q1 只可作为 Q2 warm start 候选，不能收缩 Q2 原始空间；
Q3/Q4 关系留待 Stage 3 证据裁决。

**Step 4:** 完成 L1 评分，更新 `current_stage=3` 并提交。

### Task 4: 系统文献与证据矩阵

**Files:**
- Create: `docs/research/literature-review.md`
- Create: `docs/research/literature-matrix.csv`
- Create: `docs/research/model-and-algorithm-evidence-map.md`

**Step 1:** 写入至少 24 条已核验资料，保留 DOI/稳定 URL、检索日期和元数据
核验状态。

**Step 2:** 每篇记录问题、模型、变量、目标、约束、算法、保证、验证、
可迁移和差异。

**Step 3:** 统计分类门槛，排除串题参数和不能核验的引用。

**Step 4:** 运行一个 CSV 结构测试，确保必需列齐全、DOI/URL 非空、分类计数
达标。

### Task 5: Q1 审计

**Files:**
- Create: `docs/reviews/q1-model-algorithm-audit.md`
- Create: `docs/reviews/pr5-root-cause-review.md`

**Step 1:** 静态追踪 Q1 从题面常数到场景、动力学、候选、验证和结果输出。

**Step 2:** 用调用图证明正式 Q1 是否调用连续优化器。

**Step 3:** 分别复核 10.376135 s 上界和 5.5 s 证书的前提、公式与测试。

**Step 4:** 分类旧结论为严格解析、认证结论、候选库最优、局部结果或未证明。

**Step 5:** 对 PR #5 做根因审计，不读取其数值作为当前结果。

### Task 6: 建立 toy demo 公共契约（TDD）

**Files:**
- Create: `experiments/toy_demos/__init__.py`
- Create: `experiments/toy_demos/common.py`
- Create: `tests/toy_demos/test_common.py`

**Step 1: Write failing tests**

测试 `ToyRunRecord` 必须拒绝缺失随机种子、非有限目标值、负耗时，并能输出
标准 JSON。

**Step 2: Verify RED**

Run: `python -m pytest tests/toy_demos/test_common.py -q`

Expected: FAIL because module/API does not exist。

**Step 3: Implement minimal common contract**

提供不可变 dataclass、JSON 序列化、计时与固定 seed 工具。

**Step 4: Verify GREEN and commit**

### Task 7: Q1 连续优化 demo（TDD）

**Files:**
- Create: `experiments/toy_demos/q1_continuous_optimization.py`
- Create: `tests/toy_demos/test_q1_continuous_optimization.py`

**Step 1:** 先写人工可判定二维事件目标测试；已知对称场景最优起爆中心应位于
舰船轨迹带中心，覆盖时长不超过解析上界。

**Step 2:** 观察测试因 API 缺失而失败。

**Step 3:** 实现统一评估预算下的 multi-start SLSQP、DE+SLSQP、
PSO+SLSQP、Sobol+trust-constr 和 `shgo` 小规模确定性对照。

**Step 4:** 固定种子运行，断言所有返回方案通过独立解析 verifier。

### Task 8: Q2 constraint generation demo（TDD）

**Files:**
- Create: `experiments/toy_demos/q2_constraint_generation.py`
- Create: `tests/toy_demos/test_q2_constraint_generation.py`

**Step 1:** 构造单位圆盘被两个/三个可移动圆覆盖的人工可判定案例。

**Step 2:** 测试初始有限网格会漏掉违反点，separation oracle 必须找到它。

**Step 3:** 实现有限 master、power-distance separation 和迭代见证加入。

**Step 4:** 断言 violation upper bound 单调不增并在容差内停止；保留 unresolved。

### Task 9: Q2 离散—连续联合 demo（TDD）

**Files:**
- Create: `experiments/toy_demos/q2_joint_prototype.py`
- Create: `tests/toy_demos/test_q2_joint_prototype.py`

**Step 1:** 构造最多三弹、有限顺序和连续起爆时刻的简化问题。

**Step 2:** 用小网格穷举作人工基准。

**Step 3:** 实现候选组合+连续精修与小 MINLP/枚举+oracle 两路线。

**Step 4:** 比较上下界、耗时和 unresolved；不得输出正式 Q2 场景结果。

### Task 10: Q3 多目标 demo（TDD）

**Files:**
- Create: `experiments/toy_demos/q3_multiobjective.py`
- Create: `tests/toy_demos/test_q3_multiobjective.py`

**Step 1:** 构造有限候选的三机三弹问题，穷举得到真实 Pareto 前沿。

**Step 2:** 实现 ε-约束选择和 NSGA-II 对照。

**Step 3:** 断言 ε-约束解属于穷举前沿；记录 NSGA-II 覆盖率与随机稳定性。

### Task 11: Q4 调度 demo（TDD）

**Files:**
- Create: `experiments/toy_demos/q4_scheduling.py`
- Create: `tests/toy_demos/test_q4_scheduling.py`

**Step 1:** 构造三批威胁与有限认证任务包，手工给出最优小实例。

**Step 2:** 实现离线全知 MILP、滚动时域 MILP和贪心。

**Step 3:** 测试在线模式不能读取未来威胁，离线值只作上界。

### Task 12: 运行、记录与 Stage 3 选型

**Files:**
- Create: `experiments/toy_demos/run_all.py`
- Create: `experiments/toy_demos/README.md`
- Create: `experiments/toy_demos/results/*.json`
- Create: `docs/workflow/stage-03-model-selection.md`
- Create: `docs/modeling/model-selection-decision-record.md`
- Modify: `state/decision_log.json`

**Step 1:** 先测试 `run_all` 的结果 schema 和确定性。

**Step 2:** 运行全部 demo，记录真实耗时、收敛、失败和人工案例。

**Step 3:** 建立题意适配等十维决策矩阵；每个模块至少三个候选。

**Step 4:** 加载 anti-patterns，执行 red-team；失败候选降级或否决。

**Step 5:** 完成 Stage 3 L1，设置 `toy_demos_passed` 和 `current_stage=4`，
明确“仅表示下一阶段待开始”。

### Task 13: Gate B 完整验证与发布

**Files:**
- Modify: Draft PR body

**Step 1:** 运行：

```powershell
python -m pytest -q
python -m ruff check .
python scripts/export_scenario_schema.py --check
python scripts/generate_scenarios.py --check
python experiments/toy_demos/run_all.py --check
git diff --check
```

**Step 2:** 核对没有生产代码、正式结果或 Stage 4 契约变更。

**Step 3:** 提交、推送并创建 Draft PR
“按数学建模工作流重构 B 题模型与算法选型”。

**Step 4:** 更新 PR 描述中的 Stage、证据、风险和人工决策。

**Step 5:** 在 Gate B 停止，不执行 Stage 4。
