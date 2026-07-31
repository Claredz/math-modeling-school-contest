# Fast Stage 4 and Q1–Q2 Rebuild Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不合并 main、不继承废弃 Q2 的前提下，冻结 Stage 4，并交付可复现、可认证、可直接写入论文的 Q1 与 Q2 rebuild 产物。

**Architecture:** 保留现有公共物理/几何内核，新增 rebuild 场景、候选生成、连续 solver 和独立 verifier。Q1 以两变量事件参数化运行四算法公平基准；Q2 以 Q1 warm start 扩展 1–3 枚弹，并用保守连续时空包络给出三值认证。

**Tech Stack:** Python 3.11、NumPy、SciPy、Shapely、Pydantic、Matplotlib、pytest、Ruff。

---

### Task 1: Gate B 与 Stage 4 冻结

**Files:**
- Modify: `tests/test_gate_b_consistency.py`
- Modify: `state/decision_log.json`
- Modify: `docs/modeling/model-selection-decision-record.md`
- Modify: `docs/modeling/model_contract_v0.1.md`
- Modify: `docs/modeling/assumption_register.md`
- Create: `docs/workflow/stage-04-foundation.md`

**Steps:**

1. 先把 Gate B 13 项决定、`current_stage=4`、`stage_4_started=false` 写成失败测试。
2. 运行 `python -m pytest tests/test_gate_b_consistency.py -q`，确认旧 pending 状态导致失败。
3. 更新决策日志与 Gate B 决策记录，记录人工批准时间、13 项决定和分支边界。
4. 写 Stage 4 的 3–7 条核心假设、≥10 符号、≥3 术语及事件/认证/隔离合同。
5. 自检通过后更新为 `stage_4_status=passed`、`gate_c_status=approved_fast_track`、
   `stage_5_started=true`，但 Stage 5 尚未完成。
6. 重跑目标测试和 `git diff --check`，提交。

### Task 2: 正式瞬时追踪 4×4 场景合同

**Files:**
- Modify: `src/smoke_defense/scenario.py`
- Modify: `src/smoke_defense/scenario_matrix.py`
- Modify: `scripts/generate_scenarios.py`
- Modify: `tests/test_scenario.py`
- Modify: `tests/test_scenario_matrix.py`

**Steps:**

1. 写失败测试：新正式基准允许无惯性参数的瞬时纯追踪，矩阵恰为 16 场景；旧 144
   惯性场景保留为参数化反事实。
2. 运行目标测试并确认因旧 formal/ablation 硬编码失败。
3. 最小扩展 `model_layer` 和场景生成器，不改变旧函数返回值。
4. 重跑目标测试、schema 与场景 `--check`，提交。

### Task 3: Q1 候选与独立 verifier

**Files:**
- Create: `src/smoke_defense/q1_rebuild.py`
- Create: `tests/test_q1_rebuild.py`

**Steps:**

1. 写失败测试覆盖固定 2 s 响应、3.5 s 起爆、禁止负预指令、飞行期间正常运动、
   12 km 作战半径、solver/verifier 状态分离和词典序。
2. 确认模块缺失导致 RED。
3. 实现两变量候选构造和独立连续覆盖验证适配器。
4. 逐条运行测试至 GREEN，并运行现有 Q1/路径/验证回归测试。
5. 提交。

### Task 4: Q1 四算法公平基准

**Files:**
- Modify: `src/smoke_defense/q1_rebuild.py`
- Modify: `tests/test_q1_rebuild.py`

**Steps:**

1. 写失败测试：四方法共享 seed、候选评估预算、变量边界、verifier 和目标层级；
   原生状态与最终认证分开。
2. 实现 Multi-start SLSQP、Sobol+SLSQP、SHGO、DE+SLSQP 适配器。
3. 用真实代表场景运行小预算测试，确认至少返回可验证候选；失败方法保留失败状态。
4. 重跑测试并提交。

### Task 5: Q1 正式产物与论文图表

**Files:**
- Create: `scripts/run_q1_rebuild.py`
- Create: `tests/test_q1_rebuild_artifacts.py`
- Create: `docs/q1/q1-model.md`
- Create: `docs/q1/q1-algorithm-benchmark.md`
- Create: `docs/q1/q1-verification.md`
- Create: `results/q1_rebuild/*`
- Create: `figures/q1_rebuild/*`

**Steps:**

1. 写 artifact 合同失败测试：JSON/CSV 字段、16 场景、≥2 PNG、provenance 和状态词。
2. 实现 runner，先 benchmark 代表场景，再运行胜出方法的 4×4 正式矩阵。
3. 运行 runner 生成方案表、轨迹/UAV/覆盖时间/裕度图。
4. 写模型、算法、验证文档，严格区分“最好已知”和认证状态。
5. 运行 artifact 测试、Ruff 和 `--check` 重放，提交。

### Task 6: 失锁耦合反事实

**Files:**
- Create: `src/smoke_defense/lost_counterfactual.py`
- Create: `tests/test_lost_counterfactual.py`
- Modify: `scripts/run_q1_rebuild.py`
- Modify: `docs/q1/q1-verification.md`
- Modify: `results/q1_rebuild/*`
- Modify: `figures/q1_rebuild/*`

**Steps:**

1. 根据实验分支只读审查写失败测试：标签必须是
   `experimental_counterfactual`、`formal_baseline=false`，并输出 loss/reacquisition/
   minimum separation 及 \((\tau_T,\tau_L,T_R)\)。
2. 独立实现最小状态机，不复制实验分支正式结果或假设登记。
3. 仅跑少量代表场景与小参数网格，生成对比表/图。
4. 重跑测试并提交。

### Task 7: Q2 联合连续 verifier

**Files:**
- Create: `src/smoke_defense/q2_rebuild.py`
- Create: `tests/test_q2_rebuild.py`

**Steps:**

1. 写失败测试：两个单独不足的烟幕可联合覆盖；时间区间包络可认证可行；精确裸露
   见证可认证不可行；容差内不能闭合必须 `unresolved`。
2. 实现空间 union certificate 的时间包络与 counterexample search。
3. 运行目标测试至 GREEN，并回归 `tests/test_coverage.py`。
4. 提交。

### Task 8: Q2 候选生成、连续精修与约束生成

**Files:**
- Modify: `src/smoke_defense/q2_rebuild.py`
- Modify: `tests/test_q2_rebuild.py`

**Steps:**

1. 写失败测试：1–3 枚、固定事件延时、≥1 s 投弹间隔、连续 UAV 路径、Q1 warm
   start、错位起爆、oracle 见证回灌、solver/verifier 分离。
2. 实现事件序列候选、SLSQP 连续精修和有限轮 constraint generation。
3. 确保 oracle 未认证时返回 `unresolved`，不把局部 optimizer 成功冒充认证。
4. 重跑目标测试和 Q1 回归测试，提交。

### Task 9: Q2 正式产物

**Files:**
- Create: `scripts/run_q2_rebuild.py`
- Create: `tests/test_q2_rebuild_artifacts.py`
- Create: `docs/q2/q2-model.md`
- Create: `docs/q2/q2-algorithm.md`
- Create: `docs/q2/q2-verification.md`
- Create: `results/q2_rebuild/*`
- Create: `figures/q2_rebuild/*`

**Steps:**

1. 写失败 artifact 测试，覆盖方案、弹数、事件、路径、覆盖上下界、裸露、联合增益、
   unresolved、Q1 改善、运行时间和 provenance。
2. 实现 runner 并生成可提交版结果、≥2 图和 CSV 表。
3. 写三份 Q2 文档，禁止“精确全局最优”措辞。
4. 运行 artifact 测试和 `--check` 重放，提交。

### Task 10: Q2 子检查点与全量验证

**Files:**
- Modify: `state/decision_log.json`
- Modify: `README.md`（仅在需要索引新产物时）
- Update: Draft PR #7 描述

**Steps:**

1. 更新 Stage 5 的 Q1/Q2 路径、指标、分数、跨问复用链和剩余风险；保持 Q3/Q4
   未开始。
2. 运行：
   `python -m pytest -q`、
   `python -m ruff check .`、
   `python scripts/export_scenario_schema.py --check`、
   `python scripts/generate_scenarios.py --check`、
   `python experiments/toy_demos/run_all.py --check`、
   两个 rebuild runner `--check`、`git diff --check`。
3. 检查旧 `results/q1/` 与研究 base 无差异，工作树干净。
4. 提交、推送，等待 CI；更新 Draft PR #7，但保持 Draft 且不合并。
