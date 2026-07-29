# 校赛 B 题烟幕遮蔽优化 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 建立可复现的二维舰船烟幕防御仿真与分层优化系统，完成题目四问的模型、算法、验证、图表和论文材料。

**Architecture:** 使用一个共享的时空仿真内核描述舰船、导弹、无人机、烟幕和遮蔽判定；问题一至四分别在同一内核上构建单弹优化、多弹时序、三机多目标和多威胁滚动调度。所有缺失场景量通过 YAML 参数化，所有结果由同一套计算代码导出，禁止在论文中手填无法复现的数值。

**Tech Stack:** Python 3.11、NumPy、SciPy、PyYAML、Pandas、Matplotlib、Pymoo、Pytest、Ruff、LaTeX。

---

## 开工前约束

- 题目原件：`B题：舰船烟幕遮蔽干扰优化.docx`
- 设计依据：`docs/plans/2026-07-29-b-problem-modeling-design.md`
- 所有距离使用米、时间使用秒、角度在计算层使用弧度。
- 先运行测试，再运行优化；任何优化结果必须能被独立仿真复核。
- 若主办方未补充初始坐标，不得声称得到题目要求的“唯一数值最优解”，应输出参数化规律和基准场景结果。
- 优先级采用词典序：防御成功率 → 最大裸露时间 → 稳健裕度 → 弹药 → 航程/能耗。

## 建议 72 小时排期

| 时段 | 交付 |
|---|---|
| 0–3 h | 参数核对、场景文件、符号表、数据缺口清单 |
| 3–10 h | 公共动力学、烟幕与遮蔽判定、问题一 |
| 10–18 h | 问题二多弹连续覆盖 |
| 18–30 h | 问题三三机协同与 Pareto 方案 |
| 30–42 h | 问题四威胁评估与滚动调度 |
| 42–50 h | 灵敏度、蒙特卡罗、消融和模型对比 |
| 50–64 h | 图表、结果表、论文主体 |
| 64–70 h | 摘要、结论、模型评价、附录 |
| 70–72 h | 全量复跑、数字一致性和终稿检查 |

### Task 1: 创建可复现项目骨架与参数契约

**Files:**
- Create: `pyproject.toml`
- Create: `src/smoke_defense/__init__.py`
- Create: `src/smoke_defense/config.py`
- Create: `configs/base.yaml`
- Create: `configs/scenarios/baseline.yaml`
- Create: `configs/scenarios/README.md`
- Create: `tests/test_config.py`
- Modify: `.gitignore`

**Step 1: 写配置加载的失败测试**

在 `tests/test_config.py` 中写：

```python
from smoke_defense.config import load_problem


def test_load_problem_converts_ship_speed_to_si():
    cfg = load_problem("configs/base.yaml", "configs/scenarios/baseline.yaml")
    assert cfg.ship.speed_mps == 15 * 0.514
    assert cfg.smoke.max_radius_m == 120.0
    assert cfg.missile.detection_range_m == 8000.0


def test_missing_initial_position_is_rejected(tmp_path):
    scenario = tmp_path / "missing.yaml"
    scenario.write_text("ship: {}\n", encoding="utf-8")
    try:
        load_problem("configs/base.yaml", scenario)
    except ValueError as exc:
        assert "initial_position" in str(exc)
    else:
        raise AssertionError("应拒绝缺失初始坐标的场景")
```

**Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL，提示 `ModuleNotFoundError: smoke_defense`。

**Step 3: 建立依赖和配置模型**

`pyproject.toml` 至少声明：

```toml
[project]
name = "smoke-defense"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "scipy>=1.12",
  "pandas>=2.2",
  "matplotlib>=3.8",
  "pyyaml>=6.0",
  "pymoo>=0.6",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.6"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

在 `configs/base.yaml` 中录入题面明确参数：

```yaml
ship:
  speed_knots: 15
  equivalent_radius_m: 80
uav:
  speed_mps: 28
  max_radius_m: 12000
  payload: 3
  response_delay_s: 2
  min_drop_interval_s: 1
smoke:
  detonation_delay_s: 3.5
  max_radius_m: 120
  hold_duration_s: 18
  decay_duration_s: 5
missile:
  speed_mps: 320
  detection_range_m: 8000
  half_fov_deg: 15
simulation:
  time_step_s: 0.05
  random_seed: 20260729
```

在 `baseline.yaml` 中显式标记所有待主办方确认的场景值，并给出可替换的基准值：

```yaml
metadata:
  source: "baseline_assumption"
  must_replace_if_official_data_arrives: true
ship:
  initial_position_m: [0.0, 0.0]
  heading_deg: 0.0
missiles:
  - id: M1
    initial_position_m: [10000.0, 0.0]
    heading_deg: 180.0
    launch_time_s: 0.0
uavs:
  - id: U1
    initial_position_m: [0.0, 1000.0]
minimum_uav_separation_m: 100.0
```

使用 `dataclass(frozen=True)` 将 YAML 转成带单位含义的配置对象；缺失必需坐标时抛出 `ValueError`。

**Step 4: 调整 Git 跟踪策略**

修改 `.gitignore`，继续忽略原始大结果，但允许提交论文源文件和最终摘要：

```gitignore
!results/summary.json
!results/tables/
!results/tables/*.csv
!figures/final/
!figures/final/*.pdf
!figures/final/*.png
!paper_workspace/main.tex
!paper_workspace/sections/
!paper_workspace/sections/*.tex
!paper_workspace/references.bib
```

**Step 5: 安装环境并验证**

Run: `python -m pip install -e ".[dev]"`

Expected: 安装完成，无依赖冲突。

Run: `python -m pytest tests/test_config.py -v`

Expected: 2 passed。

**Step 6: 提交**

```bash
git add pyproject.toml src/smoke_defense/__init__.py src/smoke_defense/config.py configs tests/test_config.py .gitignore
git commit -m "build: add reproducible scenario configuration"
```

### Task 2: 实现舰船、导弹和无人机公共动力学

**Files:**
- Create: `src/smoke_defense/dynamics.py`
- Create: `tests/test_dynamics.py`

**Step 1: 写解析运动与追踪速度测试**

```python
import numpy as np
from smoke_defense.dynamics import ship_position, pure_pursuit_rhs


def test_ship_position_after_ten_seconds():
    pos = ship_position(
        initial=np.array([0.0, 0.0]),
        speed_mps=7.71,
        heading_rad=0.0,
        time_s=10.0,
    )
    np.testing.assert_allclose(pos, [77.1, 0.0], atol=1e-9)


def test_pure_pursuit_preserves_missile_speed():
    velocity = pure_pursuit_rhs(
        missile_position=np.array([1000.0, 0.0]),
        ship_position=np.array([0.0, 0.0]),
        missile_speed_mps=320.0,
    )
    assert np.isclose(np.linalg.norm(velocity), 320.0)
    assert velocity[0] < 0
```

**Step 2: 确认测试失败**

Run: `python -m pytest tests/test_dynamics.py -v`

Expected: FAIL，找不到 `smoke_defense.dynamics`。

**Step 3: 实现最小动力学函数**

实现：

- `ship_position(...)`
- `pure_pursuit_rhs(...)`
- `integrate_missile_trajectory(...)`
- `uav_reachable(...)`
- `detonation_position(...)`

导弹使用 `scipy.integrate.solve_ivp`；事件函数在导弹进入舰船等效半径时停止积分。`detonation_position` 必须支持 `inertial` 和 `fixed_point` 两种解释。

**Step 4: 增加边界测试**

补充测试：

- 零距离追踪时抛出明确异常；
- 无人机到达时间小于响应延时时不可投弹；
- 超过 12000 m 作战半径不可达；
- 惯性起爆位置等于投弹点加 \(3.5\boldsymbol v_u\)。

Run: `python -m pytest tests/test_dynamics.py -v`

Expected: 全部通过。

**Step 5: 提交**

```bash
git add src/smoke_defense/dynamics.py tests/test_dynamics.py
git commit -m "feat: implement shared platform dynamics"
```

### Task 3: 实现烟幕演化和两级遮蔽几何

**Files:**
- Create: `src/smoke_defense/smoke.py`
- Create: `src/smoke_defense/geometry.py`
- Create: `tests/test_smoke.py`
- Create: `tests/test_geometry.py`

**Step 1: 写烟幕分段函数测试**

```python
import pytest
from smoke_defense.smoke import smoke_radius


@pytest.mark.parametrize(
    ("age_s", "expected"),
    [(-0.1, 0.0), (0.0, 120.0), (18.0, 120.0),
     (20.5, 60.0), (23.0, 0.0), (30.0, 0.0)],
)
def test_smoke_radius_piecewise(age_s, expected):
    assert smoke_radius(age_s, 120.0, 18.0, 5.0) == pytest.approx(expected)
```

**Step 2: 写完全覆盖和角域遮挡测试**

```python
import numpy as np
from smoke_defense.geometry import disk_fully_covers, angular_occlusion


def test_disk_cover_boundary_is_inclusive():
    assert disk_fully_covers(
        smoke_center=np.array([40.0, 0.0]),
        smoke_radius_m=120.0,
        ship_center=np.array([0.0, 0.0]),
        ship_radius_m=80.0,
    )


def test_smoke_behind_ship_cannot_occlude_line_of_sight():
    assert not angular_occlusion(
        missile=np.array([0.0, 0.0]),
        smoke=np.array([1100.0, 0.0]),
        ship=np.array([1000.0, 0.0]),
        smoke_radius_m=120.0,
        ship_radius_m=80.0,
    )
```

**Step 3: 确认失败并实现**

Run: `python -m pytest tests/test_smoke.py tests/test_geometry.py -v`

Expected: 初次 FAIL；实现后全部通过。

实现时：

- 圆盘覆盖使用 \(\|c-s\|+r_s\le R+\epsilon\)；
- 角域模型先检查烟幕是否位于导弹—舰船线段的正确纵深顺序，再比较角半径；
- 所有反三角函数输入先裁剪至 `[-1, 1]`；
- 默认数值容差为 `1e-9`。

**Step 4: 提交**

```bash
git add src/smoke_defense/smoke.py src/smoke_defense/geometry.py tests/test_smoke.py tests/test_geometry.py
git commit -m "feat: add smoke evolution and occlusion geometry"
```

### Task 4: 建立探测窗口与覆盖评估器

**Files:**
- Create: `src/smoke_defense/coverage.py`
- Create: `tests/test_coverage.py`

**Step 1: 写探测窗口和区间并集测试**

```python
from smoke_defense.coverage import merge_intervals, coverage_metrics


def test_merge_touching_intervals_has_no_gap():
    assert merge_intervals([(0.0, 10.0), (10.0, 20.0)]) == [(0.0, 20.0)]


def test_coverage_metrics_report_largest_gap():
    metrics = coverage_metrics(
        detection_window=(0.0, 20.0),
        covered_intervals=[(0.0, 8.0), (10.0, 20.0)],
    )
    assert metrics.coverage_ratio == 0.9
    assert metrics.max_gap_s == 2.0
    assert not metrics.defense_success
```

**Step 2: 实现统一评估接口**

至少实现：

- `detection_mask(...)`
- `smoke_coverage_mask(...)`
- `mask_to_intervals(...)`
- `merge_intervals(...)`
- `coverage_metrics(...)`
- `evaluate_plan(...)`

`CoverageMetrics` 至少包含：

```python
coverage_ratio: float
covered_duration_s: float
max_gap_s: float
minimum_margin_m: float
defense_success: bool
```

**Step 3: 用可人工判定场景验证**

构造 20 秒探测窗口，烟幕全程半径 120 m、与舰船中心重合。预期：

- `coverage_ratio == 1`
- `max_gap_s == 0`
- `minimum_margin_m == 40`
- `defense_success is True`

Run: `python -m pytest tests/test_coverage.py -v`

Expected: 全部通过。

**Step 4: 提交**

```bash
git add src/smoke_defense/coverage.py tests/test_coverage.py
git commit -m "feat: add detection and coverage evaluator"
```

### Task 5: 完成问题一单弹优化

**Files:**
- Create: `src/smoke_defense/q1_single.py`
- Create: `scripts/run_q1.py`
- Create: `tests/test_q1_single.py`

**Step 1: 写可行性优先测试**

```python
from smoke_defense.q1_single import lexicographic_score


def test_full_defense_always_beats_partial_defense():
    full = lexicographic_score(
        defense_success=True, coverage_ratio=1.0,
        max_gap_s=0.0, minimum_margin_m=1.0, flight_distance_m=5000.0,
    )
    partial = lexicographic_score(
        defense_success=False, coverage_ratio=0.999,
        max_gap_s=0.01, minimum_margin_m=100.0, flight_distance_m=1.0,
    )
    assert full < partial
```

**Step 2: 建立两阶段求解器**

实现流程：

1. 时间网格扫描起爆时刻；
2. 根据舰船轨迹生成烟幕中心候选带；
3. 用无人机可达性和航程筛选；
4. 对候选解使用 `scipy.optimize.differential_evolution` 精修；
5. 用 `evaluate_plan` 独立复核；
6. 若无 100% 可行解，返回 `infeasible_full_cover` 和理论最佳覆盖率。

`SingleBombSolution` 必须保存投弹时刻、投弹坐标、起爆时刻、起爆坐标、覆盖率、最大空档、最小裕度、飞行距离和求解状态。

**Step 3: 写小网格穷举对照**

在固定合成场景中，将投弹时间限制为 5 个候选、坐标限制为 3 个候选。手工穷举得到最佳索引，验证优化器选择同一候选。

Run: `python -m pytest tests/test_q1_single.py -v`

Expected: 词典序测试与穷举对照均通过。

**Step 4: 输出问题一结果**

Run: `python scripts/run_q1.py --base configs/base.yaml --scenario configs/scenarios/baseline.yaml`

Expected:

- 生成 `results/q1_solution.json`
- 生成 `results/q1_timeline.csv`
- 控制台明确打印 `full_defense_feasible: true/false`
- 不出现 NaN 或违反航程的解

**Step 5: 提交**

```bash
git add src/smoke_defense/q1_single.py scripts/run_q1.py tests/test_q1_single.py
git commit -m "feat: solve single-bomb defense problem"
```

### Task 6: 完成问题二多弹连续覆盖

**Files:**
- Create: `src/smoke_defense/q2_multibomb.py`
- Create: `scripts/run_q2.py`
- Create: `tests/test_q2_multibomb.py`

**Step 1: 写无空档时序测试**

```python
from smoke_defense.q2_multibomb import schedule_metrics


def test_three_touching_coverage_intervals_are_continuous():
    metrics = schedule_metrics([(0, 10), (10, 20), (20, 30)])
    assert metrics.continuous_duration_s == 30
    assert metrics.max_gap_s == 0
```

**Step 2: 写投弹约束测试**

覆盖以下反例：

- 同机两次投弹间隔小于 1 秒；
- 投弹数超过 3；
- 总航迹超出作战半径；
- 某枚弹的投放点在对应时刻不可达。

每个反例必须返回可解释的约束名称，而不是只返回 `False`。

**Step 3: 实现候选生成和组合选择**

先复用问题一生成每枚弹的空间—时间候选，再建立 0-1 选择模型：

- 首要最小化探测窗口内总裸露时间；
- 次要最小化最大空档；
- 再最大化最小覆盖裕度；
- 最后最小化投弹数和航程。

小规模使用枚举验证；正常规模使用 `scipy.optimize.milp` 或候选组合的分支定界。

**Step 4: 运行对比实验**

Run: `python scripts/run_q2.py --base configs/base.yaml --scenario configs/scenarios/baseline.yaml`

Expected:

- `results/q2_solution.json`
- `results/q2_comparison.csv`
- 单弹与 1/2/3 弹的覆盖率、最大空档、容错和航程对比
- 任意相邻投弹时刻差不小于 1 秒

**Step 5: 提交**

```bash
git add src/smoke_defense/q2_multibomb.py scripts/run_q2.py tests/test_q2_multibomb.py
git commit -m "feat: optimize continuous multi-bomb coverage"
```

### Task 7: 完成问题三三机协同多目标优化

**Files:**
- Create: `src/smoke_defense/q3_cooperative.py`
- Create: `scripts/run_q3.py`
- Create: `tests/test_q3_cooperative.py`

**Step 1: 写安全间距和失效容错测试**

```python
from smoke_defense.q3_cooperative import minimum_pairwise_distance


def test_minimum_pairwise_distance_detects_conflict():
    trajectories = {
        "U1": [(0.0, 0.0), (10.0, 0.0)],
        "U2": [(0.0, 50.0), (10.0, 50.0)],
    }
    assert minimum_pairwise_distance(trajectories) == 50.0
```

再写一个三弹方案测试：删除任意一枚弹后重新计算覆盖率，`single_failure_coverage` 必须等于三次删弹结果的最小值。

**Step 2: 实现 NSGA-II 问题编码**

每架无人机编码：

- 投弹时刻；
- 投弹点 \(x,y\)；
- 航向；
- 由固定延时确定起爆时刻和位置。

目标向量：

```text
[-coverage_ratio,
 max_gap_s,
 -minimum_margin_m,
 -single_failure_coverage,
 total_flight_distance_m]
```

约束：

- 每机仅 1 枚；
- 投弹点可达；
- 任意时刻安全间距不低于场景参数；
- 起爆和覆盖发生在有效作战时间内。

**Step 3: 建立 Pareto 解筛选规则**

先筛 `defense_success == True` 的方案；若存在，选最小航程且单点失效覆盖率最高者。若不存在，按覆盖率、最大空档和裕度的词典序选择。

**Step 4: 与单机方案对比**

Run: `python scripts/run_q3.py --base configs/base.yaml --scenario configs/scenarios/baseline.yaml --seed 20260729`

Expected:

- `results/q3_pareto.csv`
- `results/q3_recommended.json`
- 单机单弹与三机方案对比表
- 相同随机种子两次运行推荐方案一致或目标值在容差内一致

**Step 5: 提交**

```bash
git add src/smoke_defense/q3_cooperative.py scripts/run_q3.py tests/test_q3_cooperative.py
git commit -m "feat: optimize three-UAV cooperative defense"
```

### Task 8: 完成问题四威胁评估与滚动资源调度

**Files:**
- Create: `configs/scenarios/multi_threat_abundant.yaml`
- Create: `configs/scenarios/multi_threat_critical.yaml`
- Create: `configs/scenarios/multi_threat_shortage.yaml`
- Create: `src/smoke_defense/q4_scheduler.py`
- Create: `scripts/run_q4.py`
- Create: `tests/test_q4_scheduler.py`

**Step 1: 写威胁排序测试**

建立两个除预计命中时间外完全相同的威胁，验证预计命中时间更短者威胁分数更高。建立同一目标的两个任务包，验证高成本任务包不会在低成本任务包已满足 100% 防御时被重复分配。

**Step 2: 定义任务包**

从问题一至三抽象：

```text
P1: 1 架无人机 + 1 枚弹
P2: 1 架无人机 + 2 枚弹
P3: 1 架无人机 + 3 枚弹
P4: 3 架无人机 + 各 1 枚弹
```

每个任务包必须记录适用时间窗、所需无人机数、弹药数、预计覆盖率、最坏空档、航程和失效容错。

**Step 3: 实现威胁评分**

威胁评分只使用有物理含义且可复核的指标：

- 预计命中时间倒数；
- 距进入成像探测窗口的时间倒数；
- 入射角导致的防御机动难度；
- 未防御损失权重；
- 单位资源预期防御增益。

权重先采用等权或明确的任务优先级，再通过 ±20% 权重扰动检查排序稳定性。没有样本数据时不要伪用熵权法。

**Step 4: 实现滚动时域 MILP**

决策变量表示“威胁—任务包—开始时间—无人机”的分配。约束：

- 5 架无人机；
- 每机最多 3 枚；
- 同一无人机任务时间不重叠；
- 投弹间隔和航程可行；
- 空域冲突不可发生；
- 每个威胁最多选择一个主任务包，可选一个冗余任务包。

目标先最大化加权防御成功收益，再最小化弹药与航程。每次新批次到达或状态变化时滚动重算。

**Step 5: 验证三类资源工况**

Run:

```bash
python scripts/run_q4.py --scenario configs/scenarios/multi_threat_abundant.yaml
python scripts/run_q4.py --scenario configs/scenarios/multi_threat_critical.yaml
python scripts/run_q4.py --scenario configs/scenarios/multi_threat_shortage.yaml
```

Expected:

- 充足场景：全部威胁获分配且至少一个高威胁具有冗余；
- 临界场景：高威胁先于低威胁获得资源；
- 短缺场景：不超出 5 机、15 弹，并明确列出未覆盖风险；
- 所有调度方案通过独立时空仿真复核。

**Step 6: 提交**

```bash
git add configs/scenarios/multi_threat_*.yaml src/smoke_defense/q4_scheduler.py scripts/run_q4.py tests/test_q4_scheduler.py
git commit -m "feat: add rolling multi-threat resource scheduler"
```

### Task 9: 完成稳健性、灵敏度和模型消融

**Files:**
- Create: `src/smoke_defense/robustness.py`
- Create: `scripts/run_robustness.py`
- Create: `tests/test_robustness.py`

**Step 1: 写可重复抽样测试**

相同随机种子生成完全相同的扰动样本；不同随机种子生成不同样本。每个样本必须记录：

- 舰船和导弹速度扰动；
- 起爆延时误差；
- 烟幕最大半径误差；
- 烟幕中心风致漂移；
- 初始坐标测量误差。

**Step 2: 实现灵敏度分析**

对每个关键参数做单因素 ±5%、±10%、±20% 扰动，输出：

- 防御成功率；
- 覆盖率变化；
- 最大空档；
- 推荐投放时刻与坐标变化；
- 资源调度变化。

**Step 3: 实现蒙特卡罗验证**

使用固定种子运行至少 1000 个样本；先用 100、500、1000 样本检查置信区间收敛。报告 95% 置信区间和最坏 5% 分位结果。

**Step 4: 完成消融实验**

至少比较：

- 固定点起爆 vs 惯性飞行起爆；
- 圆盘覆盖 vs 角域遮挡；
- 无稳健裕度 vs 有稳健裕度；
- 静态调度 vs 滚动调度；
- 单弹、三弹、三机方案。

Run: `python scripts/run_robustness.py --samples 1000 --seed 20260729`

Expected: 生成 `results/robustness_summary.csv`，无 NaN，重复运行统计值一致。

**Step 5: 提交**

```bash
git add src/smoke_defense/robustness.py scripts/run_robustness.py tests/test_robustness.py
git commit -m "feat: add robustness and sensitivity analysis"
```

### Task 10: 建立统一运行入口和结果一致性检查

**Files:**
- Create: `scripts/run_all.py`
- Create: `src/smoke_defense/reporting.py`
- Create: `tests/test_reporting.py`
- Create: `results/summary.json`
- Create: `results/tables/.gitkeep`
- Create: `figures/final/.gitkeep`

**Step 1: 写摘要数字来源测试**

创建合成结果，验证 `reporting.py` 生成的摘要 JSON、CSV 表和图注中的覆盖率来自同一浮点值并使用统一舍入规则。

**Step 2: 实现统一流水线**

`run_all.py` 顺序执行：

1. 配置校验；
2. 公共轨迹生成；
3. Q1；
4. Q2；
5. Q3；
6. Q4 三场景；
7. 稳健性；
8. 汇总表与最终图。

每个阶段记录配置哈希、Git 提交 SHA、随机种子、开始时间和结束时间。

**Step 3: 生成论文所需图表**

至少生成：

1. 多主体二维轨迹图；
2. 导弹探测窗口与烟幕有效窗口时间轴；
3. 烟幕半径分段曲线；
4. Q1 覆盖裕度曲线；
5. Q2 多弹时序甘特图；
6. Q3 Pareto 前沿；
7. Q4 威胁—资源分配图；
8. 灵敏度龙卷风图；
9. 蒙特卡罗成功率分布。

**Step 4: 全量运行**

Run: `python scripts/run_all.py --base configs/base.yaml --scenario configs/scenarios/baseline.yaml`

Expected:

- 四问均有机器可读结果；
- 所有最终图存在；
- `results/summary.json` 记录配置哈希和 Git SHA；
- 运行失败时非零退出，不保留伪“成功”摘要。

**Step 5: 提交**

```bash
git add scripts/run_all.py src/smoke_defense/reporting.py tests/test_reporting.py results/summary.json results/tables/.gitkeep figures/final/.gitkeep
git commit -m "feat: add reproducible end-to-end reporting"
```

### Task 11: 撰写论文并绑定计算结果

**Files:**
- Create: `paper_workspace/main.tex`
- Create: `paper_workspace/sections/01_problem.tex`
- Create: `paper_workspace/sections/02_assumptions.tex`
- Create: `paper_workspace/sections/03_notation.tex`
- Create: `paper_workspace/sections/04_common_model.tex`
- Create: `paper_workspace/sections/05_q1.tex`
- Create: `paper_workspace/sections/06_q2.tex`
- Create: `paper_workspace/sections/07_q3.tex`
- Create: `paper_workspace/sections/08_q4.tex`
- Create: `paper_workspace/sections/09_robustness.tex`
- Create: `paper_workspace/sections/10_evaluation.tex`
- Create: `paper_workspace/references.bib`
- Create: `scripts/check_paper_numbers.py`

**Step 1: 建立论文骨架**

论文结构：

1. 摘要与关键词；
2. 问题重述；
3. 模型假设；
4. 符号说明；
5. 公共运动与烟幕模型；
6. 问题一；
7. 问题二；
8. 问题三；
9. 问题四；
10. 灵敏度与稳健性；
11. 模型评价与推广；
12. 参考文献和附录。

**Step 2: 写数字一致性检查**

`check_paper_numbers.py` 读取 `results/summary.json`，检查论文中使用标记引用的关键数字是否一致。关键结果不要手工复制，使用生成的 TeX 宏或表格。

**Step 3: 填写四问最小交付**

每问必须包含：

- 问题卡片；
- 决策变量；
- 目标函数；
- 完整约束；
- 求解流程；
- 数值/参数化结果；
- 有效性验证；
- 本问局限与向下一问的接口。

**Step 4: 编译与检查**

Run: `python scripts/check_paper_numbers.py`

Expected: `0 inconsistent values`。

Run: `xelatex -interaction=nonstopmode -halt-on-error paper_workspace/main.tex`

Expected: 退出码 0，生成 PDF，无 undefined control sequence。

**Step 5: 提交**

```bash
git add paper_workspace scripts/check_paper_numbers.py
git commit -m "docs: write reproducible modeling paper"
```

### Task 12: 最终验收、红队审核和发布

**Files:**
- Create: `docs/final-review-checklist.md`
- Modify: `README.md`

**Step 1: 执行代码质量检查**

Run: `python -m ruff check src tests scripts`

Expected: `All checks passed!`

Run: `python -m pytest -q`

Expected: 0 failed。

**Step 2: 从干净结果目录全量复跑**

先将旧结果移动到可恢复的临时备份目录，再运行：

```bash
python scripts/run_all.py --base configs/base.yaml --scenario configs/scenarios/baseline.yaml
python scripts/check_paper_numbers.py
```

Expected: 四问产物齐全，数字一致性为 0 个错误。

**Step 3: 执行物理与约束审核**

逐项确认：

- 舰速确为 7.71 m/s；
- 无人机速度 28 m/s；
- 导弹速度 320 m/s；
- 探测距离 8000 m、视场半角 15°；
- 起爆延时 3.5 s；
- 烟幕 18 s 恒定、5 s 衰减；
- 单机不超过 3 枚、投弹间隔不小于 1 s；
- 问题三每机仅 1 枚；
- 问题四不超过 5 机和 15 枚；
- 所有 100% 防御结论均由逐时刻仿真验证。

**Step 4: 执行红队问题**

在 `docs/final-review-checklist.md` 回答：

1. 单弹是否可能因探测窗口长于有效遮蔽窗口而理论不可行？
2. 视场约束在纯追踪模型中是否退化？
3. 缺失初始坐标是否被误写成官方数据？
4. 起爆点解释变化是否改变推荐方案？
5. 优化器是否只报告局部最优？
6. Q4 威胁权重变化是否改变资源优先级？
7. 图表和正文数字能否由同一结果文件复现？

**Step 5: 更新 README**

README 链接题目、设计文档、执行计划、运行命令和结果摘要，并醒目标注“当前基准场景包含假设参数”。

**Step 6: 提交**

```bash
git add docs/final-review-checklist.md README.md
git commit -m "docs: add final modeling review checklist"
```

**Step 7: 发布前核对**

Run:

```bash
git status -sb
git log --oneline --decorate -12
git diff origin/main...HEAD --stat
```

Expected: 工作区干净；提交按任务分层；差异仅包含 B 题建模项目文件。

## 完成标准

- [ ] 题面参数与场景假设完全分离；
- [ ] 公共动力学和遮蔽判定均有边界测试；
- [ ] Q1 能报告 100% 可行或严格报告不可行；
- [ ] Q2 无违反投弹间隔和航程的方案；
- [ ] Q3 给出 Pareto 解和单点失效指标；
- [ ] Q4 在三种资源工况下均给出合法调度；
- [ ] 稳健性至少包含 1000 次可重复蒙特卡罗；
- [ ] 每张最终图都能由脚本生成；
- [ ] 论文关键数字与 `results/summary.json` 一致；
- [ ] `ruff` 和 `pytest` 均通过；
- [ ] 仓库公开且题目、设计、计划和最终成果可访问。
