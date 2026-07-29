# B题烟幕防御建模 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不改变 A-001–A-020 的前提下，实现参数化场景、连续时间约束证书、联合覆盖验证器以及依次复用这些能力的 Q1–Q4 求解流水线。

**Architecture:** 按能力依赖而不是按题号拆分代码。场景契约和混合事件时间线位于最底层；共享动力学、解析证书、路径约束和联合覆盖验证器构成唯一可信计算内核；Q1–Q4 只生成候选并调用验证器。所有结论保存场景哈希、假设版本、Git SHA、随机种子和三值认证状态。

**Tech Stack:** Python 3.12、NumPy、SciPy、Pydantic、PyYAML、jsonschema、Shapely、pandas、pytest、Hypothesis、Ruff、Matplotlib。

---

## 执行约束

1. 开始每个任务前阅读：
   - `docs/modeling/model_contract_v0.1.md`
   - `docs/modeling/assumption_register.md`
   - `docs/plans/2026-07-30-b-problem-implementation-design.md`
2. 首版只实现 A-002 纯追踪；PN 配置保留但必须拒绝启用。
3. 优化器只返回候选；只有 `ContinuousCoverageVerifier` 可以返回 `certified_feasible`。
4. 每完成一个任务立即运行该任务测试和全量快速测试，再提交。
5. 不把生成的场景、结果或图表手工改成“更好看”的数字。
6. 若实现发现冻结契约自相矛盾，停止编码并新建模型变更记录，不在代码中静默修正。

## 依赖总览

```text
Task 1 场景契约与协作接口
  ↓
Task 2 场景矩阵与溯源
  ↓
Task 3 混合事件时间线
  ↓
Task 4 共享动力学与探测窗口
  ↓
Task 5 Q1 解析证书
  ↓
Task 6 UAV 路径与移动舰船约束
  ↓
Task 7 联合覆盖与连续时间验证器
  ↓
Task 8 Q1 候选与参数扫描
  ↓
Task 9 Q2/Q3 组合优化
  ↓
Task 10 Q4 任务包与滚动调度
  ↓
Task 11 鲁棒验证、报告和端到端验收
```

---

## Stage 1：契约与场景层

### Task 1：建立 Python 工程、强类型场景模型和协作边界

**Files:**
- Create: `pyproject.toml`
- Create: `src/smoke_defense/__init__.py`
- Create: `src/smoke_defense/contracts.py`
- Create: `src/smoke_defense/scenario.py`
- Create: `configs/constants.yaml`
- Create: `configs/schema/scenario.schema.json`
- Create: `configs/scenarios/examples/q1_front_d10000_nominal.yaml`
- Create: `docs/ai-task-template.md`
- Create: `tests/test_scenario.py`

**Step 1: 写场景载入的失败测试**

```python
from pathlib import Path

import pytest

from smoke_defense.scenario import ScenarioError, load_scenario


EXAMPLE = Path("configs/scenarios/examples/q1_front_d10000_nominal.yaml")


def test_example_scenario_has_frozen_semantics():
    scene = load_scenario(EXAMPLE)
    assert scene.schema_version == "1.0"
    assert scene.time_origin == "decision_start"
    assert scene.missiles[0].guidance_model == "pure_pursuit"
    assert scene.constraints.operation_radius_reference == "moving_ship"
    assert {"A-019", "A-020"} <= set(scene.assumption_ids)


def test_unknown_field_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        EXAMPLE.read_text(encoding="utf-8") + "\nunknown_knob: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ScenarioError, match="unknown_knob"):
        load_scenario(bad)
```

**Step 2: 运行测试，确认失败**

Run: `python -m pytest tests/test_scenario.py -v`

Expected: FAIL，提示 `smoke_defense.scenario` 不存在。

**Step 3: 实现最小强类型契约**

在 `contracts.py` 定义：

```python
from enum import StrEnum


class CertificationStatus(StrEnum):
    FEASIBLE = "certified_feasible"
    INFEASIBLE = "certified_infeasible"
    INDETERMINATE = "indeterminate_at_tolerance"


FROZEN_ASSUMPTIONS = tuple(f"A-{index:03d}" for index in range(1, 21))
```

在 `scenario.py` 使用 Pydantic `extra="forbid"` 定义 `ShipSpec`、`UavSpec`、`MissileSpec`、`ConstraintSpec`、`UncertaintySpec` 和 `Scenario`。载入顺序固定为：

1. YAML 解析；
2. JSON Schema 校验；
3. Pydantic 语义校验；
4. 检查基准场景只允许 `pure_pursuit` 和 `velocity_aligned`；
5. 检查所有速度、半径和时间非负；
6. 生成规范化字典。

`constants.yaml` 只保存题面常数及事实编号，不保存初始坐标、安全距离或误差。

**Step 4: 固化 AI 任务模板**

`docs/ai-task-template.md` 必须要求每次 AI 工作单包含：

```text
上游模型契约：
本次任务：
允许修改文件：
禁止修改文件：
输入场景：
必须运行的测试：
验收者：
```

并写明：聊天回答不是验收证据，提交必须经过测试和公共验证器。

**Step 5: 运行测试和静态检查**

Run: `python -m pytest tests/test_scenario.py -v`

Expected: 2 passed。

Run: `python -m ruff check src tests`

Expected: `All checks passed!`

**Step 6: 提交**

```bash
git add pyproject.toml src/smoke_defense configs/constants.yaml configs/schema configs/scenarios/examples docs/ai-task-template.md tests/test_scenario.py
git commit -m "feat: establish scenario and collaboration contracts"
```

### Task 2：生成参数化场景矩阵和可复现溯源

**Files:**
- Create: `configs/sweeps/q1_q3_matrix.yaml`
- Create: `configs/sweeps/safe_distance.yaml`
- Create: `configs/sweeps/robustness.yaml`
- Create: `configs/sweeps/pn_ablation.yaml`
- Create: `src/smoke_defense/scenario_matrix.py`
- Create: `src/smoke_defense/provenance.py`
- Create: `scripts/generate_scenarios.py`
- Create: `tests/test_scenario_matrix.py`
- Create: `tests/test_provenance.py`

**Step 1: 写 16 场景矩阵测试**

```python
import numpy as np

from smoke_defense.scenario_matrix import generate_q1_q3_scenarios


def test_q1_q3_matrix_has_four_directions_and_four_distances():
    scenes = generate_q1_q3_scenarios()
    assert len(scenes) == 16
    assert {scene.initial_distance_m for scene in scenes} == {
        8000.0, 10000.0, 12000.0, 15000.0
    }
    oblique = next(scene for scene in scenes if scene.direction == "oblique"
                    and scene.initial_distance_m == 10000.0)
    expected = 10000.0 * np.array([
        np.cos(np.deg2rad(135.0)),
        np.sin(np.deg2rad(135.0)),
    ])
    np.testing.assert_allclose(oblique.missile_position_body_m, expected)
```

再测试安全距离恰为 `{50, 100, 200, 500}`，鲁棒等级字段与设计文档完全一致，`pn_ablation.yaml` 默认 `enabled: false`。

**Step 2: 写配置哈希稳定性测试**

同一规范化场景即使 YAML 键顺序不同，也必须产生相同 SHA-256；任何数值改变必须改变哈希。

**Step 3: 运行测试，确认失败**

Run: `python -m pytest tests/test_scenario_matrix.py tests/test_provenance.py -v`

Expected: FAIL，缺少生成器和哈希函数。

**Step 4: 实现场景生成和溯源**

`Provenance` 至少包含：

```python
scenario_id: str
scenario_sha256: str
assumption_ids: tuple[str, ...]
git_sha: str
random_seed: int | None
generated_at_utc: str
```

生成文件写入 `configs/scenarios/q1_q3/generated/`。脚本必须先在临时目录生成，再逐个通过 `load_scenario`，全部合法后才替换生成目录。

**Step 5: 验证输出**

Run: `python scripts/generate_scenarios.py`

Expected: 输出 16 个 YAML，所有文件通过 schema 校验。

Run: `python -m pytest tests/test_scenario_matrix.py tests/test_provenance.py -v`

Expected: 全部通过。

**Step 6: 提交**

```bash
git add configs/sweeps configs/scenarios/q1_q3 src/smoke_defense/scenario_matrix.py src/smoke_defense/provenance.py scripts/generate_scenarios.py tests/test_scenario_matrix.py tests/test_provenance.py
git commit -m "feat: generate traceable parameterized scenarios"
```

---

## Stage 2：混合事件时间线

### Task 3：实现事件类型、因果关系和连续模式切分

**Files:**
- Create: `src/smoke_defense/events.py`
- Create: `src/smoke_defense/timeline.py`
- Create: `tests/test_timeline.py`

**Step 1: 写事件因果测试**

```python
import pytest

from smoke_defense.events import BombEvents, EventOrderError


def test_bomb_event_timing_respects_a005_a006():
    events = BombEvents.from_command(command_time_s=4.0)
    assert events.release_time_s == pytest.approx(6.0)
    assert events.burst_time_s == pytest.approx(9.5)
    assert events.hold_end_time_s == pytest.approx(27.5)
    assert events.expire_time_s == pytest.approx(32.5)


def test_release_before_response_delay_is_rejected():
    with pytest.raises(EventOrderError):
        BombEvents(
            command_time_s=4.0,
            release_time_s=5.9,
            burst_time_s=9.4,
            hold_end_time_s=27.4,
            expire_time_s=32.4,
        )
```

**Step 2: 写切段测试**

输入事件 `[0, 3.5, 21.5, 26.5]` 和分析窗口 `[0, 30]`，预期连续区间为：

```python
[(0.0, 3.5), (3.5, 21.5), (21.5, 26.5), (26.5, 30.0)]
```

重复事件必须去重，但事件点仍单独出现在 `event_times` 中。

**Step 3: 运行失败测试**

Run: `python -m pytest tests/test_timeline.py -v`

Expected: FAIL。

**Step 4: 实现**

使用有序不可变 dataclass 表示事件。`HybridTimeline` 提供：

- `add_event(event)`
- `continuous_intervals(start_s, end_s)`
- `events_at(time_s)`
- `validate_causality()`

不得把事件时刻舍入到固定时间步。

**Step 5: 验证并提交**

Run: `python -m pytest tests/test_timeline.py -v`

Expected: 全部通过。

```bash
git add src/smoke_defense/events.py src/smoke_defense/timeline.py tests/test_timeline.py
git commit -m "feat: add causal hybrid event timeline"
```

---

## Stage 3：共享动力学

### Task 4：实现坐标变换、舰船/导弹/烟幕动力学和探测窗口

**Files:**
- Create: `src/smoke_defense/coordinates.py`
- Create: `src/smoke_defense/dynamics.py`
- Create: `src/smoke_defense/detection.py`
- Create: `tests/test_coordinates.py`
- Create: `tests/test_dynamics.py`
- Create: `tests/test_detection.py`

**Step 1: 写舰载起飞位置测试**

```python
import numpy as np

from smoke_defense.coordinates import shipborne_launch_position


def test_launch_offset_rotates_with_ship_heading():
    point = shipborne_launch_position(
        ship_position_m=np.array([100.0, 200.0]),
        ship_heading_rad=np.pi / 2,
        launch_offset_body_m=np.array([10.0, 0.0]),
    )
    np.testing.assert_allclose(point, [100.0, 210.0], atol=1e-12)
```

**Step 2: 写纯追踪与烟幕边界测试**

验证：

- 导弹速度模恒为场景速度；
- 导弹速度方向指向舰船当前中心；
- 零相对距离触发命中事件而不是除零；
- 烟幕年龄 `0, 18, 23` 秒时半径分别为 `120, 120, 0`；
- 起爆位置等于投弹点加 \(3.5\boldsymbol v_u(t_d)\)；
- 起爆后中心固定。

**Step 3: 写探测语义测试**

```python
def test_pure_pursuit_with_velocity_aligned_axis_has_zero_boresight_error():
    state = synthetic_pure_pursuit_state()
    assert boresight_error_rad(state) == pytest.approx(0.0, abs=1e-12)
```

另验证 `appearance_time_s` 与 `detection_entry_time_s` 分离：导弹在场景出现后仍可能过一段时间才进入 8000 m。

**Step 4: 运行失败测试**

Run: `python -m pytest tests/test_coordinates.py tests/test_dynamics.py tests/test_detection.py -v`

Expected: FAIL。

**Step 5: 最小实现**

动力学只接受已校验的 `Scenario`，输出带物理时间的 `Trajectory`。导弹积分使用 `solve_ivp`，设置命中和 8000 m 边界事件；所有事件时间回填 `HybridTimeline`。首版遇到非纯追踪场景必须抛出 `UnsupportedExtensionError`。

**Step 6: 验证解析边界**

对共线迎面和尾追场景，探测持续时间必须分别落在：

\[
\frac{7920}{327.71}=24.1677092551\ {\rm s},
\qquad
\frac{7920}{312.29}=25.3610426206\ {\rm s}.
\]

Run: `python -m pytest tests/test_coordinates.py tests/test_dynamics.py tests/test_detection.py -v`

Expected: 全部通过，解析边界误差小于 `1e-6 s`。

**Step 7: 提交**

```bash
git add src/smoke_defense/coordinates.py src/smoke_defense/dynamics.py src/smoke_defense/detection.py tests/test_coordinates.py tests/test_dynamics.py tests/test_detection.py
git commit -m "feat: implement shared baseline dynamics"
```

---

## Stage 4：Q1 解析证书

### Task 5：实现单烟幕持续时间上界和不可行性剪枝

**Files:**
- Create: `src/smoke_defense/certificates/__init__.py`
- Create: `src/smoke_defense/certificates/q1.py`
- Create: `tests/certificates/test_q1.py`

**Step 1: 写烟幕完整覆盖时长测试**

```python
import pytest

from smoke_defense.certificates.q1 import max_single_smoke_cover_duration


def test_fixed_single_smoke_cover_duration_upper_bound():
    duration = max_single_smoke_cover_duration(
        ship_speed_mps=7.71,
        ship_radius_m=80.0,
        smoke_radius_m=120.0,
    )
    assert duration == pytest.approx(80.0 / 7.71)
```

这里的 80 m 来自舰船中心可以在固定烟幕中心两侧各移动
\(120-80=40\rm\,m\)，总可行弦长上界为 80 m。

**Step 2: 写不可行证书测试**

```python
def test_full_detection_window_is_certified_infeasible():
    certificate = certify_single_smoke_full_cover(
        detection_duration_s=24.1677092551,
        max_cover_duration_s=80.0 / 7.71,
    )
    assert certificate.status.value == "certified_infeasible"
    assert certificate.gap_lower_bound_s > 13.7
```

再写反例：若导弹在 \(t=0\) 已进入探测区内部，探测窗口只有 5 s，持续时间上界不能单独证明不可行，状态必须是 `indeterminate_at_tolerance` 而不是 `certified_feasible`。

**Step 3: 运行失败测试**

Run: `python -m pytest tests/certificates/test_q1.py -v`

Expected: FAIL。

**Step 4: 实现证书对象**

`Q1Certificate` 保存：

```python
status: CertificationStatus
theorem_id: str
premises: dict[str, float | str]
cover_duration_upper_bound_s: float
detection_duration_lower_bound_s: float
gap_lower_bound_s: float
human_readable_reason: str
```

仅当定理全部前提成立时才返回 `certified_infeasible`。证书不搜索投弹方案。

**Step 5: 验证并提交**

Run: `python -m pytest tests/certificates/test_q1.py -v`

Expected: 全部通过。

```bash
git add src/smoke_defense/certificates tests/certificates/test_q1.py
git commit -m "feat: certify q1 single-smoke infeasibility"
```

---

## Stage 5：路径与移动参考约束

### Task 6：实现连续分段直线路径及 A-020/安全距离精确检查

**Files:**
- Create: `src/smoke_defense/paths.py`
- Create: `src/smoke_defense/path_constraints.py`
- Create: `tests/test_paths.py`
- Create: `tests/test_path_constraints.py`

**Step 1: 写起飞等待和固定飞行速度测试**

构造 UAV 在 `t=5 s` 从运动舰船起飞的两段路径，验证：

- `t<5` 时不生成空中轨迹；
- 起点由 `shipborne_launch_position` 计算；
- 每个飞行段速度模为 `28 m/s`；
- 相邻段首尾位置连续；
- 任何零速度飞行段被拒绝。

**Step 2: 写 A-020 端点证书测试**

```python
def test_operation_radius_maximum_is_at_segment_endpoint():
    segment = relative_linear_segment(
        start_s=0.0,
        end_s=10.0,
        relative_start_m=[11900.0, 0.0],
        relative_velocity_mps=[5.0, 20.0],
    )
    certificate = certify_operation_radius(segment, radius_m=12000.0)
    assert certificate.checked_times_s == (0.0, 10.0)
    assert certificate.max_distance_m == pytest.approx(
        max(segment.distance_at(0.0), segment.distance_at(10.0))
    )
```

再构造一个端点均在 12000 m 内和一个终点超界的场景，分别返回可行和不可行。

**Step 3: 写安全距离内部最小值测试**

构造两机相对位置
\(\boldsymbol r(t)=(-10+2t,1)\)，区间 `[0, 10]`。两端距离均大于 1 m，但 \(t=5\) 时最小距离为 1 m。验证检查器确实计算内部驻点：

\[
t^\star=\operatorname{clip}
\left(-\frac{\boldsymbol a^\mathsf T\boldsymbol b}
{\|\boldsymbol b\|^2},t_0,t_1\right).
\]

**Step 4: 运行失败测试**

Run: `python -m pytest tests/test_paths.py tests/test_path_constraints.py -v`

Expected: FAIL。

**Step 5: 实现精确分段检查**

`certify_operation_radius` 只检查每段端点；`certify_pairwise_separation` 检查端点和内部驻点。两个函数都返回证书对象和最坏时刻，不只返回布尔值。

**Step 6: 用 Hypothesis 做性质测试**

随机生成 100 组直线段，与高密度采样对照：

- 采样最大距离不得超过 A-020 证书最大值加容差；
- 解析最小安全距离不得大于采样最小值加容差。

Run: `python -m pytest tests/test_paths.py tests/test_path_constraints.py -v`

Expected: 全部通过。

**Step 7: 提交**

```bash
git add src/smoke_defense/paths.py src/smoke_defense/path_constraints.py tests/test_paths.py tests/test_path_constraints.py
git commit -m "feat: certify moving-reference path constraints"
```

---

## Stage 6：联合覆盖验证器

### Task 7：实现空间联合覆盖和连续时间严格无裸露认证

**Files:**
- Create: `src/smoke_defense/geometry.py`
- Create: `src/smoke_defense/coverage.py`
- Create: `src/smoke_defense/verifier.py`
- Create: `tests/test_geometry.py`
- Create: `tests/test_coverage.py`
- Create: `tests/test_verifier.py`

**Step 1: 写单烟幕解析边界测试**

```python
def test_single_smoke_full_cover_boundary_is_inclusive():
    result = single_disk_margin(
        ship_center_m=[0.0, 0.0],
        ship_radius_m=80.0,
        smoke_center_m=[40.0, 0.0],
        smoke_radius_m=120.0,
    )
    assert result.margin_m == pytest.approx(0.0)
    assert result.covered
```

必须验证错误判据 `distance < 120` 不会被使用：烟幕中心距舰船中心 100 m 时应失败，因为 \(100+80>120\)。

**Step 2: 写联合覆盖测试**

构造两个烟幕都不能单独覆盖舰船、但其并集可覆盖一个小型合成目标盘的场景，验证：

- “任一烟幕独立覆盖”返回假；
- 联合覆盖器返回真；
- 删除任一烟幕后返回假。

再构造一个接近边界且当前多边形精度无法判定的场景，必须返回 `indeterminate_at_tolerance`。

**Step 3: 写连续时间反例**

在两个采样端点均覆盖、区间中点裸露的合成场景中，简单端点采样会误判；验证器必须通过 Lipschitz 裕度不足触发细分并找到反例。

**Step 4: 运行失败测试**

Run: `python -m pytest tests/test_geometry.py tests/test_coverage.py tests/test_verifier.py -v`

Expected: FAIL。

**Step 5: 实现空间内外逼近**

`JointDiskCoverChecker` 实现：

1. 舰船圆盘外接正多边形；
2. 烟幕圆盘内接正多边形；
3. 用 Shapely 做集合差；
4. 差集为空则认证覆盖；
5. 反向使用舰船内逼近和烟幕外逼近认证裸露；
6. 两者都不能确定时加倍多边形边数；
7. 达到上限仍不能确定则返回 `indeterminate_at_tolerance`。

单烟幕必须走解析公式，不经过多边形近似。

**Step 6: 实现连续时间验证**

`ContinuousCoverageVerifier.verify(plan, trajectory, timeline)`：

- 先按所有起爆、平台结束、失效、探测入口和命中事件切段；
- 单独检查所有闭端点；
- 在连续模式内部用自适应二分和保守 Lipschitz 界；
- 记录最坏覆盖裕度、最大裸露间隔、空间/时间容差；
- 只在整段获得证书时返回 `certified_feasible`。

**Step 7: 运行边界和性质测试**

Run: `python -m pytest tests/test_geometry.py tests/test_coverage.py tests/test_verifier.py -v`

Expected: 全部通过。

Run: `python -m pytest -q`

Expected: 0 failed。

**Step 8: 提交**

```bash
git add src/smoke_defense/geometry.py src/smoke_defense/coverage.py src/smoke_defense/verifier.py tests/test_geometry.py tests/test_coverage.py tests/test_verifier.py
git commit -m "feat: certify joint continuous-time smoke coverage"
```

---

## Stage 7：候选生成与 Q1→Q4 优化

### Task 8：实现 Q1 候选生成、解析剪枝和参数扫描

**Files:**
- Create: `src/smoke_defense/candidates.py`
- Create: `src/smoke_defense/q1.py`
- Create: `scripts/run_q1.py`
- Create: `tests/test_candidates.py`
- Create: `tests/test_q1.py`

**Step 1: 写“证书优先”测试**

```python
def test_q1_does_not_optimize_after_analytic_infeasibility(mocker):
    optimizer = mocker.Mock()
    result = solve_q1(full_window_scene(), optimizer=optimizer)
    assert result.status.value == "certified_infeasible"
    optimizer.assert_not_called()
```

**Step 2: 写候选可达性测试**

每个候选必须包含指令、起飞、投弹、起爆和失效时刻，以及完整 UAV 路径。对每个候选依次调用事件、载弹、投弹间隔、A-020 和联合覆盖验证器。

**Step 3: 运行失败测试**

Run: `python -m pytest tests/test_candidates.py tests/test_q1.py -v`

Expected: FAIL。

**Step 4: 实现两类输出**

- 若解析定理证明全程覆盖不可行：报告不可行证书，并继续计算“最大认证覆盖时长”作为次级结果；
- 若解析定理不能判定：生成时空候选，使用差分进化或网格精修，但最终只接受验证器认证结果。

排序采用词典序：

```text
认证状态
→ 裸露总时长
→ 最大连续裸露
→ 最小覆盖裕度（降序）
→ 鲁棒裕度（降序）
→ 航程
```

**Step 5: 执行 16 场景扫描**

Run: `python scripts/run_q1.py --scenario-dir configs/scenarios/q1_q3/generated`

Expected:

- `results/q1/scenario_results.csv`
- `results/q1/certificates/*.json`
- 16 个场景全部有结果；
- 完整经历探测窗口的场景不得被报告为单烟幕 100% 可行；
- 表中不出现“官方基准”字样。

**Step 6: 提交**

```bash
git add src/smoke_defense/candidates.py src/smoke_defense/q1.py scripts/run_q1.py tests/test_candidates.py tests/test_q1.py
git commit -m "feat: solve q1 with analytic-first candidate search"
```

### Task 9：在同一验证器上实现 Q2 多弹与 Q3 三机协同

**Files:**
- Create: `src/smoke_defense/q2.py`
- Create: `src/smoke_defense/q3.py`
- Create: `scripts/run_q2.py`
- Create: `scripts/run_q3.py`
- Create: `tests/test_q2.py`
- Create: `tests/test_q3.py`

**Step 1: 写 Q2 联合覆盖回归测试**

构造两个烟幕必须联合覆盖才能成功的场景。验证 Q2 调用 `ContinuousCoverageVerifier`，而不是合并每枚烟幕的“独立完整覆盖时间区间”。

**Step 2: 写 Q2 投弹约束测试**

反例必须分别返回明确约束名称：

- 同机相邻释放间隔小于 1 s；
- 单机用弹超过 3；
- 某一段违反 A-020；
- 指令到释放不足 2 s；
- 释放后 3.5 s 起爆位置不可由投弹航向产生。

**Step 3: 写 Q3 安全距离和失效容错测试**

验证：

- 每机至多 1 枚；
- 两机分段直线轨迹的内部最近点被检查；
- 删除任一枚烟幕后重新运行完整验证器；
- `single_failure_score` 等于所有删弹方案的最差结果。

**Step 4: 运行失败测试**

Run: `python -m pytest tests/test_q2.py tests/test_q3.py -v`

Expected: FAIL。

**Step 5: 实现分层候选选择**

Q2：

1. 从 Q1 时空候选扩展单机 1/2/3 弹；
2. 解析持续时间上界做组合剪枝；
3. 小规模枚举作为真值；
4. 正常规模用 SciPy MILP 选择候选；
5. 对入选组合运行联合验证器。

Q3：

1. 为三机分别生成路径—投弹候选；
2. 用解析安全距离证书筛选组合；
3. 首版使用离散候选的 Pareto/词典序筛选，不立即引入 NSGA-II；
4. 对完整方案和三个单点失效方案分别认证。

**Step 6: 扫描安全距离**

Run: `python scripts/run_q3.py --safe-distances 50 100 200 500`

Expected:

- `results/q3/safe_distance_scan.csv`
- 若 100–200 m 出现结构变化，脚本自动生成细化点并标记 `adaptive_refinement=true`；
- 不把任何扫描值写成题面安全距离。

**Step 7: 验证并提交**

Run: `python -m pytest tests/test_q2.py tests/test_q3.py -v`

Expected: 全部通过。

```bash
git add src/smoke_defense/q2.py src/smoke_defense/q3.py scripts/run_q2.py scripts/run_q3.py tests/test_q2.py tests/test_q3.py
git commit -m "feat: optimize verified q2 and q3 smoke plans"
```

### Task 10：实现 Q4 三类场景、任务包接口和滚动调度

**Files:**
- Create: `configs/scenarios/q4/abundant.yaml`
- Create: `configs/scenarios/q4/critical.yaml`
- Create: `configs/scenarios/q4/shortage.yaml`
- Create: `src/smoke_defense/packages.py`
- Create: `src/smoke_defense/q4.py`
- Create: `scripts/run_q4.py`
- Create: `tests/test_packages.py`
- Create: `tests/test_q4.py`

**Step 1: 写场景规模和时间语义测试**

验证三场景导弹数分别为：

- abundant：3；
- critical：5–7；
- shortage：8–10。

每枚弹必须有 `appearance_time_s`；仿真结果另行计算 `detection_entry_time_s`，两字段不得互相覆盖。

**Step 2: 写任务包可信性测试**

`DefensePackage` 必须拒绝：

- 缺少验证证书；
- 证书状态为 `indeterminate_at_tolerance` 却声称成功；
- 场景哈希与当前威胁不匹配；
- 来自 Q2/Q3 的方案未经过联合覆盖验证。

**Step 3: 写调度约束测试**

验证 5 架 UAV、每机最多 3 枚、任务时间不重叠、动态安全距离、资源占用和新批次到达后的滚动重算。

**Step 4: 运行失败测试**

Run: `python -m pytest tests/test_packages.py tests/test_q4.py -v`

Expected: FAIL。

**Step 5: 实现任务包和滚动 MILP**

任务包至少保存：

```python
package_id: str
applicable_scenario_hashes: tuple[str, ...]
uav_count: int
bomb_count: int
availability_interval_s: tuple[float, float]
terminal_states: dict[str, object]
coverage_certificate_id: str
robustness_tier: str
flight_distance_m: float
```

目标使用题面优先级的词典序近似：

1. 最大化高威胁的认证防御收益；
2. 最大化已认证威胁数；
3. 最小化最大未防御风险；
4. 最小化用弹量；
5. 最小化航程。

不得使用无数据依据的熵权法。

**Step 6: 运行三类工况**

Run:

```bash
python scripts/run_q4.py --scenario configs/scenarios/q4/abundant.yaml
python scripts/run_q4.py --scenario configs/scenarios/q4/critical.yaml
python scripts/run_q4.py --scenario configs/scenarios/q4/shortage.yaml
```

Expected:

- 三个结果均不超过 5 机、15 弹；
- abundant 输出全部威胁的分配或明确不可行证书；
- critical 显示任务包竞争和优先级；
- shortage 明确列出未防御风险，不伪造全覆盖；
- 每个已分配任务包通过独立时空复核。

**Step 7: 提交**

```bash
git add configs/scenarios/q4 src/smoke_defense/packages.py src/smoke_defense/q4.py scripts/run_q4.py tests/test_packages.py tests/test_q4.py
git commit -m "feat: schedule verified q4 defense packages"
```

### Task 11：实现有界鲁棒验证、报告绑定和端到端验收

**Files:**
- Create: `src/smoke_defense/robustness.py`
- Create: `src/smoke_defense/reporting.py`
- Create: `scripts/run_all.py`
- Create: `scripts/check_result_provenance.py`
- Create: `tests/test_robustness.py`
- Create: `tests/test_reporting.py`
- Create: `docs/final-review-checklist.md`
- Modify: `README.md`

**Step 1: 写鲁棒缩水测试**

```python
def test_medium_tier_effective_radius_in_hold_stage():
    radius = robust_effective_radius(
        nominal_radius_m=120.0,
        smoke_age_s=10.0,
        delay_error_s=0.3,
        position_error_m=10.0,
        radius_error_m=6.0,
        wind_speed_bound_mps=3.0,
        radius_slope_bound_mps=0.0,
    )
    assert radius == pytest.approx(120.0 - 10.0 - 6.0 - 30.0)
```

另测试衰减阶段加入 \(24\varepsilon_t\)，结果不得小于 0。

**Step 2: 写溯源和数字单一来源测试**

由同一个结果对象生成 JSON、CSV、LaTeX 宏和图注，验证关键覆盖率和时刻一致；缺少场景哈希、Git SHA 或认证状态时，报告生成必须失败。

**Step 3: 运行失败测试**

Run: `python -m pytest tests/test_robustness.py tests/test_reporting.py -v`

Expected: FAIL。

**Step 4: 实现三档误差盒验证**

对名义候选依次执行 `light`、`medium`、`strong`：

1. 使用保守有效半径快速筛选；
2. 枚举时间、位置、半径和风漂误差盒的关键顶点；
3. 把事件时间误差加入时间线；
4. 对每个关键场景调用完整验证器；
5. 输出最强可认证等级和第一个失效反例。

这些结果只能称为场景误差等级，不称为真实装备可靠性。

**Step 5: 实现统一运行入口**

`run_all.py` 顺序执行：

1. 校验常数和场景；
2. 生成 16 个 Q1–Q3 场景；
3. 生成公共威胁轨迹和事件；
4. Q1 解析证书与次级优化；
5. Q2；
6. Q3 安全距离扫描；
7. Q4 三类工况；
8. 三档鲁棒复核；
9. 汇总结果和论文图表。

每一步失败必须以非零状态退出，不写入伪成功摘要。

**Step 6: 完成全量验收**

Run: `python -m ruff check src tests scripts`

Expected: `All checks passed!`

Run: `python -m pytest -q`

Expected: 0 failed。

Run: `python scripts/run_all.py --seed 20260730`

Expected:

- `results/summary.json`
- `results/q1/` 至 `results/q4/`
- `figures/final/`
- 所有结果具有完整溯源；
- 相同提交、配置和随机种子重复运行得到相同离散决策及容差内相同数值。

Run: `python scripts/check_result_provenance.py results`

Expected: `0 missing or inconsistent provenance fields`。

**Step 7: 红队复核**

`docs/final-review-checklist.md` 至少回答：

1. 是否有模块绕过 A-001–A-020？
2. 是否把某个场景称为官方数据？
3. `appearance_time_s` 是否与探测入口混淆？
4. Q1 是否在解析不可行后仍声称 100% 成功？
5. Q2/Q3 是否真正检查联合空间覆盖？
6. A-020 是否按移动舰船逐段认证？
7. 安全距离是否检查了内部最近点？
8. 鲁棒误差是否被误称为真实设备精度？
9. Q4 是否只消费已验证任务包？
10. 报告数字是否都来自同一结果对象？

**Step 8: 更新 README 并提交**

README 链接最新设计、最新计划、场景生成命令、测试命令和结果状态说明。旧计划保留为历史文档，但明确标记为已被本计划替代。

```bash
git add src/smoke_defense/robustness.py src/smoke_defense/reporting.py scripts/run_all.py scripts/check_result_provenance.py tests/test_robustness.py tests/test_reporting.py docs/final-review-checklist.md README.md
git commit -m "feat: verify robust reproducible modeling pipeline"
```

---

## 暂缓任务：PN 消融

只有首版全部测试通过且团队单独批准 X-001/X-002 后，才创建后续计划。届时仅扫描：

\[
N\in\{3,4,5\},\qquad a_{\max}\in\{5g,10g,20g\}.
\]

该扩展必须保留纯追踪结果，使用同一探测和覆盖验证器，并在输出中标记 `model_layer: ablation`。本实施计划不包含 PN 代码。

## 完成定义

- [ ] A-001–A-020 在场景、代码和测试中可追溯；
- [ ] 16 个 Q1–Q3 参数场景和三类 Q4 场景均通过 schema；
- [ ] 混合事件时间轴不依赖网格舍入；
- [ ] 纯追踪解析边界与数值积分一致；
- [ ] Q1 不可行性证书通过回归测试；
- [ ] A-020 使用分段端点精确认证；
- [ ] 多机安全距离检查内部驻点；
- [ ] 多烟幕使用联合空间覆盖；
- [ ] 连续时间无裸露得到证书而非采样猜测；
- [ ] Q1–Q4 只消费公共验证器；
- [ ] 三档误差只作为场景鲁棒等级；
- [ ] 所有结果具有配置哈希、假设 ID、Git SHA、随机种子和认证状态；
- [ ] Ruff、pytest、全量流水线和溯源检查全部通过。

