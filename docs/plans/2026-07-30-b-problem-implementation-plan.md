# B题烟幕防御建模 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不改变 A-001–A-020 的前提下，实现参数化场景、连续时间约束证书、联合覆盖验证器以及依次复用这些能力的 Q1–Q4 求解流水线。

**Architecture:** 按能力依赖而不是按题号拆分代码。场景契约和混合事件时间线位于最底层；共享动力学、解析证书、路径约束和联合覆盖验证器构成唯一可信计算内核；Q1–Q4 只生成候选并调用验证器。所有结论保存场景哈希、假设版本、Git SHA、随机种子和三值认证状态。

**Tech Stack:** Python 3.12、NumPy、SciPy、Pydantic、PyYAML、Shapely、pandas、pytest、Hypothesis、Ruff、Matplotlib。

---

## 团队分工与本计划适用范围

团队人员分工固定如下：

- **建模同学**：负责假设、公式、约束、解析证书、代码交接和结果验收；
- **论文同学**：负责整篇论文、符号统一、图表叙述和论点—证据一致性；
- **仿真代码同学**：负责整套仿真代码、测试、场景运行、结果文件和绘图。

本 Implementation Plan 是交给仿真代码同学的完整开发顺序。下文 Task 1–Task 11 不是把代码拆给三个人，也不改变上述人员分工。建模同学在模型接口和结果验收节点参与，论文同学在公式冻结和结果通过验收后消费产物，但两人不分别接管代码模块。

三条工作线通过以下交接物对齐：

| 交接方向 | 交接物 | 接收条件 |
|---|---|---|
| 建模 → 仿真代码 | 模型契约、A-001–A-020、场景字段、公式、成功判据和反例测试 | 代码同学能把每个公式对应到输入、函数和测试 |
| 仿真代码 → 建模 | JSON/CSV 结果、配置哈希、Git SHA、随机种子和认证状态 | 建模同学完成解析边界、约束和连续时间覆盖验收 |
| 建模 → 论文 | 标记为 `verified_for_paper` 的结果清单与物理解释 | 每个数字都能定位到结果文件和场景 ID |
| 仿真代码 → 论文 | 自动生成图表、表格和图注元数据 | 图表与已验收结果使用相同配置哈希 |
| 论文 → 全组 | 论点—公式—结果—图表映射表 | 不存在无来源公式、数字或结论 |

## PR 边界与第一批实现验收门

当前 PR #2 只固化实现架构和 Implementation Plan，不创建任何 Python 源码、场景生成物或仿真结果，也不执行下文 Task 1–Task 11。

后续第一批实现 PR 只允许执行：

1. Task 1：场景契约；
2. Task 2：场景矩阵与溯源；
3. Task 3：混合事件时间线和最早烟幕形成公共证书。

第一批完成后必须单独验收：

1. `constants.yaml` 是题面公共常数唯一事实源，场景无重复常数；
2. Pydantic 与仓库中导出的 JSON Schema 完全一致；
3. 非零 `appearance_time_s` 的导弹坐标相对出现时刻舰船解释正确；
4. 2 s 被实现为最短响应时间，而不是固定响应时间；
5. 事件顺序、事件点和闭区间语义明确；
6. 场景哈希对 YAML 键顺序稳定，并覆盖常数版本和显式覆盖；
7. PN 配置不能启用；
8. 第一批全部测试通过。

上述接口未完成验收前，不得开始 Task 4–Task 11。执行者不得一次性执行全部 11 个任务。

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
Task 3 混合事件时间线与烟幕可用性证书
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
- Create: `scripts/export_scenario_schema.py`
- Create: `docs/ai-task-template.md`
- Create: `tests/test_scenario.py`

**Step 1: 写场景、常数和坐标语义的失败测试**

```python
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from smoke_defense.scenario import (
    Scenario,
    ScenarioError,
    load_problem_constants,
    load_scenario,
)


EXAMPLE = Path("configs/scenarios/examples/q1_front_d10000_nominal.yaml")
REPOSITORY_SCHEMA = Path("configs/schema/scenario.schema.json")


def valid_scenario_dict():
    return yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))


def test_example_scenario_has_frozen_semantics():
    scene = load_scenario(EXAMPLE)
    assert scene.schema_version == "1.0"
    assert scene.constants_version == "b-problem-v1"
    assert scene.time_origin == "decision_start"
    assert scene.missiles[0].guidance_model == "pure_pursuit"
    assert scene.constants.uav.operation_radius_m == pytest.approx(12000.0)
    assert {"A-019", "A-020"} <= set(scene.assumption_ids)


def test_unknown_field_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        EXAMPLE.read_text(encoding="utf-8") + "\nunknown_knob: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ScenarioError, match="unknown_knob"):
        load_scenario(bad)


def test_nonzero_appearance_position_is_relative_to_ship_at_appearance():
    raw = valid_scenario_dict()
    raw["ship"] = {
        "initial_position_world_m": [0.0, 0.0],
        "heading_deg": 0.0,
    }
    raw["missiles"][0].update({
        "appearance_time_s": 20.0,
        "initial_position_at_appearance_body_m": [1000.0, 200.0],
    })
    scene = Scenario.model_validate(raw).normalize(load_problem_constants())
    assert scene.missiles[0].initial_position_world_m == pytest.approx(
        [7.71 * 20.0 + 1000.0, 200.0]
    )


def test_world_and_body_position_fields_are_mutually_exclusive():
    raw = valid_scenario_dict()
    raw["missiles"][0]["initial_position_world_m"] = [1000.0, 200.0]
    with pytest.raises(ValidationError, match="exactly one"):
        Scenario.model_validate(raw)


def test_wrong_unit_field_is_rejected():
    raw = valid_scenario_dict()
    raw["uavs"][0]["available_time_ms"] = 1000.0
    with pytest.raises(ValidationError, match="available_time_ms"):
        Scenario.model_validate(raw)


def test_speed_override_requires_source():
    raw = valid_scenario_dict()
    raw["missiles"][0]["speed_override_mps"] = 300.0
    with pytest.raises(ValidationError, match="speed_source"):
        Scenario.model_validate(raw)


def test_constants_are_not_duplicated_in_scenario():
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    assert raw["constants_version"] == "b-problem-v1"
    assert "bomb" not in raw
    assert "smoke" not in raw
    assert not {"speed_mps", "equivalent_radius_m"} & raw["ship"].keys()
    assert all(
        not {"flight_speed_mps", "operation_radius_m", "max_payload"}
        & uav.keys()
        for uav in raw["uavs"]
    )
    assert all(
        not {"speed_mps", "detection_range_m", "field_of_view_half_angle_deg"}
        & missile.keys()
        for missile in raw["missiles"]
    )
    assert not {
        "minimum_release_response_s",
        "inertial_flight_s",
        "minimum_release_interval_s",
    } & raw["constraints"].keys()


def test_exported_json_schema_matches_repository_schema():
    exported = json.dumps(
        Scenario.model_json_schema(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    assert REPOSITORY_SCHEMA.read_text(encoding="utf-8") == exported
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

在 `scenario.py` 使用 Pydantic 定义 `ProblemConstants`、`ShipSpec`、`UavSpec`、`MissileSpec`、`ConstraintSpec`、`UncertaintySpec` 和 `Scenario`。所有模型及嵌套模型统一设置 `ConfigDict(extra="forbid")`。载入顺序固定为：

1. YAML 解析；
2. Pydantic 结构与语义校验；
3. 通过 `Scenario.model_json_schema()` 导出或核对 JSON Schema；
4. 检查基准场景只允许 `pure_pursuit` 和 `velocity_aligned`；
5. 检查所有速度、半径和时间非负；
6. 按 `constants_version` 载入题面常数；
7. 生成只读规范化场景对象。

Pydantic 是场景 Schema 的唯一源码。`scripts/export_scenario_schema.py` 只调用 `Scenario.model_json_schema()` 并以固定排序、缩进和 UTF-8 换行写出 `configs/schema/scenario.schema.json`；`--check` 模式只比较、不写文件。禁止人工独立修改生成的 Schema。未知字段、错误单位字段、缺少位置字段、两个位置字段同时出现、速度覆盖缺少 `speed_source` 等情况全部拒绝。

每枚导弹的位置字段必须且只能二选一：

```python
initial_position_at_appearance_body_m: tuple[float, float] | None
initial_position_world_m: tuple[float, float] | None
```

若使用体坐标，规范化时按

\[
\boldsymbol m_i(t_i^{\rm app})
=
\boldsymbol s(t_i^{\rm app})
+Q(\theta_s)\boldsymbol r_i^{\rm app}
\]

转换为世界坐标。导弹在 `appearance_time_s` 之前不存在，并从该时刻开始积分。
若使用 `initial_position_world_m`，该字段也表示
\(\boldsymbol m_i(t_i^{\rm app})\)，不是 \(t=0\) 时不存在导弹的外推位置。

`constants.yaml` 按设计文档保存舰船、导弹、无人机、干扰弹和烟幕公共常数；场景只写 `constants_version`。Q4 导弹速度只通过成对字段 `speed_override_mps` 与 `speed_source` 覆盖，禁止修改常数文件。规范化哈希必须包含常数内容哈希、覆盖值和覆盖来源。

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

Expected: 全部通过。

Run: `python scripts/export_scenario_schema.py --check`

Expected: `scenario.schema.json is up to date`。

Run: `python -m ruff check src tests scripts`

Expected: `All checks passed!`

**Step 6: 提交**

```bash
git add pyproject.toml src/smoke_defense configs/constants.yaml configs/schema configs/scenarios/examples scripts/export_scenario_schema.py docs/ai-task-template.md tests/test_scenario.py
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
    np.testing.assert_allclose(
        oblique.missiles[0].initial_position_at_appearance_body_m,
        expected,
    )
```

再测试安全距离恰为 `{50, 100, 200, 500}`；三档场景误差分别包含 `release_response_error_s`、`detonation_delay_error_s`、`smoke_center_error_m`、`smoke_radius_error_m` 和 `wind_speed_bound_mps`；`pn_ablation.yaml` 默认 `enabled: false` 且场景载入器拒绝启用。

**Step 2: 写配置哈希稳定性测试**

同一规范化场景即使 YAML 键顺序不同，也必须产生相同 SHA-256；任何场景数值、`constants_version`、常数内容、`speed_override_mps` 或 `speed_source` 改变都必须改变哈希。缺少来源的速度覆盖必须在计算哈希前被 Pydantic 拒绝。

**Step 3: 运行测试，确认失败**

Run: `python -m pytest tests/test_scenario_matrix.py tests/test_provenance.py -v`

Expected: FAIL，缺少生成器和哈希函数。

**Step 4: 实现场景生成和溯源**

`Provenance` 至少包含：

```python
scenario_id: str
scenario_sha256: str
constants_sha256: str
assumption_ids: tuple[str, ...]
git_sha: str
random_seed: int | None
generated_at_utc: str
```

生成文件写入 `configs/scenarios/q1_q3/generated/`。脚本必须先在临时目录生成，再逐个通过 `load_scenario` 的 Pydantic 管线，全部合法后才替换生成目录。

**Step 5: 验证输出**

Run: `python scripts/generate_scenarios.py`

Expected: 输出 16 个 YAML，所有文件通过 Pydantic 校验，且生成的 JSON Schema 与仓库文件一致。

Run: `python -m pytest tests/test_scenario_matrix.py tests/test_provenance.py -v`

Expected: 全部通过。

**Step 6: 提交**

```bash
git add configs/sweeps configs/scenarios/q1_q3 src/smoke_defense/scenario_matrix.py src/smoke_defense/provenance.py scripts/generate_scenarios.py tests/test_scenario_matrix.py tests/test_provenance.py
git commit -m "feat: generate traceable parameterized scenarios"
```

---

## Stage 2：混合事件时间线

### Task 3：实现事件时间线和最早烟幕形成公共证书

**Files:**
- Create: `src/smoke_defense/events.py`
- Create: `src/smoke_defense/timeline.py`
- Create: `src/smoke_defense/certificates/__init__.py`
- Create: `src/smoke_defense/certificates/availability.py`
- Create: `tests/test_timeline.py`
- Create: `tests/certificates/test_availability.py`

**Step 1: 写“2 s 是最短响应时间”的事件测试**

```python
import pytest

from smoke_defense.events import (
    BombEvents,
    EventOrderError,
    earliest_release_time,
)


def test_release_later_than_minimum_response_is_allowed():
    events = BombEvents(
        command_time_s=4.0,
        release_time_s=7.3,
        burst_time_s=10.8,
    )
    assert earliest_release_time(4.0) == pytest.approx(6.0)
    assert events.release_time_s == pytest.approx(7.3)
    assert events.burst_time_s == pytest.approx(10.8)
    assert events.hold_end_time_s == pytest.approx(28.8)
    assert events.expire_time_s == pytest.approx(33.8)


def test_release_before_minimum_response_is_rejected():
    with pytest.raises(EventOrderError):
        BombEvents(
            command_time_s=4.0,
            release_time_s=5.9,
            burst_time_s=9.4,
        )


def test_nominal_detonation_delay_must_equal_inertial_flight_time():
    with pytest.raises(EventOrderError):
        BombEvents(
            command_time_s=4.0,
            release_time_s=7.3,
            burst_time_s=10.7,
        )
```

`BombEvents` 不得提供从指令时刻自动生成实际释放时刻的构造器。`earliest_release_time(command_time_s)` 只计算可行域下界，不得写入候选方案的实际 `release_time_s`。

**Step 2: 写切段和闭端点测试**

输入事件 `[0, 3.5, 21.5, 26.5]` 和分析窗口 `[0, 30]`，预期连续区间为：

```python
[(0.0, 3.5), (3.5, 21.5), (21.5, 26.5), (26.5, 30.0)]
```

重复事件必须去重，但事件点仍单独出现在 `event_times` 中。

**Step 3: 写最早烟幕形成证书测试**

```python
import pytest

from smoke_defense.certificates.availability import (
    certify_earliest_smoke_availability,
)


def test_earliest_smoke_certificate_detects_unavoidable_initial_exposure():
    certificate = certify_earliest_smoke_availability(
        command_time_s=0.0,
        minimum_response_s=2.0,
        inertial_flight_s=3.5,
        minimum_maneuver_time_s=0.0,
        detection_entry_time_s=0.0,
        no_predeployed_smoke=True,
        full_detection_window_required=True,
    )
    assert certificate.earliest_burst_time_s == pytest.approx(5.5)
    assert certificate.unavoidably_exposed_duration_s == pytest.approx(5.5)
    assert certificate.status.value == "certified_infeasible"
```

再增加以下反例：

- 若 `detection_entry_time_s >= earliest_burst_time_s`，该定理不能据此宣布成功，只能返回 `indeterminate_at_tolerance`；
- 若允许预部署烟幕，定理前提不成立；
- 若 `minimum_maneuver_time_s > 0`，最早起爆时刻必须相应后移；
- \(d_0=8000\rm\,m\) 且 \(t=0\) 已进入探测区时，Q1、Q2、Q3 的严格全窗口目标均读取同一证书，不能报告 100% 防御，但次级优化入口仍保持可调用。

**Step 4: 运行失败测试**

Run: `python -m pytest tests/test_timeline.py tests/certificates/test_availability.py -v`

Expected: FAIL。

**Step 5: 实现事件与时间线**

使用有序不可变 dataclass 表示事件。`HybridTimeline` 提供：

- `add_event(event)`
- `continuous_intervals(start_s, end_s)`
- `events_at(time_s)`
- `validate_causality()`

不得把事件时刻舍入到固定时间步。

`BombEvents` 显式接收 `command_time_s`、`release_time_s` 和 `burst_time_s`。名义场景校验
\(t^d-t^c\ge2\) 和 \(t^e-t^d=3.5\)；鲁棒场景按独立的 `release_response_error_s` 与 `detonation_delay_error_s` 检查相应区间。平台结束和失效时刻由起爆时刻及 `constants.yaml` 派生。

**Step 6: 实现公共解析证书**

`certify_earliest_smoke_availability(...)` 放在公共 `certificates/availability.py`，由 Q1–Q3 共用。证书对象保存：

```python
status: CertificationStatus
theorem_id: str
command_time_s: float
minimum_response_s: float
inertial_flight_s: float
minimum_maneuver_time_s: float
earliest_burst_time_s: float
detection_entry_time_s: float
unavoidably_exposed_duration_s: float
premises: dict
human_readable_reason: str
```

计算：

\[
t_{\rm smoke}^{\min}
=t^c+2+3.5+T_{\rm maneuver},
\qquad
T_{\rm unavoidable}
=\max(0,t_{\rm smoke}^{\min}-t_{\rm detection}^{\rm entry}).
\]

只有在无预部署烟幕、所有投放均发生在任务开始后、严格要求全探测窗口无裸露且 \(T_{\rm unavoidable}>0\) 时，才返回 `certified_infeasible`。该状态只否定严格目标，调用方仍继续求最大覆盖时长、最小裸露时间和最佳投放方案。

**Step 7: 验证并提交**

Run: `python -m pytest tests/test_timeline.py tests/certificates/test_availability.py -v`

Expected: 全部通过。

```bash
git add src/smoke_defense/events.py src/smoke_defense/timeline.py src/smoke_defense/certificates tests/test_timeline.py tests/certificates/test_availability.py
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
- Modify: `src/smoke_defense/certificates/__init__.py`
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
def test_q1_skips_strict_search_but_continues_secondary_objective(mocker):
    strict_optimizer = mocker.Mock()
    secondary_optimizer = mocker.Mock(return_value=secondary_candidate())
    result = solve_q1(
        full_window_scene(),
        strict_optimizer=strict_optimizer,
        secondary_optimizer=secondary_optimizer,
    )
    assert result.strict_full_window_status.value == "certified_infeasible"
    strict_optimizer.assert_not_called()
    secondary_optimizer.assert_called_once()
```

**Step 2: 写候选可达性测试**

每个候选必须包含指令、实际投弹、起爆和失效时刻，以及完整 UAV 路径。实际投弹时刻可以晚于最早响应时刻。对每个候选依次调用事件、载弹、投弹间隔、A-020 和联合覆盖验证器。

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

再对 \(d_0=8000\rm\,m\)、\(t=0\) 已进入探测区、无预部署烟幕的场景，验证 Q2 和 Q3 都调用 `certify_earliest_smoke_availability`：严格全窗口状态为 `certified_infeasible`，但次级目标求解仍继续。

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
- Create: `configs/scenarios/q4/q4_case_3_threats.yaml`
- Create: `configs/scenarios/q4/q4_case_6_threats.yaml`
- Create: `configs/scenarios/q4/q4_case_9_threats.yaml`
- Create: `src/smoke_defense/packages.py`
- Create: `src/smoke_defense/q4.py`
- Create: `scripts/run_q4.py`
- Create: `tests/test_packages.py`
- Create: `tests/test_q4.py`

**Step 1: 写场景规模和时间语义测试**

验证三个中性场景的导弹数分别为 3、6、9。每枚弹必须有互相独立的：

- `reveal_time_s`；
- `appearance_time_s`；
- `initial_position_at_appearance_body_m` 或 `initial_position_world_m`；
- 仿真结果派生的 `detection_entry_time_s`。

**Step 2: 写任务包可信性测试**

`DefensePackage` 必须拒绝：

- 缺少验证证书；
- 证书状态为 `indeterminate_at_tolerance` 却声称成功；
- 场景哈希与当前威胁不匹配；
- 来自 Q2/Q3 的方案未经过联合覆盖验证。

**Step 3: 写信息揭示约束测试**

```python
def test_q4_online_scheduler_cannot_see_unrevealed_missile():
    scene = q4_scene(
        information_mode="revealed_at_appearance",
        missiles=[
            threat("M1", reveal_time_s=0.0, appearance_time_s=0.0),
            threat("M2", reveal_time_s=20.0, appearance_time_s=20.0),
        ],
    )
    decision = RollingScheduler().plan_at(time_s=10.0, scene=scene)
    assert decision.known_missile_ids == ("M1",)
    assert "M2" not in decision.assigned_missile_ids
    assert "M2" not in decision.prepositioned_for_missile_ids
```

再验证同一场景在 `offline_full_information` 模式下可以读取全部威胁，但结果必须标记为离线性能上界，不能标记为滚动在线结果。在线模式校验 `reveal_time_s == appearance_time_s`。

**Step 4: 写调度约束和结果分类测试**

验证 5 架 UAV、每机最多 3 枚、任务时间不重叠、动态安全距离、资源占用和新批次到达后的滚动重算。

```python
def test_q4_resource_label_is_derived_from_result_not_filename():
    result = synthetic_defense_result(
        scenario_path="configs/scenarios/q4/q4_case_3_threats.yaml",
        all_threats_certified=False,
        unavoidable_exposed_threat_ids=("M3",),
        used_uavs=5,
        available_uavs=5,
        used_bombs=15,
        available_bombs=15,
    )
    assert classify_resource_regime(result).value == "resource_shortage"
```

再分别构造“全部认证且有冗余”和“全部或高优先级认证但接近资源上限”的结果，预期分类为 `resource_abundant` 和 `resource_critical`。分类器不得接收文件名作为决策变量。

**Step 5: 运行失败测试**

Run: `python -m pytest tests/test_packages.py tests/test_q4.py -v`

Expected: FAIL。

**Step 6: 实现任务包、信息过滤和滚动 MILP**

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

定义：

\[
\mathcal I_{\rm known}(t)
=\left\{i:t_i^{\rm reveal}\le t\right\}.
\]

`revealed_at_appearance` 模式下，调度器在时刻 \(t\) 只能接收
\(\mathcal I_{\rm known}(t)\) 的只读威胁快照；未揭示威胁不能进入候选任务包、无人机预调动或精确投弹位置预留。`offline_full_information` 使用同一组威胁作为全知上界。

目标使用题面优先级的词典序近似：

1. 最大化高威胁的认证防御收益；
2. 最大化已认证威胁数；
3. 最小化最大未防御风险；
4. 最小化用弹量；
5. 最小化航程。

不得使用无数据依据的熵权法。

结果分类只能由求解和认证结果产生：

- 全部威胁可认证防御且存在冗余资源 → `resource_abundant`；
- 全部或高优先级威胁仅在接近资源上限时可认证防御 → `resource_critical`；
- 即使用尽资源仍存在不可避免裸露威胁 → `resource_shortage`。

分类按 `resource_shortage`、`resource_critical`、`resource_abundant` 的优先级判定。“接近上限”由容量松弛为零或删减一单位资源后认证结论恶化证明；“存在冗余”由正容量松弛或删减资源后结论不变证明。分类输入只包含认证结果、资源容量、资源占用和删减复核结果。

每个中性场景分别运行 `offline_full_information` 与 `revealed_at_appearance`。对同一可比收益标量 \(J\)，报告
\(\Delta_{\rm info}=J_{\rm offline}-J_{\rm online}\)；若使用词典序向量，则逐项报告差异。

**Step 7: 运行三类工况**

Run:

```bash
python scripts/run_q4.py --scenario configs/scenarios/q4/q4_case_3_threats.yaml --compare-information-modes
python scripts/run_q4.py --scenario configs/scenarios/q4/q4_case_6_threats.yaml --compare-information-modes
python scripts/run_q4.py --scenario configs/scenarios/q4/q4_case_9_threats.yaml --compare-information-modes
```

Expected:

- 六个模式结果均不超过 5 机、15 弹；
- 每个结果写入由认证结果导出的资源类别；
- 在线结果只使用当时已揭示的导弹；
- 离线结果明确标记为性能上界；
- 无法全覆盖时列出未防御风险，不伪造全覆盖；
- 每个已分配任务包通过独立时空复核。

**Step 8: 提交**

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
        release_response_error_s=0.3,
        detonation_delay_error_s=0.3,
        smoke_center_error_m=10.0,
        smoke_radius_error_m=6.0,
        wind_speed_bound_mps=3.0,
        radius_slope_bound_mps=0.0,
    )
    assert radius == pytest.approx(120.0 - 10.0 - 6.0 - 30.0)
```

另测试衰减阶段加入
\(24(\varepsilon_{\rm response}+\varepsilon_{\rm detonation})\)，结果不得小于 0。两个时间误差必须分别出现在反例与报告中，不能重新合并成含义不明的单字段。

**Step 2: 写溯源和数字单一来源测试**

由同一个结果对象生成 JSON、CSV、LaTeX 宏和图注，验证关键覆盖率和时刻一致；缺少场景哈希、Git SHA 或认证状态时，报告生成必须失败。

**Step 3: 运行失败测试**

Run: `python -m pytest tests/test_robustness.py tests/test_reporting.py -v`

Expected: FAIL。

**Step 4: 实现三档误差盒验证**

对名义候选依次执行 `light`、`medium`、`strong`：

1. 使用保守有效半径快速筛选；
2. 分别枚举响应时间、起爆延时、烟幕中心、半径和风漂误差盒的关键顶点；
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
7. Q4 三个中性威胁规模及两种信息模式；
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
3. `reveal_time_s`、`appearance_time_s` 是否与探测入口混淆？
4. Q1 是否在解析不可行后仍声称 100% 成功？
5. Q2/Q3 是否真正检查联合空间覆盖？
6. A-020 是否按移动舰船逐段认证？
7. 安全距离是否检查了内部最近点？
8. 鲁棒误差是否被误称为真实设备精度？
9. Q4 在线调度是否读取了尚未揭示的导弹？
10. Q4 资源类别是否由结果认证而不是文件名决定？
11. Q4 是否只消费已验证任务包？
12. 报告数字是否都来自同一结果对象？

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
- [ ] `constants.yaml` 是题面公共常数唯一事实源，场景只引用 `constants_version`；
- [ ] Pydantic 是场景结构唯一源码，仓库 JSON Schema 与重新导出结果完全一致；
- [ ] 非零 `appearance_time_s` 的体坐标相对出现时刻舰船转换，且与世界坐标输入互斥；
- [ ] 16 个 Q1–Q3 参数场景和三个中性 Q4 威胁规模均通过 Pydantic 校验；
- [ ] 混合事件时间轴不依赖网格舍入；
- [ ] 2 s 是最短响应时间，实际释放允许晚于最早时刻；
- [ ] 最早烟幕形成公共证书约束 Q1–Q3 严格目标但不阻止次级优化；
- [ ] 纯追踪解析边界与数值积分一致；
- [ ] Q1 不可行性证书通过回归测试；
- [ ] A-020 使用分段端点精确认证；
- [ ] 多机安全距离检查内部驻点；
- [ ] 多烟幕使用联合空间覆盖；
- [ ] 连续时间无裸露得到证书而非采样猜测；
- [ ] Q1–Q4 只消费公共验证器；
- [ ] Q4 在线调度无法读取尚未揭示的导弹，离线全知结果仅作为上界；
- [ ] Q4 资源类别由求解后认证结果产生，不由场景文件名产生；
- [ ] 三档误差只作为场景鲁棒等级；
- [ ] 所有结果具有配置哈希、假设 ID、Git SHA、随机种子和认证状态；
- [ ] Ruff、pytest、全量流水线和溯源检查全部通过。
