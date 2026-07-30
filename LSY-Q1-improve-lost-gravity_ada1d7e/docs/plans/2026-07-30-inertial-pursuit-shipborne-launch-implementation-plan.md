# 惯性纯追踪与舰载起飞模型 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> 状态：**当前唯一实施计划**
> 上游设计：[惯性纯追踪与舰载起飞模型修订设计](2026-07-30-inertial-pursuit-shipborne-launch-design.md)
> 冻结契约：[统一模型契约 v0.2](../modeling/model_contract_v0.1.md)
> 假设基线：[A-001–A-022](../modeling/assumption_register.md)
> 过时计划：2026-07-29 两份计划及 2026-07-30 旧实现设计/计划均禁止执行。

**Goal:** 建立可复现的二维舰船烟幕防御求解系统，使 Q1–Q4 统一使用一阶航向惯性纯追踪、舰载 UAV 起飞、多烟幕联合覆盖和连续时间三值认证。

**Architecture:** 按“场景契约 → 导弹/舰船动力学 → 混合事件 → UAV 路径 → 覆盖验证器 → Q1–Q4”依赖顺序开发。优化器只生成候选，只有独立验证器可以返回 `certified_feasible`；9 组惯性参数和瞬时纯追踪消融共用同一验证接口。

**Tech Stack:** Python 3.11、NumPy、SciPy、Pydantic、PyYAML、Pandas、Matplotlib、SciPy Optimize、PuLP 或 SciPy MILP、Pytest、Ruff。

---

## 当前进度

> 2026-07-30 Q1 最终验收：Task 1–Task 7 已完成；Task 8–Task 11 尚未实施。
> 公共物理内核和 Q1 已完成，Q2–Q4 求解代码尚未实现。本状态更新不改变冻结题面事实、模型契约或 A-001–A-022。

- [x] Task 1：Python 工程和题面常数单一事实源
- [x] Task 2：强类型场景、惯性参数矩阵和溯源
- [x] Task 3：舰船和惯性纯追踪导弹动力学
- [x] Task 4：混合事件时间线和多分量探测集合
- [x] Task 5：舰载 UAV 路径和约束证书
- [x] Task 6：烟幕动力学、联合覆盖和连续时间认证
- [x] Task 7：Q1 分量级证书和单弹求解
- [ ] Task 8：Q2 单机多弹联合覆盖
- [ ] Task 9：Q3 三机三弹协同
- [ ] Task 10：Q4 任务包和滚动调度
- [ ] Task 11：鲁棒性、端到端报告和全项目验收

---

## 执行规则

1. 每个任务先写失败测试，再写最小实现。
2. 每个物理量显式带单位后缀；配置角度用 degree，内部统一 rad。
3. `configs/constants.yaml` 只保存题面常数，不保存 \(k\)、\(\omega_{\max}\)、坐标、安全距离或误差等级。
4. 正式场景只允许 `inertial_pure_pursuit`；瞬时纯追踪必须标记 `model_layer: ablation`。
5. 场景不得提供 UAV 自由初始坐标、初始航向或发射偏置。
6. 优化器状态不得冒充物理可行性证书。
7. 每完成一个任务运行定向测试、快速全量测试、`ruff check .` 和 `git diff --check` 后提交。

---

### Task 1（已完成）：建立 Python 工程和题面常数单一事实源

**Files:**
- Create: `pyproject.toml`
- Create: `src/smoke_defense/__init__.py`
- Create: `configs/constants.yaml`
- Create: `tests/test_constants.py`

**Step 1: 写失败测试**

```python
from smoke_defense.constants import load_problem_constants


def test_problem_constants_match_statement():
    c = load_problem_constants()
    assert c.ship.speed_mps == 7.71
    assert c.ship.effective_radius_m == 80.0
    assert c.uav.speed_mps == 28.0
    assert c.uav.operation_radius_m == 12000.0
    assert c.missile.nominal_speed_mps == 320.0
    assert c.missile.detection_range_m == 8000.0
    assert c.missile.field_of_view_half_angle_deg == 15.0
    assert c.countermeasure.release_response_min_s == 2.0
    assert c.countermeasure.detonation_delay_s == 3.5
```

**Step 2: 验证失败**

Run:

```powershell
python -m pytest tests/test_constants.py -v
```

Expected: FAIL，模块尚不存在。

**Step 3: 最小实现**

建立 `src` 布局和只读常数模型。`constants.yaml` 只写 F-101–F-116 的公共数值，不写惯性参数或任何场景坐标。

**Step 4: 验证**

```powershell
python -m pytest tests/test_constants.py -v
python -m ruff check .
```

Expected: PASS。

**Step 5: 提交**

```powershell
git add -- pyproject.toml src/smoke_defense/__init__.py src/smoke_defense/constants.py configs/constants.yaml tests/test_constants.py
git commit -m "build: add problem constants and python scaffold"
```

---

### Task 2（已完成）：建立强类型场景、惯性参数矩阵和溯源

**Files:**
- Create: `src/smoke_defense/scenario.py`
- Create: `src/smoke_defense/scenario_matrix.py`
- Create: `configs/scenarios/examples/q1_front_d10000_k1_w10.yaml`
- Create: `configs/sweeps/guidance.yaml`
- Create: `configs/schema/scenario.schema.json`
- Create: `scripts/export_scenario_schema.py`
- Create: `scripts/generate_scenarios.py`
- Create: `tests/test_scenario.py`
- Create: `tests/test_scenario_matrix.py`

**Step 1: 写场景语义失败测试**

```python
def test_formal_scene_uses_inertial_pursuit(scene):
    missile = scene.missiles[0]
    assert missile.guidance_model == "inertial_pure_pursuit"
    assert missile.heading_response_rate_per_s == 1.0
    assert missile.max_turn_rate_deg_s == 10.0


def test_formal_scene_rejects_instantaneous_pursuit(valid_dict):
    valid_dict["missiles"][0]["guidance_model"] = "instantaneous_pure_pursuit"
    with pytest.raises(ValidationError):
        Scenario.model_validate(valid_dict)


@pytest.mark.parametrize(
    "field",
    ["initial_position_world_m", "initial_heading_deg", "launch_offset_body_m"],
)
def test_uav_free_initial_state_is_rejected(valid_dict, field):
    valid_dict["uavs"][0][field] = [0.0, 0.0]
    with pytest.raises(ValidationError):
        Scenario.model_validate(valid_dict)
```

`initial_position_world_m` 在此测试中指 UAV 字段；导弹仍允许互斥的世界坐标或出现时刻舰船体坐标。

**Step 2: 写矩阵失败测试**

```python
def test_q1_q3_formal_matrix_has_144_scenes():
    scenes = generate_q1_q3_matrix()
    assert len(scenes) == 4 * 4 * 3 * 3
    assert all(s.model_layer == "formal" for s in scenes)


def test_ablation_matrix_has_16_scenes():
    scenes = generate_instantaneous_ablation_matrix()
    assert len(scenes) == 16
    assert all(s.model_layer == "ablation" for s in scenes)
```

**Step 3: 实现场景模型**

Pydantic 模型统一使用 `ConfigDict(extra="forbid", frozen=True)`。惯性参数必须为正；角速度上限不超过
\(180^\circ/\rm s\)。初始导弹航向由初始视线角派生，不设输入字段。

`configs/sweeps/guidance.yaml`：

```yaml
source: team_sensitivity_assumption
heading_response_rate_per_s: [0.5, 1.0, 2.0]
max_turn_rate_deg_s: [5.0, 10.0, 20.0]
reference:
  heading_response_rate_per_s: 1.0
  max_turn_rate_deg_s: 10.0
```

**Step 4: 实现规范化哈希**

哈希必须覆盖：

- 常数内容哈希；
- 场景规范化 JSON；
- `guidance_model` 和 `model_layer`；
- \(k\)、\(\omega_{\max}\)；
- 显式导弹速度覆盖及来源；
- 假设版本。

**Step 5: 验证**

```powershell
python scripts/export_scenario_schema.py
python scripts/export_scenario_schema.py --check
python scripts/generate_scenarios.py --check
python -m pytest tests/test_scenario.py tests/test_scenario_matrix.py -v
```

Expected: 144 个正式场景、16 个消融场景，全部通过 Schema。

**Step 6: 提交**

```powershell
git add -- src/smoke_defense/scenario.py src/smoke_defense/scenario_matrix.py configs scripts tests/test_scenario.py tests/test_scenario_matrix.py
git commit -m "feat: add inertial guidance scenario matrix"
```

---

### Task 3（已完成）：实现舰船和惯性纯追踪导弹动力学

**Files:**
- Create: `src/smoke_defense/angles.py`
- Create: `src/smoke_defense/dynamics.py`
- Create: `tests/test_angles.py`
- Create: `tests/test_dynamics.py`

**Step 1: 写角度和 ODE 失败测试**

```python
def test_wrap_to_pi_range():
    values = np.linspace(-20 * np.pi, 20 * np.pi, 1001)
    wrapped = np.array([wrap_to_pi(v) for v in values])
    assert np.all(wrapped > -np.pi)
    assert np.all(wrapped <= np.pi)


def test_missile_speed_magnitude_is_constant(inertial_state, spec):
    rhs = inertial_pursuit_rhs(0.0, inertial_state, ship_at, spec)
    assert np.linalg.norm(rhs[:2]) == pytest.approx(spec.speed_mps)


def test_heading_rate_is_clipped(inertial_state, spec):
    rhs = inertial_pursuit_rhs(0.0, inertial_state, offset_ship_at, spec)
    assert abs(rhs[2]) <= np.deg2rad(spec.max_turn_rate_deg_s)


def test_initial_hit_short_circuits_los():
    result = integrate_missile(initial_distance_m=80.0, ...)
    assert result.hit_time_s == 0.0
```

**Step 2: 实现状态方程**

```python
def inertial_pursuit_rhs(t, state, ship_position, spec):
    x_m, y_m, psi = state
    ship = ship_position(t)
    lam = np.arctan2(ship[1] - y_m, ship[0] - x_m)
    error = wrap_to_pi(lam - psi)
    omega = np.clip(
        spec.heading_response_rate_per_s * error,
        -spec.max_turn_rate_rad_s,
        spec.max_turn_rate_rad_s,
    )
    return np.array([
        spec.speed_mps * np.cos(psi),
        spec.speed_mps * np.sin(psi),
        omega,
    ])
```

同时实现瞬时纯追踪参考函数，但只能由消融入口调用。

**Step 3: 验证极限关系**

使用固定测试场景验证随 \(k,\omega_{\max}\) 增大，惯性轨迹到瞬时参考轨迹的最大位置误差下降。该测试只验证趋势，不把有限参数误写成严格数学极限。

**Step 4: 验证**

```powershell
python -m pytest tests/test_angles.py tests/test_dynamics.py -v
```

Expected: PASS。

**Step 5: 提交**

```powershell
git add -- src/smoke_defense/angles.py src/smoke_defense/dynamics.py tests/test_angles.py tests/test_dynamics.py
git commit -m "feat: add inertial pursuit dynamics"
```

---

### Task 4（已完成）：实现混合事件时间线和多分量探测集合

**Files:**
- Create: `src/smoke_defense/events.py`
- Create: `src/smoke_defense/detection.py`
- Create: `src/smoke_defense/timeline.py`
- Create: `tests/test_events.py`
- Create: `tests/test_detection.py`
- Create: `tests/test_timeline.py`

**Step 1: 写失败测试**

测试：

- `appearance_time_s` 之前导弹不存在；
- 距离首次进入 8000 m 产生事件；
- \(|\alpha|\) 穿过 \(15^\circ\) 产生视场入口/出口；
- 命中事件闭端点；
- 探测集合可返回两个不相连的闭区间；
- 指令、释放、起爆分别满足
  \(t^d\ge t^c+2\)、\(t^e=t^d+3.5\)；
- 不使用“数组索引 × 步长”代替事件时间。

**Step 2: 实现事件积分**

使用 `scipy.integrate.solve_ivp`，保留稠密输出。探测集合由距离条件和视场条件的交集构造，结果结构为：

```python
DetectionSet(
    components=(ClosedInterval(start_s=..., end_s=...), ...),
    source_events=(...),
)
```

**Step 3: 验证**

```powershell
python -m pytest tests/test_events.py tests/test_detection.py tests/test_timeline.py -v
```

Expected: PASS，至少一个合成场景产生两个探测分量。

**Step 4: 提交**

```powershell
git add -- src/smoke_defense/events.py src/smoke_defense/detection.py src/smoke_defense/timeline.py tests/test_events.py tests/test_detection.py tests/test_timeline.py
git commit -m "feat: add hybrid detection timeline"
```

---

### Task 5（已完成）：实现舰载 UAV 路径和约束证书

**Files:**
- Create: `src/smoke_defense/paths.py`
- Create: `src/smoke_defense/path_constraints.py`
- Create: `tests/test_paths.py`
- Create: `tests/test_path_constraints.py`

**Step 1: 写舰载起飞失败测试**

```python
def test_uav_waits_on_moving_ship_before_takeoff():
    path = make_path(takeoff_time_s=5.0, ...)
    assert path.position(3.0) == pytest.approx(ship.position(3.0))


def test_launch_position_equals_ship_at_takeoff():
    path = make_path(takeoff_time_s=5.0, ...)
    assert path.position(5.0) == pytest.approx(ship.position(5.0))


def test_all_airborne_segments_have_fixed_speed():
    assert all(abs(seg.speed_mps - 28.0) < 1e-12 for seg in path.segments)


def test_simultaneous_colocated_launches_fail_safe_distance():
    certificate = certify_pairwise_separation(path_a, path_b, safe_distance_m=100)
    assert certificate.status == "certified_infeasible"
```

**Step 2: 实现路径对象**

起飞前位置查询返回舰船实时位置；起飞后只允许连续分段直线。路径验证必须精确检查：

- 节点连续；
- 航段速度；
- 投弹点到达；
- 每段相对舰船作战半径的端点最大值；
- 两机相对线性运动的端点和内部最近点；
- 仍在舰上的 UAV 不参加空中间距检查。

**Step 3: 验证**

```powershell
python -m pytest tests/test_paths.py tests/test_path_constraints.py -v
```

Expected: PASS。

**Step 4: 提交**

```powershell
git add -- src/smoke_defense/paths.py src/smoke_defense/path_constraints.py tests/test_paths.py tests/test_path_constraints.py
git commit -m "feat: add shipborne uav path certificates"
```

---

### Task 6（已完成）：实现烟幕动力学、联合覆盖和连续时间认证

**Files:**
- Create: `src/smoke_defense/smoke.py`
- Create: `src/smoke_defense/coverage.py`
- Create: `src/smoke_defense/verification.py`
- Create: `tests/test_smoke.py`
- Create: `tests/test_coverage.py`
- Create: `tests/test_verification.py`

**Step 1: 写烟幕边界测试**

验证烟幕年龄 \(0,18,23\) s 的半径分别为
\(120,120,0\) m；投弹点到惯性起爆点距离为 98 m；起爆后中心固定。

**Step 2: 写覆盖测试**

```python
def test_single_smoke_uses_exact_disk_containment():
    assert single_smoke_gap(ship_center=[0, 0], smoke_center=[30, 0], radius_m=120) == -10


def test_two_smokes_can_cover_when_neither_covers_alone():
    result = certify_union_coverage(ship_disk, [left_smoke, right_smoke])
    assert result.status == "certified_feasible"
```

**Step 3: 实现三值认证**

单烟幕使用解析条件。多烟幕使用有界误差内外多边形逼近；连续时间按事件切段并结合 Lipschitz 裕度。只允许：

- `certified_feasible`；
- `certified_infeasible`；
- `indeterminate_at_tolerance`。

**Step 4: 验证**

```powershell
python -m pytest tests/test_smoke.py tests/test_coverage.py tests/test_verification.py -v
```

Expected: PASS，联合覆盖反例能区分“区间并集”和真实空间联合覆盖。

**Step 5: 提交**

```powershell
git add -- src/smoke_defense/smoke.py src/smoke_defense/coverage.py src/smoke_defense/verification.py tests/test_smoke.py tests/test_coverage.py tests/test_verification.py
git commit -m "feat: add certified smoke union coverage"
```

---

### Task 7（已完成）：实现 Q1 分量级证书和单弹求解

**Files:**
- Create: `src/smoke_defense/certificates/q1.py`
- Create: `src/smoke_defense/candidates.py`
- Create: `src/smoke_defense/q1.py`
- Create: `tests/certificates/test_q1.py`
- Create: `tests/test_candidates.py`
- Create: `tests/test_q1.py`

**Step 1: 写证书失败测试**

```python
def test_long_component_is_certified_infeasible():
    result = certify_single_smoke_duration([ClosedInterval(0.0, 11.0)])
    assert result.status == "certified_infeasible"


def test_short_components_are_not_certified_feasible():
    result = certify_single_smoke_duration(
        [ClosedInterval(0.0, 5.0), ClosedInterval(8.0, 13.0)]
    )
    assert result.status == "indeterminate_at_tolerance"
```

另测试首个烟幕形成前的探测交集会产生不可避免裸露证书。

**Step 2: 实现候选降维**

依次搜索：

1. 探测分量和覆盖中心时刻；
2. 靠近舰船航线的起爆中心；
3. 半径 98 m 投弹点圆；
4. 起飞时刻和 UAV 可达路径；
5. 低维局部精修。

**Step 3: 实现词典序目标**

严格成功优先；不可行时依次最大化完整覆盖时长、最小化最大裸露区间、最大化最小裕度、最小化航程。

**Step 4: 运行 9 组参数和消融**

中值方案必须在全部 9 组惯性参数下复核；未全部通过时标记
`parameter_sensitive`。瞬时纯追踪只生成消融差异。

**Step 5: 验证和提交**

```powershell
python -m pytest tests/certificates/test_q1.py tests/test_candidates.py tests/test_q1.py -v
git add -- src/smoke_defense/certificates src/smoke_defense/candidates.py src/smoke_defense/q1.py tests
git commit -m "feat: solve q1 across inertial guidance sweep"
```

---

### Task 8（未开始）：实现 Q2 单机多弹联合覆盖

**Files:**
- Create: `src/smoke_defense/q2.py`
- Create: `tests/test_q2.py`

**Step 1: 写失败测试**

测试最多 3 枚、相邻实际投弹至少 1 s、单机路径连续、联合覆盖优于独立覆盖基线，以及最终方案通过公共验证器。

**Step 2: 实现候选组合**

由 Q1 候选库构造 0-1 选择问题，使用枚举或 MILP 选择最多 3 个候选，再连续精修起爆时刻。目标：

1. 消除探测集合裸露；
2. 最小化最大空档；
3. 最大化联合裕度；
4. 最少弹药；
5. 最短 UAV 航程。

**Step 3: 验证和提交**

```powershell
python -m pytest tests/test_q2.py -v
git add -- src/smoke_defense/q2.py tests/test_q2.py
git commit -m "feat: add q2 multi-smoke optimization"
```

---

### Task 9（未开始）：实现 Q3 三机三弹协同

**Files:**
- Create: `src/smoke_defense/q3.py`
- Create: `tests/test_q3.py`

**Step 1: 写失败测试**

验证三架 UAV 均从舰船起飞、每机最多一枚、动态安全距离检查内部最近点、同时同点起飞不可行、三烟幕联合覆盖，以及删除任一烟幕后的最坏覆盖指标。

**Step 2: 实现词典序和 \(\varepsilon\)-约束**

先最大化认证防御，再在成功层中扫描容错—航程权衡。小候选库优先枚举/MILP；只有规模超限才启用 NSGA-II，并必须通过公共验证器。

**Step 3: 验证和提交**

```powershell
python -m pytest tests/test_q3.py -v
git add -- src/smoke_defense/q3.py tests/test_q3.py
git commit -m "feat: add q3 cooperative defense"
```

---

### Task 10（未开始）：实现 Q4 任务包和滚动调度

**Files:**
- Create: `src/smoke_defense/packages.py`
- Create: `src/smoke_defense/q4.py`
- Create: `configs/scenarios/q4/`
- Create: `tests/test_packages.py`
- Create: `tests/test_q4.py`

**Step 1: 写任务包测试**

任务包必须保存：

- 适用威胁和惯性参数；
- UAV/弹药需求；
- 可用开始时间窗；
- 路径起终状态；
- 覆盖证书；
- 最大空档和鲁棒裕度；
- 航程和单点失效指标。

**Step 2: 写在线信息约束测试**

`revealed_at_appearance` 模式下，调度器在时刻 \(t\) 只能读取已经揭示的威胁；`offline_full_information` 只作为上界。

**Step 3: 实现 5 机 15 弹滚动 MILP**

变量表示“威胁—任务包—开始时刻—UAV”。约束任务不重叠、路径兼容、投弹间隔、安全距离和资源容量。新批次出现后滚动重算。

**Step 4: 实现 9 组全局惯性参数**

同一次 Q4 运行中，同型号导弹共享一组
\((k,\omega_{\max})\)。每个资源场景运行 9 次，不枚举 \(9^n\) 个逐弹组合。

**Step 5: 验证和提交**

```powershell
python -m pytest tests/test_packages.py tests/test_q4.py -v
git add -- src/smoke_defense/packages.py src/smoke_defense/q4.py configs/scenarios/q4 tests/test_packages.py tests/test_q4.py
git commit -m "feat: add q4 rolling defense scheduling"
```

---

### Task 11（未开始）：实现鲁棒性、结果溯源和端到端报告

**Files:**
- Create: `src/smoke_defense/robustness.py`
- Create: `src/smoke_defense/reporting.py`
- Create: `scripts/run_all.py`
- Create: `tests/test_robustness.py`
- Create: `tests/test_reporting.py`
- Create: `tests/test_end_to_end.py`
- Modify: `README.md`

**Step 1: 写溯源失败测试**

每个结果必须包含：

- 场景和常数哈希；
- Git SHA；
- 随机种子；
- A-001–A-022 假设 ID；
- 惯性参数和 `model_layer`；
- 三值认证状态；
- 中值、范围、最不利组合；
- `parameter_sensitive` 标记；
- 在线或离线信息模式。

**Step 2: 实现有界鲁棒验证**

先验证延时、烟幕中心、半径、风漂和速度误差盒；只有分布有依据时才增加蒙特卡罗概率评估。

**Step 3: 生成结果和图表**

至少输出：

1. 舰船、惯性导弹、UAV、烟幕轨迹；
2. 距离、视场偏角和探测分量时间轴；
3. 9 组惯性参数轨迹包络；
4. Q1 覆盖裕度与裸露区间；
5. Q2 多烟幕时序；
6. Q3 三机安全距离和失效分析；
7. Q4 威胁—任务包分配甘特图；
8. 瞬时纯追踪消融差异。

**Step 4: 端到端验证**

```powershell
python -m pytest -q
python -m ruff check .
python scripts/export_scenario_schema.py --check
python scripts/generate_scenarios.py --check
python scripts/run_all.py --seed 20260730
git diff --check
```

Expected:

- 全部测试通过；
- 144 个正式 Q1–Q3 场景和 16 个消融场景；
- Q4 不出现 \(9^n\) 场景爆炸；
- 所有正式结论绑定公共验证器和溯源信息；
- 没有结果把中值惯性参数写成题面事实。

**Step 5: 提交**

```powershell
git add -- src scripts tests results figures README.md
git commit -m "feat: add robust end-to-end defense pipeline"
```

---

## 完成定义

- [ ] A-001–A-022 在配置、代码、结果和论文中可追溯；
- [ ] 144 个正式惯性场景和 16 个瞬时纯追踪消融场景通过 Schema；
- [ ] 导弹速度恒定、航向率限幅、初始航向规则通过测试；
- [ ] 探测集合支持多个连续分量；
- [ ] 24.1677–25.3610 s 旧界只存在于消融说明；
- [ ] UAV 只能从实际起飞时刻的舰船位置起飞；
- [ ] UAV 自由初态、初始航向和发射偏置被 Schema 拒绝；
- [ ] 同时同点起飞不能绕过安全距离；
- [ ] 单烟幕解析覆盖和多烟幕联合覆盖均通过独立验证；
- [ ] Q1–Q4 只消费公共验证器；
- [ ] Q4 每个资源场景运行 9 组全局惯性参数，不展开 \(9^n\)；
- [ ] 中值、范围、最不利组合和消融差异均可复现；
- [ ] `pytest`、Ruff、Schema、场景生成和端到端流水线全部通过。
