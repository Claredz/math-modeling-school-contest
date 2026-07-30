# 惯性纯追踪与舰载起飞模型修订 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 Q1–Q4 的正式导弹动力学统一改为一阶航向惯性纯追踪，并将所有无人机的初始条件统一为可等待的舰载起飞。

**Architecture:** 先修订冻结假设和共享模型契约，再重建 Q1 解析证书边界，最后同步四问架构、审查材料和现有代码实施计划。瞬时纯追踪保留为消融对照；惯性参数进入场景扫描而非题面常数；所有文档通过关键词、公式和交叉引用检查保持一致。

**Tech Stack:** Markdown、LaTeX 数学公式、YAML/Pydantic 场景契约说明、PowerShell、ripgrep、Git。

---

### Task 1：更新冻结假设和共享模型契约

**Files:**
- Modify: `docs/modeling/assumption_register.md`
- Modify: `docs/modeling/model_contract_v0.1.md`

**Step 1: 记录预期失效的旧基准表述**

Run:

```powershell
rg -n "恒速纯追踪|视场偏角.*0|launch_offset|非零偏置|PN|有限转弯" docs/modeling/assumption_register.md docs/modeling/model_contract_v0.1.md
```

Expected: 找到 A-002 的瞬时纯追踪、A-003 的退化视场、X-002 的有限转弯扩展以及模型契约中的对应旧表述。

**Step 2: 将假设登记表升级为 v0.3**

在 `docs/modeling/assumption_register.md` 中完成以下原子修改：

1. 将版本改为 `v0.3`，审批日期保留 `2026-07-30`；
2. 将 A-002 改为正式惯性纯追踪：
   \[
   \dot\psi
   =
   \operatorname{clip}
   \left(
   k\,\operatorname{wrap}(\lambda-\psi),
   -\omega_{\max},
   \omega_{\max}
   \right);
   \]
3. 将 A-003 改为“光轴与实际速度方向一致，视场偏角一般不为零”；
4. 更新 A-009、A-019：UAV 起飞前随舰等待，起飞瞬间位置严格等于舰船实时位置，场景不得覆盖起点；
5. 增加 A-021：惯性参数扫描
   \(k\in\{0.5,1,2\}\rm\,s^{-1}\)、
   \(\omega_{\max}\in\{5,10,20\}^\circ/\rm s\)；
6. 增加 A-022：初始导弹航向等于出现时刻的视线角；
7. 将 X-002 标记为已进入基准并由 A-002/A-021 取代；
8. 将原瞬时纯追踪登记为消融对照，不再是正式基准；
9. 在变更控制区增加本次修订的理由、影响问题和失效解析结论。

**Step 3: 将共享模型契约升级为 v0.2**

在 `docs/modeling/model_contract_v0.1.md` 中：

1. 标题或版本区标为 `v0.2`，说明文件名为兼容现有引用暂不改名；
2. 用三状态 ODE \((x_m,y_m,\psi)\) 替换瞬时纯追踪 ODE；
3. 增加角度归一化、限幅、初始航向和零距离命中规则；
4. 将探测集合改为
   \[
   \mathcal D_i
   =
   \{t:\rho_i(t)\le8000,\ 
   |\operatorname{wrap}(\lambda_i-\psi_i)|\le15^\circ\};
   \]
5. 明确探测集合可不连续；
6. 增加 9 组惯性参数、1 组瞬时纯追踪消融的数据层级；
7. 将 UAV 初始条件写为
   \[
   \boldsymbol u_u(t)=\boldsymbol s(t),\ t<\tau_u,
   \qquad
   \boldsymbol u_u(\tau_u)=\boldsymbol s(\tau_u);
   \]
8. 删除可配置发射偏置和自由 UAV 初始坐标；
9. 更新 Q1–Q4 接口，使结果记录
   `guidance_model`、`heading_response_rate_per_s`、
   `max_turn_rate_deg_s` 和参数层级；
10. 将旧的“纯追踪视场退化”冲突说明改为已解决。

**Step 4: 检查两份冻结文档**

Run:

```powershell
rg -n "A-021|A-022|inertial_pure_pursuit|dot\\s*\\\\psi|omega_\\{\\\\max\\}|舰上等待|实际起飞时刻" docs/modeling/assumption_register.md docs/modeling/model_contract_v0.1.md
```

Expected: 两份文件都能定位惯性导弹和舰载起飞规则；A-021、A-022 只在假设表中定义并在契约中引用。

Run:

```powershell
rg -n "视场偏角恒为0|视场偏角恒为零|基准导弹模型采用恒速纯追踪|非零偏置只作为" docs/modeling/assumption_register.md docs/modeling/model_contract_v0.1.md
```

Expected: 无匹配。

**Step 5: 提交**

```powershell
git add -- docs/modeling/assumption_register.md docs/modeling/model_contract_v0.1.md
git commit -m "docs: revise shared pursuit and launch contract"
```

---

### Task 2：重建 Q1 解析可行性边界

**Files:**
- Modify: `docs/modeling/q1_analytic_feasibility.md`

**Step 1: 标记仍有效和已失效的命题**

保留并明确证明依赖：

- 单烟幕完全覆盖等价条件；
- 固定烟幕完整覆盖时长上界
  \(80/7.71\approx10.3761\rm\,s\)；
- 2 s 响应和 3.5 s 起爆延时的最早烟幕形成界；
- 投弹点距惯性起爆点 98 m 的反演几何；
- UAV 相对运动舰船的 12000 m 作战半径。

删除或降级为“瞬时纯追踪消融专用”：

- 探测距离必严格递减的原闭合速度证明；
- 24.1677–25.3610 s 的正式探测窗口界；
- 探测集合必为单连续区间；
- 视场偏角恒为零；
- 基于上述界对所有正式惯性场景直接下的不可行结论。

**Step 2: 写入惯性模型下的新证书流程**

新增以下逻辑：

1. 用带距离、视场、命中事件的 ODE 积分得到
   \(\mathcal D=\bigcup_\ell[a_\ell,b_\ell]\)；
2. 对每个必须连续遮蔽的分量计算
   \(L_\ell=b_\ell-a_\ell\)；
3. 若存在
   \[
   L_\ell>10.3761\rm\,s,
   \]
   则认证单固定烟幕不可能覆盖该连续分量；
4. 若所有分量均不超过该界，只能返回
   `indeterminate_at_tolerance` 并进入联合的时空数值验证；
5. 不允许从“解析证书未证明不可行”反向推导可行。

**Step 3: 更新 Q1 求解顺序**

将流程改为：

1. 按 9 组惯性参数积分威胁轨迹；
2. 计算每组探测集合和最早烟幕形成空档；
3. 使用仍有效的几何时长上界进行分量级剪枝；
4. 对未被剪枝的场景生成低维投放候选；
5. 分别认证中值、范围和最坏参数组合；
6. 最后运行瞬时纯追踪消融并报告偏差。

**Step 4: 检查解析文档**

Run:

```powershell
rg -n "10\\.3761|indeterminate_at_tolerance|多个时间区间|一阶航向惯性|最不利参数" docs/modeling/q1_analytic_feasibility.md
```

Expected: 每类新结论至少有一处匹配。

Run:

```powershell
rg -n "视场约束退化|alpha\\(t\\).*equiv0|探测窗口必|纯追踪模型下的探测窗口界" docs/modeling/q1_analytic_feasibility.md
```

Expected: 无正式基准表述匹配；如保留历史对照，必须与“消融”同段出现。

**Step 5: 提交**

```powershell
git add -- docs/modeling/q1_analytic_feasibility.md
git commit -m "docs: rebuild q1 certificate for inertial pursuit"
```

---

### Task 3：同步四问架构、审查结论和高级方法

**Files:**
- Modify: `docs/modeling/revised_four_problem_architecture.md`
- Modify: `docs/modeling/model_review_report.md`
- Modify: `docs/modeling/advanced_math_options.md`

**Step 1: 更新四问共享数据流**

在 `revised_four_problem_architecture.md` 中：

- 将共享动力学改为惯性纯追踪；
- 在 Q1–Q4 输入中删除 UAV 自由初态，增加起飞时刻；
- 在 Q1–Q4 输出中增加惯性参数和参数层级；
- 将 Q1 的“先用固定 24 s 探测界”改为“先积分探测集合，再做分量级时长证书”；
- 将 Q4 的 9 组全局参数扫描写入任务包验证流程；
- 明确同型号导弹在同一次 Q4 仿真中共享惯性参数。

**Step 2: 将模型审查报告改为变更后状态**

在 `model_review_report.md` 中：

- 将“纯追踪使视场退化”标为已通过惯性转弯修复；
- 将“有限转弯只做扩展”“PN/有限转弯未冻结”等旧状态改为历史记录；
- 将正式基准总结改为一阶惯性纯追踪；
- 将最大返工风险改为：角度单位混用、忘记限幅、误把中值参数写成题面事实、错误复用旧探测界；
- 将 UAV 发射位置缺口改为已冻结的舰载起飞规则；
- 保留“该制导律仍是团队解释而非题面唯一推论”的免责声明。

**Step 3: 调整高级方法定位**

在 `advanced_math_options.md` 中：

- 将一阶航向惯性纯追踪从可选扩展提升为基准动力学；
- 将瞬时纯追踪改为极限消融；
- PN 保持后续扩展，不与本次惯性模型混同；
- 将前向可达集的 UAV 起点固定为
  \(\boldsymbol s(\tau_u)\)；
- 在消融列表中改为“惯性纯追踪 vs 瞬时纯追踪”，另列“惯性纯追踪 vs PN（后续）”。

**Step 4: 运行跨文档语义检查**

Run:

```powershell
rg -n "惯性纯追踪|舰载起飞|参数扫描|消融" docs/modeling/revised_four_problem_architecture.md docs/modeling/model_review_report.md docs/modeling/advanced_math_options.md
```

Expected: 三份文件均至少命中两类新术语。

Run:

```powershell
rg -n "纯追踪作为可解释基准|视场在基准中退化|有限转弯.*可选|无人机初态" docs/modeling/revised_four_problem_architecture.md docs/modeling/model_review_report.md docs/modeling/advanced_math_options.md
```

Expected: 无未标注“历史/消融”的现行表述。

**Step 5: 提交**

```powershell
git add -- docs/modeling/revised_four_problem_architecture.md docs/modeling/model_review_report.md docs/modeling/advanced_math_options.md
git commit -m "docs: propagate revised dynamics across four problems"
```

---

### Task 4：更新场景契约和实现架构设计

**Files:**
- Modify: `docs/plans/2026-07-30-b-problem-implementation-design.md`

**Step 1: 更新场景 YAML 示例**

将导弹配置改为：

```yaml
missiles:
  - id: "M1"
    appearance_time_s: 0.0
    initial_position_at_appearance_body_m: [10000.0, 0.0]
    guidance_model: "inertial_pure_pursuit"
    heading_response_rate_per_s: 1.0
    max_turn_rate_deg_s: 10.0
    optical_axis_model: "velocity_aligned"
```

删除：

```yaml
uavs:
  - id: "U1"
    available_time_s: 0.0
    launch_offset_body_m: [0.0, 0.0]
```

改为只配置 UAV 标识和可用时刻；起飞位置由舰船轨迹和决策变量
\(\tau_u\) 派生。

**Step 2: 更新 Schema 语义**

设计中明确：

- 正式层只允许 `inertial_pure_pursuit`；
- `instantaneous_pure_pursuit` 只允许
  `model_layer: ablation`；
- `heading_response_rate_per_s > 0`；
- `0 < max_turn_rate_deg_s <= 180`；
- 初始航向字段禁止输入；
- UAV 自由初始位置和发射偏置为未知字段，应由
  `extra="forbid"` 拒绝；
- 场景哈希包含惯性参数和模型层。

**Step 3: 更新场景矩阵**

将原 16 个 Q1–Q3 几何场景扩展说明为：

\[
16\times9=144
\]

个正式惯性场景，并增加 16 个瞬时纯追踪消融场景。Q4 按每个资源工况运行 9 组共享导引参数，不展开 \(9^n\) 个逐弹组合。

**Step 4: 更新共享动力学和事件设计**

将动力学章节改为三状态 ODE，新增：

- `wrap_to_pi`；
- 航向限幅；
- 距离边界事件；
- 视场入口/出口事件；
- 命中事件；
- 初始已命中短路；
- 探测集合的多区间表示。

将“PN 暂缓”保留，但删除“首版只实现冻结的瞬时纯追踪”。

**Step 5: 更新舰载起飞和验证器优先级**

删除甲板偏置扩展。写明起飞前轨迹、起飞瞬间位置和多机同点同时起飞不可行规则。验证器优先级应在覆盖检查前验证：

1. 起飞时刻与舰船位置一致；
2. 起飞后固定飞行速度；
3. 相对运动舰船作战半径；
4. 多机空中安全距离。

**Step 6: 检查设计文档**

Run:

```powershell
rg -n "inertial_pure_pursuit|heading_response_rate_per_s|max_turn_rate_deg_s|144|多区间|同时同点起飞" docs/plans/2026-07-30-b-problem-implementation-design.md
```

Expected: 所有关键词均匹配。

Run:

```powershell
rg -n "guidance_model: \"pure_pursuit\"|launch_offset_body_m|首版只实现冻结的纯追踪|偏角为0" docs/plans/2026-07-30-b-problem-implementation-design.md
```

Expected: 无匹配。

**Step 7: 提交**

```powershell
git add -- docs/plans/2026-07-30-b-problem-implementation-design.md
git commit -m "docs: update implementation architecture for inertial guidance"
```

---

### Task 5：改写现有 Implementation Plan

**Files:**
- Modify: `docs/plans/2026-07-30-b-problem-implementation-plan.md`

**Step 1: 更新 Stage 1 场景层测试**

将示例断言改为：

```python
assert scene.missiles[0].guidance_model == "inertial_pure_pursuit"
assert scene.missiles[0].heading_response_rate_per_s == pytest.approx(1.0)
assert scene.missiles[0].max_turn_rate_deg_s == pytest.approx(10.0)
```

增加失败测试：

```python
def test_formal_scenario_rejects_instantaneous_pursuit():
    ...

def test_ablation_scenario_allows_instantaneous_pursuit():
    ...

def test_uav_launch_offset_is_rejected():
    ...
```

并将场景矩阵验收值由 16 改为 144 个正式场景。

**Step 2: 重写共享动力学 TDD 任务**

在 Task 4 中先写以下失败测试：

```python
def test_missile_speed_magnitude_is_constant():
    ...

def test_heading_rate_is_clipped():
    ...

def test_fixed_los_error_decays_with_first_order_response():
    ...

def test_large_response_parameters_converge_to_instantaneous_reference():
    ...

def test_boresight_error_can_open_and_close_detection_window():
    ...

def test_initial_hit_does_not_evaluate_undefined_los():
    ...
```

最小实现明确拆为：

- `wrap_to_pi(angle_rad)`；
- `inertial_pursuit_rhs(t, state, ship_trajectory, spec)`；
- `instantaneous_pursuit_reference(...)`；
- 距离、视场和命中事件；
- 多探测区间组装。

**Step 3: 重写 Q1 证书测试**

删除以 24.1677–25.3610 s 为正式模型边界的测试，替换为：

```python
def test_long_detection_component_certifies_single_smoke_infeasible():
    ...

def test_short_components_do_not_certify_feasible():
    ...

def test_old_duration_bound_is_ablation_only():
    ...
```

明确 `short_components` 返回
`indeterminate_at_tolerance`。

**Step 4: 更新路径与舰载起飞测试**

新增：

```python
def test_uav_waits_on_moving_ship_before_takeoff():
    ...

def test_uav_launch_position_equals_ship_position_at_takeoff():
    ...

def test_simultaneous_colocated_launches_violate_safe_distance():
    ...
```

删除 `shipborne_launch_position(..., launch_offset_body_m=...)` 的非零偏置测试。函数可以简化为按时刻查询舰船位置，或直接不再保留独立偏置接口。

**Step 5: 更新 Q1–Q4 优化和鲁棒任务**

计划中明确：

- 每个正式候选绑定惯性参数；
- 中值方案在 9 组参数下交叉验证；
- Q4 每个资源场景运行 9 个共享参数版本；
- 瞬时纯追踪只生成消融报告；
- 报告输出中值、范围、最坏组合和消融偏差；
- 不通过 9 组复核的方案标记为
  `parameter_sensitive`，不得称为鲁棒方案。

**Step 6: 更新完成定义**

至少加入：

- [ ] 144 个 Q1–Q3 正式惯性场景通过 Schema；
- [ ] 航向响应和最大转弯率均来自场景扫描；
- [ ] 视场偏角不再被硬编码为零；
- [ ] 探测集合支持多个连续分量；
- [ ] 旧探测时长界只用于消融；
- [ ] UAV 只能从实际起飞时刻舰船位置起飞；
- [ ] UAV 自由初始坐标和发射偏置被拒绝；
- [ ] Q4 不展开 \(9^n\) 组合；
- [ ] 正式结果包含参数范围和最不利组合。

**Step 7: 检查计划中是否还有冲突**

Run:

```powershell
rg -n "纯追踪|视场|launch_offset|偏置|24\\.1677|25\\.3610|16 个|PN" docs/plans/2026-07-30-b-problem-implementation-plan.md
```

Expected: 所有保留匹配都必须明确属于“瞬时纯追踪消融”“历史旧界”或“PN 后续扩展”，不得再描述正式基准。

Run:

```powershell
rg -n "inertial_pure_pursuit|heading_rate|144 个|parameter_sensitive|多探测区间|舰船位置起飞" docs/plans/2026-07-30-b-problem-implementation-plan.md
```

Expected: 每个新验收主题均有匹配。

**Step 8: 提交**

```powershell
git add -- docs/plans/2026-07-30-b-problem-implementation-plan.md
git commit -m "docs: replan implementation for inertial pursuit"
```

---

### Task 6：执行全局一致性审核

**Files:**
- Verify: `docs/modeling/*.md`
- Verify: `docs/plans/*.md`
- Modify if needed: 本计划 Task 1–5 中列出的文件

**Step 1: 搜索未标注的旧模型表述**

Run:

```powershell
rg -n "基准.*纯追踪|纯追踪.*基准|视场.*退化|偏角.*恒为0|偏角.*恒为零|launch_offset_body_m|非零偏置|任意.*无人机.*初始" docs/modeling docs/plans
```

Expected: 只允许在历史设计文档、已明确标记的消融说明或本次变更说明中出现。任何现行契约里的匹配都必须修正。

**Step 2: 搜索新契约覆盖情况**

Run:

```powershell
rg -l "惯性纯追踪|inertial_pure_pursuit" docs/modeling docs/plans
rg -l "舰载起飞|实际起飞时刻.*舰船|舰船.*实际起飞时刻" docs/modeling docs/plans
```

Expected: Task 1–5 的八份目标文档均被相应搜索覆盖。

**Step 3: 检查 Markdown 和 Git 差异**

Run:

```powershell
git diff --check
git status --short
```

Expected: `git diff --check` 无输出；只出现本轮审核发现并修正的预期文档。

**Step 4: 复核关键数值和单位**

逐项确认：

- \(v_m=320\rm\,m/s\)；
- \(v_u=28\rm\,m/s\)；
- \(v_s=7.71\rm\,m/s\)；
- 探测距离 \(8000\rm\,m\)；
- 视场半角 \(15^\circ\)；
- \(k=\{0.5,1,2\}\rm\,s^{-1}\)；
- \(\omega_{\max}=\{5,10,20\}^\circ/\rm s\)；
- 中值参数为 \(1\rm\,s^{-1}\) 与 \(10^\circ/\rm s\)；
- 固定烟幕单弹覆盖上界 \(10.3761\rm\,s\)；
- UAV 起飞点无偏置。

**Step 5: 提交最终一致性修正**

若 Step 1–4 产生修正：

```powershell
git add -- docs/modeling docs/plans
git commit -m "docs: verify revised model consistency"
```

若没有修正，跳过提交。

