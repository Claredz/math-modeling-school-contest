# B题实现架构与场景契约设计

> 日期：2026-07-30
> 状态：**已过时，仅供追溯；禁止作为当前实现依据**
> 上游契约：[B题统一模型契约 v0.1](../modeling/model_contract_v0.1.md)
> 假设基线：[A-001–A-020](../modeling/assumption_register.md)
> 替代设计：[惯性纯追踪与舰载起飞模型修订设计](2026-07-30-inertial-pursuit-shipborne-launch-design.md)
> 替代计划：[惯性纯追踪与舰载起飞模型修订 Implementation Plan](2026-07-30-inertial-pursuit-shipborne-launch-implementation-plan.md)

## 1. 设计目标

本阶段不再修改模型解释，而是把冻结模型变成三名组员和多个 AI 都能共同执行的工程契约。核心目标是：

1. 不虚构唯一“官方数值场景”，而是用可校验的场景文件描述缺失数据；
2. 四问复用同一套时间轴、动力学、路径约束和联合覆盖验证器；
3. 解析证书先于优化器，优化器只能生成候选，不能自行宣布成功；
4. 每个结果都能追溯到场景 ID、场景哈希、假设版本、代码提交和随机种子；
5. 三名组员可以并行开发，但不得各自复制并修改公共物理公式。

## 2. 方案选择

采用“契约优先、能力分层”的实现方式，而不采用以下两种方案：

- **按 Q1–Q4 分四套代码**：启动快，但会产生四套时间语义和遮蔽判据，团队合并成本最高；
- **先写一个巨型仿真器**：接口少，但解析证明、几何验证和优化器耦合，任何修改都会牵动全局。

选定方案把系统拆成七层：

```text
场景契约
  ↓
混合事件时间线 ──→ 最早烟幕形成公共证书
  ↓
共享动力学与探测窗口
  ↓
其余解析证书 ──→ 候选剪枝
  ↓
无人机路径与移动参考约束
  ↓
连续时间联合覆盖验证器
  ↓
Q1 → Q2 → Q3 → Q4 候选与优化
```

下层只提供经过测试的稳定接口，上层不得绕过下层重复实现判据。

## 3. 常数与场景构成唯一配置入口

### 3.1 文件分层

```text
configs/
  constants.yaml                 # 题面公共常数的唯一事实源
  schema/
    scenario.schema.json         # 由 Pydantic 导出的生成产物
  scenarios/
    q1_q3/
      generated/                 # 由矩阵生成器产生，不手工维护
    q4/
      q4_case_3_threats.yaml
      q4_case_6_threats.yaml
      q4_case_9_threats.yaml
  sweeps/
    q1_q3_matrix.yaml
    safe_distance.yaml
    robustness.yaml
    pn_ablation.yaml             # 首版不启用
```

`constants.yaml` 不保存初始坐标、安全距离或误差等级。所有题面未给出的量只能进入 `scenarios/` 或 `sweeps/`，并带来源字段。

### 3.2 题面公共常数的唯一事实源

`configs/constants.yaml` 是题面公共常数的唯一来源：

```yaml
constants_version: "b-problem-v1"

ship:
  speed_mps: 7.71
  equivalent_radius_m: 80.0

missile:
  nominal_speed_mps: 320.0
  detection_range_m: 8000.0
  field_of_view_half_angle_deg: 15.0

uav:
  flight_speed_mps: 28.0
  operation_radius_m: 12000.0
  max_payload: 3
  minimum_release_interval_s: 1.0

bomb:
  minimum_release_response_s: 2.0
  inertial_flight_s: 3.5

smoke:
  maximum_radius_m: 120.0
  hold_duration_s: 18.0
  decay_duration_s: 5.0
```

场景只引用 `constants_version`，不得重复保存上述数值。载入器先分别校验 `ProblemConstants` 和原始 `Scenario`，再合并为只读的规范化场景对象。配置哈希包含规范化场景、`constants_version` 和常数文件内容哈希，因此公共常数或任何显式覆盖改变时，哈希必然改变。

### 3.3 场景 YAML 结构

```yaml
schema_version: "1.0"
constants_version: "b-problem-v1"
scenario_id: "q1_front_d10000_nominal"
problem_scope: ["Q1", "Q2", "Q3"]
description: "正前方、初距10000 m的参数化基准场景"
source: "team_scenario_assumption"
time_origin: "decision_start"
assumption_ids:
  - "A-001"
  - "A-002"
  - "A-003"
  - "A-004"
  - "A-005"
  - "A-006"
  - "A-007"
  - "A-008"
  - "A-009"
  - "A-010"
  - "A-011"
  - "A-013"
  - "A-014"
  - "A-016"
  - "A-017"
  - "A-018"
  - "A-019"
  - "A-020"

ship:
  initial_position_world_m: [0.0, 0.0]
  heading_deg: 0.0

uavs:
  - id: "U1"
    launch_mode: "shipborne"
    available_time_s: 0.0
    launch_offset_body_m: [0.0, 0.0]

missiles:
  - id: "M1"
    appearance_time_s: 0.0
    initial_position_at_appearance_body_m: [10000.0, 0.0]
    guidance_model: "pure_pursuit"
    optical_axis_model: "velocity_aligned"

constraints:
  safe_distance_m: 100.0

uncertainty:
  tier: "nominal"
  release_response_error_s: 0.0
  detonation_delay_error_s: 0.0
  smoke_center_error_m: 0.0
  smoke_radius_error_m: 0.0
  wind_speed_bound_mps: 0.0

extensions:
  wind_drift_enabled: false
  pn_ablation_enabled: false
```

结构约束：

- `appearance_time_s` 表示威胁进入任务场景的时刻，不等于进入 8000 m 探测边界的时刻；
- 导弹在 `appearance_time_s` 之前不存在，之后按场景初态积分；
- `initial_position_at_appearance_body_m` 中的
  \(\boldsymbol r_i^{\rm app}\) 是导弹出现时相对舰船体坐标系的位置，严格满足
  \[
  \boldsymbol m_i(t_i^{\rm app})
  =
  \boldsymbol s(t_i^{\rm app})
  +Q(\theta_s)\boldsymbol r_i^{\rm app};
  \]
- 如果场景直接提供世界坐标，必须改用独立字段 `initial_position_world_m`，其值同样表示
  \(\boldsymbol m_i(t_i^{\rm app})\)；每枚导弹必须且只能提供这两个位置字段中的一个；
- `launch_offset_body_m` 是相对起飞时刻舰船体坐标系的位置；
- 基准 UAV 从舰船处起飞，偏置默认为 `[0, 0]`，非零偏置只作为后续扩展；
- 没有速度覆盖时，导弹读取 `constants.yaml` 的 `missile.nominal_speed_mps`；
- Q4 若覆盖导弹速度，必须同时提供
  `speed_override_mps` 和 `speed_source: "team_scenario_assumption"`；两字段共同进入场景哈希，且不得回写 `constants.yaml` 或被表述为题面事实；
- 所有字段必须显式带单位后缀，角度配置用 degree、计算内部转 rad；
- 未知字段拒绝载入，避免拼写错误被静默忽略。

速度覆盖示例：

```yaml
missiles:
  - id: "M1"
    appearance_time_s: 20.0
    initial_position_at_appearance_body_m: [9000.0, 1000.0]
    speed_override_mps: 300.0
    speed_source: "team_scenario_assumption"
    guidance_model: "pure_pursuit"
    optical_axis_model: "velocity_aligned"
```

### 3.4 Pydantic 是场景 Schema 的唯一源码

所有场景结构只在 Pydantic 模型中定义。`Scenario` 及其所有嵌套模型必须配置 `extra="forbid"`，并负责正值、单位字段、位置字段互斥、速度覆盖来源和模式组合等语义校验。处理管线固定为：

```text
YAML 解析
→ Pydantic 结构与语义校验
→ Scenario.model_json_schema() 导出或核对 JSON Schema
→ 合并题面常数并生成规范化场景对象
```

`configs/schema/scenario.schema.json` 可以提交到仓库，供编辑器和外部工具使用，但只能由 `Scenario.model_json_schema()` 生成。测试必须重新导出并与仓库文件逐字节比较；禁止人工独立修改 JSON Schema。未知字段、错误单位后缀以及两个初始位置字段同时出现时都必须拒绝。

### 3.5 舰载起飞位置

无人机 \(u\) 在时刻 \(\tau\) 起飞时：

\[
\boldsymbol u_u(\tau)
=\boldsymbol s(\tau)
+Q(\theta_s)\boldsymbol r^{\rm launch}_u,
\]

其中

\[
Q(\theta_s)=
\begin{bmatrix}
\cos\theta_s&-\sin\theta_s\\
\sin\theta_s&\cos\theta_s
\end{bmatrix}.
\]

首版令 \(\boldsymbol r^{\rm launch}_u=\boldsymbol0\)。保留该字段是为了以后表达甲板发射位差，而不是在首版增加自由度。

## 4. 场景矩阵

### 4.1 Q1–Q3 方位和距离

四类导弹初始方向以舰船体坐标表示：

| 名称 | 单位方向 | 说明 |
|---|---|---|
| front | \((1,0)\) | 舰首方向 |
| rear | \((-1,0)\) | 舰尾方向 |
| side | \((0,1)\) | 左舷侧向；右舷由镜像检查 |
| oblique | \((\cos135^\circ,\sin135^\circ)\) | 斜后方 |

初始距离扫描：

\[
d_0\in\{8000,10000,12000,15000\}\ {\rm m}.
\]

这形成 \(4\times4=16\) 个名义场景。结果应报告方向与距离的规律，不从中挑选一个场景冒充题面唯一答案。

### 4.2 安全距离

\[
d_{\rm safe}\in\{50,100,200,500\}\ {\rm m}.
\]

若 100–200 m 之间出现可行性或最优结构变化，再在该区间自适应细化。该数值是场景扫描，不是题面设备指标。

### 4.3 鲁棒误差等级

| 等级 | 响应误差 | 起爆误差 | 烟幕中心误差 | 半径误差 | 风速上界 |
|---|---:|---:|---:|---:|---:|
| light | 0.1 s | 0.1 s | 5 m | 3 m | 1 m/s |
| medium | 0.3 s | 0.3 s | 10 m | 6 m | 3 m/s |
| strong | 0.5 s | 0.5 s | 20 m | 12 m | 5 m/s |

这些量统一称为“场景误差等级”，不得称为真实设备精度。

字段语义固定为：

- `release_response_error_s`：指令到实际释放的响应时间误差；
- `detonation_delay_error_s`：实际释放到起爆的 3.5 s 延时误差；
- `smoke_center_error_m`：最终烟幕中心位置误差；
- `smoke_radius_error_m`：烟幕有效半径误差；
- `wind_speed_bound_mps`：烟幕起爆后的风漂速度上界。

第一版不引入舰船、导弹或无人机状态估计误差。未来如确有需要，分别增加 `ship_position_error_m`、`missile_position_error_m` 和 `uav_position_error_m`，不得重新引入含义混合的位置误差字段。

对烟幕 \(j\) 可使用保守有效半径：

\[
R^{\rm rob}_j(t)
=R_j(t)
-\varepsilon_{\rm center}
-\varepsilon_R
-v_{w,\max}(t-t^e_j)_+
-L_{R,j}(t)
\left(\varepsilon_{\rm response}+\varepsilon_{\rm detonation}\right),
\]

其中 \(L_{R,j}=0\) 位于恒定半径阶段，衰减阶段取 \(24\rm\,m/s\)。若
\(R^{\rm rob}_j(t)<0\)，验证时截断为 0。时间误差还会移动起爆和失效事件，因此最终鲁棒验证必须检查误差盒的事件端点，不能只做半径缩水。

### 4.4 Q4 资源工况

Q4 使用中性文件名描述来袭规模：

- `q4_case_3_threats.yaml`；
- `q4_case_6_threats.yaml`；
- `q4_case_9_threats.yaml`。

文件名只表达威胁数量，不预判资源是否充足。求解后按照认证结果写入：

- `resource_abundant`：全部威胁可认证防御且存在冗余资源；
- `resource_critical`：全部或高优先级威胁只有在接近资源上限时才可认证防御；
- `resource_shortage`：即使使用全部资源，仍有威胁被认证为存在不可避免裸露。

分类优先级为 `resource_shortage` > `resource_critical` > `resource_abundant`。其中“接近资源上限”必须由结果中的容量松弛量或删减一单位资源后的再认证定义；“存在冗余资源”必须由正容量松弛或可删除资源而不改变全部威胁认证结论来证明。分类器必须消费求解结果和资源占用证书，不得读取文件名推断类别。

每枚导弹独立给出 `appearance_time_s`、初始坐标以及可选的速度覆盖。`appearance_time_s`、`reveal_time_s` 与由轨迹计算得到的 `detection_entry_time_s` 是三个独立概念：

- `reveal_time_s`：调度器获知该导弹信息的时刻；
- `appearance_time_s`：导弹状态进入场景并开始积分的时刻；
- `detection_entry_time_s`：导弹轨迹首次进入有效探测集合的派生时刻。

Q4 场景必须声明：

```yaml
information_mode: "revealed_at_appearance"
```

并允许离线对照：

```yaml
information_mode: "offline_full_information"
```

在线模式下，每枚导弹满足 `reveal_time_s == appearance_time_s`，例如：

```yaml
missiles:
  - id: "M2"
    reveal_time_s: 20.0
    appearance_time_s: 20.0
    initial_position_at_appearance_body_m: [9000.0, -1200.0]
```

时刻 \(t\) 的已知信息集定义为：

\[
\mathcal I_{\rm known}(t)
=
\left\{i:t_i^{\rm reveal}\le t\right\}.
\]

在 `revealed_at_appearance` 模式中，滚动调度器只能读取
\(i\in\mathcal I_{\rm known}(t)\) 的威胁，不得利用未揭示导弹提前分配任务包、调动无人机或预留精确投弹位置。`offline_full_information` 仅作为全知性能上界。两种模式使用相同场景和目标定义；若目标 \(J\) 是同一可比收益标量，则报告

\[
\Delta_{\rm info}
=J_{\rm offline}-J_{\rm online}.
\]

若主目标是词典序向量，则逐项报告差距，不把离线全知结果表述为实时滚动结果。

### 4.5 暂缓 PN 消融

首版只实现冻结的纯追踪基准。比例导引不进入第一轮开发。后续若启用，仅扫描：

\[
N\in\{3,4,5\},\qquad
a_{\max}\in\{5g,10g,20g\},
\]

并明确标记为 X-001/X-002 消融，不能覆盖基准结果。

## 5. 混合事件时间线

系统不能只依赖等步长网格。统一事件至少包括：

- 场景出现；
- UAV 可用与起飞；
- 投弹指令；
- 实际释放；
- 烟幕起爆；
- 恒定半径结束；
- 烟幕失效；
- 导弹进入/离开探测集合；
- 导弹命中；
- Q4 新批次到达。

每个连续区间内部积分，在事件点进行状态重置和闭端点检查。结果文件同时保存事件表和连续轨迹，禁止用“数组下标 × 步长”反推法律意义上的事件时刻。

投弹事件必须显式接收实际释放时刻，不能从指令时刻自动固定为最早允许时刻：

```python
BombEvents(
    command_time_s=4.0,
    release_time_s=7.3,
    burst_time_s=10.8,
)
```

事件校验器检查：

\[
t^d-t^c\ge2,
\qquad
t^e-t^d=3.5,
\]

或在鲁棒场景中检查相应延时区间。辅助函数
`earliest_release_time(command_time_s)` 只返回
`command_time_s + 2.0`，不得替代实际 `release_time_s`。

### 5.1 最早烟幕形成公共解析证书

`certify_earliest_smoke_availability(...)` 是 Q1–Q3 共用证书，不属于 Q1 专用模块。在没有预部署烟幕、且所有无人机都从任务开始后才起飞和投弹的前提下：

\[
t^e\ge t^c+2+3.5
=t^c+5.5.
\]

若还需先机动到可投弹状态，则：

\[
t_{\rm smoke}^{\min}
=t^c+5.5+T_{\rm maneuver}.
\]

证书至少保存：

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

其中：

\[
T_{\rm unavoidable}
=
\max\left(
0,
t_{\rm smoke}^{\min}
-t_{\rm detection}^{\rm entry}
\right).
\]

若严格成功要求整个探测窗口无裸露且
\(T_{\rm unavoidable}>0\)，则当前“无预部署、任务开始后投放”方案类别可直接认证为不可行。特别地，\(d_0=8000\rm\,m\)、导弹在 \(t=0\) 已进入探测区、指令最早为 \(t=0\)、无需额外机动的场景，在首个烟幕形成前至少有 \(5.5\rm\,s\) 必然裸露，因此 Q1、Q2、Q3 都不能报告全窗口 100% 防御。

该证书只否定严格目标，不终止后续计算；求解器仍应继续优化最大覆盖时长、最小裸露时间和最佳投放方案。

## 6. A-020 的连续时间精确证书

在舰船和无人机都作直线运动的任一航段上，相对位置可写成

\[
\boldsymbol r(t)
=\boldsymbol u(t)-\boldsymbol s(t)
=\boldsymbol a+\boldsymbol b t.
\]

平方距离

\[
q(t)=\|\boldsymbol r(t)\|^2
=\|\boldsymbol b\|^2t^2
+2\boldsymbol a^\mathsf T\boldsymbol b\,t
+\|\boldsymbol a\|^2
\]

是闭区间上的凸二次函数。凸函数在闭区间的最大值必在端点取得，因此对每个分段直线航段 \([t_k,t_{k+1}]\)，

\[
\max_{t\in[t_k,t_{k+1}]}
\|\boldsymbol u(t)-\boldsymbol s(t)\|
=\max\left\{
\|\boldsymbol r(t_k)\|,
\|\boldsymbol r(t_{k+1})\|
\right\}.
\]

所以 A-020 的上界约束可以仅检查每段两个端点，并得到严格连续时间证书。该结论只用于“距离不超过上界”；多机安全距离是下界约束，其最小值可能在区间内部，必须检查二次函数驻点。

## 7. 验证器优先级

每个候选方案依次经过：

1. 场景结构和单位校验；
2. 事件因果关系校验；
3. UAV 路径连续性、固定飞行速度、载弹量和投弹间隔校验；
4. A-020 分段端点精确认证；
5. 多机安全距离的端点与内部驻点认证；
6. 单烟幕解析完整覆盖或多烟幕联合空间覆盖；
7. 按事件切段的连续时间无裸露认证；
8. 鲁棒误差盒复核；
9. 结果溯源字段完整性检查。

输出只允许三种状态：

- `certified_feasible`
- `certified_infeasible`
- `indeterminate_at_tolerance`

优化器的 `success`、`converged` 或目标值不能替代上述状态。

## 8. 三人和多 AI 的协作契约

三人的人员分工固定为“建模、论文、仿真代码”，不再按代码能力层或题号重新划分：

| 角色 | 完整主责 | 主要交付 | 不得自行决定 |
|---|---|---|---|
| A：建模 | 题意解释、假设管理、数学对象、公式推导、目标与约束、解析证书、结果验收和物理解释 | 模型契约、假设登记、代码交接表、结果验收记录 | 不得把未经仿真验证的数值写成结果 |
| B：论文 | 论文结构、符号统一、公式叙述、图表编排、结果引用、摘要和终稿一致性 | 论文正文、符号表、图表清单、论点—证据映射表 | 不得创造模型未定义的公式或代码未输出的数字 |
| C：仿真代码 | 整套仿真工程、场景载入、动力学、验证器、Q1–Q4 求解、测试、结果文件和绘图 | 可复现代码、测试报告、机器可读结果和自动生成图表 | 不得擅自修改 A-001–A-020、公式语义或成功判据 |

代码中的“场景、时间线、动力学、证书、验证器、优化器”等能力层只是仿真代码同学的内部实现顺序，不是三个人的任务分配。整套代码始终由仿真代码同学负责。

团队统一执行以下规则：

1. 建模同学把 `model_contract_v0.1.md`、A-001–A-020 和验收案例交给仿真代码同学，口头解释或 AI 聊天不能替代文件契约；
2. 仿真代码同学把结果写成 JSON/CSV，并记录 `scenario_id`、配置哈希、假设 ID、Git SHA、随机种子和验证状态；
3. 建模同学按照解析边界、约束反例和连续时间无裸露判据验收结果，只有通过验收的结果才能标记为 `verified_for_paper`；
4. 论文同学只从已冻结模型文档和 `verified_for_paper` 结果中取公式、数字和图表，不从聊天记录中摘录未经验证的结论；
5. 每个 AI 会话开头必须声明当前角色、模型版本、允许使用的输入、允许修改的文件和禁止改变的内容；
6. 建模 AI 可以推导和找反例但不能伪造仿真数字；代码 AI 可以实现和测试但不能改变模型；论文 AI 可以组织表达但不能创造公式和结果；
7. AI 提出的新假设必须先登记为待审批项，经三人确认并编号后，才能同步进入代码和论文；
8. 每次对齐只核对三份交接物：最新模型契约、最新已验证结果清单、最新论文论点—证据映射表。

这套规则让三人各自完整负责一条工作线，同时通过“模型契约 → 仿真结果 → 建模验收 → 论文引用”的单向证据链对齐。

## 9. 第一轮明确不做

- 不选择单一“官方基准场景”；
- 不实现 PN 或有限过载；
- 不引入失锁后的制导反馈；
- 不假设概率分布或真实设备精度；
- 不用蒙特卡罗替代有界鲁棒证书；
- 不在 Q1 解析不可行时继续搜索“100% 成功解”；
- 不在联合覆盖验证器完成前启动 Q2–Q4 大规模优化。
