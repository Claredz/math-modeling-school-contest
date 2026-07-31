from __future__ import annotations

import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_OBJECTIVE_HIERARCHIES = {
    "Q1": [
        "strict_full_window_coverage",
        "max_certified_total_coverage_duration",
        "min_max_continuous_exposure",
        "max_min_coverage_margin",
        "min_travel_or_resources",
    ],
    "Q2": [
        "strict_full_window_coverage",
        "max_longest_gap_free_continuous_coverage",
        "max_certified_total_effective_coverage",
        "min_max_continuous_exposure",
        "min_total_exposure",
        "max_joint_coverage_margin",
        "min_bombs",
        "min_travel",
    ],
    "Q3": [
        "strict_cooperative_defense_success",
        "max_certified_coverage_capability",
        "min_max_exposure_or_failure_risk",
        "max_single_point_failure_tolerance",
        "min_energy_and_conflict_risk",
    ],
    "Q4": [
        "max_certified_high_loss_threats",
        "max_total_certified_threat_value",
        "min_unprotected_loss",
        "min_resource_use",
        "min_rescheduling_volatility",
    ],
}

EXPECTED_OBJECTIVE_SUMMARIES = {
    "Q1": (
        "achieve strict full-window coverage; maximize certified total coverage duration; "
        "minimize maximum continuous exposure; maximize minimum coverage margin; "
        "minimize travel or resources"
    ),
    "Q2": (
        "achieve strict full-window coverage; maximize longest gap-free continuous coverage; "
        "maximize certified total effective coverage; minimize maximum continuous exposure; "
        "minimize total exposure; maximize joint coverage margin; minimize bombs; "
        "minimize travel"
    ),
    "Q3": (
        "achieve strict cooperative defense success; maximize certified coverage capability; "
        "minimize maximum exposure or failure risk; maximize single-point-failure tolerance; "
        "minimize energy and conflict risk"
    ),
    "Q4": (
        "maximize certified high-loss threats; maximize total certified threat value; "
        "minimize unprotected loss; minimize resource use; minimize rescheduling volatility"
    ),
}

EXPECTED_DECOMPOSITION_GOALS = {
    "Q1": (
        "优先实现严格全窗口覆盖；严格不可行后依次最大化认证覆盖总时长、"
        "最小化最大连续裸露、最大化最小覆盖裕度，最后最小化航程或资源"
    ),
    "Q2": (
        "优先实现严格全窗口覆盖；其后依次最大化最长无空档连续覆盖和认证有效"
        "遮蔽总时长、最小化最大连续裸露与总裸露、最大化联合覆盖裕度，最后"
        "最小化弹药和航程"
    ),
    "Q3": (
        "优先实现严格协同防御成功；其后依次最大化认证覆盖能力、最小化最大裸露"
        "或失效风险、最大化单点失效容错，最后最小化能耗与冲突风险"
    ),
    "Q4": (
        "依次最大化认证防御的高损失威胁数与认证防御威胁总价值，再最小化未防御"
        "损失、资源消耗和重调度波动"
    ),
}


def read_text(relative_path: str) -> str:
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def test_gate_b_approval_and_stage_4_fast_track_are_unambiguous() -> None:
    decision_log = json.loads(read_text("state/decision_log.json"))

    assert decision_log["current_stage"] == 5
    assert decision_log["next_stage"] == 5
    assert decision_log["gate_b_status"] == "approved"
    assert decision_log["stage_4_started"] is True
    assert decision_log["stage_4_status"] == "passed"
    assert decision_log["gate_c_status"] == "approved_fast_track"
    assert decision_log["stage_5_started"] is True
    approval = decision_log["gate_b_approval"]
    assert approval["approved_by"] == "human"
    assert approval["stage_4_started_at_approval"] is False
    assert len(approval["decisions"]) == 13
    assert all(item["approved"] for item in approval["decisions"])
    foundation = decision_log["stages"]["4"]
    assert 3 <= len(foundation["assumptions"]) <= 7
    assert all(item["support"] for item in foundation["assumptions"])
    assert len(foundation["symbols"]) >= 10
    assert all(item["unit"] and item["type"] for item in foundation["symbols"])
    assert len({item["symbol"] for item in foundation["symbols"]}) == len(
        foundation["symbols"]
    )
    assert len(foundation["terminology"]) >= 3
    assert foundation["consistency_check"]["with_stage2"] == "pass"
    assert foundation["consistency_check"]["with_stage3"] == "pass"


def test_each_subproblem_has_its_own_ordered_objective_hierarchy() -> None:
    decision_log = json.loads(read_text("state/decision_log.json"))
    objectives = decision_log["stages"]["2"]["objective_hierarchies"]

    assert objectives == EXPECTED_OBJECTIVE_HIERARCHIES
    assert decision_log["stages"]["2"]["objective_per_subproblem"] == (
        EXPECTED_OBJECTIVE_SUMMARIES
    )
    decomposition_goals = {
        item["id"]: item["goal"] for item in decision_log["stages"]["2"]["decomposition"]
    }
    assert decomposition_goals == EXPECTED_DECOMPOSITION_GOALS
    assert len({tuple(hierarchy) for hierarchy in objectives.values()}) == 4


def test_gate_b_approves_frameworks_without_freezing_q1_q2_or_q4_solver() -> None:
    decision_log = json.loads(read_text("state/decision_log.json"))
    selections = decision_log["stages"]["3"]["selected_per_subproblem"]

    assert selections["Q1"]["status"] == "framework_pending_formal_benchmark"
    assert selections["Q2"]["status"] == "priority_research_route_pending_oracle"
    assert selections["Q3"]["status"] == "framework_approved_pending_implementation"
    assert selections["Q4"]["status"] == "research_framework_pending_formal_benchmark"
    assert all(item["approval"] == "approved" for item in selections.values())


def test_stage_4_freezes_event_and_evidence_boundaries() -> None:
    foundation = read_text("docs/workflow/stage-04-foundation.md")

    for phrase in (
        "command → drop：固定 2 s",
        "drop → burst：固定 3.5 s",
        "禁止负时间预指令",
        "solver",
        "verifier",
        "certified_feasible",
        "certified_infeasible",
        "unresolved",
        "results/q1_rebuild/",
        "experimental_counterfactual",
        "synthetic",
    ):
        assert phrase in foundation


def test_research_docs_do_not_reintroduce_unified_or_frozen_solver_claims() -> None:
    stage_2 = read_text("docs/workflow/stage-02-problem-analysis.md")
    stage_3 = read_text("docs/workflow/stage-03-model-selection.md")
    decision_record = read_text("docs/modeling/model-selection-decision-record.md")
    evidence_map = read_text("docs/research/model-and-algorithm-evidence-map.md")
    research_design = read_text("docs/plans/2026-07-31-gate-b-research-design.md")

    assert r"\min(N_{\rm bomb},E_{\rm uav})" not in stage_2
    assert "只批准框架，不冻结主优化器" in decision_record
    assert "解析降维—DE—局部精修—独立认证" not in stage_3
    assert "当前证据优先级：认证任务包 + 滚动 MILP" not in evidence_map
    assert "SIP 是问题结构判断，不是算法已经完成" in stage_3
    assert "zero-forecast、whole-package-commitment" in stage_3
    assert "PSO 仅保留为启发式对照、不作为默认主方法" in decision_record
    assert "`current_stage=3`" in research_design
    assert "`next_stage=4`" in research_design
    assert "current_stage` 停在 4" not in research_design


def test_replaced_model_documents_distinguish_current_contract_from_history() -> None:
    current_paths = (
        "docs/modeling/assumption_register.md",
        "docs/modeling/model_contract_v0.1.md",
    )
    for path in current_paths:
        content = read_text(path)
        assert "状态：Stage 4 正式冻结" in content
        assert "瞬时纯追踪" in content

    legacy_paths = (
        "docs/modeling/revised_four_problem_architecture.md",
        "docs/modeling/q1_analytic_feasibility.md",
    )
    forbidden_claims = (
        "状态：当前有效",
        "当前有效的冻结假设层",
        "实现不得绕过本契约",
        "不得绕过这些假设",
    )

    for path in legacy_paths:
        content = read_text(path)
        assert "状态：历史快照，已解冻，不得作为当前正式实现依据。" in content
        assert all(claim not in content for claim in forbidden_claims)


def test_toy_readme_matches_the_actual_synthetic_abstraction_levels() -> None:
    readme = read_text("experiments/toy_demos/README.md")

    assert "每个 toy 使用人工可判定的 synthetic 代理问题" in readme
    assert "不得声称其为正式题面实例" in readme
    for question in ("Q1", "Q2", "Q3", "Q4"):
        assert f"| {question} " in readme
