# Q2 Multi-Smoke Coverage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Task 8: certified joint spatial-temporal coverage for one shipborne UAV carrying at most three smoke bombs, including reproducible results for all 144 formal scenarios.

**Architecture:** Extend the shared coverage layer before building the Q2 optimizer. Generate diverse single-smoke events, connect selected releases with one continuous fixed-speed UAV path, enumerate or branch-and-bound combinations, and rank only independently certified plans using the frozen lexicographic objective.

**Tech Stack:** Python 3.11, NumPy, SciPy, Shapely, Pydantic v2, Pytest, Ruff.

---

## Execution rules

- Follow @superpowers:test-driven-development for every production behavior.
- Preserve model contract v0.2, A-001–A-022, and Q1 ranking/results.
- Do not start Q3, Q4, or Task 11.
- Commit after each task only after its targeted tests, quick full tests,
  Ruff, and `git diff --check` pass.
- The optimizer may propose candidates; only the shared verifier may return
  `certified_feasible`.

### Task 1: Bound the exact joint gap at one time

**Files:**
- Modify: `src/smoke_defense/coverage.py`
- Modify: `tests/test_coverage.py`

**Step 1: Write failing spatial-union tests**

Add tests for:

```python
def test_union_gap_is_nonpositive_when_only_joint_union_covers():
    ship = Disk(np.zeros(2), 80.0)
    smokes = (
        Disk(np.array([-50.0, 0.0]), 110.0),
        Disk(np.array([50.0, 0.0]), 110.0),
    )
    result = evaluate_union_gap_at_time(
        ship, smokes, spatial_tolerance_m=0.05
    )
    assert all(single_smoke_gap(
        ship.center_m, smoke.center_m, smoke.radius_m
    ) > 0 for smoke in smokes)
    assert result.status is CertificationStatus.CERTIFIED_FEASIBLE
    assert result.upper_bound_m <= 0


def test_union_gap_returns_exact_positive_witness():
    result = evaluate_union_gap_at_time(
        Disk(np.zeros(2), 80.0),
        (
            Disk(np.array([-60.0, 0.0]), 50.0),
            Disk(np.array([60.0, 0.0]), 50.0),
        ),
        spatial_tolerance_m=0.05,
    )
    assert result.status is CertificationStatus.CERTIFIED_INFEASIBLE
    assert result.lower_bound_m > 0
    assert result.witness_m is not None


def test_low_polygon_cap_is_indeterminate_not_feasible():
    result = evaluate_union_gap_at_time(
        nearly_tangent_ship,
        nearly_tangent_smokes,
        initial_polygon_sides=4,
        maximum_polygon_sides=4,
        spatial_tolerance_m=1e-6,
    )
    assert result.status is CertificationStatus.INDETERMINATE
```

Also retain the existing exact single-smoke test.

**Step 2: Run the tests and observe RED**

Run:

```powershell
python -m pytest tests/test_coverage.py -v
```

Expected: collection fails because `evaluate_union_gap_at_time` does not exist.

**Step 3: Implement bounded joint-gap evaluation**

Add frozen `UnionGapEvaluation` with:

```python
status: CertificationStatus
lower_bound_m: float
upper_bound_m: float
estimated_gap_m: float
spatial_error_m: float
polygon_sides: int | None
witness_m: np.ndarray | None
reason: str
```

Implement monotone radius-offset bisection:

```python
P(delta) = target subset union_j B(center_j, radius_j + delta)
g <= delta iff P(delta)
```

Use the existing polygon certificate for each decision. Never route one smoke
through polygon approximation; use `single_smoke_gap` exactly.

**Step 4: Run GREEN and regression tests**

Run:

```powershell
python -m pytest tests/test_coverage.py tests/test_q1.py -v
python -m ruff check src/smoke_defense/coverage.py tests/test_coverage.py
git diff --check
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add -- src/smoke_defense/coverage.py tests/test_coverage.py
git commit -m "feat: bound multi-smoke union gap"
```

### Task 2: Certify multi-smoke coverage over continuous time

**Files:**
- Modify: `src/smoke_defense/verification.py`
- Modify: `tests/test_verification.py`

**Step 1: Write failing continuous-time tests**

Test:

- two smokes jointly cover while neither independently covers;
- pre-burst closed endpoint is infeasible;
- burst time is checked with post-jump radius;
- hold end and failure are explicit split points;
- a positive gap returns time and space witnesses;
- an unresolved time cell returns `indeterminate_at_tolerance`.

Desired API:

```python
result = certify_multi_smoke_coverage(
    ship_position=ship.position,
    smokes=smokes,
    detection_components=detection.components,
    source_events=detection.source_events,
    ship_radius_m=80.0,
    ship_speed_bound_mps=7.71,
    spatial_tolerance_m=0.05,
    time_tolerance_s=1e-3,
)
```

Assert fields for maximum joint gap, minimum margin, total exposure,
maximum continuous exposure, worst interval, and witnesses.

**Step 2: Run RED**

```powershell
python -m pytest tests/test_verification.py -v
```

Expected: import failure for `certify_multi_smoke_coverage`.

**Step 3: Implement event-split adaptive certification**

Add frozen records for:

- `CoverageModeCertificate`;
- `MultiSmokeCoverageCertificate`.

Build split points from detection component endpoints, all detection source
events, and every smoke burst/hold/failure time. Evaluate event times
individually. Inside continuous modes, evaluate endpoints and midpoint and use:

```python
upper = max(sample.upper_bound_m) + L_g * width / 4
lower = max(sample.lower_bound_m)
```

Recursively split unresolved cells. Mark a cell indeterminate at the requested
time tolerance instead of treating it as covered. Preserve the existing
single-smoke continuous function and tests unchanged.

**Step 4: Run GREEN and regression**

```powershell
python -m pytest tests/test_verification.py tests/test_coverage.py tests/test_q1.py -v
python -m ruff check src/smoke_defense/verification.py tests/test_verification.py
git diff --check
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add -- src/smoke_defense/verification.py tests/test_verification.py
git commit -m "feat: certify multi-smoke continuous coverage"
```

### Task 3: Define strict Q2 models and single-UAV release path

**Files:**
- Create: `src/smoke_defense/q2.py`
- Create: `tests/test_q2.py`
- Modify: `src/smoke_defense/paths.py`
- Modify: `tests/test_paths.py`

**Step 1: Write failing model and path tests**

Create tests for:

- all Q2 Pydantic models are frozen and reject extra fields;
- one plan accepts 1–3 bombs and rejects 0 or 4;
- release spacing exactly 1 s passes and 0.999 s fails;
- pre-takeoff waiting follows the ship and adds no flight distance;
- one continuous path visits every release point;
- every airborne segment has speed exactly 28 m/s;
- the path continues through the final burst time;
- the burst center equals release point plus 3.5 s times release velocity;
- individually reachable events that cannot be linked are rejected.

Desired connector:

```python
path = build_ordered_release_path(
    ship=ship,
    takeoff_time_s=takeoff,
    releases=releases,
    continue_until_s=last_burst,
)
```

**Step 2: Run RED**

```powershell
python -m pytest tests/test_q2.py tests/test_paths.py -v
```

Expected: `q2` and ordered connector imports fail.

**Step 3: Implement minimal strict models**

Use Pydantic `ConfigDict(extra="forbid", frozen=True)` for:

- `VectorEvent`;
- `MultiSmokeCandidate`;
- `PathNode`;
- `OrderedReleasePlan`;
- `MultiSmokeCertificate`;
- `IndependentCoverageBaseline`;
- `MultiSmokePlan`;
- `Q2ScenarioResult`;
- `Q2SweepResult`.

Use tuples for persisted vectors and enum/literal status fields.

**Step 4: Implement ordered path connection**

Generalize the existing terminal-heading construction from
`build_shipborne_release_path`. For each release:

1. start from the previous release or ship-at-takeoff;
2. consume exactly \(28\Delta t\) path length;
3. end at the release point with the required incoming heading;
4. concatenate continuous positive-duration linear segments;
5. append a 28 m/s post-release segment through the last burst time.

Validate release positions against `path.position(t_release)` and use the
incoming segment velocity for detonation geometry.

**Step 5: Run GREEN**

```powershell
python -m pytest tests/test_q2.py tests/test_paths.py tests/test_path_constraints.py -v
python -m ruff check src/smoke_defense/q2.py src/smoke_defense/paths.py tests/test_q2.py
git diff --check
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add -- src/smoke_defense/q2.py src/smoke_defense/paths.py tests/test_q2.py tests/test_paths.py
git commit -m "feat: add q2 models and ordered uav path"
```

### Task 4: Generate a diverse, traceable Q2 candidate library

**Files:**
- Modify: `src/smoke_defense/q2.py`
- Modify: `src/smoke_defense/candidates.py`
- Modify: `tests/test_q2.py`
- Modify: `tests/test_candidates.py`

**Step 1: Write failing diversity tests**

Assert that a library contains:

- multiple coverage-center times;
- multiple burst centers, including nonzero lateral offsets;
- more than the Q1 winner;
- single-smoke-incomplete candidates;
- event, reachability, scenario hash, constants hash, and single-coverage data.

Add a dominance test proving only candidates with no worse time/space,
margin, and distance attributes are removed. Add a test that an individually
incomplete candidate survives.

**Step 2: Run RED**

```powershell
python -m pytest tests/test_q2.py -k "candidate or dominance" -v
```

Expected: Q2 candidate-library functions are absent.

**Step 3: Implement library generation**

Refactor reusable Q1 construction helpers without changing Q1 defaults.
Generate burst centers in ship-heading coordinates using a bounded structural
grid that includes paired lateral offsets capable of true union coverage.
Construct and validate each release event independently, then persist all
required hashes and certificates.

**Step 4: Implement evidence-based pruning**

Prune only:

- exact duplicate physical events;
- strict dominance at matching geometry/time bins;
- confirmed response, reachability, or operation-radius failures.

Do not prune because single-smoke full coverage is absent.

**Step 5: Run GREEN and Q1 regression**

```powershell
python -m pytest tests/test_candidates.py tests/test_q2.py tests/test_q1.py -v
python -m ruff check src/smoke_defense/candidates.py src/smoke_defense/q2.py tests
git diff --check
```

Expected: PASS and Q1 candidate/ranking tests remain unchanged.

**Step 6: Commit**

```powershell
git add -- src/smoke_defense/candidates.py src/smoke_defense/q2.py tests/test_candidates.py tests/test_q2.py
git commit -m "feat: generate diverse q2 smoke candidates"
```

### Task 5: Enumerate combinations and apply strict lexicographic ranking

**Files:**
- Modify: `src/smoke_defense/q2.py`
- Modify: `tests/test_q2.py`

**Step 1: Write failing optimizer tests**

Test:

- all 1-, 2-, and 3-smoke combinations are considered on a small library;
- path-incompatible combinations are rejected;
- every retained combination calls the public verifier;
- true spatial union beats the independent-interval baseline;
- two-smoke plan improves the best one-smoke maximum exposure;
- maximum exposure beats margin, margin beats bomb count, and bomb count
  beats distance;
- physical tie tolerances suppress sub-nanosecond noise;
- exhaustive enumeration and branch-and-bound return the same plan.

**Step 2: Run RED**

```powershell
python -m pytest tests/test_q2.py -k "enumerat or lexicographic or branch" -v
```

Expected: optimizer functions are absent.

**Step 3: Implement enumeration**

Implement:

```python
enumerate_q2_combinations(...)
evaluate_q2_combination(...)
select_best_q2_plan(...)
```

For each ordered combination:

1. validate count and spacing;
2. build one path;
3. certify 12 km operation radius;
4. validate all release and burst geometry;
5. call `certify_multi_smoke_coverage`;
6. compute independent-interval baseline;
7. build a validated `MultiSmokePlan`.

**Step 4: Implement lexicographic key and branch-and-bound**

Rank:

1. certified full coverage;
2. lower maximum continuous exposure;
3. higher minimum joint margin;
4. fewer bombs;
5. shorter flight distance.

Use time and distance quantization based on declared physical tolerances.
Branch-and-bound may prune only on admissible lower/upper bounds and must
match exhaustive enumeration on the test library.

**Step 5: Run GREEN**

```powershell
python -m pytest tests/test_q2.py -v
python -m pytest -q
python -m ruff check .
git diff --check
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add -- src/smoke_defense/q2.py tests/test_q2.py
git commit -m "feat: optimize q2 smoke combinations"
```

### Task 6: Solve formal and ablation Q2 scenario sweeps

**Files:**
- Modify: `src/smoke_defense/q2.py`
- Create: `scripts/run_q2.py`
- Modify: `tests/test_q2.py`

**Step 1: Write failing sweep tests**

Test:

- formal Q2 matrix contains exactly 144 inertial scenarios;
- ablation contains exactly 16 and is stored separately;
- every result includes scenario/constant hashes and model provenance;
- every selected plan has an independent verifier certificate;
- deterministic seed `20260730` reproduces candidate ordering and selections.

**Step 2: Run RED**

```powershell
python -m pytest tests/test_q2.py -k "scenario or sweep or provenance" -v
```

Expected: sweep/serialization functions are absent.

**Step 3: Implement scenario solver and serialization**

Reuse Q1 trajectory/detection construction through a shared non-mutating
helper or a narrow public wrapper. Implement:

```python
solve_q2_scenario(...)
solve_all_q2_sweeps(...)
write_q2_results(...)
write_q2_markdown_summary(...)
```

Persist solver/verifier configuration, tolerances, pruning statistics,
events, path nodes, certificates, witnesses, Q1 baseline comparison, and
worst inertial combinations.

**Step 4: Add the runner**

`scripts/run_q2.py` must:

- use seed `20260730`;
- write `results/q2/q2_results.json`;
- write `results/q2/README.md`;
- print formal/ablation counts and elapsed time;
- support deterministic regeneration.

**Step 5: Run GREEN**

```powershell
python -m pytest tests/test_q2.py -v
python scripts/run_q2.py
python -m ruff check .
git diff --check
```

Expected: 144 formal and 16 ablation scenarios are written separately.

**Step 6: Commit**

```powershell
git add -- src/smoke_defense/q2.py scripts/run_q2.py tests/test_q2.py results/q2
git commit -m "results: record certified q2 guidance sweep"
```

### Task 7: Performance, full regression, and reproducibility

**Files:**
- Modify only files demonstrated by profiling or failing regression tests
- Test: all existing tests and generated artifacts

**Step 1: Establish performance evidence**

Run:

```powershell
Measure-Command { python scripts/run_q2.py }
```

If runtime is excessive, profile candidate generation, path compatibility,
spatial certificates, and repeated time evaluations. Add a failing
performance/regression test before any semantic code change.

**Step 2: Apply semantics-preserving optimization only if needed**

Allowed:

- cache immutable disk/polygon evaluations;
- cache candidate compatibility;
- vectorize distance calculations;
- deterministic admissible pruning;
- reuse scenario trajectory/detection objects.

Forbidden:

- lower verification precision;
- delete adverse candidates;
- replace continuous certification with a fixed grid;
- run only favorable scenarios.

**Step 3: Run every user gate**

```powershell
python -m pytest -q
python -m ruff check .
python scripts/export_scenario_schema.py --check
python scripts/generate_scenarios.py --check
python scripts/run_q1.py
python scripts/run_q2.py
git diff --check
```

Expected: all commands exit 0.

**Step 4: Prove Q1 result stability**

Before regenerating Q1, save its tracked hash. After regeneration compare all
Q1 scientific fields while allowing only an intentional Git SHA change if
the current script records it. Any other difference requires a failing
regression test and a documented root cause.

**Step 5: Commit any verified performance/regression fixes**

```powershell
git add -- <explicit intended paths>
git commit -m "perf: finalize q2 certified sweep"
```

### Task 8: Review, publish, and open the Draft PR

**Files:**
- Review: all changes since `44fd43b`
- Modify only files required by review findings

**Step 1: Review the complete diff**

Use @superpowers:requesting-code-review. Because this execution is
single-agent, perform an independent requirements audit against the attached
18-test checklist and inspect:

```powershell
git diff --stat 44fd43b..HEAD
git diff --check 44fd43b..HEAD
git log --oneline 44fd43b..HEAD
```

Fix every critical or important issue with a new failing test first.

**Step 2: Run fresh final verification**

Repeat all gates from Task 7 after the final fix. Do not rely on previous
outputs.

**Step 3: Check GitHub prerequisites**

```powershell
gh --version
gh auth status
git status --short --branch
```

**Step 4: Push**

```powershell
git push -u origin agent/q2-multi-smoke
```

**Step 5: Create Draft PR**

Title:

```text
实现 Q2 单机多烟幕联合时空覆盖
```

The body must cover the mathematical criterion, verifier, candidate and
combination algorithms, single-UAV path handling, tests, 144 formal results,
Q1 comparison, limitations, and files requiring human review. Do not merge.

**Step 6: Report**

Report branch/commit, Draft PR URL, verification and CI state, certified
feasible count, bomb-count distribution, Q1 exposure improvement,
indeterminate count, and review hotspots.
