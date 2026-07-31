# Stage 8 Paper Finalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在合并后的 `main` 基线上直接编写并验证一份受结果证据约束的 `cumcmthesis` LaTeX 论文，形成 Stage 8 Draft PR。

**Architecture:** 论文以 `paper/main.tex` 为入口，按章节拆分到 `paper/sections/`，复用冻结的 `results/` 与 `figures/`，通过 `paper/paper_audit.md` 记录论文声明到代码和产物的证据链。构建使用 XeLaTeX/latexmk；AI 工具说明独立放入 `supporting_materials/`。

**Tech Stack:** XeLaTeX, `cumcmthesis.cls`, BibTeX, PowerShell/Windows batch, Python project checks, GitHub PR API.

---

### Task 1: Freeze the evidence inputs

**Files:**
- Read: `results/q1_rebuild/*.csv`, `results/q1_rebuild/*.json`
- Read: `results/q2_rebuild/*.csv`, `results/q2_rebuild/*.json`
- Read: `results/q3_rebuild/*.csv`, `results/q3_rebuild/*.json`
- Read: `results/q4_rebuild/*.csv`, `results/q4_rebuild/*.json`
- Read: `results/sensitivity_rebuild/*.csv`, `results/sensitivity_rebuild/*.json`
- Read: `src/smoke_defense/q1_rebuild.py`, `q2_rebuild.py`, `q3_rebuild.py`, `q4_rebuild.py`

**Step 1:** Extract the paper-facing metrics and algorithm wording.

**Step 2:** Check each chosen number against both CSV and JSON where available.

**Step 3:** Record allowed claims and prohibited overclaims in the audit table.

### Task 2: Create the LaTeX paper skeleton

**Files:**
- Create: `paper/main.tex`
- Create: `paper/cumcmthesis.cls`
- Create: `paper/latexmkrc`
- Create: `paper/build_paper.bat`
- Create: `paper/build_paper.ps1`
- Create: `paper/references.bib`
- Create: `paper/sections/*.tex`
- Create: `paper/figures/README.md`

**Step 1:** Use `\\documentclass[withoutpreface,bwprint]{cumcmthesis}` and put the abstract first.

**Step 2:** Include sections for restatement, analysis, assumptions/symbols, Q1, Q2, Q3, Q4, sensitivity, evaluation, references and appendix.

**Step 3:** Link only frozen baseline figures; do not generate new scientific figures.

### Task 3: Write the evidence audit and AI report

**Files:**
- Create: `paper/paper_audit.md`
- Create: `supporting_materials/AI工具使用详情.tex`
- Create: `supporting_materials/build_ai_report.bat`

**Step 1:** Build the declaration-to-evidence table with evidence level and allowed wording.

**Step 2:** State that Q1–Q4 algorithms/results were not changed in Stage 8.

**Step 3:** Record ChatGPT/Codex names, uses, prompts, outputs, adopted content and human edits.

### Task 4: Compile and inspect

**Files:**
- Output: `paper/main.pdf`
- Output: `supporting_materials/AI工具使用详情.pdf`

**Step 1:** Run the Windows build script with XeLaTeX and BibTeX/latexmk.

**Step 2:** Check page count, equations, cross-references, figures, bibliography count, Chinese fonts, PDF size and absolute paths.

**Step 3:** Run all existing tests, Ruff, schema, scenario and artifact checks; resolve only Stage 8 presentation/build issues.

### Task 5: Publish the Draft PR

**Step 1:** Review `git diff --check` and confirm no scientific-result files changed.

**Step 2:** Commit and push `agent/stage8-paper-finalization`.

**Step 3:** Create a new Draft PR targeting `main`; do not merge it or upload to a contest platform.

**Step 4:** Verify the remote PR and clean working tree, then report all required identifiers and audit findings.
