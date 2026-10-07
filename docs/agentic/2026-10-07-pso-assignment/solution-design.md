# Solution Design: PSO Flappy Bird assignment

**Date:** 2026-10-07
**Artifact:** solution-design
**Status:** Draft
**Template version:** default/0.3.1
**PRD:** [Approved PRD](prd.md)
**References:** Supplied game code, report template, PSO lecture notes, [DarkMachines dimensionality study](https://arxiv.org/abs/2101.04525)
**Plugin version:** 0.5.3
**Agent Sessions:** codex: 01a116ce-7e1c-7063-9685-393463413814

---

## Summary 🔍

The assignment needs a working Particle Swarm Optimisation (PSO) controller, measured comparisons across state sizes and pipe gaps, and a short report. A readable Python hand-in will extend the supplied game class with a seed argument, implement inertia-weighted PSO, and record search and independent evaluation results. A small pilot will establish runtime and the full experiment budget. The supplied LaTeX report will use the measurements and review a published dimensionality study.

## Background & problem 🔍

The supplied game class maps a real-valued weight matrix to flap or idle actions and returns one episode's reward. It currently resets without a seed, so repeated fitness calls cannot reproduce the same pipe sequence. There is no PSO code or completed report. The report must distinguish the descriptive state from lidar states even when their nominal dimensions match.

## Goals and non-goals

### Goals

**G1: Run a reproducible PSO search**
- [ ] G1-AC1 — A valid run returns its best evaluated weight vector, mean search fitness, seed, state choice, pipe gap, particle count, PSO coefficients, episode count, and episode budget. `verify: run a small search twice with the same seed and compare records`
- [ ] G1-AC2 — The search uses inertia, personal best, and global best updates; every candidate is scored by the mean of a fixed number of seeded episodes. `verify: inspect a deterministic small run and its evaluation count`
- [ ] G1-AC3 — A budget below one complete candidate evaluation, invalid state size or gap, or missing game dependency fails with the affected setting and corrective action; no result is marked complete. `verify: exercise invalid inputs and missing-import path`

**G2: Produce comparable measurements**
- [ ] G2-AC1 — A pilot records seconds per episode and estimates the full grid's cost before full runs start. `verify: inspect pilot record and derived budget`
- [ ] G2-AC2 — Completed records cover descriptive state, at least two lidar sizes, and at least two pipe gaps; each gives nominal dimension, setup, seed, episodes used, search fitness, and fresh evaluation fitness. `verify: validate result table against condition grid`
- [ ] G2-AC3 — Pending and failed conditions have explicit status and cannot enter figures or summary statistics; weak valid results remain. `verify: inject a failed condition and inspect report inputs`

**G3: Produce the PSO hand-in**
- [ ] G3-AC1 — The Python file runs the reported experiment with clear settings and instructions; the LaTeX report uses the supplied template and cites a relevant dimensionality study. `verify: run documented command and inspect source and bibliography`
- [ ] G3-AC2 — The PDF has required sections, research questions, labeled measured results, discussion, paper critique, and a short conclusion; pages before references do not exceed eight. `verify: compile and inspect rendered PDF and page break`
- [ ] G3-AC3 — Random Search and Differential Evolution locations remain empty placeholders, with no claimed implementation or results. `verify: inspect Python and rendered report`
- [ ] G3-AC4 — Hand-in validation reports actual presence and page count of the PDF and Python file and the measured condition count; missing artifacts cannot yield a complete status. `verify: remove a copy of each required artifact or record and run the check`

### Non-goals

- Implementing or evaluating Random Search or Differential Evolution.
- Changing the controller formula or the game's existing episode ending behavior.
- Submitting to Brightspace.

## Architecture

Add an optional seed parameter to the supplied game class's episode method, passing it to environment reset. Keep its observation, action, reward, and termination logic. A separate hand-in Python file owns configuration validation, seeded episode scoring, PSO, pilot timing, condition runs, records, and a completion check. PSO maximizes mean episode reward using a global-best topology and the lecture's inertia-weighted update. Positions start in [-1, 1] per the supplied sampler; movement is clipped to those bounds, with a stated velocity cap. These are algorithm settings, not a claim that the real-valued search space is inherently bounded.

Use a seeded schedule common to candidates in each search run so their scores face the same initial random conditions. Each independent run gets its own search seeds. After search, score the returned controller on fresh seeds never used for ranking candidates. Record both scores, because selecting the best noisy score can overstate performance. The pilot uses a small valid swarm and budget, then observed runtime determines a feasible full budget before comparisons. The full grid fixes PSO settings and episode budget across conditions; only state representation, lidar count, pipe gap, and run seed vary.

```mermaid
flowchart LR
  C[Validated condition] --> P[Seeded PSO search]
  P --> R[Best weights and search record]
  R --> E[Fresh seeded episodes]
  E --> T[Completed result table]
  T --> L[LaTeX report and PDF]
```

## Components

- **`problem.py`**: accept an optional episode seed and pass it to game reset; preserve existing game behavior.
- **`pso_assignment.py`**: validate settings, evaluate controllers, run PSO and the pilot, execute the condition grid, persist status and result records, and provide the documented command entry point.
- **`assignment_latex/Assignment.tex`**: replace instructional text with the PSO report, measured tables or figures, and empty excluded-method placeholders.
- **`assignment_latex/bibliography.bib`**: add complete metadata for the dimensionality study and other cited sources.
- **`assignment_latex/` output PDF**: compiled report delivered with the Python hand-in.

## Data flow

One condition specifies state choice, lidar count where applicable, gap, and run seed. The game class supplies state size, so the search vector has `state_size × 2` weights. The scorer evaluates one vector on the condition's fixed seed list and returns mean reward. PSO records the best vector and a progress row after each complete candidate evaluation. Fresh evaluation seeds produce a separate final mean and spread. Status records distinguish pending, running, complete, and failed conditions; only complete records feed figures and conclusions.

## Error handling

An interrupted run leaves completed records intact and the active condition marked incomplete. A simulation exception marks that condition failed with its settings and error, and later conditions can still be run. Dependency import errors name the missing package and installation command. A LaTeX compilation failure or missing PDF leaves the hand-in incomplete with the compiler error available for correction.

## Testing & verification

Inspect the first completed pilot before selecting the full grid's repetitions and episode budget. Check that the same seed reproduces an episode in the installed game package; if it does not, record the limitation and do not describe results as deterministic. The local environment currently lacks the Flappy Bird package and `pdflatex`, so integration and PDF verification belong in the implementation phase after dependencies are available.

## Decisions 🔍

| ID | Title | Context | Options | Choice | Rationale | Source | Goal impact | Consequences |
|----|-------|---------|---------|--------|-----------|--------|-------------|--------------|
| D-1 | Fitness estimate | Pipe generation makes a single episode noisy. | One episode; fixed repeated episodes; conditional reevaluation | Fixed repeated episodes averaged per candidate | Gives each candidate equal effort and an interpretable episode budget. | user | G1-AC2, G2-AC2 | More runtime per candidate. |
| D-2 | Budget selection | Episode duration depends on controller behavior. | Fixed large budget; pilot first; demonstration only | Time a small pilot, then choose full budget | Runtime evidence supports a feasible comparison grid. | user | G2-AC1 | Full budget is chosen after the pilot and recorded before full runs. |
| D-3 | Game seeding | The supplied class resets without a seed. | Wrapper; edit supplied class; no seed | Add an optional seed argument to the supplied class | Direct request; keeps the game interface simple. | user | G1-AC1, G1-AC2 | Requires a small edit to supplied code. |
| D-4 | Episode ending | The supplied game ends on termination or truncation. | Add step cap; preserve behavior | Preserve behavior | Direct request. | user | G2-AC1 | Pilot must account for long episodes. |
| D-5 | PSO form and bounds | The lecture offers several PSO variants; code needs finite initialization. | Original update; inertia-weighted global best; constricted PSO | Inertia-weighted global best with fixed bounds and velocity cap | The lecture provides concrete inertia settings, and the starter sampler uses [-1, 1]. Check whether clipping dominates in the pilot. | auto (lecture notes and starter code) | G1-AC2 | Report coefficients, cap, and bounds. |
| D-6 | Independent final score | Selecting a best noisy fitness can inflate reported result. | Report search score only; score fresh episodes too | Report both search and fresh episode scores | Separates selection from evidence of general performance. | auto (PRD stochasticity requirement) | G2-AC2 | More episodes after each run. |
| D-7 | Dimensionality source | Report needs an identifiable paper about optimisation dimension. | Unspecified paper; DarkMachines comparison | Review the DarkMachines high-dimensional optimisation comparison | Its [published abstract](https://arxiv.org/abs/2101.04525) identifies analytic functions of varying dimension and includes PSO. Read the full paper before writing the critique. | auto (PRD and primary paper) | G3-AC1, G3-AC2 | Literature section distinguishes the paper's benchmarks from Flappy Bird. |

## Deviations from PRD 🔍

None.

## Key facts & assumptions

The state choices must be labeled separately even when both have 24 nominal weights. Changing lidar count also changes available information, so dimension alone cannot explain any observed difference. Seeded reset is expected to control pipe generation, but this remains unverified until the game dependency is installed. The best search score and fresh evaluation score must never be conflated.
