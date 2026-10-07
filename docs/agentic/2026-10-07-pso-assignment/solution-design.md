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

The assignment needs a working Particle Swarm Optimisation (PSO) controller, measured comparisons across state sizes, pipe gaps, and PSO settings, and a short report. A readable Python hand-in will extend the supplied game class with a seed argument and implement inertia-weighted PSO. A small pilot will guide a feasible evaluation budget. The supplied LaTeX report will use the measurements and review a published dimensionality study.

## Background & problem 🔍

The supplied game class maps a real-valued weight matrix to flap or idle actions and returns one episode's reward. It currently resets without a seed, so repeated fitness calls cannot reproduce the same pipe sequence. There is no PSO code or completed report. The report must distinguish the descriptive state from lidar states even when their nominal dimensions match.

## Goals and non-goals

### Goals

**G1: Run a reproducible PSO search**
- [ ] G1-AC1 — A valid run returns its best evaluated weight vector and fitness together with the seed, state choice, pipe gap, PSO settings, and episode budget. `verify: run a small search twice with the same seed and compare outputs`
- [ ] G1-AC2 — The search uses inertia, personal best, and global best updates; every candidate is scored by the mean of a fixed number of seeded episodes. `verify: inspect a deterministic small run and its evaluation count`
- [ ] G1-AC3 — A budget below one complete candidate evaluation, invalid state size or gap, or missing game dependency fails with the affected setting and corrective action. During a run, console progress shows episodes used and the current best fitness. `verify: exercise invalid inputs and inspect a small run's output`

**G2: Produce comparable measurements**
- [ ] G2-AC1 — A small pilot records episode runtime and estimates how many complete swarm updates fit within the runtime limit supplied for the experiment; the selected budget is stated before comparisons. `verify: inspect pilot timing, runtime limit, and budget calculation`
- [ ] G2-AC2 — Measured results cover descriptive state, at least two lidar counts, at least two pipe gaps, and at least two PSO parameter settings. Every result identifies state choice, nominal dimension, gap, PSO settings, episode budget, and fitness. `verify: inspect the results table against the planned comparisons`
- [ ] G2-AC3 — Failed or unfinished runs are excluded from reported measurements, while weak completed results remain. `verify: inspect source measurements and report table`

**G3: Produce the PSO hand-in**
- [ ] G3-AC1 — The Python file runs the reported experiment with clear settings and instructions; the LaTeX report uses the supplied template and cites a relevant dimensionality study. `verify: run documented command and inspect source and bibliography`
- [ ] G3-AC2 — The PDF has a revised abstract and introduction, problem description, experiment setup, labeled figures or tables, discussion of strong and weak outcomes, short conclusion, and references. It asks and answers research questions about PSO performance, dimension, and pipe gap; its paper review covers the application, optimisation problem, performance measure, main finding, critique, possible improvements, and relevant literature. Every student's name and the group number replace template placeholders; pages before references do not exceed eight. `verify: compile and inspect rendered PDF and page break`
- [ ] G3-AC3 — Random Search and Differential Evolution locations remain empty placeholders, with no claimed implementation or results. `verify: inspect Python and rendered report`
- [ ] G3-AC4 — The Python file and compiled PDF are present, and the PDF's page count before references is checked. `verify: inspect hand-in files and PDF`

### Non-goals

- Implementing or evaluating Random Search or Differential Evolution.
- Changing the controller formula or the game's existing episode ending behavior.
- Submitting to Brightspace.

## Architecture

Add an optional seed parameter to the supplied game class's episode method, passing it to environment reset. Keep its observation, action, reward, and termination logic. A separate hand-in Python file owns input validation, episode scoring, PSO, pilot timing, and comparison runs. PSO maximizes mean episode reward using a global-best topology and the lecture's inertia-weighted update. Positions start in [-1, 1] per the supplied sampler and are clipped to those stated bounds. These are algorithm settings, not a claim that the real-valued search space is inherently bounded.

Use the same fixed, positive number of seeded episodes for each candidate within a comparison. The hand-in command accepts that count, a positive particle count, and a runtime limit; its defaults are two episodes and five particles. A small pilot with those settings measures seconds per complete swarm update. Divide the runtime limit by the pilot time, round down to whole swarm updates, and report that estimated budget before comparisons. Reject a limit too small for one update. The estimate is advisory because later controllers may survive longer. Use the same episode budget across compared conditions. At minimum, compare descriptive state and two lidar counts at gap 100, then descriptive state at one other gap. To meet the assignment's optimiser-parameter requirement, compare the baseline inertia setting with a lower inertia setting on the descriptive state at gap 100, keeping other settings, episode seeds, and budget equal.

Use baseline inertia 0.7298 and cognitive and social coefficients 1.49618 each, as in the lecture notes; use 0.4 for the lower-inertia comparison. Expose particle count and position bounds as settings. Clip new positions to the bounds. Count each game episode against the evaluation budget. The report states the final chosen settings, pilot result, and any limits of the comparison.

```mermaid
flowchart LR
  C[Validated condition] --> P[Seeded PSO search]
  P --> R[Best weights and search record]
  R --> T[Measured result table]
  T --> L[LaTeX report and PDF]
```

## Components

- **`problem.py`**: accept an optional episode seed and pass it to game reset; preserve existing game behavior.
- **`pso_assignment.py`**: validate settings, evaluate controllers, run PSO and the pilot, execute comparisons, print or save measured results, and provide the documented command entry point.
- **`assignment_latex/Assignment.tex`**: replace instructional text with the PSO report, measured tables or figures, and empty excluded-method placeholders.
- **`assignment_latex/bibliography.bib`**: add complete metadata for the dimensionality study and other cited sources.
- **`assignment_latex/` output PDF**: compiled report delivered with the Python hand-in.

## Data flow

One condition specifies state choice, lidar count where applicable, gap, PSO settings, and run seed. The game class supplies state size, so the search vector has `state_size × 2` weights. The scorer evaluates one vector on fixed episode seeds and returns mean reward. The best vector, measured fitness, and setup feed the report table. A failed run yields an error and no measured result.

## Error handling

An invalid setting or simulation failure names the affected condition and does not produce a valid measurement. An interrupted run's console output states how many episodes finished, without presenting that run as a final result. Dependency import errors name the missing package and installation command. A LaTeX compilation failure is corrected before treating the PDF as a hand-in file.

## Testing & verification

Inspect the pilot and budget calculation before comparison runs. Check that the same seed reproduces an episode in the installed game package; if it does not, record the limitation and do not describe results as deterministic. The local environment currently lacks the Flappy Bird package and `pdflatex`, so integration and PDF verification belong in the implementation phase after dependencies are available. Obtain the student names and group number before final compilation.

## Decisions 🔍

| ID | Title | Context | Options | Choice | Rationale | Source | Goal impact | Consequences |
|----|-------|---------|---------|--------|-----------|--------|-------------|--------------|
| D-1 | Fitness estimate | Pipe generation makes a single episode noisy. | One episode; fixed repeated episodes; conditional reevaluation | Fixed repeated episodes averaged per candidate | Gives each candidate equal effort and an interpretable episode budget. | user | G1-AC2, G2-AC2 | More runtime per candidate. |
| D-2 | Budget selection | Episode duration depends on controller behavior. | Fixed large budget; pilot first; demonstration only | Pilot, then calculate whole swarm updates under a supplied runtime limit | User chose a small pilot first; a supplied limit avoids imposing an arbitrary duration. | user for pilot; auto (runtime calculation) for budget rule | G2-AC1 | Estimated budget is fixed and stated before comparisons. |
| D-3 | Game seeding | The supplied class resets without a seed. | Wrapper; edit supplied class; no seed | Add an optional seed argument to the supplied class | Direct request; keeps the game interface simple. | user | G1-AC1, G1-AC2 | Requires a small edit to supplied code. |
| D-4 | Episode ending | The supplied game ends on termination or truncation. | Add step cap; preserve behavior | Preserve behavior | Direct request. | user | G2-AC1 | Pilot must account for long episodes. |
| D-5 | PSO form and bounds | The lecture offers several PSO variants; code needs finite initialization. | Original update; inertia-weighted global best; constricted PSO | Inertia-weighted global best with configurable particle count and bounded positions | The lecture supplies baseline coefficients, and the starter sampler uses [-1, 1]. | auto (lecture notes and starter code) | G1-AC2 | Report chosen particle count, coefficients, and bounds. |
| D-6 | Parameter comparison | The assignment asks for variation in optimiser parameters. | One fixed setting; vary inertia; larger parameter grid | Compare inertia 0.7298 and 0.4 on descriptive state at gap 100 | Varying one coefficient isolates its effect without multiplying every condition. | auto (assignment and lecture notes) | G2-AC2 | Report both settings and the limited scope of this comparison. |
| D-7 | Dimensionality source | Report needs an identifiable paper about optimisation dimension. | Unspecified paper; DarkMachines comparison | Review the DarkMachines high-dimensional optimisation comparison | Its [published abstract](https://arxiv.org/abs/2101.04525) identifies analytic functions of varying dimension and includes PSO. Read the full paper before writing the critique. | auto (PRD and primary paper) | G3-AC1, G3-AC2 | Literature section distinguishes the paper's benchmarks from Flappy Bird. |

## Deviations from PRD 🔍

None.

## Key facts & assumptions

The state choices must be labeled separately even when both have 24 nominal weights. Changing lidar count also changes available information, so dimension alone cannot explain any observed difference. Seeded reset is expected to control pipe generation, but this remains unverified until the game dependency is installed. A single seeded episode is not a guaranteed score.
