# PRD: PSO portion of the Natural Computing assignment

**Date:** 2026-10-07
**Status:** Draft
**Template version:** default/0.2.1
**References:** Supplied assignment, starter code, and PSO lecture notes
**Plugin version:** 0.5.3
**Agent Sessions:** codex: 01a116c0-a614-7561-b57a-99bd92e51eb3

---

## TL;DR

Complete the Particle Swarm Optimisation (PSO) portion of the Natural Computing Flappy Bird assignment so the student can hand in working Python code and a report written in LaTeX and delivered as PDF. The work examines controller size and pipe gap, discusses measured results, and reviews a relevant dimensionality paper. A working PSO experiment ships first; the full report and hand-in files follow. Report locations for Random Search (RS) and Differential Evolution (DE) remain empty placeholders.

## Overview

- **Purpose:** Produce the PSO code, experiments, evidence, and report required by the assignment.
- **User Value:** The student can run and inspect a PSO solution, explain its measured behavior, and hand in the required files.
- **Scope:** Flappy Bird controller optimisation with PSO, experiments varying state representation and pipe gap, a dimensionality literature review, and the supplied report template in LaTeX and PDF.

## User Stories

| Story | Description | Priority |
|-------|-------------|----------|
| S1: Run PSO | Find and evaluate a Flappy Bird controller using PSO | P1 |
| S2: Compare conditions | Measure PSO across controller dimensions and pipe gaps | P1 |
| S3: Submit findings | Produce the required LaTeX report, PDF, and Python code | P1 |

### Story 1: Run PSO

- **As a** student
- **I want to** run PSO on the supplied Flappy Bird problem and inspect its best controller
- **So that** I have a working solution and measured fitness to report

#### Acceptance Scenarios

**Scenario 1: Happy Path**
- **Given** a valid simulation setup and experiment budget
- **When** I run PSO
- **Then** the run finishes with a best controller, its fitness, and the configuration needed to repeat the run
- **And** I can evaluate or display that controller using the supplied problem interface

**Scenario 2: Error Path**
- **Given** an invalid setup or unavailable game dependency
- **When** I try to run PSO
- **Then** I receive a plain-language error naming the problem and the action needed to correct it

#### Edge Cases

| Edge Case | Scenario | Expected Behavior |
|-----------|----------|-------------------|
| In progress | A run has not finished | Its active status and progress or budget used are visible. |
| Small valid budget | A budget sufficient to evaluate at least one controller is selected | It produces a valid evaluated controller. A budget allowing zero evaluations is rejected before starting. |
| Variable outcomes | The same controller is evaluated in stochastic episodes | The evaluation protocol is stated; one episode is not presented as a guaranteed score. |

#### Error Handling

| Error Condition | User Message | System Action |
|-----------------|--------------|---------------|
| Invalid run setting | “The selected [setting name] value is invalid. Use [supported range or values] and run again.” | Name the affected setting and its supported values; do not start the invalid run. |
| Missing dependency | “The Flappy Bird dependency is unavailable. Install the required packages and run again.” | Stop cleanly. |

### Story 2: Compare conditions

- **As a** student
- **I want to** compare PSO results across controller dimensions and pipe gaps
- **So that** I can answer how problem size and difficulty affect solution quality

#### Acceptance Scenarios

**Scenario 1: Happy Path**
- **Given** a working PSO run
- **When** I run the planned conditions
- **Then** results cover the descriptive state and multiple lidar sizes, with varied pipe gaps
- **And** each result identifies its state representation, dimension, gap, PSO settings, evaluation budget, and fitness outcome

**Scenario 2: Error Path**
- **Given** an invalid condition or a failed simulation
- **When** that condition is run
- **Then** it is identifiable and cannot be mistaken for a valid measured result

#### Edge Cases

| Edge Case | Scenario | Expected Behavior |
|-----------|----------|-------------------|
| In progress | Some conditions remain pending | Completed results are identifiable; pending ones are not final findings. |
| Equal dimensions | Two state choices have the same dimension | Each result still identifies its state choice. |
| Poor controller | A condition yields weak behavior | Its measured fitness is retained and discussed. |

#### Error Handling

| Error Condition | User Message | System Action |
|-----------------|--------------|---------------|
| Invalid condition | “This experiment condition is invalid. Check the state size or pipe gap.” | Stop the experiment before recording results for that condition. |
| Simulation failure | “This condition failed. Review the error and rerun it.” | Exclude it from valid comparisons until rerun. |

### Story 3: Submit findings

- **As a** student
- **I want to** hand in the required code and report files
- **So that** the PSO portion can be reviewed against the assignment

#### Acceptance Scenarios

**Scenario 1: Happy Path**
- **Given** completed PSO experiments and a selected dimensionality paper
- **When** I prepare the hand-in
- **Then** I have readable Python code and a report written in the supplied LaTeX template and available as PDF
- **And** the report has a revised abstract and introduction, problem description, research questions, experiment setup, labeled figures or tables tied to measured results, discussion, dimensionality literature review, short conclusion, and references
- **And** the report is at most eight pages excluding references
- **And** RS and DE locations remain empty placeholders with no claimed implementation, result, or conclusion

**Scenario 2: Error Path**
- **Given** missing results, an uncompiled report, or a report over the page limit
- **When** I check the hand-in
- **Then** the missing or invalid deliverable is identified before submission

#### Edge Cases

| Edge Case | Scenario | Expected Behavior |
|-----------|----------|-------------------|
| In progress | Report content is being assembled | Draft status is clear and hand-in is not described as complete. |
| Excluded approaches | The template refers to RS or DE | Empty placeholders preserve those locations without content. |
| Unsuitable source | A paper does not address optimisation dimensionality | It is not used as the required literature review. |

#### Error Handling

| Error Condition | User Message | System Action |
|-----------------|--------------|---------------|
| PDF unavailable | “The report PDF is missing. Compile the LaTeX report before submission.” | Mark the report incomplete. |
| Page limit exceeded | “The report exceeds eight pages excluding references. Shorten it before submission.” | Mark the report incomplete. |
| Results missing | “The PSO results are missing. Run the remaining conditions and update the report.” | Mark the report incomplete. |

## Story Dependencies

- S2 depends on S1 because comparisons require a working PSO run.
- S3 depends on S2 because the report needs measured results and discussion.

## Functional Requirements

- **FR-001:** The code MUST use PSO to optimise a real-valued Flappy Bird controller for the supplied problem.
- **FR-002:** A run MUST expose its best evaluated controller and fitness, and the setup needed to repeat it.
- **FR-003:** Experiments MUST compare the descriptive state with multiple lidar reading counts and report the resulting controller dimensions.
- **FR-004:** Experiments MUST vary pipe gap and report simulation and PSO settings for each result.
- **FR-005:** Results MUST include labeled figures or tables that identify measured conditions, units or fitness values, and a discussion of what they show, including weak outcomes.
- **FR-006:** The report MUST formulate research questions about PSO performance, dimensionality, and problem difficulty, then answer them from the experiments.
- **FR-007:** The report MUST review at least one identifiable paper that investigates how dimensionality affects discrete or continuous optimisation, cite it, and cover its problem, performance measure, main finding, critique, and possible improvements.
- **FR-008:** The report MUST use the provided LaTeX template, include the assignment's required report sections, compile to PDF, and stay within eight pages excluding references.
- **FR-009:** The hand-in MUST include Python code as a Python file with clear variable names and instructions sufficient to run the reported experiment, and the report as a PDF.
- **FR-010:** Report locations reserved for RS and DE MUST remain empty placeholders; no RS or DE code, experiments, findings, or prose are required.
- **FR-011:** Invalid runs and incomplete deliverables MUST be distinguishable from valid results and complete hand-in files.

## Success Metrics

| Metric | What it tells us | Direction |
|--------|------------------|-----------|
| Required hand-in files present, read from the hand-in directory | The specified files exist | At least one Python file and one PDF |
| PSO execution, read from a fresh run of the hand-in code | The submitted optimiser works | One completed run producing an evaluated controller and fitness |
| Repetition record, read from the result table or run records | The measured evidence can be checked | Every reported result identifies its setup and evaluation budget |
| Experiment coverage, read from the result table | Dimension and difficulty are examined | Descriptive state, at least two lidar counts, at least two pipe gaps |
| Report completeness, checked in the compiled PDF | The required PSO report content is present | All required sections |
| Guardrail: pages before references, read from the compiled PDF | Protects the assignment page limit | At most 8 |
| Guardrail: RS or DE substantive content, checked in code and report | Protects the requested exclusion | 0 |

## Release Slicing

| Release | Contents (stories / FRs) | Rationale for this cut |
|---------|--------------------------|------------------------|
| v1 | S1; FR-001, FR-002, FR-011 | A working PSO run gives the first usable result. |
| fast-follow | S2, S3; FR-003 through FR-010 | Completes experiments and all hand-in files. |
| later | None | No further work is requested. |

## Non-Functional Requirements

- The report must stay within eight pages excluding references.
- Results must be interpretable despite stochastic pipe generation: state the evaluation protocol and avoid claims from an unexplained single episode.
- The Python file must be readable, with clear names and concise comments where needed, as required by the assignment.

## Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| Supplied Flappy Bird problem and example | Required | Available |
| Game packages named by the starter code | Required | Installation not verified |
| Supplied LaTeX report template and bibliography | Required | Available |
| Suitable dimensionality paper | Required | Selection pending |
| LaTeX compilation environment | Required | Availability not verified |

## Out of Scope

- Random Search implementation, experiments, results, and explanatory content.
- Differential Evolution implementation, experiments, results, and explanatory content.
- Submission to Brightspace itself.

## Decisions

| ID | Title | Context | Options | Choice | Rationale | Source | FR impact | Consequences |
|----|-------|---------|---------|--------|-----------|--------|-----------|--------------|
| D-1 | PSO-only work | The assignment names three optimisers. | All three; PSO only | PSO only | The user explicitly excluded RS and DE. | User request | FR-001, FR-010 | Cross-optimiser analysis cannot be claimed. |
| D-2 | Complete hand-in | The assignment requires code and a report. | Code only; code and report | Code and report | The user requested everything specified for hand-in. | User clarification | FR-008, FR-009 | Report production is in scope. |
| D-3 | LaTeX report | The supplied report is a LaTeX template. | Another format; LaTeX | LaTeX source and PDF output | The user explicitly requested LaTeX. | User clarification | FR-008 | The report must compile. |
| D-4 | Empty excluded sections | The template refers to RS and DE. | Remove references; preserve placeholders | Preserve empty placeholders | The user explicitly requested placeholders. | User clarification | FR-010 | Excluded locations contain no substantive content. |

## Review record

CAP_REACHED — round-2 findings fixed, not re-reviewed.

- “The minimum supported budget” is never defined, so the valid boundary and rejection behavior cannot be tested externally.
- “This setting is invalid. Choose a supported value and run again.” does not satisfy “an error naming the problem and the action needed to correct it” because it identifies neither the affected setting nor its supported values.
