# Particle Swarm Optimisation

`pso.py` is the entry point for deterministic checks, headless Flappy Bird
smoke tests, runtime measurement, individual runs and the experiment suite.

The optimiser maximises reward with inertia weighted global best PSO. The
defaults are `w = 0.7298` and `c1 = c2 = 1.49618`. Positions start uniformly
in `[-1, 1]`, matching `GymProblem.sample()`, and are clipped to this interval
after every update. Velocities start uniformly in `[-0.2, 0.2]` and are
clipped to the same interval after every update. Every objective call,
including initial swarm evaluation, consumes one unit of the budget.

Run the checks and all experiments with:

```text
python3 pso.py all
```

The experiment command accepts the pipe gap, state type, lidar count, swarm
size, PSO coefficients, evaluation budget and random seeds. Use `python3
pso.py experiments -h` to see every setting. If the budget is omitted, a
small budget is selected after measuring episode runtime. The default suite
runs three independent optimiser seeds for each setting.

Each experiment run creates a new timestamped directory under `pso_results`.
It contains the runtime probe, complete settings, one JSON file per trial and
a CSV summary. Trial JSON stores the evaluation indexed history, returned
weights, best fitness observed during search and fresh episode fitness values.

The PSO seed reproduces optimiser random numbers on a deterministic objective.
It does not reproduce Flappy Bird episodes. `GymProblem` creates a new
environment and calls `reset()` without a seed, so pipe sequences remain
unseeded. Search best fitness and fresh controller fitness must therefore be
reported separately.

Comparisons across lidar counts also change the number of weights and the
information available to the controller. Both effects must be considered when
interpreting the results.
