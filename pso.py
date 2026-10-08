"""Run and evaluate Particle Swarm Optimisation on the supplied GymProblem."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from problem import GymProblem


DEFAULT_INERTIA = 0.7298
DEFAULT_COGNITIVE = 1.49618
DEFAULT_SOCIAL = 1.49618
DEFAULT_VELOCITY_LIMIT = 0.2


@dataclass(frozen=True)
class PSOConfig:
    dimensions: int
    swarm_size: int = 8
    evaluation_budget: int = 30
    inertia: float = DEFAULT_INERTIA
    cognitive: float = DEFAULT_COGNITIVE
    social: float = DEFAULT_SOCIAL
    position_min: float = -1.0
    position_max: float = 1.0
    velocity_limit: float = DEFAULT_VELOCITY_LIMIT
    seed: int = 1

    def validate(self) -> None:
        if self.dimensions < 1:
            raise ValueError("dimensions must be positive")
        if self.swarm_size < 1:
            raise ValueError("swarm_size must be positive")
        if self.evaluation_budget < self.swarm_size:
            raise ValueError("evaluation_budget must be at least swarm_size")
        if self.position_min >= self.position_max:
            raise ValueError("position_min must be less than position_max")
        if self.velocity_limit <= 0:
            raise ValueError("velocity_limit must be positive")


@dataclass
class PSOResult:
    best_weights: np.ndarray
    best_fitness: float
    evaluations: int
    history: list[dict[str, int | float | str]]


@dataclass(frozen=True)
class GameConfig:
    state_type: str = "lidar"
    n_lidar: int = 8
    pipe_gap: int = 100

    @property
    def use_lidar(self) -> bool:
        if self.state_type not in {"lidar", "features"}:
            raise ValueError("state_type must be lidar or features")
        return self.state_type == "lidar"


def advance_swarm(
    positions: np.ndarray,
    velocities: np.ndarray,
    personal_best_positions: np.ndarray,
    global_best_position: np.ndarray,
    config: PSOConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one inertia weighted, global best PSO update."""
    r1 = rng.random(positions.shape)
    r2 = rng.random(positions.shape)
    new_velocities = (
        config.inertia * velocities
        + config.cognitive * r1 * (personal_best_positions - positions)
        + config.social * r2 * (global_best_position - positions)
    )
    new_velocities = np.clip(
        new_velocities,
        -config.velocity_limit,
        config.velocity_limit,
    )
    new_positions = np.clip(
        positions + new_velocities,
        config.position_min,
        config.position_max,
    )
    return new_positions, new_velocities


def run_pso(objective: Callable[[np.ndarray], float], config: PSOConfig) -> PSOResult:
    """Maximise an objective while counting every call against the budget."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    positions = rng.uniform(
        config.position_min,
        config.position_max,
        size=(config.swarm_size, config.dimensions),
    )
    velocities = rng.uniform(
        -config.velocity_limit,
        config.velocity_limit,
        size=positions.shape,
    )
    personal_best_positions = positions.copy()
    personal_best_fitness = np.full(config.swarm_size, -np.inf)
    global_best_position = positions[0].copy()
    global_best_fitness = -np.inf
    history: list[dict[str, int | float | str]] = []
    evaluation = 0
    generation = 0

    while evaluation < config.evaluation_budget:
        if generation > 0:
            positions, velocities = advance_swarm(
                positions,
                velocities,
                personal_best_positions,
                global_best_position.reshape(1, -1),
                config,
                rng,
            )

        for particle_index in range(config.swarm_size):
            if evaluation >= config.evaluation_budget:
                break
            fitness = float(objective(positions[particle_index].copy()))
            evaluation += 1
            if fitness > personal_best_fitness[particle_index]:
                personal_best_fitness[particle_index] = fitness
                personal_best_positions[particle_index] = positions[particle_index]
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = positions[particle_index].copy()
            history.append(
                {
                    "evaluation": evaluation,
                    "generation": generation,
                    "particle": particle_index,
                    "fitness": fitness,
                    "best_fitness": global_best_fitness,
                }
            )
        generation += 1

    return PSOResult(
        best_weights=global_best_position,
        best_fitness=global_best_fitness,
        evaluations=evaluation,
        history=history,
    )


def make_problem(config: GameConfig) -> GymProblem:
    return GymProblem(
        use_lidar=config.use_lidar,
        n_lidar=config.n_lidar,
        pipe_gap=config.pipe_gap,
    )


def close_problem(problem: GymProblem) -> None:
    env = getattr(problem, "env", None)
    if env is not None:
        env.close()


def run_game_trial(
    game: GameConfig,
    swarm_size: int,
    evaluation_budget: int,
    inertia: float,
    cognitive: float,
    social: float,
    seed: int,
    fresh_evaluations: int,
) -> dict:
    problem = make_problem(game)
    try:
        pso_config = PSOConfig(
            dimensions=int(problem.n_variables),
            swarm_size=swarm_size,
            evaluation_budget=evaluation_budget,
            inertia=inertia,
            cognitive=cognitive,
            social=social,
            seed=seed,
        )
        started = time.perf_counter()
        result = run_pso(problem, pso_config)
        search_seconds = time.perf_counter() - started
        fresh_fitness = [float(problem(result.best_weights)) for _ in range(fresh_evaluations)]
    finally:
        close_problem(problem)

    return {
        "game": asdict(game),
        "pso": asdict(pso_config),
        "bounds": {
            "position": [pso_config.position_min, pso_config.position_max],
            "initial_velocity": [
                -pso_config.velocity_limit,
                pso_config.velocity_limit,
            ],
            "velocity_clamp": [
                -pso_config.velocity_limit,
                pso_config.velocity_limit,
            ],
            "position_rule": "clip after each update",
        },
        "search_best_fitness": result.best_fitness,
        "fresh_fitness": fresh_fitness,
        "fresh_mean_fitness": statistics.fmean(fresh_fitness),
        "returned_weights": result.best_weights.tolist(),
        "weight_count": int(result.best_weights.size),
        "history": result.history,
        "evaluations": result.evaluations,
        "search_seconds": search_seconds,
        "reproducibility": (
            "The PSO seed fixes optimiser randomness only. GymProblem creates a new "
            "environment and resets it without a seed, so pipe sequences remain unseeded."
        ),
    }


def deterministic_verification() -> dict:
    target = np.array([0.25, -0.5, 0.75])
    calls = 0

    def objective(weights: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return -float(np.sum((weights - target) ** 2))

    config = PSOConfig(
        dimensions=3,
        swarm_size=4,
        evaluation_budget=11,
        seed=1234,
    )
    first = run_pso(objective, config)
    first_calls = calls
    calls = 0
    second = run_pso(objective, config)
    if first_calls != config.evaluation_budget or calls != config.evaluation_budget:
        raise AssertionError("fitness call count does not match the budget")
    if len(first.history) != config.evaluation_budget:
        raise AssertionError("history length does not match the budget")
    if not np.array_equal(first.best_weights, second.best_weights):
        raise AssertionError("the deterministic objective was not reproduced")
    if first.history != second.history:
        raise AssertionError("the deterministic history was not reproduced")
    return {
        "status": "passed",
        "evaluation_budget": config.evaluation_budget,
        "fitness_calls": first_calls,
        "best_fitness": first.best_fitness,
    }


def smoke_verification() -> dict:
    checks = []
    for game in (
        GameConfig(state_type="lidar", n_lidar=5, pipe_gap=100),
        GameConfig(state_type="features", n_lidar=5, pipe_gap=100),
    ):
        problem = make_problem(game)
        try:
            expected_weights = 2 * problem.state_size
            if problem.n_variables != expected_weights:
                raise AssertionError("weight count is not twice the observation count")
            weights = np.zeros(problem.n_variables)
            fitness = float(problem(weights))
            checks.append(
                {
                    "state_type": game.state_type,
                    "observation_count": int(problem.state_size),
                    "weight_count": int(problem.n_variables),
                    "fitness": fitness,
                }
            )
        finally:
            close_problem(problem)
    return {"status": "passed", "checks": checks}


def measure_episode_runtime(repeats: int = 3) -> dict:
    rng = np.random.default_rng(8001)
    measurements = []
    for game in (
        GameConfig(state_type="lidar", n_lidar=8, pipe_gap=100),
        GameConfig(state_type="features", n_lidar=8, pipe_gap=100),
    ):
        problem = make_problem(game)
        try:
            for repeat in range(repeats):
                weights = rng.uniform(-1.0, 1.0, problem.n_variables)
                started = time.perf_counter()
                fitness = float(problem(weights))
                elapsed = time.perf_counter() - started
                measurements.append(
                    {
                        "state_type": game.state_type,
                        "repeat": repeat,
                        "seconds": elapsed,
                        "fitness": fitness,
                    }
                )
        finally:
            close_problem(problem)
    durations = [row["seconds"] for row in measurements]
    return {
        "measurements": measurements,
        "median_seconds": statistics.median(durations),
        "maximum_seconds": max(durations),
        "warning": (
            "GymProblem has no episode time limit. Better controllers can run much longer "
            "than these random runtime probes."
        ),
    }


def choose_budget(runtime: dict, largest_swarm: int) -> int:
    median_seconds = max(float(runtime["median_seconds"]), 0.001)
    runtime_based = int(2.0 / median_seconds)
    return max(largest_swarm, min(30, max(12, runtime_based)))


def experiment_specs(include_extended_sensitivity: bool) -> list[dict]:
    specs = []
    for n_lidar in (4, 8, 12):
        specs.append(
            {
                "suite": "observation",
                "label": f"lidar_{n_lidar}",
                "game": GameConfig("lidar", n_lidar, 100),
            }
        )
    specs.append(
        {
            "suite": "observation",
            "label": "features_12",
            "game": GameConfig("features", 8, 100),
        }
    )
    for pipe_gap in (80, 100, 120):
        specs.append(
            {
                "suite": "pipe_gap",
                "label": f"gap_{pipe_gap}",
                "game": GameConfig("lidar", 8, pipe_gap),
            }
        )

    sensitivity = [
        ("swarm_4", 4, DEFAULT_INERTIA, DEFAULT_COGNITIVE, DEFAULT_SOCIAL),
        ("baseline", 6, DEFAULT_INERTIA, DEFAULT_COGNITIVE, DEFAULT_SOCIAL),
        ("swarm_8", 8, DEFAULT_INERTIA, DEFAULT_COGNITIVE, DEFAULT_SOCIAL),
        ("inertia_0.5", 6, 0.5, DEFAULT_COGNITIVE, DEFAULT_SOCIAL),
        ("social_2.0", 6, DEFAULT_INERTIA, DEFAULT_COGNITIVE, 2.0),
    ]
    if not include_extended_sensitivity:
        sensitivity = sensitivity[:3]
    for label, swarm_size, inertia, cognitive, social in sensitivity:
        specs.append(
            {
                "suite": "sensitivity",
                "label": label,
                "game": GameConfig("lidar", 8, 100),
                "swarm_size": swarm_size,
                "inertia": inertia,
                "cognitive": cognitive,
                "social": social,
            }
        )
    return specs


def make_output_directory(root: Path, label: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / f"{stamp}_{label}"
    suffix = 1
    while output.exists():
        output = root / f"{stamp}_{label}_{suffix}"
        suffix += 1
    output.mkdir(parents=True)
    return output


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_summary(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = [
        "trial_id",
        "suite",
        "label",
        "seed",
        "state_type",
        "n_lidar",
        "pipe_gap",
        "observation_count",
        "weight_count",
        "swarm_size",
        "evaluation_budget",
        "inertia",
        "cognitive",
        "social",
        "search_best_fitness",
        "fresh_mean_fitness",
        "fresh_fitness",
        "search_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_summary(path: Path, rows: Sequence[dict]) -> None:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["suite"], row["label"]), []).append(row)
    fieldnames = [
        "suite",
        "label",
        "trials",
        "mean_search_best_fitness",
        "stdev_search_best_fitness",
        "mean_fresh_fitness",
        "stdev_fresh_fitness",
        "minimum_fresh_fitness",
        "maximum_fresh_fitness",
        "mean_search_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for suite, label in sorted(grouped):
            group = grouped[(suite, label)]
            search_values = [float(row["search_best_fitness"]) for row in group]
            fresh_values = [float(row["fresh_mean_fitness"]) for row in group]
            elapsed_values = [float(row["search_seconds"]) for row in group]
            writer.writerow(
                {
                    "suite": suite,
                    "label": label,
                    "trials": len(group),
                    "mean_search_best_fitness": statistics.fmean(search_values),
                    "stdev_search_best_fitness": (
                        statistics.stdev(search_values) if len(search_values) > 1 else 0.0
                    ),
                    "mean_fresh_fitness": statistics.fmean(fresh_values),
                    "stdev_fresh_fitness": (
                        statistics.stdev(fresh_values) if len(fresh_values) > 1 else 0.0
                    ),
                    "minimum_fresh_fitness": min(fresh_values),
                    "maximum_fresh_fitness": max(fresh_values),
                    "mean_search_seconds": statistics.fmean(elapsed_values),
                }
            )


def run_experiments(args: argparse.Namespace, runtime: dict) -> Path:
    include_extended = float(runtime["median_seconds"]) <= 0.25
    specs = experiment_specs(include_extended)
    largest_swarm = max(spec.get("swarm_size", args.swarm_size) for spec in specs)
    budget = args.budget or choose_budget(runtime, largest_swarm)
    if budget < largest_swarm:
        raise ValueError("budget is smaller than a configured swarm")
    output = make_output_directory(args.output, "experiments")
    write_json(output / "runtime_measurement.json", runtime)
    settings = {
        "trial_seeds": args.seeds,
        "fresh_evaluations": args.fresh_evaluations,
        "evaluation_budget": budget,
        "sensitivity_setting_count": sum(
            spec["suite"] == "sensitivity" for spec in specs
        ),
        "sensitivity_selection": (
            "five settings because median probe runtime was at most 0.25 seconds"
            if include_extended
            else "three settings because median probe runtime exceeded 0.25 seconds"
        ),
        "specifications": [
            {
                **{key: value for key, value in spec.items() if key != "game"},
                "game": asdict(spec["game"]),
            }
            for spec in specs
        ],
        "interpretation_note": (
            "Changing n_lidar changes both controller weight count and available information."
        ),
        "reproducibility_limit": (
            "PSO seeds do not seed GymProblem resets, so game pipe sequences differ."
        ),
    }
    write_json(output / "experiment_settings.json", settings)

    summary_rows = []
    trial_id = 0
    for spec in specs:
        for seed in args.seeds:
            trial_id += 1
            swarm_size = spec.get("swarm_size", args.swarm_size)
            trial = run_game_trial(
                game=spec["game"],
                swarm_size=swarm_size,
                evaluation_budget=budget,
                inertia=spec.get("inertia", args.inertia),
                cognitive=spec.get("cognitive", args.cognitive),
                social=spec.get("social", args.social),
                seed=seed,
                fresh_evaluations=args.fresh_evaluations,
            )
            trial["trial_id"] = trial_id
            trial["suite"] = spec["suite"]
            trial["label"] = spec["label"]
            write_json(output / f"trial_{trial_id:03d}.json", trial)
            summary_rows.append(
                {
                    "trial_id": trial_id,
                    "suite": spec["suite"],
                    "label": spec["label"],
                    "seed": seed,
                    "state_type": trial["game"]["state_type"],
                    "n_lidar": trial["game"]["n_lidar"],
                    "pipe_gap": trial["game"]["pipe_gap"],
                    "observation_count": trial["weight_count"] // 2,
                    "weight_count": trial["weight_count"],
                    "swarm_size": trial["pso"]["swarm_size"],
                    "evaluation_budget": trial["evaluations"],
                    "inertia": trial["pso"]["inertia"],
                    "cognitive": trial["pso"]["cognitive"],
                    "social": trial["pso"]["social"],
                    "search_best_fitness": trial["search_best_fitness"],
                    "fresh_mean_fitness": trial["fresh_mean_fitness"],
                    "fresh_fitness": json.dumps(trial["fresh_fitness"]),
                    "search_seconds": trial["search_seconds"],
                }
            )
            print(
                f"trial {trial_id}/{len(specs) * len(args.seeds)} "
                f"{spec['suite']} {spec['label']} seed={seed} "
                f"search={trial['search_best_fitness']:.1f} "
                f"fresh={trial['fresh_mean_fitness']:.1f}",
                flush=True,
            )
    write_summary(output / "summary.csv", summary_rows)
    write_aggregate_summary(output / "aggregate_summary.csv", summary_rows)
    return output


def run_single(args: argparse.Namespace, runtime: dict) -> Path:
    game = GameConfig(args.state_type, args.n_lidar, args.pipe_gap)
    budget = args.budget or choose_budget(runtime, args.swarm_size)
    trial = run_game_trial(
        game=game,
        swarm_size=args.swarm_size,
        evaluation_budget=budget,
        inertia=args.inertia,
        cognitive=args.cognitive,
        social=args.social,
        seed=args.seed,
        fresh_evaluations=args.fresh_evaluations,
    )
    output = make_output_directory(args.output, "single")
    write_json(output / "runtime_measurement.json", runtime)
    write_json(output / "trial_001.json", trial)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("verify", "runtime", "run", "experiments", "all"),
    )
    parser.add_argument("--output", type=Path, default=Path("pso_results"))
    parser.add_argument("--state-type", choices=("lidar", "features"), default="lidar")
    parser.add_argument("--n-lidar", type=int, default=8)
    parser.add_argument("--pipe-gap", type=int, default=100)
    parser.add_argument("--swarm-size", type=int, default=6)
    parser.add_argument("--inertia", type=float, default=DEFAULT_INERTIA)
    parser.add_argument("--cognitive", type=float, default=DEFAULT_COGNITIVE)
    parser.add_argument("--social", type=float, default=DEFAULT_SOCIAL)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--fresh-evaluations", type=int, default=2)
    parser.add_argument("--runtime-repeats", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    args = build_parser().parse_args(argv)
    if args.command in {"verify", "all"}:
        verification = {
            "deterministic": deterministic_verification(),
            "flappy_smoke": smoke_verification(),
        }
        print(json.dumps(verification, indent=2))
        if args.command == "verify":
            return 0

    runtime = measure_episode_runtime(args.runtime_repeats)
    print(json.dumps(runtime, indent=2))
    if args.command == "runtime":
        return 0
    if args.command == "run":
        output = run_single(args, runtime)
    else:
        output = run_experiments(args, runtime)
    print(f"saved results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
