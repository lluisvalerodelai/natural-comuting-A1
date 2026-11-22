"""
requirements:
    pip install flappy-bird-gymnasium pillow
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import partial
from importlib.metadata import version
from inspect import Parameter, signature
import io
from itertools import product
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import pygame
from PIL import Image

# All study settings and defaults in a dict
parameters = {
    "lidar_counts": (5, 25, 100),
    "gaps": (100, 150, 200),
    "focal_setting": ("lidar25", 150),
    "controllers": 16,  # 16 alternative controllers that take flap or idle actions
    "epochs": 12,  # 12 rounds of evaluating a controller
    "seed": 3,
    "layouts": 2,  # the seeded random layouts that we swap each game on for each run.
    "runs": 10,  # A complete optimization attempt: each algorithm, with an observation type (features, lidar) and one pipe gap setting is evaluated 10 times across 12 epochs over 16 controllers
    "held_out_layouts": 8,  # 8 unseen layouts are ran after the full optimizer workflow
    "frames": 1000,  # A study episode runs for 1000 frames in-game
    "workers": 4,  # default workers for multi-processing to speed it up
    "hill_step_length": 0.75,  # proposed local step length before clipping
    "de": {
        "differential_weight": 0.6,
        "crossover": 0.8,
        "mutation_base": "rand",
    },  # DE hyperparameters
    "pso": {
        "inertia": 0.7,
        "cognitive": 1.4,
        "social": 1.4,
        "speed_limit": 0.3,
        "topology": "global",
    },  # PSO hyperparameters
    "colours": {"RS": "silver", "DE": "blue", "PSO": "orange"},
    "markers": {"RS": "o", "DE": "s", "PSO": "^"},
}

# expose all features and settings in CLI
parameters["observations"] = (
    "features",
    *(f"lidar{n}" for n in parameters["lidar_counts"]),
)
# count initial candidates and all update rounds to make up the default budget
parameters["budget"] = parameters["controllers"] * (1 + parameters["epochs"])


# The Flappy Bird controller
class GymProblem:
    def __init__(
        self,
        env_name="FlappyBird-v0",
        use_lidar=True,
        n_lidar=12,
        pipe_gap=100,
        *,
        frames=None,
        watch=False,
    ):
        self.env_name = env_name  # game name
        self.pipe_gap = pipe_gap
        self.use_lidar = use_lidar  # lidar or features switch
        self.n_lidar = n_lidar  # store number of lidar readings
        self.frames = frames  # set episode frame limit when supplied by caller
        self.watch = watch  # enable visualization of the game to gauge performance
        self.env = self._make_env(
            "rgb_array" if watch else None
        )  # request RGB frames only when watched

        original_state_size = self.env.observation_space.shape[
            0
        ]  # read number of sensor values a game provides
        if self.use_lidar:
            self.lidar_indices = np.linspace(
                0, original_state_size - 1, self.n_lidar, dtype=int
            )
            self.state_size = self.n_lidar
        else:
            self.lidar_indices = None
            self.state_size = original_state_size

        self.n_outputs = int(
            self.env.action_space.n
        )  # convert to int for json compatibility
        self.n_variables = self.state_size * self.n_outputs

        self.M = np.zeros((self.state_size, self.n_outputs))
        self.rewards = []

        if watch:
            pygame.display.init()
            pygame.font.init()
            self.font = pygame.font.Font(None, 22)
            self.clock = pygame.time.Clock()

    # to build a rendering mode with same settings
    def _make_env(self, render_mode):
        # apply selected sensors, gap, render mode and episodes limit
        return gym.make(
            self.env_name,
            use_lidar=self.use_lidar,
            pipe_gap=self.pipe_gap,
            render_mode=render_mode,
            max_episode_steps=self.frames,
            disable_env_checker=True,
        )

    def process_observation(self, observation):
        if self.use_lidar:
            return observation[self.lidar_indices]
        return observation

    # draw a random controller sample
    def sample(self):
        return np.random.uniform(-1, 1, size=self.n_variables)

    # Update a frame of the replay window
    def _draw_replay(self, screen, total_reward, pipes):

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                raise KeyboardInterrupt

        frame = self.env.render()

        pygame.surfarray.blit_array(screen, frame.swapaxes(0, 1))

        status = f"Reward: {total_reward}, Pipes: {pipes}"

        screen.blit(self.font.render(status, True, "white", "black"), (6, 6))
        pygame.display.flip()
        self.clock.tick(30)

    # Play a game with fixed weights, default result is total reward
    def play_episode(
        self,
        x,
        render_mode=None,
        gif_path=None,
        fps=30,
        frame_skip=1,
        *,
        seed=None,
        caption="Flappy Bird Game",
        return_details=False,
    ):
        self.M = np.asarray(x).reshape(self.state_size, self.n_outputs)
        env = self.env if render_mode is None else self._make_env(render_mode)

        record = gif_path is not None and render_mode == "human"
        gif_frames = []

        try:
            observation, _ = env.reset(seed=seed)
            observation = self.process_observation(observation)
            screen = None

            if self.watch and render_mode is None:
                height, width = env.render().shape[:2]
                screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption(caption)

            self.rewards = []
            total_reward = 0.0

            while True:
                action = int(np.argmax(self.M.T @ observation))
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                self.rewards.append(reward)

                observation = self.process_observation(observation)

                if screen is not None:
                    self._draw_replay(screen, total_reward, info["score"])
                if record:
                    surface = pygame.display.get_surface()
                    if surface is not None:
                        gif_frames.append(
                            pygame.surfarray.array3d(surface).transpose(1, 0, 2)
                        )

                if terminated or truncated:
                    break
        finally:
            if render_mode is not None:
                env.close()

        if gif_frames:
            images = [Image.fromarray(frame) for frame in gif_frames[::frame_skip]]
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=int(1000 * frame_skip / fps),
                loop=0,
            )

        if return_details:
            return dict(
                seed=seed,
                reward=total_reward,
                pipes=int(info["score"]),
                steps=len(self.rewards),
                capped=truncated and not terminated,
            )

        return total_reward

    def __call__(self, x):
        return self.play_episode(x)

    def show(self, x, gif_path=None, fps=30, frame_skip=1, *, seed=None):
        return self.play_episode(x, "human", gif_path, fps, frame_skip, seed=seed)


# All Optimizers for the report
def random_search(evaluate, dimension, budget, rng):
    """
    Independent random sampling

    Args:
        evaluate: the evaluation function
        dimension: state_size [12 features, 5 lidar,25 lidar, 100 lidar] * n_outputs [idle, flap]
        budget: total budget = 16 controllers + (16 controllers * 12 epochs)
        rng: the random number geneator function
    """

    # Spend one evaluation on each draw
    for i in range(budget):
        weights = rng.uniform(
            -1, 1, dimension
        )  # draw a fresh value for every controller weight according to a dimension
        evaluate(weights)  # Play with the random controller


def random_hill_climb(
    evaluate,
    dimension,
    budget,
    rng,
    initial_samples=parameters["controllers"],
    step_length=parameters["hill_step_length"],
):
    """
    RS-HC = Combination of Random Sampling and Stochastic Hill Climbing

    Args:

    """

    if (
        dimension < 1
        or initial_samples < 1
        or not np.isfinite(step_length)
        or step_length <= 0
    ):
        raise ValueError("Invalid hyperparameters")

    best_score = -np.inf

    for i in range(budget):
        if i < initial_samples:
            candidate = rng.uniform(-1, 1, dimension)

        else:
            direction = rng.standard_normal(dimension)
            length = np.linalg.norm(direction)
            while length == 0:
                direction = rng.standard_normal(dimension)
                length = np.linalg.norm(direction)
            step = step_length * (direction / length)
            candidate = np.clip(best + step, -1, 1)

        score = evaluate(candidate)
        if score > best_score:
            best, best_score = candidate, score


def differential_evolution(
    evaluate,
    dimension,
    budget,
    rng,
    controllers=parameters["controllers"],
    differential_weight=parameters["de"]["differential_weight"],
    crossover=parameters["de"]["crossover"],
    mutation_base=parameters["de"]["mutation_base"],
):
    """
    Differential Evolution algorithms following DE rand1 bin and DE best1 bin
    Args:
        evaluate: Returns one controller's mean training reward.
        dimension: Number of weights per controller, varies per method (features / lidar)
        budget: Total candidate evaluations
        rng: Seeded numpy random generator
        controllers: Number of (parents) controllers in each generation, needs at least 4
        differential_weight: Mutation Scale F
        crossover: Probability from 0 to 1 to crossover a particular donor weight
        mutation_base: Selects parent A in donor formula. "rand" draws a random, "best" uses highest scoring one
    """

    if mutation_base not in ("rand", "best"):
        raise ValueError("Invalid mutation_base")

    if not 4 <= controllers <= budget:
        raise ValueError("DE requires at least 4 controllers with enough evaluations")

    positions = rng.uniform(-1, 1, (controllers, dimension))

    scores = np.array([evaluate(x) for x in positions], dtype=float)

    for start in range(controllers, budget, controllers):
        parents = positions.copy()

        best_index = int(np.argmax(scores))

        for target in range(min(controllers, budget - start)):
            pool = np.delete(np.arange(controllers), target)

            if mutation_base == "best":
                a = best_index
                b, c = rng.choice(pool[pool != a], size=2, replace=False)

            else:
                a, b, c = rng.choice(pool, size=3, replace=False)

            donor = parents[a] + differential_weight * (parents[b] - parents[c])

            mask = rng.random(dimension) < crossover

            mask[rng.integers(dimension)] = True

            trial = np.clip(np.where(mask, donor, parents[target]), -1, 1)
            score = evaluate(trial)
            if score >= scores[target]:
                positions[target] = trial
                scores[target] = score


# Particle Swarm Optimization
def particle_swarm(
    evaluate,
    dimension,
    budget,
    rng,
    controllers=parameters["controllers"],
    inertia=parameters["pso"]["inertia"],
    cognitive=parameters["pso"]["cognitive"],
    social=parameters["pso"]["social"],
    speed_limit=parameters["pso"]["speed_limit"],
    topology=parameters["pso"]["topology"],
):
    """
    Synchronous, inertia weighted PSO

    Args:
        evaluate: Return one controller's mean training reward.
        dimension: Number of weights per controller, varies per method (features / lidar)
        budget: Total candidate evaluations
        rng: Seeded numpy random generator
        controllers: Number of (particles) controllers in the swarm.
        inertia: Weight w on previous velocity, default of 0.7
        cognitive: Strength of the pull towards the partcile's personal best position
        social: Strenght of pull towards the leader selected by topology
        speed_limit: Maximum absolute velocity per weight, per update
        topology: Selects social leader. "global" uses swarm best record, "ring" uses best record among particle and immediate neighbors in a ring
    """

    if topology not in ("global", "ring"):
        raise ValueError("Invalid Topology")

    if not 1 <= controllers <= budget:
        raise ValueError("PSO requires positive swarm within budget")

    positions = rng.uniform(-1, 1, (controllers, dimension))

    velocities = rng.uniform(-speed_limit, speed_limit, positions.shape)
    personal_best = positions.copy()

    scores = np.array([evaluate(x) for x in positions], dtype=float)

    global_best = personal_best[np.argmax(scores)].copy()
    global_score = max(scores)

    if topology == "ring":
        particles = np.arange(controllers)
        neighbours = np.column_stack(
            (particles, np.roll(particles, 1), np.roll(particles, -1))
        )

    for start in range(controllers, budget, controllers):
        cognitive_pull = (
            cognitive * rng.random(positions.shape) * (personal_best - positions)
        )
        leader = global_best

        if topology == "ring":
            winners = np.argmax(scores[neighbours], axis=1)
            leader = personal_best[neighbours[particles, winners]]

        social_pull = social * rng.random(positions.shape) * (leader - positions)
        velocities = np.clip(
            inertia * velocities + cognitive_pull + social_pull,
            -speed_limit,
            speed_limit,
        )

        positions = np.clip(positions + velocities, -1, 1)
        for i in range(min(controllers, budget - start)):
            score = evaluate(positions[i])

            if score > scores[i]:
                scores[i] = score
                personal_best[i] = positions[i]

            if score > global_score:
                global_score = score
                global_best = positions[i].copy()


parameters["methods"] = {
    "RS": random_search,
    "DE": differential_evolution,
    "PSO": particle_swarm,
}

parameters["experiments"] = {
    "operators": {
        "RS-HC": (
            "RS",
            random_hill_climb,
            {"step_length": parameters["hill_step_length"]},
        ),
        "PSO-ring": ("PSO", particle_swarm, {"topology": "ring"}),
        "DE-best1": ("DE", differential_evolution, {"mutation_base": "best"}),
    },
    "parameters": {
        "DE-F0.3": ("DE", differential_evolution, {"differential_weight": 0.3}),
        "DE-F0.9": ("DE", differential_evolution, {"differential_weight": 0.9}),
        "PSO-w0.4": ("PSO", particle_swarm, {"inertia": 0.4}),
        "PSO-w0.9": ("PSO", particle_swarm, {"inertia": 0.9}),
    },
}

parameters["algorithms"] = {
    name: (name, function, {}) for name, function in parameters["methods"].items()
}
parameters["algorithms"].update(
    {
        label: spec
        for group in parameters["experiments"].values()
        for label, spec in group.items()
    }
)


# Experiment harness
def for_study(observation, gap, frames, watch=False):
    """
    Convert a study label (i.e. lidar25) into constructor arguments
    """

    if observation == "features":
        return GymProblem(use_lidar=False, pipe_gap=gap, frames=frames, watch=watch)

    count = observation.removeprefix("lidar")
    if count == observation or not count.isdigit() or not 1 <= int(count) <= 180:
        raise ValueError("Use features or lidarN with N between 1 and 180")

    return GymProblem(
        use_lidar=True, n_lidar=int(count), pipe_gap=gap, frames=frames, watch=watch
    )


def run_trial(
    method,
    observation,
    gap,
    run=1,
    options=None,
    *,
    budget=parameters["budget"],
    layouts=parameters["layouts"],
    held_out_layouts=parameters["held_out_layouts"],
    frames=parameters["frames"],
    seed=parameters["seed"],
    watch=False,
):
    """
    "Train" an optimizer on one task, evaluate the winner on unseen layouts
    """

    if min(run, budget, layouts, held_out_layouts, frames, gap) < 1:
        raise ValueError("Invalid settings")

    baseline, search, overrides = parameters["algorithms"][method]
    options = {**overrides, **(options {})}

    optimizer, layout_stream = np.random.SeedSequence([seed, run]).spawn(2)
    rng = np.random.default_rng(optimizer)
    layout_rng = np.random.default_rng(layout_stream)

    seeds = layout_rng.choice(
        2**31 - 1, layouts + held_out_layouts, replace=False
    ).tolist()
    train = seeds[:layouts]
    test = seeds[layouts:]

    problem = for_study(observation, gap, frames, watch)
    rewards = []
    candidates = []
    best_score = -np.inf
    best_weights = None

    def evaluate(weights):
        nonlocal best_score, best_weights
        caption = f"{method} | candidate {len(rewards) + 1}/{budget}"

        if rewards:
            caption += f" | best {best_score}"
        score = float(
            np.mean(
                [
                    problem.play_episode(weights, seed=layout, caption=caption)
                    for layout in train
                ]
            )
        )
        rewards.append(score)
        candidates.append(weights.tolist())

        if score > best_score:
            best_score = score
            best_weights = weights.copy()

        return score

    try:
        search(evaluate, problem.n_variables, budget, rng, **options)
        held_out = [
            problem.play_episode(
                best_weights,
                seed=layout,
                caption=f"{method} | held-out game {i}",
                return_details=True,
            )
            for i, layout in enumerate(test, 1)
        ]

    finally:
        problem.env.close()

    return dict(
        method=baseline,
        algorithm=method,
        parameters=options,
        observation=observation,
        gap=gap,
        run=run,
        dimension=problem.n_variables,
        train_seeds=train,
        scores=rewards,
        candidates=candidates,
        weights=best_weights.tolist(),
        held_out=held_out,
    )


def run_study(args):
    """
    Run all selected algorithms across the entire grid of parameters
    """
    budget = (
        args.budget if args.budget is not None else args.controllers * (1 + args.epochs)
    )

    if (
        min(
            budget,
            args.controllers,
            args.runs,
            args.layouts,
            args.held_out_layouts,
            args.frames,
            args.workers,
        )
        < 1
        or args.seed < 0
    ):
        raise ValueError("counts must be positive and seeds >= 0")

    if "DE" in args.method and args.controllers < 4:
        raise ValueError("DE requires 4 controllers at minimum")

    if any(method != "RS" for method in args.method) and budget < args.controllers:
        raise ValueError("The budget must cover all initial controllers")

    for values in (args.method, args.observation, args.gap, args.experiments):
        if len(set(values)) != len(values):
            raise ValueError(
                "each method, observation, gap and experiment set must be selected only once"
            )

    protocol = dict(
        budget=budget,
        layouts=args.layouts,
        held_out_layouts=args.held_out_layouts,
        frames=args.frames,
        seed=args.seed,
    )
    conditions = list(product(args.observation, args.gap, args.method))
    focal_observation, focal_gap = parameters["focal_setting"]

    selected_variants = {}
    for experiment in args.experiments:
        if focal_observation not in args.observation or focal_gap not in args.gap:
            raise ValueError(
                f"The paired expeiments require --observation {focal_observation} "
                f"and --gap {focal_gap} in the grid"
            )

        matching = {
            label: spec
            for label, spec in parameters["experiments"][experiment].items()
            if spec[0] in args.method
        }

        if not matching:
            raise ValueError(f"No {experiment} variants apply to selected method")

        for label, (baseline, search, overrides) in matching.items():
            conditions.append((*parameters["focal_setting"], label))
            selected_variants[label] = dict(
                baseline=baseline, update=search.__name__, overrides=overrides
            )

    tasks = []
    for observation, gap, label in conditions:
        baseline, search, i = parameters["algorithms"][label]
        options = {}

        if baseline != "RS":
            options["controllers"] = args.controllers
        elif search is random_hill_climb:
            options["initial_samples"] = args.controllers

        tasks.extend(
            (label, observation, gap, run, options) for run in range(1, args.runs + 1)
        )

    output = args.output
    if output.exists():
        raise ValueError(f"{output} already exists. Choose a new --output directory.")
    output.mkdir(parents=True)

    packages = ("flappy-bird-gymnasium", "gymnasium", "numpy", "pygame")
    defaults = {
        method: {
            name: param.default
            for name, param in signature(
                parameters["methods"][method]
            ).parameters.items()
            if param.default is not Parameter.empty
        }
        for method in args.method
    }

    settings = dict(
        **protocol,
        runs=args.runs,
        methods=args.method,
        observations=args.observation,
        gaps=args.gap,
        controllers=args.controllers,
        last_epoch=(budget - 1) // args.controllers,
        experiments=args.experiments,
        focal_setting=parameters["focal_setting"],
        variants=selected_variants,
        optimizer_defaults=defaults,
        de_selection="replace target on equal or higher reward",
        versions={name: version(name) for name in packages},
    )

    (output / "settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    partial_path = output / "trials.partial"

    print(
        f"{len(tasks)} trials x {budget} cnadidates x {args.layouts} training layouts; {args.held_out_layouts} held-out layouts per final controller",
        flush=True,
    )

    with (
        partial_path.open("x", encoding="utf-8") as stream,
        ProcessPoolExecutor(args.workers) as pool,
    ):
        worker = partial(run_trial, **protocol)
        results = pool.map(worker, methods, observations, gaps, run_numbers, options)
        methods, observations, gaps, run_numbers, options = zip(*tasks)
        for i, trial in enumerate(results, start=1):
            record = dict(epoch_size=args.controllers, **trial)
            print(json.dumps(record, allow_nan=False), file=stream, flush=True)

            if i % args.runs == 0:
                print(
                    f"{i}/{len(tasks)} {trial['algorithm']} {trial['observation']} gap {trial['gap']}",
                    flush=True,
                )

    partial_path.rename(output / "trials.jsonl")
    export_data(output)
    print(f"Saved {output / 'ioh.zip'} and CSV tables")


def load_trials(directory):
    with (directory / "trials.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_ioh_group(archive, problem_id, name, algorithm, trials):
    """
    Write an IOHAnalyzer compatible ddataset
    JSON description and a DAT table of training scores

    Each function: observation/gap pair
    Dimension: No. of weights
    Run: Own block of candidate evaluations in the DAT table

    IOHAnalyzer reads the files to draw fixed-budget and fixed-target plots
    """

    dimension = trials[0]["dimension"]

    prefix = f"IOHprofiler_f{problem_id}_{name}"
    data_path = f"data_f{problem_id}/{prefix}_DIM{dimension}.dat"
    trace = io.StringIO()
    run_metadata = []
    attributes = ["evaluations", "raw_y"]
    if "candidates" in trials[0]:
        attributes.extend(["epoch", *(f"x{i}" for i in range(dimension))])

    for trial in trials:
        scores = trial["scores"]
        trace.write(" ".join(attributes) + "\n")
        for evaluation, score in enumerate(scores, 1):
            row = [evaluation, score]
            if "candidates" in trial:
                index = evaluation - 1
                row.extend([index // trial["epoch_size"], *trial["candidates"][index]])

            trace.write(" ".join(map(str, row)) + "\n")

        best_reward = max(scores)

        first_best_evaluation = scores.index(best_reward) + 1
        run_metadata.append(
            dict(
                instance=trial["run"],
                evals=len(scores),
                best=dict(
                    evals=first_best_evaluation, y=best_reward, x=trial["weights"]
                ),
            )
        )

    metadata = dict(
        version="0.3.5",
        suite="FlappyBird",
        function_id=problem_id,
        function_name=name,
        maximization=True,
        algorithm=dict(name=algorithm, info=""),
        attributes=attributes,
        scenarios=[dict(dimension=dimension, path=data_path, runs=run_metadata)],
    )

    archive.writestr(f"{algorithm}/{prefix}.json", json.dumps(metadata))
    archive.writestr(f"{algorithm}/{data_path}", trace.getvalue())


# Boilerplate helpers and functions that support with exporting, data reading etc.


def export_data(directory):
    """
    Export training evaluations to IOHAnalyzer and CSV
    Unseen games exported to held-out.csv
    """

    keys = ("problem", "dimension", "algorithm", "run")
    trial_offsets = defaultdict(list)
    pending = directory / "ioh.zip.partial"

    with (
        (directory / "trials.jsonl").open("rb") as source,
        (directory / "evaluations.csv").open(
            "w", newline="", encoding="utf-8"
        ) as search,
        (directory / "held-out.csv").open("w", newline="", encoding="utf-8") as test,
        ZipFile(pending, "w", ZIP_DEFLATED) as archive,
    ):
        evaluations, episodes = csv.writer(search), csv.writer(test)
        evaluations.writerow((*keys, "evaluation", "epoch", "reward", "best_reward"))
        episodes.writerow((*keys, "seed", "reward", "pipes", "steps", "capped"))
        offset = 0

        for line in source:
            trial = json.loads(line)

            observation, gap, algorithm = (
                trial["observation"],
                trial["gap"],
                trial["algorithm"],
            )
            trial_offsets[observation, gap, algorithm].append(offset)
            offset += len(line)

            key = (
                f"{observation}-gap{gap}",
                trial["dimension"],
                algorithm,
                trial["run"],
            )
            best = -float("inf")

            for i, reward in enumerate(trial["scores"], 1):
                best = max(best, reward)
                epoch = (
                    (i - 1) // trial["epoch_size"] if "epoch_size" in trial else None
                )
                evaluations.writerow((*key, i, epoch, reward, best))

            for ep in trial["held_out"]:
                episodes.writerow(
                    (
                        *key,
                        ep["seed"],
                        ep["reward"],
                        ep["pipes"],
                        ep["steps"],
                        ep["capped"],
                    )
                )

        for (observation, gap, algorithm), offsets in trial_offsets.items():
            trials = []
            for offset in offsets:
                source.seek(offset)
                trials.append(json.loads(source.readline()))

            ray_count = (
                0
                if observation == "features"
                else int(observation.removeprefix("lidar"))
            )

            function_id = 1000 * ray_count + gap

            write_ioh_group(
                archive, function_id, f"{observation}-gap{gap}", algorithm, trials
            )
    pending.replace(directory / "ioh.zip")


def replay_episodes(trial, *, training, epochs, episode):
    """
    A Pygame replay that selects the weights and seeds for each replay

    Yields (weights, layout seed, window title) in the requested order
    """

    label = trial["algorithm"]

    if training:
        for (
            index,
            weights,
        ) in enumerate(trial["candidates"]):
            epoch = index // trial["epoch_size"]
            for layout, seed in enumerate(trial["train_seeds"], 1):
                yield weights, seed, f"{label} | epoch {epoch} | candidate {index + 1} | layout {layout}"

        return

    seed = trial["held_out"][episode - 1]["seed"]
    title = f"run {trial['run']} | held-out {episode}"

    if not epochs:
        yield trial["weights"], seed, f"{label} | final | {title}"
        return

    for epoch in epochs:
        scores = trial["scores"][: (epoch + 1) * trial["epoch_size"]]

        best = scores.index(max(scores))
        yield trial["candidates"][best], seed, f"{label} | epoch {epoch} | {title}"


def watch_study(args):
    """
    Select saved runs, and play the original layouts
    """

    settings = json.loads(
        (args.directory / "settings.json").read_text(encoding="utf-8")
    )
    gap = settings["gaps"][0] if args.gap is None else args.gap
    selected = {
        t["algorithm"]: t
        for t in load_trials(args.directory)
        if (t["observation"], t["gap"], t["run"]) == (args.observation, gap, args.run)
    }

    for label in args.algorithms:
        if label not in selected:
            raise ValueError(f"{label} Unavailable for this selection")
        trial = selected[label]

        if not 1 <= args.episode <= len(trial["held_out"]):
            raise ValueError(
                f"held-out episode must be between 1 and {len(trial['held_out'])}"
            )

        if (args.training or args.epochs) and "candidates" not in trial:
            raise ValueError("Invalid trial archive")

        if args.epochs and any(
            e < 0 or e * trial["epoch_size"] >= len(trial["scores"])
            for e in args.epochs
        ):
            raise ValueError(
                f"Selecte epochs between 0 and {(len(trial['scores']) - 1) // trial['epoch_size']}"
            )

    frames = settings["frames"] if "frames" in settings else settings["max_steps"]
    game = for_study(args.observation, gap, frames, watch=True)

    try:
        for label in args.algorithms:
            episodes = replay_episodes(
                selected[label],
                training=args.training,
                epochs=args.epochs,
                episode=args.episode,
            )

            for weights, seed, title in episodes:
                result = game.play_episode(
                    weights, seed=seed, caption=title, return_details=True
                )
                if not args.training:
                    print(
                        f"{title}: reward {result['reward']}, pipes {result['pipes']}",
                        flush=True,
                    )

    except KeyboardInterrupt:
        pass
    finally:
        game.env.close()


def trial_summary(trials):
    """
    Reduce saved runs to the numbers used by every report plot and summary.csv
    """

    groups = defaultdict(list)
    for trial in trials:
        episodes = trial["held_out"]
        key = (trial["observation"], trial["gap"], trial["algorithm"])
        groups[key].append(
            dict(
                run=trial["run"],
                method=trial["method"],
                dimension=trial["dimension"],
                best_so_far=np.maximum.accumulate(trial["scores"]),
                reward=float(np.mean([ep["reward"] for ep in episodes])),
                pipes=float(np.mean([ep["pipes"] for ep in episodes])),
                passed=sum(ep["pipes"] >= 1 for ep in episodes),
                games=len(episodes),
            )
        )

    summary = {}
    for key, runs in groups.items():
        stats = {
            metric: np.quantile(
                [run[metric] for run in runs], [0.25, 0.5, 0.75], axis=0
            )
            for metric in ("best_so_far", "reward", "pipes")
        }
        stats.update(
            method=runs[0]["method"],
            dimension=runs[0]["dimension"],
            runs=len(runs),
            reward_by_run={run["run"]: run["reward"] for run in runs},
            fraction_passed=sum(run["passed"] for run in runs)
            / sum(run["games"] for run in runs),
        )
        summary[key] = stats

    return summary


# OPTIONAL STUFF


def save_figure(figure, folder, name):
    """
    Save figures with exepriment visualizations, to build intuition
    """

    import matplotlib.pyplot as plt

    for extension in ("pdf", "png"):
        figure.savefig(folder / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def make_figures(summary, folder):
    """
    Draws training progres, held-out reward and pipes across same baseline task
    """
    import matplotlib.pyplot as plt 

    observations = sorted(
        {obs for obs, _, _ in summary},
        key=lambda obs: 181 if obs == "features" else int(obs.removeprefix("lidar")),
    )
    gaps = sorted({gap for _, gap, _ in summary})
    labels = {label for _, _, label in summary}
    methods = [method for method in parameters["methods"] if method in labels]
    colours = parameters["colours"]

    fig, axes = plt.subplots(
        len(observations),
        len(gaps),
        squeeze=False,
        sharex=True,
        sharey=True,
        figsize=(3.3 * len(gaps) + 0.8, 2.1 * len(observations) + 0.7),
        layout="constrained",
    )

    for row, obs in enumerate(observations):
        for col, gap in enumerate(gaps):
            ax = axes[row, col]
            ax.set_title(f"{obs} . gap {gap}")
            for method in methods:
                low, median, high = summary[obs, gap, method]["best_so_far"]
                evaluations = np.arange(1, len(median) + 1)
                ax.step(
                    evaluations,
                    median,
                    where="post",
                    color=colours[method],
                    label=method,
                )
                ax.fill_between(
                    evaluations,
                    low,
                    high,
                    step="post",
                    color=colours[method],
                    alpha=0.15,
                )

    axes[0, 0].legend()
    fig.supxlabel("Candidate Evaluations")

    fig.supylabel("Best Mean Trainnig Reward")
    save_figure(fig, folder, "convergence")

    cols = min(4, len(observations))
    rows = (len(observations) + cols - 1) // cols
    for metric, ylabel, filename in (
        ("reward", "Held-Out Reward", "held-out"),
        ("pipes", "Held-Out pipes cleared", "pipes"),
    ):
        fig, axes = plt.subplots(
            rows,
            cols,
            squeeze=False,
            sharex=True,
            sharey=True,
            figsize=(3.3 * cols + 0.8, 2.1 * rows + 0.7),
            layout="constrained",
        )

        for ax in axes.flat[len(observations) :]:
            ax.set_visible(False)
        for ax, obs in zip(axes.flat, observations):
            ax.set_title(obs)
            ax.set_xticks(gaps)

            for method in methods:
                low, median, high = np.array(
                    [summary[obs, gap, method][metric] for gap in gaps]
                ).T
                ax.errorbar(
                    gaps,
                    median,
                    yerr=[median - low, high - median],
                    marker=parameters["markers"][method],
                    color=colours[method],
                    capsize=3,
                    label=method,
                )

        axes[0, 0].legend()
        fig.supxlabel("Pipe gap (px)")
        fig.supylabel(ylabel)
        save_figure(fig, folder, filename)


def sensitivity_figure(summary, folder):
    """
    Each dot to represent the difference in mean held-out reward between matched runs
    """

    import matplotlib.pyplot as plt

    variants = [key for key, stats in summary.items() if key[2] != stats["method"]]
    if not variants:
        return
    fig, ax = plt.subplots(figsize=(6, 1.5 + 0.6 * len(variants)), layout="constrained")
    ax.axvline(0, color="0.65", linewidth=1)
    for i, (obs, gap, label) in enumerate(variants):
        variant = summary[obs, gap, label]
        method = variant["method"]
        baseline = summary[obs, gap, method]["reward_by_run"]
        delta = [
            reward - baseline[run] for run, reward in variant["reward_by_run"].items()
        ]
        low, median, high = np.quantile(delta, [0.25, 0.5, 0.75])
        colour = parameters["colours"][method]
        ax.scatter(delta, np.full(len(delta), i), s=16, color=colour, alpha=0.45)
        ax.hlines(i, low, high, color=colour, linewidth=3)
        ax.plot(median, i, "D", color=colour)

    ax.set_yticks(range(len(variants)), [label for _, _, label in variants])
    ax.set_xlabel("Paired held-out reward difference (variant - default)")
    save_figure(fig, folder, "sensitivity")


def make_plots(directory):
    """
    Create the report figures and table from a completed study
    """

    import matplotlib

    matplotlib.use("Agg")
    summary = trial_summary(load_trials(directory))
    folder = directory / "figures"
    folder.mkdir(exist_ok=True)
    make_figures(summary, folder)
    sensitivity_figure(summary, folder)
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "observation",
                "gap",
                "algorithm",
                "dimension",
                "runs",
                "reward_q1",
                "reward_median",
                "reward_q3",
                "pipes_median",
                "fraction_clearing_one_pipe",
            )
        )

        for key, stats in summary.items():
            writer.writerow(
                (
                    *key,
                    stats["dimension"],
                    stats["runs"],
                    *stats["reward"],
                    stats["pipes"][1],
                    stats["fraction_passed"],
                )
            )
    print(
        f"Saved PNG and PDF figgures in {folder} and a table in {directory/'summary.csv'}"
    )


def starter_example():
    problem = GymProblem(use_lidar=True, n_lidar=10, pipe_gap=100)
    try:
        x = problem.sample()
        f = problem(x)
        print("Fitness:", f)
        f = problem.show(x)
        print("Fitness:", f)
    finally:
        problem.env.close()


# The entire argparse CLI setup
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    focal_observation, focal_gap = parameters["focal_setting"]
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run Experiment grid and build IOHA zip")
    run.add_argument(
        "--method",
        nargs="+",
        choices=parameters["methods"],
        default=list(parameters["methods"]),
    )
    run.add_argument(
        "--experiments", nargs="+", choices=parameters["experiments"], default=[]
    )
    run.add_argument(
        "--observation",
        nargs="+",
        choices=parameters["observations"],
        default=parameters["observations"],
    )
    run.add_argument(
        "--gap", type=int, nargs="+", choices=parameters["gaps"], default=parameters["gaps"]
    )
    run.add_argument("--controllers", type=int, default=parameters["controllers"])
    duration = run.add_mutually_exclusive_group()
    duration.add_argument("--epochs", type=int, default=parameters["epochs"])
    duration.add_argument("--budget", type=int)
    run.add_argument("--runs", type=int, default=parameters["runs"])
    run.add_argument("--layouts", type=int, default=parameters["layouts"])
    run.add_argument(
        "--held-out-layouts", type=int, default=parameters["held_out_layouts"]
    )
    run.add_argument("--frames", type=int, default=parameters["frames"])
    run.add_argument("--seed", type=int, default=parameters["seed"])
    run.add_argument("--workers", type=int, default=parameters["workers"])
    run.add_argument("--output", type=Path, default=Path("runs/eval"))

    watch = commands.add_parser("watch")
    watch.add_argument("directory", type=Path)
    watch.add_argument("algorithms", nargs="*", default=list(parameters["methods"]))
    watch.add_argument("--observation", default="features")
    watch.add_argument("--gap", type=int)
    watch.add_argument("--run", type=int, default=1)
    watch.add_argument("--episode", type=int, default=1)
    mode = watch.add_mutually_exclusive_group()
    mode.add_argument("--epochs", type=int, nargs="+")
    mode.add_argument("--training", action="store_true")

    solve = commands.add_parser(
        "solve", help="run a single live search optionally in Pygame"
    )
    solve.add_argument("method", choices=parameters["algorithms"])
    solve.add_argument("--watch", action="store_true")
    solve.add_argument("--observation", default="features")
    solve.add_argument("--gap", type=int, default=parameters["gaps"][0])
    solve.add_argument("--budget", type=int, default=3 * parameters["controllers"])
    solve.add_argument("--seed", type=int, default=parameters["seed"])

    commands.add_parser("example")
    for name, purpose in (
        ("export", "rebuild CSV Tables and ZIP"),
        ("plots", "generate PDF/PNG figures and summary.csv"),
    ):
        command = commands.add_parser(name, help=purpose)
        command.add_argument("directory", type=Path)

    args = parser.parse_args()
    try:
        if args.command == "run":
            run_study(args)
        elif args.command == "watch":
            watch_study(args)
        elif args.command == "solve":
            trial = run_trial(
                args.method,
                args.observation,
                args.gap,
                budget=args.budget,
                seed=args.seed,
                watch=args.watch,
                layouts=1,
                held_out_layouts=1,
            )
            episode = trial["held_out"][0]

            print(f"Best training reward {max(trial['scores'])}")
            print(
                f"Held-out reward {episode['reward']}, pipes {episode['pipes']}, steps {episode['steps']}"
            )
        elif args.command == "example":
            starter_example()
        elif args.command == "export":
            export_data(args.directory)
        elif args.command == "plots":
            make_plots(args.directory)
    except ValueError as error:
        parser.error(str(error))

    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
