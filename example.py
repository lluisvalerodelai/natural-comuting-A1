from problem import GymProblem

if __name__ == "__main__":
    problem = GymProblem(use_lidar=True, n_lidar=10, pipe_gap=100) # initialise the problem with specific environment setup
    x = problem.sample() # sample a random candidate solution

    # You can evaluate the fitness of a candidate solution like so:
    f = problem(x)
    print('Fitness: ', f)

    # You can also evaluate and display a solution this way:
    f = problem.show(x) # optionally, if you set gif_path to a file like solution.gif for example, the run will be saved.
    print('Fitness: ', f)
