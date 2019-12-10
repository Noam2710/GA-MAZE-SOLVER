from termcolor import colored
import random
import numpy
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt


SIZE_OF_CHROMOSOME = 20
POP_SIZE = 100
GENES_OPTIONS = ['RIGHT', 'LEFT', 'UP', 'DOWN']
WALL = '@'
GENOME_MIN = 0
GENOME_MAX = 3
GENERATIONS_NUMBER = 150
PRINT_FINAL_SOLUTION = False
APPROACH = 3


class Maze:
    def __init__(self):
        self.maze = [
        ['@', '@', '@', '@', '@', '@', '@'],
        ['S', ' ', '@', ' ', ' ', ' ', '@'],
        ['@', ' ', '@', ' ', '@', ' ', 'E'],
        ['@', ' ', ' ', ' ', ' ', ' ', '@'],
        ['@', '@', '@', '@', '@', '@', '@']
        ]

        self.start_loc = [1, 0]
        self.end_loc = [2, 6]
        self.current_loc = self.start_loc

    def print_maze(self):
        temp = self.maze[self.current_loc[0]][self.current_loc[1]]
        self.maze[self.current_loc[0]][self.current_loc[1]] = 'C'
        for line in self.maze:
            line_to_print = ''
            for column in line:
                line_to_print += Maze.get_element_by_column(column) + ' '
            print(line_to_print)
        self.maze[self.current_loc[0]][self.current_loc[1]] = temp

    @classmethod
    def get_element_by_column(cls, column):
        if column == '@':
            return colored(u"\u2588",'red')
        elif column == ' ':
            return colored(u"\u2588",'white')
        elif column == 'C':
            return colored(u"\u2588",'magenta')
        else:
            return colored(u"\u2588",'green')


def get_length_to_reach_the_goal(individual):
    maze = test_maze
    maze.current_loc = maze.start_loc
    num_of_not_valid_steps = 0
    for idx, step in enumerate(individual):
        step = GENES_OPTIONS[step]
        if step == 'RIGHT':
            new_index = [maze.current_loc[0], maze.current_loc[1] + 1]
        if step == 'LEFT':
            new_index = [maze.current_loc[0], maze.current_loc[1] - 1]
        if step == 'UP':
            new_index = [maze.current_loc[0] - 1, maze.current_loc[1]]
        if step == 'DOWN':
            new_index = [maze.current_loc[0] + 1, maze.current_loc[1]]

        valid_step = check_if_valid_step(new_index, maze.maze)
        if valid_step:
            maze.current_loc = new_index
        else:
            num_of_not_valid_steps += 4

        if maze.current_loc == maze.end_loc:
            break
    if APPROACH == 3:
        return num_of_not_valid_steps + idx + 1,
    else:
        return idx + 1,


def check_if_valid_step(new_index, maze):
    number_of_rows = len(maze)
    number_of_columns = len(maze[0])

    if new_index[0] < 0 or new_index[0] > number_of_rows-1:
        return False
    if new_index[1] < 0 or new_index[1] > number_of_columns-1:
        return False
    if maze[new_index[0]][new_index[1]] == WALL:
        return False

    return True


def print_graph(logbook):
    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("avg")
    fit_maxs = logbook.select('max')
    fit_med = logbook.select('med')

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")

    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    line2 = ax1.plot(gen, fit_avgs, "r-", label="Average Fitness")
    line3 = ax1.plot(gen, fit_maxs, "g-", label="Maximum Fitness")
    line4 = ax1.plot(gen, fit_med, "y-", label="Median Fitness")


    lns = line1 + line2 + line3 + line4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()


def update_fitnesses_to_population():
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit


def register_stats():
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    stats.register("med", numpy.median)


def print_initial_maze():
    print('--------------------')
    test_maze.print_maze()
    print('--------------------')


def print_the_best_solution():
    if PRINT_FINAL_SOLUTION:
        maze = Maze()
        best_path = best_ind.items
        print("Initial Maze")
        maze.print_maze()
        for idx, step in enumerate(best_path[0]):
            step = GENES_OPTIONS[step]
            if step == 'RIGHT':
                new_index = [maze.current_loc[0], maze.current_loc[1] + 1]
                valid_step = check_if_valid_step(new_index, maze.maze)
            if step == 'LEFT':
                new_index = [maze.current_loc[0], maze.current_loc[1] - 1]
                valid_step = check_if_valid_step(new_index, maze.maze)
            if step == 'UP':
                new_index = [maze.current_loc[0] - 1, maze.current_loc[1]]
                valid_step = check_if_valid_step(new_index, maze.maze)
            if step == 'DOWN':
                new_index = [maze.current_loc[0] + 1, maze.current_loc[1]]
                valid_step = check_if_valid_step(new_index, maze.maze)

            if valid_step:
                maze.current_loc = new_index

            print("Step No. {}".format(str(idx+1)))
            maze.print_maze()

            if maze.current_loc == maze.end_loc:
                break


def cxOnePoint(ind1, ind2):
    """Executes a one point crossover on the input :term:`sequence` individuals.
    The two individuals are modified in place. The resulting individuals will
    respectively have the length of the other.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.randint` function from the
    python base :mod:`random` module.
    """
    position = min(ind1.fitness.values[0], ind2.fitness.values[0])
    cxpoint = random.randint(1, position - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

    return ind1, ind2


test_maze = Maze()
print_initial_maze()
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attribute", random.randint, 0, 3)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=SIZE_OF_CHROMOSOME)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", get_length_to_reach_the_goal)

if APPROACH == 2:
    toolbox.register("mate", cxOnePoint)
else:
    toolbox.register("mate", tools.cxOnePoint)

toolbox.register("mutate", tools.mutUniformInt, low=GENOME_MIN, up=GENOME_MAX, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=100)

pop = toolbox.population(n=POP_SIZE)
fitnesses = toolbox.map(toolbox.evaluate, pop)
best_ind = tools.HallOfFame(1)
update_fitnesses_to_population()

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
register_stats()

pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.8, ngen=GENERATIONS_NUMBER, stats=stats, halloffame=best_ind,
                                   verbose=True)


print_the_best_solution()
print_graph(logbook)
