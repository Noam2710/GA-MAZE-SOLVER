from termcolor import colored
import copy
import random
import numpy
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt


SIZE_OF_CHROMOSOME = 20
POP_SIZE = 100
GENES_OPTIONS = ['RIGHT', 'LEFT', 'UP', 'DOWN']
WALL = '@'


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
        self.best_way_length = 9

    def print_maze(self):
        for line in self.maze:
            line_to_print = ''
            for column in line:
                line_to_print += Maze.get_element_by_column(column) + ' '
            print(line_to_print)

    @classmethod
    def get_element_by_column(cls, column):
        if column == '@':
            return colored(u"\u2588",'red')
        elif column == ' ':
            return colored(u"\u2588",'white')
        else:
            return colored(u"\u2588",'green')

    def get_length_to_reach_the_goal(self, genes):
        self.current_loc = self.start_loc
        for idx, step in enumerate(genes):
            if step == 'RIGHT':
                new_index = [self.current_loc[0],self.current_loc[1]+1]
                valid_step = self.check_if_valid_step(new_index)
            if step == 'LEFT':
                new_index = [self.current_loc[0], self.current_loc[1] - 1]
                valid_step = self.check_if_valid_step(new_index)
            if step == 'UP':
                new_index = [self.current_loc[0]-1, self.current_loc[1]]
                valid_step = self.check_if_valid_step(new_index)
            if step == 'DOWN':
                new_index = [self.current_loc[0] + 1, self.current_loc[1]]
                valid_step = self.check_if_valid_step(new_index)

            if valid_step:
                self.current_loc = new_index

            if self.current_loc == self.end_loc:
                break

        return idx+1

    def check_if_valid_step(self,new_index):
        number_of_rows = len(self.maze)
        number_of_columns = len(self.maze[0])

        if new_index[0] < 0 or new_index[0] > number_of_rows-1:
            return False
        if new_index[1] < 0 or new_index[1] > number_of_columns-1:
            return False
        if self.maze[new_index[0]][new_index[1]] == WALL:
            return False

        return True


def get_length_to_reach_the_goal(individual):
    maze = test_maze
    maze.current_loc = maze.start_loc
    for idx, step in enumerate(individual):
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

        if maze.current_loc == maze.end_loc:
            break

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

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")

    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    line2 = ax1.plot(gen, fit_avgs, "r-", label="Average Fitness")

    lns = line1 + line2
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


test_maze = Maze()
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attribute", random.randint, 0, 3)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=SIZE_OF_CHROMOSOME)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", get_length_to_reach_the_goal)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=0.08)
toolbox.register("select", tools.selTournament, tournsize=5)


pop = toolbox.population(n=POP_SIZE)
fitnesses = toolbox.map(toolbox.evaluate, pop)
Best_2_ind = tools.HallOfFame(2)
update_fitnesses_to_population()


stats = tools.Statistics(key=lambda ind: ind.fitness.values)
register_stats()

pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1000, stats=stats, halloffame=Best_2_ind, verbose=True)
print_graph(logbook)







