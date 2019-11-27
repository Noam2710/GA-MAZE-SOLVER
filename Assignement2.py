import random
from termcolor import colored

SIZE_OF_CHROMOSOME = 20
GENES_OPTIONES = ['RIGHT', 'LEFT', 'UP', 'DOWN']


class Population():
    def __init__(self,size):
        self.list_of_individuals = []
        self.fittest_individual = None
        self.size = size

        for index in range(size):
            new_individual = Individual()
            self.list_of_individuals.append(new_individual)


class Individual():
    def __init__(self):
        self.genes = []
        for index in range(SIZE_OF_CHROMOSOME):
            self.genes.append(random.choice(GENES_OPTIONES))
        self.fitness = self.update_fitness()

    def update_fitness(self):
        pass


class Maze():
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
                line_to_print += self.get_element_by_column(column) + ' '
            print(line_to_print)

    def get_element_by_column(self, column):
        if column == '@':
            return colored(u"\u2588",'red')
        elif column == ' ':
            return colored(u"\u2588",'white')
        else:
            return colored(u"\u2588",'green')


test_maze = Maze()
test_maze.print_maze()