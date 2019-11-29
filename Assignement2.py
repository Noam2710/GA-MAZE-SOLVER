from termcolor import colored
import copy
import random
import time

SIZE_OF_CHROMOSOME = 20
GENES_OPTIONS = ['RIGHT', 'LEFT', 'UP', 'DOWN']
WALL = '@'


class Population:
    def __init__(self,size):
        self.list_of_individuals = []
        self.best_individual_score = None
        self.best_individual_index = None
        self.second_individual_score = None
        self.second_individual_index = None
        self.size = size

        for index in range(size):
            new_individual = Individual()
            self.list_of_individuals.append(new_individual)

    def update_fitness_according_to_maze(self,maze):
        for individual in self.list_of_individuals:
            individual.update_fitness(maze)

        self.find_the_best_individual()
        self.find_the_second_best_individual()

    def find_the_best_individual(self):
        best_fitness = self.list_of_individuals[0].fitness
        best_fitness_index = 0
        for idx, individual in enumerate(self.list_of_individuals):
            if individual.fitness < best_fitness:
                best_fitness = individual.fitness
                best_fitness_index = idx

        self.best_individual_score = best_fitness
        self.best_individual_index = best_fitness_index

    def find_the_second_best_individual(self):
        copy_of_individuals_list = copy.deepcopy(self.list_of_individuals)

        best_second_fitness = copy_of_individuals_list[0].fitness
        best_second_fitness_index = 0
        for idx, individual in enumerate(copy_of_individuals_list):
            if individual.fitness <= best_second_fitness and idx != self.best_individual_index:
                best_second_fitness = individual.fitness
                best_second_fitness_index = idx

        self.second_individual_score = best_second_fitness
        self.second_individual_index = best_second_fitness_index

    def copy_two_best_to_two_worst_individuals(self):
        worst_fitness = self.list_of_individuals[0]
        for idx, individual in enumerate(self.list_of_individuals):
            if individual.fitness > worst_fitness.fitness:
                worst_fitness = individual

        best_individual = self.list_of_individuals[self.best_individual_index]
        worst_fitness.genes = copy.deepcopy(best_individual.genes)

        worst_fitness = self.list_of_individuals[0]
        for idx, individual in enumerate(self.list_of_individuals):
            if individual.fitness > worst_fitness.fitness:
                worst_fitness = individual

        second_best_individual = self.list_of_individuals[self.second_individual_index]
        worst_fitness.genes = copy.deepcopy(second_best_individual.genes)

    def print_all_indivuauls_not_maximum(self):
        temp = []
        for individual in self.list_of_individuals:
            if individual.fitness < 20:
                temp.append(individual.fitness)
        print(temp)

class Individual:
    def __init__(self,fill_with_chromosome=True):
        self.genes = []
        self.fitness = 0

        if fill_with_chromosome:
            for index in range(SIZE_OF_CHROMOSOME):
                self.genes.append(random.choice(GENES_OPTIONS))

    def update_fitness(self, maze):
        self.fitness = maze.get_length_to_reach_the_goal(self.genes)

    def add_mutation_to_chromosome(self):
        index = random.randint(0, SIZE_OF_CHROMOSOME-1)
        self.genes[index] = random.choice(GENES_OPTIONS)


class GA:
    def __init__(self, maze, population, prob):
        self.maze = maze
        self.population = population
        self.generation_index = 0
        self.probability_to_mutation = prob

    def generate_generations(self):
        self.population.update_fitness_according_to_maze(self.maze)
        # self.population.print_all_indivuauls_not_maximum()
        if self.population.best_individual_score == 20:
            return

        print("Generation {} | Best  - score {} index {} | Second - score {} index {}".
              format(self.generation_index,
                     self.population.best_individual_score,
                     self.population.best_individual_index,
                     self.population.second_individual_score,
                     self.population.second_individual_index))

        while self.population.best_individual_score > self.maze.best_way_length:

            self.selection()
            self.crossover()

            self.population.update_fitness_according_to_maze(self.maze)
            self.population.copy_two_best_to_two_worst_individuals()
            self.population.update_fitness_according_to_maze(self.maze)
            self.generation_index += 1

            # self.population.print_all_indivuauls_not_maximum()

            print("Generation {} | Best  - score {} index {} | Second - score {} index {}".
                  format(self.generation_index,
                         self.population.best_individual_score,
                         self.population.best_individual_index,
                         self.population.second_individual_score,
                         self.population.second_individual_index))
        print("We solved the maze after {} generations, the solution is {}".format(self.generation_index,
                                                                                   self.population.list_of_individuals[self.population.best_individual_index]))

    def selection(self):
        "The best and Second best can be found by population variables."
        pass

    def crossover(self):
        "We use here Single-point crossover"
        point = random.randint(0, SIZE_OF_CHROMOSOME-1)

        child1 = Individual(fill_with_chromosome=False)
        child2 = Individual(fill_with_chromosome=False)

        for index in range(0, SIZE_OF_CHROMOSOME):
            if index <= point:
                child1.genes.append(self.population.list_of_individuals[self.population.best_individual_index].genes[index])
                child2.genes.append(self.population.list_of_individuals[self.population.second_individual_index].genes[index])
            else:
                child1.genes.append(self.population.list_of_individuals[self.population.second_individual_index].genes[index])
                child2.genes.append(self.population.list_of_individuals[self.population.best_individual_index].genes[index])

        if random.uniform(0, 1) > self.probability_to_mutation:
            self.mutation(child1, child2)

        self.population.list_of_individuals[self.population.best_individual_index] = child1
        self.population.list_of_individuals[self.population.second_individual_index] = child2

    def mutation(self, child1, child2):

        child1.update_fitness(self.maze)
        child2.update_fitness(self.maze)
        child1.add_mutation_to_chromosome()
        child2.add_mutation_to_chromosome()
        child1.update_fitness(self.maze)
        child2.update_fitness(self.maze)
        # print("Child1 Score {} Child2 Score {}".format(child1.fitness,child2.fitness))


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

test_maze = Maze()
population = Population(1000)
GA_instance = GA(test_maze, population, 0)
GA_instance.generate_generations()

