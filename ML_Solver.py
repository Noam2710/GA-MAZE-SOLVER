import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import random
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

list_of_decision_trees = None


def register_stats():
    global stats
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)


def create_forest(total_number_of_trees, num_of_features):
    RF = []
    #X, y = get_train_set_of_current_data_set(num_of_features)
    X, y = X_train, y_train
    for i in range(2, num_of_features):
        curr_tree = RandomForestClassifier(n_estimators=int(total_number_of_trees / (num_of_features - 2)),
                                           max_features=i)
        curr_tree.fit(X, y)
        RF.append(curr_tree)

    # Flatten the list to one list of N decition trees
    RF = [item for sublist in RF for item in sublist]
    return RF


def create_population(list_of_decision_trees, pop_size, chromosome_length):
    population = []
    for _ in range(pop_size):
        individual = random.sample(list_of_decision_trees, chromosome_length)
        population.append(individual)
    return population


def cx_one_point(ind1, ind2):
    cxpoint = random.randint(0, len(ind1))

    old_ind2 = ind2[cxpoint:]
    old_ind1 = ind1[cxpoint:]

    intersection = set(ind1[cxpoint:]).intersection(set(old_ind2))
    while len(intersection) > 0:
        for intersect in intersection:
            old_ind2.remove(intersect)
            old_ind2.append(random.randint(0,999))
        intersection = set(ind1[cxpoint:]).intersection(set(old_ind2))

    intersection = set(ind2[cxpoint:]).intersection(set(old_ind1))
    while len(intersection) > 0:
        for intersect in intersection:
            old_ind1.remove(intersect)
            old_ind1.append(random.randint(0, 999))
        intersection = set(ind2[cxpoint:]).intersection(set(old_ind1))

    ind1[cxpoint:], ind2[cxpoint:] = old_ind2, old_ind1
    return ind1, ind2


def initialize_evolution_functions(total_number_of_trees, chromosome_length, tour_size):

    global toolbox

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox.register("attribute", random.randint, 0, total_number_of_trees-1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, chromosome_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", cx_one_point)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=total_number_of_trees-1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=tour_size)
    register_stats()

    pop = toolbox.population(n=200)
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    best_ind = tools.HallOfFame(1)
    pop = update_fitnesses_to_population(pop,fitnesses)

    return pop, best_ind


def evaluate_individual(individual, test=False):

    if test:
        X, y = X_test,y_test
    else:
        X, y = X_val,y_val

    prediction_results = []

    for decision_tree_index in individual:
        prediction_results.append(list_of_decision_trees[decision_tree_index].predict(X))

    prediction_results = np.array(prediction_results)
    predictions_of_individual = np.sum(prediction_results, axis=0) > len(individual) / 2
    fitness = accuracy_score(predictions_of_individual, y)

    return (float(fitness),)


def create_population_to_deap(pop_not_for_deap):
    pop_to_deap = []
    for ind in pop_not_for_deap:
        pop_to_deap.append(creator.Individual(ind))

    return pop_to_deap


def update_fitnesses_to_population(pop, fitnesses):
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    return pop


def print_graph(logbook):
    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("avg")
    fit_maxs = logbook.select('max')
    fit_med = logbook.select('med')

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness/Accuracy", color="b")

    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness/Accuracy")
    line2 = ax1.plot(gen, fit_avgs, "r-", label="Average Fitness/Accuracy")
    line3 = ax1.plot(gen, fit_maxs, "g-", label="Maximum Fitness/Accuracy")
    line4 = ax1.plot(gen, fit_med, "y-", label="Median Fitness/Accuracy")


    lns = line1 + line2 + line3 + line4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()


def initialize_data_set(fake):
    global X_train, X_val, y_train, y_val, X_test, y_test

    if fake:
        num_of_features = 202 #Arbitrary number
        X,y = make_classification(n_samples=100, n_features=num_of_features, n_informative=num_of_features,n_redundant=0, shuffle=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    else:
        num_of_features = 0 #Should be extracted from the dataset
        #print("Here we should add the project data-sets")
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    return X_train, X_val, y_train, y_val, X_test, y_test, num_of_features


def HGARF(total_number_of_trees, num_of_features, chromosome_length, number_of_generations, tour_size, crossover_rate,mutation_rate):

    global list_of_decision_trees

    list_of_decision_trees = create_forest(total_number_of_trees=total_number_of_trees, num_of_features=num_of_features)

    pop, best_ind = initialize_evolution_functions(total_number_of_trees, chromosome_length, tour_size)

    pop, logbook = algorithms.eaSimple(pop,
                               toolbox,
                               cxpb=crossover_rate,
                               mutpb=mutation_rate,
                               ngen=number_of_generations,
                               stats=stats,
                               halloffame=best_ind,
                               verbose=True)

    return pop, logbook, best_ind




X_train, X_val, y_train, y_val, X_test, y_test, num_of_features = initialize_data_set(fake=True)

toolbox = base.Toolbox()
stats = tools.Statistics(key=lambda ind: ind.fitness.values)


pop, logbook, best_ind = HGARF(total_number_of_trees=1000,
                        num_of_features=num_of_features,
                        chromosome_length=10,
                        number_of_generations=100,
                        tour_size=20,
                        crossover_rate=0.5,
                        mutation_rate=0.8)

print("best validation score is " + str(evaluate_individual(best_ind[0])[0]))
print("best test score is " + str(evaluate_individual(best_ind[0], test=True)[0]))

print_graph(logbook)






