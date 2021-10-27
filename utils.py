import random

import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt


def read_distance_matrix_from_txt(path, header=None, skiprows=4, sep=','):
    '''
    :param path: 'data(TSP)/data-1.txt'
    :return: distance_matrix (e.g. distance_matrix[1][2] = distance between city '1' and '2')
    '''
    distance_matrix = pd.read_csv(path, header=header, skiprows=skiprows, sep=sep)
    return distance_matrix


def init_population(pop_size=100, num_cities=200):
    '''
    :param pop_size: # of chromosomes
    :param num_cities: number of total cities
    :return: population (e.g. [ [2,4,1,0,6,9,...], ... ], where integers stand for 'city' indices.)
    '''
    population = [list(rand.sample(range(num_cities), k=num_cities)) for _ in range(pop_size)]
    return population


def get_fitness(distance_matrix, chromosome):
    '''
    :param distance_matrix: from read_distance_matrix_from_txt()
    :param chromosome: individual string from init_population(), Here, "chromosome == route for traveling salesman"
    :return: fitness_score of the current chromosome (route)
    '''
    total_distance = [distance_matrix[chromosome[i-1]][chromosome[i]] for i in range(1, len(chromosome))]
    fitness = sum(total_distance)
    return fitness


def is_promising_for_salesman(cur, best):
    return True if cur < best else False     # shortest route is better.


def store(population, fitnesses, filename='test.txt'):
    txt = ''
    for chromosome, fitness in zip(population, fitnesses):
        string = '-'.join(map(str, chromosome))
        txt += f'{string},{fitness:.6f}\n'
    with open(filename, 'w') as f:
        f.write(txt)


def elitism(population, fitnesses, elitism_ratio):
    num_elites = int(len(population) * elitism_ratio)
    top_ids = sorted(np.arange(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[-num_elites:]
    bottom_ids = sorted(np.arange(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:num_elites]
    # reverse=True? -> The objective is to minimize fitness score in TSP.

    for i, (top_id, bottom_id) in enumerate(zip(top_ids, bottom_ids)):
        population[bottom_id] = population[top_id]
        population[i], population[top_id] = population[top_id], population[i]

    elite_chromosomes = population[:num_elites]
    population = population[num_elites:]
    fitnesses = fitnesses[num_elites:]
    return elite_chromosomes, population, fitnesses


def tournament_selection(population, distance_matrix):
    pop_size = len(population)
    for i in range(pop_size):
        pos = random.randint(0, pop_size-1)
        f_i, f_pos = get_fitness(distance_matrix, population[i]), get_fitness(distance_matrix, population[pos])
        if is_promising_for_salesman(f_pos, f_i):
            population[i] = population[pos]


def order_one_crossover(population, prob, num_offsprings=1):
    '''
    :param num_offsprings: [1 or 2]
    :return: crossover result
    '''
    string_length, pop_size = len(population[0]), len(population)
    pop_size_half = pop_size // 2
    random.shuffle(population)   # for bias issue
    for i in range(pop_size_half):
        if random.random() <= prob:
            dad, mom = population[i], population[i + pop_size_half]
            start_idx, end_idx = sorted(np.random.choice(list(range(string_length)), size=2, replace=False))
            # offspring 1
            partial_dad_1 = dad[start_idx:end_idx+1]
            partial_mom_1 = [gene for gene in mom if gene not in partial_dad_1]
            offspring_1 = [partial_mom_1.pop(0) for _ in range(0, min(len(partial_mom_1), start_idx))] + \
                           partial_dad_1 + partial_mom_1
            population[i] = offspring_1
            # offspring 2 (If you want to birth another offspring!)
            if num_offsprings >= 2:
                partial_mom_2 = mom[start_idx:end_idx+1]
                partial_dad_2 = [gene for gene in dad if gene not in partial_mom_2]
                offspring_2 = [partial_dad_2.pop(0) for _ in range(0, min(len(partial_dad_2), start_idx))] + \
                               partial_mom_2 + partial_dad_2
                population[i + pop_size_half] = offspring_2


def mutation(population, prob):
    '''
    :return: mutation result
    '''
    string_length, pop_size = len(population[0]), len(population)
    for i in range(pop_size):
        if random.random() <= prob:
            # swap
            cursor_1, cursor_2 = np.random.choice(list(range(string_length)), size=2, replace=False)
            population[i][cursor_1], population[i][cursor_2] = population[i][cursor_2], population[i][cursor_1]


def plot(d1, d2, name):
    plt.title(f"Traveling Salesman Problem Fitness Trace ({name})")
    plt.plot(range(len(d1)), np.log(np.array(d1)), label="Log(Fitness), average")
    plt.plot(range(len(d2)), np.log(np.array(d2)), label="Log(Fitness), best")
    plt.legend()
    plt.savefig(name)
    # plt.show()