import random

import copy
import numpy as np
import pandas as pd
import random as rand


def read_distance_matrix_from_txt(path, header=None, skiprows=4, sep=','):
    '''
    :param path: 'data(TSP)/data-1.txt'
    :return: distance_matrix (e.g. distance_matrix[1][2] = distance between city '1' and '2')
    '''
    distance_matrix = pd.read_csv(path, header=header, skiprows=skiprows, sep=sep)
    return distance_matrix


def init_population(pop_size=100, gen_size=6000, num_cities=200):
    '''
    :param pop_size: chromosome (string) size
    :param gen_size: generation size
    :param num_cities: number of total cities
    :return: population (e.g. [ [2,4,1,0,6,9,...], ... ], where integers stand for 'city' indices.)
    '''
    population = [list(rand.sample(range(num_cities), k=pop_size)) for _ in range(gen_size)]
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
        txt += f'{chromosome},{fitness:.6f}\n'
    with open(filename, 'w') as f:
        f.write(txt)


def order_one_crossover(population, prob, num_offsprings=1):
    '''
    :param num_offsprings: [1 or 2]
    :return: crossover result
    '''
    pop_size, gen_size = len(population[0]), len(population)
    gen_size_half = gen_size // 2
    random.shuffle(population)   # for bias issue
    for i in range(gen_size_half):
        if random.random() <= prob:
            dad, mom = population[i], population[i + gen_size_half]
            start_idx, end_idx = sorted(np.random.choice(list(range(pop_size)), size=2, replace=False))
            # offspring 1
            partial_dad_1 = dad[start_idx:end_idx+1]
            partial_mom_1 = [gene for gene in mom if gene not in partial_dad_1]
            offspring_1 = [partial_mom_1.pop(0) for _ in range(0, min(len(partial_mom_1), start_idx))] + \
                           partial_dad_1 + partial_mom_1
            population[i] = offspring_1[:pop_size]

            # offspring 2 (If you want to birth another offspring!)
            if num_offsprings >= 2:
                partial_mom_2 = mom[start_idx:end_idx+1]
                partial_dad_2 = [gene for gene in dad if gene not in partial_mom_2]
                offspring_2 = [partial_dad_2.pop(0) for _ in range(0, min(len(partial_dad_2), start_idx))] + \
                               partial_mom_2 + partial_dad_2
                population[i + gen_size_half] = offspring_2[:pop_size]


def mutation(population, prob):
    pop_size = len(population[0])
    for i in range(pop_size):
        if random.random() <= prob:
            # swap
            cursor_1, cursor_2 = np.random.choice(list(range(pop_size)), size=2, replace=False)
            population[i][cursor_1], population[i][cursor_2] = population[i][cursor_2], population[i][cursor_1]
