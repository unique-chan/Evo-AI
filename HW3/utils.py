import random
import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
import copy


def init_population(population_size=100, individual_length=50):
    '''
    :return: population (array: shape=(population_size, individual_length))
             e.g. array([[0, 1, 1, ..., 1, 1, 1],
                         ...,
                         [1, 1, 0, ..., 1, 1, 1]])
    '''
    rng = np.random.default_rng(seed=42)
    population = rng.integers(2, size=(population_size, individual_length))
    return population


def get_fitness(chromosome):
    n = len(chromosome)
    u = chromosome
    one_minus_u = 1 - chromosome
    fitness = np.maximum(np.sum(u[:n // 2]), np.sum(one_minus_u[:n // 2])) + \
              np.maximum(np.sum(u[n // 2:]), np.sum(one_minus_u[n // 2:]))
    return fitness


def get_hamming_distance(chromosome_a, chromosome_b):
    '''
    :param chromosome_a:                                            e.g. np.array([1, 1, 1, 1, 1, 0, 0, 1])
    :param chromosome_b:                                            e.g. np.array([1, 1, 1, 1, 1, 1, 0, 0])
    :return: hamming_distance between chromosome_a & chromosome_b   e.g. 2 (two elements differ in this case.)
    '''
    return np.sum(chromosome_a ^ chromosome_b)


def get_hamming_distance_matrix(population):
    '''
    :return: hamming_distance_matrix H
             where H[i][j] refers to hamming_distance between i-th chromosome and j-th chromosome in the population.
    '''
    pop_size = len(population)
    hamming_distance_matrix = np.zeros((pop_size, pop_size))
    for i in range(pop_size):
        current_chromosome = population[i]
        for j in range(pop_size):
            if i != j:
                hamming_distance_matrix[i][j] = get_hamming_distance(current_chromosome, population[j])
    return hamming_distance_matrix


def get_modified_fitness(chromosome_index, population, hamming_distance_matrix, threshold, eps=1e-7):
    '''
    :param chromosome_index <=> i
    :param hamming_distance_matrix <=> d
    :param threshold <=> D
    :return: (modified fitness) = fitness / sum( h(d[i][j]) )
                                  where h(d[i][j]) = 1 - d[i][j] / D   if d[i][j] < D
                                                   = 0                 otherwise
                                  here, d[i][j] = get_hamming_distance(i-th chromosome, j-th chromosome).
    '''
    n = hamming_distance_matrix.shape[0]
    h = np.array([hamming_distance_matrix[chromosome_index][j] for j in range(n) if chromosome_index != j])
    h = h[h < threshold] / threshold
    h = 1 - h         # h(d[i][j]) = 1 - d[i][j] / D   if d[i][j] < D
                      #            = 0                 otherwise
    zeta = np.sum(h)
    return get_fitness(population[chromosome_index]) / (zeta + eps)  # eps for avoiding zero division.


def store(population, filename='fourmax.txt'):
    txt = ''
    # for chromosome, fitness in zip(population, fitnesses):
    #     string = '-'.join(map(str, chromosome))
    #     txt += f'{string},{fitness:.6f}\n'
    # with open(filename, 'w') as f:
    #     f.write(txt)
    for chromosome in population:
        string = ''.join(map(str, chromosome))
        txt += f'{string}\n'
    with open(filename, 'w') as f:
        f.write(txt)


def is_promising_for_FourMax(cur, best):
    return True if cur > best else False  # we want to find the optimal maximum.


def elitism(population, fitnesses, elitism_ratio):
    num_elites = int(len(population) * elitism_ratio)
    top_ids = sorted(np.arange(len(fitnesses)), key=lambda i: fitnesses[i], reverse=False)[-num_elites:]
    bottom_ids = sorted(np.arange(len(fitnesses)), key=lambda i: fitnesses[i], reverse=False)[:num_elites]
    # reverse=False? -> The objective is to maximize fitness score in FourMax.

    for i, (top_id, bottom_id) in enumerate(zip(top_ids, bottom_ids)):
        population[bottom_id] = population[top_id]
        population[i], population[top_id] = population[top_id], population[i]

    elite_chromosomes = population[:num_elites]
    population = population[num_elites:]
    fitnesses = fitnesses[num_elites:]
    return elite_chromosomes, population, fitnesses


def tournament_selection(population, soft_tournament_prob=0):
    '''
    :param soft_tournament_prob:
        if soft_tournament_prob <= 0:   'strict' tournament selection
        otherwise:                      'soft' tournament selection (in general, soft_tournament_prob >= 0.5)
    '''
    pop_size = len(population)
    for i in range(pop_size):
        pos = random.randint(0, pop_size - 1)
        f_i, f_pos = get_fitness(population[i]), get_fitness(population[pos])
        if soft_tournament_prob <= 0:  # strict tournament
            if is_promising_for_FourMax(f_pos, f_i):
                population[i] = population[pos]
        else:  # soft tournament
            if is_promising_for_FourMax(f_pos, f_i):
                if random.random() <= soft_tournament_prob:
                    population[i] = population[pos]
    return population


def crossover(population, prob):
    individual_length = len(population[0])
    pop_size = len(population)
    pop_size_half = pop_size // 2
    random.shuffle(population)  # To address positional bias issue, shuffling is required before crossover.
    for i in range(pop_size_half):
        # 3-pt crossover
        if random.random() <= prob:
            positions = sorted(np.random.choice(list(range(0, individual_length)), size=3, replace=False))
            dad, mom = population[i], population[i + pop_size_half]
            offspring1 = np.concatenate((dad[0:positions[0]], mom[positions[0]:positions[1]],
                                        dad[positions[1]:positions[2]], mom[positions[2]:]))
            offspring2 = np.concatenate((mom[0:positions[0]], dad[positions[0]:positions[1]],
                                        mom[positions[1]:positions[2]], dad[positions[2]:]))
            population[i] = offspring1
            population[i + pop_size_half] = offspring2


def bitwise_mutation(population, prob):
    '''
    [Assumption] All chromosomes consist of only 0 and 1.
    :param population: numpy array (mxn matrix)
    :param prob: scalar R: [0, 1] (e.g. 0.8)
    :return:
    '''
    individual_length = len(population[0])
    pop_size = len(population)
    mask_matrix = np.random.rand(pop_size, individual_length) < prob
    for i in range(mask_matrix.shape[0]):
        for j in range(mask_matrix.shape[1]):
            if mask_matrix[i][j]:
                population[i][j] = 1 if population[i][j] == 0 else 0


def plot(d1, d2, filename='fourmax.png', message=''):
    plt.title(f"FourMax Problem fitness Trace ({filename}) \n(avg-best: {d2[-1]:.6f})" + message)
    plt.plot(range(len(d1)), np.array(d1), label="Fitness, average")
    plt.plot(range(len(d2)), np.array(d2), label="Fitness, best")
    plt.legend()
    plt.savefig(filename)
    # plt.show()
