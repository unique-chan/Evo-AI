import random
import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt


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
    '''
    :return: fitness score
             e.g. chromosome = array([1,1,1,0,0])
                  -> fitness_score = max(1+1+1+0+0, (1-1)+(1-1)+(1-1)+(1-0)+(1-0))
    '''
    u = np.sum(chromosome)
    bar_u = np.sum(1 - chromosome)
    fitness = np.maximum(u, bar_u)
    return fitness


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


# def roulette_selection(population, fitnesses, logarithmic_scaling_T=0):
#     '''
#     :param logarithmic_scaling_T:
#            if T <= 0, no logarithmic scaling is applied to the fitnesses (i.e. 'ordinary' roulette selection).
#            otherwise, logarithmic scaling is applied to the fitnesses:
#                       new_fitnesses = exp(fitnesses / T).
#     '''
#     population_size = len(population)
#     new_fitnesses = np.array(fitnesses)
#     if logarithmic_scaling_T > 0:
#         new_fitnesses = np.exp(new_fitnesses / logarithmic_scaling_T)
#     selections = []
#     for i in range(population_size):
#         threshold = random.random()
#         j = 0
#         while j < population_size and threshold > sum(new_fitnesses[:j+1]) / sum(new_fitnesses):
#             j += 1
#         selections.append(population[j])
#     return selections


# def tournament_selection(population, distance_matrix, soft_tournament_prob=0):
#     '''
#     :param soft_tournament_prob:
#         if soft_tournament_prob <= 0:   'strict' tournament selection
#         otherwise:                      'soft' tournament selection (in general, soft_tournament_prob >= 0.5)
#     '''
#     pop_size = len(population)
#     for i in range(pop_size):
#         pos = random.randint(0, pop_size-1)
#         f_i, f_pos = get_fitness(distance_matrix, population[i]), get_fitness(distance_matrix, population[pos])
#         if soft_tournament_prob <= 0:                       # strict tournament
#             if is_promising_for_salesman(f_pos, f_i):
#                 population[i] = population[pos]
#         else:                                               # soft tournament
#             if is_promising_for_salesman(f_pos, f_i):
#                 if random.random() <= soft_tournament_prob:
#                     population[i] = population[pos]
#     return population


# def boltzmann_selection(population, distance_matrix, prob, T):
#     pop_size = len(population)
#     for i in range(pop_size):
#         pos = random.randint(0, pop_size-1)
#         f_i, f_pos = get_fitness(distance_matrix, population[i]), get_fitness(distance_matrix, population[pos])
#         prob_selecting_pos = 1 / (1 + np.exp(-(f_pos - f_i) / T))
#         if prob_selecting_pos >= prob:
#             population[i] = population[pos]
#     return population


# def order_one_crossover(population, prob, num_offsprings=1):
#     '''
#     :param num_offsprings: [1 or 2]
#     :return: crossover result
#     '''
#     string_length, pop_size = len(population[0]), len(population)
#     pop_size_half = pop_size // 2
#     random.shuffle(population)   # for bias issue
#     for i in range(pop_size_half):
#         if random.random() <= prob:
#             dad, mom = population[i], population[i + pop_size_half]
#             start_idx, end_idx = sorted(np.random.choice(list(range(string_length)), size=2, replace=False))
#             # offspring 1
#             partial_dad_1 = dad[start_idx:end_idx+1]
#             partial_mom_1 = [gene for gene in mom if gene not in partial_dad_1]
#             offspring_1 = [partial_mom_1.pop(0) for _ in range(0, min(len(partial_mom_1), start_idx))] + \
#                            partial_dad_1 + partial_mom_1
#             population[i] = offspring_1
#             # offspring 2 (If you want to birth another offspring!)
#             if num_offsprings >= 2:
#                 partial_mom_2 = mom[start_idx:end_idx+1]
#                 partial_dad_2 = [gene for gene in dad if gene not in partial_mom_2]
#                 offspring_2 = [partial_dad_2.pop(0) for _ in range(0, min(len(partial_dad_2), start_idx))] + \
#                                partial_mom_2 + partial_dad_2
#                 population[i + pop_size_half] = offspring_2


# def mutation(population, prob):
#     '''
#     :return: mutation result
#     '''
#     string_length, pop_size = len(population[0]), len(population)
#     for i in range(pop_size):
#         if random.random() <= prob:
#             # swap
#             cursor_1, cursor_2 = np.random.choice(list(range(string_length)), size=2, replace=False)
#             population[i][cursor_1], population[i][cursor_2] = population[i][cursor_2], population[i][cursor_1]


def plot(d1, d2, name='fourmax.png'):
    plt.title(f"FourMax Problem itness Trace ({name}) \n(avg-best: {d2[-1]:.6f})")
    plt.plot(range(len(d1)), np.array(d1), label="Fitness, average")
    plt.plot(range(len(d2)), np.array(d2), label="Fitness, best")
    plt.legend()
    plt.savefig(name)
    # plt.show()