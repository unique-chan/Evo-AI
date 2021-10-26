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
    :return: fitness_score of the current chromsome (route)
    '''
    total_distance = [distance_matrix[chromosome[i-1]][chromosome[i]] for i in range(1, len(chromosome))]
    fitness = sum(total_distance)
    return fitness


def is_promising_for_salesman(cur, best):
    return True if cur < best else False     # shortest route is better.


def


def selection(mode):
    pass


def crossover(mode):
    pass


def mutation(mode):
    pass
