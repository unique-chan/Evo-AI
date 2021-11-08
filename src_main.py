import numpy as np
import random
import utils

''' [How to run?]
1) Install the python libraries as in requirements.txt
    pip install -r requirements.txt
2) 계속 작성해야 함.
'''

''' [Read me]
ID: 20212047
Name: 김예찬
추후 작성해야 함.
'''

# hyper-parameters
pop_size = 100
individual_length = 50
gen_size = 300
elitism_ratio = 0.02

soft_tournament_prob = 0.5
crossover_prob = 0.9

# for replication
np.random.seed(42)
random.seed(34)

# initialize
P = utils.init_population(population_size=100, individual_length=50)

# best, avg_fitnesses, best_fitnesses = 0, [], []
# for i in range(gen_size):
#     # calculate fitness scores
#     fitnesses = [utils.get_fitness(chromosome) for chromosome in P]
#     avg_fitness = sum(fitnesses) / len(fitnesses)
#     avg_fitnesses.append(avg_fitness)
#     if i % 50 == 0:
#         print(f'Iteration {i+1:6d} - avg_fitness: {avg_fitness:.6f} - best: {best:.6f}')
#
#     # store the current result if so far best
#     if utils.is_promising_for_FourMax(avg_fitness, best):
#         utils.store(P, filename='fourmax.txt')
#         best = avg_fitness
#     best_fitnesses.append(best)
#
#     # elitism (1): separate elite strings from the population, P.
#     elite_P, P, fitnesses = utils.elitism(P, fitnesses, elitism_ratio)
#
#     # selection
#     utils.tournament_selection(P, soft_tournament_prob=soft_tournament_prob)
#
#     # crossover
#     utils.crossover(P, crossover_prob)
#
#     # mutation
#
#
#
# # visualization
