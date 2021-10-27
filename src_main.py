import sys
import random
import utils
import numpy as np

''' How to run?
python src_main.py [file_path]
e.g. python src_main.py data(TSP)/data-1.txt
'''

# hyper-parameters
pop_size = 100
gen_size = 6000
crossover_prob = 0.9
mutation_prob = 0.01
elitism_ratio = 0.02
num_offsprings_for_crossover = 2  # 1 or 2

# data
file = sys.argv[1]

# for replication
np.random.seed(12)
random.seed(34)

# initialize
D = utils.read_distance_matrix_from_txt(file)
P = utils.init_population(pop_size=pop_size, num_cities=D.shape[0])
out_file = file.replace('data(TSP)/', '').replace('data-', 'fitness-')  # 'fitness-x.txt' where x is integer.

best, avg_fitnesses, best_fitnesses = np.infty, [], []
for i in range(gen_size):
    # calculate fiteness scores for each chromosome in P
    fitnesses = [utils.get_fitness(D, chromosome) for chromosome in P]
    avg_fitness = sum(fitnesses) / len(fitnesses)
    avg_fitnesses.append(avg_fitness)
    if i % 300 == 0:
        print(f'file: {file} \t Iteration {i+1:6d} - avg_fitness: {avg_fitness:.6f} - best: {best:.6f}')

    # store the current result if so far best
    if utils.is_promising_for_salesman(avg_fitness, best):
        utils.store(P, fitnesses, out_file)
        best = avg_fitness
    best_fitnesses.append(best)

    # elitism (1): seperate elite strings from the population, P.
    elite_P, P, fitnesses = utils.elitism(P, fitnesses, elitism_ratio)

    # selection
    utils.tournament_selection(P, D)

    # crossover
    utils.order_one_crossover(P, crossover_prob, num_offsprings=num_offsprings_for_crossover)

    # mutation (re-ordering)
    utils.mutation(P, mutation_prob)

    # elitism (2)
    P += elite_P

# visualization
png_file = out_file.replace('.txt', '.png')  # 'fitness-x.png'
utils.plot(avg_fitnesses, best_fitnesses, name=png_file)
