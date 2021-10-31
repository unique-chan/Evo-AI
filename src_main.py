import sys
import random
import utils
import numpy as np

''' How to run?
1) Install the python libraries as in requirements.txt
    pip install -r requirements.txt
2) Then, run the code as follows:
    python src_main.py [file_path]
    e.g. python src_main.py data(TSP)/data-1.txt
'''

# hyper-parameters
pop_size = 100
gen_size = 6000
crossover_prob = 0.9
soft_tournament_prob = 0.5
# boltzmann_prob = 0.8
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

# for logarithmic scaling on boltzmann_selection / roulette_selection
# T = 40

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
    # [Note] I tried to implement various selection algorithms as in below (A)-(E).
    # Yechan's opinion:     I strongly recommend you use 'tournament selection.'
    #                       Especially, please try to use 'soft tournament selection' for your experiments.

    #########################################################################################################
    # selection option (A): tournament selection - strict version
    #                                                   -> [opinion] great option for the given dataset!
    # utils.tournament_selection(P, D)
    #########################################################################################################

    #########################################################################################################
    # selection option (B): tournament selection - soft version
    #                                                   -> [opinion] very great option for the given dataset!
    # [Note] All the submission results (fitness-*.txt / fitness-*.png) are based on using (B)!
    utils.tournament_selection(P, D, soft_tournament_prob=soft_tournament_prob)
    #########################################################################################################

    #########################################################################################################
    # selection option (C): boltzmann_selection
    #                                                   -> [opinion] really terrible for the given dataset!
    # if i >= 1:
    #     if utils.is_promising_for_salesman(best_fitnesses[i], best_fitnesses[i-1]):
    #         T *= 0.99  # it should be tuned!
    # utils.boltzmann_selection(P, D, boltzmann_prob, T)
    #########################################################################################################

    #########################################################################################################
    # selection option (D): roulette selection
    #                                                   -> [opinion] really terrible for the given dataset!
    # P = utils.roulette_selection(P, fitnesses)
    #########################################################################################################

    #########################################################################################################
    # selection option (E): roulette selection with logarithmic_scaling
    #                                                   -> [opinion] really terrible for the given dataset!
    # if i >= 1:
    #     if utils.is_promising_for_salesman(best_fitnesses[i], best_fitnesses[i-1]):
    #         T *= 0.99  # it should be tuned!
    # P = utils.roulette_selection(P, fitnesses, logarithmic_scaling_T=T)
    #########################################################################################################

    # crossover
    utils.order_one_crossover(P, crossover_prob, num_offsprings=num_offsprings_for_crossover)

    # mutation (re-ordering)
    utils.mutation(P, mutation_prob)

    # elitism (2): elite strings must be alive without being crossover and mutation.
    P += elite_P

# visualization
png_file = out_file.replace('.txt', '.png')  # 'fitness-x.png'
utils.plot(avg_fitnesses, best_fitnesses, name=png_file)
