import os
import utils

os.chdir('data(TSP)')

file = 'data-1.txt'
pop_size = 100
gen_size = 6000

D = utils.read_distance_matrix_from_txt(file)
P = utils.init_population(pop_size=pop_size, gen_size=gen_size, num_cities=D.shape[0])

out_file = file.replace('data(TSP)', '').replace('data-', 'fitness')

best, avg_fitnesses = 0, []
for i in range(gen_size):
    fitnesses = [utils.get_fitness(D, chromosome) for chromosome in P]
    avg_fitness = sum(fitnesses) / len(fitnesses)
    avg_fitnesses.append(avg_fitness)

    if utils.is_promising_for_salesman(avg_fitness, best):
        # store ->
        best = avg_fitness
        # (?)

        # (?)

        # (?)

# ?
# print(avg_fitness)