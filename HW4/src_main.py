import copy

import pandas as pd

from src_operators import *
from src_utils import *

''' [How to run?]
1) Install the python libraries as in requirements.txt
    pip install -r requirements.txt
2) Run the code
    python src_main.py
'''

''' [Read me]
ID: 20212047
Name: 김예찬
'''

# hyper-parameters
pop_size = 500
gen_size = 500
tree_depth_max = 10

soft_tournament_prob = 0.5
crossover_prob = 0.9
mutation_prob = 0.01

file_paths = ['data(gp)/data-gp1.txt', 'data(gp)/data-gp2.txt']
for file_path in file_paths:
    print(f'{file_path} is opened.')
    data_gp = pd.read_csv(file_path, sep=',')

    # initialize
    symbol = 'x'
    operands = generate_operands(symbol, num_symbol=6, num_constant=10, min_constant=-5, max_constant=5)
    print(f"symbol: '{symbol}', bi_operators: {bi_operators}, uni-operators: {uni_operators}, operands: {operands}")

    population = init_population(pop_size, tree_depth_max, operands)

    best_fitness, best_population, best_fitnesses, avg_fitnesses = np.infty, population, [], []
    for i in range(gen_size):
        # calculate fitness scores for each chromosome in population
        fitnesses = [get_fitness_score(chromosome, symbol, data_gp) for chromosome in population]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        avg_fitnesses.append(avg_fitness)
        # if i % 50 == 0:
        print(f'Iteration {i+1: 6d} - avg_fitness: {avg_fitness: .6f} - best: {best_fitness: .6f}')

        # store the current result if so far best
        if avg_fitness < best_fitness:  # in this task, fitness score (~= error) should be minimized.
            best_fitness = avg_fitness
            best_population = copy.deepcopy(population)
        best_fitnesses.append(best_fitness)

        # selection
        tournament_selection(population, symbol, data_gp, soft_tournament_prob)

        # crossover
        crossover(population, crossover_prob)

        # mutation
        mutation(population, operands, mutation_prob)

    # get and visualize the best symbolic formula
    best_symbolic_formula, fitness = get_best_symbolic_formula(best_population, symbol, data_gp)
    print('best symbolic formula:')
    visualize_tree(best_symbolic_formula)

    # with given dataset, visualize the estimation result of the best symbolic formula
    prediction = [calculate(best_symbolic_formula, symbol, symbol_val=x) for x in data_gp['x']]
    visualize_plot(data_gp, prediction, title=f'{file_path} \n- best fitness_score: {fitness}')
