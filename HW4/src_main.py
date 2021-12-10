import copy
import argparse

import pandas as pd

from src_operators import *
from src_utils import *

''' [How to run?]
1) Install the python libraries as in requirements.txt
    pip install -r requirements.txt
2) Run the code
    python src_main.py --file_path="data(gp)/data-gp1.txt" --pop_size=500 --gen_size=500 --tree_depth_max=15
'''

# parser
parser = argparse.ArgumentParser(description='2021 Evo AI HW4 - Yechan Kim (20212047)')
parser.add_argument('--file_path', type=str)
parser.add_argument('--pop_size', default=500, type=int)
parser.add_argument('--gen_size', default=500, type=int)
parser.add_argument('--tree_depth_max', default=15, type=int)
args = parser.parse_args()

# hyper-parameters
pop_size = args.pop_size
gen_size = args.gen_size
tree_depth_max = args.tree_depth_max

soft_tournament_prob = 0.8
crossover_prob = 0.9
mutation_prob = 0.01

file_path = args.file_path

print(f'{file_path} is opened.')
# data normalization ('y' value) // z-score normalization for better learning
data_gp = pd.read_csv(file_path, sep=',')
y_mean, y_std = data_gp['y'].mean(), data_gp['y'].std()
print(f'y_mean, y_std: {y_mean}, {y_std}')
data_gp['y'] = (data_gp['y'] - y_mean) / y_std

# initialize
symbol = 'x'
operands = generate_operands(symbol, num_symbol=5, num_constant=5, min_constant=-1, max_constant=1)
print(f"symbol: '{symbol}', bi_operators: {bi_operators}, uni-operators: {uni_operators}, operands: {operands}")

population = init_population(pop_size, tree_depth_max, operands, symbol)

best_avg_fitness, best_population, best_avg_fitnesses, avg_fitnesses = np.infty, population, [], []
for i in range(gen_size):
    # calculate fitness scores for each chromosome in population
    fitnesses = [get_fitness_score(chromosome, symbol, data_gp) for chromosome in population]
    avg_fitness = sum(fitnesses) / len(fitnesses)
    avg_fitnesses.append(avg_fitness)

    # store the current result if so far best
    if avg_fitness < best_avg_fitness:  # in this task, fitness score (~= error) should be minimized.
        best_avg_fitness = avg_fitness
        best_population = copy.deepcopy(population)
    best_avg_fitnesses.append(best_avg_fitness)

    _, fitness = get_best_symbolic_formula(best_population, symbol, data_gp)

    # log print
    print(f"Iteration {i+1: 6d} - avg_fitness: {avg_fitness: .6f} - best_avg_fitness: {best_avg_fitness: .6f} "
          f"- best_model's fitness: {fitness}")

    # temp code
    # prediction = [calculate(_, symbol, symbol_val=x) for x in data_gp['x']]
    # visualize_prediction_plot(data_gp, prediction, title=f'{file_path[:-4]} \n- best fitness_score: {fitness}')
    # visualize_tree(_, visualize=False, output_file=f'{file_path[:-4]} - formula.txt')

    # selection
    tournament_selection(population, symbol, data_gp, soft_tournament_prob)

    # crossover
    crossover(population, crossover_prob)

    # mutation
    mutation(population, operands, mutation_prob)

# visualize fitnesses plot
visualize_fitnesses_plot(best_avg_fitnesses, avg_fitnesses,
                         title=f'{file_path[:-4]} - fitnesses')

# get and visualize the best symbolic formula
best_symbolic_formula, fitness = get_best_symbolic_formula(best_population, symbol, data_gp)
print('best symbolic formula:')
visualize_tree(best_symbolic_formula, output_file=f'{file_path[:-4]} - formula.txt')

# with given dataset, visualize the estimation result of the best symbolic formula
prediction = [calculate(best_symbolic_formula, symbol, symbol_val=x) for x in data_gp['x']]
# VERY IMPORTANT!!!
data_gp['y'] = data_gp['y'] * y_std + y_mean                # original scale!
prediction = [pre * y_std + y_mean for pre in prediction]   # prediction scale should be increased!
visualize_prediction_plot(data_gp, prediction, title=f'{file_path[:-4]} - prediction')
