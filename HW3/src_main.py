import numpy as np
import random
import utils

''' [How to run?]
1) Install the python libraries as in requirements.txt
    pip install -r requirements.txt
2) Run the code
    python src_main.py
'''

''' [Read me]
ID: 20212047
Name: 김예찬

본 과제를 위해 "overlap selection"과 "sharing method"를 접목해보았습니다.
특히, "sharing method"를 취하여 fitness function을 modification한 결과, 
보다 더욱 유의미하게 diverse한 (본래 4-max problem 취지에 맞는) chromosome solution set을 구할 수 있음을 확인하였습니다. 
위 "overlap selection" + "sharing method"이 함께 적용된 실험 결과는 fourmax.png / fourmax.txt을 확인하면 됩니다.

참고로, "sharing method"를 적용하지 않은 방법의 실험 결과는 
'./if-no-sharing-method-is-applied' 디렉토리에 동봉된 파일을 확인하면 됩니다.
'''

# hyper-parameters
pop_size = 100
individual_length = 50
gen_size = 300
elitism_ratio = 0.2

soft_tournament_prob = 0.5
crossover_prob = 0.9
mutation_prob = 0.01

sharing_threshold = 55

# for replication
np.random.seed(42)
random.seed(34)

# initialize
P = utils.init_population(population_size=100, individual_length=50)

best, avg_fitnesses, best_fitnesses = 0, [], []
for i in range(gen_size):
    # calculate fitness scores (no sharing method)
    fitnesses = [utils.get_fitness(chromosome) for chromosome in P]
    avg_fitness = sum(fitnesses) / len(fitnesses)
    avg_fitnesses.append(avg_fitness)
    if (i + 1) % 50 == 0:
        print(f'Iteration {i+1:6d} - avg_fitness: {avg_fitness:.6f} - best: {best:.6f}')

    # store the current result if so far best
    if utils.is_promising_for_FourMax(avg_fitness, best):
        utils.store(P, filename='fourmax.txt')
        best = avg_fitness
    best_fitnesses.append(best)

    # calculate modified fitness scores (adopting sharing method with hamming distance!)
    H = utils.get_hamming_distance_matrix(P)
    fitnesses = [utils.get_modified_fitness(i, P, H, sharing_threshold) for i in range(len(P))]

    # elitism (overlap selection) (1): separate elite strings from the population, P.
    elite_P, P, fitnesses = utils.elitism(P, fitnesses, elitism_ratio)

    # selection
    utils.tournament_selection(P, soft_tournament_prob=soft_tournament_prob)

    # crossover
    utils.crossover(P, crossover_prob)

    # mutation
    utils.bitwise_mutation(P, mutation_prob)

    # elitism (overlap selection) (2): elite strings must be alive without being crossover and mutation.
    P = np.concatenate((P, elite_P))


# visualization
utils.plot(avg_fitnesses, best_fitnesses, filename=f'fourmax.png')