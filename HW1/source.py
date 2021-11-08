import copy

import matplotlib.pyplot as plt
import numpy as np
import random


def read_data(filename):
    """Parse problem specifications from the data file."""
    with open(filename, "r") as f:
        # header
        for line in f:
            iwp = line.strip().split()
            if len(iwp) >= 4 and iwp[2] == "capacity":
                capacity = float(iwp[3])
            elif iwp == ["item_index", "weight", "profit"]:
                table = True
                break
        if not table:
            raise ValueError("table not found.")
        # body
        weights = []
        profits = []
        for line in f:
            i, w, p = line.strip().split()
            weights.append(float(w))
            profits.append(float(p))
    return capacity, weights, profits


def fitness_function(individual, capacity, weights, profits):
    """Calculate fitness value of an individual."""
    sum_weight = 0
    sum_profit = 0
    for bit, weight, profit in zip(individual, weights, profits):
        if bit == '1':
            sum_weight += weight
            sum_profit += profit

    fitness = sum_profit if sum_weight <= capacity else 0
    return fitness


def initialize(pop_size=10000, gen_size=100):
    """Initialize 100 individuals, each of which consists of 10000 bits"""
    population = []
    for _ in range(gen_size):
        individual = ""
        for _ in range(pop_size):
            individual += "1" if random.random() > 0.5 else "0"
        population.append(individual)
    return population


def roulette_selection(individuals, fitnesses):
    '''
    Yechan Kim's Roulette Wheel Selection
    (Note) This code is implemented based on the pseudo-code in the given lecture note.
    '''
    population_size = len(individuals)
    selections = []
    for i in range(population_size):
        threshold = random.random()
        j = 0
        while j < population_size and threshold > sum(fitnesses[:j+1]) / sum(fitnesses):
            j += 1
        selections.append(individuals[j])
    return selections


def tournament_selection(individuals, capacity, weights, profits):
    '''
    Yechan Kim's Tournament Selection
    (Note) This code is implemented based on the pseudo-code in the given lecture note.
    '''
    population_size = len(individuals)
    for i in range(population_size):
        pos = random.randint(0, population_size-1)
        f_i = fitness_function(individuals[i], capacity, weights, profits)
        f_pos = fitness_function(individuals[pos], capacity, weights, profits)
        if f_i < f_pos:
            individuals[i] = individuals[pos]
    return individuals


def crossover_3points(individuals, prob):
    '''
    Yechan Kim's 3-points Crossover
    (Note) This code is implemented by extending the crossover pseudo-code for one-point to the code for 3-points.
    '''
    individual_length = len(individuals[0])
    population_size = len(individuals)
    population_size_half = population_size // 2
    random.shuffle(individuals)  # To address positional bias issue, shuffling is needed at first.
    for i in range(population_size_half):
        # 3-point crossover
        if random.random() <= prob:
            positions = sorted(np.random.choice(list(range(0, individual_length)), size=3, replace=False))
            dad, mom = individuals[i], individuals[i + population_size_half]
            offspring1 = dad[0:positions[0]] + mom[positions[0]:positions[1]] +\
                         dad[positions[1]:positions[2]] + mom[positions[2]:]
            offspring2 = mom[0:positions[0]] + dad[positions[0]:positions[1]] +\
                         mom[positions[1]:positions[2]] + dad[positions[2]:]
            individuals[i] = offspring1
            individuals[i + population_size_half] = offspring2
    return individuals


def bitwise_mutation(individuals, prob):
    '''
    Yechan Kim's Bitwise_mutation
    (Note) This code is implemented based on the pseudo-code in the given lecture note.
    '''
    for i, individual in enumerate(individuals):
        mutated = ''
        for gene in individual:
            if random.random() < prob:
                mutated += '1' if gene == '0' else '0'
            else:
                mutated += gene
        individuals[i] = mutated
    return individuals


def elitism(elitism_ratio, individuals, fitnesses):
    '''
    Yechan Kim's Elitism
    1) Top-N elites must be survived without being cross-overed or mutated.
    2) Bottom-N elites (= Top-N idiots) must be replaced with copies of Top-N elites.
    3) The remaining individuals including (replaced) top-N idiots must be selected, cross-overed, and mutated.
    4) Here N is computed as: population_size * elitism_ratio.
    '''
    num_elitism = int(len(individuals) * elitism_ratio)
    top_ids = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[-num_elitism:]
    bottom_ids = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:num_elitism]

    for i, (top_id, bottom_id) in enumerate(zip(top_ids, bottom_ids)):
        individuals[bottom_id] = individuals[top_id]
        individuals[i], individuals[top_id] = individuals[top_id], individuals[i]

    elite_individuals = individuals[:num_elitism]
    individuals = individuals[num_elitism:]
    fitnesses = fitnesses[num_elitism:]
    return elite_individuals, individuals, fitnesses


def plot(d1, d2, figure='trace.png'):
    plt.title("0/1 Knapsack fitness value trace")
    plt.plot(range(len(d1)), d1, label="Roulette Wheel Selection")
    plt.plot(range(len(d2)), d2, label="Pairwise Tournament Selection")
    plt.legend()
    plt.savefig(figure)
    plt.show()


def store(individuals, fitnesses, filename='test.txt'):
    txt = ''
    for individual, fitness in zip(individuals, fitnesses):
        txt += "{},{:.6f}\n".format(individual, fitness)
    with open(filename, "w") as f:
        f.write(txt)


if __name__ == '__main__':
    # 0) Settings
    spec = read_data(filename="Data(0-1Knapsack).txt")
    Pc = 0.9    # crossover probability
    Pm = 0.01   # mutation probability
    N_generations = 100   # generations

    # 1) Roulette wheel selection, 3-point crossover, and bitwise mutation
    # 1-1) Initialize
    print("1) Roulette wheel selection, 3-point crossover, and bitwise mutation")
    elitism_ratio_1 = 0.05  # 0.0: No elitism, 0.1: Top-10% elites from the population will be automatically survived.
    avg_fitnesses_1 = []
    population1 = initialize(pop_size=10000, gen_size=100)
    individuals = np.array(population1)

    best = 0
    for i in range(N_generations):
        # 1-2) Calc fitness for each individual chromosome
        fitnesses = [fitness_function(individual, *spec) for individual in individuals]
        avg_fitness = np.mean(fitnesses)
        avg_fitnesses_1.append(avg_fitness)
        print(f'{i}: {avg_fitness}')

        # 1-2-A) Store the best result so far
        if avg_fitness > best:
            store(individuals, fitnesses, 'roulette.txt')
            best = avg_fitness

        # 1-3) Choose elite chromosomes
        elite_individuals = None
        if elitism_ratio_1 > 0.0:
            elite_individuals, individuals, fitnesses = elitism(elitism_ratio_1, individuals, fitnesses)

        # 1-4) Roulette wheel selection
        individuals = roulette_selection(individuals, fitnesses)

        # 1-5) 3-points Crossover
        individuals = crossover_3points(individuals, prob=Pc)

        # 1-6) Bitwise Mutation
        individuals = bitwise_mutation(individuals, prob=Pm)

        # 1-7) Elite chromosomes survived without being cross-overed or mutated
        if elite_individuals is not None:
            individuals = np.concatenate((individuals, elite_individuals))

    print('best:', best)

    ################################################################################################
    # 2) Pairwise tournament selection, 3-point crossover, and bitwise mutation
    # 2-1) Initialize
    print('=' * 70)
    print("2) Pairwise tournament selection, 3-point crossover, and bitwise mutation")
    elitism_ratio_2 = elitism_ratio_1  # for fair comparison, use same elitism ratio as in 1)
    avg_fitnesses_2 = []
    population2 = initialize(pop_size=10000, gen_size=100)
    individuals = copy.deepcopy(population2)

    best = 0
    for i in range(N_generations):
        # 2-2) Calc fitness for each individual chromosome
        fitnesses = [fitness_function(individual, *spec) for individual in individuals]
        avg_fitness = np.mean(fitnesses)
        avg_fitnesses_2.append(avg_fitness)
        print(f'{i}: {avg_fitness}')

        # 2-2-A) Store the best result so far
        if avg_fitness > best:
            store(individuals, fitnesses, 'tournament.txt')
            best = avg_fitness

        # 2-3) Choose elite chromosomes
        elite_individuals = None
        if elitism_ratio_2 > 0.0:
            elite_individuals, individuals, fitnesses = elitism(elitism_ratio_2, individuals, fitnesses)

        # 2-4) Tournament selection
        individuals = tournament_selection(individuals, *spec)

        # 2-5) 3-points crossover
        individuals = crossover_3points(individuals, prob=Pc)

        # 2-6) Bitwise mutation
        individuals = bitwise_mutation(individuals, prob=Pm)

        # 2-7) Elite chromosomes survived without being cross-overed or mutated
        if elite_individuals is not None:
            individuals = np.concatenate((individuals, elite_individuals))

    print('best:', best)

    # Visualization: Roulette wheel selection vs Tournament selection
    plot(avg_fitnesses_1, avg_fitnesses_2, figure='trace.png')