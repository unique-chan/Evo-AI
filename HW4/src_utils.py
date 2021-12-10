import random
import copy

import pandas as pd
from matplotlib import pyplot as plt
from anytree import Node, RenderTree, PostOrderIter

from src_operators import *


def visualize_tree(root: Node, visualize: bool = True, output_file: str = None,
                   y_mean: float = None, y_std: float = None):
    if visualize:
        for pre, fill, node in RenderTree(root):
            print(f'{pre} {node.name}')

    if output_file:
        f = open(output_file, 'w')
        for pre, fill, node in RenderTree(root):
            f.write(f'{pre} {node.name} \n')
        if y_mean and y_std:
            f.write(f'위 수식 f(x)으로 계산한 결과에, {y_mean} (y_mean)을 곱하고 {y_std} (y_std)을 더한 값이 \n'
                    f'실제 스케일에 해당하는 예측 값입니다. (즉, prediction = f(x) * y_mean + y_std)')
        f.close()


def generate_operands(symbol: str, num_symbol: int, num_constant: int, min_constant: int, max_constant: int) -> list:
    operands = [symbol] * num_symbol + list(np.linspace(min_constant, max_constant, num_constant - 1))
    if 0 in operands:
        operands.remove(0)  # if 0 exists, remove it.
    return operands


def __expand_root_node(operands: list, tree_depth_max: int, tree_depth_cur: int, root: Node, node_cur: Node,
                       symbol: str):
    def ____choice(candidates: list, node_cur_name: str = None, symbol: str = 'x'):
        chosen = random.choice(candidates)
        if chosen == bi_operators:
            return random.choice(bi_operators)
        elif chosen == uni_operators:
            return random.choice(uni_operators)
        else:
            return random.choice(operands)

    if tree_depth_cur < tree_depth_max:
        if node_cur.name in bi_operators:       # case (1-1): parent is bi-operator (such as +, -, *, ...)
                                                # everything can be children and
                                                # num of children must be 2.
            candidates = [bi_operators, uni_operators, operands]
            # candidates = [bi_operators, operands]
            child1 = Node(____choice(candidates), parent=node_cur)
            child2 = Node(____choice(candidates), parent=node_cur)
            __expand_root_node(operands, tree_depth_max,
                               tree_depth_cur=tree_depth_cur + 1, root=root, node_cur=child1, symbol=symbol)
            __expand_root_node(operands, tree_depth_max,
                               tree_depth_cur=tree_depth_cur + 1, root=root, node_cur=child2, symbol=symbol)
        elif node_cur.name in uni_operators:    # case (1-2): parent is uni-operator (such as pow, exp, ...)
                                                # everything can be children and
                                                # num of children must be 1.
            candidates = [bi_operators, uni_operators, operands]
            # candidates = [bi_operators, operands]
            child = Node(____choice(candidates, node_cur.name), parent=node_cur)
            __expand_root_node(operands, tree_depth_max,
                               tree_depth_cur=tree_depth_cur + 1, root=root, node_cur=child, symbol=symbol)
        else:                                   # case (1-3): parent is operand,
                                                # children is not allowed.
            pass
    else:  # tree_depth_cur == tree_depth_max
        if node_cur.name in bi_operators:       # case (2-1): parent is bi-operator (such as +, -, *, ...)
                                                # only 'operand' can be children and
                                                # num of children must be 2.
            candidates = [operands]
            child1 = Node(____choice(candidates), parent=node_cur)
            child2 = Node(____choice(candidates), parent=node_cur)
            __expand_root_node(operands, tree_depth_max,
                               tree_depth_cur=tree_depth_cur + 1, root=root, node_cur=child1, symbol=symbol)
            __expand_root_node(operands, tree_depth_max,
                               tree_depth_cur=tree_depth_cur + 1, root=root, node_cur=child2, symbol=symbol)
        elif node_cur.name in uni_operators:    # case (2-2): parent is uni-operator (such as pow, exp, ...)
                                                # only 'operand' can be children and
                                                # num of children must be 1.
            candidates = [operands]
            child = Node(____choice(candidates), parent=node_cur)
            __expand_root_node(operands, tree_depth_max,
                               tree_depth_cur=tree_depth_cur + 1, root=root, node_cur=child, symbol=symbol)
        else:                                   # case (2-3): parent is operand
                                                # children is not allowed.
            pass


def generate_random_symbol_tree(operands: list, tree_depth_max: int, symbol: str = 'x',
                                visualize: bool = False) -> Node:
    # here, we only assume that root is always bi-operator.
    # initially, we generate root node (linking node).
    op = random.choice(bi_operators)
    root = Node(op)
    __expand_root_node(operands, tree_depth_max,
                       tree_depth_cur=1, root=root, node_cur=root, symbol=symbol)
    if visualize:
        visualize_tree(root)
    return root


def calculate(root: Node, symbol: str = 'x', symbol_val: float = 1.0) -> float:
    def __filter(node_name: str, symbol: str, symbol_val: float):
        return node_name if node_name != symbol else symbol_val

    tokens = [__filter(node.name, symbol, symbol_val) for node in PostOrderIter(root)]
    i = 0
    n = len(tokens)
    stack = []
    try:
        while i < n:
            token = tokens[i]
            stack.append(token)
            if token in bi_operators:
                stack.pop()  # remove operator
                right, left = stack.pop(), stack.pop()
                result = eval(f'{token}({left}, {right})')
                stack.append(result)
            elif token in uni_operators:
                stack.pop()  # remove operator
                child = stack.pop()
                result = eval(f'{token}({child})')
                stack.append(result)
            i += 1
        return stack.pop()
    except Exception as e:
        print(e)
        print('i', i)
        print('token', tokens[i])
        print('tokens:', tokens)
        print('stack:', stack)


def init_population(population_size: int, tree_depth_max: int, operands: list, symbol: str) -> list:
    return [generate_random_symbol_tree(operands=operands,
                                        tree_depth_max=random.randint(tree_depth_max // random.choice([2, 3]),
                                                                      tree_depth_max),
                                        symbol=symbol)
            for _ in range(population_size)]


def get_fitness_score(tree: Node, symbol: str, data_gp: pd.DataFrame) -> float:
    l1_error = 0
    l2_error = 0
    for i in range(len(data_gp)):
        x = data_gp['x'][i]
        y = data_gp['y'][i]
        y_hat = calculate(tree, symbol, symbol_val=x)
        l1_error += abs(y - y_hat)
        l2_error += (y - y_hat) ** 2
    return 0.3 * l1_error + 0.7 * l2_error


def tournament_selection(population: list, symbol: str, data_gp: pd.DataFrame, soft_tournament_prob: float = 0):
    '''
    :param soft_tournament_prob:
        if soft_tournament_prob <= 0:   'strict' tournament selection
        otherwise:                      'soft' tournament selection (in general, soft_tournament_prob >= 0.5)
    '''
    pop_size = len(population)
    for i in range(pop_size):
        pos = random.randint(0, pop_size - 1)
        f_i, f_pos = get_fitness_score(population[i], symbol, data_gp), \
                     get_fitness_score(population[pos], symbol, data_gp)
        if soft_tournament_prob <= 0:  # strict tournament
            if f_i > f_pos:  # in this task, fitness score (~= error) should be minimized.
                population[i] = copy.deepcopy(population[pos])
        else:  # soft tournament
            if f_i > f_pos:
                if random.random() <= soft_tournament_prob:
                    population[i] = copy.deepcopy(population[pos])


def crossover(population: list, crossover_prob: float = 0.9):
    pop_size = len(population)
    for i in range(pop_size):
        if random.random() <= crossover_prob:
            root_tree = population[i]
            if root_tree.children[0] in bi_operators and \
                    root_tree.children[1] in bi_operators:
                for _ in range(random.choice([1, 2, 3])):
                    left_bi_op_nodes = [node for _, _, node in RenderTree(root_tree.children[0])
                                        if node.name in bi_operators]
                    right_bi_op_nodes = [node for _, _, node in RenderTree(root_tree.children[1])
                                         if node.name in bi_operators]

                    if len(left_bi_op_nodes) == 0 or \
                            len(right_bi_op_nodes) == 0:
                        continue

                    left_op, right_op = random.choice(left_bi_op_nodes), random.choice(right_bi_op_nodes)
                    left_op_parent, right_op_parent = copy.deepcopy(left_op.parent), copy.deepcopy(right_op.parent)

                    # cross-over
                    left_op.parent = right_op_parent
                    right_op.parent = left_op_parent


def mutation(population: list, operands: list, mutation_prob: float = 0.01):
    pop_size = len(population)
    for i in range(pop_size):
        if random.random() <= mutation_prob:
            root_tree = population[i]

            cases = [1, 2]
            if random.choice(cases) == 1:
                op_nodes = [node for _, _, node in RenderTree(root_tree)
                            if node.name in bi_operators + uni_operators]
                op_node = random.choice(op_nodes)
                candidates = []
                if op_node.name in bi_operators:
                    candidates = copy.deepcopy(bi_operators)
                elif op_node.name in uni_operators:
                    candidates = copy.deepcopy(uni_operators)
                candidates.remove(op_node.name)
                op_node.name = random.choice(candidates)
            else:
                operand_nodes = [node for _, _, node in RenderTree(root_tree)
                                 if node.name in operands]
                operand_node = random.choice(operand_nodes)
                candidates = []
                if operand_node.name in operands:
                    candidates = copy.deepcopy(operands)
                candidates.remove(operand_node.name)
                operand_node.name = random.choice(candidates)


def get_best_symbolic_formula(population: list, symbol: str, data_gp: pd.DataFrame):
    fitnesses = [get_fitness_score(chromosome, symbol, data_gp) for chromosome in population]
    lowest_error = min(fitnesses)
    idx = fitnesses.index(lowest_error)
    return population[idx], lowest_error


def visualize_prediction_plot(data_gp: pd.DataFrame, prediction: list, title: str):
    plt.title(title)
    plt.scatter(data_gp['x'], data_gp['y'], marker='x', label='Ground Truth')
    plt.scatter(data_gp['x'], prediction, marker='o', label='Prediction')
    plt.legend()
    plt.savefig(title.replace("\n", "") + '.png')
    plt.clf()
    # plt.show()


def visualize_fitnesses_plot(best_avg_fitnesses: list, avg_fitnesses: list, title: str):
    plt.title(title)
    plt.plot(range(1, len(best_avg_fitnesses) + 1), np.array(best_avg_fitnesses), label="Best avg fitnesses")
    plt.plot(range(1, len(avg_fitnesses) + 1), np.array(avg_fitnesses), label="Avg fitnesses")
    plt.legend()
    plt.savefig(title.replace("\n", "") + '.png')
    plt.clf()
    # plt.show()