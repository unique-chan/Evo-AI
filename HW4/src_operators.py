import numpy as np

ADD = 'add'
SUB = 'sub'
MUL = 'mul'
POW = 'pow'
# EXP = 'exp'
SIN = 'sin'
COS = 'cos'
TAN = 'tan'
LOG = 'log_abs'         # no negative value is allowed!
SQRT = 'sqrt_abs'       # no negative value is allowed!

bi_operators = [ADD, SUB, MUL]
# operators that require two operands.  (e.g. ADD(3, 4))
uni_operators = [POW, SIN, COS, TAN, LOG, SQRT] * 2
# operators that require one operand.   (e.g. POW(10))


def add(left, right):
    return left + right


def sub(left, right):
    return left - right


def mul(left, right):
    return left * right


def pow(child):
    return child ** 2


# def exp(child):
#     print('exp-child', child)
#     return np.exp(child)


def log_abs(child):
    return np.log(abs(child) + 1e-4)
    # 1e-4 is added not to allow the case [log(0) = -infty].


def sqrt_abs(child):
    return np.sqrt(abs(child))


def sin(child):
    return np.sin(child)


def cos(child):
    return np.cos(child)


def tan(child):
    return np.tanh(child)
