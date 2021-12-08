import numpy as np

ADD = 'add'
SUB = 'sub'
MUL = 'mul'
POW = 'pow'
EXP = 'exp'
LOG = 'log'
SQRT = 'sqrt'
SIN = 'sin'
COS = 'cos'
ATAN = 'atan'


bi_operators = [ADD, SUB, MUL]                          # operators that require two operands.  (e.g. ADD(3, 4))
uni_operators = [POW, EXP, LOG, SQRT, SIN, COS, ATAN]   # operators that require one operand.   (e.g. POW(10))


def add(left, right):
    return left + right


def sub(left, right):
    return left - right


def mul(left, right):
    return left * right


def pow(child):
    return child ** 2


def exp(child):
    return np.exp(child)


def log(child):
    return np.log(child)


def sqrt(child):
    return np.sqrt(child)


def sin(child):
    return np.sin(child)


def cos(child):
    return np.cos(child)


def atan(child):
    return np.arctanh(child)
