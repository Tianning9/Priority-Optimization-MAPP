import random
import numpy as np

def swap_adjacent(order):
    o = list(order)
    i = random.randint(0, len(o)-2)
    o[i], o[i+1] = o[i+1], o[i]
    return tuple(o)

def swap_random(order):
    o = list(order)
    i, j = random.sample(range(len(o)), 2)
    o[i], o[j] = o[j], o[i]
    return tuple(o)

def rotate_three(order):
    o = list(order)
    i = random.randint(0, len(o)-3)
    o[i], o[i+1], o[i+2] = o[i+1], o[i+2], o[i]
    return tuple(o)

def insert_move(order):
    N = len(order)
    i, j = random.sample(range(N), 2)

    order = list(order)
    elem = order.pop(i)
    order.insert(j, elem)

    return tuple(order)

def neighbor(order):
    """Return a local-random mutated permutation."""
    ops = [swap_adjacent, swap_random, rotate_three, insert_move]
    return random.choice(ops)(order)

def crossover(o1, o2):
    N = len(o1)
    cut = np.random.randint(1, N-1)
    child = list(o1[:cut])
    for x in o2:
        if x not in child:
            child.append(x)
    return tuple(child)

def mutate(order, p=0.25):
    if random.random() < p:
        return neighbor(order)
    return order
