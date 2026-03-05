import time
import math
import random
import numpy as np
from operator import neighbor
from planner import simulate_order_once

def simulated_annealing(players, goals, power, env, T0=5.0, Tmin=1e-1, cool=0.99, steps=500):

    t_start = time.time()
    N = len(players)

    # logs
    iter_hist = []
    best_hist = []
    curr_hist = []
    valid_flags = []
    # initial state
    curr = tuple(np.random.permutation(N))
    C, S, W, G, B, E, paths, valid = simulate_order_once(curr, players, goals, power, env.copy())

    best_order = curr
    best_metrics = (C, S, W, G, B, E)
    best_paths = paths
    curr_cost = E
    best_cost = E

    iter_hist.append(0)
    best_hist.append(best_cost)
    curr_hist.append(curr_cost)
    valid_flags.append(valid)
    T = T0
    flag = valid
    # SA loop
    for step in range(1, steps):

        cand = neighbor(curr)
        Cm, Sm, Wm, Gm, Bm, Em, paths_m, valid = simulate_order_once(cand, players, goals, power, env.copy())
        
        accept = (Em < curr_cost) or (random.random() < math.exp(-(Em - curr_cost)/T))
        if accept:
            curr = cand
            curr_cost = Em

        # update global best
        if Em < best_cost:
            best_order = cand
            best_cost = Em
            best_metrics = (Cm, Sm, Wm, Gm, Bm, Em)
            best_paths = paths_m
            flag = valid
        valid_flags.append(flag)
        iter_hist.append(step)
        best_hist.append(best_cost)
        curr_hist.append(curr_cost)

        # cool down
        T = max(Tmin, T * cool)

    total_time = time.time() - t_start

    return best_order, best_metrics, best_paths, iter_hist, best_hist, curr_hist, valid_flags, total_time
