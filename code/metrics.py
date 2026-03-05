import numpy as np

def compute_multiobjective_metrics(player_costs, powers, paths, dyn_blocks):
    norm_costs = np.array(player_costs) / (powers+1e-6)
    C = np.sum(player_costs)  
    F = np.max(norm_costs) - np.min(norm_costs)
    # Waiting time
    W = 0
    for pid, path in paths.items():
        for k in range(1, len(path)):
            if path[k][:2] == path[k-1][:2]:
                W += 1

    # Blocking penalty
    B = sum(len(v) for v in dyn_blocks.values())

    # Congestion penalty
    G = 0
    for p1, pa in paths.items():
        for p2, pb in paths.items():
            if p1 >= p2:
                continue
            T = min(len(pa), len(pb))
            for t in range(T):
                ax, ay, _ = pa[t]
                bx, by, _ = pb[t]
                if abs(ax - bx) + abs(ay - by) <= 1:
                    G += 1
    E = C + F
    return C, F, W, G, B 
