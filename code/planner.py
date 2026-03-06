import heapq
import numpy as np
from metrics import compute_multiobjective_metrics

class Environment:
    def __init__(self, grid, static_obstacles=None):
        self.grid = grid
        self.static_obstacles = set(static_obstacles or set())

        # Reservations for higher-priority agents
        self.vertex_res = set()       # (x, y, t)
        self.edge_res = set()         # ((x1,y1,t1), (x2,y2,t2))
        self.goal_res = {}            # (x,y) -> earliest time goal is blocked forever

    def copy(self):
        env2 = Environment(self.grid, self.static_obstacles.copy())
        env2.vertex_res = self.vertex_res.copy()
        env2.edge_res = self.edge_res.copy()
        env2.goal_res = self.goal_res.copy()
        return env2

    # Reserve goal cell from arrival time → infinity
    def block_goal_cell(self, cell, t_goal):
        self.goal_res[cell] = t_goal

    # Check if next cell is blocked
    def is_blocked(self, x, y, t):
        # static obstacles
        if (x, y) in self.static_obstacles:
            return True

        # vertex conflicts
        if (x, y, t) in self.vertex_res:
            return True

        # dynamic goal blocking
        if (x, y) in self.goal_res and t >= self.goal_res[(x, y)]:
            return True

        return False

    # For checking edge conflicts inside A*
    def edge_conflict(self, prev, nxt):
        (x1, y1, t1) = prev
        (x2, y2, t2) = nxt 
        return (((x2, y2, t1), (x1, y1, t2)) in self.edge_res)

    # Reserve a full path (vertex & edges)
    def reserve_path(self, path):
        for i in range(len(path)):
            x, y, t = path[i]
            self.vertex_res.add((x, y, t))

            # store directed edge (t-1 → t)
            if i > 0:
                px, py, pt = path[i-1]
                self.edge_res.add(((px, py, pt), (x, y, t)))

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def time_astar(env, start, goal, max_t=200):
    open_set = []
    heapq.heappush(open_set, (0, 0, (start[0], start[1], 0), [(start[0], start[1], 0)]))
    visited = set()

    while open_set:
        _, g, (x, y, t), path = heapq.heappop(open_set)

        if (x, y, t) in visited:
            continue
        visited.add((x, y, t))

        # goal reached
        if (x, y) == goal:
            return path, True

        if t >= max_t:
            continue

        # explore 5 actions
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(0,0)]:
            nx, ny = x + dx, y + dy
            nt = t + 1

            # bounds
            if not (0 <= nx < env.grid[0] and 0 <= ny < env.grid[1]):
                continue

            # static/vertex/goal block
            if env.is_blocked(nx, ny, nt):
                continue

            # check edge conflict
            if env.edge_conflict((x, y, t), (nx, ny, nt)):
                continue

            # success → expand neighbor
            new_cost = g + 1
            priority = new_cost + abs(nx - goal[0]) + abs(ny - goal[1])
            heapq.heappush(open_set, (priority, new_cost,
                                      (nx, ny, nt),
                                      path + [(nx, ny, nt)]))
          
    return [], False

def validate_paths(paths):
    # Gather for basic conflicts
    vertices = set()
    edges = set()

    # Record each agent’s goal occupancy interval
    goal_intervals = {}   # (x,y) -> earliest time occupied forever
    
    # First pass: detect basic MAPF conflicts (vertex + edge swap)
    for pid, path in paths.items():
        # Record goal interval
        gx, gy, gt = path[-1]
        if (gx,gy) not in goal_intervals:
            goal_intervals[(gx,gy)] = gt
        else:
            # take earliest arrival
            goal_intervals[(gx,gy)] = min(goal_intervals[(gx,gy)], gt)
    
    # SECOND PASS: Detect GOAL OCCUPANCY COLLISIONS    
    for pid, path in paths.items():
        for (x,y,t) in path:
            for (gx,gy), Tg in goal_intervals.items():
                # If this (x,y,t) is a goal of another agent
                # and t >= Tg, that is a conflict
                if (x == gx and y == gy and t > Tg):
                    # But allow if this is the same agent's own goal
                    if not (path[-1][0] == gx and path[-1][1] == gy):
                        return False
    
    return True

def simulate_order_once(order, players, goals, powers, env_local):

    paths = {}
    player_costs = []
    dyn_blocks = {}
    power_reorder = np.array(powers)[list(order)]
    for pid in order:
        path, ok = time_astar(env_local, players[pid], goals[pid])
        
        if not ok or len(path) == 0:
            # If ANY agent fails, STOP EARLY with huge cost
            return 9999, 9999, 0, 0, 0, 9999, {}, False
        
        paths[pid] = path 
        player_costs.append(len(path))
        # Reserve goal forever
        gx, gy, gt = path[-1]
        env_local.block_goal_cell((gx,gy), gt)
        # Reserve path in table
        env_local.reserve_path(path)
    
    if not validate_paths(paths):
        return 9999,9999,0,0,0,9999,{}, False
    
    # Compute multi-objective metrics
    C, F, W, G, B = compute_multiobjective_metrics(player_costs, power_reorder, paths, dyn_blocks)
    S = W + G + B
    alph = 0.5
    E = alph * C + (1-alph) * S
    return C, S, W, G, B, E, paths, True
