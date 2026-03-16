import numpy as np
from planner import Environment

from experiments import extract_best
from experiments import run_50_trials
from experiments import plot_environment_clean
from experiments import summarize_and_print_tables
from experiments import plot_50_trial_trends_clean
from experiments import plot_valid_invalid_by_iteration
from experiments import plot_random_valid_invalid_histogram
from experiments import plot_random_distribution_valid_only

from visualization import simulate_paths

# grid size
grid_10 = (7, 7)

# first obstacle configuration - easy
obs_10_a = {(2, 2)}

# second obstacle configuration - medium
obs_10_b = set()
obs_10_b |= {(1, 2), (3, 2)} 

# third obstacle configuration - hard
obs_10_c = {(1, 1), (1, 3), (3, 3)}

# configuraiton of agents - start and goal positions
starts_10 = [
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
    (0, 4), (1, 4), (2, 4), (3, 4), (4, 4)
]
goals_10 = [
    (0, 4), (1, 4), (2, 4), (3, 4), (4, 4),
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)
]

obstacles_10 = [obs_10_a, obs_10_b, obs_10_c]
# The start battery volume of each agent
powers_10 = [30, 25, 20, 25, 30, 15, 20, 25, 20, 15]

env = Environment(grid_10, obs_10_a)
all_valid_E_10_a, invalid_counts_10_a = plot_random_distribution_valid_only(starts_10, goals_10, powers_10, env,
                                        iterations=40, trials=30,
                                        env_name="(10 agents, Type A)",
                                        invalid_threshold=9999)
plot_random_valid_invalid_histogram(all_valid_E_10_a, invalid_counts_10_a, env_name="(10 agents, Type A)")

env = Environment(grid_10, obs_10_b)
all_valid_E_10_b, invalid_counts_10_b = plot_random_distribution_valid_only(starts_10, goals_10, powers_10, env,
                                        iterations=40, trials=30,
                                        env_name="(10 agents, Type B)",
                                        invalid_threshold=9999)
plot_random_valid_invalid_histogram(all_valid_E_10_b, invalid_counts_10_b, env_name="(10 agents, Type B)")

env = Environment(grid_10, obs_10_c)
all_valid_E_10_c, invalid_counts_10_c = plot_random_distribution_valid_only(starts_10, goals_10, powers_10, env,
                                        iterations=40, trials=30,
                                        env_name="(10 agents, Type C)",
                                        invalid_threshold=9999)
plot_random_valid_invalid_histogram(all_valid_E_10_c, invalid_counts_10_c, env_name="(10 agents, Type C)")

# Grid world map with three types of obstacle configurations 
rows = 7
cols = 7
font = 16
size = (8,6)
fonts = 16
s = 8
plot_environment_clean(
    s, fonts, size, font, rows, cols, starts_10, goals_10, obs_10_a,
    title="10-agent 7×7 Grid World (Obstacle Type A)"
)

plot_environment_clean(
    s, fonts, size, font, rows, cols, starts_10, goals_10, obs_10_b,
    title="10-agent 7×7 Grid World (Obstacle Type B)"
)
plot_environment_clean(
    s, fonts, size, font, rows, cols, starts_10, goals_10, obs_10_c,
    title="10-agent 7×7 Grid World (Obstacle Type C)"
)

# case a - simple obstacle configuration
obstacles_10 = [obs_10_a]
for i, static_obs in enumerate(obstacles_10):
    print(f"\n===== Subcase {i+1} =====")
    env_10 = Environment(grid_10, static_obs)
    SA_scores_10_a, GA_scores_10_a, BO_scores_10_a, RAND_scores_10_a, SA_trends_10_a, GA_trends_10_a, BO_trends_10_a = \
        run_50_trials(
            starts_10,
            goals_10,
            powers_10,
            env_10,
            step=40, 
            pop_size=6, 
            generations=40,
            n_iter=40, 
            trials=3) 
    plot_50_trial_trends_clean(SA_trends_10_a, GA_trends_10_a, BO_trends_10_a, "a", 10)
    
    plot_valid_invalid_by_iteration(SA_trends_10_a, "SA", "a", 10)
    plot_valid_invalid_by_iteration(GA_trends_10_a, "GA", "a", 10)
    plot_valid_invalid_by_iteration(BO_trends_10_a, "BO", "a", 10)

    # tables
    SA_stats_10_a, GA_stats_10_a, BO_stats_10_a, RAND_stats_10_a = summarize_and_print_tables(
        SA_scores_10_a, GA_scores_10_a, BO_scores_10_a, RAND_scores_10_a,
        case_name=f"10-Player Case — Obstacle Set a"
    )
best_rand_10_a = extract_best(RAND_scores_10_a)
best_sa_10_a   = extract_best(SA_scores_10_a)
best_ga_10_a   = extract_best(GA_scores_10_a)
best_bo_10_a   = extract_best(BO_scores_10_a)

positions = list(obs_10_a) 
results_sim_10_a = {
    "Random": best_rand_10_a,
    "SA": best_sa_10_a,
    "GA": best_ga_10_a,
    "BO": best_bo_10_a
}

for name in ["Random", "SA", "GA", "BO"]:
    print(f"Simulating: {name}")
    best = results_sim_10_a[name]
    order = best["order"]
    paths_dict = best["paths"]

    if not paths_dict or paths_dict == {} or order is None or (isinstance(order, tuple) and order[0] == 9999):
        print(f"{name} has no valid safe solution. Skipping simulation.")
        continue

    ordered_paths = {pid: paths_dict[pid] for pid in order}

    simulate_paths(ordered_paths, grid_10, positions)
