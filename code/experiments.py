import time
import numpy as np
import matplotlib.pyplot as plt
from planner import simulate_order_once
from optimization.SA import simulated_annealing
from optimization.GA import genetic_algorithm
from optimization.BO import BO_permutation_optimize

def run_all_algorithms(players, goals, power, env, step, pop_size, generations, n_iter):

    results = {}

    # ================= SA =================
    best_sa, sa_metrics, sa_paths, it_sa, best_sa_hist, curr_sa_hist, sa_valid, sa_time = \
        simulated_annealing(players, goals, power, env, steps=step)

    C1, S1, W1, G1, B1, E1 = sa_metrics
    results["SA"] = {
        "order": best_sa,
        "paths": sa_paths,
        "C": C1, "S": S1, "W": W1, "G": G1, "B": B1, "E": E1,
        "time": sa_time,
        "trend": {"iter": it_sa, "best": best_sa_hist, "curr": curr_sa_hist, "valid": sa_valid}
    }

    # ================= GA =================
    best_ga, ga_metrics, ga_paths, it_ga, best_ga_hist, ga_valid, ga_time = \
        genetic_algorithm(players, goals, power, env,
                          pop_size=pop_size, generations=generations)

    C2, S2, W2, G2, B2, E2 = ga_metrics

    results["GA"] = {
        "order": best_ga,
        "paths": ga_paths,
        "C": C2, "S": S2, "W": W2, "G": G2, "B": B2, "E": E2,
        "time": ga_time,
        "trend": {"iter": it_ga, "best": best_ga_hist, "valid": ga_valid}
    }

    # ================= BO =================
    best_bo, bo_metrics, bo_paths, it_bo, cost_bo_hist, best_bo_hist, bo_valid, bo_time = \
        BO_permutation_optimize(players, goals, power, env,
                                eval_budget=n_iter)

    C3, S3, W3, G3, B3, E3 = bo_metrics

    results["BO"] = {
        "order": best_bo,
        "paths": bo_paths,
        "C": C3, "S": S3, "W": W3, "G": G3, "B": B3, "E": E3,
        "time": bo_time,
        "trend": {"iter": it_bo, "curr": cost_bo_hist, "best": best_bo_hist, "valid": bo_valid}
    }

    return results

def plot_valid_invalid_by_iteration(trends, algo_name, obstalce_name, scale_name):
    """
    trends = list of tuples from run_50_trials:
        SA: (iters, best, curr, valid)
        GA: (iters, best, valid)
        BO: (iters, curr, best, valid)

    For each iteration index k:
        - Count how many trials were valid at iteration k
        - Count how many were invalid at iteration k

    Produces a histogram-like plot:
        x-axis = iteration index
        y-axis = counts
    """
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    # Extract validity logs
    valid_logs = []

    for entry in trends:
        if algo_name == "SA":
            _, _, _, valid = entry
        elif algo_name == "GA":
            _, _, valid = entry
        elif algo_name == "BO":
            _, _, _, valid = entry

        valid_logs.append(valid)

    # Pad with False for trials that ended early
    max_len = max(len(v) for v in valid_logs)
    padded = np.full((len(valid_logs), max_len), False)

    for i, v in enumerate(valid_logs):
        padded[i, :len(v)] = v

    # Count valid/invalid per iteration
    valid_counts = padded.sum(axis=0)
    invalid_counts = padded.shape[0] - valid_counts

    # Plot
    x = np.arange(max_len)

    plt.figure(figsize=(10,5))
    plt.bar(x, valid_counts, label="Valid", color="green", alpha=0.7)
    plt.bar(x, invalid_counts, bottom=valid_counts,
            label="Invalid", color="red", alpha=0.7)

    plt.xlabel("Iteration")
    plt.ylabel("Count Across Trials")
    plt.title(f"{algo_name} - Valid vs Invalid by Iteration ({scale_name} agents, Type {obstalce_name})")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

    return valid_counts, invalid_counts

def random_baseline_trial(players, goals, power, env, eval_budget):
    """
    One trial of random baseline:
    - Uses eval_budget random permutations
    - Computes mean C, F, M, E over these evaluations
    - Returns the aggregated trial values
    """
    best_C = None
    best_S = None
    best_E = float("inf")
    best_time = None
    best_order = None
    best_paths = None
    for _ in range(eval_budget):
        order = tuple(np.random.permutation(len(players)))

        t0 = time.time()
        C, S, W, G, B, E, paths, valid = simulate_order_once(
            order, players, goals, power, env.copy()
        )
        t1 = time.time()

        if E < best_E:
            best_E = E
            best_C = C
            best_S = S
            best_time = t1 - t0
            best_order = order
            best_paths = paths
    return best_C, best_S, best_E, best_time, best_order, best_paths

def run_50_trials(players, goals, power, env, step=40, pop_size=6, generations=40, n_iter=40, trials=50):

    SA_scores, GA_scores, BO_scores = [], [], []
    RAND_scores = []
    SA_trends, GA_trends, BO_trends = [], [], []

    # Decide evaluation budget for random baseline (fairness)
    eval_budget = step   
 
    for t in range(trials):
        print(f"=== Trial {t+1}/{trials} ===")

        # --- RANDOM BASELINE ---
        C, S, E, time_rand, order_rand, paths_rand = random_baseline_trial(
            players, goals, power, env,
            eval_budget=eval_budget
        )
        RAND_scores.append((C, S, E, time_rand, order_rand, paths_rand))

        # --- SA / GA / BO ---
        out = run_all_algorithms(
            players, goals, power, env,
            step, pop_size, generations, n_iter
        )

        # Store optimization results
        SA_scores.append((out["SA"]["C"], out["SA"]["S"], out["SA"]["E"], out["SA"]["time"], out["SA"]["order"], out["SA"]["paths"]))
        GA_scores.append((out["GA"]["C"], out["GA"]["S"], out["GA"]["E"], out["GA"]["time"], out["GA"]["order"], out["GA"]["paths"]))
        BO_scores.append((out["BO"]["C"], out["BO"]["S"], out["BO"]["E"], out["BO"]["time"], out["BO"]["order"], out["BO"]["paths"]))

        # Store trend data
        SA_trends.append((out["SA"]["trend"]["iter"], out["SA"]["trend"]["best"], out["SA"]["trend"]["curr"], out["SA"]["trend"]["valid"]))
        GA_trends.append((out["GA"]["trend"]["iter"], out["GA"]["trend"]["best"],out["GA"]["trend"]["valid"]))
        BO_trends.append((out["BO"]["trend"]["iter"], out["BO"]["trend"]["curr"], out["BO"]["trend"]["best"],out["BO"]["trend"]["valid"]))

    return SA_scores, GA_scores, BO_scores, RAND_scores, SA_trends, GA_trends, BO_trends

def summarize_and_print_tables(SA_scores, GA_scores, BO_scores, RAND_scores, case_name="Results"):
    
    def compute_stats(scores):

        filtered = [s for s in scores if s[2] < 9999]

        if len(filtered) == 0:
            return None  # no valid samples

        numeric = np.array([s[:4] for s in filtered], float)
        C = numeric[:, 0]
        S = numeric[:, 1]
        E = numeric[:, 2]
        T = numeric[:, 3]

        return {
            "C_mean": C.mean(), "C_std": C.std(),
            "S_mean": S.mean(), "S_std": S.std(),
            "E_mean": E.mean(), "E_std": E.std(),
            "T_mean": T.mean(), "T_std": T.std(),
            "num_valid": len(filtered)
        }
        
    SA_stats = compute_stats(SA_scores)
    GA_stats = compute_stats(GA_scores)
    BO_stats = compute_stats(BO_scores)
    RAND_stats = compute_stats(RAND_scores)

    print(f"\n===== {case_name}: Mean ± Std over {len(SA_scores)} Trials =====")
    print("{:<10} {:>20} {:>20} {:>20} {:>20}".format(
        "Alg", "Cost", "Safety", "Objective", "Time (s)"
    ))

    def fm(x):
        """Format number as: 64.100 \( \pm \) 1.399"""
        if x[0] is None:
            return "Invalid"
        return f"& {x[0]:.3f} \\( \\pm \\) {x[1]:.3f}"
    
    def print_line(name, s):
        if s is None:
            print("{:<10} {:>20} {:>20} {:>20} {:>20}".format(
                name, "Invalid", "Invalid", "Invalid", "Invalid"
            ))
        else:
            print("{:<10} {:>20} {:>20} {:>20} {:>20}".format(
                name,
                fm([s["C_mean"], s["C_std"]]),
                fm([s["S_mean"], s["S_std"]]),
                fm([s["E_mean"], s["E_std"]]),
                fm([s["T_mean"], s["T_std"]])
            ))
        """
        s["C_mean"], s["C_std"],
        s["S_mean"], s["S_std"],
        s["E_mean"], s["E_std"],
        s["T_mean"], s["T_std"]
        """
    print_line("Random", RAND_stats)
    print_line("SA", SA_stats)
    print_line("GA", GA_stats)
    print_line("BO", BO_stats)

    return SA_stats, GA_stats, BO_stats, RAND_stats

def plot_50_trial_trends_clean(SA_trends, GA_trends, BO_trends, obstacles_name, scale_name):

    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12

    # -------- helper: pad variable-length sequences --------
    def pad(arrays):
        max_len = max(len(a) for a in arrays)
        A = np.full((len(arrays), max_len), np.nan)
        for i, a in enumerate(arrays):
            A[i, :len(a)] = a
        return A

    def extract_iters_and_best(trends, name):
        iters_list = []
        best_list = []

        if name == "SA":
            for (iters, best, curr, valid) in trends:
                T = len(iters)
                fixed_iters = np.array(iters, float)
                fixed_best = np.array([
                    best[i] if valid[i] else np.nan
                    for i in range(T)
                ], dtype=float)

                iters_list.append(fixed_iters)
                best_list.append(fixed_best)

        elif name == "GA":
            for (iters, best, valid) in trends:
                T = len(best)

                # GA does not have explicit iteration numbers → use 0..T-1
                fixed_iters = np.arange(T, dtype=float)

                fixed_best = np.array([
                    best[i] if valid[i] else np.nan
                    for i in range(T)
                ], dtype=float)

                iters_list.append(fixed_iters)
                best_list.append(fixed_best)

        elif name == "BO":
            for (iters, curr, best, valid) in trends:
                T = len(iters)
                fixed_iters = np.array(iters, float)
                fixed_best = np.array([
                    best[i] if valid[i] else np.nan
                    for i in range(T)
                ], dtype=float)

                iters_list.append(fixed_iters)
                best_list.append(fixed_best)

        return iters_list, best_list

    # -------- main loop for SA / GA / BO --------
    for name, trends, color in [
        ("SA", SA_trends, "blue"),
        ("GA", GA_trends, "green"),
        ("BO", BO_trends, "red")
    ]:

        iters_list, best_list = extract_iters_and_best(trends, name)

        # pad globally
        I = pad(iters_list)
        B = pad(best_list)

        # compute statistics
        x = np.nanmean(I, axis=0)          # TRUE iteration axis
        mean_curve = np.nanmean(B, axis=0)
        std_curve = np.nanstd(B, axis=0)

        plt.figure(figsize=(7,4))
        for xx, yy in zip(iters_list, best_list):
            plt.scatter(xx, yy, color=color, alpha=0.15, s=15)

        plt.title(f"{name} –  Scatter Convergence ({len(SA_trends)} Trials)")
        plt.xlabel("Iteration")
        plt.ylabel("Objective J")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


        plt.figure(figsize=(7,4))
        # Std shading
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                            color=color, alpha=0.25, label="Std deviation band")

        # Mean curve
        plt.plot(x, mean_curve, color=color, linewidth=1, label="Mean best-so-far")

        plt.title(f"{name} – Mean ± Std Convergence ({scale_name} agents, Type {obstacles_name})") 
        plt.xlabel("Iteration")
        plt.ylabel("Objective J")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def extract_best(scores):
    # scores = list of tuples (C, S, E, time, order, paths)
    best_item = min(scores, key=lambda x: x[2])  # minimize E
    return {
        "C": best_item[0],
        "S": best_item[1],
        "E": best_item[2],
        "time": best_item[3],
        "order": best_item[4],
        "paths": best_item[5]
    }

def plot_random_distribution_valid_only(players, goals, power, env,iterations=40, trials=30, env_name="Environment",invalid_threshold=9999):

    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12
    
    all_valid_E = []      # valid objective values per trial
    invalid_counts = []   # number of invalid samples per trial

    for t in range(trials):
        valid_vals = []
        invalid_ct = 0

        for _ in range(iterations):
            order = tuple(np.random.permutation(len(players)))
            C, S, W, G, B, E, paths, valid = simulate_order_once(
                order, players, goals, power, env.copy()
            )

            if not valid or E >= invalid_threshold:
                invalid_ct += 1
                continue

            valid_vals.append(E)

        all_valid_E.append(valid_vals)
        invalid_counts.append(invalid_ct)

    plt.figure(figsize=(7, 5))

    for trial_id, vals in enumerate(all_valid_E):
        x = [trial_id + 1] * len(vals)     # shift axis to start at 1
        plt.scatter(x, vals, s=12, alpha=0.5)

    plt.xlabel("Trial Index")
    plt.ylabel("Valid Objective J")
    plt.title(f"Random Sample Distribution - Valid Only {env_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return all_valid_E, invalid_counts

def plot_random_valid_invalid_histogram(valid_values, invalid_counts, env_name="Environment"):
    """
    Histogram per trial:
       - valid sample count
       - invalid sample count
    """
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 14
    trials = len(invalid_counts)
    valid_counts = [len(v) for v in valid_values]

    x = np.arange(1, trials + 1)

    plt.figure(figsize=(9, 5))

    # Stacked bar: valid at bottom, invalid on top
    plt.bar(x, valid_counts, label="Valid", color="green", alpha=0.7)
    plt.bar(x, invalid_counts, bottom=valid_counts,
            label="Invalid", color="red", alpha=0.7)

    plt.xlabel("Trial Index")
    plt.ylabel("Sample Count")
    plt.title(f"Valid vs Invalid Random Samples {env_name}")
    xticks = np.arange(1, trials + 1, 5)
    plt.xticks(xticks)
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

def plot_environment_clean(s, fonts, size, font, rows, cols, starts, goals, obstacles, title):
    fig, ax = plt.subplots(figsize=size)

    # ======== GRID WORLD FIXED ========
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)

    ax.set_xticks(np.arange(0, cols+1, 1))
    ax.set_yticks(np.arange(0, rows+1, 1))
    ax.set_xticklabels(np.arange(0, cols+1, 1))
    ax.set_yticklabels(np.arange(0, rows+1, 1))

    ax.grid(True, color='gray', linewidth=0.8)

    # ======== OBSTACLES (centered squares) ========
    for (r, c) in obstacles:
        ax.add_patch(
            plt.Rectangle((c, r), 1, 1, color='gray', alpha=0.7)
        )

    # ======== LABEL FUNCTION (center of cell) ========
    def label_cell(r, c, text, color, dx=0, dy=0):
        ax.text(c + 0.5 + dx, r + 0.5 + dy, text,
                color=color, ha='center', va='center',
                fontsize=font) 

    # ======== START LABELS (shift left slightly) ========
    for i, (r, c) in enumerate(starts):
        label_cell(r, c, f"S{i+1}", 'blue', dx=-0.25)

    # ======== GOAL LABELS (shift right slightly) ========
    for i, (r, c) in enumerate(goals):
        label_cell(r, c, f"G{i+1}", 'red', dx=+0.25)

    # ======== LEGEND ========
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='gray', markersize=s, label='Obstacle'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='blue', markersize=s, label='Start'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='red', markersize=s, label='Goal')
    ]
    ax.legend(handles=legend_elements,
              loc='upper center', ncol=3, fontsize=9, frameon=False)

    # ======== FINAL TOUCHES ========
    ax.invert_yaxis()  
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=fonts, pad=15)
    plt.tight_layout()
    plt.show()

