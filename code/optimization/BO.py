import torch
import gpytorch
import numpy as np
from gpytorch.models import ExactGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from operator import neighbor
from planner import simulate_order_once


def kendall_tau_distance(p, q):
    """Efficient Kendall tau distance."""
    index = {v: i for i, v in enumerate(p)}
    mapped = [index[v] for v in q]

    inv = 0
    BIT = [0] * (len(p) + 1)

    def update(i):
        while i < len(BIT):
            BIT[i] += 1
            i += i & -i

    def query(i):
        s = 0
        while i > 0:
            s += BIT[i]
            i -= i & -i
        return s

    for i, x in enumerate(mapped):
        inv += i - query(x + 1)
        update(x + 1)

    return inv

class KendallKernel(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(self, lam=0.1):
        super().__init__()
        self.lam = lam

    def forward(self, X1, X2, diag=False, **params):
        # X1, X2: integer tensors [n1, d], [n2, d]
        X1_np = X1.cpu().numpy()
        X2_np = X2.cpu().numpy()

        n1, n2 = len(X1_np), len(X2_np)
        device = X1.device
        K = torch.zeros(n1, n2, dtype=torch.double, device=device)
        if n1 == n2:
            K = K + 1e-6 * torch.eye(n1, dtype=torch.double, device=device)

        for i in range(n1):
            for j in range(n2):
                d = kendall_tau_distance(tuple(X1_np[i]), tuple(X2_np[j]))
                K[i, j] = torch.exp(torch.tensor(-self.lam * d, dtype=torch.double, device=device))
        return K

class PermutationGP(ExactGP):
    def __init__(self, X, y, likelihood):
        super().__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = KendallKernel(lam=0.1)
    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)
 
def BO_permutation_optimize(players, goals, power, env, eval_budget=80):

    import time
    t_start = time.time()

    N = len(players)

    # storage
    X_list = []
    Y_list = []
    cost_history = []
    best_history = []
    iter_history = []
    valid_flags = []
    # function wrapper that returns FULL metrics
    def eval_order(order):
        C, S, W, G, B, E, paths, valid = simulate_order_once(
            order, players, goals, power, env.copy()
        )
        return (E, C, S, W, G, B, paths, valid)

    # -------------------------
    # Initial random evaluations
    # -------------------------
    init_samples = max(N, 5)

    for it in range(init_samples):
        o = tuple(np.random.permutation(N))
        E, C, S, W, G, B, paths, valid = eval_order(o)

        X_list.append(o)
        Y_list.append(E)
        cost_history.append(E)
        iter_history.append(it)

        if len(best_history) == 0 or E < best_history[-1]:
            best_history.append(E)
            best_order = o
            best_metrics = (C, S, W, G, B, E)
            best_paths = paths
            flag = valid 
            valid_flags.append(flag)
        else:
            best_history.append(best_history[-1])
            valid_flags.append(valid_flags[-1])
    # prepare GP training tensors
    X_train = torch.tensor([list(x) for x in X_list], dtype=torch.long)
    Y_train = torch.tensor(Y_list, dtype=torch.double)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = PermutationGP(X_train, Y_train, likelihood)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll)

    eval_count = init_samples

    # -------------------------
    # BO optimization loop
    # -------------------------
    while eval_count < eval_budget:

        # neighborhood proposals
        neighs = [neighbor(best_order) for _ in range(30)]
        X_local = torch.tensor([list(o) for o in neighs], dtype=torch.long)

        model.eval()
        likelihood.eval()

        posterior = likelihood(model(X_local))
        samples = posterior.rsample()

        cand = neighs[int(torch.argmin(samples))]

        # evaluate candidate
        E, C, S, W, G, B, paths, valid = eval_order(cand)
        eval_count += 1

        cost_history.append(E)
        iter_history.append(eval_count)
        
        # update global best
        if E < best_metrics[-1]:  # compare with best E
            best_order = cand
            best_metrics = (C, S, W, G, B, E)
            best_paths = paths
            flag = valid
        
        valid_flags.append(flag)
        best_history.append(best_metrics[-1])

        # update GP training set
        X_train = torch.cat([X_train, torch.tensor([list(cand)], dtype=torch.long)], dim=0)
        Y_train = torch.cat([Y_train, torch.tensor([E], dtype=torch.double)], dim=0)

        model = PermutationGP(X_train, Y_train, likelihood)
        mll = ExactMarginalLogLikelihood(likelihood, model)
        fit_gpytorch_mll(mll)

    total_time = time.time() - t_start

    return best_order, best_metrics, best_paths, iter_history, cost_history, best_history, valid_flags, total_time
