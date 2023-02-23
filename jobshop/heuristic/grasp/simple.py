import numpy as np
from jobshop.heuristic.operations import Graph
from jobshop.params import JobShopParams
from jobshop.heuristic.construction import semi_greedy_makespan, semi_greedy_time_remaining
from jobshop.heuristic.evaluation import calc_makespan, calc_tails
from jobshop.heuristic.local_search import get_critical, local_search


def grasp(
    params: JobShopParams, maxiter=1000, alpha=(0.0, 1.0), target=None,
    mixed_construction=True, verbose=False, seed=None,
):
    """Initialize a Pool a solutions using basic GRASP with minimal diversity

    Parameters
    ----------
    params : JobShopParams
        Problem parameters
    
    maxiter : int, optional
        Number of iterations (construction + local search), by default 1000
    
    alpha : float | tuple, optional
        Greediness parameter defined in the range (0, 1) in which 0 is random and 1 is greedy.
        If a tuple is passed a random uniform generator is used. By default (0.0, 1.0)
    
    target : float | int | None, optional
        Taget that stops optimization, by default None
    
    mixed_construction : bool, optional
        Either or not greedy makespan and time-remaining should be alternated
        in the GRASP construction phase, by default True
    
    verbose : bool, optional
        Either or not to print messages while the algorithm runs, by default False
    
    seed : int | None, optional
        Random seed, by default None

    Returns
    -------
    S : Graph
        Best solution
    """
    # Evaluate alpha
    if hasattr(alpha, "__iter__"):
        get_alpha = lambda: np.random.uniform(alpha[0], alpha[-1])
    else:
        get_alpha = lambda: alpha
    
    # Initialize target if None
    if target is None:
        target = - float("inf")
    
    # Initialize seed and solutions pool
    np.random.seed(seed)
    S_best = None
    C_best = np.inf
    
    for i in range(maxiter):
        
        # Initialize a solution S
        S = Graph(params.machines, params.jobs, params.p_times, params.seq)
        if (i % 2 == 0) or (not mixed_construction):
            semi_greedy_makespan(S, alpha=get_alpha())
        else:
            semi_greedy_time_remaining(S, alpha=get_alpha())
        calc_makespan(S)
        calc_tails(S)
        get_critical(S)
        S = local_search(S)
        
        # Update if better than previous
        if S.C < C_best:
            S_best = S
            C_best = S.C
            if verbose:
                print(f"New best solution {C_best}")
        
        # Break if target
        if C_best <= target:
            break
    
    return S_best