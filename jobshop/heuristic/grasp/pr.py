import numpy as np
from math import ceil
from itertools import combinations
from jobshop.heuristic.operations import Graph
from jobshop.params import JobShopParams
from jobshop.heuristic.construction import semi_greedy_makespan, semi_greedy_time_remaining
from jobshop.heuristic.evaluation import calc_makespan, calc_tails
from jobshop.heuristic.local_search import get_critical, local_search
from jobshop.heuristic.path_relinking import PathRelinking, get_delta_module, get_delta_solutions


def update_pool(S: Graph, P: np.array, C_pool=np.array, min_delta=2, verbose=False):
    if S.C < C_pool[0]:
        P[-1] = S
        C_pool[-1] = S.C
        new_sort = np.argsort(C_pool)
        C_pool = C_pool[new_sort]
        P = P[new_sort]
        if verbose:
            print(f"New best solution: {S.C}")
            print(f"Updated Pool: {C_pool}")
        return P, C_pool
    elif S.C <= C_pool[-1]:
        for Sp in P:
            delta_mod = get_delta_module(S, Sp)
            if delta_mod < min_delta:
                return P, C_pool
            else:
                continue
        P[-1] = S
        C_pool[-1] = S.C
        new_sort = np.argsort(C_pool)
        C_pool = C_pool[new_sort]
        P = P[new_sort]
        if verbose:
            print(f"New quality solution: {S.C}")
            print(f"Updated Pool: {C_pool}")
        return P, C_pool
    else:
        return P, C_pool


def append_to_pool(S: Graph, P: np.array, C_pool=np.array, min_delta=2, verbose=False):
    if len(P) == 0:
        P = np.append(P, S)
        C_pool = np.append(C_pool, S.C)
        if verbose:
            print(f"First solution: {S.C}")
        return P, C_pool
    elif S.C < C_pool[0]:
        P = np.append(P, S)
        C_pool = np.append(C_pool, S.C)
        new_sort = np.argsort(C_pool)
        C_pool = C_pool[new_sort]
        P = P[new_sort]
        if verbose:
            print(f"New best solution: {S.C}")
            print(f"Updated Pool: {C_pool}")
        return P, C_pool
    elif S.C <= C_pool[-1]:
        for Sp in P:
            delta_mod = get_delta_module(S, Sp)
            if delta_mod < min_delta:
                return P, C_pool
            else:
                continue
        P = np.append(P, S)
        C_pool = np.append(C_pool, S.C)
        new_sort = np.argsort(C_pool)
        C_pool = C_pool[new_sort]
        P = P[new_sort]
        if verbose:
            print(f"New quality solution: {S.C}")
            print(f"Updated Pool: {C_pool}")
        return P, C_pool
    else:
        P = np.append(P, S)
        C_pool = np.append(C_pool, S.C)
        return P, C_pool


def intensificaton(P, C_pool, path_relinking=None, min_delta=2, verbose=False):
    """Do intensification routine in which PR is performed in every pair of solutions in a pool (including new)

    Parameters
    ----------
    P : np.array
        Pool of solutions
    
    C_pool : np.array
        Cost of solutions
    
    min_delta : int, optional
        Minimum difference of solutions to update pool, by default 2
    
    verbose : bool, optional
        Either or not to print messages, by default False

    Returns
    -------
    P : np.array
        Pool of solutions (Graph instances)
    
    C_pool : np.array
        Makespan of solutions
    """
    # Create a set of elite solutions not yet evaluated
    Q = set(P.copy())
    
    # Instantiate path relinking if None
    if path_relinking is None:
        path_relinking = PathRelinking()
    
    # Do path relinking from S to every other in Pool
    while len(Q) > 0:
        S = Q.pop()
        for T in P:
            S_gmin = path_relinking(S, T, min_delta=2)
            P, C_pool = update_pool(S_gmin, P, C_pool, min_delta=min_delta, verbose=verbose)
            # S_gmin as a copy is considered different from S
            if S_gmin in P:
                if verbose:
                    print(f"New solution to Q: {S_gmin.C}")
                Q.add(S_gmin)
    return P, C_pool


def grasp_pool(
    params: JobShopParams, maxiter=1000, alpha=(0.0, 1.0), maxpool=10, min_diff=0.25,
    verbose=False, seed=None,
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
    
    maxpool : int, optional
        Number of solutions in elite pool, by default 10
    
    min_diff : float, optional
        Variation factor to include a new solution in the pool, by default 2
    
    verbose : bool, optional
        Either or not to print messages while the algorithm runs, by default False
    
    seed : int | None, optional
        Random seed, by default None

    Returns
    -------
    P : np.array
        Pool of solutions (Graph instances)
    
    C_pool : np.array
        Makespan of solutions
    """
    # Evaluate alpha
    if hasattr(alpha, "__iter__"):
        get_alpha = lambda: np.random.uniform(alpha[0], alpha[-1])
    else:
        get_alpha = lambda: alpha
    
    # Obtain min delta from params
    min_delta = ceil(min_diff * len(params.machines) * len(params.jobs))
    
    # Initialize seed and solutions pool
    np.random.seed(seed)
    P = np.array([])
    C_pool = np.array([])
    
    for i in range(maxiter):
        
        # Initialize a solution S by GRASP
        S = Graph(params.machines, params.jobs, params.p_times, params.seq)
        semi_greedy_makespan(S, alpha=get_alpha())
        calc_makespan(S)
        calc_tails(S)
        get_critical(S)
        S = local_search(S)
        
        # If pool is full update, else append
        if len(P) == maxpool:
            P, C_pool = update_pool(S, P, C_pool, min_delta=min_delta, verbose=verbose)
        else:
            P, C_pool = append_to_pool(S, P, C_pool, min_delta=min_delta, verbose=verbose)
    
    # Sort by C_pool
    new_sort = np.argsort(C_pool)
    C_pool = C_pool[new_sort]
    P = P[new_sort]
    
    return P, C_pool


def grasp_pr(
    params: JobShopParams, maxiter=500, init_iter=0.5, alpha=(0.0, 1.0),
    maxpool=10, ifreq=100, min_diff=0.25, post_opt=False, mixed_construction=True,
    verbose=False, seed=None,
):
    """Perform GRASP with Path Relinking in the job-shop scheduling problem

    Parameters
    ----------
    params : JobShopParams
        Problem parameters
    
    maxiter : int, optional
        Number of iterations, by default 1000
    
    init_iter : float, optional
        Fraction of iterations used to consider acceptance criterion from pool, by default 100
    
    alpha : float | tuple, optional
        Greediness parameter defined in the range (0, 1) in which 0 is random and 1 is greedy.
        If a tuple is passed a random uniform generator is used. By default (0.0, 1.0)
    
    maxpool : int, optional
        Number of solutions in elite pool, by default 10
    
    ifreq : int, optional
        Frequence of intensification strategy, by default 10
    
    min_diff : float, optional
        Variation factor to include a new solution in the pool, by default 2
    
    post_opt : bool, optional
        Either or not to do post optimation by pairwise PR between the elite pool members, by default False
    
    mixed_construction : bool, optional
        Either or not greedy makespan and time-remaining should be alternated
        in the GRASP construction phase, by default False
    
    verbose : bool, optional
        Either or not to print messages while the algorithm runs, by default False
    
    seed : int | None, optional
        Random seed, by default None

    Returns
    -------
    P : np.array
        Pool of solutions (Graph instances)
    
    C_pool : np.array
        Makespan of solutions
    """
    # Evaluate alpha
    if hasattr(alpha, "__iter__"):
        get_alpha = lambda: np.random.uniform(alpha[0], alpha[-1])
    else:
        get_alpha = lambda: alpha

    # Obtain min delta from params
    min_delta = ceil(min_diff * len(params.machines) * len(params.jobs))
    
    # Instantiate path_relinking that stores visited paths
    path_relinking = PathRelinking()
    
    # Obtain init_iter if float
    n_init = ceil(maxiter * init_iter)
    
    # Initialize seed and solutions pool
    np.random.seed(seed)
    P = np.array([])
    C_pool = np.array([])
    grasp_solutions = []
    last_int_pool = P.copy()
    
    for i in range(maxiter):
        
        # Initialize a solution S by GRASP
        S = Graph(params.machines, params.jobs, params.p_times, params.seq)
        if (i % 2 == 0) or (not mixed_construction):
            semi_greedy_makespan(S, alpha=get_alpha())
        else:
            semi_greedy_time_remaining(S, alpha=get_alpha())
        calc_makespan(S)
        calc_tails(S)
        get_critical(S)
        S = local_search(S)
        grasp_solutions.append(S.C)
        last_half = maxpool // 2
        
        # If pool is full
        if len(P) == maxpool:
            if i <= n_init:
                accepted = S.C <= C_pool[-1]
            else:
                lim = max(C_pool[-1], np.mean(grasp_solutions) - 2 * np.std(grasp_solutions))
                accepted = S.C <= lim
            if accepted:
                for T in P:
                    S_gmin = path_relinking(S, T, min_delta=2)
                    P, C_pool = update_pool(S_gmin, P, C_pool, min_delta=min_delta, verbose=verbose)
                    S_gmin = path_relinking(T, S, min_delta=2)
                    P, C_pool = update_pool(S_gmin, P, C_pool, min_delta=min_delta, verbose=verbose)
                    
        # Else if pool is not yet full, add new solution
        else:
            P, C_pool = append_to_pool(S, P, C_pool, min_delta=min_delta, verbose=verbose)
        
        # Do intensification in random pair of solutions
        if i % ifreq == 0:
            if verbose:
                print("Starting intensification")
            P, C_pool = intensificaton(
                P, C_pool, path_relinking=path_relinking,
                min_delta=min_delta, verbose=verbose,
            )
            if verbose:
                print("Finished intensification")
            
            # Assign half the pool to inf
            if np.array_equiv(last_int_pool, P):
                if verbose:
                    print("Assign half the pool inf values")
                C_pool[-last_half:] = float("inf")
            
            # Update last pool
            last_int_pool = P.copy()
                
    
    # Post optimization
    if post_opt:
        if verbose:
                print("Post optimization")
        P, C_pool = intensificaton(
            P, C_pool, path_relinking=path_relinking,
            min_delta=min_delta, verbose=verbose,
        )
    
    return P


def grasp_pr_alt(
    params: JobShopParams, init_iter=1000, main_iter=100, alpha=(0.0, 1.0),
    maxpool=10, ifreq=10, min_diff=0.2, acceptance_quantile=0.05, post_opt=False,
    verbose=False, seed=None,
):
    """Perform GRASP with Path Relinking in the job-shop scheduling problem and alternative acceptance criterion

    Parameters
    ----------
    params : JobShopParams
        Problem parameters
    
    init_iter : int, optional
        Number of iterations in initial GRASP, by default 1000
    
    main_iter : int, optional
        Number of iterations in main loop (including PR), by default 100
    
    alpha : float | tuple, optional
        Greediness parameter defined in the range (0, 1) in which 0 is random and 1 is greedy.
        If a tuple is passed a random uniform generator is used. By default (0.0, 1.0)
    
    maxpool : int, optional
        Number of solutions in elite pool, by default 10
    
    ifreq : int, optional
        Frequence of intensification strategy, by default 10
    
    min_diff : float, optional
        Variation factor to include a new solution in the pool, by default 2
    
    acceptance_quantile : float, optional
        Quantile that must be outperformed for a solution to be considered in PR, by default 0.05
    
    post_opt : bool, optional
        Either or not to do post optimation by pairwise PR between the elite pool members, by default False
    
    verbose : bool, optional
        Either or not to print messages while the algorithm runs, by default False
    
    seed : int | None, optional
        Random seed, by default None

    Returns
    -------
    P : np.array
        Pool of solutions (Graph instances)
    
    C_pool : np.array
        Makespan of solutions
    """
    # Evaluate alpha
    if hasattr(alpha, "__iter__"):
        get_alpha = lambda: np.random.uniform(alpha[0], alpha[-1])
    else:
        get_alpha = lambda: alpha
    
    # Obtain min delta from params
    min_delta = ceil(min_diff * len(params.machines) * len(params.jobs))
    
    # Initialize seed and solutions pool
    np.random.seed(seed)
    P, C_pool = grasp_pool(
        params, maxiter=init_iter, alpha=alpha, min_delta=min_delta,
        verbose=False, seed=seed,
    )
    all_makespans = C_pool.copy()
    if len(P) > maxpool:
        P = P[:maxpool]
        C_pool = C_pool[:maxpool]
    
    if verbose:
        print(f"Initial pool {C_pool}")
    
    for i in range(main_iter):
        
        # Initialize a solution S by GRASP
        S = Graph(params.machines, params.jobs, params.p_times, params.seq)
        semi_greedy_makespan(S, alpha=get_alpha())
        calc_makespan(S)
        calc_tails(S)
        get_critical(S)
        S = local_search(S)
        all_makespans = np.append(all_makespans, S.C)
        
        # If pool is full
        if len(P) == maxpool:
            accepted = S.C <= np.quantile(all_makespans, acceptance_quantile)
            if accepted:
                for p_idx in range(maxpool):
                    T = P[p_idx]
                    S_gmin = path_relinking(S, T, min_delta=2)
                    P, C_pool = update_pool(S_gmin, P, C_pool, min_delta=min_delta, verbose=verbose)
                    S_gmin = path_relinking(T, S, min_delta=2)
                    P, C_pool = update_pool(S_gmin, P, C_pool, min_delta=min_delta, verbose=verbose)
            
            elif S.C <= C_pool[-1]:
                P, C_pool = update_pool(S, P, C_pool, min_delta=min_delta, verbose=verbose)
                    
        # Else if pool is not yet full, add new solution
        else:
            P, C_pool = append_to_pool(S, P, C_pool, min_delta=min_delta, verbose=verbose)
        
        # Do intensification in random pair of solutions
        if i % ifreq == 0:
            S = np.random.choice(P)
            T = np.random.choice(P)
            S_gmin = path_relinking(S, T, min_delta=2)
            P, C_pool = update_pool(S_gmin, P, C_pool, min_delta=min_delta, verbose=verbose)
            S_gmin = path_relinking(T, S, min_delta=2)
            P, C_pool = update_pool(S_gmin, P, C_pool, min_delta=min_delta, verbose=verbose)
    
    # Post optimization
    if post_opt:
        if verbose:
                print("Post optimization")
        for i, j in combinations(range(len(P)), 2):
            S = P[i]
            T = P[j]
            S_gmin = path_relinking(S, T, min_delta=2)
            P, C_pool = update_pool(S_gmin, P, C_pool, min_delta=min_delta, verbose=verbose)
            S_gmin = path_relinking(T, S, min_delta=2)
            P, C_pool = update_pool(S_gmin, P, C_pool, min_delta=min_delta, verbose=verbose)
    
    return P
