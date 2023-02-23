import numpy as np
from jobshop.heuristic.operations import Graph
from jobshop.heuristic.evaluation import calc_makespan


class PathRelinking:
    
    def __init__(self) -> None:
        self.visited_paths = {}
    
    @staticmethod
    def get_delta_solutions(S: Graph, T: Graph):
        return get_delta_solutions(S, T)
    
    @staticmethod
    def get_delta_module(S: Graph, T: Graph):
        return get_delta_module(S, T)
    
    def __call__(self, S: Graph, T: Graph, min_delta=2):
        sig_s = S.signature
        sig_t = T.signature
        if (sig_s, sig_t, min_delta) in self.visited_paths:
            return self.visited_paths[sig_s, sig_t, min_delta].copy()
        else:
            S_gmin = path_relinking(S, T)
            self.visited_paths[sig_s, sig_t, min_delta] = S_gmin
            return S_gmin


def get_delta_solutions(S: Graph, T: Graph):
    delta_set = {}
    for m in S.machines:
        delta_set[m] = []
        for k, j in enumerate(S.M[m].jobs):
            i = T.M[m].jobs[k]
            if i != j:
                if (j, i) not in delta_set[m]:
                    delta_set[m].append((i, j))
    return delta_set


def get_delta_module(S: Graph, T: Graph):
    delta_sol = get_delta_solutions(S, T)
    return sum(len(delta_machine) for delta_machine in delta_sol.values())


def path_relinking(S: Graph, T: Graph, min_delta=2):
    
    # Initialize values
    c_gmin = S.C
    S_gmin = S.copy()
    delta_sol = get_delta_solutions(S, T)
    iter_count = 0
    total_lenght = sum(len(delta_machine) for delta_machine in delta_sol.values())
    max_iter = total_lenght * 100
    
    # Do path
    while total_lenght >= min_delta and iter_count <= max_iter:
        
        # Initialize values of iteration
        c_min = float("inf")
        
        # Iterate over machines
        for m in S.machines:
            
            # Iterate over swaps of machine
            for (i, j) in delta_sol[m]:
                S_alt = S.copy()
                S_alt.M[m].jobs.swap(i, j)
                c_alt = calc_makespan(S_alt)
                
                # If strictly better than the current path solution
                if c_alt < S_gmin.C:
                    c_min = c_alt
                    S_min = S_alt
                    best_swap = (i, j)
                    m_min = m
                
                # If better than previous update
                elif c_alt <= c_min:
                    c_min = c_alt
                    S_min = S_alt
                    best_swap = (i, j)
                    m_min = m
        
        # Update after move
        S = S_min
        makespan = c_min
        delta_sol[m_min].remove(best_swap)
        total_lenght = sum(len(delta_machine) for delta_machine in delta_sol.values())
        
        # Update global best
        if makespan <= c_gmin:
            c_gmin = makespan
            S_gmin = S
         
        # Update iterations
        iter_count = iter_count + 1
    
    return S_gmin