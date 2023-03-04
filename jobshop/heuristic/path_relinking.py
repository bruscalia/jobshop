import numpy as np
from jobshop.heuristic.operations import Graph
from jobshop.params import JobSequence
from jobshop.heuristic.evaluation import calc_makespan


class PathRelinking:
    
    def __init__(self, seed=None) -> None:
        self.visited_paths = {}
        self.rng = np.random.default_rng(seed)
    
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
            S_gmin = self._path_relinking(S, T)
            self.visited_paths[sig_s, sig_t, min_delta] = S_gmin
            return S_gmin
    
    def _path_relinking(self, S: Graph, T: Graph, min_delta=2):
        
        # Initialize values
        c_gmin = S.C
        S_gmin = S.copy()
        delta_sol = self.get_delta_solutions(S, T)
        iter_count = 0
        total_lenght = sum(len(delta_machine) for delta_machine in delta_sol.values())
        max_iter = total_lenght * 10
        
        # Do path
        while total_lenght >= min_delta and iter_count <= max_iter:
            
            # Initialize values of iteration
            c_min = float("inf")
            
            # Iterate over machines
            for m in self.rng.permutation(S.machines):
                
                # Iterate over swaps of machine
                for (i, j) in self.rng.permutation(delta_sol[m]):
                    S_alt = S.copy()
                    S_alt.M[m].jobs.swap(i, j)
                    c_alt = calc_makespan(S_alt)
                    
                    # If better than previous update
                    if c_alt <= c_min:
                        c_min = c_alt
                        S_min = S_alt
                        # best_swap = (i, j) - Currently ignored
                        m_min = m
            
            # Update after move
            S = S_min
            makespan = c_min
            delta_sol[m_min] = get_delta_machine(S.M[m_min].jobs, T.M[m_min].jobs)
            total_lenght = sum(len(delta_machine) for delta_machine in delta_sol.values())
            
            # Update global best
            if makespan <= c_gmin:
                c_gmin = makespan
                S_gmin = S.copy()
            
            # Update iterations
            iter_count = iter_count + 1
        
        return S_gmin


def get_delta_solutions(S: Graph, T: Graph):
    delta_set = {}
    for m in S.machines:
        delta_set[m] = get_delta_machine(S.M[m].jobs, T.M[m].jobs)
    return delta_set


def get_delta_machine(s: JobSequence, t: JobSequence):
    delta_set = []
    for k, j in enumerate(s):
        i = t[k]
        if (i != j) and (j, i) not in delta_set:
            delta_set.append((i, j))
    return delta_set


def get_delta_module(S: Graph, T: Graph):
    delta_sol = np.sum(S.pheno != T.pheno)
    return delta_sol


