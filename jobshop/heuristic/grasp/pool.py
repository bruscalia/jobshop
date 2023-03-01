import numpy as np
from jobshop.heuristic.operations import Graph
from jobshop.heuristic.path_relinking import get_delta_module


class Pool:
    
    def __init__(self, *solutions, min_delta=0) -> None:
        self.P = np.array(solutions)
        self.C = np.array([S.C for S in self.P])
        self.min_delta = min_delta
    
    def __repr__(self) -> str:
        return str(self.C)
    
    def update(self, S: Graph, verbose=False) -> None:
        if S.C < self.C[0]:
            self._update_best(S, verbose=verbose)
        elif S.C < self.C[-1]:
            self._update_quality(S, verbose=verbose)
        else:
            return
    
    def append(self, S: Graph, verbose=False):
        self.P = np.append(self.P, S)
        self.C = np.append(self.C, S.C)
        new_sort = np.argsort(self.C)
        self.C = self.C[new_sort]
        self.P = self.P[new_sort]
        if verbose:
            print(f"New solution: {S.C}")
            print(f"Updated Pool: {self}")
    
    def _update_best(self, S: Graph, verbose=False):
        self.P[-1] = S
        self.C[-1] = S.C
        new_sort = np.argsort(self.C)
        self.C = self.C[new_sort]
        self.P = self.P[new_sort]
        if verbose:
            print(f"New best solution: {S.C}")
            print(f"Updated Pool: {self}")
    
    def _update_quality(self, S: Graph, verbose=False):
        for Sp in self.P:
            delta_mod = get_delta_module(S, Sp)
            if delta_mod < self.min_delta:
                return
            else:
                continue
        self.P[-1] = S
        self.C[-1] = S.C
        new_sort = np.argsort(self.C)
        self.C = self.C[new_sort]
        self.P = self.P[new_sort]
        if verbose:
            print(f"New quality solution: {S.C}")
            print(f"Updated Pool: {self}")