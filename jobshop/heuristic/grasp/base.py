import numpy as np
from jobshop.params import JobShopParams
from jobshop.heuristic.operations import Graph
from jobshop.heuristic.construction import SemiGreedyMakespan, SemiGreedyAlternate
from jobshop.heuristic.evaluation import calc_tails
from jobshop.heuristic.local_search import get_critical, local_search


LARGE_INT = 10000000000000000

class GRASP:
    
    def __init__(
        self,
        params: JobShopParams,
        alpha=(0.0, 1.0),
        mixed_construction=True,
        seed=None,
    ) -> None:
        """GRASP algorithm for the Job-shop scheduling problem

        Parameters
        ----------
        params : JobShopParams
            Problem parameters
        
        alpha : tuple | float, optional
            Greediness factor of constructive heuristics, by default (0.0, 1.0)
        
        mixed_construction : bool, optional
            Either of not to alternate between greedy construction by makespan (main)
            and greedy construction by time-remaining job (auxiliary), by default True
        
        seed : int | None, optional
            numpy random seed, by default None
        """
        
        # Base params
        self.params = params
        self.seed = seed
        
        # Construction operator
        if mixed_construction:
            self.construction = SemiGreedyAlternate(alpha)
        else:
            self.construction = SemiGreedyMakespan(alpha)
    
    def build_solution(self) -> Graph:
        """Build a graph structure solution from parameters

        Returns
        -------
        Graph
            Solution
        """
        S = Graph(self.params)
        S = self.construction(S)
        calc_tails(S)
        get_critical(S)
        S = local_search(S)
        return S
    
    def iter(self, maxiter=LARGE_INT) -> Graph:
        """Generator of solutions by semi-greedy construction and local search

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations, by default LARGE_INT

        Yields
        ------
        Graph
            Solution
        """
        k = 0
        while k < maxiter:
            k = k + 1
            yield self.build_solution()

    def __call__(self, maxiter=1000, target=None, verbose=False, seed=None) -> Graph:
        """Solve problem by generating maxiter solutions

        Parameters
        ----------
        maxiter : int, optional
            Number of iterations, by default 1000
        
        target : float | int | None, optional
            Target value of objective (stops iterations), by default None
        
        verbose : bool, optional
            Either or not to print messages while solving, by default False
        
        seed : int | None, optional
            numpy random seed, by default None

        Returns
        -------
        Graph
            Solution
        """
        
        # Random seed
        if seed is None:
            np.random.seed(self.seed)
        else:
            np.random.seed(seed)
        
        # Target
        if target is None:
            target = -float("inf")
        
        # Initialize
        S_best = None
        C_best = np.inf
        
        # Iterate using solution generator
        for S in self.iter(maxiter):
            
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
            
    def reset_alpha(self, alpha=(0.0, 1.0)):
        """Reset semi-greedy construction alpha

        Parameters
        ----------
        alpha : tuple | float, optional
            Greediness factor of constructive heuristics, by default (0.0, 1.0)
        """
        self.construction.reset_alpha(alpha)