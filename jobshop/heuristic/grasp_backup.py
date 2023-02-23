import numpy as np
from jobshop.heuristic.operations import Graph
from jobshop.params import JobShopParams
from jobshop.heuristic.construction import semi_greedy_makespan
from jobshop.heuristic.evaluation import calc_tails, calc_makespan
from jobshop.heuristic.local_search import get_critical, local_search


class GraspSolution:
    
    def __init__(self, value, graph: Graph, sol_values: list):
        self.value = value
        self.graph = graph
        self.sol_values = sol_values


def simple_grasp(params: JobShopParams, maxiter=1000, alpha=0.8, seed=None):
    np.random.seed(seed)
    best_obj = np.inf
    best_sol = None
    sol_values = []
    
    for _ in range(maxiter):
        
        graph = Graph(params.machines, params.jobs, params.p_times, params.seq)
        semi_greedy_makespan(graph, alpha=alpha)
        calc_makespan(graph)
        calc_tails(graph)
        get_critical(graph)
        graph = local_search(graph)
        
        sol_value = graph.C
        sol_values.append(sol_value)
        
        if sol_value <= best_obj:
            best_obj = sol_value
            best_sol = graph
    
    return GraspSolution(best_obj, best_sol, sol_values)
