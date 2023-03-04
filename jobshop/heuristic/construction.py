import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from jobshop.heuristic.operations import Graph, Operation
from jobshop.heuristic.evaluation import calc_makespan


class SemiGreedyBase(ABC):
    
    def __init__(self, alpha=(0.0, 1.0)):
        """Semi-greedy construction heuristics

        Parameters
        ----------
        alpha : tuple | float, optional
            Greediness factor, by default (0.0, 1.0)
        """
        self.alpha = alpha
    
        # Set to random uniform generation if iterables
        if hasattr(alpha, "__iter__"):
            self.get_alpha = lambda: np.random.uniform(alpha[0], alpha[-1])
        else:
            self.get_alpha = lambda: self.alpha
    
    def __call__(self, graph, alpha=None) -> Graph:
        if alpha is not None:
            self.reset_alpha(alpha=alpha)
        return self.do(graph)
    
    @staticmethod
    def get_unpreceding_ops(graph: Graph):
        Q = []
        for j, machines in graph.seq.items():
            Q.append((machines[0], j))
        return Q
    
    @staticmethod
    def clear_unpreceding_ops(graph: Graph):
        Q = SemiGreedyBase.get_unpreceding_ops(graph)
        for m, j in Q:
            graph.O[m, j].release = 0.0
        return Q

    @abstractmethod
    def do(self, graph):
        pass
    
    def reset_alpha(self, alpha=(0.0, 1.0)):
        self.alpha = alpha
        if hasattr(alpha, "__iter__"):
            self.get_alpha = lambda: np.random.uniform(alpha[0], alpha[-1])
        else:
            self.get_alpha = lambda: self.alpha
    
    @staticmethod
    def get_preceding_machine(graph: Graph, m: Any, j: Any):
        prev_jobs = graph.M[m].jobs
        if len(prev_jobs) >= 1:
            last_job = graph.M[m].jobs[-1]
            return graph.O[m, last_job]
        else:
            return None
    
    @staticmethod
    def get_earliest_release(PJ: Operation, PM: Operation):
        r = 0.0
        if PJ is not None:
            r = max(r, PJ.release + PJ.duration)
        if PM is not None:
            r = max(r, PM.release + PM.duration)
        return r
    
    @staticmethod
    def get_rcl(H: dict, alpha):
        # Initialize restricted candidate list
        RCL = []
        min_sol = min(H.values())
        max_sol = max(H.values())
        atol = min_sol + (1 - alpha) * (max_sol - min_sol)

        # Iterate over candidates
        for q, C_pot in H.items():
            if C_pot <= atol:
                RCL.append(q)
            else:
                pass
        
        return RCL


class SemiGreedyMakespan(SemiGreedyBase):
    
    def do(self, graph: Graph):
        
        # Get initial candidates
        Q = self.clear_unpreceding_ops(graph)
        L = set()
        C = 0
        alpha = self.get_alpha()
        k = 0
        max_iter = len(graph.jobs) * len(graph.machines) + 1

        # Iterate if there's additional jobs to include
        while len(Q) >= 1 and k <= max_iter:
            
            k = k + 1

            # Start
            H = {}
            R = {}

            # Iterate over feasible solutions
            for q in Q:
                
                # Add new item to labeled
                m, j = q
                d = graph.O[m, j].duration
                
                # Get preceding
                PJ = graph.precede_job(m, j)
                PM = self.get_preceding_machine(graph, m, j)
                
                # Update earliest release
                r = self.get_earliest_release(PJ, PM)
                R[q] = r
                
                # Update C_pot
                C_pot = max(r + d, C)
                H[q] = C_pot
                
            # Get restricted candidate list
            RCL = self.get_rcl(H, alpha)
                
            # Add random element from RCL
            idx = np.random.choice(len(RCL))
            new_element = RCL[idx]
            m, j = new_element
            
            graph.O[m, j].release = R[m, j]
            graph.M[m].add_job(j)
            Q.remove((m, j))
            L.add((m, j))
            
            # Update C
            C = max(C, H[m, j])
            
            # Get next machine of job (now feasible)
            SJ = graph.follow_job(m, j)
            if SJ is not None:
                Q.append(SJ.code)
        
        if len(L) == len(graph.O):
            graph.C = C
        else:
            graph.C = graph.V * (1.0 + 1e-2)
        
        return graph


class SemiGreedyTR(SemiGreedyBase):
    
    def do(self, graph: Graph):
        
        # Get initial candidates
        Q = self.clear_unpreceding_ops(graph)
        L = set()
        alpha = self.get_alpha()
        k = 0
        max_iter = len(graph.jobs) * len(graph.machines) + 1
        time_remaining = {job: get_job_total_duration(job, graph) for job in graph.jobs}

        # Iterate if there's additional jobs to include
        while len(Q) >= 1 and k <= max_iter:
            
            k = k + 1

            # Start
            H = {}

            # Iterate over feasible solutions
            for q in Q:
                
                # Add new item to labeled
                m, j = q
                H[q] = - (time_remaining[j] - graph.O[m, j].duration)
                
            # Get restricted candidate list
            RCL = self.get_rcl(H, alpha)
                
            # Add random element
            idx = np.random.choice(len(RCL))
            new_element = RCL[idx]
            m, j = new_element
            graph.M[m].add_job(j)
            time_remaining[j] = time_remaining[j] - graph.O[m, j].duration
            Q.remove((m, j))
            L.add((m, j))
            
            # Get next machine of job (now feasible)
            SJ = graph.follow_job(m, j)
            if SJ is not None:
                Q.append(SJ.code)

        # Obtain makespan
        graph.C = calc_makespan(graph)
        return graph


def get_job_total_duration(job, graph: Graph):
    total = 0
    for m in graph.seq[job]:
        total = total + graph.p_times[m, job]
    return total


class SemiGreedyAlternate(SemiGreedyBase):
    
    def __init__(self, alpha=(0, 1)):
        self.makespan = SemiGreedyMakespan(alpha)
        self.time_remaining = SemiGreedyTR(alpha)
        self.pair = True
    
    def do(self, graph):
        pair = self.pair
        self.pair = not pair
        if pair:
            return self.makespan(graph)
        else:
            return self.time_remaining(graph)
    
    def reset_alpha(self, alpha=(0, 1)):
        self.makespan.reset_alpha(alpha)
        self.time_remaining.reset_alpha(alpha)