import numpy as np
from jobshop.params import JobShopParams
from jobshop.heuristic.operations import Graph
from jobshop.heuristic.evaluation import calc_makespan, calc_tails
from jobshop.heuristic.local_search import get_critical, local_search


class Decoder(JobShopParams):
    
    def __init__(self, params):
        """Decoder for Genetic Algorithms applied to the job-shop schedule problem

        Parameters
        ----------
        params : JobShopParams
            Parameters that define the problem
        """
        super().__init__(params.machines, params.jobs, params.p_times, params.seq)
        _x = []
        for key, value in self.seq.items():
            _x.extend([key] * len(value))
        self.base_vector = np.array(_x)
        self.known_solutions = {}
    
    def decode(self, x):
        """From a string x obtains phenotype and objective function

        Parameters
        ----------
        x : numpy.array
            String of independent variabless

        Returns
        -------
        tuple
            phenotype, corrected genes, and objective value
        """
        
        # Get sorted values of base vector
        order = self.get_order(x)
        order_hash = hash(str(order))
        
        # Avoid re-calculation if order is known
        if order_hash in self.known_solutions:
            pheno, C = self.known_solutions[order_hash]
            
        else:
            graph = self.build_graph(order)
            C = graph.C
            pheno = graph.pheno
            self.known_solutions[order_hash] = pheno, C
        
        return pheno, x, C
    
    def build_graph(self, order):
        """Build and evaluate problem graph from ordered operations

        Parameters
        ----------
        order : numpy.array
            Order of operations

        Returns
        -------
        Graph
            Job-shop graph
        """
        
        # Count how many times each job was assigned
        assigned = {
            key: 0
            for key in self.jobs
        }
        
        # Create a list of elements (m, j) to assign
        Q = []
        for j in order:
            k = assigned[j]
            m = self.seq[j][k]
            Q.append((m, j))
            assigned[j] = assigned[j] + 1
        
        # Initialize graph
        graph = Graph(self)
        for (m, j) in Q:
            graph.M[m].add_job(j)
        
        # Calculate makespan
        calc_makespan(graph)
        
        return graph
    
    def get_order(self, x):
        idx = np.argsort(x)
        order = self.base_vector[idx]
        return order
    
    def build_graph_from_string(self, x):
        """Build and evaluate problem graph from string

        Parameters
        ----------
        x : numpy.array
            String of solution

        Returns
        -------
        Graph
            Job-shop graph
        """
        order = self.get_order(x)
        return self.build_graph(order)
        

class LSDecoder(Decoder):
    
    def decode(self, x):
        # Get sorted values of base vector
        order = self.get_order(x)
        order_hash = hash(str(order))
        
        # Avoid re-calculation if pheno is known
        if order_hash in self.known_solutions:
            pheno, x_new, C = self.known_solutions[order_hash]
            
        else:
            graph = self.build_graph(order)
            C = graph.C
            pheno = graph.pheno
            order_new = graph.order
  
            x_new = np.zeros_like(x)
            for i in np.unique(order_new):
                idx = np.flatnonzero(order_new == i)
                x_new[idx] = x[idx]
            
            order_new_hash = hash(str(order_new))
            self.known_solutions[order_hash] = pheno, x_new, C
            self.known_solutions[order_new_hash] = pheno, x_new, C
        
        return pheno, x_new, C
    
    def build_graph(self, order):
        graph = super().build_graph(order)
        calc_tails(graph)
        get_critical(graph)
        return local_search(graph)