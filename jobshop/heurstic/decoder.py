import numpy as np
from jobshop.params import JobShopParams
from jobshop.heurstic.operations import Graph
from jobshop.heurstic.evaluation import calc_makespan


class Decoder(JobShopParams):
    
    def __init__(self, machines, jobs, p_times, seq):
        """Decoder for Genetic Algorithms applied to the job-shop schedule problem

        Parameters
        ----------
        machines : Iterable
            Machines
        
        jobs : Iterable
            Jobs
        
        p_times : dict
            Duration of operations (m, j)
        
        seq : dict
            Sequence of machines for job j
        """
        super().__init__(machines, jobs, p_times, seq)
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
            phenotype and objective value
        """
        
        # Get sorted values of base vector
        pheno = self.get_pheno(x)
        pheno_hash = hash(str(pheno))
        
        # Avoid re-calculation if pheno is known
        if pheno_hash in self.known_solutions:
            C = self.known_solutions[pheno_hash]
            
        else:
            graph = self.build_graph(pheno)
            C = graph.C
            self.known_solutions[pheno_hash] = C
        
        return pheno, C
    
    def build_graph(self, pheno):
        """Build and evaluate problem graph from phenotype

        Parameters
        ----------
        pheno : numpy.array
            Phenotype of solution

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
        for j in pheno:
            k = assigned[j]
            m = self.seq[j][k]
            Q.append((m, j))
            assigned[j] = assigned[j] + 1
        
        # Initialize graph
        graph = Graph(self.machines, self.jobs, self.p_times, self.seq)
        for (m, j) in Q:
            graph.M[m].add_job(j)
        
        # Calculate makespan
        calc_makespan(graph)
        
        return graph
    
    def get_pheno(self, x):
        idx = np.argsort(x)
        pheno = self.base_vector[idx]
        return pheno
    
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
        pheno = self.get_pheno(x)
        return self.build_graph(pheno)
        
        