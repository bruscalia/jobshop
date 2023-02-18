import numpy as np


class JobSequence(list):
    
    def prev(self, x):
        if self.is_first(x):
            return None
        else:
            i = self.index(x)
            return self[i - 1]
    
    def next(self, x):
        if self.is_last(x):
            return None
        else:
            i = self.index(x)
            return self[i + 1]
    
    def is_first(self, x):
        return x == self[0]
    
    def is_last(self, x):
        return x == self[-1]
    
    def swap(self, x, y):
        i = self.index(x)
        j = self.index(y)
        self[i] = y
        self[j] = x
    
    def append(self, __object) -> None:
        if __object not in self:
            super().append(__object)
        else:
            pass


class JobShopParams:
    
    def __init__(self, machines, jobs, p_times, seq):
        self.machines = machines
        self.jobs = jobs
        self.p_times = p_times
        self.seq = seq


class JobShopRandomParams(JobShopParams):
    
    def __init__(self, n_machines, n_jobs, t_span=(0, 20), seed=None):
        self.t_span = t_span
        self.seed = seed
        
        machines = np.arange(n_machines) + 1
        jobs = np.arange(n_jobs) + 1
        p_times = self._random_times(machines, jobs, t_span)
        seq = self._random_sequences(machines, jobs)
        super().__init__(machines, jobs, p_times, seq)
    
    def _random_times(self, machines, jobs, t_span):
        np.random.seed(self.seed)
        t = np.arange(t_span[0], t_span[1])
        return {
            (m, j): np.random.choice(t)
            for m in machines
            for j in jobs
        }
    
    def _random_sequences(self, machines, jobs):
        np.random.seed(self.seed)
        return {
            j: JobSequence(np.random.permutation(machines))
            for j in jobs
        }

