import numpy as np
from typing import Any, Union
from jobshop.params import JobShopParams, JobSequence
import copy


class Operation:
    
    def __init__(
        self,
        machine: Any,
        job: Any,
        duration: Union[int, float],
        release=None,
    ) -> None:
        self.machine = machine
        self.job = job
        self.code = machine, job
        self.duration = duration
        self.release = release
        self.tail = None
        self.critical = False
    
    def __repr__(self) -> str:
        return str(self.__dict__)
    
    def reset_release(self, release):
        self.release = release


class Machine:
    
    def __init__(
        self,
        key,
        jobs=JobSequence([]),
    ) -> None:
        self.key = key
        self.jobs = jobs
    
    def __repr__(self) -> str:
        return str(self.__dict__)
    
    def add_job(self, job):
        self.jobs = JobSequence(np.append(self.jobs, job))


class Graph(JobShopParams):
    
    def __init__(self, machines, jobs, p_times, seq):
        """Graph structure to job-shop problem

        Parameters
        ----------
        machines : Iterable
            Iterable of machine labels
        
        jobs : Iterable
            Iterable of job labels
        
        p_times : dict
            Dictionary with duration of each opeartion (m, j)
        
        seq : dict
            Dictionary of sequence of machines per job
        """
        super().__init__(machines, jobs, p_times, seq)
        self.M = {
            m: Machine(m)
            for m in self.machines
        }
        self.O = {}
        self._start_operations()
        self.V = sum(self.p_times[key] for key in self.p_times)
    
    def _start_operations(self):
        for m in self.machines:
            for j in self.jobs:
                self.O[m, j] = Operation(m, j, self.p_times[m, j], 0.0)
    
    def precede_job(self, machine, job):
        """Returns the operation belonging to job processed just before (machine, job)

        Parameters
        ----------
        machine : Any
            Machine key
        
        job : Any
            Job key

        Returns
        -------
        Operation
        """
        last_machine = self.seq[job].prev(machine)
        if last_machine is not None:
            return self.O[last_machine, job]
        else:
            return None
    
    def follow_job(self, machine, job):
        """Returns the operation belonging to job processed right after (machine, job)

        Parameters
        ----------
        machine : Any
            Machine key
        
        job : Any
            Job key

        Returns
        -------
        Operation
        """
        next_machine = self.seq[job].next(machine)
        if next_machine is not None:
            return self.O[next_machine, job]
        else:
            return None
    
    def precede_machine(self, machine, job):
        """Returns the operation processed on machine just before (machine, job)

        Parameters
        ----------
        machine : Any
            Machine key
        
        job : Any
            Job key

        Returns
        -------
        Operation
        """
        last_job = self.M[machine].jobs.prev(job)
        if last_job is not None:
            return self.O[machine, last_job]
        else:
            return None
    
    def follow_machine(self, machine, job):
        """Returns the operation processed on machine right after (machine, job)

        Parameters
        ----------
        machine : Any
            Machine key
        
        job : Any
            Job key

        Returns
        -------
        Operation
        """
        last_job = self.M[machine].jobs.next(job)
        if last_job is not None:
            return self.O[machine, last_job]
        else:
            return None
    
    def copy(self):
        return copy.deepcopy(self)