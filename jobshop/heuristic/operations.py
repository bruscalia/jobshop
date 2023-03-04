import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
        jobs=None,
    ) -> None:
        if jobs is None:
            jobs = JobSequence()
        self.key = key
        self.jobs = jobs
    
    def __repr__(self) -> str:
        return str(self.__dict__)
    
    def add_job(self, job):
        self.jobs.append(job)


class Graph(JobShopParams):
    
    colors = mpl.colormaps["Dark2"].colors + mpl.colormaps["Set2"].colors
    
    def __init__(self, params: JobShopParams):
        """Graph structure to job-shop problem

        Parameters
        ----------
        params : JobShopParams
            Parameters that define the problem
        """
        super().__init__(params.machines, params.jobs, params.p_times, params.seq)
        self.M = {
            m: Machine(m, jobs=JobSequence())
            for m in self.machines
        }
        self.O = {}
        self._start_operations()
        self.V = sum(self.p_times[key] for key in self.p_times)
    
    def restart(self):
        self.__init__(self)
    
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
    
    @property
    def order(self):
        releases = np.array([o.release for o in self.O.values()])
        unordered = np.array([o.job for o in self.O.values()])
        seq = np.argsort(releases)
        order = unordered[seq]
        return order
    
    @property
    def pheno(self):
        pheno = []
        for m in self.M.values():
            pheno = pheno + m.jobs
        return np.array(pheno)
    
    @property
    def signature(self):
        return hash(str(self.order))
    
    def plot(self, horizontal=True, figsize=[7, 3], dpi=100, colors=None):
        if horizontal:
            self._plot_horizontal(figsize=figsize, dpi=dpi, colors=colors)
        else:
            self._plot_vertical(figsize=figsize, dpi=dpi, colors=colors)

    def _plot_vertical(self, figsize=[7, 3], dpi=100, colors=None):
        
        if colors is None:
            colors = self.colors
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for i, j in enumerate(self.jobs):
            machines, starts, spans = self._get_elements(j)
            
            if i >= len(colors):
                i = i % len(colors)
            
            color = colors[i]
            ax.bar(machines, spans, bottom=starts, label=f"Job {j}", color=color)

        ax.set_xticks(self.machines)
        ax.set_xlabel("Machine")
        ax.set_ylabel("Time")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.03))
        fig.tight_layout()
        plt.show()

    def _plot_horizontal(self, figsize=[7, 3], dpi=100, colors=None):
        
        colors = self._get_colors(colors)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for i, j in enumerate(self.jobs):
            machines, starts, spans = self._get_elements(j)
            
            if i >= len(colors):
                i = i % len(colors)
            
            color = colors[i]
            ax.barh(machines, spans, left=starts, label=f"Job {j}", color=color)

        ax.set_yticks(self.machines)
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.03))
        fig.tight_layout()
        plt.show()
    
    def _get_colors(self, colors):
        if colors is None:
            colors = self.colors
        return colors
    
    def _get_elements(self, j):
        machines = self.machines
        starts = [self.O[m, j].release for m in self.machines]
        spans = [self.O[m, j].duration for m in self.machines]
        return machines, starts, spans