import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.core.selection import Selection
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.operators.mutation.nom import NoMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util.display.single import SingleObjectiveOutput



# =========================================================================================================
# Elite Selection
# =========================================================================================================

class EliteBiasedSelection(Selection):
    
    def __init__(self, n_elite, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_elite = n_elite

    def _do(self, problem, pop, n_select, n_parents, **kwargs):

        # do the mating selection - always one elite and one non-elites
        s_elite = np.random.choice(np.arange(self.n_elite), size=n_select, replace=True)
        s_non_elite = np.random.choice(np.arange(len(pop) - self.n_elite) + self.n_elite, size=n_select)

        return np.column_stack([s_elite, s_non_elite])


# =========================================================================================================
# BRKGA
# =========================================================================================================

class BRKGA(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 perc_elite=0.2,
                 perc_mutants=0.15,
                 bias=0.7,
                 sampling=FloatRandomSampling(),
                 survival=None,
                 output=SingleObjectiveOutput(),
                 eliminate_duplicates=False,
                 **kwargs
                 ):
        """BRKGA implementation adpated from pymoo

        Parameters
        ----------
        pop_size : int, optional
            Population size, by default 100
        
        perc_elite : float, optional
            Part of pop_size reserved to elite solutions, by default 0.2
        
        perc_mutants : float, optional
            Part of pop_size reserved to randomly generated solutions, by default 0.15
        
        bias : float, optional
            Bias in crossover, by default 0.7
        
        sampling : pymoo Sampling, optional
            Sampling operator, by default FloatRandomSampling()
        
        survival : pymoo Survival, optional
            Survival operator, by default None
        
        output : pymoo Output, optional
            By default SingleObjectiveOutput()
        
        eliminate_duplicates : bool, optional
            Pymoo strategy to eliminate duplicates, by default False
        """

        if survival is None:
            survival = FitnessSurvival()
        
        n_elites = int(pop_size * perc_elite)
        n_mutants = int(pop_size * perc_mutants)
        n_offsprings = pop_size - n_elites - n_mutants

        super().__init__(pop_size=pop_size,
                         n_offsprings=n_offsprings,
                         sampling=sampling,
                         selection=EliteBiasedSelection(n_elites),
                         crossover=BinomialCrossover(bias, prob=1.0),
                         mutation=NoMutation(),
                         survival=survival,
                         output=output,
                         eliminate_duplicates=eliminate_duplicates,
                         advance_after_initial_infill=True,
                         **kwargs)

        self.n_elites = n_elites
        self.n_mutants = n_mutants
        self.bias = bias
        self.termination = DefaultSingleObjectiveTermination()

    def _infill(self):
        pop = self.pop

        # actually do the mating given the elite selection and biased crossover
        off = self.mating.do(self.problem, pop, n_offsprings=self.n_offsprings, algorithm=self)

        # create the mutants randomly to fill the population with
        mutants = FloatRandomSampling().do(self.problem, self.n_mutants, algorithm=self)

        # evaluate all the new solutions
        return Population.merge(off, mutants)

    def _advance(self, infills=None, **kwargs):
        pop = self.pop

        # get all the elites from the current population
        elites = pop[:self.n_elites]

        # finally merge everything together and sort by fitness
        pop = Population.merge(elites, infills)

        # the do survival selection - set the elites for the next round
        self.pop = self.survival.do(self.problem, pop, n_survive=len(pop), algorithm=self)