import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.core.selection import Selection
from pymoo.operators.mutation.nom import NoMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util.display.single import SingleObjectiveOutput
from jobshop.heuristic.brkga.survival import EliteSurvival
from jobshop.heuristic.brkga.crossover import BinomialCrossover



# =========================================================================================================
# Elite Selection
# =========================================================================================================

class EliteBiasedSelection(Selection):
    
    def __init__(self, elite_size, **kwargs) -> None:
        super().__init__(**kwargs)
        self.elite_size = elite_size

    def _do(self, problem, pop, n_select, n_parents, **kwargs):

        # do the mating selection - always one elite and one non-elites
        s_elite = np.random.choice(np.arange(self.elite_size), size=n_select, replace=True)
        s_non_elite = np.random.choice(np.arange(len(pop) - self.elite_size) + self.elite_size, size=n_select)

        return np.column_stack([s_elite, s_non_elite])


# =========================================================================================================
# BRKGA
# =========================================================================================================

class BRKGA(GeneticAlgorithm):

    def __init__(
        self,
        pop_size=100,
        perc_elite=0.2,
        perc_mutants=0.15,
        bias=0.7,
        sampling=FloatRandomSampling(),
        survival=None,
        mutation=None,
        eliminate_duplicates=True,
        output=SingleObjectiveOutput(),
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
            Survival operator. If None, EliteSurvival with duplicate elimination is used. By default None
        
        mutation : pymoo Survival, optional
            Survival operator, by default None
        
        eliminate_duplicates : bool | DuplicateElimination, optional
            Pymoo strategy to eliminate duplicates passed to the EliteSurvival, by default True
        
        output : pymoo Output, optional
            By default SingleObjectiveOutput()
        """
        
        # perc_elite + perc_mutants can't be more than one
        if perc_mutants + perc_elite >= 1.0:
            raise ValueError("Elite and mutants can't be more than 100 percent of the population")

        if survival is None:
            survival = EliteSurvival(eliminate_duplicates=eliminate_duplicates)
        
        if mutation is None:
            mutation = NoMutation()
        
        elite_size = int(pop_size * perc_elite)
        n_mutants = int(pop_size * perc_mutants)
        n_offsprings = pop_size - elite_size - n_mutants

        super().__init__(
            pop_size=pop_size,
            n_offsprings=n_offsprings,
            sampling=sampling,
            selection=EliteBiasedSelection(elite_size),
            crossover=BinomialCrossover(bias, n_offsprings=1, prob=1.0),
            mutation=mutation,
            survival=survival,
            output=output,
            eliminate_duplicates=True,
            advance_after_initial_infill=True,
            **kwargs,
        )

        self.elite_size = elite_size
        self.n_mutants = n_mutants
        self.bias = bias
        self.termination = DefaultSingleObjectiveTermination()

    def _infill(self):
 
        # actually do the mating given the elite selection and biased crossover
        off = self.mating.do(self.problem, self.pop, n_offsprings=self.n_offsprings, algorithm=self)

        # create the mutants randomly to fill the population with
        mutants = FloatRandomSampling().do(self.problem, self.n_mutants, algorithm=self)

        # evaluate all the new solutions
        return Population.merge(off, mutants)

    def _advance(self, infills=None, **kwargs):
        
        # Get current population
        pop = self.pop

        # get all the elites from the current population
        elites = pop[:self.elite_size]

        # finally merge everything together and sort by fitness
        pop = Population.merge(elites, infills)

        # the do survival selection - set the elites for the next round
        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self)