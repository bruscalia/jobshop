import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.duplicate import DefaultDuplicateElimination, DuplicateElimination
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# =========================================================================================================
# Implementation
# =========================================================================================================


class EliteSurvival(Survival):

    def __init__(self, eliminate_duplicates=True, base_survival=None):
        super().__init__(False)
        if base_survival is None:
            base_survival = FitnessSurvival()
        self.base_survival = base_survival
        if isinstance(eliminate_duplicates, bool) and eliminate_duplicates:
            eliminate_duplicates = DefaultDuplicateElimination()
        self.eliminate_duplicates = eliminate_duplicates

    def _do(self, problem, pop, n_survive=None, algorithm=None, **kwargs):

        # Do base survival (likely to be a sorting operator)
        pop = self.base_survival.do(problem, pop)
        pop = self.eliminate_duplicates.do(pop)

        return pop