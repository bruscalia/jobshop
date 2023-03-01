import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.operators.crossover.binx import mut_binomial


# =========================================================================================================
# Derived from pymoo with a minor fix
# =========================================================================================================


class BinomialCrossover(Crossover):

    def __init__(self, bias=0.5, n_offsprings=2, **kwargs):
        """Binomial crossover operator

        Parameters
        ----------
        bias : float, optional
            Bias in inheritance of elements from the first parent, by default 0.5
        
        n_offsprings : int, optional
            Number of offsprings generated from each pair of parents, by default 2
        """
        super().__init__(2, n_offsprings, **kwargs)
        self.bias = bias

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape

        M = mut_binomial(n_matings, n_var, self.bias, at_least_once=True)

        if self.n_offsprings == 1:
            Xp = X[0].copy()
            Xp[~M] = X[1][~M]
            Xp = np.array([Xp])
        elif self.n_offsprings == 2:
            Xp = np.copy(X)
            Xp[0][~M] = X[1][~M]
            Xp[1][~M] = X[0][~M]
        else:
            raise Exception

        return Xp