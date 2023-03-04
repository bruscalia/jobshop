import numpy as np
from pymoo.core.duplicate import ElementwiseDuplicateElimination


def to_float(val):
    if isinstance(val, bool) or isinstance(val, np.bool_):
        return 0.0 if val else 1.0
    else:
        return val


class SortedDuplicates(ElementwiseDuplicateElimination):
    
    def _do(self, pop, other, is_duplicate):

        if other is None:
            for i in range(len(pop)):
                for j in range(i + 1, len(pop)):
                    val = to_float(self.cmp_func(pop[i], pop[j]))
                    if val < self.epsilon:
                        is_duplicate[j] = True
                        break
        else:
            for i in range(len(pop)):
                for j in range(len(other)):
                    val = to_float(self.cmp_func(pop[i], other[j]))
                    if val < self.epsilon:
                        is_duplicate[i] = True
                        break

        return is_duplicate

    def is_equal(self, a, b):
        dx = a.get("X") - b.get("X")
        return dx.dot(dx) <= self.epsilon


class PhenoDuplicates(SortedDuplicates):
    
    def __init__(self, min_diff=0.25, **kwargs) -> None:
        """This is an operator to eliminate duplicates based on the minimum percentual difference
        between the phenotype of two individuals. The first one is preserved.

        Parameters
        ----------
        min_diff : float, optional
            Minimum percentual difference, by default 0.25
        """
        super().__init__(None, **kwargs)
        self.min_diff = min_diff
    
    def is_equal(self, a, b):
        comp = a.get("pheno") != b.get("pheno")
        return np.sum(comp) <= len(comp) * self.min_diff