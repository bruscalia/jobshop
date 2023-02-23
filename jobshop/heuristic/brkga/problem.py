import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from jobshop.params import JobShopParams
from jobshop.heuristic.brkga.decoder import Decoder


class JobShopProblem(ElementwiseProblem):
    
    def __init__(self, params: JobShopParams, decoder_class=Decoder):
        self.params = params
        n_var = 0
        for j, machines in self.params.seq.items():
            n_var = n_var + len(machines)
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        self.decoder = decoder_class(
            self.params.machines, self.params.jobs,
            self.params.p_times, self.params.seq
        )
        super().__init__(elementwise=True, n_var=n_var, n_obj=1, xl=xl, xu=xu)
    
    def _evaluate(self, x, out, *args, **kwargs):
        pheno, x_new, C = self.decoder.decode(x)
        out["pheno"] = pheno
        out["X"] = x_new
        out["hash"] = hash(str(pheno))
        out["F"] = C


class DuplicatesEncoder(ElementwiseDuplicateElimination):
    
    def __init__(self, x_tol=1e-3) -> None:
        super().__init__()
        self.x_tol = x_tol

    def is_equal(self, a, b):
        same_pheno = a.get("hash") == b.get("hash")
        diff_x = a.get("X") - b.get("X")
        dist_x = np.sqrt(diff_x.dot(diff_x))
        return same_pheno and dist_x <= self.x_tol * len(diff_x)