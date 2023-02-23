# jobshop 
Python package for modeling the job-shop scheduling problem using mixed-integer programming (MIP) and meta-heuristics.

See here a [MIP](#mip) and a [GRASP](#grasp) example.

## MIP

Let us see an example of how to implement the disjunctive model for the job-shop problem, described by the equations below.

$$
\begin{align*}
    \text{min} \quad & C \\
    \text{s.t.}~~ & x_{\sigma_{h-1}^j, j} + p_{\sigma_{h-1}^j, j} \leq x_{\sigma_{h}^j, j}
        & \forall ~ j \in J; h \in (2, ..., |M|)\\
    & x_{m, j} + p_{m, j} \leq mx_{m, k} + V (1 - z_{m, j, k})
        & \forall ~ j, k \in J, j \neq k; m \in M\\
    & z_{m, j, k} + z_{m, k, j} = 1
        & \forall ~ j, k \in J, j \neq k; m \in M\\
    & x_{\sigma_{|M|}^j, j} + p_{\sigma_{|M|}^j, j} \leq C
        & \forall ~ j \in J\\
    & x_{m, j} \geq 0 & \forall ~ j \in J; m \in M\\
    & z_{m, j, k} \in \{0, 1\} & \forall ~ j, k \in J; m \in M\\
\end{align*}
$$

```python
import numpy as np
import pyomo.environ as pyo
from jobshop.params import JobShopRandomParams
from jobshop.mip.disjunctive import DisjModel
from jobshop.mip.timeindex import TimeModel
```

```python
params = JobShopRandomParams(5, 4, t_span=(5, 20), seed=12)
disj_model = DisjModel(params)

solver = pyo.SolverFactory(
    "cbc", 
    options=dict(cuts="on", sec=20, heur="on", RINS="both", DINS="both"),
)
print(solver.solve(disj_model, tee=True))
```

```python
disj_model.plot()
```

![jobshop_plot](./data/jobshop_plot.png)


## GRASP

This is still under development, however you can have a glimpse of the final results.

The next steps are being included in this [unstable notebook](./notebooks/test_grasp.ipynb).

```python
import numpy as np
from jobshop.params import JobShopRandomParams
from jobshop.heuristic.grasp.simple import grasp
from jobshop.heuristic.grasp.pr import grasp_pr
```

```python
sol_grasp = grasp(params, maxiter=10000, alpha=(0.3, 0.9))
print(sol_grasp.C)
```

```python
pool_pr = grasp_pr(params, maxiter=10000, init_iter=0.5, alpha=(0.3, 0.9), maxpool=20)
sol_grasp_pr = pool_pr[0]
print(sol_grasp_pr.C)
```

## BRKGA

```python
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.optimize import minimize
from jobshop.params import JobShopRandomParams, JobShopParams
from jobshop.heuristic.decoder import Decoder
from jobshop.heuristic.brkga import BRKGA
```

```python
params = JobShopRandomParams(10, 10, t_span=(5, 20), seed=12)
```

```python
class JobShopProblem(ElementwiseProblem):
    
    def __init__(self, params: JobShopParams):
        self.params = params
        n_var = 0
        for j, machines in self.params.seq.items():
            n_var = n_var + len(machines)
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        self.decoder = Decoder(
            self.params.machines, self.params.jobs,
            self.params.p_times, self.params.seq
        )
        super().__init__(elementwise=True, n_var=n_var, n_obj=1, xl=xl, xu=xu)
    
    def _evaluate(self, x, out, *args, **kwargs):
        z, C = self.decoder.decode(x)
        out["pheno"] = z
        out["hash"] = hash(str(z))
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
```

```python
brkga = BRKGA(
    pop_size=100,
    perc_elite=0.15,
    perc_mutants=0.15,
    bias=0.8,
    eliminate_duplicates=DuplicatesEncoder(1e-5),
)
problem = JobShopProblem(params)
res = minimize(problem, brkga, ("n_gen", 200), verbose=True, seed=12)
```

```python
graph = problem.decoder.build_graph_from_string(res.X)
print(graph.C)
```
