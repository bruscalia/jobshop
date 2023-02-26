# jobshop 
Python package for modeling the job-shop scheduling problem using mixed-integer programming (MIP) and meta-heuristics.

See here a [MIP](#mip) and a [GRASP](#grasp) example.

Examples on how to run experiments on benchmark test problems can be found here for [GRASP](./notebooks/grasp_instance.ipynb) and [BRKGA](./notebooks/brkga_instance.ipynb)

## Install

First, make sure you have a Python 3 environment installed.

From the current version on github:
```
pip install git+https://github.com/bruscalia/jobshop
```

## MIP

Let us see an example of how to implement the disjunctive model for the job-shop problem, described by the equations below.

$$
\begin{align*}
    \text{min} \quad & C \\
    \text{s.t.} \quad & x_{\sigma_{h-1}^j, j} + p_{\sigma_{h-1}^j, j} \leq x_{\sigma_{h}^j, j}
        & \forall ~ j \in J; h \in (2, ..., |M|)\\
    & x_{m, j} + p_{m, j} \leq x_{m, k} + V (1 - z_{m, j, k})
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

Although a very slow solution process, the following GRASP with Path Relinking implementation would produce a good quality solution of 154 versus known optima 153.

```python
import numpy as np
from jobshop.params import JobShopRandomParams
from jobshop.heuristic.grasp import grasp, grasp_pr
```

```python
params = JobShopRandomParams(10, 10, t_span=(0, 20), seed=12)
```

```python
sol_grasp = grasp(params, maxiter=10000, alpha=(0.3, 0.9))
print(sol_grasp.C)
```

```python
pool_pr = grasp_pr(
    params, maxiter=10000, ifreq=1000, mixed_construction=True,
    alpha=(0.0, 1.0), maxpool=12, seed=12,
)
sol_grasp_pr = pool_pr[0]
print(sol_grasp_pr.C)
```

## BRKGA

The following code is expected to return the good solution of 154 versus known optima 153. The termination operator passed would force the algorithm to stop if either the number of generations reaches 200 or a solution with objective value of 153 is found.

```python
from pymoo.optimize import minimize
from jobshop.params import JobShopRandomParams
from jobshop.heuristic.brkga import BRKGA, LSDecoder, JobShopProblem
from jobshop.heuristic.brkga.termination import TargetTermination
```

```python
params = JobShopRandomParams(10, 10, t_span=(0, 20), seed=12)
```

```python
brkga = BRKGA(
    pop_size=100,
    perc_elite=0.2,
    perc_mutants=0.15,
    bias=0.8,
)
problem = JobShopProblem(params, LSDecoder)
res = minimize(problem, brkga, termination=TargetTermination(200, 153), verbose=True, seed=12)
```

```python
graph = problem.decoder.build_graph_from_string(res.X)
print(graph.C)
```
