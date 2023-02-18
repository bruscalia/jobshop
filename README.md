# jobshop 
Python package for modeling the job-shop scheduling problem using mixed-integer programming (MIP) and meta-heuristics.

See here a [MIP](#mip) and a [GRASP](#grasp) example.

## MIP

Let us see an example of how to implement the disjunctive model for the job-shop problem, described by the equations below.

$$
\begin{align}
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
\end{align}
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
from jobshop.heurstic.grasp import simple_grasp
```

```python
sol_grasp = simple_grasp(params, n_iter=1000, alpha=0.8, seed=12)
print(sol_grasp.value)
```
