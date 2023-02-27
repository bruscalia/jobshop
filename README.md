# jobshop 
Python package for modeling the job-shop scheduling problem using mixed-integer programming (MIP) and meta-heuristics.

See here a [MIP](#mip) and [Metaheuristics](#metaheuristics) examples.

## Install

First, make sure you have a Python 3 environment installed.

From the current version on github:
```
pip install git+https://github.com/bruscalia/jobshop
```

## Problems

In this framework, problems are defined by ``JobShopParams`` instances. There is available a random generator and a convenience to instantiate problems from the literature using the ``from jobshop.params import job_params_from_json`` interface. Json files are available for some instances [here](./instances/orlib).

```python
from jobshop.params import JobShopRandomParams
from jobshop.params import job_params_from_json
```

## MIP

Let us see an example of how to implement the disjunctive model for the job-shop problem, described by the equations below. Those interested in a detailed review of MIP formulations to the job-shop problem can refer to [Ku & Beck (2016)](#mipjssp).

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

![jobshop_mip_plot](./data/jobshop_plot.png)


## Metaheuristics

The problem could be formulated in a similar manner with logical relationships between elements, in which we must define the starting time $t$ of each operation $\sigma_{k}^j$ of processing time $p$. For more details of this formulation or [GRASP and GRASP-PR](#grasp) implementation, we suggest referring to [Aiex et al. (2003)](#graspprjssp). For more details of the [BRKGA](#brkga) decoding process, one can refer to [Gonçalves & Resende (2014)](#brkgajssp).

$$
\begin{align}
    \text{min} \quad & C_{max} \\
    \text{s.t.} \quad & t(\sigma_{k}^j) + p(\sigma_{k}^j) \leq C_{max}
        & \forall ~ \sigma_{k}^j \in O \\
    & t(\sigma_{l}^j) + p(\sigma_{l}^j) \leq t(\sigma_{k}^j)
        & \forall ~ \sigma_{l}^j \prec \sigma_{k}^j \\
        & \begin{split}
            & t(\sigma_{l}^i) + p(\sigma_{l}^i) \leq t(\sigma_{k}^j) ~\lor \\
            & t(\sigma_{k}^j) + p(\sigma_{k}^j) \leq t(\sigma_{l}^i)
        \end{split} & \forall ~ \sigma_{k}^j, \sigma_{l}^i \in O, ~ M_{\sigma_{k}^j} = M_{\sigma_{l}^i} \\
    & t(\sigma_{k}^j) \geq 0 & \forall ~ \sigma_{k}^j \in O \\
\end{align}
$$

### GRASP

Now consider the instance *mt10*, a 10x10 problem from the literature. This problem has a known optimal solution of 933, although 950 is already a challenging target.

```python
from jobshop.params import job_params_from_json
from jobshop.heuristic.grasp import grasp, grasp_pr
```

```python
params = job_params_from_json("./instances/orlib/mt10.json")
```

```python
# Pure GRASP (fast solutions with lower quality)
sol_grasp = grasp(params, maxiter=100000, alpha=(0.3, 1.0))
print(sol_grasp.C)
```

```python
# GRASP-PR
pool_pr = grasp_pr(
    params, maxiter=100000, ifreq=10000, post_opt=True,
    init_iter=0.6, alpha=(0.3, 1.0), maxpool=30, min_diff=0.3,
    seed=12, target=950,
)
sol_grasp_pr = pool_pr[0]
print(sol_grasp_pr.C)
```

These configurations would return solutions with makespan of 979 for GRASP and 945 for GRASP-PR, although the latter would take several hours to be obtained.

```python
sol_grasp_pr.plot()
```

![jobshop_grasppr_plot](./data/grasp_pr_mt10_results_945.png)

### BRKGA

Let us once again consider problem *mt10* and re-use the previously defined ``params`` instance. Those interested in understanging better the mechanisms of BRKGA might refer to [this presentation](http://www.decom.ufop.br/prof/marcone/Disciplinas/InteligenciaComputacional/brkga.pdf).

```python
from pymoo.optimize import minimize
from jobshop.heuristic.brkga import BRKGA, LSDecoder, JobShopProblem, DuplicatesEncoder
from jobshop.heuristic.brkga.termination import TargetTermination
```

```python
brkga = BRKGA(
    pop_size=200,
    perc_elite=0.25,
    perc_mutants=0.1,
    bias=0.8,
    eliminate_duplicates=DuplicatesEncoder(),
)
problem = JobShopProblem(params, LSDecoder)
res = minimize(problem, brkga, termination=TargetTermination(1000, 950), verbose=True, seed=12)
```

```python
graph = problem.decoder.build_graph_from_string(res.X)
print(graph.C)
```

Using this configuration, BRKGA would return a solution with makespan of 951, which would also take a few hours, although good quality solutions (lesser than 970) could be found within a few minutes.

```python
graph.plot()
```

![jobshop_brkga_plot](./data/brkga_mt10_results_951.png)


## References

<a id="graspprjssp"></a> 
Aiex, R. M., Binato, S., & Resende, M. G. (2003). Parallel GRASP with path-relinking for job shop scheduling. Parallel Computing, 29(4), 393-430.

<a id="brkgajssp"></a> 
Gonçalves, J. F., & Resende, M. G. (2014). An extended Akers graphical method with a biased random‐key genetic algorithm for job‐shop scheduling. International Transactions in Operational Research, 21(2), 215-246.

<a id="mipjssp"></a> 
Ku, W. Y., & Beck, J. C. (2016). Mixed integer programming models for job shop scheduling: A computational analysis. Computers & Operations Research, 73, 165-173.