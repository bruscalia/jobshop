import pyomo.environ as pyo
from jobshop.mip.base import JobShopModel


def cstr_unique_start(model, m, j):
    return sum(model.x[m, j, t] for t in model.T) == 1


def cstr_unique_machine(model, m, t):
    total = 0
    start = model.T.first()
    for j in model.J:
        duration = model.p[m, j]
        t0 = max(start, t - duration + 1)
        for t1 in range(t0, t + 1):
            total = total + model.x[m, j, t1]
    return total <= 1


def cstr_precede(model, m, j):
    o = model.seq[j].prev(m)
    if o is None:
        prev_term = 0
    else:
        prev_term = sum(
            (t + model.p[o, j]) * model.x[o, j, t]
            for t in model.T
        )
    current_term = sum(
        t * model.x[m, j, t]
        for t in model.T
    )
    return prev_term <= current_term


def cstr_total_time(model, j):
    m = model.seq[j][-1]
    return sum((t + model.p[m, j]) * model.x[m, j, t] for t in model.T) <= model.C


class TimeModel(JobShopModel):

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.T = pyo.RangeSet(self.V)
        self._create_vars()
        self._create_cstr()
        self.obj = pyo.Objective(rule=self.C, sense=pyo.minimize)

    def _create_vars(self):
        self.x = pyo.Var(self.M, self.J, self.T, within=pyo.Binary)
        self.C = pyo.Var(within=pyo.NonNegativeReals)

    def _create_cstr(self):
        self.cstr_unique_start = pyo.Constraint(self.M, self.J, rule=cstr_unique_start)
        self.cstr_unique_machine = pyo.Constraint(self.M, self.T, rule=cstr_unique_machine)
        self.cstr_precede = pyo.Constraint(self.M, self.J, rule=cstr_precede)
        self.cstr_total_time = pyo.Constraint(self.J, rule=cstr_total_time)

    def _get_elements(self, j):
        machines = [x.index()[0] for x in self.x[:, j, :] if x.value == 1]
        starts = [x.index()[2] for x in self.x[:, j, :] if x.value == 1]
        spans = [self.p[m, j] for m in machines]
        return machines, starts, spans
