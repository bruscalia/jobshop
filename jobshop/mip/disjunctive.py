import pyomo.environ as pyo
from jobshop.mip.base import JobShopModel


def cstr_seq(model, m, j):
    o = model.seq[j].prev(m)
    if o is not None:
        return model.x[o, j] + model.p[o, j] <= model.x[m, j]
    else:
        return pyo.Constraint.Skip


def cstr_precede(model, m, j, k):
    if j != k:
        return model.x[m, j] + model.p[m, j] <= model.x[m, k] + model.V * (1 - model.z[m, j, k])
    else:
        return pyo.Constraint.Skip


def cstr_comp_precede(model, m, j, k):
    if j != k:
        return model.z[m, j, k] + model.z[m, k, j] == 1
    else:
        return model.z[m, j, k] == 0


def cstr_total_time(model, j):
    m = model.seq[j][-1]
    return model.x[m, j] + model.p[m, j] <= model.C


class DisjModel(JobShopModel):

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self._create_vars()
        self._create_cstr()
        self.obj = pyo.Objective(rule=self.C, sense=pyo.minimize)

    def _create_vars(self):
        self.x = pyo.Var(self.M, self.J, within=pyo.NonNegativeReals)
        self.z = pyo.Var(self.M, self.J, self.J, within=pyo.Binary)
        self.C = pyo.Var(within=pyo.NonNegativeReals)

    def _create_cstr(self):
        self.cstr_seq = pyo.Constraint(self.M, self.J, rule=cstr_seq)
        self.cstr_precede = pyo.Constraint(self.M, self.J, self.J, rule=cstr_precede)
        self.cstr_comp_precede = pyo.Constraint(self.M, self.J, self.J, rule=cstr_comp_precede)
        self.cstr_total_time = pyo.Constraint(self.J, rule=cstr_total_time)

    def _get_elements(self, j):
        machines = [x.index()[0] for x in self.x[:, j]]
        starts = [x.value for x in self.x[:, j]]
        spans = [self.p[m, j] for m in machines]
        return machines, starts, spans
