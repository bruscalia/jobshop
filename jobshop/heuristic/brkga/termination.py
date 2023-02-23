from pymoo.core.termination import Termination, TerminateIfAny
from pymoo.termination.max_gen import MaximumGenerationTermination


class BaseTarget(Termination):
    
    def __init__(self, target=None) -> None:
        super().__init__()
        if target is None:
            target = - float("inf")
        self.target = target
    
    def _update(self, algorithm):
        F = float("inf")
        if algorithm.opt is not None:
            F = algorithm.opt.get("F")
        if F <= self.target:
            return 1.0
        else:
            return 0.0


class TargetTermination(TerminateIfAny):
    
    def __init__(self, n_gen=float("inf"), target=None) -> None:
        super().__init__()
        self.max_gen = MaximumGenerationTermination(n_gen)
        self.target = BaseTarget(target)
        self.criteria = [self.max_gen, self.target]