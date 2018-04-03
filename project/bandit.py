"""
Multi-armed bandit problems - multiple generation solutions.

TODO:
  - provide tools to generate "arbitrary" bandit problems
  - be able to replay a sequence of samples (so we can exactly compare
    algorithms)
  - save results
  - display performance and regret graphs
"""
import abc
from typing import (  # Type hinting (for Python 3.5 and above)
    Sequence, Optional, Union, Mapping
)

from project.utils import RandomStateMixin


class Arm(RandomStateMixin, metaclass=abc.ABCMeta):
    """An arm for a Bandit (i.e. a probability distribution)."""
    def __init__(self, **kwargs):
        """Base constructor for a random arm.

        When subclassing `Arm`, one should call `super().__init__` to get
        access to the private `_random` RandomState.
        """
        super().__init__(**kwargs)
        self.kwargs = kwargs

    @abc.abstractmethod
    def draw(self) -> Union[int, float]:
        pass


class BernouilliArm(Arm):
    def __init__(self, p: Optional[float]=None, **kwargs):
        super().__init__(**kwargs)

        if p is None:
            # Generate a parameter uniformly in [0, 1]
            p = self._random.rand()
        self.p = p

    def draw(self) -> int:
        return int(self._random.rand() < self.p)


class Bandit:
    """A simple shell for bandit problems."""
    def __init__(self, arms: Sequence[Arm]):
        self.arms = arms

    @property
    def n_arms(self):
        return len(self.arms)

    @classmethod
    def generate(
        cls,
        n_arms: Optional[int]=None,
        arm_class: type=BernouilliArm,
        arm_kwargs: Optional[Sequence[Mapping]]=None
    ):
        """Generate a Bandit with required arms."""
        if arm_kwargs is None:
            assert n_arms is not None, (
                "If no `arm_kwargs` provided, one must set `n_arms`"
            )
            arm_kwargs = ({} for _ in range(n_arms))
        else:
            assert n_arms is None or n_arms == len(arm_kwargs), (
                "If both `arm_kwargs` and `n_arms` provided, they must be"
                "compatible (got {} and {})".format(len(arm_kwargs), n_arms)
            )

        return cls(arms=tuple(arm_class(**arm_kw) for arm_kw in arm_kwargs))

    def draw(self, arm: int) -> int:
        return self.arms[arm].draw()

    def seed(self, seed: Optional[Union[int, Sequence[int]]]=None):
        """Seed the random states of this Bandit arms."""
        if seed is None or isinstance(seed, int):
            for arm in self.arms:
                arm.seed(seed)
        else:
            n_seeds, n_arms = len(seed), len(self.arms)
            assert n_seeds == n_arms, (
                "If multiple seeds are provided (got {}), there must be as "
                "many as this Bandit has arms ({}).".format(n_seeds, n_arms))

            for seed, arm in zip(seed, self.arms):
                arm.seed(seed)
