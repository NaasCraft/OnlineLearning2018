import abc
from typing import Tuple

import numpy as np

from project import bandit
from project.utils import RandomStateMixin


class Policy(RandomStateMixin, metaclass=abc.ABCMeta):
    """Policy base-class for interacting with the Bandit environment."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(
        self, bandit: bandit.Bandit, n_steps: int, **kwargs
    ) -> Tuple[int, float]:
        self.prepare(n_arms=bandit.n_arms, n_steps=n_steps)

        action, reward = np.zeros(n_steps), np.zeros(n_steps)

        for step in range(n_steps):
            arm = self.pick()
            action[step] = arm
            reward[step] = bandit.draw(arm)

        return action, reward

    @abc.abstractmethod
    def prepare(self, **kwargs):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def pick(self) -> int:
        pass


class RandomPolicy(Policy):
    """Random (uniform) pick policy."""
    def prepare(self, n_arms: int, **kwargs):
        self.n_arms = n_arms

    def reset(self):
        pass

    def pick(self) -> int:
        return self._random.randint(self.n_arms)
