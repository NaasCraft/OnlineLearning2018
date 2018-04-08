import abc
import collections
from typing import Union, Tuple

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
            self.receive(arm, reward[step])

        return action, reward

    @abc.abstractmethod
    def prepare(self, **kwargs):
        """Initialize internal state variables for a full run."""
        pass

    @abc.abstractmethod
    def pick(self) -> int:
        """Choose an arm according to previously observed results."""
        pass

    @abc.abstractmethod
    def receive(self, arm: int, reward: 'Union[int, float]'):
        """Update internal state variables according to received reward."""
        pass


class RandomPolicy(Policy):
    """Random (uniform) pick policy."""
    def prepare(self, n_arms: int, **kwargs):
        self.n_arms = n_arms

    def pick(self) -> int:
        return self._random.randint(self.n_arms)

    def receive(self, arm: int, reward: Union[int, float]):
        pass


class UCBPolicy(Policy):
    """Implementation of the Upper Confidence Bound algorithm."""
    def __init__(
        self,
        delta: float,
        arm_prior: 'Tuple[int, int]'=(1, 1),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.delta = delta
        self.arm_prior = arm_prior

    def prepare(self, n_arms: int, **kwargs):
        self._counts = [0] * n_arms
        self._states = [self.arm_prior] * n_arms

    def pick(self):
        best_arm = np.argmax(self._get_arm_values())

        self._counts[best_arm] += 1
        return best_arm

    def receive(self, arm, reward):
        last_state = self._states[arm]

        if reward == 1:
            self._states[arm] = (last_state[0] + 1, last_state[1])
        else:
            self._states[arm] = (last_state[0], last_state[1] + 1)

    def _get_arm_values(self):
        """Return the array of values attributed to each arm.

        If any of the arms was not picked yet, the values will be 1 for any
        unpicked arm and 0 for the others.
        When all arms were once drawn, an arm value has the shape:

            s / (s + f)  +  delta * sqrt( log(T) / T(i) )

        where T(i) is the number of times the arm i was picked until now.
        """
        if any(c == 0 for c in self._counts):
            # Test all arms first
            return np.array([
                1 if c == 0 else 0
                for c in self._counts
            ])

        return np.array([
            self.get_prob(state) + self.get_confidence_bound(arm)
            for arm, state in enumerate(self._states)
        ])

    @staticmethod
    def get_prob(state):
        """success / (success + failure)"""
        if not all(state):
            raise ValueError(state)

        return state[0] / sum(state)

    def get_confidence_bound(self, arm):
        """delta * sqrt( log(T) / T(i) )"""
        return self.delta * np.sqrt(
            np.log(np.sum(self._counts)) / (self._counts[arm])
        )


class GittinsIndexPolicy(Policy):
    """Gittins Index based policy.

    Uses a pre-computed, lazily evaluated Gittins Index.
    """
    _cached_index = collections.defaultdict(None)

    def prepare(
        self,
        n_arms: int,
        n_steps: int,
        discount: float,
        **kwargs
    ):
        self.n_arms = n_arms

        self.cache_index(n_steps, discount)

    def reset(self):
        pass

    def pick(self) -> int:
        pass

    def cache_index(self, n_steps: int, discount: float):
        if self._cached_index[discount] is None:
            # todo
            return

        # todo: check if required size is cached
