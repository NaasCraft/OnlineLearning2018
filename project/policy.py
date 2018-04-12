import abc
from multiprocessing import Pool
import os
import re
from typing import Union, Tuple, Optional

import numpy as np

from project import bandit
from project import value_iteration as vi
from project.utils import RandomStateMixin, timeit


class BayesianBernouilliMixin:
    """Mixin used to track arms belief state."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._arm_prior = kwargs.get('arm_prior', (0, 0))

    @staticmethod
    def _next_state(state: Tuple[int, int], direction: str) -> Tuple[int, int]:
        assert direction in ('s', 'f'), 'Invalid direction'

        if direction == 's':
            return (state[0] + 1, state[1])
        return (state[0], state[1] + 1)

    def prepare(self, n_arms: int, **kwargs):
        self._arms_state = [self._arm_prior] * n_arms
        super().prepare(n_arms=n_arms, **kwargs)

    def receive(self, arm, reward):
        self._arms_state[arm] = self._next_state(
            state=self._arms_state[arm],
            direction='s' if reward else 'f'
        )
        super().receive(arm, reward)

    @property
    def states(self):
        return self._arms_state


BBMixin = BayesianBernouilliMixin


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


class UCBPolicy(BBMixin, Policy):
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
        super().prepare(n_arms=n_arms, **kwargs)
        self._counts = [0] * n_arms

    def pick(self):
        best_arm = np.argmax(self._get_arm_values())

        self._counts[best_arm] += 1
        return best_arm

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
            for arm, state in enumerate(self._arms_state)
        ])

    @staticmethod
    def get_prob(state):
        """success / (success + failure)"""
        _state = (state[0] + 1, state[1] + 1)
        if not all(_state):
            raise ValueError('Invalid state: {}'.format(_state))

        return _state[0] / sum(_state)

    def get_confidence_bound(self, arm):
        """delta * sqrt( log(T) / T(i) )"""
        return self.delta * np.sqrt(
            np.log(np.sum(self._counts)) / (self._counts[arm])
        )


class GittinsIndexPolicy(BBMixin, Policy):
    """Policy based on pre-computed approximations of the Gittins index."""

    STORAGE_FOLDER = 'saved_gittins'
    FNAME_REGEX = re.compile(
        r'(?P<filename>'
        r'gittins_(?P<n_steps>\d+\d)_(?P<discount>\d{2})\.(?P<extension>npy)'
        r')'
    )

    def __init__(self, n_steps: int, discount: float, **kwargs):
        """Load the pre-computed Gittins index, checking required values.

        Stored values are under `saved_gittins/`, with a filename as such:

            gittins_{n_steps}_{discount * 100}.npy
        """
        super().__init__(**kwargs)
        filename = 'gittins_{}_{}.npy'.format(n_steps, int(discount * 100))
        full_path = os.path.join(self.STORAGE_FOLDER, filename)

        assert os.path.isfile(full_path), (
            "No pre-computed index exists for (n_steps={0}, "
            "discount={1:.2f}). Available files are:\n{2}"
        ).format(
            n_steps,
            discount,
            self.get_stored_files('\t{filename}\n')
        )

        self.gittins_index = np.load(full_path)

    @classmethod
    def get_stored_files(cls, fmt_string=None):
        files = [
            cls.FNAME_REGEX.search(fname)
            for fname in os.listdir(cls.STORAGE_FOLDER)
        ]

        if fmt_string is None:
            return [file.string for file in files if file is not None]

        full_string = ''
        for file in files:
            if file is None:
                # Not a file of interest
                continue

            kwds = dict(
                file.groupdict(),
                fullpath=os.path.join(cls.STORAGE_FOLDER, file.string)
            )
            full_string += fmt_string.format(**kwds)

        return full_string

    def pick(self):
        return np.argmax([
            self.gittins_index[state]
            for state in self._arms_state
        ])


class VIPolicy(BBMixin, Policy):
    """Greedy policy from value iteration approximation result."""
    _cached_values = {}

    def __init__(self, discount=0.9, **kwargs):
        super().__init__(discount=discount, **kwargs)
        self.discount = discount

    @staticmethod
    def _compute_value(n_steps, discount):
        _def = vi.MDPDef(n_steps=n_steps)
        VI = vi.ValueIteration(discount=discount, **_def.unpack())
        return VI.run()

    @property
    def values(self):
        args = (self._n_steps, self.discount)
        values = self._cached_values.get(args, None)

        if values is None:
            values = self._compute_value(*args)
            self._cached_values[args] = values

        return values

    def prepare(self, n_steps, **kwargs):
        super().prepare(n_steps=n_steps, **kwargs)
        self._n_steps = n_steps

    def pick(self):
        values = [self.values[state] for state in self.states]
        best_arm = np.argmax(values)
        return best_arm


class EpsilonVIPolicy(VIPolicy):
    """Epsilon-greedy variant."""
    def __init__(self, epsilon=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def pick(self):
        if self._random.rand() <= self.epsilon:
            return self._random.randint(len(self.states))

        return super().pick()


class GittinsVIPolicy(BBMixin, Policy):
    FOLDER = 'saved_gittins'
    FNAME_REGEX = re.compile(
        r'gittins_vi_'
        r'(?P<n_steps>\d+\d)_(?P<precision>\d+)_(?P<discount>\d{2})'
        r'\.npy'
    )
    # NOTE: Precision is expressed as the number of digits to keep
    #       for the index values.
    DEFAULT_PRECISION = 3
    DEFAULT_DISCOUNT = 0.9
    DEBUG = True

    def __init__(self, **kwargs):
        self.precision = kwargs.get('precision', self.DEFAULT_PRECISION)
        self.discount = kwargs.get('discount', self.DEFAULT_DISCOUNT)
        self.parallel = kwargs.get('parallel', False)
        if self.parallel and isinstance(self.parallel, int):
            self.pool = Pool(self.parallel)
        elif self.parallel:
            self.pool = Pool()

    def prepare(self, n_steps: int, **kwargs):
        super().prepare(n_steps=n_steps, **kwargs)
        self._GI = self._get_or_compute_index(
            n_steps, self.precision, self.discount)

    def pick(self):
        return np.argmax([self._GI[state] for state in self._arms_state])

    @classmethod
    def _iter_saved_files(cls):
        for file in os.listdir(cls.FOLDER):
            s = cls.FNAME_REGEX.search(file)
            if s is None:
                continue
            yield (
                file,
                int(s.group('n_steps')),
                int(s.group('precision')),
                round(int(s.group('discount')) / 100, 2),
            )

    @classmethod
    def _get_or_compute_index(
        cls, n_steps: int, precision: int, discount: float,
        n_jobs: Optional[int]=None
    ):
        for _file, _steps, _prec, _disc in cls._iter_saved_files():
            valid_file = (
                _steps >= n_steps and _prec >= precision
                and abs(_disc - discount) < 1e-3
            )
            if valid_file:
                file = _file
                break
        else:
            file = None

        if file is not None:
            return np.load(os.path.join(cls.FOLDER, file))

        print(
            "Computing index for (n_steps={}, precision={}, discount={:.2f})"
            .format(n_steps, precision, discount)
        )
        with timeit('Index computation', debug=cls.DEBUG):
            index = cls._compute_index(n_steps, precision, discount, n_jobs)

        fullpath = os.path.join(
            cls.FOLDER,
            'gittins_vi_{}_{}_{}.npy'.format(
                n_steps, precision, int(discount * 100))
        )
        print("Saving to '{}'".format(fullpath))
        np.save(fullpath, index)

        return index

    @classmethod
    def _compute_index(
        cls, n_steps: int, precision: int, discount: float,
        n_jobs: Optional[int]=None
    ):
        step_size = 10 ** -precision
        M_vals = np.arange(0, 1, step_size)

        values = [0] * len(M_vals)

        if n_jobs:
            _pool = Pool(n_jobs)
            _results = []

        for i, M in enumerate(M_vals):
            definition = vi.MDPDef(n_steps=n_steps, M=M)
            VI = vi.ValueIteration(discount=discount, **definition.unpack())

            if n_jobs:
                _results.append(_pool.apply_async(VI.run))
            else:
                values[i] = VI.run()

        if n_jobs:
            for i, res in enumerate(_results):
                values[i] = res.get()

        gittins_index = np.zeros((n_steps, n_steps))
        for ns in range(n_steps):
            for nf in range(n_steps):
                # NOTE: M is a candidate if value(M) = M
                M_candidates = [
                    M for val, M in zip(values, M_vals)
                    if np.abs(M - val[ns, nf]) < step_size
                ]
                # NOTE: the (1 - discount) factor does not yield the correct
                #       results, probably because of time indexing.
                gittins_index[ns, nf] = min(M_candidates)

        return gittins_index


if __name__ == '__main__':
    pass
