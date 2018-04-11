from warnings import warn
from itertools import chain

import numpy as np


class CallableMapping:
    def __init__(self, func):
        self.func = func
        self._cache = {}

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]

        val = self.func(key[:-1], key[-1])  # split state and action
        self._cache[key] = val

        return val


def flatten(*args):
    """Combine two values (int or tuple) into a single flat tuple.

    assert combine(1, 2) == (1, 2)
    assert combine((1, 2), 3) == (1, 2, 3)
    assert combine(1, (2, 3)) == (1, 2, 3)
    assert combine((1,), 2) == (1, 2)
    assert combine((1, 2), (3, 4)) == (1, 2, 3, 4)
    assert combine(1, 2, 3, (4, 5)) == (1, 2, 3, 4, 5)
    """
    _vals = [(v,) if isinstance(v, int) else v for v in args]
    return tuple(chain(*_vals))


class ValueIteration:
    """Flexible implementation of the value iteration algorithm."""

    # Allowed keys for configuration of the algorithm
    CONFIG_KEYS = frozenset((
        'expected_reward',
        # Expected reward (no default): can be provided as a function of
        # state and action, or as a numpy array of same shape as the value
        # function, with an extra dimension for action (assume a non-negative
        # integer).
        'n_actions',
        # Number of actions (no default): may be inferred from expected_reward
        # if it was provided as a numpy array, as its last dimension.
        # Else, this value MUST be set.
        'transition',
        # Transition dynamics (no default): can be set as a numpy array of
        # shape (value_shape, n_actions, value_shape), or as a generator
        # function, taking a state and an action as arguments and yielding
        # non-zero probabilities with according follower state.
        'init_value',
        # Initial value for the value function (default: 0)
        'discount',
        # Discount for ahead of time rewards (default: 0.9)
        'synchronous_updates'
        # Whether to update the value function synchronously or not
        # (default: False) not synchronous, value is updated immediately
    ))

    def __init__(self, shape: 'Union[int, tuple]', **kwargs):
        self.value_shape = shape

        kwargs.setdefault('init_value', 0)
        kwargs.setdefault('discount', 0.9)
        kwargs.setdefault('synchronous_updates', False)
        self.configure(**kwargs)

    def configure(self, **kwargs):
        invalid_keys = set(kwargs) - self.CONFIG_KEYS
        assert not invalid_keys, (
            'Invalid keys: {}'.format(', '.join(invalid_keys))
        )

        for key, value in kwargs.items():
            setattr(self, key, value)

        if 'transition' not in kwargs:
            warn('Transition dynamics should be provided before running.')

        if not hasattr(self, 'expected_reward'):
            warn('`expected_reward` should be set before running.')
        elif callable(self.expected_reward) and \
                not hasattr(self, 'n_actions'):
            warn(
                '`expected_reward` was provided as a function, so you should'
                ' configure `n_actions` explicitly too.'
            )

    @property
    def expected_reward(self):
        return self._expected_reward

    @expected_reward.setter
    def expected_reward(self, value):
        if callable(value):
            self._expected_reward = CallableMapping(value)
        else:
            assert value.shape[:-1] == self.value_shape
            self._expected_reward = value
            self.n_actions = value.shape[-1]

    @property
    def transition(self):
        return self._transition

    @transition.setter
    def transition(self, value):
        if callable(value):
            self.transition_mode = 'generator'
        else:
            assert value.shape == flatten(
                self.value_shape, self.n_actions, self.value_shape)
            self.transition_mode = 'matrix'

        self._transition = value

    def run(self, threshold=None):
        values = self.init_value * np.ones(self.value_shape)

        if threshold is None:
            threshold = getattr(self, 'progression_threshold', 1e-3)

        delta = threshold * 2

        while delta >= threshold:
            delta = 0

            for state, value in self.iter_values(values):
                new_value = max(
                    self.get_update(state, action, values)
                    for action in self.iter_actions()
                )
                delta = max(delta, np.abs(value - new_value))
                values[state] = new_value

        return values

    def get_update(self, state, action, values):
        return self.expected_reward[flatten(state, action)] + sum(
            prob * values[next_state]
            for prob, next_state in self.get_next_state(state, action)
        )

    def get_next_state(self, state, action):
        assert hasattr(self, '_transition'), (
            'One of two transition dynamics descriptions must be '
            'configured.'
        )

        if self.transition_mode == 'matrix':
            for next_state, prob in self._iter_array(
                self.transition[flatten(state, action)]
            ):
                if prob == 0:
                    continue
                yield prob, next_state

        elif self.transition_mode == 'generator':
            return self.transition(state, action)

        else:
            warn('Should not get in this state: no transition dynamics.')

    def iter_actions(self):
        return range(self.n_actions)

    def iter_values(self, values):
        if self.synchronous_updates:
            _vals = values.copy()
        else:
            _vals = values

        return self._iter_array(_vals)

    @staticmethod
    def _iter_array(array):
        iterator = np.nditer(array, flags=['multi_index'])
        for v in iterator:
            yield iterator.multi_index, v


def check_direction(func):
    def wrapper(state, direction):
        assert direction in ('s', 'f'), (
            'Invalid direction: {}'.format(direction)
        )
        return func(state, direction)
    return wrapper


class MDPDef:
    def __init__(self, n_steps: int=100, M=None):
        # We first set the total number of steps to compute
        self.n_steps = n_steps

        # We store the retirement option, if provided
        self.M = M

        # If there M is provided, we have 2 available actions
        self.n_actions = 2 if M is not None else 1

    @staticmethod
    @check_direction
    def _next(state, direction):
        if direction == 's':
            return (state[0] + 1, state[1])

        return (state[0], state[1] + 1)

    @staticmethod
    @check_direction
    def _get_prob(state, direction):
        ns, nf = state[0] + 1, state[1] + 1
        if direction == 's':
            return ns / (ns + nf)

        return nf / (ns + nf)

    def transition_gen(self, state, action):
        if action == 1:
            # Retirement option: no following state
            return

        if sum(state) == self.n_steps:
            # check extremities
            return

        for direction in ('s', 'f'):
            yield (
                self._get_prob(state, direction), self._next(state, direction)
            )

    def expected_reward(self, state, action):
        if action == 1:
            return self.M

        return self._get_prob(state, 's')

    def unpack(self):
        return {
            'shape': (self.n_steps, self.n_steps),
            'transition': self.transition_gen,
            'expected_reward': self.expected_reward,
            'n_actions': self.n_actions
        }