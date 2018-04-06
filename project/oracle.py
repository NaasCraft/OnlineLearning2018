import collections
from typing import List, Sequence, Optional

import numpy as np

from project.policy import Policy
from project.bandit import Bandit


class Oracle:
    """Central class responsible for generating and evaluating Bandit problems.

    Receives a set of `Policy` objects to benchmark against some Bandit problem
    that was generated using this `Oracle`, or provided by hand.
    """
    Report = collections.namedtuple('OracleReport', (
        "id", "parameters", "results"
    ))

    _past_reports: List[Report] = []

    def __init__(self, policies: Sequence[Policy]):
        self.policies = policies

    @classmethod
    def generate(cls):
        """Oracle object factory.

        Generate policies from a sequence of classes or tuples
        (class, init_kwargs).
        """
        pass

    def generate_bandit(self, **kwargs) -> Bandit:
        self.bandit = Bandit.generate(**kwargs)
        return self.bandit

    def set_bandit(self, bandit: Bandit):
        self.bandit = bandit

    def evaluate(
        self,
        n_runs: int=10,
        run_length: int=100,
        seed: Optional[int]=None,
        **kwargs
    ) -> Report:
        """<TBD>

        Evaluate a set of policies against a bandit problem.
        """
        parameters = {
            'n_runs': n_runs, 'run_length': run_length,
            'seed': None,
            'kwargs': kwargs
        }

        actions = np.zeros((n_runs, len(self.policies), run_length))
        rewards = np.zeros((n_runs, len(self.policies), run_length))

        for run in range(n_runs):
            for i, policy in enumerate(self.policies):
                self.bandit.seed(seed)
                actions[run, i], rewards[run, i] = policy.run(
                    bandit=self.bandit, n_steps=run_length,
                    **kwargs
                )

        report = self.Report(
            id=len(self._past_reports),
            parameters=parameters,
            results=rewards
        )
        self._past_reports.append(report)
        return report
