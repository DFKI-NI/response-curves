from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class Transition(ABC):
    """
    Abstract base class Transition for non-abrupt concept drift.
    """

    def __init__(
        self,
        min_length: int,
        max_length: int,
        seed: int = None,
    ):
        """
        Abstract base constructor for transitions

        :param min_length: the minimum length of the transition
        :param max_length: the maximum length of the transition
        :param seed: the random seed
        """
        self.min_length = min_length
        self.max_length = max_length
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def __iter__(self):
        """
        Iterate over the transition factors.

        Abstract method, must be implemented by inheriting class.
        """
        raise NotImplementedError


class LinearTransition(Transition):
    """
    LinearTransition provides a sequence of linearly spaced transition factors for concept drifts.
    """

    def __iter__(self):
        """
        Iterate over the transition factors
        """
        length = self.rng.integers(self.min_length, self.max_length + 1)
        # we use length + 2 because 0 and 1 are included in the output of this linspace
        transition_factors = np.linspace(0, 1, length + 2)
        # remove leading 0 and trailing 1 from transition factors
        transition_factors = transition_factors[1:-1]
        yield from transition_factors


class ConceptDrift(ABC, Iterable):
    """
    Abstract base class Transition for non-abrupt concept drift.
    """

    def __init__(
        self,
    ):
        """
        Abstract base constructor for concept drift
        """
        self.old_concept = None
        self.new_concept = None

    def start(self, old_concept, new_concept):
        """
        Start a new concept drift.

        :param old_concept: the old concept
        :param new_concept: the new concept
        """
        self.old_concept = old_concept
        self.new_concept = new_concept

    @abstractmethod
    def __iter__(self):
        """
        Iterate over data during the concept drift. Terminates after the concept drift concluded.

        Must be implemented in inheriting class.
        """
        raise NotImplementedError("Must implement __iter__")


class AbruptDrift(ConceptDrift):
    """
    AbruptDrift provdes an abrupt concept drift by immediately yielding data from the new concept.
    """

    def __iter__(self):
        yield next(iter(self.new_concept))


class IncrementalDrift(ConceptDrift):
    """
    IncrementalDrift performs an incremental concept drift with the given concepts and transition.
    """

    def __init__(
        self,
        transition: Transition,
    ):
        """
        Init a new IncrementalDrift instance

        :param transition: the transition used during the concept drifts
        """
        super().__init__()
        self.transition = transition

    @staticmethod
    def _get_features(transition_factor: float, old_features: np.ndarray, new_features: np.ndarray) -> np.ndarray:
        """
        Get features during a concept drift by combining the old features and new features according to the current
        transition factor.

        :param transition_factor: the transition factor used to weigh the old and new features
        :param old_features: the features of the old concept
        :param new_features: the features of the new concept
        :return: the features
        """
        old_features = old_features * (1 - transition_factor)
        new_features = new_features * transition_factor
        features = old_features + new_features
        return features

    def __iter__(self):
        for factor, old_features, new_features in zip(
            self.transition, self.old_concept, self.new_concept
        ):
            features = self._get_features(factor, old_features, new_features)
            yield features
