from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class Feature(ABC):
    """
    Abstract base class for synthetic parameters, which provide data via the `__iter__` function.
    """

    def __init__(
        self,
        min_: int or float,
        max_: int or float,
        seed=None,
    ):
        """
        Constructor

        :param min_: the minimum possible value
        :param max_: the maximum possible value
        :param seed: the seed
        """
        self.min = min_
        self.max = max_
        self.rng = np.random.default_rng(seed=seed)

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()


class UniformFeature(Feature):
    """
    A parameter that generates uniform values in the interval given by the minimum and maximum possible values.
    """

    def __iter__(self) -> Iterator[float]:
        """
        Generate this parameter.

        :return: a generator yielding uniformly distributed values
        """
        while True:
            value = self.rng.uniform(self.min, self.max)
            yield value
