import time
from itertools import islice

import numpy as np

from data.concept import Concept
from data.concept_drift import ConceptDrift


class Stream:
    """
    A class providing a Stream. It offers a wrapper around concepts and concept drifts, alternating between them during
    data generation.
    """

    def __init__(
        self,
        concepts: list[Concept],
        concept_min_len: int,
        concept_max_len: int,
        concept_drift: ConceptDrift,
        min_len: int,
        name: str = None,
        seed=None,
    ):
        """
        Init a new Stream instance

        :param concepts: the concepts
        :param concept_min_len: the minimum duration of a concept
        :param concept_max_len: the maximum duration of a concept
        :param concept_drift: the concept drift
        :param min_len: the minimum length of the stream
        :param name: the name of the stream to be used in plotting
        :param seed: the random seed
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.concepts = concepts
        self.active_concept = self.rng.choice(self.concepts, 1)[0]
        self.concept_min_len = concept_min_len
        self.concept_max_len = concept_max_len
        self.concept_drift = concept_drift
        self.min_len = min_len
        self.name = name

    def __iter__(self):
        """
        Generate data from the concepts and concept drift.
        """
        i = 0
        while True:
            concept_len = self.rng.integers(
                self.concept_min_len, self.concept_max_len + 1
            )
            for x in islice(self.active_concept, concept_len):
                i += 1
                yield x, None, False
            new_concept = self._get_new_concept()
            if i >= self.min_len:
                break
            self.concept_drift.start(self.active_concept, new_concept)
            for x in self.concept_drift:
                i += 1
                yield x, None, True
            self.active_concept = new_concept

    def _get_new_concept(self):
        """
        Get a new concept, which will be different from the current concept.

        :return: the new concept
        """
        new_concept = self.rng.choice(self.concepts, 1)[0]
        if new_concept == self.active_concept:
            return self._get_new_concept()
        else:
            return new_concept
