import numpy as np

from .features import Feature


class Concept:
    """
    The Concept contains different features representing a stationary period in a data stream.
    """

    def __init__(
        self,
        features: list[Feature],
    ):
        """
        Init a new Concept instance

        :param features: the features
        """
        self.features = features

    def __iter__(self):
        """
        Iterate over data from the features. Does not terminate unless the features do.
        """
        for values in zip(*self.features):
            values = np.array(values)
            yield values
