"""This module provides the INSECTS drifting data streams."""
from os import path

from river import stream
from river.datasets import base


class InsectsAbruptBalanced(base.FileDataset):
    """
    This class provides the abrupt balanced INSECTS datastream.

    References
    ----------
    [^1]: [Challenges in benchmark stream learning algorithms with real-world
    data](https://doi.org/10.1007/s10618-020-00698-5)
    """

    def __init__(
        self,
        name: str = "",
        directory_path: str = "data",
        **desc,
    ):
        """
        Init a new InsectsAbruptBalanced instance
        :param name: the name of the dataset to be used in plotting
        :param directory_path: the directory containing the csv file
        :param desc: kwargs
        """
        desc.update(
            {
                "n_samples": 52848,
                "n_features": 33,
                "task": base.MULTI_CLF,
                "filename": "INSECTS-abrupt_balanced_norm.csv",
            }
        )
        super().__init__(**desc)
        self.name = name
        self.full_path = ""
        self.drifts = [14_352, 19_500, 33_240, 38_682, 39_510]
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        converters = {f"Att{i}": float for i in range(1, 34)}
        converters["class"] = str
        """return stream.iter_csv(
            self.full_path,
            target="class",
            converters=converters,
        )"""
        for i, (x, y) in enumerate(
            stream.iter_csv(
                self.full_path,
                target="class",
                converters=converters,
            )
        ):
            yield x, y, i in self.drifts
