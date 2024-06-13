from collections import defaultdict
from typing import Iterable, Optional

from river.tree import HoeffdingTreeClassifier

from metrics.eval import evaluate
from .config_generator import ConfigGenerator
from .parameter import Parameter


class ModelOptimizer:
    """
    ModelOptimizer provides methods to test different configurations of a given unsupervised concept drift detector.
    """

    def __init__(
        self,
        base_model: callable,
        parameters: list[Parameter],
        n_runs: int,
        name: str = "",
        seeds: Optional[Iterable] = None,
    ):
        """
        Init a new ModelOptimizer.

        :param base_model: a callable of the detector under test
        :param parameters: the configuration parameters
        :param n_runs: the number of test runs for each configuration
        :param name: the name of the model under test
        :param seeds: the seeds or None
        """
        self.base_model = base_model
        self.configs = ConfigGenerator(parameters, seeds=seeds)
        self.n_runs = n_runs
        self.name = name

    def _model_generator(self):
        """
        A generator that yields initialized models using configurations provided by the ConfigGenerator.

        :return: the initialized models
        """
        for config in self.configs:
            yield self.base_model(**config), config

    def optimize(self, stream, experiment_name, verbose=False) -> dict[str, list]:
        """
        Optimize the model on the given data stream and log the results using the ExperimentLogger.

        :param stream: the data stream
        :param experiment_name: the name of the experiment
        :param verbose: print the currently optimized model and its config
        :return: a dict containing the predictive results
        """
        results = defaultdict(list)
        for run in range(self.n_runs):
            for model, config in self._model_generator():
                if verbose:
                    print(f"{model}: {config}")
                ground_truth = []
                predictions = []
                for x, y, drift in stream:
                    prediction = model.update(x)
                    predictions.append(prediction)
                    ground_truth.append(drift)
                metrics = evaluate(ground_truth, predictions)
                results[self._config_to_string(config)].append(metrics)
        return results

    @staticmethod
    def _config_to_string(config):
        """
        Converts the given configuration to a string without the seed for easier plotting and tracking of multiple
        configurations.

        :param config: the config
        :return: a string representation of the config without the seed
        """
        string = ", ".join(f"{value}" for key, value in config.items() if key != "seed")
        return string


class SupervisedModelOptimizer(ModelOptimizer):
    """
    SupervisedModelOptimizer provides methods to test different configurations of a given supervised concept drift
    detector.
    """

    def __init__(
        self,
        base_model: callable,
        parameters: list[Parameter],
        n_runs: int,
        name: str = "",
        seeds: Optional[list[int]] = None,
    ):
        """
        Init a new SupervisedModelOptimizer instance

        :param base_model: a callable of the detector under test
        :param parameters: the configuration parameters
        :param n_runs: the number of test runs for each configuration
        :param name: the name of the model under test
        :param seeds: the seeds or None
        """
        super().__init__(
            base_model=base_model,
            parameters=parameters,
            n_runs=n_runs,
            name=name,
            seeds=seeds,
        )
        self.classifier = None

    def _model_generator(self):
        """
        A generator that yields initialized models using configurations provided by the ConfigGenerator.

        :return: the initialized models
        """
        for config in self.configs:
            config.pop("seed")
            yield self.base_model(**config), config

    def optimize(self, stream, experiment_name, verbose=False) -> dict[str, list]:
        """
        Optimize the model on the given data stream

        :param stream: the data stream
        :param experiment_name: the name of the experiment
        :param verbose: print the currently optimized model and its config
        :return: a dict containing the predictive results
        """
        results = defaultdict(list)
        for run in range(self.n_runs):
            for model, config in self._model_generator():
                self.classifier = HoeffdingTreeClassifier()
                if verbose:
                    print(f"{model}: {config}")
                ground_truth = []
                predictions = []
                for x, y, drift in stream:
                    y_pred = self.classifier.predict_one(x)
                    model.update(y_pred != y)
                    if model.drift_detected:
                        self.classifier = HoeffdingTreeClassifier()
                    predictions.append(model.drift_detected)
                    ground_truth.append(drift)
                    self.classifier.learn_one(x, y)
                metrics = evaluate(ground_truth, predictions)
                results[self._config_to_string(config)].append(metrics)
        return results
