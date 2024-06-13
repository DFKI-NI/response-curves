import copy
from collections import defaultdict

from plot.response_curves import plot_response_curves


def run(experiment_name, config):
    """
    Run the experiment with the given config.

    :param experiment_name: the name of the experiment
    :param config: the config of the experiment
    """
    print(f"Running experiment {experiment_name}")
    for base_stream in config.streams:
        stream_results = defaultdict(dict)
        for model in config.models:
            stream = copy.copy(base_stream)
            model_results = model.optimize(stream, experiment_name, verbose=True)
            stream_results[model.name].update(model_results)
        plot_response_curves(stream_results, base_stream.name)
