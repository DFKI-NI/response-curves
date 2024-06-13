from metrics.intervals import match_intervals
from metrics.utils import ground_truth_to_time_stamps, prediction_to_time_stamps
from metrics.scores import get_f1_scores


def evaluate(ground_truth, predictions) -> list:
    """
    Evaluate the concept drift.

    :param ground_truth: the ground truth of the concept drift
    :param predictions: the predictions given by the concept drift detector
    :return: the delta max scores of the detection
    """
    ground_truth_intervals = ground_truth_to_time_stamps(ground_truth)
    prediction_intervals = prediction_to_time_stamps(predictions)
    match_intervals(ground_truth_intervals, prediction_intervals)
    scores = get_f1_scores(ground_truth_intervals, prediction_intervals)
    return scores
