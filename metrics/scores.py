import math
import numpy as np


def get_f1_scores(ground_truth_intervals, prediction_intervals) -> list:
    """
    Get the F1 scores for the given ground truth and prediction by evaluating lim Delta_max -> inf.

    :param ground_truth_intervals: the ground truth
    :param prediction_intervals: the prediction
    :return: a list of F1 scores
    """
    ground_truth_response_times = np.array(
        [interval.time_to_response for interval in ground_truth_intervals]
    )
    delta_max = math.ceil(np.nanmax(ground_truth_response_times))
    scores = []
    for delta in range(delta_max + 1):
        true_positives = sum(ground_truth_response_times <= delta)
        false_positives = get_false_positives(
            ground_truth_intervals, prediction_intervals, delta
        )
        false_negatives = len(ground_truth_intervals) - true_positives
        recall = true_positives / (true_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives)
        if recall + precision > 0:
            f1_score = 2 * (recall * precision) / (recall + precision)
        else:
            f1_score = np.nan
        scores.append(f1_score)
    return scores


def get_false_positives(ground_truth_intervals, prediction_intervals, delta):
    correct_predictions = []
    for ground_truth in ground_truth_intervals:
        if ground_truth.time_to_response <= delta:
            correct_predictions += ground_truth.predictions
    false_positives = len(prediction_intervals) - len(correct_predictions)
    return false_positives
