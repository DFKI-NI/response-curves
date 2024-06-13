import numpy as np

from metrics.intervals import (
    GroundTruthInterval,
    PredictionInterval,
    link_ground_truths,
)


def ground_truth_to_time_stamps(ground_truth: list[bool]) -> list[GroundTruthInterval]:
    time_stamps = _get_change_time_stamps(ground_truth)
    results = []
    for i in range(len(time_stamps) // 2):
        start = time_stamps[2 * i]
        end = time_stamps[2 * i + 1]
        results.append(GroundTruthInterval(start=start, end=end))
    link_ground_truths(results)
    return results


def prediction_to_time_stamps(prediction: list[bool]) -> list[PredictionInterval]:
    time_stamps = _get_change_time_stamps(prediction)
    results = []
    for i in range(len(time_stamps) // 2):
        start = time_stamps[2 * i]
        end = time_stamps[2 * i + 1]
        results.append(PredictionInterval(start=start, end=end))
    return results


def _get_change_time_stamps(set_: list[bool]):
    set_ = np.array(set_)
    changes = set_[1:] != set_[:-1]
    time_stamps = list(
        np.where(changes == True)[0] + 1
    )  # offset 1 due to prior slicing
    return time_stamps
