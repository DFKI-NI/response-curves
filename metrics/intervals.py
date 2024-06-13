from abc import ABC
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Interval(ABC):
    """
    Interval class contains start and end time step as well as distances until detection, adaptation and response.
    """
    start: int
    end: int
    time_to_detection = np.NAN
    time_to_adaptation = np.NAN
    time_to_response = np.NAN

    def _set_time_to_response(self):
        """
        Set the time to response.
        """
        self.time_to_response = (self.time_to_detection + self.time_to_adaptation) / 2

    def set(self, time_to_detection, time_to_adaptation):
        """
        Set the interval's time to detection and time to adaptation.

        :param time_to_detection: the time to detection
        :param time_to_adaptation: the time to adaptation
        """
        self.time_to_detection = time_to_detection
        self.time_to_adaptation = time_to_adaptation
        self._set_time_to_response()


@dataclass
class GroundTruthInterval(Interval):
    """
    A class containing information about the concept drift.
    """
    next_start = np.inf
    detected: bool = False
    predictions: list = field(default_factory=list)

    def set(self, time_to_detection, time_to_adaptation):
        """
        Set the interval's time to detection and time to adaptation. By appending the given times to the ground truth's
        time lists. Must call finalize in the end.

        :param time_to_detection: the time to detection
        :param time_to_adaptation: the time to adaptation
        """
        if isinstance(self.time_to_detection, float):
            self.time_to_detection = []
        self.time_to_detection.append(time_to_detection)
        if isinstance(self.time_to_adaptation, float):
            self.time_to_adaptation = []
        self.time_to_adaptation.append(time_to_adaptation)
        self.detected = True

    def finalize(self):
        """
        Finalizes the ground truth interval by averaging the times to detection and times to adaptation.
        :return:
        """
        if self.detected:
            self.time_to_detection = np.mean(self.time_to_detection)
            self.time_to_adaptation = np.mean(self.time_to_adaptation)
            self._set_time_to_response()


@dataclass
class PredictionInterval(Interval):
    """
    A class containing information about the prediction.
    """
    true_positive: bool = False

    def set(self, time_to_detection, time_to_adaptation):
        super().set(time_to_detection, time_to_adaptation)
        self.true_positive = True


def match_intervals(
    ground_truth_intervals: list[GroundTruthInterval],
    prediction_intervals: list[PredictionInterval],
):
    """
    Matches the ground truth and predictions to determine true positives, false positives and false negatives.
    :param ground_truth_intervals: the ground truth
    :param prediction_intervals: the prediction
    """
    for ground_truth in ground_truth_intervals:
        for prediction in prediction_intervals:
            if ground_truth.start <= prediction.start < ground_truth.next_start:
                link_ground_truth_and_prediction(ground_truth, prediction)
        ground_truth.finalize()


def link_ground_truth_and_prediction(
    ground_truth: GroundTruthInterval, prediction: PredictionInterval
):
    """
    Links the ground truth and prediction if the ground truth was not detected yet or the prediction is interrupted.
    :param ground_truth: the ground truth
    :param prediction: the prediction
    """
    if not isinstance(ground_truth, GroundTruthInterval):
        raise TypeError("ground_truth must be of type GroundTruthInterval")
    if not isinstance(prediction, PredictionInterval):
        raise TypeError("prediction must be of type PredictionInterval")

    if (not ground_truth.detected and not prediction.true_positive) or (
        prediction.start < ground_truth.end
    ):
        time_to_detection = prediction.start - ground_truth.start
        time_to_adaptation = abs(prediction.end - ground_truth.end)
        ground_truth.set(time_to_detection, time_to_adaptation)
        ground_truth.predictions.append(prediction)
        prediction.set(time_to_detection, time_to_adaptation)


def link_ground_truths(ground_truths: list[GroundTruthInterval]):
    """
    Links the given ground truths by giving each ground truth the start time of the following one.

    :param ground_truths: the ground truths
    """
    for i in range(len(ground_truths) - 1):
        ground_truths[i].next_start = ground_truths[i + 1].start
