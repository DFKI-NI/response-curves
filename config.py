import numpy as np
from river.drift.binary import DDM
from river.drift import ADWIN

from data.concept import Concept
from data.concept_drift import IncrementalDrift, LinearTransition, AbruptDrift
from data.features import UniformFeature
from data.insects import InsectsAbruptBalanced
from data.stream import Stream
from detectors import *
from optimization.model_optimizer import ModelOptimizer, SupervisedModelOptimizer
from optimization.parameter import Parameter


N_SAMPLES = 3_0_000


class D3Configuration:
    # EXPERIMENT 2: response curve showing impact of sample size
    #             & response curve showing impact of threshold
    seed = 112716392
    stream = Stream(
        name=r"Response Curves for different configurations of D3 (abrupt drift)",
        min_len=N_SAMPLES,
        concepts=[
            Concept(features=[UniformFeature(0, 1, seed=seed)]),
            Concept(features=[UniformFeature(10, 11, seed=seed)]),
        ],
        concept_drift=IncrementalDrift(
            transition=LinearTransition(min_length=1, max_length=25, seed=seed)
        ),
        concept_min_len=1000,
        concept_max_len=1500,
        seed=seed,
    )
    streams = [stream]
    models = [
        ModelOptimizer(
            name="D3",
            base_model=DiscriminativeDriftDetector2019,
            parameters=[
                Parameter("n_reference_samples", values=[100, 250]),
                Parameter("threshold", values=[0.55, 0.8]),
            ],
            seeds=np.arange(20 * 4) + seed,
            n_runs=20,
        ),
    ]


class BNDMvsD3Configuration:
    # EXPERIMENT 1: response curve of D3 vs BNDM
    seed = 122716391
    stream = Stream(
        name=r"Response Curves for BNDM and D3 (abrupt drift)",
        min_len=N_SAMPLES,
        concepts=[
            Concept(features=[UniformFeature(0, 1, seed=seed)]),
            Concept(features=[UniformFeature(10, 11, seed=seed)]),
        ],
        concept_drift=IncrementalDrift(
            transition=LinearTransition(min_length=1, max_length=25, seed=seed)
        ),
        concept_min_len=1000,
        concept_max_len=1500,
        seed=seed,
    )
    streams = [stream]
    models = [
        ModelOptimizer(
            name="D3",
            base_model=DiscriminativeDriftDetector2019,
            parameters=[
                Parameter("n_reference_samples", values=[100]),
            ],
            seeds=np.arange(20) + seed,
            n_runs=20,
        ),
        ModelOptimizer(
            name="BNDM",
            base_model=BayesianNonparametricDetectionMethod,
            parameters=[
                Parameter("n_samples", values=[100]),
            ],
            seeds=None,
            n_runs=1,
        ),
    ]


class AbruptConfiguration:
    seed = 132716394
    stream = Stream(
        name=r"Response Curves for different step sizes of D3 (abrupt drift)",
        min_len=N_SAMPLES,
        concepts=[
            Concept(features=[UniformFeature(0, 1, seed=seed)]),
            Concept(features=[UniformFeature(10, 11, seed=seed)]),
        ],
        concept_drift=AbruptDrift(),
        concept_min_len=1000,
        concept_max_len=1500,
        seed=seed,
    )
    streams = [stream]
    models = [
        ModelOptimizer(
            name="D3",
            base_model=DiscriminativeDriftDetector2019,
            parameters=[
                Parameter("n_reference_samples", values=[100]),
                Parameter("step", values=["step half", "step 1"]),
            ],
            seeds=np.arange(20 * 2) + seed,
            n_runs=20,
        ),
    ]


class IncrementalConfiguration:
    # EXPERIMENT 5: reset on incremental
    seed = 142716395
    stream = Stream(
        name=r"Response Curves for different reset modes of BNDM (incremental drift)",
        min_len=N_SAMPLES,
        concepts=[
            Concept(features=[UniformFeature(0, 1, seed=seed)]),
            Concept(features=[UniformFeature(10, 11, seed=seed)]),
        ],
        concept_drift=IncrementalDrift(
            transition=LinearTransition(min_length=100, max_length=500, seed=seed),
        ),
        concept_min_len=1000,
        concept_max_len=1500,
        seed=seed,
    )
    streams = [stream]
    models = [
        ModelOptimizer(
            name="BNDM",
            base_model=BayesianNonparametricDetectionMethod,
            parameters=[
                Parameter("n_samples", values=[100]),
                Parameter(
                    "reset_mode", values=["full reset", "half reset", "no reset"]
                ),
            ],
            seeds=None,
            n_runs=1,
        ),
    ]


class InsectsConfiguration:
    # EXPERIMENT 6: insects with DDM, ADWIN, D3, BNDM all in default configurations
    seed = 152716396
    streams = [
        InsectsAbruptBalanced(
            name="Response Curves for ADWIN, BNDM, D3 and DDM (INSECTS abrupt-bal.)",
            directory_path="data",
        ),
    ]
    models = [
        SupervisedModelOptimizer(
            name="DDM",
            base_model=DDM,
            parameters=[
                Parameter("drift_threshold", value=3),
            ],
            seeds=None,
            n_runs=1,
        ),
        SupervisedModelOptimizer(
            name="ADWIN",
            base_model=ADWIN,
            parameters=[Parameter("delta", value=0.002)],
            seeds=None,
            n_runs=1,
        ),
        ModelOptimizer(
            name="D3",
            base_model=DiscriminativeDriftDetector2019,
            parameters=[
                Parameter("n_reference_samples", values=[250]),
                Parameter("threshold", values=[0.7]),
            ],
            seeds=np.arange(20) + seed,
            n_runs=20,
        ),
        ModelOptimizer(
            name="BNDM",
            base_model=BayesianNonparametricDetectionMethod,
            parameters=[
                Parameter("n_samples", values=[250]),
                Parameter("threshold", values=[0.6]),
            ],
            seeds=None,
            n_runs=1,
        ),
    ]
