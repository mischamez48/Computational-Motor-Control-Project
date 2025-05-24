import numpy as np


def define_hyperparameters():
    """
    Define hyperparameters for the zebrafish model.
    """

    hyperparameters = {}

    hyperparameters["REF_JOINT_AMP"] = np.array([
        0.06580,
        0.02810,
        0.02781,
        0.03047,
        0.03623,
        0.04127,
        0.04864,
        0.05398,
        0.06508,
        0.08945,
        0.10271,
        0.11789,
        0.14929,
        0.0,      # Note: Tail moves passively,
        0.0,      # Note: Tail moves passively,
    ])  # type: ignore unit:radian

    hyperparameters["ws_ref"] = 1 / \
        np.mean(hyperparameters["REF_JOINT_AMP"][:-2])

    return hyperparameters

