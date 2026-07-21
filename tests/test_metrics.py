import numpy as np

from encoderbench.metrics import (aggregate_subjects, binary_metrics, force_invalid_wrong,
                                  paired_bootstrap_difference)


def test_invalid_predictions_are_forced_wrong_per_row():
    truth = np.array(["AD", "CN", "AD"])
    prediction, invalid = force_invalid_wrong(truth, np.array(["UNK", "ERR", "AD"]), "AD")
    assert prediction.tolist() == ["CN", "AD", "AD"]
    assert invalid == 2
    assert binary_metrics(truth, prediction, None, "AD")["balanced_accuracy"] == 0.25


def test_subject_probabilities_average_repeat_scans():
    truth = np.array(["AD", "AD", "CN"])
    labels, prediction, probability, subjects = aggregate_subjects(
        truth, np.array([0.2, 0.9, 0.1]), np.array(["s1", "s1", "s2"]), "AD"
    )
    assert subjects.tolist() == ["s1", "s2"]
    assert labels.tolist() == ["AD", "CN"]
    assert np.allclose(probability, [0.55, 0.1])
    assert prediction.tolist() == ["AD", "CN"]


def test_paired_bootstrap_uses_common_subjects():
    truth = {"s1": "AD", "s2": "AD", "s3": "CN", "s4": "CN"}
    better = dict(truth)
    worse = {subject: "CN" for subject in truth}
    result = paired_bootstrap_difference(truth, better, worse, "AD", samples=100, seed=1)
    assert result["difference"] == 0.5
    assert result["n_subjects"] == 4

