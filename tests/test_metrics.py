import numpy as np
from fraud_detection.evaluation.metrics import at_review_rate, evaluate


def test_perfect_ranking():
    y = np.array([0] * 98 + [1] * 2)
    scores = y.astype(float)
    r = at_review_rate(y, scores, rate=0.02)
    assert r["precision"] == 1.0 and r["recall"] == 1.0
    assert evaluate(y, scores)["pr_auc"] == 1.0
