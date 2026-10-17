"""Métriques adaptées à la détection de fraude en contexte déséquilibré."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def at_review_rate(y_true, scores, rate: float = 0.02) -> dict:
    """Précision et rappel si l'on examine les `rate` % de transactions les plus suspectes.

    Traduit le score en termes métier : une équipe antifraude ne peut examiner
    qu'une petite fraction des transactions.
    """
    y_true = np.asarray(y_true)
    k = max(1, int(np.ceil(rate * len(y_true))))
    top = np.argsort(-np.asarray(scores))[:k]
    caught = y_true[top].sum()
    return {"precision": caught / k, "recall": caught / y_true.sum()}


def evaluate(y_true, scores, review_rates=(0.01, 0.02, 0.05)) -> dict:
    results = {
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
    }
    for rate in review_rates:
        r = at_review_rate(y_true, scores, rate)
        results[f"precision@{rate:.0%}"] = r["precision"]
        results[f"recall@{rate:.0%}"] = r["recall"]
    return results
