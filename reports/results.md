# Résultats des expériences

Scores sur la période de validation (30 jours précédant le test).
Le jeu de test reste réservé à l'évaluation finale.

| Modèle | PR-AUC | ROC-AUC | Précision@2% | Rappel@2% |
|---|---|---|---|---|
| XGBoost baseline | 0,606 | 0,922 | 0,768 | 0,453 |
| XGBoost optimisé (Optuna TPE, N essais) | 0,607 | 0,924 | 0,765 | 0,451 |

**Optimisation des hyperparamètres.** L'optimisation bayésienne n'améliore pas
significativement les paramètres initiaux (PR-AUC 0,607 contre 0,606), un écart
du même ordre que la variabilité aléatoire. Les paramètres optimisés sont retenus
pour toutes les variantes suivantes, afin qu'elles partent des mêmes réglages.
