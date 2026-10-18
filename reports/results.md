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

## Étude du déséquilibre de classes

| Variante | PR-AUC | Précision@1% | Précision@2% | Rappel@2% | Rappel@5% |
|---|---|---|---|---|---|
| Sans traitement | 0,605 ± 0,003 | 0,924 ± 0,001 | 0,763 ± 0,002 | 0,450 ± 0,001 | 0,644 ± 0,004 |
| Pondération | 0,592 ± 0,005 | 0,930 ± 0,000 | 0,768 ± 0,005 | 0,453 ± 0,003 | 0,624 ± 0,006 |
| SMOTE par produit (0,25) | 0,586 ± 0,001 | 0,934 ± 0,007 | 0,748 ± 0,006 | 0,441 ± 0,004 | 0,621 ± 0,005 |
| SMOTE par produit (0,1) | 0,604 ± 0,005 | 0,930 ± 0,005 | 0,766 ± 0,006 | 0,452 ± 0,003 | 0,638 ± 0,007 |

**Conclusion.** : Aucun traitement du déséquilibre n'améliore la PR-AUC. Un SMOTE léger (ratio 0,1) fait jeu égal avec l'absence de traitement, et une pondération et un SMOTE plus fort (ratio 0,25) la dégradent nettement. Le rééquilibrage apporte seulement un léger gain de précision sur les 1 % de transactions les plus suspectes. Dans ce cas précis, il peut être intéressant de procéder au SMOTE si les usecase le requièrent.
