"""Entraînement de XGBoost, avec optimisation bayésienne optionnelle des hyperparamètres."""

import optuna
import xgboost as xgb

# Paramètres techniques, jamais optimisés
FIXED_PARAMS = {
    "tree_method": "hist",
    "enable_categorical": True,
    "eval_metric": "aucpr",
    "n_jobs": -1,
}

SAMPLERS = {
    "tpe": lambda seed: optuna.samplers.TPESampler(seed=seed),
    "gp": lambda seed: optuna.samplers.GPSampler(seed=seed),
}


def train_xgb(
    X_train, y_train, X_val, y_val, params: dict, verbose=200
) -> xgb.XGBClassifier:
    """Entraîne XGBoost avec arrêt anticipé sur la PR-AUC de validation."""
    model = xgb.XGBClassifier(**FIXED_PARAMS, **params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)
    return model


def _suggest(trial: optuna.Trial, name: str, spec: dict):
    log = spec.get("log", False)
    if spec["type"] == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], log=log)
    if spec["type"] == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=log)
    raise ValueError(f"Type inconnu pour {name} : {spec['type']}")


def tune_xgb(X_train, y_train, X_val, y_val, base_params: dict, tuning: dict):
    """Cherche les hyperparamètres maximisant la PR-AUC de validation."""

    def objective(trial):
        params = {
            **base_params,
            **{n: _suggest(trial, n, s) for n, s in tuning["search_space"].items()},
        }
        model = train_xgb(X_train, y_train, X_val, y_val, params, verbose=False)
        trial.set_user_attr("best_iteration", model.best_iteration)
        return model.best_score

    sampler = SAMPLERS[tuning.get("sampler", "tpe")](base_params.get("random_state", 0))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=tuning["n_trials"],
        timeout=tuning.get("timeout_minutes", 0) * 60 or None,
    )
    return {**base_params, **study.best_params}, study


def train_from_config(X_train, y_train, X_val, y_val, config: dict):
    """Entraîne XGBoost selon la config, en optimisant d'abord si `tuning.enabled`."""
    params, study = dict(config["params"]), None
    if config["tuning"]["enabled"]:
        params, study = tune_xgb(
            X_train, y_train, X_val, y_val, params, config["tuning"]
        )
    model = train_xgb(X_train, y_train, X_val, y_val, params)
    return model, params, study
