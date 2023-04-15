import pandas as pd
from sklearn.datasets import fetch_openml
from boostora.boostora import run_framework


def custom_param_tuning(trial):
    param = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }
    # 省略（独自のパラメータ設定のコード）

    return param


data = fetch_openml(name='Boston', version=1, as_frame=True)
data_frame = data.frame
data_frame["target"] = data.target

run_framework(data_frame, "target", experiment_name="My_XGBoost_experiment",
              param_tuning_func=custom_param_tuning, n_trials=5)
