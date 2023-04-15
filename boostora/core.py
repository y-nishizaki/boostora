import itertools
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import optuna
import optuna.visualization as vis
import pandas as pd
import plotly.io as pio
import shap
import shap.plots as sp
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost.callback import TrainingCallback


class MlflowXgboostCallback(TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        if not evals_log:
            return False
        for data, metric in evals_log.items():
            for metric_name, log in metric.items():
                mlflow.log_metric(f'{data}-{metric_name}', log[-1])
        return False


def load_data(data_frame, target_col, test_size=0.2, random_state=42):
    X, y = data_frame.drop(target_col, axis=1), data_frame[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def objective(trial, X_train, y_train, X_test, y_test, param_tuning_func):
    params = param_tuning_func(trial)
    dtrain = create_dmatrix(X_train, y_train)
    dtest = create_dmatrix(X_test, y_test)

    model = xgb.train(params, dtrain)
    preds = model.predict(dtest)
    rmse = mean_squared_error(y_test, preds, squared=False)

    return rmse


def save_and_log_visualizations(study, model, X_test):
    # Optimization History
    fig = vis.plot_optimization_history(study)
    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = os.path.join(tmpdir, "optimization_history.png")
        pio.write_image(fig, fig_path, format='png')
        mlflow.log_artifact(fig_path, "optuna_visualizations")

    # Parameter Importance
    fig = vis.plot_param_importances(study)
    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = os.path.join(tmpdir, "parameter_importance.png")
        pio.write_image(fig, fig_path, format='png')
        mlflow.log_artifact(fig_path, "optuna_visualizations")

    # Slice Plot
    fig = vis.plot_slice(study)
    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = os.path.join(tmpdir, "slice_plot.png")
        pio.write_image(fig, fig_path, format='png')
        mlflow.log_artifact(fig_path, "optuna_visualizations")

    # Parallel Coordinate Plot
    fig = vis.plot_parallel_coordinate(study)
    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = os.path.join(tmpdir, "parallel_coordinate_plot.png")
        pio.write_image(fig, fig_path, format='png')
        mlflow.log_artifact(fig_path, "optuna_visualizations")

    # Contour Plot
    param_combinations = list(itertools.combinations(
        study.best_trial.params.keys(), 2))
    for param1, param2 in param_combinations:
        fig = vis.plot_contour(study, params=[param1, param2])
        with tempfile.TemporaryDirectory() as tmpdir:
            fig_path = os.path.join(
                tmpdir, f"contour_plot_{param1}_vs_{param2}.png")
            pio.write_image(fig, fig_path, format='png')
            mlflow.log_artifact(fig_path, "optuna_visualizations")

    # SHAP Summary Plot
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = os.path.join(tmpdir, "shap_summary_plot.png")
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path, "shap_plots")

    # Compute the mean SHAP values
    mean_shap_values = np.mean(shap_values.values, axis=0)
    mean_expected_value = np.mean(explainer.expected_value)

    # force_plot for the mean SHAP values
    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = os.path.join(tmpdir, "shap_force_plot_mean.png")
        shap.force_plot(mean_expected_value, mean_shap_values, X_test.columns, show=False, matplotlib=True)
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path, "shap_plots")

def default_param_tuning(trial):
    param = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }
    if param["booster"] in ["gbtree", "dart"]:
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float(
            "rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float(
            "skip_drop", 1e-8, 1.0, log=True)

    return param


def run_framework(data_frame, target_col, experiment_name="XGBoost_optuna_tuning", param_tuning_func=default_param_tuning, n_trials=100):
    X_train, X_test, y_train, y_test = load_data(data_frame, target_col)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train,
                   X_test, y_test, param_tuning_func), n_trials=n_trials)

    best_params = study.best_trial.params
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        dtrain = create_dmatrix(X_train, y_train)
        dtest = create_dmatrix(X_test, y_test)
        best_params["objective"] = "reg:squarederror"
        best_params["eval_metric"] = "rmse"

        model = xgb.train(best_params, dtrain, evals=[
                          (dtest, "validation")], verbose_eval=False, callbacks=[MlflowXgboostCallback()])
        y_pred = model.predict(dtest)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_params(best_params)
        mlflow.xgboost.log_model(model, "xgboost_model")

        save_and_log_visualizations(study, model, X_test)

        print("Model and plots saved in run %s" %
              mlflow.active_run().info.run_id)


def create_dmatrix(X, y=None, enable_categorical=True):
    if y is not None:
        return xgb.DMatrix(X, label=y, enable_categorical=enable_categorical)
    else:
        return xgb.DMatrix(X, enable_categorical=enable_categorical)
