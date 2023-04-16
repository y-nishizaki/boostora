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
from sklearn.metrics import mean_squared_error, accuracy_score
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


class Boostora:
    def __init__(self, classification=False):
        self.classification = classification

    def load_data(self, data_frame, target_col, test_size=0.2, random_state=42):
        X, y = data_frame.drop(target_col, axis=1), data_frame[target_col]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def objective(self, trial, X_train, y_train, X_test, y_test, param_tuning_func):
        params = param_tuning_func(trial)
        dtrain = self.create_dmatrix(X_train, y_train)
        dtest = self.create_dmatrix(X_test, y_test)

        model = xgb.train(params, dtrain)
        preds = model.predict(dtest)

        if self.classification:
            acc = accuracy_score(y_test, preds.round())
            return 1 - acc
        else:
            rmse = mean_squared_error(y_test, preds, squared=False)
            return rmse

    def save_and_log_visualizations(self, study, model, X_test):
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

    def default_param_tuning(self, trial):
        param = {
            "objective": "reg:squarederror" if not self.classification else "binary:logistic",
            "eval_metric": "rmse" if not self.classification else "logloss",
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

    def run_framework(self, data_frame, target_col, experiment_name="XGBoost_optuna_tuning", param_tuning_func=None, n_trials=100):
        X_train, X_test, y_train, y_test = self.load_data(data_frame, target_col)

        if param_tuning_func is None:
            param_tuning_func = self.default_param_tuning

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: self.objective(
                trial, X_train, y_train, X_test, y_test, param_tuning_func), n_trials=n_trials)

            best_trial = study.best_trial

            mlflow.log_params(best_trial.params)

            best_params = best_trial.params
            best_params["objective"] = "reg:squarederror" if not self.classification else "binary:logistic"
            best_params["eval_metric"] = "rmse" if not self.classification else "logloss"

            dtrain = self.create_dmatrix(X_train, y_train)
            dtest = self.create_dmatrix(X_test, y_test)

            best_model = xgb.train(best_params, dtrain, callbacks=[
                                   MlflowXgboostCallback()])
            preds = best_model.predict(dtest)

            if self.classification:
                acc = accuracy_score(y_test, preds.round())
                mlflow.log_metric("accuracy", acc)
            else:
                rmse = mean_squared_error(y_test, preds, squared=False)
                mlflow.log_metric("rmse", rmse)

            mlflow.xgboost.log_model(
                best_model, "model", registered_model_name="XGBoost_optuna_tuning")

            self.save_and_log_visualizations(study, best_model, X_test)

    def create_dmatrix(self, X, y):
        if self.classification:
            return xgb.DMatrix(X, label=y)
        else:
            return xgb.DMatrix(X, label=y)



