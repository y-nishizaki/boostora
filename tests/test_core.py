import pytest
import pandas as pd
import numpy as np
import optuna
from boostora.core import Boostora
from sklearn.datasets import make_classification, make_regression

@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    return df

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    return df

def test_load_data_classification(classification_data):
    boostora = Boostora(classification=True)
    X_train, X_test, y_train, y_test = boostora.load_data(classification_data, 'target')
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20

def test_load_data_regression(regression_data):
    boostora = Boostora()
    X_train, X_test, y_train, y_test = boostora.load_data(regression_data, 'target')
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20

def test_default_param_tuning_classification():
    boostora = Boostora(classification=True)
    fixed_params = {
        "booster": "gbtree",
        "lambda": 1.0,
        "alpha": 1e-8,  # Change from 0.0 to 1e-8
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "eta": 0.3,
        "max_depth": 6,
        "gamma": 1e-8,
        "grow_policy": "depthwise",
    }
    trial = optuna.trial.FixedTrial(fixed_params)
    params = boostora.default_param_tuning(trial)
    assert params['objective'] == 'binary:logistic'
    assert params['eval_metric'] == 'logloss'

def test_default_param_tuning_regression():
    boostora = Boostora()
    fixed_params = {
        "booster": "gbtree",
        "lambda": 1.0,
        "alpha": 1e-8,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "eta": 0.3,
        "max_depth": 6,
        "gamma": 1e-8,
        "grow_policy": "depthwise",
    }
    trial = optuna.trial.FixedTrial(fixed_params)
    params = boostora.default_param_tuning(trial)
    assert params['objective'] == 'reg:squarederror'
    assert params['eval_metric'] == 'rmse'    

# 追加のテストを実行する場合は、ここに記述してください。
