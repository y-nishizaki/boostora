import os
import pandas as pd
import unittest
from boostora import Boostora

class TestBoostora(unittest.TestCase):
    def test_classification(self):
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
        df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        df['species'] = df['species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

        target_col = 'species'

        boostora = Boostora(classification=True)
        boostora.run_framework(df, target_col, experiment_name="Boostora_Classification_Test", n_trials=10)

    def test_regression(self):
        df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data', header=None, sep='\s+')
        df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

        target_col = 'MEDV'

        boostora = Boostora(classification=False)
        boostora.run_framework(df, target_col, experiment_name="Boostora_Regression_Test", n_trials=10)


if __name__ == '__main__':
    unittest.main()
