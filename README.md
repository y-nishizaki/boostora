# Boostora

Boostora is a Python package that simplifies the process of hyperparameter tuning and visualization for XGBoost models using Optuna and SHAP.

## Features

- Automated hyperparameter tuning with Optuna
- Visualization of hyperparameter optimization results
- Interpretation of model predictions using SHAP values
- Logging of results and plots with MLflow

## Installation

To install Boostora, run the following command:

```bash
pip install git+https://github.com/y-nishizaki/boostora.git
```

# Usage
```Python
import pandas as pd
from boostora import run_framework

data_frame = pd.read_csv("path/to/your/data.csv")
target_col = "target"

run_framework(data_frame, target_col, n_trials=100)
```
run_framework function accepts additional parameters to customize the optimization process. Please refer to the documentation for more information.

# Customization
You can create your own parameter tuning function and pass it to run_framework as an argument. Please refer to the documentation for a detailed guide on how to create a custom parameter tuning function.

# License
Boostora is released under the [MIT License](https://opensource.org/license/mit/).


# Appendix
「Boostora」は、"Boost"（ブースト）と、"Exploratory"（探索的）の一部である "ora" を組み合わせた造語です。ブーストはXGBoostの能力を強調し、"ora" は探索的なオプティマイゼーションプロセスを表現しています。Boostoraは、XGBoostのパフォーマンスを最大限に引き出すための探索的なフレームワークを意味しています。

# Directory
```
tree
```
.
├── LICENSE
├── README.md
├── boostora
│   ├── __init__.py
│   └── core.py
├── requirements.txt
├── requirements_dev.txt
├── setup.py
└── tests
    ├── __init__.py
    └── test_core.py
```
brew install tree 
```
