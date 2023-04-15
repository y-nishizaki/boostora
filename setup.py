from setuptools import setup, find_packages

def read_requirements(file):
    with open(file, "r") as f:
        return f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="boostora",
    version="0.1.0",
    author="Yoshihisa Nishizaki",
    author_email="your.email@example.com",
    description="Boostora is a Python package that simplifies the process of hyperparameter tuning and visualization for XGBoost models using Optuna and SHAP.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/y-nishizaki/boostora",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires='>=3.6',
    install_requires=read_requirements("requirements.txt"),
)
