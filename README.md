# Changepoint detection penalty learning

## Folders:
- **`acc_rate`:** Contains CSV files detailing the accuracy rates for each implemented method.
- **`figures`:** Holds visualizations and figures generated during the project.
- **`raw_data`:** Stores raw CSV data files comprising sequences and corresponding labels.
- **`record_dataframe`:** Houses CSV files documenting learned model statistics, including method specifics such as log_lambda, total_labels, and number_of_errors.
- **`training_data`:** Consists of data pertaining to error counts for each lambda, sequence features, and target intervals.

## Python Files:
- **`opart_functions.py`:** Houses a collection of utility functions utilized across the project.
- **`accuracy_rate_comparison_figure.py`:** Implements functionality to generate figures comparing accuracy rates across methods.
- **`BIC.py`:** Implements the computation of log_lambda using the Bayesian Information Criterion (BIC) approach.
- **`linear.py`:** Provides functionality to learn log_lambda utilizing linear regression techniques.
- **`MLP.py`:** Implements learning log_lambda from a set of sequence features using a Multi-Layer Perceptron (MLP) approach.
- **`main.py`:** Serves as the main entry point, responsible for generating accuracy rate files in the 'acc_rate' folder and recording dataframes for each implemented method.

## Generating Figures from Scratch:
1. Run `main.py` to generate a CSV file containing accuracy rates for each method. Add new methods if necessary.
2. Execute `accuracy_rate_comparison_figure.py`. The resulting figure will be generated in the `figures` folder.
