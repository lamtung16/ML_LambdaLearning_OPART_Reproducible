# Changepoint detection penalty learning

## Folders:
- **`acc_rate`:** Contains CSV files detailing the accuracy rates for each implemented method.
- **`figures`:** Holds figures generated.
- **`raw_data`:** Stores raw CSV data files comprising sequences and labels.
- **`record_dataframe`:** CSV files including method specifics log_lambda, total_labels, and number_of_errors.
- **`training_data`:** Consists of data for training pertaining to error counts for each lambda, sequence features, and target intervals.

## Python Files:
- **`opart_functions.py`:** Collection of utility functions.
- **`accuracy_rate_comparison_figure.py`:** Implements functionality to generate figures comparing accuracy rates across methods.
- **`BIC.py`:** Implements the computation of log_lambda using the Bayesian Information Criterion (BIC) approach.
- **`linear.py`:** Provides functionality to learn log_lambda utilizing linear regression techniques.
- **`MLP.py`:** Implements learning log_lambda from a set of sequence features using a Multi-Layer Perceptron (MLP) approach.
- **`main.py`:** Serves as the main entry point, responsible for generating accuracy rate files in the `acc_rate` folder and recording dataframes for each implemented method to `record_dataframe` folder.

## Generating Figures from Scratch:
- Accuracy Rate Comparison figure:
  - Run `main.py` to generate a CSV file containing accuracy rates for each method. Add new methods if necessary.
  - Execute `accuracy_rate_comparison_figure.py`. The resulting figure will be generated in the `figures` folder.
