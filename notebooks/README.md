# **Notebooks Folder**

This folder contains Jupyter notebooks documenting the step-by-step execution of the project.

## **Contents**
1. **`01_data_preparation.ipynb`**:
   - Generates the synthetic dataset using predefined stellar parameters.
   - Outputs: `data/synthetic_star_dataset.csv`.

2. **`02_psd_calculation.ipynb`**:
   - Computes power spectral density (PSD) for synthetic light curves.
   - Visualizes PSD plots for different stellar types.

3. **`03_fliper_extraction.ipynb`**:
   - Extracts FliPer values from PSDs.
   - Processes and saves data to `data/processed/synthetic_fliper_data.csv`.

4. **`04_model_training.ipynb`**:
   - Trains a Random Forest classifier using the processed dataset.
   - Evaluates and saves the model to the `models/` folder.

5. **`05_analysis.ipynb`**:
   - Analyzes and visualizes the Random Forest model's performance.
   - Includes metrics such as confusion matrices, feature importance plots, and classification reports.

6. **`06_nn_training_and_analysis.ipynb`**:
   - Trains a neural network (MLP) for star classification.
   - Evaluates the neural network model and compares its performance to other approaches.

## **Purpose**
The notebooks provide a clear, sequential workflow for the entire project, with detailed code, visualizations, and analyses.


