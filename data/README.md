# **Data Folder**

This folder contains all the data and scripts used for data generation and processing.

## **Contents**
- **`synthetic_data_generation.py`**: 
  - Python script for generating synthetic stellar data, including light curves for different star types.
  - The generated dataset is stored as `synthetic_star_dataset.csv`.

- **`synthetic_star_dataset.csv`**: 
  - The main synthetic dataset containing basic stellar parameters (`Teff`, `logg`, `Lum`, `Star_Type`).

- **`processed/`**: 
  - Contains processed datasets for further analysis and modeling.
  - **`synthetic_fliper_data.csv`**: Processed dataset with computed FliPer values and other features ready for model training.

## **Purpose**
This folder ensures reproducibility of the data generation and processing pipeline, serving as the foundation for the entire project.


