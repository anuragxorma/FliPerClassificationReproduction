# Reproducing $FliPer_{Class}$: Machine Learning for Stellar Classification

This repository contains the implementation for reproducing the results of the **$FliPer_{Class}$** paper, titled *"FliPer_Class: In search of solar-like pulsators among TESS targets"*, authored by **L. Bugnet, R. A. García, S. Mathur, G. R. Davies, O. J. Hall, M. N. Lund, and B. M. Rendle**. The paper focuses on using stellar power spectral density (PSD) and machine learning for star classification.

The approach involves extracting physical features from stellar light curves, such as **FliPer** (Flicker in Power Spectrum), to classify stars into various types like SPB, Gamma Doradus, RR Lyrae, and Solar-Like stars.

---

## **Motivation**

The original paper relies on actual stellar data from astrophysical simulations, which are not publicly accessible. To overcome this limitation, **synthetic data** is generated based on predefined light curve characteristics for different stellar types. While synthetic, this data retains meaningful variability features inspired by stellar physics, allowing us to mimic the analysis pipeline.

This repository also includes an **additional neural network approach** (not part of the original paper) for star classification, which demonstrates the potential of modern deep learning techniques in this context.

---

## **Pipeline Overview**

The implementation is broken into modular steps, with each step handled in a dedicated file or notebook:

1. **Data Generation**
   - Synthetic light curves are generated for different star types using predefined variability parameters.
   - **Output:** `data/synthetic_star_dataset.csv`

2. **PSD and FliPer Calculation**
   - Power spectral density (PSD) is computed for each light curve using the Welch method.
   - FliPer values are calculated as a high-frequency power-to-total power ratio for each star.

3. **Model Training**
   - A **Random Forest classifier** is trained using the extracted features (Teff, logg, Lum, FliPer).
   - Performance metrics include accuracy, precision, recall, and F1-score.
   - A **Neural Network model** is also implemented to explore deep learning potential.

4. **Analysis and Visualization**
   - Model performance is evaluated through a confusion matrix, feature importance analysis, and ROC curves.
   - Results demonstrate strong classification performance on the synthetic dataset.

---

## **Results and Discussion**

### Random Forest Classifier
The Random Forest classifier achieved high accuracy across all star types, demonstrating the power of feature engineering with PSDs and FliPer values.

### Neural Network Classifier
The neural network approach was an exploratory addition to assess whether deep learning could provide comparable results without extensive feature engineering. While the neural network also performed well, the Random Forest model remains computationally efficient and interpretable, aligning with the paper's original objectives.

### Why Are the Results So Good?
1. **Clean Synthetic Data:** No real-world observational noise or uncertainties.
2. **Feature Distinctiveness:** Synthetic variability parameters ensure clear differences between classes.
3. **Balanced Dataset:** The data generation ensures well-represented star types.

While promising, these results should be validated on real-world astrophysical datasets to account for observational challenges.

---

## **Repository Structure**
    
    ├── data/
    │   ├── synthetic_data_generation.py    # Data Generation
    │   ├── synthetic_star_dataset.csv      # Generated dataset
    │   ├── processed/                      # Processed datasets (with FliPer values)
    │      ├── synthetic_fliper_data.csv
    ├── src/
    │   ├── data_preprocessing.py           # PSD/FliPer calculations
    │   ├── model_training.py               # Random Forest model training
    │   ├── neural_network.py               # Neural network implementation
    ├── notebooks/
    │   ├── 01_data_preparation.ipynb       # Visualizes the synthetic dataset
    │   ├── 02_psd_calculation.ipynb        # PSD calculation visualization
    │   ├── 03_fliper_extraction.ipynb      # FliPer calculation
    │   ├── 04_model_training.ipynb         # Random Forest training workflow
    │   ├── 05_analysis.ipynb               # Model evaluation and visualization
    │   ├── 06_nn_training.ipynb            # Neural network training and evaluation
    ├── models/
    │   ├── random_forest.pkl               # Saved Random Forest model
    │   ├── neural_network_classifier.keras # Saved Neural Network model
    └── README.md                           # This file
---

## **How to Run**

### **Prerequisites**
- Python 3.x
- Required packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `tensorflow`

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/anuragxorma/FliPerClassificationReproduction
2. Install dependencies
    ```bash
    pip install numpy pandas matplotlib scikit-learn scipy tensorflow
3. Run the pipeline:

- Data Generation: `python src/data_preprocessing.py`
- PSD Calculation: `Run notebooks/02_psd_calculation.ipynb`
- Model Training: `Run notebooks/04_model_training.ipynb`
- Analysis: `Run notebooks/05_analysis.ipynb`

## **Future Directions**
1. Incorporating Time Series Models: Exploring methods such as LSTM or GRU networks to directly analyze stellar light curves, leveraging temporal dependencies in the data.
2. Transfer Learning: Applying pre-trained models on large astrophysical datasets for better generalization.
3. Multimodal Approaches: Combining light curve data with other astrophysical features (e.g., spectral data) for enhanced classification.
These extensions represent exciting opportunities for further improving classification accuracy and exploring new frontiers in stellar physics.

## **Acknowledgements**
This work reproduces the analysis pipeline described in the FliPer_Class paper. Special thanks to the authors for their groundbreaking work in applying machine learning techniques to stellar classification.

Paper Reference: "FliPer_Class: In search of solar-like pulsators among TESS targets" by L. Bugnet, R. A. García, S. Mathur, G. R. Davies, O. J. Hall, M. N. Lund, and B. M. Rendle.

I acknowledge the authors' innovative methodology, which forms the foundation of this reproduction and extension effort. This project aims to reproduce and learn from their methods while expanding the analysis with additional neural network approaches.

