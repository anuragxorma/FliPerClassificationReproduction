import numpy as np
import pandas as pd

# Define star types and their feature ranges, along with sample sizes
STAR_TYPES = {
    "Solar_Like": {"teff_range": (5200, 6300), "logg_range": (3.8, 4.7), "lum_range": (0.08, 1.2), "num_samples": 3668},
    "sdBV": {"teff_range": (18000, 42000), "logg_range": (4.8, 6.2), "lum_range": (8, 120), "num_samples": 129},
    "Beta_Cephei": {"teff_range": (14000, 32000), "logg_range": (3.3, 4.2), "lum_range": (800, 12000), "num_samples": 298},
    "SPB": {"teff_range": (9000, 21000), "logg_range": (3.3, 4.2), "lum_range": (80, 1200), "num_samples": 1846},
    "Delta_Scuti": {"teff_range": (5800, 8500), "logg_range": (3.3, 4.7), "lum_range": (0.8, 12), "num_samples": 115},
    "Gamma_Doradus": {"teff_range": (5800, 8500), "logg_range": (3.3, 4.2), "lum_range": (0.08, 1.2), "num_samples": 1569},
    "roAp": {"teff_range": (6800, 9200), "logg_range": (3.8, 4.7), "lum_range": (0.008, 0.12), "num_samples": 287},
    "RRLyrae": {"teff_range": (5800, 7700), "logg_range": (1.8, 3.2), "lum_range": (25, 120), "num_samples": 646},
    "LPV": {"teff_range": (2800, 5200), "logg_range": (-0.2, 1.7), "lum_range": (800, 12000), "num_samples": 965},
    "Cepheid": {"teff_range": (4800, 6200), "logg_range": (0.8, 2.2), "lum_range": (80, 3500), "num_samples": 1289}
}

def generate_synthetic_data(ranges):
    num_samples = ranges["num_samples"]
    
    # Add more realistic noise and variations
    base_noise = 0.15  # Increased base noise
    
    teff = np.random.normal(
        np.mean(ranges["teff_range"]), 
        (ranges["teff_range"][1] - ranges["teff_range"][0])/4, 
        size=num_samples
    )
    logg = np.random.normal(
        np.mean(ranges["logg_range"]), 
        (ranges["logg_range"][1] - ranges["logg_range"][0])/4, 
        size=num_samples
    )
    lum = np.random.lognormal(
        np.mean(np.log(ranges["lum_range"])), 
        0.5, 
        size=num_samples
    )
    
    # Add correlated noise
    noise = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=[[base_noise, base_noise/2, base_noise/2],
             [base_noise/2, base_noise, base_noise/2],
             [base_noise/2, base_noise/2, base_noise]],
        size=num_samples
    )
    
    return np.column_stack([teff, logg, lum]) + noise

# Create dataset
data = []
for star_type, ranges in STAR_TYPES.items():
    star_data = generate_synthetic_data(ranges)
    labels = np.full(star_data.shape[0], star_type)  # Label for the star type
    data.append(np.column_stack([star_data, labels]))

# Combine and save dataset
columns = ["Teff", "logg", "Lum", "Star_Type"]
synthetic_dataset = pd.DataFrame(np.vstack(data), columns=columns)
synthetic_dataset.to_csv("data/synthetic_star_dataset.csv", index=False)

print("Synthetic dataset created and saved as 'synthetic_star_dataset.csv'.")
