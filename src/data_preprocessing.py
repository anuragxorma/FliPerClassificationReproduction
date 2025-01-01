import numpy as np
from scipy.signal import welch

# Define light curve properties for different star types
LIGHT_CURVE_PROPERTIES = {
    "Solar_Like": {"amplitude_range": (0.008, 0.06), "frequency_range": (0.4, 1.7)},
    "sdBV": {"amplitude_range": (0.08, 0.35), "frequency_range": (4.5, 11.0)},
    "Beta_Cephei": {"amplitude_range": (0.08, 0.6), "frequency_range": (0.8, 3.5)},
    "SPB": {"amplitude_range": (0.04, 0.25), "frequency_range": (0.15, 1.2)},
    "Delta_Scuti": {"amplitude_range": (0.04, 0.12), "frequency_range": (8.0, 22.0)},
    "Gamma_Doradus": {"amplitude_range": (0.008, 0.06), "frequency_range": (0.8, 5.5)},
    "roAp": {"amplitude_range": (0.08, 0.35), "frequency_range": (28.0, 52.0)},
    "RRLyrae": {"amplitude_range": (0.4, 1.2), "frequency_range": (0.04, 0.12)},
    "LPV": {"amplitude_range": (0.25, 0.9), "frequency_range": (0.008, 0.06)},
    "Cepheid": {"amplitude_range": (0.4, 1.2), "frequency_range": (0.04, 0.25)},
}

def generate_light_curve(teff, star_type, num_points=1000, time_step=0.01):
    time = np.arange(0, num_points * time_step, time_step)
    properties = LIGHT_CURVE_PROPERTIES.get(star_type)
    
    # Add multiple frequency components
    num_components = np.random.randint(2, 5)
    signal = np.zeros_like(time)
    
    for _ in range(num_components):
        amplitude = np.random.uniform(*properties["amplitude_range"]) * np.random.uniform(0.5, 1.0)
        frequency = np.random.uniform(*properties["frequency_range"]) * np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2*np.pi)
        signal += amplitude * np.sin(2 * np.pi * frequency * time + phase)
    
    # Add red noise
    red_noise = np.cumsum(np.random.normal(0, 0.02, size=time.shape))
    red_noise = red_noise / np.std(red_noise) * 0.15
    
    # Add white noise
    white_noise = np.random.normal(0, 0.15, size=time.shape)
    
    return time, signal + red_noise + white_noise

def compute_psd(time, flux):
    """
    Compute the Power Spectral Density (PSD) of a light curve.

    Parameters:
        time (array): Time array.
        flux (array): Flux array.

    Returns:
        tuple: Frequency array and PSD array.
    """
    freq, psd = welch(flux, fs=1 / (time[1] - time[0]), nperseg=256)
    return freq, psd

def compute_fliper(psd, freq, high_freq_limit=10.0):
    """
    Compute FliPer value for a PSD.

    Parameters:
        psd (array): Power Spectral Density array.
        freq (array): Frequency array.
        high_freq_limit (float): Frequency threshold for high frequencies.

    Returns:
        float: FliPer value.
    """
    total_power = np.sum(psd)
    high_freq_power = np.sum(psd[freq > high_freq_limit])
    return high_freq_power / total_power
