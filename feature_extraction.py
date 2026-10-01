import numpy as np
import pandas as pd


# Feature extraction for a single 1-D signal (one spatial channel of one sample).
# Input:  either the differentiated signal (9999,) or the raw signal (10000,).
# Output: a vector of 16 hand-crafted time/frequency-domain features.
# These features are what the SVM model is trained on (see get_das_data.py).
def feature_extraction(data):  # data: (9999,) or (10000,)
    # Work in float64: the raw .mat data is uint16, and operations such as
    # x ** 4 or subtraction would silently overflow / wrap around in integer types.
    data = np.asarray(data, dtype=np.float64)

    def fft_fft(data):
        # Magnitude spectrum, keeping only the positive frequencies (skip the DC bin 0).
        fft_trans = np.abs(np.fft.fft(data))
        freq_spectrum = fft_trans[1:int(np.floor(len(data) * 1.0 / 2)) + 1]
        _freq_sum_ = np.sum(freq_spectrum)
        return freq_spectrum, _freq_sum_
    freq_spectrum, _freq_sum_ = fft_fft(data)
    abs_mean = np.abs(data).mean()
    # Maximum value
    dif_max = data.max()
    # Minimum value
    dif_min = data.min()
    # Peak-to-peak value (max - min)
    dif_pk = dif_max - dif_min
    # Mean
    dif_mean = data.mean()
    # Variance
    dif_var = data.var()
    # Standard deviation
    dif_std = data.std()
    # Energy (average power of the magnitude spectrum)
    dif_energy = np.sum(freq_spectrum ** 2) / len(freq_spectrum)
    # Root mean square: sqrt(mean(x^2)) == sqrt(mean^2 + std^2)
    dif_rms = np.sqrt(pow(dif_mean, 2) + pow(dif_std, 2))
    # Average rectified value (mean of |x|)
    dif_arv = abs_mean
    # Shape (waveform) factor = RMS / mean(|x|)
    dif_boxing = dif_rms / abs_mean
    # Impulse factor (IF) = max / mean(|x|)
    dif_maichong = dif_max / abs_mean
    # Crest (peak) factor (CF) = max / RMS
    dif_fengzhi = dif_max / dif_rms
    # Clearance (margin) factor (CL) = max / (mean(sqrt(|x|)))^2
    dif_yudu = dif_max / pow(np.mean(np.sqrt(np.abs(data))), 2)
    # Kurtosis (pandas' unbiased excess kurtosis)
    dif_kurt = pd.Series(data).kurt()
    # Kurtosis factor = mean(x^4) / RMS^4
    dif_qiaodu = np.mean(data ** 4) / pow(dif_rms, 4)
    # Spectral (information) entropy: treat the normalised spectrum as a probability
    # distribution and compute -sum(p * log2(p)). 1e-5 avoids log2(0).
    pr_freq = freq_spectrum * 1.0 / _freq_sum_
    dif_entropy = -1 * np.sum(np.log2(pr_freq + 1e-5) * pr_freq)

    feature_list = [dif_max, dif_min, dif_pk, dif_mean,
                    dif_energy, dif_var, dif_std, dif_rms,
                    dif_arv, dif_boxing, dif_maichong, dif_fengzhi,
                    dif_yudu, dif_kurt, dif_qiaodu, dif_entropy]
    # Round every feature to 3 decimals, as in the original implementation.
    return np.round(np.array(feature_list, dtype=np.float64), 3)
