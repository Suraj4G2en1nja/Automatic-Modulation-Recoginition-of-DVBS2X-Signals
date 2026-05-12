import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def process_iq_data(df, snr=None):

    # ✅ FIX 1: Make randomness consistent
    np.random.seed(42)

    snr = df["snr"].iloc[0] if snr is None else snr

    raw = df[[col for col in df.columns if col.startswith("sig_")]].values.astype(np.float32)

    I = raw[:, :128]
    Q = raw[:, 128:]

    complex_sig = I + 1j * Q

    # ✅ FIX 2: Add noise only if SNR is provided AND dataset doesn't already include it
    if snr is not None and "snr" not in df.columns:
        noise_factor = 10 ** (-snr / 20)
        noise = noise_factor * np.random.randn(*complex_sig.shape)
        complex_sig = complex_sig + noise

    amp = np.abs(complex_sig)
    phase = np.angle(complex_sig)
    freq = np.diff(phase, axis=1, prepend=0)

    fft_mag = np.abs(np.fft.fft(complex_sig, axis=1))

    m20 = np.mean(complex_sig**2, axis=1)
    m21 = np.mean(np.abs(complex_sig)**2, axis=1)
    m40 = np.mean(complex_sig**4, axis=1)
    m42 = np.mean(np.abs(complex_sig)**4, axis=1)

    c21 = m21
    c40 = np.abs(m40 - 3 * (m20**2))
    c42 = np.abs(m42 - np.abs(m20)**2 - 2 * (m21**2))

    amp_kurt = kurtosis(amp, axis=1)
    amp_skew = skew(amp, axis=1)

    df_amp = pd.DataFrame(amp, columns=[f'a_{i}' for i in range(128)])
    df_frq = pd.DataFrame(freq, columns=[f'f_{i}' for i in range(128)])
    df_fft = pd.DataFrame(fft_mag, columns=[f's_{i}' for i in range(128)])

    df_stats = pd.DataFrame({
        'c21': c21,
        'c40': c40,
        'c42': c42,
        'kurt': amp_kurt,
        'skew': amp_skew
    })

    final_df = pd.concat([df_amp, df_frq, df_fft, df_stats], axis=1)

    return final_df, snr