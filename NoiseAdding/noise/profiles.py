# rfuav/NoiseAdding/noise/profiles.py

import numpy as np


def add_awgn_noise(signal, snr_db):
    """Additive White Gaussian Noise (AWGN), uniformly spread across all frequencies"""
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise


def add_awgn_local_noise(signal, snr_db, window_size=1024):
    """AWGN scaled locally according to signal energy in a sliding window"""
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    base_noise_power = signal_power / snr_linear

    abs_signal = np.abs(signal)
    energy = np.convolve(abs_signal ** 2, np.ones(window_size) / window_size, mode='same')
    norm_energy = energy / (np.max(energy) + 1e-12)

    noise_std = np.sqrt((base_noise_power / 2) * norm_energy)
    noise = (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)) * noise_std
    return signal + noise


def add_tone_noise(signal, fs, tone_freq=1e6, amplitude=0.3):
    """Add a continuous wave tone (narrowband sinusoid) at a fixed frequency"""
    t = np.arange(len(signal)) / fs
    tone = amplitude * np.exp(2j * np.pi * tone_freq * t)
    return signal + tone


def add_band_noise(signal, fs, center_freq=5e6, bandwidth=1e6, amplitude=0.2):
    """Add band-limited Gaussian noise centered at a given frequency"""
    n = len(signal)
    noise = (np.random.randn(n) + 1j * np.random.randn(n))
    freqs = np.fft.fftfreq(n, d=1 / fs)
    mask = np.abs(freqs - center_freq) < (bandwidth / 2)
    spectrum = np.fft.fft(noise)
    spectrum[~mask] = 0
    filtered_noise = amplitude * np.fft.ifft(spectrum)
    return signal + filtered_noise


def add_impulse_noise(signal, num_impulses=10, magnitude=5.0):
    """Inject sparse high-energy impulses (short spikes) into the signal"""
    impulse_positions = np.random.choice(len(signal), num_impulses, replace=False)
    impulse_signal = np.copy(signal)
    impulse_signal[impulse_positions] += magnitude * (
            np.random.randn(num_impulses) + 1j * np.random.randn(num_impulses)
    )
    return impulse_signal


def add_frequency_drift(signal, fs, max_shift_hz=1e4):
    """Simulate gradual random frequency drift over time (phase rotation)"""
    t = np.arange(len(signal)) / fs
    drift = np.cumsum(np.random.uniform(-max_shift_hz, max_shift_hz, size=len(signal))) / fs
    phase_drift = np.exp(2j * np.pi * drift)
    return signal * phase_drift


def add_random_dropout(signal, dropout_rate=0.01):
    """Randomly zero out samples in the signal"""
    mask = np.random.rand(len(signal)) > dropout_rate
    return signal * mask


def add_gain_fluctuation(signal, fluctuation_strength=0.3):
    """Apply fast gain variation (amplitude modulation)"""
    gain = 1 + fluctuation_strength * np.random.randn(len(signal))
    return signal * gain


def add_phase_jitter(signal, jitter_std=0.1):
    """Apply random phase noise (jitter) to each sample"""
    phase_noise = np.exp(1j * np.random.normal(0, jitter_std, len(signal)))
    return signal * phase_noise
