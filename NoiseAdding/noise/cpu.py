# rfuav/NoiseAdding/noise/cpu.py

import numpy as np
from NoiseAdding.noise.base import NoiseProcessor
import os
import gc
import matplotlib.pyplot as plt
from scipy.signal import stft, windows


class CPUProcessor(NoiseProcessor):
    def __init__(self, fs, stft_point, duration_time, cmap):
        # Initializes the CPUProcessor class that handles noise and spectrogram generation on the CPU
        super().__init__(fs, stft_point, duration_time, cmap)
        self.fs = fs
        self.stft_point = stft_point
        self.duration_time = duration_time
        self.cmap = cmap

    def generate_spectrogram(self, signal, save_dir, base_name):
        """
        Generate a spectrogram for each segment of the signal and save it as an image.
        """
        samples_per_segment = int(self.fs * self.duration_time)  # Number of samples per segment
        os.makedirs(save_dir, exist_ok=True)  # Create save directory if it doesn't exist

        num_segments = len(signal) // samples_per_segment  # Split signal into segments
        for i in range(num_segments):
            img_name = f"{base_name}_frame_{i:04d}.jpg"  # Image filename with frame number
            img_path = os.path.join(save_dir, img_name)
            if os.path.exists(img_path):
                continue  # Skip if the image already exists

            segment = signal[i * samples_per_segment:(i + 1) * samples_per_segment]
            f, t, Zxx = stft(segment, fs=self.fs, nperseg=self.stft_point,
                             window=windows.hamming(self.stft_point), return_onesided=False)
            Zxx = np.fft.fftshift(Zxx, axes=0)  # Shift FFT to center zero frequency
            f = np.fft.fftshift(f)

            spectrum_db = 10 * np.log10(np.abs(Zxx) + 1e-12)  # Convert to decibels
            extent = [t.min(), t.max(), f.min(), f.max()]  # Set spectrogram's axis limits

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(spectrum_db, extent=extent, aspect='auto', origin='lower', cmap=self.cmap)
            ax.axis('off')  # Hide axes
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(img_path, dpi=300)  # Save image with high resolution
            plt.close(fig)  # Close the figure to free up memory
            plt.close('all')
            gc.collect()  # Perform garbage collection to free memory

    def apply_noise_profile(self, signal, profile_name, **kwargs):
        """
        Apply the specified noise profile to the signal.
        """
        # Import noise profiles
        from NoiseAdding.noise.profiles import (
            add_awgn_noise, add_awgn_local_noise, add_tone_noise,
            add_band_noise, add_impulse_noise, add_frequency_drift,
            add_random_dropout, add_gain_fluctuation, add_phase_jitter
        )

        # Define a map of noise profiles and their corresponding functions
        noise_map = {
            'awgn': lambda s: add_awgn_noise(s, kwargs.get('snr_db', 0)),
            'awgn_local': lambda s: add_awgn_local_noise(s, kwargs.get('snr_db', 0), kwargs.get('window_size', 1024)),
            'tone': lambda s: add_tone_noise(s, self.fs, kwargs.get('tone_freq', 1e6), kwargs.get('amplitude', 0.3)),
            'band': lambda s: add_band_noise(s, self.fs, kwargs.get('center_freq', 5e6), kwargs.get('bandwidth', 1e6),
                                             kwargs.get('amplitude', 0.2)),
            'impulse': lambda s: add_impulse_noise(s, kwargs.get('num_impulses', 10), kwargs.get('magnitude', 5.0)),
            'drift': lambda s: add_frequency_drift(s, self.fs, kwargs.get('max_shift_hz', 1e4)),
            'dropout': lambda s: add_random_dropout(s, kwargs.get('dropout_rate', 0.01)),
            'gain': lambda s: add_gain_fluctuation(s, kwargs.get('fluctuation_strength', 0.3)),
            'phase': lambda s: add_phase_jitter(s, kwargs.get('jitter_std', 0.1)),
        }

        if profile_name not in noise_map:
            raise ValueError(f"Unknown noise profile: {profile_name}")

        return noise_map[profile_name](signal)  # Apply the chosen noise profile

    def get_noise_configs(self):
        """
        Generate noise configurations for various SNR levels and window sizes.
        """
        SNR_LEVELS = list(range(-20, 25, 5))  # SNR levels from -20 dB to +20 dB
        WINDOW_SIZES = [64, 128, 256, 512, 1024, 2048]  # Window sizes for local AWGN noise

        configs = [
            *[(f"AWGN_SNR_{snr:+03d}dB", dict(profile_name="awgn", snr_db=snr)) for snr in SNR_LEVELS],  # AWGN with different SNRs
            *[(f"AWGN_Local_{snr:+03d}dB_win{ws}", dict(profile_name="awgn_local", snr_db=snr, window_size=ws))
              for snr in SNR_LEVELS for ws in WINDOW_SIZES]  # Local AWGN with different window sizes
        ]
        return configs
