# rfuav/NoiseAdding/noise/gpu.py

import os
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from scipy.signal import stft as cpu_stft
from NoiseAdding.noise.base import NoiseProcessor


class GPUProcessor(NoiseProcessor):
    def __init__(self, fs, stft_point, duration_time, cmap):
        super().__init__(fs, stft_point, duration_time, cmap)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_spectrogram(self, signal, save_dir, base_name):
        """
        Generate a spectrogram for each segment of the signal and save it as an image.
        """
        samples_per_segment = int(self.fs * self.duration_time)
        os.makedirs(save_dir, exist_ok=True)

        num_segments = len(signal) // samples_per_segment
        for i in range(num_segments):
            img_name = f"{base_name}_frame_{i:04d}.jpg"
            img_path = os.path.join(save_dir, img_name)
            if os.path.exists(img_path):
                continue  # Skip if the image already exists

            print(f"📷 [GPU] Generating spectrogram: {img_path}")
            segment = signal[i * samples_per_segment:(i + 1) * samples_per_segment]
            # Currently using the CPU version of STFT
            f, t, Zxx = cpu_stft(
                segment,
                fs=self.fs,
                nperseg=self.stft_point,
                window=windows.hamming(self.stft_point),
                return_onesided=False
            )
            Zxx = np.fft.fftshift(Zxx, axes=0)
            f = np.fft.fftshift(f)

            spectrum_db = 10 * np.log10(np.abs(Zxx) + 1e-12)
            extent = [t.min(), t.max(), f.min(), f.max()]

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(spectrum_db, extent=extent, aspect='auto', origin='lower', cmap=self.cmap)
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(img_path, dpi=300)
            plt.close(fig)
            plt.close('all')
            gc.collect()

    def apply_noise_profile(self, signal, profile_name, **kwargs):
        """
        Apply the specified noise profile to the signal.
        """
        signal_tensor = torch.from_numpy(signal).to(self.device)
        noisy_signal = self._apply_noise_profile(signal_tensor, profile_name, **kwargs)
        return noisy_signal.cpu().numpy()

    def _apply_noise_profile(self, signal, profile_name, **kwargs):
        """
        Internal method to apply noise profiles (currently supports AWGN).
        """
        if profile_name == 'awgn':
            snr_db = kwargs.get('snr_db', 0)
            power = torch.mean(signal.abs() ** 2)
            snr_linear = 10 ** (snr_db / 10)
            noise_power = power / snr_linear
            std = torch.sqrt(noise_power / 2)
            noise = std * (torch.randn_like(signal) + 1j * torch.randn_like(signal))
            return signal + noise

        raise ValueError(f"[GPU] Unknown noise profile: {profile_name}")

    def get_noise_configs(self):
        """
        Generate noise configurations for various SNR levels.
        """
        SNR_LEVELS = list(range(-20, 25, 5))  # SNR levels from -20 dB to +20 dB
        return [(f"AWGN_SNR_{snr:+03d}dB", dict(profile_name="awgn", snr_db=snr)) for snr in SNR_LEVELS]
