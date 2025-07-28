# rfuav/NoiseAdding/noise/base.py
import gc
import os
import time
from abc import ABC, abstractmethod
import numpy as np
from multiprocessing import current_process
from datetime import datetime


class NoiseProcessor(ABC):
    def __init__(self, fs=100_000_000, stft_point=1024, duration_time=0.1,
                 cmap='hsv', log_path="logs/noise_log.txt"):
        self.fs = fs
        self.stft_point = stft_point
        self.duration_time = duration_time
        self.cmap = cmap
        self.log_path = log_path

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    @abstractmethod
    def apply_noise_profile(self, signal, profile_name, **kwargs):
        pass

    @abstractmethod
    def generate_spectrogram(self, signal, save_dir, base_name):
        pass

    @abstractmethod
    def get_noise_configs(self):
        pass

    def skip_if_exists(self, base_name, save_dir, signal):
        samples_per_segment = int(self.fs * self.duration_time)
        num_segments = len(signal) // samples_per_segment
        return all(
            os.path.exists(os.path.join(save_dir, f"{base_name}_frame_{i:04d}.jpg"))
            for i in range(num_segments)
        )

    def log_event(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        proc = current_process().name
        with open(self.log_path, "a") as f:
            f.write(f"[{timestamp}] [{proc}] {message}\n")

    def apply_noise_and_generate(self, signal, save_dir, base_name, noise_tag, noise_params, save_noisy=False):
        profile_name = noise_params.pop('profile_name')
        self.log_event(f"Start apply_noise_profile: {base_name} | {noise_tag}")
        t0 = time.perf_counter()
        noisy = self.apply_noise_profile(signal, profile_name, **noise_params)
        t1 = time.perf_counter()
        self.log_event(f"Finished apply_noise_profile: {base_name} | {noise_tag} in {t1 - t0:.2f}s")

        if save_noisy:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{base_name}_noisy.npy")
            np.save(path, noisy)
            self.log_event(f"Saved noisy .npy: {path}")

        self.log_event(f"Start generate_spectrogram: {base_name} | {noise_tag}")
        t2 = time.perf_counter()
        self.generate_spectrogram(noisy, save_dir, base_name)
        t3 = time.perf_counter()
        self.log_event(f"Finished generate_spectrogram: {base_name} | {noise_tag} in {t3 - t2:.2f}s")

        del noisy
        gc.collect()
