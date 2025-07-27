from NoiseAdding.noise.cpu import CPUProcessor
from NoiseAdding.noise.gpu import GPUProcessor
from NoiseAdding.noise.base import NoiseProcessor
import os
import gc
import time
import numpy as np
from multiprocessing import Pool, current_process
from multiprocessing.shared_memory import SharedMemory


class RFUAVPipeline:
    def __init__(self, num_processes, fs=100_000_000, stft_point=1024, duration_time=0.1,
                 cmap='hsv', device='cpu'):
        self.fs = fs
        self.stft_point = stft_point
        self.duration_time = duration_time
        self.cmap = cmap
        self.device = device
        self.num_processes = num_processes

        if device == 'cpu':
            self.processor: NoiseProcessor = CPUProcessor(
                fs=fs, stft_point=stft_point,
                duration_time=duration_time, cmap=cmap
            )
        elif device == 'cuda':
            self.processor: NoiseProcessor = GPUProcessor(
                fs=fs, stft_point=stft_point,
                duration_time=duration_time, cmap=cmap
            )
        else:
            raise NotImplementedError(f"Device '{device}' is not supported.")

    def get_reorganized_path(self, save_root, relative_path, noise_tag):
        parts = relative_path.split(os.sep)
        band = parts[0] if len(parts) > 1 else "default_band"
        return os.path.join(save_root, band, noise_tag)

    def handle_noise_task(self, args):
        (
            shm_name, shape, dtype_str,
            relative_path, save_root,
            noise_tag, noise_params, save_noisy,
            base_name
        ) = args

        pid = current_process().name

        try:
            existing_shm = SharedMemory(name=shm_name)
            flat_data = np.ndarray(shape, dtype=np.float32, buffer=existing_shm.buf)
            complex_signal = flat_data[::2] + 1j * flat_data[1::2]
        except Exception as e:
            print(f"[{pid}] ❌ Ошибка доступа к shared memory '{shm_name}': {e}")
            return

        save_dir = self.get_reorganized_path(save_root, relative_path, noise_tag)
        if self.processor.skip_if_exists(base_name, save_dir, complex_signal):
            print(f"[{pid}] ⏩ {base_name} | {noise_tag}: уже существует, пропуск.")
            existing_shm.close()
            return

        print(f"[{pid}] 🔧 {base_name} | {noise_tag}: начало обработки.")
        start_time = time.time()

        self.processor.apply_noise_and_generate(
            complex_signal, save_dir, base_name, noise_tag, noise_params, save_noisy
        )

        existing_shm.close()
        print(f"[{pid}] ✅ {base_name} | {noise_tag}: завершено за {time.time() - start_time:.2f} сек")

    def process_single_iq(self, iq_path, relative_path, save_root, save_noisy):
        base_name = os.path.splitext(os.path.basename(iq_path))[0]

        try:
            raw_data = np.fromfile(iq_path, dtype=np.float32)
            complex_signal = raw_data[::2] + 1j * raw_data[1::2]
        except Exception as e:
            print(f"⚠️ Ошибка при чтении {iq_path}: {e}")
            return

        print(f"📂 Оригинал: {relative_path}")
        original_dir = self.get_reorganized_path(save_root, relative_path, "original")
        start_original = time.time()
        self.processor.generate_spectrogram(complex_signal, original_dir, base_name)
        print(f"📷 Оригинал {base_name} сохранён за {time.time() - start_original:.2f} сек")

        # === Подготовка shared memory ===
        stacked = np.empty((complex_signal.size * 2,), dtype=np.float32)
        stacked[::2] = complex_signal.real
        stacked[1::2] = complex_signal.imag

        shm = SharedMemory(create=True, size=stacked.nbytes)
        shm_buffer = np.ndarray(stacked.shape, dtype=np.float32, buffer=shm.buf)
        np.copyto(shm_buffer, stacked)

        # === Подготовка задач ===
        noise_configs = self.processor.get_noise_configs()
        tasks = [
            (
                shm.name, stacked.shape, stacked.dtype.name,
                relative_path, save_root,
                noise_tag, noise_params, save_noisy,
                base_name
            )
            for noise_tag, noise_params in noise_configs
        ]

        # === Параллельная обработка ===
        with Pool(processes=self.num_processes) as pool:
            pool.map(self.handle_noise_task, tasks)

        shm.close()
        shm.unlink()

    def process_all(self, root_path, save_root, save_noisy=False):
        for subdir, _, files in os.walk(root_path):
            for file in files:
                if file.lower().endswith(".iq"):
                    full_path = os.path.join(subdir, file)
                    relative_path = os.path.relpath(full_path, root_path)
                    print(f"🔹 Обработка файла: {relative_path}")
                    self.process_single_iq(full_path, relative_path, save_root, save_noisy)

        gc.collect()
