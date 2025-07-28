# rfuav/NoiseAdding/main.py

import os
from multiprocessing import cpu_count
from NoiseAdding.pipeline.processor import RFUAVPipeline

# Path to the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = r'D:\DATASETS\RFUAV\SIGNALS\FLYSKY FS I6X'

# Here, SAVE_PATH will refer to the project root
SAVE_PATH = os.path.join(BASE_DIR, "SAVE_PATH", "MANY_NOISES", "FLYSKY FS I6X (TEST)")

if __name__ == "__main__":
    pipeline = RFUAVPipeline(device='cpu', num_processes=min(4, cpu_count()))
    pipeline.process_all(DATA_PATH, SAVE_PATH)
