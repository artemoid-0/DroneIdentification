import os
from multiprocessing import cpu_count
from NoiseAdding.pipeline.processor import RFUAVPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "DATASETS", "RFUAV", "SIGNALS", "FLYSKY FS I6X")
SAVE_PATH = os.path.join(BASE_DIR, "SAVE_PATH", "MANY_NOISES", "FLYSKY FS I6X (TEST)")

if __name__ == "__main__":
    pipeline = RFUAVPipeline(device='cpu', num_processes=min(4, cpu_count()))
    pipeline.process_all(DATA_PATH, SAVE_PATH)
