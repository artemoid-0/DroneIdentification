# ClassificationModel/data_processing.py
import os
import random
from collections import defaultdict
from glob import glob

# === Constants ===
NUM_CLASSES = 8
TRAIN_SPLIT_RATIO = 0.8

TRAIN_SNR_RANGE = [-15, 15, 20]
VAL_SNR_VALUES = [-5, -10]
TEST_SNR_VALUES = [-20, ]

TRAIN_LOC_WINS = [256, 512]
VAL_LOC_WINS = [64, 256]
TEST_LOC_WINS = [64, 1024, 2048]


# === Auto-selection of folders ===
def auto_select_folders(all_folders):
    train_folders = []
    val_folders = []
    test_folders = []

    for folder in all_folders:
        if folder == "original":
            train_folders.append(folder)
            continue

        if folder.startswith("AWGN_SNR_"):
            snr_str = folder.replace("AWGN_SNR_", "").replace("dB", "")
            try:
                snr = int(snr_str)
            except ValueError:
                continue

            if snr in TRAIN_SNR_RANGE:
                train_folders.append(folder)
            elif snr in VAL_SNR_VALUES:
                val_folders.append(folder)
            elif snr in TEST_SNR_VALUES:
                test_folders.append(folder)

        elif folder.startswith("AWGN_Local_"):
            parts = folder.split('_')
            try:
                snr = int(parts[2].replace("dB", ""))
                win = int(parts[3].replace("win", ""))
            except (IndexError, ValueError):
                continue

            if snr in TRAIN_SNR_RANGE and win in TRAIN_LOC_WINS:
                train_folders.append(folder)
            elif snr in VAL_SNR_VALUES and win in VAL_LOC_WINS:
                val_folders.append(folder)
            elif snr in TEST_SNR_VALUES or win in TEST_LOC_WINS:
                test_folders.append(folder)

    return sorted(train_folders), sorted(val_folders), sorted(test_folders)


# === Collecting a balanced dataset from different sources ===
def collect_balanced_datasets(root_path, train_folders, val_folders, test_folders, train_ratio=0.8, val_ratio=0.166,
                              test_ratio=0.034):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "The sum of the ratios must equal 1.0"

    drone_class_dirs = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
    class_to_idx = {name: idx for idx, name in enumerate(drone_class_dirs)}

    train_pairs, val_pairs, test_pairs = [], [], []
    min_total_images = float('inf')
    image_index = defaultdict(lambda: defaultdict(list))  # drone -> folder -> list[paths]

    for drone in drone_class_dirs:
        base_path = os.path.join(root_path, drone)
        vtsbw_dirs = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        if not vtsbw_dirs:
            continue
        first_band = vtsbw_dirs[0]
        band_base_path = os.path.join(base_path, first_band)

        original_path = os.path.join(band_base_path, "original")
        if not os.path.isdir(original_path):
            continue

        original_images = sorted(glob(os.path.join(original_path, '*.jpg')))
        image_index[drone]['original'] = original_images
        min_total_images = min(min_total_images, len(original_images))

        for folder in set(train_folders + val_folders + test_folders):
            if folder == 'original':
                continue
            folder_path = os.path.join(band_base_path, folder)
            if os.path.isdir(folder_path):
                image_index[drone][folder] = sorted(glob(os.path.join(folder_path, '*.jpg')))

    IMAGES_PER_CLASS = min_total_images
    train_count = int(IMAGES_PER_CLASS * train_ratio)
    val_count = int(IMAGES_PER_CLASS * val_ratio)
    test_count = IMAGES_PER_CLASS - train_count - val_count

    print(f"\n📏 IMAGES_PER_CLASS = {IMAGES_PER_CLASS} (Train: {train_count}, Val: {val_count}, Test: {test_count})")

    distribution_summary = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    reserved_filenames = defaultdict(lambda: {'val': set(), 'test': set()})

    for drone in drone_class_dirs:
        class_idx = class_to_idx[drone]
        print(f"\n📦 Drone model: {drone} (class {class_idx})")

        all_val_candidates = []
        for folder in val_folders:
            all_val_candidates.extend(image_index[drone].get(folder, []))

        all_test_candidates = []
        for folder in test_folders:
            all_test_candidates.extend(image_index[drone].get(folder, []))

        all_filenames = list(set(os.path.basename(p) for p in all_val_candidates + all_test_candidates))
        if len(all_filenames) < val_count + test_count:
            print(f"  ⚠️ Not enough unique images ({len(all_filenames)} < {val_count + test_count})")
            continue

        reserved = random.sample(all_filenames, val_count + test_count)
        reserved_filenames[drone]['val'].update(reserved[:val_count])
        reserved_filenames[drone]['test'].update(reserved[val_count:])

        for folder in val_folders:
            for path in image_index[drone].get(folder, []):
                filename = os.path.basename(path)
                if filename in reserved_filenames[drone]['val']:
                    val_pairs.append((path, class_idx))
                    distribution_summary[drone]['val'] += 1
                    print(f"\033[91m🟥 {filename:<35} -> VAL   ({folder})\033[0m")

        for folder in test_folders:
            for path in image_index[drone].get(folder, []):
                filename = os.path.basename(path)
                if filename in reserved_filenames[drone]['test']:
                    test_pairs.append((path, class_idx))
                    distribution_summary[drone]['test'] += 1
                    print(f"\033[94m🟦 {filename:<35} -> TEST  ({folder})\033[0m")

        for folder in train_folders:
            usable = [p for p in image_index[drone].get(folder, [])
                      if os.path.basename(p) not in reserved_filenames[drone]['val']
                      and os.path.basename(p) not in reserved_filenames[drone]['test']]
            if len(usable) < train_count:
                continue
            chosen = random.sample(usable, train_count)
            for path in chosen:
                train_pairs.append((path, class_idx))
                distribution_summary[drone]['train'] += 1
                print(f"\033[92m🟩 {os.path.basename(path):<35} -> TRAIN ({folder})\033[0m")

    print("\n📊 Final distribution by class:")
    for drone, dist in distribution_summary.items():
        print(f"  {drone:<25} | TRAIN: {dist['train']:>3} | VAL: {dist['val']:>3} | TEST: {dist['test']:>3}")

    return train_pairs, val_pairs, test_pairs, class_to_idx
