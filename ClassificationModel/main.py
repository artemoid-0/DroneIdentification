# ClassificationModel/main.py
from ClassificationModel.data_processing import collect_balanced_datasets, auto_select_folders
from ClassificationModel.classifier import train_classifier
import os


def main():
    # === Paths ===
    PROJECT_ROOT = os.path.dirname(__file__)
    DATASET_PATH = os.path.join(PROJECT_ROOT, "..", "SAVE_PATH", "MANY_NOISES")
    RESUME = True
    SAVE_PATH = os.path.join(PROJECT_ROOT, "..", "training_output",
                             "run_005 (diverse train (-15, 15, 20), moderate val (-5, -10), hard test (-20))")

    print("📂 DATASET_PATH:", DATASET_PATH)
    print("💾 SAVE_PATH:", SAVE_PATH)

    # === Collect all noise folders from all drones and ranges ===
    all_noise_folders = set()

    for drone_model in sorted(os.listdir(DATASET_PATH)):
        model_path = os.path.join(DATASET_PATH, drone_model)
        if not os.path.isdir(model_path):
            continue

        for band_folder in sorted(os.listdir(model_path)):
            band_path = os.path.join(model_path, band_folder)
            if not os.path.isdir(band_path):
                continue

            for noise_folder in os.listdir(band_path):
                full_path = os.path.join(band_path, noise_folder)
                if os.path.isdir(full_path):
                    all_noise_folders.add(noise_folder)

    all_noise_folders = sorted(all_noise_folders)
    print(f"\n📦 Found {len(all_noise_folders)} unique noise folders:")
    for f in all_noise_folders:
        print("   •", f)

    # === Split into train/val/test ===
    train_folders, val_folders, test_folders = auto_select_folders(all_noise_folders)

    print("\n🎓 TRAIN_FOLDERS:", train_folders)
    print("🧪 VAL_FOLDERS:", val_folders)
    print("🔬 TEST_FOLDERS:", test_folders)

    # === Collect dataset ===
    train_data, val_data, test_data, class_to_idx = collect_balanced_datasets(
        root_path=DATASET_PATH,
        train_folders=train_folders,
        val_folders=val_folders,
        test_folders=test_folders
    )

    print(
        f"\n🔎 Using {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test images from {len(class_to_idx)} classes.")

    train_classifier(train_data, val_data, class_to_idx,
                     test_data=test_data,
                     save_path=SAVE_PATH if RESUME else None,
                     resume=RESUME)


if __name__ == '__main__':
    main()
