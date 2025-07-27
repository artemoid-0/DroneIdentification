from ClassificationModel.data_processing import collect_balanced_datasets, auto_select_folders
from ClassificationModel.classifier import train_classifier
import os


def main():
    # === Пути ===
    PROJECT_ROOT = os.path.dirname(__file__)
    DATASET_PATH = os.path.join(PROJECT_ROOT, "..", "SAVE_PATH", "MANY_NOISES")
    RESUME = True
    SAVE_PATH = os.path.join(PROJECT_ROOT, "..", "training_output", "run_002_(less noise to train)")

    print("📂 DATASET_PATH:", DATASET_PATH)
    print("💾 SAVE_PATH:", SAVE_PATH)

    # === Сбор всех шумовых папок из всех дронов и диапазонов ===
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
    print(f"\n📦 Найдено {len(all_noise_folders)} уникальных шумовых папок:")
    for f in all_noise_folders:
        print("   •", f)

    # === Деление на train/val/test ===
    train_folders, val_folders, test_folders = auto_select_folders(all_noise_folders)

    print("\n🎓 TRAIN_FOLDERS:", train_folders)
    print("🧪 VAL_FOLDERS:", val_folders)
    print("🔬 TEST_FOLDERS:", test_folders)


    # === Сбор датасета ===
    train_data, val_data, test_data, class_to_idx = collect_balanced_datasets(
        root_path=DATASET_PATH,
        train_folders=train_folders,
        val_folders=val_folders,
        test_folders=test_folders
    )

    print(
        f"\n🔎 Используем {len(train_data)} тренировочных, {len(val_data)} валидационных и {len(test_data)} тестовых изображений из {len(class_to_idx)} классов.")


    train_classifier(train_data, val_data, class_to_idx,
                     test_data=test_data,
                     save_path=SAVE_PATH if RESUME else None,
                     resume=RESUME)


if __name__ == '__main__':
    main()
