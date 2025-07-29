# ClassificationModel/classifier.py
import json
import os
import random
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

IMG_SIZE = 224

# === Training transformations ===
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img + 0.01 * torch.randn_like(img)),  # Light noise
    transforms.RandomErasing(p=0.3, scale=(0.01, 0.05), ratio=(0.3, 3.3), value=0),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# === Validation/Test transformations ===
REALISTIC_EVAL = False

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.05, contrast=0.05) if REALISTIC_EVAL else transforms.Lambda(lambda x: x),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img + 0.005 * torch.randn_like(img) if REALISTIC_EVAL else img),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class DroneSpectrogramDataset(Dataset):
    def __init__(self, image_label_pairs, transform=None):
        self.image_label_pairs = image_label_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        img_path, label = self.image_label_pairs[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_next_run_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("run_")]
    numbers = [int(d.split('_')[1]) for d in existing if d.split('_')[1].isdigit()]
    next_id = max(numbers, default=0) + 1
    return os.path.join(base_dir, f"run_{next_id:03d}")


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))  # Slightly increase the figure size
    im = ax.imshow(cm, cmap="Blues")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, ha="right")  # Rotate and align
    ax.set_yticklabels(labels)

    # Labels and title
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Display numbers inside the cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



def train_classifier(train_data, val_data, class_to_idx, test_data=None, epochs=20, batch_size=16,
                     lr=1e-4, patience=5, save_base="training_output", save_path=None, resume=False):
    # === Reproducibility ===
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if save_path is None:
        save_path = get_next_run_dir(save_base)
    else:
        os.makedirs(save_path, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file_path = os.path.join(save_path, f"log_classifier.txt")
    log_file = open(log_file_path, "a" if resume else "w", encoding="utf-8")

    def log(msg):
        print(msg)
        print(msg, file=log_file)

    def classification_report_to_str(y_true, y_pred):
        return classification_report(y_true, y_pred, digits=4)

    train_loader = DataLoader(DroneSpectrogramDataset(train_data, train_transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DroneSpectrogramDataset(val_data, eval_transform), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(DroneSpectrogramDataset(test_data, eval_transform), batch_size=batch_size,
                             shuffle=False) if test_data else None

    log("\n🔍 Train class distribution: " + str(Counter([label for _, label in train_data])))
    log("🔍 Val class distribution: " + str(Counter([label for _, label in val_data])))

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
    if torch.cuda.is_available():
        model = model.cuda()

    model_save_path = os.path.join(save_path, "best_model.pth")
    if resume and os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        log(f"🔄 Continuing training from {model_save_path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch + 1}/{epochs}] Training", leave=False)
        for images, labels in train_bar:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        # === Validation ===
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="[Validation]", leave=False):
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = sum([p == t for p, t in zip(y_pred, y_true)]) / len(y_true)
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        log(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            log("✅ New best model saved.")
            log("📄 Classification Report (Validation):")
            log(classification_report_to_str(y_true, y_pred))

            pred_csv_path = os.path.join(save_path, "val_predictions.csv")
            with open(pred_csv_path, "w") as f:
                f.write("true,predicted\n")
                for t, p in zip(y_true, y_pred):
                    f.write(f"{t},{p}\n")

            cm_path = os.path.join(save_path, "confusion_matrix_val.png")
            plot_confusion_matrix(y_true, y_pred, list(class_to_idx.keys()), cm_path)
        else:
            patience_counter += 1
            log(f"⏳ EarlyStopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                log("🛑 Early stopping.")
                break

    meta_info = {
        "timestamp": timestamp,
        "epochs_trained": epoch + 1,
        "best_f1": best_f1,
        "train_distribution": dict(Counter([label for _, label in train_data])),
        "val_distribution": dict(Counter([label for _, label in val_data])),
        "class_to_idx": class_to_idx,
        "img_size": IMG_SIZE,
        "batch_size": batch_size,
        "learning_rate": lr
    }
    meta_path = os.path.join(save_path, "best_model_info.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    if test_loader:
        log("\n🧪 Model evaluation on test data:")
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        y_true_test, y_pred_test = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="[Testing]", leave=False):
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                y_true_test.extend(labels.cpu().numpy())
                y_pred_test.extend(predicted.cpu().numpy())

        log("📄 Classification Report (Test):")
        log(classification_report_to_str(y_true_test, y_pred_test))

        pred_csv_path = os.path.join(save_path, "test_predictions.csv")
        with open(pred_csv_path, "w") as f:
            f.write("true,predicted\n")
            for t, p in zip(y_true_test, y_pred_test):
                f.write(f"{t},{p}\n")

        cm_path = os.path.join(save_path, "confusion_matrix_test.png")
        plot_confusion_matrix(y_true_test, y_pred_test, list(class_to_idx.keys()), cm_path)

    log_file.close()
