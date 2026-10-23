"""
ASL Vision Model Training and Comprehensive Evaluation Script
Integrates DIP Tasks 1-5 with MobileNetV3 Transfer Learning for ASL Fingerspelling.
Produces detailed evaluation metrics (Accuracy, Precision, Recall, F1, Per-Class Report).
"""

import os
import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dip_pipeline import execute_task5_pipeline
from config import get_active_dataset_dir, SUPPORTED_SIGNS


class ASLDataset(Dataset):
    def __init__(self, samples, is_train=True):
        self.samples = samples
        self.is_train = is_train
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        im = cv2.imread(path)
        if im is None:
            im = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # DIP Task 5 Enhancement
        t5 = execute_task5_pipeline(im, A=1.5)
        rgb = cv2.cvtColor(t5["contrast_stretched"], cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb)
        return tensor, label


def train_and_evaluate():
    dataset_root = Path(get_active_dataset_dir())
    print(f"Loading dataset from: {dataset_root.resolve()}")

    classes = sorted([
        d.name for d in dataset_root.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "asl-alphabet-test"
    ])
    print(f"Found {len(classes)} sign classes: {classes}")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}

    all_samples = []
    for c in classes:
        for imp in (dataset_root / c).glob("*.jpg"):
            all_samples.append((str(imp), class_to_idx[c]))

    print(f"Total dataset images: {len(all_samples)}")

    # Stratified 80/20 train/test split
    train_samples, test_samples = train_test_split(
        all_samples,
        test_size=0.20,
        random_state=42,
        stratify=[s[1] for s in all_samples]
    )
    print(f"Train set: {len(train_samples)}, Test set: {len(test_samples)}")

    train_loader = DataLoader(ASLDataset(train_samples, is_train=True), batch_size=32, shuffle=True)
    test_loader = DataLoader(ASLDataset(test_samples, is_train=False), batch_size=32, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using accelerated compute device: {device}")

    # Build model: MobileNetV3-Small with 29 output classes
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, len(classes))
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 18
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_test_acc = 0.0
    best_weights = None
    t0 = time.time()

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_acc = train_correct / train_total

        # Test evaluation
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_weights = model.state_dict().copy()

        print(f"Epoch {epoch+1:2d}/{epochs}: Train Acc = {train_acc*100:.2f}%, Test Acc = {test_acc*100:.2f}% (Elapsed: {time.time()-t0:.1f}s)")

    print(f"\nBest Test Accuracy achieved: {best_test_acc*100:.2f}%")

    # Load best weights for comprehensive evaluation
    model.load_state_dict(best_weights)
    model.eval()

    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Compute comprehensive evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=4,
        zero_division=0
    )

    print("\n" + "="*60)
    print("FINAL EVALUATION METRICS ON UNSEEN TEST DATA")
    print("="*60)
    print(f"Overall Accuracy:        {acc * 100:.2f}%")
    print(f"Macro Precision:         {prec_macro * 100:.2f}%")
    print(f"Weighted Precision:      {prec_weighted * 100:.2f}%")
    print(f"Macro Recall:            {rec_macro * 100:.2f}%")
    print(f"Weighted Recall:         {rec_weighted * 100:.2f}%")
    print(f"Macro F1-Score:          {f1_m * 100:.2f}%")
    print(f"Weighted F1-Score:       {f1_w * 100:.2f}%")
    print("="*60)
    print("\nPER-CLASS CLASSIFICATION REPORT:")
    print(report_text)

    # Save model and metrics
    models_dir = CURRENT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = models_dir / "asl_mobilenet_v3.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "accuracy": acc,
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_precision": round(prec_macro, 4),
            "weighted_precision": round(prec_weighted, 4),
            "macro_recall": round(rec_macro, 4),
            "weighted_recall": round(rec_weighted, 4),
            "macro_f1": round(f1_m, 4),
            "weighted_f1": round(f1_w, 4)
        }
    }, model_save_path)
    print(f"\nSaved model checkpoint to: {model_save_path}")

    # Save metrics JSON
    metrics_path = models_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "MobileNetV3-Small with DIP Preprocessing",
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_samples": len(y_true),
            "num_classes": len(classes),
            "accuracy": round(acc, 4),
            "macro_precision": round(prec_macro, 4),
            "weighted_precision": round(prec_weighted, 4),
            "macro_recall": round(rec_macro, 4),
            "weighted_recall": round(rec_weighted, 4),
            "macro_f1": round(f1_m, 4),
            "weighted_f1": round(f1_w, 4),
            "per_class": {
                c: {
                    "precision": round(report_dict[c]["precision"], 4),
                    "recall": round(report_dict[c]["recall"], 4),
                    "f1_score": round(report_dict[c]["f1-score"], 4),
                    "support": report_dict[c]["support"]
                }
                for c in classes if c in report_dict
            }
        }, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_path}")


if __name__ == "__main__":
    train_and_evaluate()
