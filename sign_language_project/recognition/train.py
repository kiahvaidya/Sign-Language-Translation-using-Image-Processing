"""
Model Training Script & Pipeline
Extracts DIP features across the dataset, trains a classical ML model,
evaluates performance, saves evaluation plots, and stores the model artifact.
Can be executed as a CLI command or invoked programmatically from Streamlit.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add project root to sys.path to enable direct CLI execution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.acquisition import find_dataset_images
from preprocessing.filtering import task5_integrated_pipeline
from preprocessing.segmentation import hsv_skin_segmentation
from preprocessing.morphology import clean_hand_mask
from features.feature_extraction import extract_hand_features
from recognition.classifier import SignLanguageClassifier


def find_default_dataset_dir():
    """
    Locates the dataset directory automatically across common workspace locations.
    """
    candidates = [
        PROJECT_ROOT / "data" / "dataset",
        PROJECT_ROOT.parent / "dataset",
        PROJECT_ROOT / "dataset",
        Path("dataset"),
        Path("../dataset")
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            # Check if directory contains non-empty subfolders
            subdirs = [d for d in c.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if len(subdirs) > 0:
                return str(c.resolve())
    return None


def extract_features_from_dataset(dataset_dir, selected_classes=None, max_per_class=None, progress_callback=None):
    """
    Iterates through dataset, runs full DIP pipeline on every image,
    and collects extracted feature vectors.
    """
    dataset_map = find_dataset_images(dataset_dir)
    if not dataset_map:
        raise ValueError(f"No valid images found in dataset directory: {dataset_dir}")

    if selected_classes is not None:
        target_classes = [c for c in selected_classes if c in dataset_map]
    else:
        target_classes = sorted(list(dataset_map.keys()))

    X, y = [], []
    total_imgs = sum(
        min(len(dataset_map[c]), max_per_class if max_per_class else 999999)
        for c in target_classes
    )
    processed_count = 0

    print(f"Starting DIP feature extraction for {len(target_classes)} classes ({total_imgs} total images)...")

    for class_name in target_classes:
        img_paths = dataset_map[class_name]
        if max_per_class:
            img_paths = img_paths[:max_per_class]

        for p in img_paths:
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                continue

            # DIP Pipeline
            # 1. Task 5 Enhancement
            t5 = task5_integrated_pipeline(img_bgr)
            high_boost_gray = t5["high_boost"]

            # 2. Task 6 Segmentation (Robust skin mask)
            raw_mask = hsv_skin_segmentation(img_bgr)

            # 3. Task 8 Morphological Cleaning
            cleaned_mask = clean_hand_mask(raw_mask)

            # 4. Feature Extraction
            feat_data = extract_hand_features(
                cleaned_mask,
                original_bgr=img_bgr,
                gray_image=high_boost_gray
            )

            if feat_data["has_hand"]:
                X.append(feat_data["features_vector"])
                y.append(class_name)

            processed_count += 1
            if progress_callback:
                progress_callback(processed_count, total_imgs, f"Processing {class_name} ({processed_count}/{total_imgs})")

    return np.array(X, dtype=np.float32), np.array(y), target_classes


def plot_and_save_confusion_matrix(cm, classes, output_path):
    """
    Plots and saves confusion matrix heatmap.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - DIP Sign Language Recognition", fontsize=14, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Label values in matrix cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, format(val, "d"),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color=color, fontsize=9)

    plt.ylabel("True Class", fontsize=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_file), dpi=300, bbox_inches="tight")
    plt.close()
    return str(out_file)


def run_training(
    dataset_dir=None,
    model_type="rf",
    selected_classes=None,
    max_per_class=None,
    test_size=0.25,
    output_model_path=None,
    progress_callback=None
):
    """
    Full training routine returning performance metrics and trained classifier.
    """
    if dataset_dir is None:
        dataset_dir = find_default_dataset_dir()
        if dataset_dir is None:
            raise FileNotFoundError("Could not automatically locate the dataset directory.")

    if output_model_path is None:
        output_model_path = PROJECT_ROOT / "models" / "sign_language_model.pkl"

    start_time = time.time()
    X, y, classes = extract_features_from_dataset(
        dataset_dir,
        selected_classes=selected_classes,
        max_per_class=max_per_class,
        progress_callback=progress_callback
    )

    if len(X) < 10:
        raise ValueError(f"Extracted too few valid samples ({len(X)}) to train a classifier.")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Train Classifier
    classifier = SignLanguageClassifier(model_type=model_type)
    classifier.fit(X_train, y_train)

    # Evaluation
    eval_results = classifier.evaluate(X_test, y_test)
    eval_results["train_samples"] = len(X_train)
    eval_results["test_samples"] = len(X_test)
    eval_results["feature_dim"] = int(X.shape[1])
    eval_results["training_time_sec"] = round(time.time() - start_time, 2)

    # Save model
    classifier.save(output_model_path)
    eval_results["saved_model_path"] = str(Path(output_model_path).resolve())

    # Save confusion matrix plot
    cm_path = PROJECT_ROOT / "outputs" / "recognition" / "confusion_matrix.png"
    plot_and_save_confusion_matrix(eval_results["confusion_matrix"], classifier.classes_, cm_path)
    eval_results["confusion_matrix_image"] = str(cm_path.resolve())

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Model Type       : {model_type.upper()}")
    print(f"Number of Classes: {len(classifier.classes_)} ({', '.join(classifier.classes_)})")
    print(f"Total Samples    : {len(X)} (Train: {len(X_train)}, Test: {len(X_test)})")
    print(f"Feature Dimension: {X.shape[1]}")
    print(f"Accuracy         : {eval_results['accuracy'] * 100:.2f}%")
    print(f"Precision        : {eval_results['precision'] * 100:.2f}%")
    print(f"Recall           : {eval_results['recall'] * 100:.2f}%")
    print(f"F1 Score         : {eval_results['f1_score'] * 100:.2f}%")
    print(f"Model Saved At   : {output_model_path}")
    print(f"CM Plot Saved At : {cm_path}")
    print("=" * 60)

    return classifier, eval_results


def main():
    parser = argparse.ArgumentParser(description="Train Sign Language DIP Classifier")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--model_type", type=str, default="rf", choices=["rf", "svm", "knn"], help="Model type")
    parser.add_argument("--classes", nargs="+", default=None, help="Subset of class names to train on")
    parser.add_argument("--max_per_class", type=int, default=None, help="Max images per class")
    parser.add_argument("--test_size", type=float, default=0.25, help="Test set fraction")
    parser.add_argument("--output", type=str, default=None, help="Output model path")

    args = parser.parse_args()

    run_training(
        dataset_dir=args.dataset,
        model_type=args.model_type,
        selected_classes=args.classes,
        max_per_class=args.max_per_class,
        test_size=args.test_size,
        output_model_path=args.output
    )


if __name__ == "__main__":
    main()
