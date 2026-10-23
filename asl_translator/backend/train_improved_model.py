"""
Retrains the ASL recognition model on the complete dataset (all 1,740 images across 29 classes)
with scale-invariant feature extraction, adaptive forearm pruning, and scale augmentation.
Saves the improved model strictly to asl_translator/backend/models/sign_language_model.pkl.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from image_preprocessing import normalize_spatial_resolution
from dip_pipeline import execute_task5_pipeline
from hand_segmentation import hsv_skin_segmentation
from morphological_processing import clean_hand_mask
from sign_recognition import extract_hand_features

def main():
    dataset_root = CURRENT_DIR.parent.parent / "dataset"
    if not dataset_root.is_dir():
        dataset_root = Path("dataset")

    print(f"Loading full dataset from: {dataset_root.resolve()}")

    classes = sorted([
        d.name for d in dataset_root.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "asl-alphabet-test"
    ])
    print(f"Found {len(classes)} classes: {classes}")

    X, y = [], []
    t0 = time.time()
    total_imgs = 0

    for c in classes:
        # Collect from both base and test directories to maximize training variety
        p1 = list((dataset_root / c).glob("*.jpg")) + list((dataset_root / c).glob("*.png"))
        p2 = list((dataset_root / "asl-alphabet-test" / c).glob("*.jpg")) + list((dataset_root / "asl-alphabet-test" / c).glob("*.png"))
        all_paths = p1 + p2
        total_imgs += len(all_paths)

        for p in all_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue

            # Standard DIP pipeline matching inference
            norm = normalize_spatial_resolution(img, (200, 200))
            t5 = execute_task5_pipeline(norm, A=1.5)
            raw = hsv_skin_segmentation(norm)
            clean = clean_hand_mask(raw, prune_wrist=True)
            feats = extract_hand_features(clean, original_bgr=norm, gray_image=t5["high_boost"])

            if feats["has_hand"]:
                X.append(feats["features_vector"])
                y.append(c)

                # Scale augmentation: 0.85x centered on canvas to ensure robust scale invariance
                h, w = norm.shape[:2]
                scaled_w, scaled_h = int(w * 0.85), int(h * 0.85)
                resized_clean = cv2.resize(clean, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                resized_gray = cv2.resize(t5["high_boost"], (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                padded_clean = np.zeros_like(clean)
                padded_gray = np.zeros_like(t5["high_boost"])
                ox, oy = (w - scaled_w) // 2, (h - scaled_h) // 2
                padded_clean[oy:oy+scaled_h, ox:ox+scaled_w] = resized_clean
                padded_gray[oy:oy+scaled_h, ox:ox+scaled_w] = resized_gray

                feats_aug = extract_hand_features(padded_clean, gray_image=padded_gray)
                if feats_aug["has_hand"]:
                    X.append(feats_aug["features_vector"])
                    y.append(c)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    elapsed_extract = time.time() - t0
    print(f"Extracted DIP features for {len(X)} samples (including scale augmentation) in {elapsed_extract:.2f}s")

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation validation
    print("Evaluating 5-Fold Stratified Cross Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, max_features="sqrt")
    scores = cross_val_score(eval_model, X_scaled, y, cv=cv)
    print(f"5-Fold CV Accuracy: {scores.mean() * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")

    # Train final production model on all samples
    print("Training final production classifier on all samples...")
    t_train = time.time()
    final_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_split=2
    )
    final_model.fit(X_scaled, y)
    print(f"Model trained in {time.time() - t_train:.2f}s")

    # Save model checkpoint
    output_path = CURRENT_DIR / "models" / "sign_language_model.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": final_model,
        "scaler": scaler,
        "classes_": list(final_model.classes_),
        "model_type": "rf",
        "is_trained": True,
        "metadata": {
            "model_type": "RandomForestClassifier (300 estimators with scale augmentation)",
            "num_classes": len(final_model.classes_),
            "classes": list(final_model.classes_),
            "num_samples": int(len(X)),
            "cv_accuracy": f"{scores.mean() * 100:.2f}%",
            "feature_dim": int(X.shape[1]),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    joblib.dump(checkpoint, output_path, compress=3)
    print(f"Successfully saved updated model to: {output_path}")
    print(f"Checkpoint size: {output_path.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
