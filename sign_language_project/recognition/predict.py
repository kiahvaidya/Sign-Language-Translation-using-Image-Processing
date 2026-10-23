"""
Inference & Prediction Pipeline Module
Connects the complete DIP processing pipeline (Tasks 1 -> 8) directly to the trained classifier,
returning structured predictions, confidence ratings, and English translation mappings.
"""

import cv2
import numpy as np
from pathlib import Path
try:
    from preprocessing.acquisition import load_image
    from preprocessing.filtering import task5_integrated_pipeline
    from preprocessing.segmentation import (
        otsu_threshold,
        global_threshold,
        adaptive_threshold,
        hsv_skin_segmentation
    )
    from preprocessing.morphology import clean_hand_mask
    from features.feature_extraction import extract_hand_features
    from recognition.classifier import SignLanguageClassifier
except ImportError:
    from ..preprocessing.acquisition import load_image
    from ..preprocessing.filtering import task5_integrated_pipeline
    from ..preprocessing.segmentation import (
        otsu_threshold,
        global_threshold,
        adaptive_threshold,
        hsv_skin_segmentation
    )
    from ..preprocessing.morphology import clean_hand_mask
    from ..features.feature_extraction import extract_hand_features
    from .classifier import SignLanguageClassifier


# Dictionary mapping sign class identifiers to user-friendly English outputs
ENGLISH_TRANSLATION_MAP = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "E": "E",
    "F": "F",
    "G": "G",
    "H": "H",
    "I": "I",
    "J": "J",
    "K": "K",
    "L": "L",
    "M": "M",
    "N": "N",
    "O": "O",
    "P": "P",
    "Q": "Q",
    "R": "R",
    "S": "S",
    "T": "T",
    "U": "U",
    "V": "V",
    "W": "W",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "SPACE": " ",
    "DEL": "[BACKSPACE]",
    "NOTHING": "[NOTHING]",
    "HELLO": "Hello",
    "THANK YOU": "Thank You",
    "YES": "Yes",
    "NO": "No",
    "PLEASE": "Please"
}


class SignPredictor:
    """
    End-to-end DIP sign language inference engine.
    """

    def __init__(self, model_path=None):
        self.classifier = None
        self.model_path = model_path
        if model_path is not None and Path(model_path).is_file():
            self.load_model(model_path)

    def load_model(self, model_path):
        """
        Loads the trained model checkpoint.
        """
        self.classifier = SignLanguageClassifier.load(model_path)
        self.model_path = model_path
        return self

    def predict_image(self, image_source, segmentation_method="hsv_skin", A=1.5, prune_wrist=False, min_sat=26):
        """
        Runs the full DIP pipeline on an input image:
        1. Acquisition & validation
        2. Task 5 Integrated Enhancement (Contrast Stretch -> Median -> High Boost)
        3. Task 6 Segmentation (Selected method)
        4. Task 8 Morphological Cleaning (Opening -> Closing -> Largest Hand Component -> Optional Wrist Pruning)
        5. Feature Extraction (Geometry, Hu Moments, Projections, Zoning, HOG)
        6. Classification & English Translation

        Returns:
            dict containing prediction results, probabilities, DIP stages, and metrics.
        """
        # STEP 1: Image Acquisition
        original_bgr, original_rgb, original_gray, metadata = load_image(image_source)

        # STEP 2: Task 5 Integrated Enhancement
        enhancement_stages = task5_integrated_pipeline(original_bgr, A=A)
        enhanced_bgr = enhancement_stages["contrast_stretched"]
        filtered_bgr = enhancement_stages["median_filtered"]
        high_boost_gray = enhancement_stages["high_boost"]

        # STEP 3: Task 6 Segmentation
        if segmentation_method == "otsu":
            raw_mask, _ = otsu_threshold(high_boost_gray, invert=False)
        elif segmentation_method == "otsu_inv":
            raw_mask, _ = otsu_threshold(high_boost_gray, invert=True)
        elif segmentation_method == "adaptive":
            raw_mask = adaptive_threshold(high_boost_gray)
        elif segmentation_method == "global":
            raw_mask = global_threshold(high_boost_gray, 127)
        elif segmentation_method == "hsv_skin":
            raw_mask = hsv_skin_segmentation(original_bgr, min_sat=min_sat)
        elif segmentation_method == "hybrid":
            skin = hsv_skin_segmentation(original_bgr, min_sat=min_sat)
            otsu_m, _ = otsu_threshold(high_boost_gray, invert=False)
            raw_mask = cv2.bitwise_or(skin, otsu_m)
        else:
            raw_mask = hsv_skin_segmentation(original_bgr, min_sat=min_sat)

        # STEP 4: Task 8 Morphological Processing
        cleaned_mask = clean_hand_mask(raw_mask, open_iter=1, close_iter=2, keep_largest=True, prune_wrist=prune_wrist)

        # STEP 5: Feature Extraction
        feature_data = extract_hand_features(
            cleaned_mask,
            original_bgr=original_bgr,
            gray_image=high_boost_gray
        )

        has_hand = feature_data["has_hand"]
        features_vector = feature_data["features_vector"]
        vis_image = feature_data["visualization"]
        metrics = feature_data["metrics"]

        predicted_sign = "UNKNOWN"
        confidence = 0.0
        top_probs = {}

        # STEP 6: Classifier Inference
        if self.classifier is not None and self.classifier.is_trained:
            if has_hand:
                pred_label = self.classifier.predict(features_vector)[0]
                predicted_sign = str(pred_label)

                # Get confidence
                probs = self.classifier.predict_proba(features_vector)[0]
                classes = self.classifier.classes_

                class_prob_pairs = sorted(
                    zip(classes, probs),
                    key=lambda x: x[1],
                    reverse=True
                )
                top_probs = {str(c): float(round(p, 4)) for c, p in class_prob_pairs[:3]}
                confidence = float(round(top_probs.get(str(predicted_sign), 0.0), 4))
            else:
                predicted_sign = "No Hand Detected"
                confidence = 0.0
        else:
            predicted_sign = "Model Not Loaded"

        # English Translation
        clean_sign = predicted_sign.upper().strip()
        english = ENGLISH_TRANSLATION_MAP.get(clean_sign, predicted_sign)

        stages = {
            "original_rgb": original_rgb,
            "original_bgr": original_bgr,
            "enhanced_bgr": enhanced_bgr,
            "filtered_bgr": filtered_bgr,
            "high_boost_gray": high_boost_gray,
            "raw_mask": raw_mask,
            "cleaned_mask": cleaned_mask,
            "visualization": cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        }

        return {
            "predicted_sign": predicted_sign,
            "english_translation": english,
            "confidence": confidence,
            "top_probabilities": top_probs,
            "has_hand": has_hand,
            "stages": stages,
            "metrics": metrics,
            "metadata": metadata,
            "bbox": feature_data["bbox"]
        }
