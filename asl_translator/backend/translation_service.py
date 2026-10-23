"""
Translation Service Module
Coordinates DIP Tasks 1 through 8 and performs intelligent word-level synthesis
and dictionary-assisted autocorrection using comprehensive English vocabulary.
"""

import difflib
import numpy as np
from pathlib import Path
from typing import List, Tuple

from config import (
    HIGH_BOOST_A,
    STANDARD_VOCABULARY,
    MULTILINGUAL_DICTIONARY
)
from image_preprocessing import (
    load_image,
    normalize_spatial_resolution,
    encode_image_base64
)
from dip_pipeline import execute_task5_pipeline
from hand_segmentation import (
    hsv_skin_segmentation,
    adaptive_hand_segmentation,
    otsu_threshold,
    adaptive_threshold,
    region_growing
)
from morphological_processing import clean_hand_mask
from sign_recognition import extract_hand_features, SignRecognizer


# ==========================================================
# COMPREHENSIVE ENGLISH DICTIONARY LOADER
# ==========================================================

# ==========================================================
# COMPREHENSIVE ENGLISH DICTIONARY LOADER
# ==========================================================

_COMMON_WORDS_SET = None
_DICTIONARY_SET = None

def get_common_words() -> set:
    """
    Returns prioritized high-frequency conversational English vocabulary.
    """
    global _COMMON_WORDS_SET
    if _COMMON_WORDS_SET is not None:
        return _COMMON_WORDS_SET

    words = set()
    for w in STANDARD_VOCABULARY:
        for sub_w in w.upper().split():
            if len(sub_w) > 1:
                words.add(sub_w)

    top_common = [
        "HOLLOW", "HELLO", "WORLD", "HOUSE", "WATER", "FRIEND", "FAMILY",
        "SCHOOL", "MOTHER", "FATHER", "SISTER", "BROTHER", "MORNING", "NIGHT",
        "TODAY", "TOMORROW", "PLEASE", "THANK", "THANKS", "WELCOME", "SORRY", "EXCUSE",
        "HAPPY", "SMILE", "PEACE", "LEARN", "TEACH", "STUDENT", "DOCTOR",
        "NURSE", "ANIMAL", "TIGER", "HORSE", "ZEBRA", "YELLOW", "ORANGE",
        "PURPLE", "BLACK", "WHITE", "GREEN", "BROWN", "APPLE", "BANANA",
        "BREAD", "CHEESE", "CHICKEN", "DINNER", "LUNCH", "BREAKFAST",
        "WINDOW", "DOOR", "FLOOR", "TABLE", "CHAIR", "PHONE", "CAMERA",
        "BOOK", "GOOD", "FINE", "YES", "NO", "MAYBE", "OKAY", "HELP",
        "LOVE", "NAME", "SIGN", "LANGUAGE", "IMAGE", "COMPUTER", "VISION",
        "CAT", "DOG", "CAR", "CAB", "TIME", "WORK", "PLAY", "READ", "WRITE",
        "TALK", "LOOK", "SEE", "HEAR", "FEEL", "TOUCH", "RUN", "WALK",
        "OPEN", "CLOSE", "LIGHT", "DARK", "HOT", "COLD", "WARM", "TREE",
        "FLOWER", "RIVER", "OCEAN", "STAR", "SKY", "SUN", "MOON", "RAIN",
        "CITY", "ROAD", "TRAIN", "PLANE", "BOAT", "MUSIC", "SONG", "DANCE"
    ]
    for w in top_common:
        words.add(w.upper())

    _COMMON_WORDS_SET = words
    return _COMMON_WORDS_SET


def get_english_dictionary() -> set:
    """
    Loads comprehensive English vocabulary set combining macOS /usr/share/dict/words
    with standard ASL and conversational word lists for O(1) instant validation.
    """
    global _DICTIONARY_SET
    if _DICTIONARY_SET is not None:
        return _DICTIONARY_SET

    words = set(get_common_words())

    # macOS system dictionary if available
    system_dict = Path("/usr/share/dict/words")
    if system_dict.is_file():
        try:
            with open(system_dict, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    cand = line.strip().upper()
                    if 2 <= len(cand) <= 18 and cand.isalpha():
                        words.add(cand)
        except Exception as e:
            print(f"Notice: Could not load system dict: {e}")

    _DICTIONARY_SET = words
    return _DICTIONARY_SET


def smart_word_autocorrect(raw_word: str) -> Tuple[str, float]:
    """
    Performs intelligent word-level spelling resolution:
    1. If raw_word is already recognized (e.g. 'HOLLOW', 'CAT', 'HELLO'):
       Accepts it directly with 1.0 confidence!
    2. If not exact, searches prioritized common words first (len +/- 1, cutoff >= 0.70).
    3. If still unmatched, searches the broader dictionary (len +/- 1, cutoff >= 0.80).
    4. Otherwise preserves the user's spelled letters as-is.
    """
    cleaned = raw_word.upper().strip()
    if not cleaned:
        return "", 1.0

    common_words = get_common_words()
    dict_words = get_english_dictionary()

    # Direct exact dictionary match (e.g. HOLLOW, HELLO, CAT)
    if cleaned in common_words or cleaned in dict_words:
        return cleaned, 1.0

    # Tier 1: Check prioritized common vocabulary first (resolves typos like HBLLO -> HELLO)
    if len(cleaned) >= 3:
        common_cands = [w for w in common_words if abs(len(w) - len(cleaned)) <= 1]
        matches = difflib.get_close_matches(cleaned, common_cands, n=1, cutoff=0.70)
        if matches:
            return matches[0], 0.90

    # Tier 2: Check broader dictionary with strict cutoff
    if len(cleaned) >= 3:
        broad_cands = [w for w in dict_words if abs(len(w) - len(cleaned)) <= 1]
        matches = difflib.get_close_matches(cleaned, broad_cands, n=1, cutoff=0.80)
        if matches:
            return matches[0], 0.85

    # Fallback to user's exact letters
    return cleaned, 0.70


def format_sentence_from_words(words: List[str]) -> str:
    """
    Formats recognized word list into natural English punctuation and casing.
    """
    valid_words = [w.strip() for w in words if w.strip()]
    if not valid_words:
        return "No sign gestures detected."

    full_phrase = " ".join(valid_words).upper()
    exclamations = {"HELLO", "WELCOME", "HELP", "THANK YOU", "THANKS", "I LOVE YOU", "GOOD MORNING", "GOOD NIGHT"}
    questions = {"HOW ARE YOU", "WHAT IS YOUR NAME", "WHERE", "WHY", "WHAT", "WHEN"}

    capitalized = " ".join(w.capitalize() for w in valid_words)

    if full_phrase in exclamations:
        return capitalized + "!"
    elif full_phrase in questions or full_phrase.startswith(("WHAT", "HOW", "WHERE", "WHY", "WHO")):
        return capitalized + "?"
    elif not capitalized.endswith((".", "!", "?")):
        return capitalized + "."

    return capitalized


def format_sentence(tokens_or_words) -> str:
    """
    General sentence formatter accepting tokens or words list.
    """
    if not tokens_or_words:
        return "No sign gestures detected."

    # If list of single character tokens, assemble into words
    if all(len(t) == 1 or t in ("space", "del", "nothing") for t in tokens_or_words):
        words = []
        curr = []
        for t in tokens_or_words:
            if t in ("space", " "):
                if curr:
                    words.append("".join(curr))
                    curr = []
            elif t not in ("del", "nothing", "", "?"):
                curr.append(t)
        if curr:
            words.append("".join(curr))
        resolved_words = [smart_word_autocorrect(w)[0] for w in words]
        return format_sentence_from_words(resolved_words)

    resolved_words = [smart_word_autocorrect(w)[0] for w in tokens_or_words if w]
    return format_sentence_from_words(resolved_words)



class TranslationService:
    """
    Core translation coordinator executing DIP Tasks 1 through 8.
    """

    def __init__(self):
        self.recognizer = SignRecognizer.get_instance()

    def translate_single_image(self, image_source, prune_wrist=True, segmentation_method="hsv_skin"):
        """
        Executes full DIP pipeline on single image:
        1. Task 1: Image Acquisition
        2. Task 2: Image Representation (RGB, Gray, HSV, Normalized 200x200)
        3. Tasks 3, 4, 5: Integrated Pipeline (Contrast Stretch -> Median -> High-Boost)
        4. Tasks 6 & 7: Hand Segmentation
        5. Task 8: Morphological Cleaning & Adaptive Wrist Pruning
        6. Feature Extraction & Classifier Prediction
        """
        # Step 1: Acquisition & Validation
        img_bgr, orig_rgb, orig_gray, orig_hsv, metadata = load_image(image_source)

        # Step 2: Spatial Representation Normalization
        img_norm_bgr = normalize_spatial_resolution(img_bgr)

        # Step 3: Tasks 3, 4, 5 Integrated Enhancement
        task5_stages = execute_task5_pipeline(img_norm_bgr, A=HIGH_BOOST_A)
        high_boost_gray = task5_stages["high_boost"]

        # Step 4: Tasks 6 & 7 Segmentation
        if segmentation_method == "otsu":
            raw_mask, _ = otsu_threshold(high_boost_gray)
        elif segmentation_method == "adaptive":
            raw_mask = adaptive_threshold(high_boost_gray)
        elif segmentation_method == "region_growing":
            raw_mask, seed_pt = region_growing(high_boost_gray, threshold=25)
        elif segmentation_method == "hsv_skin":
            raw_mask = hsv_skin_segmentation(img_norm_bgr)
        else:
            raw_mask = adaptive_hand_segmentation(img_norm_bgr)

        # Step 5: Task 8 Morphological Processing
        cleaned_mask = clean_hand_mask(raw_mask, open_iter=1, close_iter=2, keep_largest=True, prune_wrist=prune_wrist)

        # Step 6: Feature Extraction
        feat_data = extract_hand_features(cleaned_mask, original_bgr=img_norm_bgr, gray_image=high_boost_gray)

        predicted_sign = "UNKNOWN"
        confidence = 0.0
        top_probs = {}

        if feat_data["has_hand"]:
            predicted_sign, confidence, top_probs = self.recognizer.predict(
                feat_data["features_vector"],
                image_bgr=img_norm_bgr
            )
        else:
            predicted_sign = "No Hand Detected"
            confidence = 0.0

        clean_sign = predicted_sign.upper().strip()
        english_translation = clean_sign

        stages_b64 = {
            "acquisition_rgb": encode_image_base64(orig_rgb),
            "high_boost_gray": encode_image_base64(high_boost_gray),
            "raw_mask": encode_image_base64(raw_mask),
            "cleaned_mask": encode_image_base64(cleaned_mask),
            "visualization": encode_image_base64(feat_data["visualization"], is_bgr=True)
        }

        multilingual = MULTILINGUAL_DICTIONARY.get(clean_sign, {})

        return {
            "predicted_sign": predicted_sign,
            "english_translation": english_translation,
            "confidence": round(confidence, 4),
            "top_probabilities": top_probs,
            "has_hand": feat_data["has_hand"],
            "stages": stages_b64,
            "metrics": feat_data["metrics"],
            "metadata": metadata,
            "multilingual": multilingual
        }

    def translate_image_sequence(self, image_sources, prune_wrist=True):
        """
        Translates an ordered list/sequence of gesture images into an English word or sentence.
        Letters are grouped into words and resolved with smart dictionary preservation.
        """
        if not image_sources:
            return {
                "translated_sentence": "",
                "characters": [],
                "confidence": 0.0,
                "dip_evidence": [],
                "multilingual": {}
            }

        characters = []
        confidences = []
        dip_evidence = []

        for idx, src in enumerate(image_sources):
            if src is None:
                characters.append(" ")
                confidences.append(1.0)
                continue

            try:
                res = self.translate_single_image(src, prune_wrist=prune_wrist)
                sign = res["predicted_sign"].upper()
                conf = res["confidence"]

                if sign == "SPACE":
                    characters.append(" ")
                elif sign == "DEL":
                    if characters:
                        characters.pop()
                elif sign != "NOTHING" and res["has_hand"]:
                    characters.append(sign)
                    confidences.append(conf)

                dip_evidence.append({
                    "index": idx + 1,
                    "predicted_sign": res["predicted_sign"],
                    "confidence": conf,
                    "stages": res["stages"],
                    "metrics": res["metrics"],
                    "metadata": res["metadata"]
                })
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                characters.append("?")
                confidences.append(0.0)

        # Assemble letters into spelled words
        words = []
        curr_chars = []
        for ch in characters:
            if ch == " ":
                if curr_chars:
                    words.append("".join(curr_chars))
                    curr_chars = []
            elif ch not in ("?", "DEL", "NOTHING"):
                curr_chars.append(ch)
        if curr_chars:
            words.append("".join(curr_chars))

        # Resolve spelling using smart dictionary
        resolved_words = []
        for w in words:
            resolved_w, _ = smart_word_autocorrect(w)
            resolved_words.append(resolved_w)

        final_sentence = format_sentence_from_words(resolved_words)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        lookup_key = "".join(resolved_words).upper()
        multilingual = MULTILINGUAL_DICTIONARY.get(lookup_key, {})

        return {
            "translated_sentence": final_sentence,
            "raw_phrase": "".join(characters).strip(),
            "characters": characters,
            "confidence": round(mean_conf, 4),
            "dip_evidence": dip_evidence,
            "multilingual": multilingual
        }
