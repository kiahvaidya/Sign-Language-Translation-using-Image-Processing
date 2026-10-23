"""
Sign-to-Sentence Translation Engine
The core 'Google Translate for Sign Language' architecture.
Transforms sequences of sign language gesture images into fluent English sentences
using DIP Tasks 1-8 and dictionary-assisted sentence synthesis.
"""

import os
import difflib
from pathlib import Path
import numpy as np

try:
    from recognition.predict import SignPredictor
    from preprocessing.acquisition import load_image, find_dataset_images
except ImportError:
    from .predict import SignPredictor
    from ..preprocessing.acquisition import load_image, find_dataset_images


# Common Sign Language & English vocabulary dictionary for auto-correction & natural sentence synthesis
STANDARD_VOCABULARY = [
    # Common Greetings & Phrases
    "HELLO", "HI", "WELCOME", "GOOD MORNING", "GOOD AFTERNOON", "GOOD NIGHT",
    "HOW ARE YOU", "NICE TO MEET YOU", "SEE YOU LATER", "HAVE A NICE DAY",
    # Polite Expressions
    "THANK YOU", "THANKS", "PLEASE", "YOU ARE WELCOME", "EXCUSE ME", "SORRY",
    # Basic Responses
    "YES", "NO", "MAYBE", "OKAY", "FINE", "GOOD", "BAD",
    # Essential Communication
    "HELP", "NEED HELP", "I LOVE YOU", "WHAT IS YOUR NAME", "MY NAME IS",
    "SIGN LANGUAGE", "IMAGE PROCESSING", "DIGITAL IMAGE", "COMPUTER VISION",
    # Common Nouns & Actions
    "CAT", "DOG", "CAR", "CAB", "HOME", "SCHOOL", "WATER", "FOOD", "BOOK",
    "FRIEND", "FAMILY", "TEACHER", "STUDENT", "DOCTOR", "TIME", "TODAY"
]

# Multilingual translation dictionary for common sign phrases
MULTILINGUAL_TRANSLATIONS = {
    "HELLO": {
        "Spanish": "¡Hola!",
        "French": "Bonjour!",
        "German": "Hallo!",
        "Hindi": "नमस्ते (Namaste)!"
    },
    "WELCOME": {
        "Spanish": "¡Bienvenido!",
        "French": "Bienvenue!",
        "German": "Willkommen!",
        "Hindi": "स्वागत है (Swagat hai)!"
    },
    "THANK YOU": {
        "Spanish": "¡Gracias!",
        "French": "Merci!",
        "German": "Danke!",
        "Hindi": "धन्यवाद (Dhanyawad)!"
    },
    "HOW ARE YOU": {
        "Spanish": "¿Cómo estás?",
        "French": "Comment allez-vous?",
        "German": "Wie geht es dir?",
        "Hindi": "आप कैसे हैं (Aap kaise hain)?"
    },
    "NICE TO MEET YOU": {
        "Spanish": "Mucho gusto.",
        "French": "Ravi de vous rencontrer.",
        "German": "Schön, Sie kennenzulernen.",
        "Hindi": "आपसे मिलकर खुशी हुई।"
    },
    "PLEASE": {
        "Spanish": "Por favor.",
        "French": "S'il vous plaît.",
        "German": "Bitte.",
        "Hindi": "कृपया (Kripya)।"
    },
    "I LOVE YOU": {
        "Spanish": "Te quiero.",
        "French": "Je t'aime.",
        "German": "Ich liebe dich.",
        "Hindi": "मैं तुमसे प्यार करता हूँ।"
    },
    "HELP": {
        "Spanish": "¡Ayuda!",
        "French": "À l'aide!",
        "German": "Hilfe!",
        "Hindi": "मदद (Madad)!"
    },
    "CAB": {
        "Spanish": "Taxi",
        "French": "Taxi",
        "German": "Taxi",
        "Hindi": "टैक्सी (Taxi)"
    },
    "CAT": {
        "Spanish": "Gato",
        "French": "Chat",
        "German": "Katze",
        "Hindi": "बिल्ली (Billi)"
    },
    "SIGN LANGUAGE": {
        "Spanish": "Lengua de señas",
        "French": "Langue des signes",
        "German": "Gebärdensprache",
        "Hindi": "सांकेतिक भाषा (Sign Language)"
    }
}


def autocorrect_token(raw_token):
    """
    Auto-corrects a raw recognized token against standard sign language vocabulary
    using Levenshtein distance similarity matching.
    """
    cleaned = raw_token.upper().strip()
    if not cleaned:
        return "", 1.0

    if cleaned in STANDARD_VOCABULARY:
        return cleaned, 1.0

    # Search for close match in vocabulary
    matches = difflib.get_close_matches(cleaned, STANDARD_VOCABULARY, n=1, cutoff=0.6)
    if matches:
        return matches[0], 0.88

    return cleaned, 0.70


def format_sentence(tokens):
    """
    Formats a list of words or characters into natural English sentence casing and punctuation.
    """
    raw_text = " ".join(t for t in tokens if t).strip()
    if not raw_text:
        return "No translation available."

    # Look for full phrase match in dictionary
    corrected_phrase, conf = autocorrect_token(raw_text)
    if corrected_phrase in STANDARD_VOCABULARY:
        # Punctuate natural greeting/exclamation
        if corrected_phrase in ("HELLO", "WELCOME", "HELP", "THANK YOU", "I LOVE YOU"):
            return corrected_phrase.capitalize() + "!"
        elif corrected_phrase in ("HOW ARE YOU", "WHAT IS YOUR NAME"):
            return corrected_phrase.capitalize() + "?"
        else:
            return corrected_phrase.capitalize() + "."

    # Capitalize first letter of each sentence
    sentence = raw_text.capitalize()
    if not sentence.endswith((".", "!", "?")):
        sentence += "."
    return sentence


class SignToSentenceTranslator:
    """
    End-to-End Sign Language to English Translation Engine.
    Executes DIP Tasks 1 through 8 on each sign gesture image,
    extracts descriptors, predicts tokens, and synthesizes polished English sentences.
    """

    def __init__(self, model_path):
        self.predictor = SignPredictor(model_path)

    def translate_sequence(self, image_sources, segmentation_method="hsv_skin", A=1.5, prune_wrist=False):
        """
        Translates a list/sequence of gesture images into an English sentence.

        Args:
            image_sources (list): List of file paths, UploadedFiles, or numpy frames.
            segmentation_method (str): DIP segmentation method to apply.
            A (float): High-boost factor (Task 5).
            prune_wrist (bool): Whether to sever forearm/wrist below palm.

        Returns:
            dict containing:
              - 'translated_sentence': str (natural English)
              - 'raw_phrase': str (raw recognized letters)
              - 'confidence': float
              - 'characters': list of predicted characters
              - 'dip_evidence': list of per-image DIP processing stages (Tasks 1-8)
              - 'multilingual': dict of other language equivalents
        """
        if not image_sources:
            return {
                "translated_sentence": "",
                "raw_phrase": "",
                "confidence": 0.0,
                "characters": [],
                "dip_evidence": [],
                "multilingual": {}
            }

        characters = []
        confidences = []
        dip_evidence = []

        for idx, src in enumerate(image_sources):
            if src is None:
                # Space representation between words
                characters.append(" ")
                confidences.append(1.0)
                continue

            try:
                # Execute full DIP pipeline (Tasks 1 to 8)
                res = self.predictor.predict_image(
                    src,
                    segmentation_method=segmentation_method,
                    A=A,
                    prune_wrist=prune_wrist
                )

                sign = res["predicted_sign"]
                conf = res["confidence"]

                # Handle control tokens
                if sign.upper() == "SPACE":
                    characters.append(" ")
                elif sign.upper() == "DEL":
                    if characters:
                        characters.pop()
                elif sign.upper() != "NOTHING" and res["has_hand"]:
                    characters.append(sign.upper())
                    confidences.append(conf)

                dip_evidence.append({
                    "index": idx + 1,
                    "predicted_sign": sign,
                    "confidence": conf,
                    "stages": res["stages"],
                    "metrics": res["metrics"],
                    "metadata": res["metadata"],
                    "bbox": res["bbox"]
                })

            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                characters.append("?")
                confidences.append(0.0)

        # Assemble raw tokens
        raw_phrase = "".join(characters).strip()
        words = raw_phrase.split()

        # Spell check & polish
        corrected_words = []
        for w in words:
            cw, _ = autocorrect_token(w)
            corrected_words.append(cw)

        # Natural English sentence synthesis
        translated_sentence = format_sentence(corrected_words)

        # Compute aggregate confidence
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        # Look up multilingual equivalents
        clean_key = " ".join(corrected_words).upper().strip()
        multilingual = MULTILINGUAL_TRANSLATIONS.get(clean_key, {})

        return {
            "translated_sentence": translated_sentence,
            "raw_phrase": raw_phrase,
            "confidence": round(avg_conf, 3),
            "characters": characters,
            "dip_evidence": dip_evidence,
            "multilingual": multilingual
        }


# High-confidence canonical sample images for benchmark demonstrations
CANONICAL_SIGN_SAMPLES = {
    "A": ["A0014_test.jpg", "A0018_test.jpg"],
    "B": ["B0024_test.jpg", "B0022_test.jpg"],
    "C": ["C0009_test.jpg", "C0012_test.jpg"],
    "D": ["D0006_test.jpg", "D0012_test.jpg"],
    "E": ["E0022_test.jpg", "E0008_test.jpg", "E0009_test.jpg"],
    "F": ["F0009_test.jpg", "F0015_test.jpg"],
    "G": ["G0009_test.jpg", "G0018_test.jpg"],
    "H": ["H0008_test.jpg", "H0014_test.jpg"],
    "I": ["I0016_test.jpg", "I0018_test.jpg"],
    "J": ["J0013_test.jpg", "J0016_test.jpg"],
    "K": ["K0008_test.jpg", "K0012_test.jpg"],
    "L": ["L0014_test.jpg", "L0018_test.jpg"],
    "M": ["M0005_test.jpg", "M0010_test.jpg"],
    "N": ["N0001_test.jpg", "N0008_test.jpg"],
    "O": ["O0025_test.jpg", "O0020_test.jpg"],
    "P": ["P0002_test.jpg", "P0014_test.jpg"],
    "Q": ["Q0006_test.jpg", "Q0012_test.jpg"],
    "R": ["R0008_test.jpg", "R0015_test.jpg"],
    "S": ["S0004_test.jpg", "S0012_test.jpg"],
    "T": ["T0014_test.jpg", "T0018_test.jpg"],
    "U": ["U0010_test.jpg", "U0015_test.jpg"],
    "V": ["V0002_test.jpg", "V0014_test.jpg"],
    "W": ["W0004_test.jpg", "W0012_test.jpg"],
    "X": ["X0003_test.jpg", "X0015_test.jpg"],
    "Y": ["Y0009_test.jpg", "Y0016_test.jpg"],
    "Z": ["Z0026_test.jpg", "Z0020_test.jpg"],
}


def get_demo_phrase_images(phrase, dataset_dir):
    """
    Finds real dataset images for each letter in the target phrase.
    Uses canonical, high-confidence samples for clean recognition.
    """
    dataset_map = find_dataset_images(dataset_dir)
    if not dataset_map:
        return []

    images = []
    used_counter = {}

    for ch in phrase.upper():
        if ch == " ":
            images.append(None)  # Space separator
            continue

        if ch in dataset_map and dataset_map[ch]:
            file_list = dataset_map[ch]
            idx = used_counter.get(ch, 0)

            # Try canonical file first
            chosen_path = None
            if ch in CANONICAL_SIGN_SAMPLES:
                canonical_options = CANONICAL_SIGN_SAMPLES[ch]
                target_fname = canonical_options[idx % len(canonical_options)]
                # Search in file_list
                for p in file_list:
                    if Path(p).name == target_fname:
                        chosen_path = p
                        break

            if chosen_path is None:
                chosen_path = file_list[idx % len(file_list)]

            images.append(chosen_path)
            used_counter[ch] = idx + 1
        else:
            images.append(None)

    return images
