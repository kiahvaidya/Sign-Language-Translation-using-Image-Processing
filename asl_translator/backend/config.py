"""
ASL-to-English Translator Configuration Module
Centralized settings, model paths, vocabulary mappings, and DIP constants.
"""

import os
from pathlib import Path

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent.parent

# Model Paths (primary in asl_translator, fallback to sign_language_project)
MODEL_PATH = BACKEND_DIR / "models" / "sign_language_model.pkl"
PYTORCH_MODEL_PATH = BACKEND_DIR / "models" / "asl_mobilenet_v3.pth"
EVALUATION_METRICS_PATH = BACKEND_DIR / "models" / "evaluation_metrics.json"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "sign_language_project" / "models" / "sign_language_model.pkl"

def get_active_model_path():
    if PYTORCH_MODEL_PATH.is_file():
        return str(PYTORCH_MODEL_PATH)
    if MODEL_PATH.is_file():
        return str(MODEL_PATH)
    if FALLBACK_MODEL_PATH.is_file():
        return str(FALLBACK_MODEL_PATH)
    return str(MODEL_PATH)

# Dataset Paths
DATASET_DIR = PROJECT_ROOT / "dataset"
FALLBACK_DATASET_DIR = PROJECT_ROOT / "sign_language_project" / "data" / "dataset"

def get_active_dataset_dir():
    if DATASET_DIR.is_dir() and any(DATASET_DIR.iterdir()):
        return str(DATASET_DIR)
    if FALLBACK_DATASET_DIR.is_dir() and any(FALLBACK_DATASET_DIR.iterdir()):
        return str(FALLBACK_DATASET_DIR)
    return str(DATASET_DIR)

# Sample Video Path
SAMPLE_VIDEO_PATH = BACKEND_DIR / "data" / "sample_video.mp4"
FALLBACK_SAMPLE_VIDEO = PROJECT_ROOT / "sign_language_project" / "data" / "sample_video.mp4"

def get_active_sample_video():
    if SAMPLE_VIDEO_PATH.is_file():
        return str(SAMPLE_VIDEO_PATH)
    if FALLBACK_SAMPLE_VIDEO.is_file():
        return str(FALLBACK_SAMPLE_VIDEO)
    return None

# Temporary storage for uploaded videos and files
TEMP_DIR = BACKEND_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Vocabulary: 29 Classes in standard ASL alphabet dataset
SUPPORTED_SIGNS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
]

# Standard English vocabulary for dictionary-assisted sign sequence synthesis
STANDARD_VOCABULARY = [
    "HELLO", "HI", "WELCOME", "GOOD MORNING", "GOOD AFTERNOON", "GOOD NIGHT",
    "HOW ARE YOU", "NICE TO MEET YOU", "SEE YOU LATER", "HAVE A NICE DAY",
    "THANK YOU", "THANKS", "PLEASE", "YOU ARE WELCOME", "EXCUSE ME", "SORRY",
    "YES", "NO", "MAYBE", "OKAY", "FINE", "GOOD", "BAD",
    "HELP", "NEED HELP", "I LOVE YOU", "WHAT IS YOUR NAME", "MY NAME IS",
    "SIGN LANGUAGE", "IMAGE PROCESSING", "DIGITAL IMAGE", "COMPUTER VISION",
    "CAT", "DOG", "CAR", "CAB", "HOME", "SCHOOL", "WATER", "FOOD", "BOOK",
    "FRIEND", "FAMILY", "TEACHER", "STUDENT", "DOCTOR", "TIME", "TODAY"
]

# Multilingual equivalents for common phrases
MULTILINGUAL_DICTIONARY = {
    "HELLO": {"Spanish": "¡Hola!", "French": "Bonjour!", "German": "Hallo!", "Hindi": "नमस्ते (Namaste)!"},
    "WELCOME": {"Spanish": "¡Bienvenido!", "French": "Bienvenue!", "German": "Willkommen!", "Hindi": "स्वागत है (Swagat hai)!"},
    "THANK YOU": {"Spanish": "¡Gracias!", "French": "Merci!", "German": "Danke!", "Hindi": "धन्यवाद (Dhanyawad)!"},
    "PLEASE": {"Spanish": "Por favor.", "French": "S'il vous plaît.", "German": "Bitte.", "Hindi": "कृपया (Kripya)।"},
    "HELP": {"Spanish": "¡Ayuda!", "French": "À l'aide!", "German": "Hilfe!", "Hindi": "मदद (Madad)!"},
    "I LOVE YOU": {"Spanish": "Te quiero.", "French": "Je t'aime.", "German": "Ich liebe dich.", "Hindi": "मैं तुमसे प्यार करता हूँ।"},
    "CAB": {"Spanish": "Taxi", "French": "Taxi", "German": "Taxi", "Hindi": "टैक्सी (Taxi)"},
    "CAT": {"Spanish": "Gato", "French": "Chat", "German": "Katze", "Hindi": "बिल्ली (Billi)"},
    "WATER": {"Spanish": "Agua", "French": "Eau", "German": "Wasser", "Hindi": "पानी (Paani)"},
    "YES": {"Spanish": "Sí", "French": "Oui", "German": "Ja", "Hindi": "हाँ (Haan)"},
    "NO": {"Spanish": "No", "French": "Non", "German": "Nein", "Hindi": "नहीं (Nahin)"}
}

# DIP Pipeline Processing Parameters
TARGET_SIZE = (200, 200)          # Standard resolution matching training dataset
HIGH_BOOST_A = 1.5                # Task 5 High-Boost scaling factor
MEDIAN_KERNEL = 5                 # Task 4 5x5 Median filter for impulse noise
MIN_SKIN_SATURATION = 26          # HSV saturation threshold floor to exclude walls/white shirts
MAX_SKIN_HUE = 25                 # HSV hue upper bound
WRIST_CUTOFF_FACTOR = 1.25        # Distance-transform palm multiplier to prune forearm

# File size & request validation limits
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
