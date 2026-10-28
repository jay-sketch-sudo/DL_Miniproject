"""
Configuration and Hyperparameters for Image Captioning System.
"""
import os
import sys
from pathlib import Path

# Configure UTF-8 for Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Base Paths
BASE_DIR = Path(__file__).resolve().parent

# Ensure Keras and Matplotlib store their caches locally in the workspace
os.environ["KERAS_HOME"] = str(BASE_DIR / ".keras")
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".matplotlib")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOADS_DIR = STATIC_DIR / "uploads"

# Create required directories
for directory in [DATA_DIR, CHECKPOINTS_DIR, STATIC_DIR, TEMPLATES_DIR, UPLOADS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Paths (supports standard Flickr8k variations)
IMAGES_DIR = DATA_DIR / "Flickr8k_Dataset"
TEXT_DIR = DATA_DIR / "Flickr8k_text"
CAPTIONS_FILE_TOKEN = TEXT_DIR / "Flickr8k.token.txt"
CAPTIONS_FILE_CSV = DATA_DIR / "captions.txt"
TRAIN_IMAGES_TXT = TEXT_DIR / "Flickr_8k.trainImages.txt"
DEV_IMAGES_TXT = TEXT_DIR / "Flickr_8k.devImages.txt"
TEST_IMAGES_TXT = TEXT_DIR / "Flickr_8k.testImages.txt"

# Saved Model & Artifact Paths
FEATURES_FILE = CHECKPOINTS_DIR / "features_inceptionv3.pkl"
TOKENIZER_FILE = CHECKPOINTS_DIR / "tokenizer.pkl"
MODEL_FILE = CHECKPOINTS_DIR / "image_caption_model.keras"
BEST_MODEL_FILE = CHECKPOINTS_DIR / "best_model.keras"
CONFIG_JSON_FILE = CHECKPOINTS_DIR / "config.json"

# InceptionV3 Image Specs
IMAGE_SIZE = (299, 299)
IMAGE_SHAPE = (299, 299, 3)
FEATURE_DIM = 2048

# NLP & Model Hyperparameters
VOCAB_MIN_FREQUENCY = 1      # Retain words across the downloaded Flickr8k split
EMBEDDING_DIM = 256
LSTM_UNITS = 256
DENSE_UNITS = 256
DROPOUT_RATE = 0.4
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

# Special Tokens
START_TOKEN = "startseq"
END_TOKEN = "endseq"
UNK_TOKEN = "unkseq"
PAD_TOKEN = "padseq"

