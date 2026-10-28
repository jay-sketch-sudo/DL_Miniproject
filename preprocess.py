"""
Data Preprocessing Module for Image Captioning.

Handles:
1. Feature extraction using pretrained InceptionV3 CNN.
2. Caption loading, cleaning, and formatting (<startseq> ... <endseq>).
3. Vocabulary construction, filtering rare words, and Tokenizer serialization.
4. Train/Validation/Test split parsing.
"""
import os
import sys

# Configure UTF-8 for Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import re
import string
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import config

# Lazy load keras modules to allow running without full TF import if needed
def get_inception_feature_extractor():
    """Initializes and returns InceptionV3 feature extraction model."""
    import tensorflow as tf
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    
    local_weights = config.BASE_DIR / ".keras" / "models" / "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    if local_weights.exists():
        print(f"🧠 Loading pretrained InceptionV3 from cached local weights: {local_weights}...")
        base_model = InceptionV3(weights=str(local_weights), include_top=False, pooling="avg")
    else:
        print("🧠 Loading pretrained InceptionV3 (ImageNet)...")
        base_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    return base_model

def preprocess_single_image(image_path_or_file):
    """
    Loads and preprocesses an image for InceptionV3 (299x299, scaled to [-1, 1]).
    """
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    
    if isinstance(image_path_or_file, (str, Path)):
        img = Image.open(image_path_or_file).convert("RGB")
    else:
        # File-like object (e.g. from Flask request)
        img = Image.open(image_path_or_file).convert("RGB")
        
    img = img.resize(config.IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) # shape (1, 299, 299, 3)
    return preprocess_input(img_array)

def extract_features_from_directory(images_dir=config.IMAGES_DIR, save_path=config.FEATURES_FILE):
    """
    Extracts 2048-dimensional features for all .jpg images in directory using InceptionV3.
    """
    if save_path.exists():
        print(f"📦 Features already cached at {save_path}. Loading...")
        with open(save_path, "rb") as f:
            return pickle.load(f)
            
    extractor = get_inception_feature_extractor()
    features = {}
    image_paths = list(Path(images_dir).glob("*.jpg"))
    
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in {images_dir}")
        
    print(f"📸 Extracting features for {len(image_paths)} images using InceptionV3...")
    for img_path in tqdm(image_paths, desc="Extracting visual features"):
        try:
            processed_img = preprocess_single_image(img_path)
            feature_vector = extractor.predict(processed_img, verbose=0) # shape (1, 2048)
            features[img_path.name] = feature_vector.flatten() # shape (2048,)
        except Exception as e:
            print(f"⚠️ Failed to process {img_path.name}: {e}")
            
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(features, f)
    print(f"✅ Extracted & saved {len(features)} image feature vectors to {save_path}")
    return features

def clean_caption_text(text):
    """
    Cleans caption: lowercase, removes punctuation, numbers, and excess whitespace.
    """
    text = text.lower().strip()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Split into words and remove single-letter words except 'a'
    words = [w for w in text.split() if len(w) > 1 or w == "a"]
    return " ".join(words)

def load_captions():
    """
    Loads captions from Flickr8k.token.txt or captions.txt.
    Returns:
        dict: {image_filename: [caption1, caption2, ...]}
    """
    descriptions = {}
    
    if config.CAPTIONS_FILE_TOKEN.exists():
        print(f"📄 Loading captions from token file: {config.CAPTIONS_FILE_TOKEN}")
        with open(config.CAPTIONS_FILE_TOKEN, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_token, caption = parts[0], parts[1]
                # Format is usually 'filename.jpg#0'
                img_id = img_token.split("#")[0]
                cleaned = clean_caption_text(caption)
                if img_id not in descriptions:
                    descriptions[img_id] = []
                descriptions[img_id].append(f"{config.START_TOKEN} {cleaned} {config.END_TOKEN}")
                
    elif config.CAPTIONS_FILE_CSV.exists():
        print(f"📄 Loading captions from CSV file: {config.CAPTIONS_FILE_CSV}")
        with open(config.CAPTIONS_FILE_CSV, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]: # Skip header
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                img_id = parts[0].strip()
                caption = ",".join(parts[1:]).strip().strip('"')
                cleaned = clean_caption_text(caption)
                if img_id not in descriptions:
                    descriptions[img_id] = []
                descriptions[img_id].append(f"{config.START_TOKEN} {cleaned} {config.END_TOKEN}")
    else:
        raise FileNotFoundError(
            f"Captions file not found. Looked for {config.CAPTIONS_FILE_TOKEN} and {config.CAPTIONS_FILE_CSV}."
        )
        
    print(f"✅ Loaded descriptions for {len(descriptions)} images.")
    return descriptions

def load_split_image_ids(split_file):
    """Loads a set of image filenames listed in a split text file."""
    if not Path(split_file).exists():
        return None
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    return set(ids)

def build_and_save_tokenizer(descriptions, min_frequency=config.VOCAB_MIN_FREQUENCY, save_path=config.TOKENIZER_FILE):
    """
    Constructs a Keras Tokenizer from descriptions, fits directly on corpus text,
    computes max_length, and saves the tokenizer.
    """
    from tensorflow.keras.preprocessing.text import Tokenizer
    
    all_captions = []
    word_counts = {}
    
    for caps in descriptions.values():
        for cap in caps:
            all_captions.append(cap)
            for word in cap.split():
                word_counts[word] = word_counts.get(word, 0) + 1
                
    print(f"📊 Total captions: {len(all_captions)}, Total unique words: {len(word_counts)}")
    
    # Fit Tokenizer directly on all captions so word frequencies and mappings are accurate
    tokenizer = Tokenizer(oov_token=config.UNK_TOKEN, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(all_captions)
    
    # Calculate maximum caption sequence length
    max_length = max(len(cap.split()) for cap in all_captions)
    vocab_size = len(tokenizer.word_index) + 1 # +1 for 0-padding
    
    print(f"✅ Vocabulary size (including 0-padding): {vocab_size}")
    print(f"📏 Maximum caption length: {max_length}")
    print(f"  • Token for '{config.START_TOKEN}': {tokenizer.word_index.get(config.START_TOKEN)}")
    print(f"  • Token for '{config.END_TOKEN}': {tokenizer.word_index.get(config.END_TOKEN)}")
    print(f"  • Token for 'a': {tokenizer.word_index.get('a')}")
    
    tokenizer_metadata = {
        "tokenizer": tokenizer,
        "max_length": max_length,
        "vocab_size": vocab_size,
        "word_counts": word_counts
    }
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(tokenizer_metadata, f)
    print(f"💾 Tokenizer metadata saved to {save_path}")
    
    return tokenizer_metadata

if __name__ == "__main__":
    descriptions = load_captions()
    meta = build_and_save_tokenizer(descriptions)
