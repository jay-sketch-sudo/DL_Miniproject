"""
Training Pipeline for Image Captioning Model.

Executes:
1. Feature extraction & caching (InceptionV3)
2. Tokenizer initialization & serialization
3. Batch Data Generator for sequence-to-sequence training
4. Model Compilation, Callbacks (Checkpoint, EarlyStopping, ReduceLROnPlateau)
5. Model Training & History Plotting
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

import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

import config
from preprocess import (
    extract_features_from_directory,
    load_captions,
    build_and_save_tokenizer,
    load_split_image_ids
)
from model import build_captioning_model

def create_sequences(tokenizer, max_length, descriptions, features, image_ids, use_sparse=True):
    """
    Yields input features, input sequences, and target words in batches.
    """
    X_img, X_seq, y_target = [], [], []
    
    for img_id in image_ids:
        if img_id not in features or img_id not in descriptions:
            continue
            
        feat = features[img_id]
        captions = descriptions[img_id]
        
        for caption in captions:
            # Convert words to integer token IDs
            seq = tokenizer.texts_to_sequences([caption])[0]
            
            # Create prefix sequences and target word
            for i in range(1, len(seq)):
                in_seq = seq[:i]
                out_seq = seq[i]
                
                # Pad sequence to fixed max_length
                in_seq = pad_sequences([in_seq], maxlen=max_length, padding="post")[0]
                
                X_img.append(feat)
                X_seq.append(in_seq)
                y_target.append(out_seq)
                
    X_img = np.array(X_img, dtype=np.float32)
    X_seq = np.array(X_seq, dtype=np.int32)
    y_target = np.array(y_target, dtype=np.int32)
    
    return X_img, X_seq, y_target

def batch_data_generator(descriptions, features, tokenizer, max_length, image_ids, batch_size=config.BATCH_SIZE, use_sparse=True):
    """
    Memory-efficient generator that yields batches of image features, padded input sequences, and target labels.
    """
    image_ids = list(image_ids)
    while True:
        np.random.shuffle(image_ids)
        X_img_batch, X_seq_batch, y_batch = [], [], []
        
        for img_id in image_ids:
            if img_id not in features or img_id not in descriptions:
                continue
            feat = features[img_id]
            for cap in descriptions[img_id]:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding="post")[0]
                    
                    X_img_batch.append(feat)
                    X_seq_batch.append(in_seq)
                    y_batch.append(out_seq)
                    
                    if len(X_img_batch) >= batch_size:
                        yield (
                            (np.array(X_img_batch, dtype=np.float32), np.array(X_seq_batch, dtype=np.int32)),
                            np.array(y_batch, dtype=np.int32)
                        )
                        X_img_batch, X_seq_batch, y_batch = [], [], []

def count_training_samples(descriptions, features, image_ids):
    """Counts total number of (X_img, X_seq -> target) samples across given image IDs."""
    total = 0
    for img_id in image_ids:
        if img_id in features and img_id in descriptions:
            for cap in descriptions[img_id]:
                words = cap.split()
                if len(words) > 1:
                    total += (len(words) - 1)
    return total

def run_training(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE):
    """Executes the full training pipeline."""
    print("=" * 60)
    print("🚀 Initializing Image Captioning Training Pipeline")
    print("=" * 60)

    # 1. Feature Extraction / Loading
    features = extract_features_from_directory()
    print(f"📦 Total image features ready: {len(features)}")

    # 2. Descriptions Loading
    descriptions = load_captions()

    # 3. Tokenizer
    if config.TOKENIZER_FILE.exists():
        print(f"📄 Loading existing tokenizer from {config.TOKENIZER_FILE}")
        with open(config.TOKENIZER_FILE, "rb") as f:
            tokenizer_meta = pickle.load(f)
    else:
        tokenizer_meta = build_and_save_tokenizer(descriptions)

    tokenizer = tokenizer_meta["tokenizer"]
    max_length = tokenizer_meta["max_length"]
    vocab_size = tokenizer_meta["vocab_size"]

    # 4. Train / Validation Splits
    all_image_ids = [k for k in descriptions.keys() if k in features]
    
    train_ids = load_split_image_ids(config.TRAIN_IMAGES_TXT)
    dev_ids = load_split_image_ids(config.DEV_IMAGES_TXT)
    
    if train_ids and dev_ids:
        train_ids = [img for img in train_ids if img in features and img in descriptions]
        dev_ids = [img for img in dev_ids if img in features and img in descriptions]
        print(f"📂 Used Flickr8k official splits: {len(train_ids)} train, {len(dev_ids)} val")
    else:
        np.random.seed(42)
        np.random.shuffle(all_image_ids)
        split = int(0.85 * len(all_image_ids))
        train_ids, dev_ids = all_image_ids[:split], all_image_ids[split:]
        print(f"📂 Created random 85/15 split: {len(train_ids)} train, {len(dev_ids)} val")

    total_train_samples = count_training_samples(descriptions, features, train_ids)
    total_val_samples = count_training_samples(descriptions, features, dev_ids)
    
    steps_per_epoch = max(1, total_train_samples // batch_size)
    validation_steps = max(1, total_val_samples // batch_size)

    print(f"📊 Training samples: {total_train_samples} (Steps/Epoch: {steps_per_epoch})")
    print(f"📊 Validation samples: {total_val_samples} (Validation Steps: {validation_steps})")

    # 5. Model Architecture
    model = build_captioning_model(
        vocab_size=vocab_size,
        max_length=max_length,
        use_sparse_loss=True
    )
    model.summary()

    # 6. Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=str(config.BEST_MODEL_FILE),
            monitor="loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="loss",
            patience=8,
            min_delta=0.01,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        ),
        CSVLogger(
            str(config.CHECKPOINTS_DIR / "training_log.csv"),
            separator=",",
            append=False
        )
    ]

    train_gen = batch_data_generator(descriptions, features, tokenizer, max_length, train_ids, batch_size=batch_size)
    val_gen = batch_data_generator(descriptions, features, tokenizer, max_length, dev_ids, batch_size=batch_size)

    # 7. Model Training
    print("\n🏋️ Starting Model Training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model & config
    model.save(str(config.MODEL_FILE))
    print(f"💾 Model saved to {config.MODEL_FILE} and {config.BEST_MODEL_FILE}")

    run_config = {
        "max_length": max_length,
        "vocab_size": vocab_size,
        "embedding_dim": config.EMBEDDING_DIM,
        "lstm_units": config.LSTM_UNITS,
        "start_token": config.START_TOKEN,
        "end_token": config.END_TOKEN,
        "unk_token": config.UNK_TOKEN
    }
    with open(config.CONFIG_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)

    # 8. Plot Loss Curve
    try:
        plt.figure(figsize=(9, 5))
        plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
        plt.title("Image Captioning Training Loss Curve (InceptionV3 + LSTM)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Sparse Categorical Crossentropy)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plot_path = config.STATIC_DIR / "loss_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📈 Training loss curve saved to {plot_path}")
    except Exception as e:
        print(f"⚠️ Could not plot training curve: {e}")

    print("🎉 Training completed successfully!")
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train InceptionV3 + LSTM Image Captioning model.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size for training.")
    args = parser.parse_args()
    run_training(epochs=args.epochs, batch_size=args.batch_size)
