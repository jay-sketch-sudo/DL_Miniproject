"""
Hugging Face Flickr8k Dataset Loader & Importer.

Downloads the official `intro/flickr8k` dataset from Hugging Face:
https://huggingface.co/datasets/intro/flickr8k

Extracts real photographic images and 5 human reference captions per image,
writes them into standard Flickr8k format, and cleans up any old synthetic/sample data.
"""
import os
import sys
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Configure UTF-8 for Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config

def cleanup_synthetic_data():
    """Deletes all previously generated synthetic sample data and cached checkpoints."""
    print("🧹 Cleaning up old synthetic data points and caches...")
    
    # 1. Delete synthetic images
    if config.IMAGES_DIR.exists():
        for file in config.IMAGES_DIR.glob("sample_*.jpg"):
            try:
                file.unlink()
            except Exception:
                pass
                
    # 2. Delete old cached features and tokenizers
    files_to_remove = [
        config.FEATURES_FILE,
        config.TOKENIZER_FILE,
        config.MODEL_FILE,
        config.BEST_MODEL_FILE,
        config.CHECKPOINTS_DIR / "bleu_scores.json",
        config.CHECKPOINTS_DIR / "training_log.csv"
    ]
    for file in files_to_remove:
        if file.exists():
            try:
                file.unlink()
                print(f"  • Removed {file.name}")
            except Exception:
                pass
                
    print("✅ Cleaned up old synthetic data points.")

def download_and_prepare_flickr8k(max_images=1000):
    """
    Streams and extracts images and captions from `intro/flickr8k` on Hugging Face.
    
    Args:
        max_images (int or None): Number of real images to download (None for all 8,000).
    """
    from datasets import load_dataset

    print("=" * 65)
    print("🌐 Downloading Real Flickr8k Dataset from Hugging Face (intro/flickr8k)")
    print("=" * 65)

    cleanup_synthetic_data()

    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.TEXT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📥 Loading dataset stream from Hugging Face: intro/flickr8k...")
    # Use streaming=True so we download images smoothly without requiring massive initial storage
    dataset = load_dataset("intro/flickr8k", split="train", streaming=True)

    token_lines = []
    train_ids = []
    dev_ids = []
    test_ids = []

    count = 0
    desc = f"Downloading real Flickr8k images" if max_images is None else f"Downloading {max_images} real Flickr8k images"
    pbar = tqdm(total=max_images if max_images is not None else 8000, desc=desc)

    for item in dataset:
        if max_images is not None and count >= max_images:
            break

        img = item.get("image")
        if img is None:
            continue

        # Unique real image filename
        img_filename = f"flickr8k_{count:05d}.jpg"
        img_save_path = config.IMAGES_DIR / img_filename

        # Convert to RGB and save
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(img_save_path, "JPEG", quality=90)

        # Collect captions (caption_0 .. caption_4)
        captions = []
        for i in range(5):
            cap_key = f"caption_{i}"
            cap_val = item.get(cap_key)
            if cap_val:
                cap_clean = str(cap_val).strip()
                captions.append(cap_clean)
                token_lines.append(f"{img_filename}#{i}\t{cap_clean}\n")

        # Split assignment (split column or 80/10/10)
        split_val = str(item.get("split", "train")).lower()
        if "test" in split_val:
            test_ids.append(img_filename)
        elif "dev" in split_val or "val" in split_val:
            dev_ids.append(img_filename)
        else:
            # Fallback 80/10/10 if split is uniform
            mod = count % 10
            if mod == 8:
                dev_ids.append(img_filename)
            elif mod == 9:
                test_ids.append(img_filename)
            else:
                train_ids.append(img_filename)

        count += 1
        pbar.update(1)

    pbar.close()

    # Save token captions file
    with open(config.CAPTIONS_FILE_TOKEN, "w", encoding="utf-8") as f:
        f.writelines(token_lines)

    # Save split files
    with open(config.TRAIN_IMAGES_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(train_ids) + "\n")
    with open(config.DEV_IMAGES_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(dev_ids) + "\n")
    with open(config.TEST_IMAGES_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(test_ids) + "\n")

    print("\n" + "=" * 65)
    print(f"🎉 Successfully prepared {count} real Flickr8k images with {len(token_lines)} captions!")
    print(f"  • Images directory: {config.IMAGES_DIR}")
    print(f"  • Captions file:    {config.CAPTIONS_FILE_TOKEN}")
    print(f"  • Train set:        {len(train_ids)} images")
    print(f"  • Validation set:   {len(dev_ids)} images")
    print(f"  • Test set:         {len(test_ids)} images")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare intro/flickr8k from Hugging Face.")
    parser.add_argument("--num", type=int, default=1000, help="Number of real images to download (default: 1000, 0 for all).")
    args = parser.parse_args()

    num_images = None if args.num <= 0 else args.num
    download_and_prepare_flickr8k(max_images=num_images)

