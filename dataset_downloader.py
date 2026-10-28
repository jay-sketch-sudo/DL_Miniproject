"""
Dataset Downloader and Verification Utility for Flickr8k.

Provides automated checks, guides for downloading the Flickr8k dataset,
and can generate/download a verified sample dataset for immediate testing.
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

import urllib.request
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw

import config

def check_dataset_status():
    """Checks whether the Flickr8k images and captions exist."""
    print("=" * 60)
    print("🔍 Checking Flickr8k Dataset Status")
    print("=" * 60)
    
    images_exist = config.IMAGES_DIR.exists() and len(list(config.IMAGES_DIR.glob("*.jpg"))) > 0
    token_txt_exist = config.CAPTIONS_FILE_TOKEN.exists()
    csv_txt_exist = config.CAPTIONS_FILE_CSV.exists()
    
    num_images = len(list(config.IMAGES_DIR.glob("*.jpg"))) if config.IMAGES_DIR.exists() else 0
    print(f"📁 Images Directory: {config.IMAGES_DIR}")
    print(f"   Found {num_images} images.")
    
    print(f"📄 Captions (Token format): {'Found' if token_txt_exist else 'Missing'} ({config.CAPTIONS_FILE_TOKEN})")
    print(f"📄 Captions (CSV format):   {'Found' if csv_txt_exist else 'Missing'} ({config.CAPTIONS_FILE_CSV})")
    
    is_ready = images_exist and (token_txt_exist or csv_txt_exist)
    if is_ready:
        print("\n✅ Dataset is ready for training and feature extraction!")
    else:
        print("\n⚠️ Dataset is incomplete or not yet downloaded.")
        print("\nTo obtain the complete Flickr8k Dataset:")
        print("Option 1 (Kaggle CLI):")
        print("  kaggle datasets download -d adityajn105/flickr8k")
        print("  Extract files into the 'data/' folder.")
        print("\nOption 2 (Direct Download):")
        print("  Download Flickr8k_Dataset.zip and Flickr8k_text.zip,")
        print(f"  and place the extracted files inside {config.DATA_DIR}.")
        print("\nAlternatively, run `python dataset_downloader.py --sample` to generate a")
        print("functional mini-dataset for smoke testing the entire pipeline immediately.")
    print("=" * 60)
    return is_ready

def create_sample_dataset(num_samples=25):
    """
    Creates a small synthetic sample dataset with distinct geometric objects
    and matching captions for rapid end-to-end pipeline verification.
    """
    print(f"\n🎨 Generating {num_samples} sample images & captions in {config.DATA_DIR}...")
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.TEXT_DIR.mkdir(parents=True, exist_ok=True)
    
    colors = [
        ("red", (220, 50, 50)),
        ("blue", (50, 100, 220)),
        ("green", (50, 180, 50)),
        ("yellow", (230, 210, 50)),
        ("purple", (160, 50, 200))
    ]
    shapes = ["circle", "rectangle", "triangle"]
    
    captions_lines = []
    image_names = []
    
    count = 0
    for color_name, rgb in colors:
        for shape in shapes:
            if count >= num_samples:
                break
            img_id = f"sample_{count:03d}.jpg"
            img_path = config.IMAGES_DIR / img_id
            
            # Create synthetic image
            img = Image.new("RGB", (300, 300), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            
            if shape == "circle":
                draw.ellipse([60, 60, 240, 240], fill=rgb, outline=(0, 0, 0), width=3)
            elif shape == "rectangle":
                draw.rectangle([60, 70, 240, 230], fill=rgb, outline=(0, 0, 0), width=3)
            elif shape == "triangle":
                draw.polygon([(150, 50), (60, 250), (240, 250)], fill=rgb, outline=(0, 0, 0))
                
            img.save(img_path, "JPEG")
            image_names.append(img_id)
            
            # 5 captions per image (standard Flickr8k format)
            caps = [
                f"a {color_name} {shape} on a gray background",
                f"there is a bright {color_name} {shape}",
                f"a photo showing a {color_name} {shape} in the center",
                f"an image with a single {color_name} {shape}",
                f"a clean graphic depicting a {color_name} {shape}"
            ]
            for idx, cap in enumerate(caps):
                captions_lines.append(f"{img_id}#{idx}\t{cap}\n")
            count += 1
            
    # Write captions file
    with open(config.CAPTIONS_FILE_TOKEN, "w", encoding="utf-8") as f:
        f.writelines(captions_lines)
        
    # Write train/dev/test splits
    split_train = image_names[: int(0.7 * len(image_names))]
    split_dev = image_names[int(0.7 * len(image_names)) : int(0.85 * len(image_names))]
    split_test = image_names[int(0.85 * len(image_names)) :]
    
    with open(config.TRAIN_IMAGES_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(split_train) + "\n")
    with open(config.DEV_IMAGES_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(split_dev) + "\n")
    with open(config.TEST_IMAGES_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(split_test) + "\n")
        
    print(f"✅ Generated {len(image_names)} sample images and {len(captions_lines)} captions.")
    print("Files created:")
    print(f"  • {config.CAPTIONS_FILE_TOKEN}")
    print(f"  • {config.TRAIN_IMAGES_TXT}")
    print(f"  • {config.DEV_IMAGES_TXT}")
    print(f"  • {config.TEST_IMAGES_TXT}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        create_sample_dataset()
    else:
        check_dataset_status()
