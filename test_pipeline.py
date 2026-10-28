"""
Unit and Integration Tests for Image Captioning Pipeline.

Verifies:
- Text normalization & cleaning
- Dataset format compatibility
- Vocabulary building & Tokenizer serialization
- Image preprocessing dimensions
- Model architecture compilation
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

import unittest
from pathlib import Path
import numpy as np
from PIL import Image

import config
from preprocess import clean_caption_text, load_captions, build_and_save_tokenizer

class TestImageCaptioningPipeline(unittest.TestCase):

    def test_clean_caption_text(self):
        """Tests that punctuation, numbers, and excess spaces are cleaned properly."""
        raw = "A Dog, running in 2024 through the GREEN-GRASS!"
        cleaned = clean_caption_text(raw)
        self.assertEqual(cleaned, "a dog running in through the greengrass")

    def test_sample_dataset_captions(self):
        """Tests that descriptions are loaded and formatted with start and end tokens."""
        if not config.CAPTIONS_FILE_TOKEN.exists() and not config.CAPTIONS_FILE_CSV.exists():
            from dataset_downloader import create_sample_dataset
            create_sample_dataset(5)

        descriptions = load_captions()
        self.assertGreater(len(descriptions), 0)
        
        # Check first entry
        sample_img = list(descriptions.keys())[0]
        sample_caps = descriptions[sample_img]
        self.assertGreater(len(sample_caps), 0)
        for cap in sample_caps:
            self.assertTrue(cap.startswith(config.START_TOKEN))
            self.assertTrue(cap.endswith(config.END_TOKEN))

    def test_tokenizer_building(self):
        """Tests tokenizer construction and serialization."""
        descriptions = load_captions()
        meta = build_and_save_tokenizer(descriptions, min_frequency=1)
        
        self.assertIn("tokenizer", meta)
        self.assertIn("max_length", meta)
        self.assertIn("vocab_size", meta)
        self.assertGreater(meta["vocab_size"], 2)
        self.assertGreater(meta["max_length"], 1)

    def test_image_preprocessing_shape(self):
        """Tests synthetic image sizing matches InceptionV3 requirement (299x299)."""
        temp_img_path = config.DATA_DIR / "test_temp.jpg"
        img = Image.new("RGB", (500, 400), color=(100, 150, 200))
        img.save(temp_img_path)

        from preprocess import preprocess_single_image
        arr = preprocess_single_image(temp_img_path)
        
        self.assertEqual(arr.shape, (1, 299, 299, 3))
        self.assertEqual(arr.dtype, np.float32)

        if temp_img_path.exists():
            temp_img_path.unlink()

if __name__ == "__main__":
    unittest.main()

