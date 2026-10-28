"""
Evaluation Module for Image Captioning Model.

Calculates BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores on the test/evaluation dataset
using nltk.translate.bleu_score.
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
import argparse
from pathlib import Path
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

import config
from preprocess import load_captions, load_split_image_ids
from inference import ImageCaptionPredictor

def evaluate_model(num_eval_images=None, beam_width=1):
    """
    Evaluates model on the test split (or available images) and reports BLEU-1..4.
    """
    print("=" * 60)
    print(f"🔬 Starting BLEU Evaluation (Beam Width = {beam_width})")
    print("=" * 60)

    # 1. Load ground truth captions
    descriptions = load_captions()

    # 2. Determine evaluation image list
    test_ids = load_split_image_ids(config.TEST_IMAGES_TXT)
    if not test_ids:
        dev_ids = load_split_image_ids(config.DEV_IMAGES_TXT)
        test_ids = dev_ids if dev_ids else list(descriptions.keys())

    eval_image_ids = [img for img in test_ids if (config.IMAGES_DIR / img).exists()]
    if num_eval_images is not None and num_eval_images < len(eval_image_ids):
        eval_image_ids = eval_image_ids[:num_eval_images]

    print(f"📊 Evaluating on {len(eval_image_ids)} images...")

    # 3. Initialize predictor
    predictor = ImageCaptionPredictor()

    actual_references = []
    predicted_hypotheses = []
    qualitative_samples = []

    for img_name in tqdm(eval_image_ids, desc="Generating test captions"):
        img_path = config.IMAGES_DIR / img_name
        
        # Ground truth references (cleaned, tokenized)
        raw_caps = descriptions.get(img_name, [])
        cleaned_refs = []
        for c in raw_caps:
            cleaned = c.replace(config.START_TOKEN, "").replace(config.END_TOKEN, "").strip()
            cleaned_refs.append(cleaned.split())
            
        actual_references.append(cleaned_refs)

        # Model prediction
        if beam_width > 1:
            pred_caption = predictor.generate_caption_beam_search(img_path, beam_index=beam_width)
        else:
            pred_caption = predictor.generate_caption_greedy(img_path)

        pred_tokens = pred_caption.lower().replace(".", "").split()
        predicted_hypotheses.append(pred_tokens)

        if len(qualitative_samples) < 5:
            qualitative_samples.append({
                "image": img_name,
                "prediction": pred_caption,
                "references": [" ".join(ref) for ref in cleaned_refs]
            })

    # 4. Compute Corpus BLEU Scores
    print("\n📈 Calculating Corpus BLEU Scores...")
    bleu_1 = corpus_bleu(actual_references, predicted_hypotheses, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(actual_references, predicted_hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(actual_references, predicted_hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(actual_references, predicted_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    results = {
        "num_images_evaluated": len(eval_image_ids),
        "beam_width": beam_width,
        "BLEU-1": round(float(bleu_1), 4),
        "BLEU-2": round(float(bleu_2), 4),
        "BLEU-3": round(float(bleu_3), 4),
        "BLEU-4": round(float(bleu_4), 4)
    }

    print("\n" + "=" * 45)
    print("🏆 BLEU EVALUATION RESULTS")
    print("=" * 45)
    print(f"  • BLEU-1 (Unigram Precision): {results['BLEU-1'] * 100:.2f}%")
    print(f"  • BLEU-2 (Bigram Precision):  {results['BLEU-2'] * 100:.2f}%")
    print(f"  • BLEU-3 (Trigram Precision): {results['BLEU-3'] * 100:.2f}%")
    print(f"  • BLEU-4 (4-gram Precision):  {results['BLEU-4'] * 100:.2f}%")
    print("=" * 45)

    print("\n🔍 Qualitative Sample Previews:")
    for i, sample in enumerate(qualitative_samples, 1):
        print(f"\n[Sample {i}] Image: {sample['image']}")
        print(f"  👉 Predicted: {sample['prediction']}")
        print(f"  📖 Ground Truth 1: {sample['references'][0] if sample['references'] else 'N/A'}")
        if len(sample['references']) > 1:
            print(f"  📖 Ground Truth 2: {sample['references'][1]}")

    # Save results to json
    results_path = config.CHECKPOINTS_DIR / "bleu_scores.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n💾 Evaluation scores saved to {results_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image Captioning model using BLEU scores.")
    parser.add_argument("--num", type=int, default=None, help="Number of test images to evaluate (default: all).")
    parser.add_argument("--beam", type=int, default=1, help="Beam width for decoding.")
    args = parser.parse_args()
    
    evaluate_model(num_eval_images=args.num, beam_width=args.beam)
