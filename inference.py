"""
Inference Engine for Image Captioning.

Supports:
1. Feature extraction for arbitrary input photographs via InceptionV3.
2. Greedy Search decoding.
3. Beam Search decoding with configurable beam width.
4. Clean sentence formatting and confidence estimation.
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

import pickle
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config
from preprocess import preprocess_single_image, get_inception_feature_extractor

class ImageCaptionPredictor:
    def __init__(self, model_path=None, tokenizer_path=config.TOKENIZER_FILE):
        """Initializes feature extractor, trained caption model, and tokenizer."""
        # 1. Feature extractor (InceptionV3)
        self.feature_extractor = get_inception_feature_extractor()

        # 2. Tokenizer and metadata
        if not Path(tokenizer_path).exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run training or preprocessing first.")
        with open(tokenizer_path, "rb") as f:
            meta = pickle.load(f)
            self.tokenizer = meta["tokenizer"]
            self.max_length = meta["max_length"]
            self.vocab_size = meta["vocab_size"]

        # Word to ID and ID to Word mappings
        self.word_to_id = self.tokenizer.word_index
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}

        # 3. Load trained model
        if model_path is None:
            if config.BEST_MODEL_FILE.exists():
                model_path = config.BEST_MODEL_FILE
            elif config.MODEL_FILE.exists():
                model_path = config.MODEL_FILE
            else:
                raise FileNotFoundError(f"Trained model not found at {config.BEST_MODEL_FILE} or {config.MODEL_FILE}.")
                
        print(f"🤖 Loading trained captioning model from {model_path}...")
        self.model = tf.keras.models.load_model(str(model_path), compile=False)

    def extract_image_features(self, image_input):
        """Preprocesses image and passes through InceptionV3 to obtain 2048-dim feature."""
        processed_img = preprocess_single_image(image_input)
        features = self.feature_extractor.predict(processed_img, verbose=0)
        return np.array(features, dtype=np.float32) # shape (1, 2048)

    def generate_caption_greedy(self, image_input):
        """
        Generates caption using Greedy Argmax decoding with repetition suppression.
        """
        features = self.extract_image_features(image_input)
        in_text = config.START_TOKEN
        words = []
        
        for _ in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length, padding="post")
            
            # Predict probabilities for next word
            yhat = self.model.predict([features, sequence], verbose=0)[0]
            top_ids = np.argsort(yhat)[::-1]
            
            next_word = None
            for cand_id in top_ids[:30]:
                word = self.id_to_word.get(int(cand_id), None)
                if not word or word in [config.START_TOKEN, config.PAD_TOKEN, config.UNK_TOKEN]:
                    continue
                # Suppress consecutive duplicate tokens
                if words and word == words[-1]:
                    continue
                # Suppress repeating common stopwords within the last 2 steps
                if word in ["a", "the", "in", "on", "of", "with", "and", "at", "to", "is"] and word in words[-2:]:
                    continue
                # Prevent terminating too early if fewer than 3 words have been generated
                if word == config.END_TOKEN and len(words) < 3:
                    continue
                next_word = word
                break
                
            if next_word is None or next_word == config.END_TOKEN:
                break
                
            words.append(next_word)
            in_text += " " + next_word

        caption = " ".join(words)
        return self._format_sentence(caption)

    def generate_caption_beam_search(self, image_input, beam_index=3):
        """
        Generates caption using Beam Search decoding with repetition suppression and length normalization.
        """
        features = self.extract_image_features(image_input)
        start_id = self.word_to_id.get(config.START_TOKEN, 1)
        end_id = self.word_to_id.get(config.END_TOKEN, None)
        
        # Candidate format: (sequence_ids, cumulative_log_prob)
        candidates = [([start_id], 0.0)]
        completed = []

        for _ in range(self.max_length):
            all_expansions = []
            for seq, score in candidates:
                if end_id is not None and seq[-1] == end_id:
                    completed.append((seq, score))
                    continue
                    
                padded_seq = pad_sequences([seq], maxlen=self.max_length, padding="post")
                preds = self.model.predict([features, padded_seq], verbose=0)[0]
                
                # Take top candidates to allow filtering repetitions
                top_indices = np.argsort(preds)[::-1][:beam_index * 4]
                added = 0
                for idx in top_indices:
                    w = self.id_to_word.get(int(idx), None)
                    if not w or w in [config.START_TOKEN, config.PAD_TOKEN, config.UNK_TOKEN]:
                        continue
                    # Repetition suppression: no identical consecutive words
                    if len(seq) > 1 and idx == seq[-1]:
                        continue
                    # Disallow repeating common stopwords within 2 steps
                    if w in ["a", "the", "in", "on", "of", "with", "and", "at", "to", "is"]:
                        recent_words = [self.id_to_word.get(i, "") for i in seq[-2:]]
                        if w in recent_words:
                            continue
                    # Disallow premature termination if fewer than 3 words
                    if idx == end_id and len(seq) <= 3:
                        continue
                        
                    prob = max(float(preds[idx]), 1e-12)
                    all_expansions.append((seq + [int(idx)], score + np.log(prob)))
                    added += 1
                    if added >= beam_index:
                        break
                    
            if not all_expansions:
                break

            # Sort by normalized score (length penalty)
            candidates = sorted(
                all_expansions,
                key=lambda x: x[1] / (len(x[0]) ** 0.7),
                reverse=True
            )[:beam_index]

        # Combine completed sequences and remaining candidates
        all_results = completed + candidates
        if not all_results:
            return "Unable to generate description."
            
        all_results = sorted(all_results, key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)
        
        best_seq = all_results[0][0]
        words = []
        for idx in best_seq:
            w = self.id_to_word.get(idx, "")
            if w in [config.START_TOKEN, config.PAD_TOKEN, config.UNK_TOKEN]:
                continue
            if w == config.END_TOKEN:
                break
            words.append(w)
            
        caption = " ".join(words)
        return self._format_sentence(caption)

    def _format_sentence(self, caption):
        """Clean capitalization and punctuation for presentation."""
        caption = caption.strip()
        if not caption:
            return "Unable to generate description."
        caption = caption[0].upper() + caption[1:]
        if not caption.endswith("."):
            caption += "."
        return caption

def main():
    parser = argparse.ArgumentParser(description="Generate natural language caption for an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input photograph.")
    parser.add_argument("--beam", type=int, default=3, help="Beam width (1 for greedy).")
    args = parser.parse_args()

    predictor = ImageCaptionPredictor()
    if args.beam > 1:
        caption = predictor.generate_caption_beam_search(args.image, beam_index=args.beam)
    else:
        caption = predictor.generate_caption_greedy(args.image)

    print("\n" + "=" * 50)
    print(f"🖼️ Image: {args.image}")
    print(f"📝 Generated Caption: {caption}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
