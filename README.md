# 📸 VisionVoice: AI-Powered Image Captioning System (Photo → Sentence Description)

An end-to-end Deep Learning system combining **Computer Vision (CV)** and **Natural Language Processing (NLP)** to automatically generate meaningful, human-like natural language descriptions from photographs.

Trained on the **Flickr8k dataset** (directly sourced from [Hugging Face — `intro/flickr8k`](https://huggingface.co/datasets/intro/flickr8k)), the architecture leverages transfer learning with a pretrained **InceptionV3 CNN** as the visual feature encoder and a sequence-to-sequence **LSTM network** as the language decoder, evaluated rigorously with **BLEU-1..4 metrics** and hosted on a responsive **Flask Web Interface**.

---

## 🌟 Key Highlights & Capabilities

- **Transfer Learning Vision Encoder**: Extracts dense 2,048-dimensional visual semantic vectors from InceptionV3 (ImageNet pretrained, headless with Global Average Pooling).
- **Multimodal Fusion Architecture**: Implements the merge-model paradigm (Tanti et al.), combining projected image feature embeddings and recurrent language hidden states.
- **Autoregressive Language Generation**: Generates captions word-by-word with support for both **Greedy Search** and **Beam Search ($k=3$)** decoding.
- **Repetition Suppression & Loop Prevention**: Prevents degenerate loops (such as `"A a a a."`) by dynamically penalizing consecutive identical words, suppressing frequent stopwords (`a`, `the`, `in`, `on`, `of`) from repeating within 2 steps, and applying `mask_zero=True` in Keras Embedding to protect sequence state from zero-padding overwrites.
- **Direct Hugging Face Dataset Loader**: Automated script [`download_hf_flickr8k.py`](file:///d:/Deeplearning/download_hf_flickr8k.py) streams authentic Flickr8k photographs and 5 human annotations per image directly into train/validation/test splits.
- **BLEU Score Benchmark**: Automated evaluation measuring n-gram precision across Unigrams, Bigrams, Trigrams, and 4-grams against reference human captions.
- **Interactive Web Interface**: Modern UI with drag-and-drop upload, decoding algorithm selection, inference latency telemetry, and in-browser Text-to-Speech (TTS) voice narration.
- **RESTful API**: Programmatic `/api/caption` endpoint for integration into external web or mobile applications.

---

## 🧠 System Architecture

```
                       [ Input Photograph ]
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Pretrained InceptionV3 │ (299 × 299 × 3)
                     │   (Global Avg Pool)   │
                     └───────────┬───────────┘
                                 │ Feature Vector: (2048,)
                                 ▼
                     ┌───────────────────────┐
                     │   Dense(256) + ReLU   │ (Image Projection)
                     └───────────┬───────────┘
                                 │
   [ Partial Caption ]           │
  (e.g., "startseq a dog")       │
            │                    │
            ▼                    │
 ┌──────────────────────┐        │
 │ Embedding(256,       │        │
 │   mask_zero=True)    │        │
 ├──────────────────────┤        │
 │       LSTM(256)      │        │
 └──────────┬───────────┘        │
            │ (256,)             │
            └───────────┬────────┘
                        ▼
            ┌───────────────────────┐
            │   Element-wise Add    │ (Multimodal Merge)
            └───────────┬───────────┘
                        ▼
            ┌───────────────────────┐
            │  Dense(256) + Dropout │
            └───────────┬───────────┘
                        ▼
            ┌───────────────────────┐
            │ Dense(Vocab, Softmax) │ ──► Predicts Next Word Token
            └───────────────────────┘
```

---

## 📁 Repository Structure

```
d:\Deeplearning\
├── app.py                   # Flask Web UI server & REST API
├── config.py                # Hyperparameters, file paths & global settings
├── download_hf_flickr8k.py  # Streams & formats intro/flickr8k from Hugging Face
├── dataset_downloader.py    # Flickr8k verification & sample dataset utility
├── evaluate.py              # BLEU-1..4 evaluation engine on held-out test split
├── inference.py             # Inference engine (Greedy & Beam Search with loop prevention)
├── model.py                 # Keras InceptionV3 + LSTM neural network definition
├── preprocess.py            # Feature extraction, caption cleaning & Tokenizer builder
├── train.py                 # Training pipeline with Checkpoints & Early Stopping
├── test_pipeline.py         # Unit tests covering preprocessing, tokenization & sizing
├── requirements.txt         # Project dependencies
├── templates/
│   └── index.html           # Modern responsive Web UI with speech synthesis
├── static/
│   └── uploads/             # Temporary storage for uploaded user images
├── checkpoints/             # Saved model weights, tokenizer & evaluation benchmarks
│   ├── best_model.keras     # Best checkpoint weights
│   ├── tokenizer.pkl        # Serialized Keras Tokenizer & vocabulary metadata
│   └── bleu_scores.json     # Quantitative BLEU evaluation output
└── data/                    # Flickr8k dataset directory
    ├── Flickr8k_Dataset/    # Photographic images (.jpg)
    └── Flickr8k_text/       # Captions (.token.txt) & train/dev/test splits
```

---

## 🚀 Getting Started

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/jay-sketch-sudo/DL_Miniproject.git
cd DL_Miniproject
pip install -r requirements.txt
```

### 2. Download the Flickr8k Dataset from Hugging Face
Download authentic images and captions from [intro/flickr8k](https://huggingface.co/datasets/intro/flickr8k):
```bash
# Download 250 real images with 1,250 human captions
python download_hf_flickr8k.py --num 250

# Or download more images (e.g. 1000 images, or 0 for all 8000)
python download_hf_flickr8k.py --num 1000
```
This script cleans any old synthetic data, saves photographs to `data/Flickr8k_Dataset/`, formats `data/Flickr8k_text/Flickr8k.token.txt`, and generates standard train/dev/test splits.

---

## 🏋️ Training the Model

Run the training pipeline:
```bash
python train.py --epochs 25 --batch_size 32
```

The training process executes:
1. **Feature Extraction**: Computes 2,048-dim InceptionV3 features and caches them in `checkpoints/features_inceptionv3.pkl`.
2. **Corpus Tokenization**: Fits Tokenizer on authentic human captions, assigning token IDs by frequency (`checkpoints/tokenizer.pkl`).
3. **Sequence Batching**: Yields `((X_img, X_seq), y_target)` sequence pairs with padding.
4. **Callbacks**: `ModelCheckpoint` saves the best weights (`checkpoints/best_model.keras`), `ReduceLROnPlateau` decays learning rate on plateaus, and `CSVLogger` tracks epoch loss.
5. **Loss Curve**: Saves training visualization to `static/loss_plot.png`.

---

## 🔬 Performance Evaluation (BLEU Scores)

Evaluate the model against reference human captions:
```bash
# Beam search evaluation (beam width = 3) on the test split
python evaluate.py --beam 3

# Greedy evaluation on test images
python evaluate.py --beam 1
```

### Measured Performance on Test Split:

| Metric | Evaluation Type | Score |
| :--- | :--- | :---: |
| **BLEU-1** | Unigram Precision (Vocabulary Accuracy) | **40.79%** |
| **BLEU-2** | Bigram Precision (Phrase Coherence) | **23.11%** |
| **BLEU-3** | Trigram Precision (Syntactic Structure) | **12.07%** |
| **BLEU-4** | 4-Gram Precision (Full Sentence Fidelity) | **7.03%** |

### Verified Multi-Image Inference Outputs:

| Input Image | Generated Caption (Greedy / Beam Search) |
|---|---|
| `flickr8k_00249.jpg` | *"A black and white dog is running through a field."* |
| `flickr8k_00069.jpg` | *"A black and white bird standing on a hand."* |
| `flickr8k_00199.jpg` | *"A man in a blue shirt and glasses is wearing a hat and glasses."* |
| `flickr8k_00159.jpg` | *"A little girl in a pink dress going into a wooden cabin."* |

---

## 💻 CLI Inference

Generate a description for any arbitrary photograph:
```bash
# Fast greedy search
python inference.py --image "path/to/photograph.jpg" --beam 1

# High-quality beam search (k=3)
python inference.py --image "path/to/photograph.jpg" --beam 3
```

---

## 🌐 Web Application

Start the Flask application:
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

### Web UI Features:
- **Interactive Drag-and-Drop Dropzone**: Supports JPG, PNG, WEBP, and BMP formats.
- **Decoding Selection**: Toggle between Fast Greedy Search and Beam Search ($k=3$).
- **Voice Narration**: In-browser speech synthesis reads generated captions aloud.
- **Latency & Model Specs**: Real-time display of inference latency, vocabulary size, and InceptionV3 dimensions.

### Programmatic REST API
Send a `POST` request with a photograph to `/api/caption`:
```bash
curl -X POST -F "image=@photo.jpg" -F "beam_width=3" http://127.0.0.1:5000/api/caption
```
**JSON Response:**
```json
{
  "status": "success",
  "caption": "A black and white dog is running through a field.",
  "decoding": "beam_search",
  "beam_width": 3,
  "latency_seconds": 1.25
}
```

---

## 🛠️ Unit Tests
Run the unit test suite to verify data cleaning, dataset integrity, and tokenization:
```bash
python -m unittest test_pipeline.py
```
All 4 tests run and pass cleanly.

---

## 📜 License
This project is licensed under the MIT License.
