"""
Model Architecture for Image Captioning.

Implements the Encoder-Decoder Merge Architecture:
- Encoder: Takes 2048-dim visual feature vector from InceptionV3 -> Dropout -> Dense(256)
- Decoder: Takes sequence of input tokens -> Embedding(256) -> Dropout -> LSTM(256)
- Fusion: Adds visual representation and language sequence representation -> Dense(256) -> Dense(vocab_size, softmax)
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import config

def build_captioning_model(vocab_size, max_length, 
                           feature_dim=config.FEATURE_DIM,
                           embedding_dim=config.EMBEDDING_DIM,
                           lstm_units=config.LSTM_UNITS,
                           dense_units=config.DENSE_UNITS,
                           dropout_rate=config.DROPOUT_RATE,
                           learning_rate=config.LEARNING_RATE,
                           use_sparse_loss=True):
    """
    Builds the combined CNN-LSTM Image Captioning neural network.

    Args:
        vocab_size (int): Total size of the vocabulary including padding index 0.
        max_length (int): Maximum length of a caption sequence.
        feature_dim (int): Dimensionality of InceptionV3 image features (2048).
        embedding_dim (int): Dimensionality of word embedding space.
        lstm_units (int): Number of units in LSTM hidden state.
        dense_units (int): Number of hidden units in fusion layer.
        dropout_rate (float): Dropout probability for regularization.
        learning_rate (float): Adam optimizer learning rate.
        use_sparse_loss (bool): If True, uses sparse_categorical_crossentropy for lower memory usage.

    Returns:
        tf.keras.Model: Compiled Keras Model ready for training.
    """
    # 1. Feature Extractor (Encoder input from InceptionV3)
    inputs_image = layers.Input(shape=(feature_dim,), name="image_features_input")
    feat_dropout = layers.Dropout(dropout_rate, name="image_dropout")(inputs_image)
    feat_dense = layers.Dense(dense_units, activation="relu", name="image_dense")(feat_dropout)

    # 2. Sequence Processor (Decoder input from word tokens)
    inputs_text = layers.Input(shape=(max_length,), name="text_sequence_input")
    text_embed = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="text_embedding"
    )(inputs_text)
    text_dropout = layers.Dropout(dropout_rate, name="text_dropout")(text_embed)
    text_lstm = layers.LSTM(lstm_units, name="text_lstm")(text_dropout)

    # 3. Fusion Layer (Merge image features and text features)
    decoder_merged = layers.add([feat_dense, text_lstm], name="fusion_add")
    decoder_dense = layers.Dense(dense_units, activation="relu", name="fusion_dense")(decoder_merged)
    decoder_dropout = layers.Dropout(dropout_rate, name="fusion_dropout")(decoder_dense)

    # 4. Output Layer (Predict probability distribution over vocabulary)
    outputs = layers.Dense(vocab_size, activation="softmax", name="output_vocab")(decoder_dropout)

    # Build model
    model = Model(inputs=[inputs_image, inputs_text], outputs=outputs, name="InceptionV3_LSTM_Captioner")

    loss_fn = "sparse_categorical_crossentropy" if use_sparse_loss else "categorical_crossentropy"
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    test_model = build_captioning_model(vocab_size=5000, max_length=35)
    test_model.summary()

