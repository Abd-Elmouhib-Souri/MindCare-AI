"""
Voice emotion model (MobileNetV2 on 3-channel MFCC, TESS), reproduced EXACTLY from the
original code (MLL/src/data/audio_pipeline.py, src/models/audio_classifier.py,
src/training/train_audio.py) so that a re-evaluation measures the same method:
  librosa 22,050 Hz mono -> 40 MFCC + delta + delta-delta -> pad/truncate to 150 frames
  -> tensor (40, 150, 3), raw values (no scaling, as in the original).
Only the SPLIT changes: leave-one-speaker-out instead of a random split.
"""

import os
from glob import glob

import numpy as np

SAMPLE_RATE, N_MFCC, MAX_PAD_LEN = 22050, 40, 150
# Same matching order as the original: 'surprise' also catches 'pleasant_surprise(d)'
_EMOTION_KEYS = [("angry", 0), ("disgust", 1), ("fear", 2), ("happy", 3), ("neutral", 4),
                 ("sad", 5), ("surprise", 6), ("pleasant_surprised", 6), ("pleasant_surprise", 6)]


def parse_emotion(folder_name: str):
    """Emotion index from a TESS folder name, e.g. 'OAF_Pleasant_surprise' -> 6."""
    low = folder_name.lower()
    for key, label in _EMOTION_KEYS:
        if key in low:
            return label
    return None


def parse_speaker_word(file_name: str):
    """'OAF_back_angry.wav' -> ('OAF', 'back'). TESS: 2 speakers (OAF older, YAF younger)."""
    parts = os.path.splitext(file_name)[0].split("_")
    return parts[0].upper(), parts[1].lower()


def list_tess(tess_dir: str) -> list:
    """[(path, emotion, speaker, word)] with the original de-duplication by file name.
    Sorted, so the order is the same on every operating system.
    The speaker comes from the FOLDER name (OAF_angry -> OAF): TESS has at least one file
    with a typo in its name ('OA_...' instead of 'OAF_...'), which created a fake 3rd speaker
    when the speaker was read from the file name (bug found on the real data, 6 Nov 2026)."""
    items, seen = [], set()
    for path in sorted(glob(os.path.join(tess_dir, "**", "*.wav"), recursive=True)):
        name = os.path.basename(path)
        if name in seen:
            continue
        seen.add(name)
        folder = os.path.basename(os.path.dirname(path))
        emotion = parse_emotion(folder)
        if emotion is None:
            continue
        speaker = folder.split("_")[0].upper()
        _, word = parse_speaker_word(name)
        items.append((path, emotion, speaker, word))
    return items


def naming_anomalies(items: list) -> list:
    """File names whose speaker prefix differs from their folder (reported, not silently fixed)."""
    return [os.path.basename(p) for p, _, spk, _ in items
            if parse_speaker_word(os.path.basename(p))[0] != spk]


def extract_3d_mfcc(path: str) -> np.ndarray:
    """(40, 150, 3) tensor: MFCC, delta, delta-delta; zero-padded or truncated to 150 frames."""
    import librosa
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    chans = [mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2)]
    out = []
    for c in chans:
        c = c[:, :MAX_PAD_LEN]
        out.append(np.pad(c, ((0, 0), (0, MAX_PAD_LEN - c.shape[1])), mode="constant"))
    return np.stack(out, axis=-1).astype(np.float32)


def build_audio_model(weights: str = "imagenet"):
    """Same graph as the original AudioCNNClassifier.build_model (frozen backbone)."""
    import keras
    from keras import layers
    inp = layers.Input(shape=(N_MFCC, MAX_PAD_LEN, 3), name="audio_mfcc_input")
    backbone = keras.applications.MobileNetV2(input_shape=(N_MFCC, MAX_PAD_LEN, 3),
                                              include_top=False, weights=weights)
    backbone.trainable = False
    x = backbone(inp, training=False)
    x = layers.GlobalAveragePooling2D(name="gap2d")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(256, activation="relu", name="feature_dense")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    out = layers.Dense(7, activation="softmax", name="emotion_probabilities")(x)
    return keras.Model(inp, out, name="MindCare_Audio_CNN"), backbone


def train_two_stage(X, y, weights: str = "imagenet", verbose: int = 0):
    """The original two-stage recipe, unchanged (no tuning):
    stage 1: frozen backbone, Adam 1e-3, <=15 epochs, patience 5
    stage 2: last 30 backbone layers unfrozen (BatchNorm stays frozen), Adam 1e-5, <=20 epochs, patience 7
    validation_split 0.15, balanced class weights, early stopping on val_loss."""
    import keras
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    cw = dict(zip(classes.tolist(), compute_class_weight("balanced", classes=classes, y=y).tolist()))
    model, backbone = build_audio_model(weights)

    def fit(lr, epochs, patience):
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
               keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7)]
        return model.fit(X, y, validation_split=0.15, epochs=epochs, batch_size=32,
                         class_weight=cw, callbacks=cbs, verbose=verbose)

    h1 = fit(1e-3, 15, 5)
    # Exactly the original logic: backbone.trainable = True makes EVERY layer trainable; then the
    # first layers are frozen and only non-BatchNorm layers are (re)set in the last 30. So the
    # BatchNorm layers among the last 30 stay trainable (the original comment said otherwise).
    # The backbone is still called with training=False, so BatchNorm statistics are not updated.
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    for layer in backbone.layers[-30:]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True
    h2 = fit(1e-5, 20, 7)
    return model, {"stage1_epochs": len(h1.history["loss"]), "stage2_epochs": len(h2.history["loss"])}
