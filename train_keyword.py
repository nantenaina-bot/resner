import os
import numpy as np
import librosa
import tensorflow as tf

from tensorflow.keras import layers, models

# ============================================================
# CONFIGURATIONa
# ============================================================

DATASET_PATH = "dataset"

MODEL_PATH = "keyword_model.keras"
CLASSES_PATH = "keyword_classes.txt"

CLASSES = [
    "bonjour",
    "autre"
]

SAMPLE_RATE = 16000
DURATION = 2
N_SAMPLES = SAMPLE_RATE * DURATION

N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 256
MAX_LEN = 125

EPOCHS = 50
BATCH_SIZE = 8

# ============================================================
# EXTRACTION MFCC
# ============================================================

def extract_mfcc(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=DURATION
    )

    # Force exactement 2 secondes
    if len(audio) < N_SAMPLES:
        audio = np.pad(
            audio,
            (0, N_SAMPLES - len(audio))
        )
    else:
        audio = audio[:N_SAMPLES]

    # Normalisation audio
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Force exactement 63 frames
    if mfcc.shape[1] < MAX_LEN:

        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, MAX_LEN - mfcc.shape[1]))
        )

    else:

        mfcc = mfcc[:, :MAX_LEN]

    # Normalisation MFCC
    mean = np.mean(mfcc)
    std = np.std(mfcc)

    if std > 0:
        mfcc = (mfcc - mean) / std

    return mfcc.astype(np.float32)


# ============================================================
# CHARGEMENT DATASET
# ============================================================

X = []
y = []

print()
print("==============================================")
print("       CHARGEMENT DATASET KEYWORD")
print("==============================================")

for class_index, class_name in enumerate(CLASSES):

    folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.exists(folder):

        print(
            f"⚠️ Dossier absent : {folder}"
        )

        continue

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".wav", ".mp3", ".flac", ".ogg")
        )
    ]

    print(
        f"{class_index} -> {class_name} : "
        f"{len(files)} fichiers"
    )

    for file_name in files:

        file_path = os.path.join(
            folder,
            file_name
        )

        try:

            mfcc = extract_mfcc(file_path)

            X.append(mfcc)
            y.append(class_index)

        except Exception as e:

            print(
                f"❌ Erreur : {file_path}"
            )

            print(e)


# ============================================================
# VERIFICATION
# ============================================================

if len(X) == 0:

    raise RuntimeError(
        "❌ Aucun fichier audio trouvé dans le dataset keyword."
    )

X = np.array(X, dtype=np.float32)

y = np.array(y, dtype=np.int32)

# Ajouter canal pour Conv2D
X = X[..., np.newaxis]

print()
print("Shape X :", X.shape)
print("Shape y :", y.shape)

# ============================================================
# SAUVEGARDE DES CLASSES
# ============================================================

with open(
    CLASSES_PATH,
    "w",
    encoding="utf-8"
) as f:

    for class_name in CLASSES:

        f.write(
            class_name + "\n"
        )

print()
print(
    f"✅ Classes sauvegardées : {CLASSES_PATH}"
)

# ============================================================
# MODEL
# ============================================================

model = models.Sequential([

    layers.Input(
        shape=(N_MFCC, MAX_LEN, 1)
    ),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dropout(0.4),

    layers.Dense(
        len(CLASSES),
        activation="softmax"
    )
])

# ============================================================
# COMPILATION
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

print()
print("==============================================")
print("          ENTRAINEMENT KEYWORD")
print("==============================================")

model.summary()

# ============================================================
# TRAINING
# ============================================================

history = model.fit(

    X,
    y,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_split=0.2,

    shuffle=True
)

# ============================================================
# SAUVEGARDE
# ============================================================

model.save(
    MODEL_PATH
)

print()
print("==============================================")
print("          ENTRAINEMENT TERMINE")
print("==============================================")

print(
    f"✅ Modèle : {MODEL_PATH}"
)

print(
    f"✅ Classes : {CLASSES_PATH}"
)

print()
print("Classes :")

for i, class_name in enumerate(CLASSES):

    print(
        f"  {i} -> {class_name}"
    )

print()