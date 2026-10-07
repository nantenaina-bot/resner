import os
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "dataset"
SAMPLE_RATE = 16000
DURATION = 2

N_SAMPLES = SAMPLE_RATE * DURATION

N_MFCC = 40
MAX_LEN = 63

EPOCHS = 50
BATCH_SIZE = 8


# ============================================================
# EXTRACT MFCC
# ============================================================

def extract_mfcc(file_path):

    try:

        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            duration=DURATION
        )

        # 2 secondes exactement
        if len(audio) < N_SAMPLES:

            audio = np.pad(
                audio,
                (0, N_SAMPLES - len(audio))
            )

        else:

            audio = audio[:N_SAMPLES]


        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=512,
            hop_length=512
        )


        # Même taille pour tous
        if mfcc.shape[1] < MAX_LEN:

            mfcc = np.pad(
                mfcc,
                (
                    (0, 0),
                    (0, MAX_LEN - mfcc.shape[1])
                )
            )

        else:

            mfcc = mfcc[:, :MAX_LEN]


        return mfcc.astype(np.float32)


    except Exception as e:

        print(
            f"Erreur avec {file_path}: {e}"
        )

        return None


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    X = []
    y = []

    classes = [
        "moi",
        "personne",
        "autre"
    ]

    print("\n================================")
    print("CHARGEMENT DU DATASET")
    print("================================")


    for label in classes:

        folder = os.path.join(
            DATASET_PATH,
            label
        )


        if not os.path.exists(folder):

            print(
                f"\nERREUR : dossier absent -> {folder}"
            )

            continue


        files = os.listdir(folder)

        print(
            f"\n[{label.upper()}]"
        )


        for filename in files:

            if not filename.lower().endswith(
                (".wav", ".mp3", ".flac", ".ogg")
            ):
                continue


            file_path = os.path.join(
                folder,
                filename
            )


            mfcc = extract_mfcc(
                file_path
            )


            if mfcc is not None:

                X.append(mfcc)
                y.append(label)

                print(
                    f"  OK : {filename}"
                )


    return (
        np.array(X, dtype=np.float32),
        np.array(y)
    )


# ============================================================
# CREATE CNN
# ============================================================

def create_model():

    model = models.Sequential([

        layers.Input(
            shape=(N_MFCC, MAX_LEN, 1)
        ),


        # -------------------------
        # CONVOLUTION 1
        # -------------------------

        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),


        # -------------------------
        # CONVOLUTION 2
        # -------------------------

        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),


        # -------------------------
        # CONVOLUTION 3
        # -------------------------

        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.GlobalAveragePooling2D(),


        # -------------------------
        # DENSE
        # -------------------------

        layers.Dense(
            64,
            activation="relu"
        ),

        layers.Dropout(
            0.4
        ),


        # -------------------------
        # OUTPUT
        # -------------------------

        layers.Dense(
            3,
            activation="softmax"
        )

    ])


    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )


    return model


# ============================================================
# TRAIN
# ============================================================

def main():

    print("\n")
    print("==============================================")
    print("       VOICE RECOGNITION TRAINING")
    print("==============================================")


    # Load
    X, y = load_dataset()


    if len(X) == 0:

        print(
            "\nAucun fichier audio trouvé."
        )

        return


    print("\n--------------------------------")
    print("DATASET")
    print("--------------------------------")

    print(
        "Nombre total :",
        len(X)
    )


    # Encode classes
    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)


    print(
        "Classes :",
        list(encoder.classes_)
    )


    # Save class names
    with open(
        "classes.txt",
        "w",
        encoding="utf-8"
    ) as f:

        for label in encoder.classes_:

            f.write(
                label + "\n"
            )


    # Add channel
    X = X[..., np.newaxis]


    # Train / validation
    X_train, X_val, y_train, y_val = train_test_split(

        X,
        y_encoded,

        test_size=0.25,

        random_state=42,

        stratify=y_encoded

    )


    print(
        "\nTrain :",
        len(X_train)
    )

    print(
        "Validation :",
        len(X_val)
    )


    # Create model
    model = create_model()


    print("\n")
    model.summary()


    # Early stopping
    early_stop = EarlyStopping(

        monitor="val_loss",

        patience=10,

        restore_best_weights=True

    )


    # Train
    print("\n================================")
    print("TRAINING")
    print("================================")


    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            early_stop
        ],

        verbose=1

    )


    # Evaluate
    loss, accuracy = model.evaluate(

        X_val,
        y_val,

        verbose=0

    )


    print("\n")
    print("================================")
    print("RESULTAT")
    print("================================")


    print(
        f"Validation accuracy : "
        f"{accuracy * 100:.2f}%"
    )


    # Save model
    model.save(
        "model.keras"
    )


    print(
        "\nModel sauvegardé : model.keras"
    )

    print(
        "Classes sauvegardées : classes.txt"
    )


    print("\n================================")
    print("TRAINING TERMINE")
    print("================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()