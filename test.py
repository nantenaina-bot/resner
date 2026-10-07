import os
import numpy as np
import librosa
import tensorflow as tf

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset"

KEYWORD_MODEL = "keyword_model.keras"
SPEAKER_MODEL = "speaker_model.keras"

KEYWORD_CLASSES = [
    "bonjour",
    "autre_mot"
]

SPEAKER_CLASSES = [
    "moi",
    "personne"
]

SAMPLE_RATE = 16000
DURATION = 2
SAMPLES = SAMPLE_RATE * DURATION

N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 256


# ============================================================
# EXTRACTION MFCC
# ============================================================

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Normalisation
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    # Durée fixe : 2 secondes
    if len(audio) < SAMPLES:

        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )

    else:

        audio = audio[:SAMPLES]

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Normalisation MFCC
    mfcc = (
        mfcc - np.mean(mfcc)
    ) / (
        np.std(mfcc) + 1e-8
    )

    return mfcc.astype(np.float32)


# ============================================================
# CHARGER UN DATASET
# ============================================================

def load_dataset(classes):

    X = []
    y = []

    print("\n--------------------------------------------")
    print("Chargement du dataset")
    print("--------------------------------------------")

    for label, class_name in enumerate(classes):

        folder = os.path.join(
            DATASET_DIR,
            class_name
        )

        if not os.path.exists(folder):

            print(
                f"[ERREUR] Dossier absent : {folder}"
            )

            continue

        files = [
            file
            for file in os.listdir(folder)
            if file.lower().endswith(".wav")
        ]

        print(
            f"{class_name:<15} : {len(files)} fichiers"
        )

        for file in files:

            file_path = os.path.join(
                folder,
                file
            )

            try:

                features = extract_features(
                    file_path
                )

                X.append(features)
                y.append(label)

            except Exception as error:

                print(
                    f"[ERREUR] {file} : {error}"
                )

    if len(X) == 0:

        return None, None

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int32
    )

    return X, y


# ============================================================
# TEST D'UN MODELE
# ============================================================

def test_model(
    model_path,
    classes,
    title
):

    print("\n")
    print("=" * 65)
    print(title)
    print("=" * 65)

    # --------------------------------------------------------
    # Vérification modèle
    # --------------------------------------------------------

    if not os.path.exists(model_path):

        print(
            f"[ERREUR] Modèle introuvable : {model_path}"
        )

        return

    # --------------------------------------------------------
    # Chargement modèle
    # --------------------------------------------------------

    print(
        f"\nChargement du modèle : {model_path}"
    )

    model = tf.keras.models.load_model(
        model_path
    )

    print("✓ Modèle chargé")

    # --------------------------------------------------------
    # Chargement dataset
    # --------------------------------------------------------

    X, y = load_dataset(
        classes
    )

    if X is None:

        print(
            "\n[ERREUR] Aucun fichier audio trouvé."
        )

        return

    print(
        "\nShape MFCC :",
        X.shape
    )

    # --------------------------------------------------------
    # Ajout du channel pour CNN
    # --------------------------------------------------------

    X = X[..., np.newaxis]

    print(
        "Shape entrée modèle :",
        X.shape
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\nPrediction en cours...")

    probabilities = model.predict(
        X,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    print("\n")
    print("--------------------------------------------")
    print("RESULTAT")
    print("--------------------------------------------")

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report")
    print("--------------------------------------------")

    print(
        classification_report(
            y,
            predictions,
            labels=list(range(len(classes))),
            target_names=classes,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("Confusion Matrix")
    print("--------------------------------------------")

    matrix = confusion_matrix(
        y,
        predictions,
        labels=list(range(len(classes)))
    )

    print(matrix)

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    total = len(y)

    correct = np.sum(
        y == predictions
    )

    incorrect = total - correct

    print("\nRésumé")
    print("--------------------------------------------")

    print(
        f"Total       : {total}"
    )

    print(
        f"Correct     : {correct}"
    )

    print(
        f"Incorrect   : {incorrect}"
    )

    if incorrect == 0:

        print(
            "\n✓ EXCELLENT : toutes les prédictions sont correctes."
        )

    else:

        print(
            "\n⚠ Il existe des erreurs de classification."
        )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("\n")
    print("=" * 65)
    print("       TEST FINAL - RECONNAISSANCE VOCALE")
    print("=" * 65)

    print("""
Structure utilisée :

dataset/
├── bonjour/
├── autre_mot/
├── moi/
└── personne/
""")

    # ========================================================
    # 1. TEST KEYWORD
    # ========================================================

    test_model(

        model_path=KEYWORD_MODEL,

        classes=KEYWORD_CLASSES,

        title="TEST 1 : KEYWORD"
    )

    # ========================================================
    # 2. TEST SPEAKER
    # ========================================================

    test_model(

        model_path=SPEAKER_MODEL,

        classes=SPEAKER_CLASSES,

        title="TEST 2 : SPEAKER"
    )

    # ========================================================
    # FIN
    # ========================================================

    print("\n")
    print("=" * 65)
    print("                TEST TERMINÉ")
    print("=" * 65)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()