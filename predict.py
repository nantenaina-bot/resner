import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
import os

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model.keras"

# Fréquence utilisée par le modèle
SAMPLE_RATE = 16000

# Fréquence utilisée pour enregistrer avec le microphone
RECORD_SAMPLE_RATE = 48000

DURATION = 2

# DEVICE MICROPHONE
MICROPHONE_DEVICE = 19

TEMP_FILE = "temp.wav"

CLASSES = [
    "MOI",
    "PERSONNE",
    "AUTRE"
]

# Seuil de confiance
THRESHOLD = 0.60


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

print("======================================")
print("       RECONNAISSANCE VOCALE")
print("======================================")
print()

print("Chargement du modèle...")

try:

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("✓ Modèle chargé")

except Exception as e:

    print("❌ Erreur lors du chargement du modèle :")
    print(e)

    input("\nAppuyez sur ENTER pour quitter...")
    raise SystemExit


print()

# ============================================================
# VERIFICATION DU MICROPHONE
# ============================================================

print("======================================")
print("       MICROPHONE")
print("======================================")

try:

    device_info = sd.query_devices(
        MICROPHONE_DEVICE
    )

    print(
        "Device :",
        MICROPHONE_DEVICE
    )

    print(
        "Nom :",
        device_info["name"]
    )

    print(
        "Entrées :",
        device_info["max_input_channels"]
    )

    print(
        "Host API :",
        sd.query_hostapis(
            device_info["hostapi"]
        )["name"]
    )

except Exception as e:

    print("❌ Impossible d'accéder au device 19")
    print(e)

    input("\nAppuyez sur ENTER pour quitter...")
    raise SystemExit

print()


# ============================================================
# EXTRACTION MFCC
# ============================================================

def extract_mfcc(filename):

    # --------------------------------------------------------
    # Chargement + conversion vers 16 kHz
    # --------------------------------------------------------

    audio, sr = librosa.load(
        filename,
        sr=SAMPLE_RATE,
        mono=True
    )

    # --------------------------------------------------------
    # Normalisation audio
    # --------------------------------------------------------

    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 0:

        audio = audio / max_value

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40,
        n_fft=512,
        hop_length=256
    )

    # --------------------------------------------------------
    # Taille fixe
    # --------------------------------------------------------

    MAX_LENGTH = 126

    if mfcc.shape[1] < MAX_LENGTH:

        pad = MAX_LENGTH - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            (
                (0, 0),
                (0, pad)
            ),
            mode="constant"
        )

    else:

        mfcc = mfcc[:, :MAX_LENGTH]

    # --------------------------------------------------------
    # Normalisation MFCC
    # --------------------------------------------------------

    mfcc = (
        mfcc - np.mean(mfcc)
    ) / (
        np.std(mfcc) + 1e-8
    )

    # --------------------------------------------------------
    # Dimension CNN
    # --------------------------------------------------------

    mfcc = mfcc[..., np.newaxis]

    return np.expand_dims(
        mfcc,
        axis=0
    )


# ============================================================
# ENREGISTREMENT MICROPHONE
# ============================================================

def record_audio():

    print()
    print("======================================")
    print("          TEST VOCAL")
    print("======================================")
    print()

    print("Appuyez sur ENTER puis dites :")
    print()
    print("             BONJOUR")
    print()

    input()

    print("🎤 Enregistrement...")
    print(
        f"Durée : {DURATION} secondes"
    )

    try:

        # ----------------------------------------------------
        # Nombre d'échantillons
        # ----------------------------------------------------

        num_samples = int(
            DURATION * RECORD_SAMPLE_RATE
        )

        # ----------------------------------------------------
        # Enregistrement
        # ----------------------------------------------------

        audio = sd.rec(
            num_samples,
            samplerate=RECORD_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=MICROPHONE_DEVICE
        )

        sd.wait()

        # ----------------------------------------------------
        # Conversion mono
        # ----------------------------------------------------

        audio = audio.flatten()

        # ----------------------------------------------------
        # Vérification du signal
        # ----------------------------------------------------

        niveau = np.max(
            np.abs(audio)
        )

        print()

        print(
            "Niveau audio : {:.5f}".format(
                float(niveau)
            )
        )

        if niveau < 0.005:

            print(
                "⚠️ Attention : signal microphone très faible."
            )

        # ----------------------------------------------------
        # Sauvegarde
        # ----------------------------------------------------

        sf.write(
            TEMP_FILE,
            audio,
            RECORD_SAMPLE_RATE
        )

        print("✓ Enregistrement terminé")

        return True

    except Exception as e:

        print()
        print("======================================")
        print("       ERREUR MICROPHONE")
        print("======================================")
        print()

        print(e)

        return False


# ============================================================
# PREDICTION
# ============================================================

def predict():

    # --------------------------------------------------------
    # Enregistrement
    # --------------------------------------------------------

    success = record_audio()

    if not success:

        return

    # --------------------------------------------------------
    # Extraction MFCC
    # --------------------------------------------------------

    print()
    print("Analyse de la voix...")

    try:

        x = extract_mfcc(
            TEMP_FILE
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            x,
            verbose=0
        )[0]

        # ----------------------------------------------------
        # Classe maximale
        # ----------------------------------------------------

        index = np.argmax(
            prediction
        )

        confidence = float(
            prediction[index]
        )

        result = CLASSES[index]

        # ----------------------------------------------------
        # Seuil
        # ----------------------------------------------------

        if confidence < THRESHOLD:

            result = "AUTRE"

        # ----------------------------------------------------
        # Affichage
        # ----------------------------------------------------

        print()
        print("======================================")
        print("             RESULTAT")
        print("======================================")
        print()

        print(
            "MOI       : {:.2f}%".format(
                prediction[0] * 100
            )
        )

        print(
            "PERSONNE  : {:.2f}%".format(
                prediction[1] * 100
            )
        )

        print(
            "AUTRE     : {:.2f}%".format(
                prediction[2] * 100
            )
        )

        print()
        print("--------------------------------------")

        print(
            "RESULTAT  : {}".format(
                result
            )
        )

        print(
            "CONFIANCE : {:.2f}%".format(
                confidence * 100
            )
        )

        print("--------------------------------------")
        print()

    except Exception as e:

        print()
        print("❌ ERREUR PREDICTION")
        print(e)


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    try:

        while True:

            predict()

            again = input(
                "Faire un autre test ? (o/n) : "
            )

            if again.lower() not in [
                "o",
                "oui"
            ]:

                break

    except KeyboardInterrupt:

        print()
        print("Programme arrêté.")

    finally:

        if os.path.exists(
            TEMP_FILE
        ):

            try:

                os.remove(
                    TEMP_FILE
                )

            except:

                pass

        print()
        print("Fin du programme.")