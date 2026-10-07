import os
import time
import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
import soundfile as sf


# ============================================================
# CONFIGURATION
# ============================================================

KEYWORD_MODEL_PATH = "keyword_model.keras"
SPEAKER_MODEL_PATH = "speaker_model.keras"

KEYWORD_CLASSES = [
    "bonjour",
    "autre_mot"
]

SPEAKER_CLASSES = [
    "moi",
    "personne"
]

# Paramètres utilisés pendant l'entraînement
SAMPLE_RATE = 16000
DURATION = 2
MAX_LENGTH = SAMPLE_RATE * DURATION
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 256

# Fréquence d'enregistrement du microphone
RECORD_SAMPLE_RATE = 48000

TEMP_AUDIO = "temp_test.wav"


# ============================================================
# TITRE
# ============================================================

print("==============================================")
print("       RECONNAISSANCE VOCALE IA")
print("       KEYWORD + SPEAKER")
print("==============================================")


# ============================================================
# CHARGEMENT DES MODELES
# ============================================================

print("\nChargement des modèles...")

try:

    keyword_model = tf.keras.models.load_model(
        KEYWORD_MODEL_PATH
    )

    speaker_model = tf.keras.models.load_model(
        SPEAKER_MODEL_PATH
    )

    print("✅ Keyword model chargé.")
    print("✅ Speaker model chargé.")

except Exception as e:

    print("\n❌ ERREUR : impossible de charger les modèles.")
    print(e)

    input("\nAppuyez sur Entrée pour quitter...")
    raise SystemExit


# ============================================================
# RECHERCHE DES MICROPHONES
# ============================================================

def trouver_microphones():

    print("\n==============================================")
    print("       RECHERCHE DU MICROPHONE")
    print("==============================================")

    try:

        devices = sd.query_devices()

    except Exception as e:

        print("❌ Impossible de lire les périphériques audio.")
        print(e)

        return None

    candidats_realtek = []
    candidats_autres = []

    for i, device in enumerate(devices):

        nom = device["name"]
        inputs = device["max_input_channels"]

        if inputs > 0:

            print(
                f"[{i}] {nom} | INPUT={inputs}"
            )

            nom_lower = nom.lower()

            if (
                "microphone" in nom_lower
                and "realtek" in nom_lower
            ):

                candidats_realtek.append(i)

            else:

                candidats_autres.append(i)

    # Priorité Realtek
    if len(candidats_realtek) > 0:

        print("\nMicrophones Realtek trouvés :")

        for device_id in candidats_realtek:

            print(
                f"  → Device {device_id} : "
                f"{sd.query_devices(device_id)['name']}"
            )

        return candidats_realtek

    # Sinon autres microphones
    if len(candidats_autres) > 0:

        print("\nAutres microphones disponibles :")

        for device_id in candidats_autres:

            print(
                f"  → Device {device_id} : "
                f"{sd.query_devices(device_id)['name']}"
            )

        return candidats_autres

    return None


# ============================================================
# TEST D'UN MICROPHONE
# ============================================================

def tester_microphone(device_id):

    print(
        f"\nTest du microphone device {device_id}..."
    )

    try:

        info = sd.query_devices(device_id)

        print(
            "Nom :",
            info["name"]
        )

        host_api = sd.query_hostapis(
            info["hostapi"]
        )["name"]

        print(
            "Host API :",
            host_api
        )

        print(
            "Entrées :",
            info["max_input_channels"]
        )

        # Petit test d'une seconde
        nombre_echantillons = (
            RECORD_SAMPLE_RATE * 1
        )

        print(
            "🎤 Test pendant 1 seconde..."
        )

        audio = sd.rec(
            nombre_echantillons,
            samplerate=RECORD_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device_id
        )

        sd.wait()

        audio = audio.flatten()

        niveau = np.max(
            np.abs(audio)
        )

        print(
            "Niveau audio :",
            round(float(niveau), 5)
        )

        print(
            "✅ Microphone fonctionnel."
        )

        return True

    except Exception as e:

        print(
            "❌ Échec du microphone :"
        )

        print(e)

        return False


# ============================================================
# SELECTION DU MICROPHONE
# ============================================================

microphones = trouver_microphones()

if microphones is None:

    print(
        "\n❌ Aucun microphone disponible."
    )

    print("""
Vérifiez dans Windows :

1. Paramètres
2. Système
3. Son
4. Entrée
5. Sélectionner le microphone
6. Confidentialité
7. Microphone
8. Autoriser l'accès au microphone
9. Autoriser les applications de bureau
""")

    input(
        "\nAppuyez sur Entrée pour quitter..."
    )

    raise SystemExit


MICROPHONE_DEVICE = None

# Tester les microphones un par un
for device_id in microphones:

    if tester_microphone(device_id):

        MICROPHONE_DEVICE = device_id

        break


# ============================================================
# SI AUCUN MICROPHONE NE FONCTIONNE
# ============================================================

if MICROPHONE_DEVICE is None:

    print("\n==============================================")
    print("       AUCUN MICROPHONE FONCTIONNEL")
    print("==============================================")

    print("""
PortAudio/sounddevice n'arrive pas à ouvrir
les microphones détectés.

Vérifiez :

- Windows → Paramètres → Son
- Microphone activé
- Autorisation microphone
- Pilote Realtek
- Aucune autre application ne bloque le microphone
""")

    input(
        "\nAppuyez sur Entrée pour quitter..."
    )

    raise SystemExit


# ============================================================
# INFORMATIONS MICROPHONE
# ============================================================

info = sd.query_devices(
    MICROPHONE_DEVICE
)

print("\n==============================================")
print("       MICROPHONE SELECTIONNE")
print("==============================================")

print(
    "Device :",
    MICROPHONE_DEVICE
)

print(
    "Nom :",
    info["name"]
)

print(
    "Host API :",
    sd.query_hostapis(
        info["hostapi"]
    )["name"]
)

print(
    "Fréquence d'enregistrement :",
    RECORD_SAMPLE_RATE,
    "Hz"
)

print(
    "Durée :",
    DURATION,
    "secondes"
)


# ============================================================
# EXTRACTION MFCC
# ============================================================

def extract_mfcc(file_path):

    # --------------------------------------------------------
    # Chargement du fichier
    # Conversion automatique vers 16 kHz
    # --------------------------------------------------------

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # --------------------------------------------------------
    # Normalisation du signal
    # --------------------------------------------------------

    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 0:

        audio = audio / max_value

    # --------------------------------------------------------
    # Ajustement exactement à 2 secondes
    # --------------------------------------------------------

    if len(audio) < MAX_LENGTH:

        audio = np.pad(
            audio,
            (
                0,
                MAX_LENGTH - len(audio)
            )
        )

    else:

        audio = audio[:MAX_LENGTH]

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # --------------------------------------------------------
    # Normalisation MFCC
    # --------------------------------------------------------

    mfcc = (
        mfcc - np.mean(mfcc)
    ) / (
        np.std(mfcc) + 1e-8
    )

    return mfcc


# ============================================================
# ENREGISTREMENT DE 2 SECONDES
# ============================================================

def enregistrer_audio():

    print("\n==============================================")
    print("          ENREGISTREMENT VOCAL")
    print("==============================================")

    print()
    print("🎤 PRÉPAREZ-VOUS...")
    
    time.sleep(1)

    print()
    print("🔴 ENREGISTREMENT EN COURS")
    print("👉 Dites : « bonjour »")
    print()
    print("⏱️ Durée : 2 secondes")

    try:

        # ----------------------------------------------------
        # EXACTEMENT 2 secondes
        # ----------------------------------------------------

        nombre_echantillons = (
            RECORD_SAMPLE_RATE * DURATION
        )

        # ----------------------------------------------------
        # Démarrage du microphone
        # ----------------------------------------------------

        audio = sd.rec(
            nombre_echantillons,
            samplerate=RECORD_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=MICROPHONE_DEVICE
        )

        # ----------------------------------------------------
        # IMPORTANT :
        # attendre obligatoirement la fin
        # ----------------------------------------------------

        sd.wait()

        # ----------------------------------------------------
        # FIN ENREGISTREMENT
        # ----------------------------------------------------

        print()
        print("🟢 ENREGISTREMENT TERMINÉ.")
        print("✅ 2 secondes complètes enregistrées.")

        # Conversion 1D
        audio = audio.flatten()

        # Niveau audio
        niveau = np.max(
            np.abs(audio)
        )

        print(
            "Niveau audio :",
            round(float(niveau), 5)
        )

        if niveau < 0.005:

            print(
                "⚠️ ATTENTION : signal très faible."
            )

        # ----------------------------------------------------
        # Sauvegarde temporaire
        # ----------------------------------------------------

        sf.write(
            TEMP_AUDIO,
            audio,
            RECORD_SAMPLE_RATE
        )

        print(
            "Audio temporaire sauvegardé."
        )

        return True

    except Exception as e:

        print("\n==============================================")
        print("         ERREUR MICROPHONE")
        print("==============================================")

        print(e)

        return False


# ============================================================
# PREDICTION
# ============================================================

def prediction():

    try:

        print("\n==============================================")
        print("          ANALYSE IA EN COURS")
        print("==============================================")

        # ----------------------------------------------------
        # Extraction MFCC
        # ----------------------------------------------------

        mfcc = extract_mfcc(
            TEMP_AUDIO
        )

        # ----------------------------------------------------
        # Shape :
        #
        # (40, temps)
        #
        # devient :
        #
        # (1, 40, temps, 1)
        # ----------------------------------------------------

        X = mfcc[
            np.newaxis,
            ...,
            np.newaxis
        ]

        # ====================================================
        # KEYWORD MODEL
        # ====================================================

        keyword_probabilities = (
            keyword_model.predict(
                X,
                verbose=0
            )[0]
        )

        keyword_index = np.argmax(
            keyword_probabilities
        )

        keyword = KEYWORD_CLASSES[
            keyword_index
        ]

        keyword_confidence = (
            keyword_probabilities[
                keyword_index
            ] * 100
        )

        # ====================================================
        # SPEAKER MODEL
        # ====================================================

        speaker_probabilities = (
            speaker_model.predict(
                X,
                verbose=0
            )[0]
        )

        speaker_index = np.argmax(
            speaker_probabilities
        )

        speaker = SPEAKER_CLASSES[
            speaker_index
        ]

        speaker_confidence = (
            speaker_probabilities[
                speaker_index
            ] * 100
        )

        # ====================================================
        # RESULTATS
        # ====================================================

        print("\n==============================================")
        print("                  RESULTAT")
        print("==============================================")

        print()
        print(
            f"Keyword : {keyword}"
        )

        print(
            f"Speaker : {speaker}"
        )

        print()

        print(
            f"Confiance Keyword : "
            f"{keyword_confidence:.2f}%"
        )

        print(
            f"Confiance Speaker : "
            f"{speaker_confidence:.2f}%"
        )

        print("----------------------------------------------")

        # ====================================================
        # DECISION
        # ====================================================

        if (
            keyword == "bonjour"
            and
            speaker == "moi"
        ):

            print(
                "✅ COMMANDE AUTORISEE"
            )

            print(
                "Bonjour Nantenaina !"
            )

        elif (
            keyword == "bonjour"
            and
            speaker == "personne"
        ):

            print(
                " MOT CORRECT"
            )

            print(
                " UTILISATEUR NON AUTORISE"
            )

        else:

            print(
                "❌ COMMANDE REFUSEE"
            )

        print(
            "=============================================="
        )

        return True

    except Exception as e:

        print("\n❌ ERREUR PREDICTION :")
        print(e)

        return False


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

while True:

    print("\n")
    print("==============================================")
    print("       SYSTEME DE RECONNAISSANCE VOCALE")
    print("==============================================")

    input(
        "\nAppuyez sur ENTRÉE pour commencer..."
    )

    # ========================================================
    # 1. ENREGISTREMENT
    # ========================================================

    succes = enregistrer_audio()

    if not succes:

        print(
            "\n❌ Enregistrement impossible."
        )

        continuer = input(
            "\nRéessayer ? (o/n) : "
        )

        if continuer.lower() not in [
            "o",
            "oui"
        ]:

            break

        continue

    # ========================================================
    # 2. PREDICTION
    # ========================================================
    #
    # IMPORTANT :
    # Cette partie n'est exécutée qu'après
    # les 2 secondes complètes.
    #

    prediction()

    # ========================================================
    # 3. NOUVEAU TEST
    # ========================================================

    print()

    continuer = input(
        "Nouveau test ? (o/n) : "
    )

    if continuer.lower() not in [
        "o",
        "oui"
    ]:

        print(
            "\n=============================================="
        )

        print(
            "Programme terminé."
        )

        print(
            "=============================================="
        )

        # Suppression fichier temporaire
        if os.path.exists(TEMP_AUDIO):

            try:

                os.remove(
                    TEMP_AUDIO
                )

            except:

                pass

        break