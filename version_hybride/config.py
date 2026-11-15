"""Configuration commune à tous les modules."""
from pathlib import Path

# --- Pendule -------------------------------------------------------------
N_SEGMENTS = 3
DT = 0.01
DUREE_EPISODE = 5.0
LONGUEUR = 1.0
MASSE = 1.0
GRAVITE = 9.81
AMORTISSEMENT = 0.08
COUPLE_MAX = 15.0
OMEGA_MAX = 30.0  # garde-fou numérique sur la vitesse angulaire

# --- QEWO (valeurs de l'article, expérience 1) ---------------------------
EPOCHS = 150
N_CANDIDATS_CACHES = 32
N_CANDIDATS_SORTIE = 64
ALPHA_INITIAL = 0.1
GAMMA_DOWN = 0.95
GAMMA_UP = 1.05
ALPHA_MIN = 0.01
ALPHA_MAX = 1.0
TOL_RATIO_CACHE = 0.05
TOL_RATIO_SORTIE = 0.10
LAMBDA_L2 = 0.0  # régularisation L2 (Frobenius) ; 0 = désactivée

# --- Exécution quantique --------------------------------------------------
# Une mesure du registre peut tomber hors de M (bruit matériel) : on relance
# le circuit. Si aucune mesure valide après ce nombre d'essais, le poids
# n'est pas modifié.
MAX_ESSAIS_MESURE = 20

# --- Données d'entraînement / validation ---------------------------------
N_GRAINES_ENTRAINEMENT = 3
GRAINES_VALIDATION = tuple(range(5000, 5010))
GRAINE_INIT = 0

# --- Baseline PD ------------------------------------------------------------
PD_KP = 60.0
PD_KD = 12.0

# --- Fichiers -----------------------------------------------------------------
DOSSIER = Path(__file__).resolve().parent
FICHIER_MODELE = DOSSIER.parent / "modele_pendule_qewo150.npz"
FICHIER_QSHARP = DOSSIER / "Main.qs"