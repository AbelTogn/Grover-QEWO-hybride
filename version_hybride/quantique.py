"""
Frontière classique / quantique de QEWO.

Côté classique (calculé ici, en Python) :
  - étape 4 : ensemble M = {k : L_k <= min(L) + tol_ratio * min(L)} ;
  - taille du registre ceil(log2 N) et nombre optimal d'itérations de Grover.
Côté quantique (Main.qs) :
  - étape 5 : superposition, oracle sur M, diffusion, mesure du registre.

Un « exécuteur » est une fonction (nb_qubits, marques, iterations) -> indice mesuré.
Trois sont fournis :
  - executeur_qdk    : le circuit Q# sur le simulateur du QDK ;
  - executeur_numpy  : émulation exacte du même circuit (tests, sans QDK) ;
  - pour du vrai matériel, soumettre exporter_qir(...) puis renvoyer l'indice
    mesuré (voir la docstring).
"""
import math
from dataclasses import dataclass

import numpy as np

import config as cfg


@dataclass
class StatsQuantiques:
    mesures: int = 0            # nombre d'exécutions du circuit (1 mesure chacune)
    requetes_oracle: int = 0    # applications de l'oracle (= itérations de Grover, relances comprises)
    mesures_invalides: int = 0  # mesure tombée hors de M (bruit, ou N non puissance de 2)
    poids_sans_mise_a_jour: int = 0  # aucune mesure valide après MAX_ESSAIS_MESURE
    recherches: int = 0         # nombre de recherches de Grover (un poids = une recherche)
    somme_marques: int = 0      # somme des |M| (pour la taille moyenne de M)
    recherches_sans_iteration: int = 0  # |M|/2^n >= 1/2 : aucune amplification, tirage uniforme
    temps: float = 0.0


# ------------------------------------------------------------------
# Partie classique de l'étape 4 / préparation de l'étape 5
# ------------------------------------------------------------------

def indices_marques(pertes, tol_ratio):
    """Étape 4 : M = {k : L_k <= min(L) + tol_ratio * min(L)}."""
    pertes = np.asarray(pertes, dtype=float)
    minimum = float(np.min(pertes))
    return np.flatnonzero(pertes <= minimum + tol_ratio * minimum)


def nb_qubits_registre(n_candidats):
    """Nombre de qubits d'indice : ceil(log2 N)."""
    return max(1, math.ceil(math.log2(n_candidats)))


def iterations_optimales(n_marques, nb_qubits):
    """
    Règle standard de Grover : k = round(pi / (4 theta) - 1/2), avec
    theta = arcsin(sqrt(M / 2^n)). Elle place l'amplitude des états marqués à sa
    PREMIÈRE pointe, sin^2((2k+1) theta). La taille de l'espace est celle du
    registre, 2^n, pas le nombre de candidats.
    Si M >= 2^n / 2, la première pointe est k=0 : une mesure uniforme tombe déjà
    dans M avec une probabilité >= 1/2, on n'itère pas. En cas d'égalité entre
    deux valeurs de k, on garde la plus petite (moins de portes, donc moins de
    bruit sur du matériel réel).
    """
    taille = 2**nb_qubits
    if n_marques <= 0 or 2 * n_marques >= taille:
        return 0
    theta = math.asin(math.sqrt(n_marques / taille))
    return max(0, math.ceil(math.pi / (4.0 * theta) - 1.0))


def rechercher_grover(n_candidats, marques, executeur, stats):
    """
    Étape 5 : exécute le circuit de Grover et retourne l'indice mesuré.
    Une mesure hors de M est rejetée et le circuit est relancé. Si aucune
    mesure valide n'est obtenue, retourne None (le poids reste inchangé).
    """
    if n_candidats == 1:
        return 0
    nb_qubits = nb_qubits_registre(n_candidats)
    iterations = iterations_optimales(len(marques), nb_qubits)
    valides = {int(m) for m in marques}
    stats.recherches += 1
    stats.somme_marques += len(marques)
    if iterations == 0:
        stats.recherches_sans_iteration += 1
    liste = [int(m) for m in marques]

    for _ in range(cfg.MAX_ESSAIS_MESURE):
        indice = executeur(nb_qubits, liste, iterations)
        stats.mesures += 1
        stats.requetes_oracle += iterations
        if indice in valides:
            return indice
        stats.mesures_invalides += 1

    stats.poids_sans_mise_a_jour += 1
    return None


# ------------------------------------------------------------------
# Exécuteurs
# ------------------------------------------------------------------

def creer_executeur_qdk():
    """Circuit Q# (Main.qs) sur le simulateur du QDK."""
    from qdk import qsharp  # import tardif : inutile en mode argmin

    if not cfg.FICHIER_QSHARP.is_file():
        raise FileNotFoundError(f"Fichier Q# introuvable : {cfg.FICHIER_QSHARP}")

    qsharp.init(target_profile=qsharp.TargetProfile.Unrestricted)
    if hasattr(qsharp, "set_quantum_seed"):
        qsharp.set_quantum_seed(cfg.GRAINE_INIT)  # mesures reproductibles
    qsharp.eval(cfg.FICHIER_QSHARP.read_text(encoding="utf-8"))

    def executeur(nb_qubits, marques, iterations):
        liste = ", ".join(str(m) for m in marques)
        resultat = qsharp.eval(f"GroverIndice({nb_qubits}, [{liste}], {iterations})")
        try:
            return int(resultat)
        except (TypeError, ValueError) as erreur:
            raise RuntimeError(f"Résultat inattendu : {resultat!r}") from erreur

    return executeur


def exporter_qir(nb_qubits, marques, iterations):
    """
    NON TESTÉ ici (QDK absent de l'environnement de développement).
    Compile le circuit Grover (retour : Result[]) pour le profil Base, prêt à
    être soumis à un service matériel. Le programme compilé contient les
    données de M et le nombre d'itérations sous forme de constantes.
    La mesure obtenue (bits, poids faible en premier) se convertit en indice
    par sum(bit << i), puis rechercher_grover() la valide comme d'habitude.
    """
    from qdk import qsharp

    qsharp.init(target_profile=qsharp.TargetProfile.Base)
    qsharp.eval(cfg.FICHIER_QSHARP.read_text(encoding="utf-8"))
    liste = ", ".join(str(int(m)) for m in marques)
    return qsharp.compile(f"Grover({nb_qubits}, [{liste}], {iterations})")


def probabilites_grover(nb_qubits, marques, iterations):
    """Vecteur d'état exact du circuit : probabilité de mesurer chaque indice."""
    taille = 2**nb_qubits
    amplitudes = np.full(taille, 1.0 / math.sqrt(taille))
    masque = np.zeros(taille, dtype=bool)
    masque[list(marques)] = True
    for _ in range(iterations):
        amplitudes[masque] *= -1.0                   # oracle (kickback de phase)
        amplitudes = 2.0 * amplitudes.mean() - amplitudes  # diffusion (phase globale près)
    return np.abs(amplitudes) ** 2


def creer_executeur_numpy(graine=None):
    """Émulation exacte du circuit de Main.qs, avec tirage de la mesure."""
    graine = cfg.GRAINE_INIT if graine is None else graine
    rng = np.random.default_rng(graine)

    def executeur(nb_qubits, marques, iterations):
        p = probabilites_grover(nb_qubits, marques, iterations)
        return int(rng.choice(len(p), p=p / p.sum()))

    return executeur
