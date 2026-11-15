"""Réseau de neurones QEWO, évaluation classique des candidats (étape 3) et sauvegarde."""
import os

import numpy as np

import config as cfg
from pendule import construire_etat, simuler_lot


class ReseauQEWO:
    """
    Réseau sans biais, activation tanh (comme dans l'article).
    Entrées : cos(theta_i), sin(theta_i), omega_i / 10 pour chaque segment.
    Sorties : commandes moteur normalisées dans [-1, 1].
    """

    def __init__(self, architecture, poids=None, graine=None):
        self.architecture = list(architecture)
        self.alpha_couches = None

        if poids is None:
            graine = cfg.GRAINE_INIT if graine is None else graine
            rng = np.random.default_rng(graine)
            # Étape 1 : poids uniformes dans [-1, 1].
            self.poids = [
                rng.uniform(-1.0, 1.0, size=(sortie, entree))
                for entree, sortie in zip(self.architecture[:-1], self.architecture[1:])
            ]
        else:
            self.poids = [np.array(w, dtype=float) for w in poids]

    def forward(self, X):
        """X de forme (..., entrée) -> (..., sortie)."""
        A = X
        for W in self.poids:
            A = np.tanh(A @ W.T)
        return A

    def predire(self, etat):
        return self.forward(np.asarray(etat, dtype=float))


def evaluer_candidats(nn, t, i, j, candidats, graines, duree=None):
    """
    Étape 3 de QEWO (classique) : perte du réseau pour chaque valeur candidate
    de nn.poids[t][i, j], les autres poids restant inchangés.
    Les candidats sont simulés en parallèle. Retourne un tableau de forme (k,).
    """
    candidats = np.asarray(candidats, dtype=float)
    k = len(candidats)
    poids_lot = [W.T for W in nn.poids]  # (entrée, sortie), partagés par les k candidats
    Wt = np.broadcast_to(nn.poids[t], (k, *nn.poids[t].shape)).copy()
    Wt[:, i, j] = candidats
    poids_lot[t] = np.swapaxes(Wt, 1, 2)  # (k, entrée, sortie)

    def controle(theta, omega):
        A = construire_etat(theta, omega)  # (k, s, 3n)
        for W in poids_lot:
            A = np.tanh(A @ W)
        return cfg.COUPLE_MAX * A

    pertes = simuler_lot(controle, k, graines, duree)

    if cfg.LAMBDA_L2 > 0.0:
        norme = sum(float(np.sum(W**2)) for W in nn.poids)
        norme_candidats = norme - nn.poids[t][i, j] ** 2 + candidats**2
        pertes = pertes + cfg.LAMBDA_L2 * norme_candidats
    return pertes


def evaluer_reseau(nn, graines, duree=None):
    """Perte moyenne du réseau tel qu'il est."""
    valeur = nn.poids[0][0, 0]
    return float(evaluer_candidats(nn, 0, 0, 0, [valeur], graines, duree)[0])


# ------------------------------------------------------------------
# Sauvegarde (npz, sans pickle)
# ------------------------------------------------------------------

def sauvegarder_modele(nn, alphas, epochs_effectuees, perte, perte_validation):
    donnees = {f"W{t}": W for t, W in enumerate(nn.poids)}
    donnees.update(
        architecture=np.array(nn.architecture),
        alphas=np.array(alphas, dtype=float),
        n_segments=np.array(cfg.N_SEGMENTS),
        epochs_effectuees=np.array(epochs_effectuees),
        perte=np.array(perte),
        perte_validation=np.array(perte_validation),
        methode=np.array("QEWO"),
    )
    temporaire = cfg.FICHIER_MODELE.with_name(cfg.FICHIER_MODELE.name + ".tmp")
    with open(temporaire, "wb") as fichier:
        np.savez(fichier, **donnees)
    os.replace(temporaire, cfg.FICHIER_MODELE)


def charger_modele():
    """Retourne (réseau, nombre d'epochs déjà effectuées)."""
    with np.load(cfg.FICHIER_MODELE, allow_pickle=False) as donnees:
        if int(donnees["n_segments"]) != cfg.N_SEGMENTS:
            raise ValueError(
                f"Le modèle a {int(donnees['n_segments'])} segments, "
                f"mais N_SEGMENTS vaut {cfg.N_SEGMENTS}."
            )
        architecture = donnees["architecture"].tolist()
        poids = [donnees[f"W{t}"] for t in range(len(architecture) - 1)]
        nn = ReseauQEWO(architecture, poids)
        nn.alpha_couches = donnees["alphas"].tolist()
        return nn, int(donnees["epochs_effectuees"])
