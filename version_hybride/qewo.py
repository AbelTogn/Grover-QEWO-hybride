"""
QEWO (Quantum-Enhanced Weight Optimization), Jura & Udrescu, arXiv:2504.14568.

Correspondance avec l'algorithme de l'article :
  1. poids uniformes dans [-1, 1], réseau sans biais, activation tanh   (reseau.py)
  2. par couche t et par poids : intervalle w ± alpha_t * sigma_t, N candidats
     x_k = w - alpha*sigma + k * 2*alpha*sigma/(N-1)                    (StrategieQEWO)
  3. évaluation CLASSIQUE de la perte de chaque candidat                (reseau.py)
  4. M = {k : L_k <= min(L) + tol_ratio * min(L)}                       (quantique.py)
  5. circuit de Grover sur le registre d'indices, mesure -> k*          (Main.qs)
  6. w <- x_k*  (sans retour arrière)
  7. alpha_t *= GAMMA_DOWN si la perte baisse, *= GAMMA_UP sinon (bornes)

Adaptations au contrôle (l'article traite de la classification) : perte =
coût d'épisode du pendule, sortie tanh, pas de dropout (il rendrait la perte
d'un candidat aléatoire).

Une « stratégie » (génération des candidats + règle d'acceptation) peut être
remplacée pour construire des baselines de même budget (voir baselines.py).
"""
import time

import numpy as np

import config as cfg
from pendule import evaluer_pd
from quantique import StatsQuantiques, indices_marques, rechercher_grover
from reseau import (
    ReseauQEWO, evaluer_candidats, evaluer_reseau, sauvegarder_modele,
)


class StrategieQEWO:
    """Étapes 2 et 6 de l'article : grille régulière, mise à jour inconditionnelle."""

    nom = "qewo"

    def candidats(self, valeur, borne, n):
        return np.linspace(valeur - borne, valeur + borne, n)

    def accepter(self, perte_candidat, perte_avant):
        return True


def formater_duree(secondes):
    secondes = max(0, int(secondes))
    heures, reste = divmod(secondes, 3600)
    minutes, secondes = divmod(reste, 60)
    if heures:
        return f"{heures} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secondes:02d} s"
    return f"{secondes} s"


def graines_entrainement(epoch):
    """
    Lot de graines de l'epoch (équivalent d'un mini-batch). Dépend de la graine
    du run : les runs voient des épisodes différents. Reste disjoint des graines
    de validation (5000...) et de test final (20000...).
    """
    base = 100_000 + 10_000 * cfg.GRAINE_INIT + 100 * epoch
    return [base + k for k in range(cfg.N_GRAINES_ENTRAINEMENT)]


def choisir_candidat(pertes, tol_ratio, executeur, stats):
    """
    Étapes 4 et 5. Retourne l'indice retenu, ou None si le circuit n'a donné
    aucune mesure valide. executeur=None : baseline classique (argmin).
    """
    pertes = np.asarray(pertes, dtype=float)
    if executeur is None:
        return int(np.argmin(pertes))

    marques = indices_marques(pertes, tol_ratio)        # étape 4 (classique)
    debut = time.perf_counter()
    indice = rechercher_grover(len(pertes), marques, executeur, stats)  # étape 5
    stats.temps += time.perf_counter() - debut
    return indice


def entrainer(nn=None, epochs_deja_faites=0, executeur=None, journal=None,
              verbose=True, strategie=None, chemin_modele=None, meta=None):
    """
    Entraînement complet.
      executeur : fonction (nb_qubits, marques, iterations) -> indice, ou None (argmin).
      journal   : fonction appelée avec un dict par epoch (voir journal.JournalCSV).
      verbose   : False supprime la barre de progression par poids.
      strategie : objet avec .candidats(valeur, borne, n) et .accepter(perte, avant).
    """
    strategie = StrategieQEWO() if strategie is None else strategie
    architecture = [3 * cfg.N_SEGMENTS, 32, 32, cfg.N_SEGMENTS]
    if nn is None:
        nn = ReseauQEWO(architecture)
    elif nn.architecture != architecture:
        raise ValueError(
            "L'architecture du modèle chargé ne correspond pas à la configuration actuelle."
        )

    alphas = nn.alpha_couches
    if alphas is None or len(alphas) != len(nn.poids):
        alphas = [cfg.ALPHA_INITIAL for _ in nn.poids]

    nb_couches = len(nn.poids)
    candidats_par_couche = [
        cfg.N_CANDIDATS_SORTIE if t == nb_couches - 1 else cfg.N_CANDIDATS_CACHES
        for t in range(nb_couches)
    ]
    tol_par_couche = [
        cfg.TOL_RATIO_SORTIE if t == nb_couches - 1 else cfg.TOL_RATIO_CACHE
        for t in range(nb_couches)
    ]
    poids_par_epoch = sum(W.size for W in nn.poids)
    evals_par_epoch = sum(
        W.size * c for W, c in zip(nn.poids, candidats_par_couche)
    )
    mode = "grover" if executeur is not None else "argmin"

    perte_pd = evaluer_pd(list(cfg.GRAINES_VALIDATION))
    print(f"Référence PD (validation) : {perte_pd:.6f}")

    # Calibration : temps d'évaluation d'un lot, rapporté à un candidat.
    graines = graines_entrainement(epochs_deja_faites)
    debut = time.perf_counter()
    evaluer_candidats(
        nn, 0, 0, 0, np.full(cfg.N_CANDIDATS_CACHES, nn.poids[0][0, 0]), graines
    )
    temps_par_eval = (time.perf_counter() - debut) / cfg.N_CANDIDATS_CACHES
    print(f"Temps mesuré par candidat évalué : {temps_par_eval * 1000:.2f} ms")
    print(
        f"Entraînement : {poids_par_epoch} poids/epoch, "
        f"{evals_par_epoch:,} évaluations/epoch | stratégie {strategie.nom} | "
        f"sélection : {mode}".replace(",", " ")
    )

    stats = StatsQuantiques()
    temps_evals = 0.0
    nb_evals = 0
    poids_total = poids_par_epoch * cfg.EPOCHS
    evals_totales = evals_par_epoch * cfg.EPOCHS
    poids_faits = 0
    debut_entrainement = time.perf_counter()

    precedent = {"recherches": 0, "somme": 0, "sans_iteration": 0}

    def ecrire_journal(epoch, perte_train, perte_val):
        if journal is None:
            return
        # taille de M et part de recherches sans amplification : valeurs de CETTE epoch
        d_rech = stats.recherches - precedent["recherches"]
        d_somme = stats.somme_marques - precedent["somme"]
        d_sans = stats.recherches_sans_iteration - precedent["sans_iteration"]
        precedent.update(
            recherches=stats.recherches, somme=stats.somme_marques,
            sans_iteration=stats.recherches_sans_iteration,
        )
        journal({
            "epoch": epoch,
            "evaluations": nb_evals,
            "requetes_oracle": stats.requetes_oracle,
            "mesures": stats.mesures,
            "mesures_hors_M": stats.mesures_invalides,
            "poids_sans_maj": stats.poids_sans_mise_a_jour,
            "taille_M_moy": d_somme / d_rech if d_rech else 0.0,
            "part_sans_iteration": d_sans / d_rech if d_rech else 0.0,
            "perte_train": perte_train,
            "perte_val": perte_val,
            "perte_pd_val": perte_pd,
            "temps_s": time.perf_counter() - debut_entrainement,
        })

    # Ligne epoch=0 : réseau initial (point de départ des courbes).
    ecrire_journal(
        epochs_deja_faites,
        evaluer_reseau(nn, graines),
        evaluer_reseau(nn, list(cfg.GRAINES_VALIDATION)),
    )

    def temps_restant():
        eval_moy = temps_evals / nb_evals if nb_evals else temps_par_eval
        quantique_moy = stats.temps / poids_faits if poids_faits else 0.0
        return (evals_totales - nb_evals) * eval_moy + (poids_total - poids_faits) * quantique_moy

    for epoch in range(cfg.EPOCHS):
        numero = epochs_deja_faites + epoch
        graines = graines_entrainement(numero)
        perte_courante = evaluer_reseau(nn, graines)
        ameliorations = 0
        invalides_debut = stats.mesures_invalides
        sans_maj_debut = stats.poids_sans_mise_a_jour

        for t, W in enumerate(nn.poids):
            n_candidats = candidats_par_couche[t]
            tol_ratio = tol_par_couche[t]
            nom_couche = f"{t + 1}/{nb_couches}"
            print(
                f"\nEpoch {epoch + 1}/{cfg.EPOCHS} — couche {nom_couche} "
                f"({W.shape[1]} entrées → {W.shape[0]} neurones) | "
                f"perte de départ {perte_courante:.6f}"
            )

            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    poids_faits += 1
                    valeur = float(W[i, j])
                    sigma = max(float(np.std(W)), 1e-12)
                    perte_avant = perte_courante
                    borne = alphas[t] * sigma

                    # Étape 2 : N candidats (grille régulière pour QEWO).
                    candidats = strategie.candidats(valeur, borne, n_candidats)

                    # Étape 3 : évaluation classique.
                    debut_eval = time.perf_counter()
                    pertes = evaluer_candidats(nn, t, i, j, candidats, graines)
                    temps_evals += time.perf_counter() - debut_eval
                    nb_evals += n_candidats

                    # Étapes 4-5 : ensemble M puis Grover.
                    indice = choisir_candidat(pertes, tol_ratio, executeur, stats)

                    if indice is not None and strategie.accepter(
                        float(pertes[indice]), perte_avant
                    ):
                        # Étape 6 : mise à jour (sans retour arrière pour QEWO).
                        W[i, j] = float(candidats[indice])
                        perte_courante = float(pertes[indice])

                    # Étape 7 : adaptation de alpha. (Sans mise à jour, la
                    # perte n'a pas baissé : alpha augmente.)
                    if perte_courante < perte_avant:
                        alphas[t] = max(cfg.GAMMA_DOWN * alphas[t], cfg.ALPHA_MIN)
                        ameliorations += 1
                    else:
                        alphas[t] = min(cfg.GAMMA_UP * alphas[t], cfg.ALPHA_MAX)

                    if verbose:
                        print(
                            f"\rEpoch {epoch + 1}/{cfg.EPOCHS} | couche {nom_couche} | "
                            f"poids {poids_faits}/{poids_total} "
                            f"({100 * poids_faits / poids_total:.1f} %) | "
                            f"restant ~{formater_duree(temps_restant())}",
                            end="", flush=True,
                        )

        if verbose:
            print()
        nn.alpha_couches = list(alphas)
        perte_validation = evaluer_reseau(nn, list(cfg.GRAINES_VALIDATION))
        ligne = (
            f"Epoch {epoch + 1}/{cfg.EPOCHS} terminée | perte entraînement "
            f"{perte_courante:.6f} | perte validation {perte_validation:.6f} "
            f"(PD : {perte_pd:.6f}) | améliorations {ameliorations}"
        )
        if executeur is not None:
            ligne += (
                f" | mesures hors de M {stats.mesures_invalides - invalides_debut}"
                f" | poids non modifiés {stats.poids_sans_mise_a_jour - sans_maj_debut}"
            )
        print(ligne, flush=True)
        ecrire_journal(numero + 1, perte_courante, perte_validation)
        sauvegarder_modele(
            nn, alphas, numero + 1, perte_courante, perte_validation,
            chemin=chemin_modele, meta=meta,
        )

    nn.alpha_couches = list(alphas)
    destination = chemin_modele if chemin_modele is not None else cfg.FICHIER_MODELE
    print(f"Modèle sauvegardé : {destination}")
    print(f"Durée totale : {formater_duree(time.perf_counter() - debut_entrainement)}")
    return nn
