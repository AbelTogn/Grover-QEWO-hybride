"""
Lance UN run d'entraînement sans menu et enregistre ses résultats.

    python version_hybride/experiences.py --methode emulation --graine 0
    python version_hybride/experiences.py --methode argmin --graine 2 --n-cache 19

Méthodes :
    grover     QEWO, circuit Q# sur le simulateur du QDK
    emulation  QEWO, même circuit émulé en NumPy (sans QDK)
    argmin     QEWO avec sélection classique du minimum
    aleatoire  baseline : recherche aléatoire, acceptation gloutonne

Sorties (dans resultats/) :
    courbes/<run>.csv   une ligne par epoch
    modeles/<run>.npz   modèle final (ignoré par git)
"""
import argparse
import sys
import time

import config as cfg
from baselines import StrategieAleatoire
from journal import JournalCSV, run_termine
from qewo import entrainer, formater_duree
from reseau import ReseauQEWO

METHODES = ("grover", "emulation", "argmin", "aleatoire")
VARIANTES = ("normale", "sans_moteur_base")


def identifiant(methode, graine, n_cache, tol_cache, epochs, variante, duree):
    morceaux = [methode, f"s{graine}", f"n{n_cache}", f"t{tol_cache:g}", f"e{epochs}"]
    if variante != "normale":
        morceaux.append(variante)
    if duree != 5.0:
        morceaux.append(f"d{duree:g}")
    return "_".join(morceaux)


def creer_executeur(methode, graine):
    if methode == "grover":
        from quantique import creer_executeur_qdk
        return creer_executeur_qdk()
    if methode == "emulation":
        from quantique import creer_executeur_numpy
        return creer_executeur_numpy(graine)
    return None


def executer_run(methode, graine=0, n_cache=None, tol_cache=None, tol_sortie=None,
                 epochs=None, variante="normale", duree_episode=None,
                 dossier=None, forcer=False, verbose=False):
    """Retourne 'termine' ou 'deja_fait'. Restaure la configuration à la sortie."""
    if methode not in METHODES:
        raise ValueError(f"Méthode inconnue : {methode}")
    if variante not in VARIANTES:
        raise ValueError(f"Variante inconnue : {variante}")

    dossier = cfg.DOSSIER_RESULTATS if dossier is None else dossier
    ancienne = {
        nom: getattr(cfg, nom) for nom in (
            "GRAINE_INIT", "N_CANDIDATS_CACHES", "TOL_RATIO_CACHE", "TOL_RATIO_SORTIE",
            "EPOCHS", "DUREE_EPISODE", "SANS_MOTEUR_BASE",
        )
    }
    try:
        cfg.GRAINE_INIT = graine
        if n_cache is not None:
            cfg.N_CANDIDATS_CACHES = n_cache
        if tol_cache is not None:
            cfg.TOL_RATIO_CACHE = tol_cache
        if tol_sortie is not None:
            cfg.TOL_RATIO_SORTIE = tol_sortie
        if epochs is not None:
            cfg.EPOCHS = epochs
        if duree_episode is not None:
            cfg.DUREE_EPISODE = duree_episode
        cfg.SANS_MOTEUR_BASE = variante == "sans_moteur_base"

        run = identifiant(
            methode, graine, cfg.N_CANDIDATS_CACHES, cfg.TOL_RATIO_CACHE,
            cfg.EPOCHS, variante, cfg.DUREE_EPISODE,
        )
        chemin_csv = dossier / "courbes" / f"{run}.csv"
        chemin_modele = dossier / "modeles" / f"{run}.npz"

        if not forcer and run_termine(chemin_csv, cfg.EPOCHS):
            print(f"[déjà fait] {run}")
            return "deja_fait"

        dossier.mkdir(parents=True, exist_ok=True)
        gitignore = dossier / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("modeles/\n", encoding="utf-8")

        champs = {
            "methode": methode, "graine": graine,
            "n_cache": cfg.N_CANDIDATS_CACHES, "n_sortie": cfg.N_CANDIDATS_SORTIE,
            "tol_cache": cfg.TOL_RATIO_CACHE, "tol_sortie": cfg.TOL_RATIO_SORTIE,
            "variante": variante, "duree_episode": cfg.DUREE_EPISODE,
        }
        meta = {**champs, "epochs": cfg.EPOCHS, "run": run}

        print(f"\n=== {run} ===")
        debut = time.perf_counter()
        executeur = creer_executeur(methode, graine)
        strategie = StrategieAleatoire(graine) if methode == "aleatoire" else None
        journal = JournalCSV(chemin_csv, champs)
        try:
            entrainer(
                ReseauQEWO([3 * cfg.N_SEGMENTS, 32, 32, cfg.N_SEGMENTS]), 0, executeur,
                journal=journal, verbose=verbose, strategie=strategie,
                chemin_modele=chemin_modele, meta=meta,
            )
        finally:
            journal.fermer()
        print(f"=== {run} terminé en {formater_duree(time.perf_counter() - debut)} ===")
        return "termine"
    finally:
        for nom, valeur in ancienne.items():
            setattr(cfg, nom, valeur)


def main(argv=None):
    p = argparse.ArgumentParser(description="Lance un run QEWO ou une baseline.")
    p.add_argument("--methode", required=True, choices=METHODES)
    p.add_argument("--graine", type=int, default=0)
    p.add_argument("--n-cache", type=int, default=None, help="candidats des couches cachées")
    p.add_argument("--tol-cache", type=float, default=None, help="tol_ratio des couches cachées")
    p.add_argument("--tol-sortie", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--variante", choices=VARIANTES, default="normale")
    p.add_argument("--duree-episode", type=float, default=None,
                   help="NE PAS utiliser pour la campagne (tests rapides seulement)")
    p.add_argument("--dossier", default=None, help="dossier des résultats")
    p.add_argument("--forcer", action="store_true", help="refaire un run déjà terminé")
    p.add_argument("--verbose", action="store_true", help="barre de progression par poids")
    a = p.parse_args(argv)

    from pathlib import Path
    executer_run(
        a.methode, a.graine, a.n_cache, a.tol_cache, a.tol_sortie, a.epochs,
        a.variante, a.duree_episode,
        Path(a.dossier) if a.dossier else None, a.forcer, a.verbose,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
