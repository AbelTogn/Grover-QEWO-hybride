"""
Évaluation finale : tous les modèles de resultats/modeles + le PD, sur des
graines jamais utilisées pour entraîner ni valider (GRAINES_TEST_FINAL).

    python version_hybride/evaluation_finale.py

Écrit resultats/test_final.csv, une ligne par modèle (et une par PD) :
perte moyenne / médiane / écart-type sur les épisodes de test, et taux de
réussite (|erreur| < SEUIL_REUSSITE rad sur tous les segments pendant la
dernière FENETRE_REUSSITE seconde).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import config as cfg
from pendule import construire_etat, controle_pd, simuler_details
from reseau import charger_modele, lire_meta

COLONNES = [
    "modele", "methode", "graine", "n_cache", "tol_cache", "variante", "epochs",
    "n_episodes", "perte_moyenne", "perte_mediane", "ecart_type", "taux_reussite",
]


def evaluer_reseau_details(nn, graines, duree=None):
    """Retourne (couts, succes), de forme (len(graines),) chacun."""
    def controle(theta, omega):
        return cfg.COUPLE_MAX * nn.forward(construire_etat(theta, omega))

    cout, succes = simuler_details(controle, 1, graines, duree)
    return cout[0], succes[0]


def evaluer_pd_details(graines, duree=None):
    cout, succes = simuler_details(controle_pd(), 1, graines, duree)
    return cout[0], succes[0]


def _ligne(nom, methode, graine, n_cache, tol_cache, variante, epochs, cout, succes):
    return {
        "modele": nom, "methode": methode, "graine": graine, "n_cache": n_cache,
        "tol_cache": tol_cache, "variante": variante, "epochs": epochs,
        "n_episodes": len(cout), "perte_moyenne": float(np.mean(cout)),
        "perte_mediane": float(np.median(cout)), "ecart_type": float(np.std(cout)),
        "taux_reussite": float(np.mean(succes)),
    }


def evaluer_dossier(dossier, graines=None, duree=None):
    """Évalue tous les modèles de dossier/modeles ; retourne la liste des lignes."""
    graines = list(cfg.GRAINES_TEST_FINAL if graines is None else graines)
    modeles = sorted((Path(dossier) / "modeles").glob("*.npz"))
    if not modeles:
        raise FileNotFoundError(f"Aucun modèle dans {Path(dossier) / 'modeles'}")

    ancien = cfg.SANS_MOTEUR_BASE
    lignes, variantes = [], {"normale"}
    try:
        for chemin in modeles:
            meta = lire_meta(chemin)
            if not meta:
                print(f"[ignoré] {chemin.name} : pas de métadonnées")
                continue
            variante = meta.get("variante", "normale")
            variantes.add(variante)
            cfg.SANS_MOTEUR_BASE = variante == "sans_moteur_base"
            nn, epochs = charger_modele(chemin)
            cout, succes = evaluer_reseau_details(nn, graines, duree)
            lignes.append(_ligne(
                meta.get("run", chemin.stem), meta["methode"], meta["graine"],
                meta["n_cache"], meta["tol_cache"], variante, epochs, cout, succes,
            ))
        for variante in sorted(variantes):  
            cfg.SANS_MOTEUR_BASE = variante == "sans_moteur_base"
            cout, succes = evaluer_pd_details(graines, duree)
            lignes.append(_ligne(f"pd_{variante}", "pd", 0, 0, 0.0, variante, 0, cout, succes))
    finally:
        cfg.SANS_MOTEUR_BASE = ancien
    return lignes


def ecrire_csv(lignes, chemin):
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", newline="", encoding="utf-8") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=COLONNES)
        ecrivain.writeheader()
        ecrivain.writerows(lignes)


def lire_csv(chemin):
    entiers = {"graine", "n_cache", "epochs", "n_episodes"}
    textes = {"modele", "methode", "variante"}
    lignes = []
    with open(chemin, newline="", encoding="utf-8") as fichier:
        for brut in csv.DictReader(fichier):
            lignes.append({
                c: (v if c in textes else int(float(v)) if c in entiers else float(v))
                for c, v in brut.items()
            })
    return lignes


def main(argv=None):
    p = argparse.ArgumentParser(description="Évaluation finale sur graines jamais vues.")
    p.add_argument("--dossier", default=None, help="dossier des résultats")
    p.add_argument("--duree-episode", type=float, default=None,
                   help="NE PAS utiliser pour les vrais résultats")
    a = p.parse_args(argv)

    dossier = Path(a.dossier) if a.dossier else cfg.DOSSIER_RESULTATS
    lignes = evaluer_dossier(dossier, duree=a.duree_episode)
    sortie = dossier / "test_final.csv"
    ecrire_csv(lignes, sortie)

    print(f"{len(cfg.GRAINES_TEST_FINAL)} épisodes de test par modèle "
          f"(graines {cfg.GRAINES_TEST_FINAL[0]} à {cfg.GRAINES_TEST_FINAL[-1]})\n")
    print(f"{'modèle':<42}{'perte moy.':>12}{'médiane':>10}{'réussite':>10}")
    for l in lignes:
        print(f"{l['modele']:<42}{l['perte_moyenne']:>12.4f}{l['perte_mediane']:>10.4f}"
              f"{100 * l['taux_reussite']:>9.0f} %")
    print(f"\nÉcrit : {sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
