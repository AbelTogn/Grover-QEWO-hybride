"""
Campagne d'expériences : enchaîne les runs des séries S1 à S8.

    python version_hybride/campagne.py --liste                 # affiche le plan, sans rien lancer
    python version_hybride/campagne.py --serie S1              # une série
    python version_hybride/campagne.py --serie S1 --serie S2   # plusieurs séries
    python version_hybride/campagne.py --toutes                # S1 à S6 (obligatoires)
    python version_hybride/campagne.py --toutes --avec-optionnelles

Un run déjà terminé est sauté : on peut interrompre (Ctrl+C) et relancer la
même commande. Pour paralléliser, lancez des séries différentes dans des
terminaux différents, avec OMP_NUM_THREADS=1.
"""
import argparse
import sys
import time
from pathlib import Path

import config as cfg
from experiences import executer_run, identifiant
from qewo import formater_duree

MINUTES_PAR_EPOCH = 4.1  # mesuré par l'auteur de ce code ; à recalibrer sur votre machine

SERIES = {
    "S1": ("QEWO, Grover émulé, configuration par défaut",
           [dict(methode="emulation", graine=g) for g in range(5)]),
    "S2": ("QEWO, sélection argmin (classique)",
           [dict(methode="argmin", graine=g) for g in range(5)]),
    "S3": ("Baseline : recherche aléatoire, même budget",
           [dict(methode="aleatoire", graine=g) for g in range(5)]),
    "S4": ("QEWO émulé, 19 candidats par couche cachée",
           [dict(methode="emulation", graine=g, n_cache=19) for g in range(3)]),
    "S5": ("QEWO émulé, tolérances ×0 puis ×4 (tol_sortie = 2 × tol_cache, rapport du papier)",
           [dict(methode="emulation", graine=g, tol_cache=t, tol_sortie=2 * t)
            for t in (0.0, 0.2) for g in range(3)]),
    "S6": ("QEWO sur le simulateur Q# du QDK (une graine)",
           [dict(methode="grover", graine=0)]),
    "S7": ("[optionnel] argmin sur 30 epochs",
           [dict(methode="argmin", graine=0, epochs=30)]),
    "S8": ("[optionnel] variante sans moteur à la base",
           [dict(methode=m, graine=g, variante="sans_moteur_base")
            for m in ("emulation", "argmin", "aleatoire") for g in range(3)]),
}
OBLIGATOIRES = ["S1", "S2", "S3", "S4", "S5", "S6"]
OPTIONNELLES = ["S7", "S8"]


def construire_plan(noms, epochs=None, duree_episode=None):
    plan = []
    for nom in noms:
        for spec in SERIES[nom][1]:
            spec = dict(spec)
            if epochs is not None:
                spec.setdefault("epochs", epochs)
            if duree_episode is not None:
                spec["duree_episode"] = duree_episode
            plan.append((nom, spec))
    return plan


def duree_estimee(spec, epochs_defaut):
    return spec.get("epochs", epochs_defaut) * MINUTES_PAR_EPOCH * 60


def nom_run(spec, epochs_defaut):
    return identifiant(
        spec["methode"], spec.get("graine", 0),
        spec.get("n_cache", cfg.N_CANDIDATS_CACHES),
        spec.get("tol_cache", cfg.TOL_RATIO_CACHE),
        spec.get("epochs", epochs_defaut),
        spec.get("variante", "normale"),
        spec.get("duree_episode", cfg.DUREE_EPISODE),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="Campagne d'expériences QEWO.")
    p.add_argument("--serie", action="append", choices=sorted(SERIES), default=[])
    p.add_argument("--toutes", action="store_true", help="S1 à S6")
    p.add_argument("--avec-optionnelles", action="store_true", help="ajoute S7 et S8")
    p.add_argument("--liste", action="store_true", help="affiche le plan et quitte")
    p.add_argument("--forcer", action="store_true", help="refaire les runs déjà terminés")
    p.add_argument("--epochs", type=int, default=None, help="remplace le nombre d'epochs")
    p.add_argument("--duree-episode", type=float, default=None,
                   help="NE PAS utiliser pour la vraie campagne (tests rapides seulement)")
    p.add_argument("--dossier", default=None)
    a = p.parse_args(argv)

    noms = list(a.serie)
    if a.toutes:
        noms += [n for n in OBLIGATOIRES if n not in noms]
    if a.avec_optionnelles:
        noms += [n for n in OPTIONNELLES if n not in noms]
    if not noms:
        p.error("choisissez --serie Sx, --toutes ou --liste avec une série")
    noms = sorted(set(noms))

    plan = construire_plan(noms, a.epochs, a.duree_episode)
    dossier = Path(a.dossier) if a.dossier else cfg.DOSSIER_RESULTATS
    epochs_defaut = a.epochs if a.epochs is not None else cfg.EPOCHS

    total = sum(duree_estimee(spec, epochs_defaut) for _, spec in plan)
    print(f"{len(plan)} runs, durée estimée ≈ {formater_duree(total)} "
          f"({MINUTES_PAR_EPOCH} min/epoch, à recalibrer)")
    if a.liste:
        for nom in noms:
            print(f"\n{nom} — {SERIES[nom][0]}")
            for s, spec in plan:
                if s == nom:
                    print(f"   {nom_run(spec, epochs_defaut)}")
        return 0

    echecs = []
    debut = time.perf_counter()
    for k, (nom, spec) in enumerate(plan, 1):
        print(f"\n##### [{k}/{len(plan)}] série {nom} #####")
        try:
            executer_run(dossier=dossier, forcer=a.forcer, **spec)
        except KeyboardInterrupt:
            print("\nInterrompu : relancez la même commande pour reprendre.")
            return 130
        except Exception as erreur:  # un run raté ne doit pas arrêter la nuit
            print(f"ÉCHEC {nom_run(spec, epochs_defaut)} : {erreur!r}")
            echecs.append((nom_run(spec, epochs_defaut), repr(erreur)))

    print(f"\nCampagne terminée en {formater_duree(time.perf_counter() - debut)}.")
    if echecs:
        print(f"{len(echecs)} run(s) en échec :")
        for run, erreur in echecs:
            print(f"  - {run} : {erreur}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
