"""
Analyse des résultats : figures et tableaux pour le TIPE.

    python version_hybride/analyse.py            # lit resultats/, écrit resultats/figures/
    python version_hybride/analyse.py --tout     # courbes de TOUS les groupes (pas seulement la config par défaut)

Entrées :  resultats/courbes/*.csv (obligatoire), resultats/test_final.csv (facultatif,
           produit par evaluation_finale.py)
Sorties :  resultats/figures/*.png, tableau_racine_N.{csv,md}, resume_runs.{csv,md}
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config as cfg  # noqa: E402
import evaluation_finale as ef  # noqa: E402
import quantique as q  # noqa: E402
from journal import lire_csv  # noqa: E402

LIBELLES = {
    "emulation": "QEWO (Grover émulé)",
    "grover": "QEWO (Grover, QDK)",
    "argmin": "QEWO-argmin (classique)",
    "aleatoire": "Recherche aléatoire",
}
COULEURS = {
    "emulation": "#1f77b4", "grover": "#17becf", "argmin": "#d62728", "aleatoire": "#2ca02c",
}
MARQUEURS = {"emulation": "o", "grover": "D", "argmin": "s", "aleatoire": "^"}
ORDRE = list(LIBELLES)
# couleurs des groupes supplémentaires d'une même méthode (ablations : N, tol…)
COULEURS_EXTRA = ["#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]


def couleur_groupe(methode, rang):
    """Couleur de la méthode pour son 1er groupe, couleur supplémentaire ensuite."""
    if rang == 0:
        return COULEURS.get(methode, "gray")
    return COULEURS_EXTRA[(rang - 1) % len(COULEURS_EXTRA)]


# ------------------------------------------------------------------
# Chargement et regroupement
# ------------------------------------------------------------------

def charger_runs(dossier):
    """Un dict par run terminé : nom, lignes, clé de groupe."""
    runs = []
    for chemin in sorted((Path(dossier) / "courbes").glob("*.csv")):
        try:
            lignes = lire_csv(chemin)
        except (ValueError, KeyError) as erreur:
            print(f"[ignoré] {chemin.name} : {erreur!r}")
            continue
        if len(lignes) < 2:
            continue
        prevu = re.search(r"_e(\d+)", chemin.stem)
        if prevu and lignes[-1]["epoch"] < int(prevu.group(1)):
            print(f"[incomplet] {chemin.name} : {lignes[-1]['epoch']}/{prevu.group(1)} epochs, ignoré")
            continue
        l0 = lignes[0]
        cle = (l0["methode"], l0["n_cache"], l0["tol_cache"], l0["variante"],
               l0["duree_episode"], lignes[-1]["epoch"])
        runs.append({"nom": chemin.stem, "lignes": lignes, "cle": cle, "graine": l0["graine"]})
    return runs


def grouper(runs):
    groupes = defaultdict(list)
    for run in runs:
        groupes[run["cle"]].append(run)
    return dict(sorted(groupes.items(), key=lambda kv: (ORDRE.index(kv[0][0])
                                                        if kv[0][0] in ORDRE else 99, kv[0][1:])))


def est_defaut(cle):
    _, n, tol, _, duree, epochs = cle
    return (n == cfg.N_CANDIDATS_CACHES and abs(tol - cfg.TOL_RATIO_CACHE) < 1e-12
            and duree == cfg.DUREE_EPISODE and epochs == cfg.EPOCHS)


def libelle(cle):
    methode, n, tol, variante, duree, epochs = cle
    details = []
    if n != cfg.N_CANDIDATS_CACHES:
        details.append(f"N={n}")
    if abs(tol - cfg.TOL_RATIO_CACHE) > 1e-12:
        details.append(f"tol={tol:g}")
    if variante != "normale":
        details.append("sans moteur base")
    if epochs != cfg.EPOCHS:
        details.append(f"{epochs} epochs")
    if duree != cfg.DUREE_EPISODE:
        details.append(f"épisodes {duree:g} s")
    base = LIBELLES.get(methode, methode)
    return base + (" — " + ", ".join(details) if details else "")


def ecart_type(valeurs):
    return float(np.std(valeurs, ddof=1)) if len(valeurs) > 1 else 0.0


def perte_pd(runs):
    valeurs = [r["lignes"][0]["perte_pd_val"] for r in runs]
    return float(np.median(valeurs)) if valeurs else None


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def figure_courbes(groupes, fichier, titre, pd_val):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    styles = ["-", "--", ":", "-."]
    vus = defaultdict(int)
    for cle, runs in groupes:
        n_lignes = min(len(r["lignes"]) for r in runs)
        X = np.array([[l["evaluations"] for l in r["lignes"][:n_lignes]] for r in runs])
        Y = np.array([[l["perte_val"] for l in r["lignes"][:n_lignes]] for r in runs])
        x, m = X.mean(axis=0), Y.mean(axis=0)
        s = Y.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(m)
        couleur = couleur_groupe(cle[0], vus[cle[0]])
        style = styles[vus[cle[0]] % len(styles)]
        vus[cle[0]] += 1
        for y in Y:
            ax.plot(x, y, color=couleur, alpha=0.15, lw=0.8)
        ax.plot(x, m, color=couleur, ls=style, marker=MARQUEURS.get(cle[0], "o"), ms=4,
                label=f"{libelle(cle)} (n={len(runs)})")
        ax.fill_between(x, np.maximum(m - s, 1e-9), m + s, color=couleur, alpha=0.18)
    if pd_val:
        ax.axhline(pd_val, color="k", ls="--", lw=1, label="PD (référence)")
    ax.set_yscale("log")
    ax.set_xlabel("Évaluations de la perte (cumulées)")
    ax.set_ylabel("Perte de validation (échelle log)")
    ax.set_title(titre)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fichier, dpi=200)
    plt.close(fig)


def figure_boites(donnees, fichier, titre, ylabel, pd_val=None, log=True):
    """donnees : liste de (libellé, valeurs)."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.9 * len(donnees)), 5.2))
    ax.boxplot([v for _, v in donnees], showfliers=False)
    ax.set_xticks(range(1, len(donnees) + 1))
    ax.set_xticklabels([l.replace(" — ", "\n") for l, _ in donnees], fontsize=8)
    for i, (_, valeurs) in enumerate(donnees, 1):
        ax.scatter(i + rng.uniform(-0.09, 0.09, len(valeurs)), valeurs, s=18, alpha=0.8,
                   color="#444", zorder=3)
    if pd_val:
        ax.axhline(pd_val, color="k", ls="--", lw=1, label="PD (référence)")
        ax.legend(fontsize=8)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(titre)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(fichier, dpi=200)
    plt.close(fig)


def figure_ablation(groupes, parametre, fichier, pd_val):
    """Perte de validation finale en fonction de N (n_cache) ou de tol_cache."""
    position = {"n_cache": 1, "tol_cache": 2}[parametre]
    par_methode = defaultdict(lambda: defaultdict(list))
    for cle, runs in groupes.items():
        methode, n, tol, variante, duree, epochs = cle
        if variante != "normale" or duree != cfg.DUREE_EPISODE or epochs != cfg.EPOCHS:
            continue
        autre_defaut = (abs(tol - cfg.TOL_RATIO_CACHE) < 1e-12 if parametre == "n_cache"
                        else n == cfg.N_CANDIDATS_CACHES)
        if autre_defaut:
            par_methode[methode][cle[position]] += [r["lignes"][-1]["perte_val"] for r in runs]
    par_methode = {m: v for m, v in par_methode.items() if len(v) >= 2}
    if not par_methode:
        print(f"[ablation {parametre}] pas assez de données (il faut au moins 2 valeurs), ignorée")
        return False

    valeurs = sorted({v for d in par_methode.values() for v in d})
    fig, ax = plt.subplots(figsize=(7, 5))
    decalage = {m: (i - (len(par_methode) - 1) / 2) * 0.12 for i, m in enumerate(par_methode)}
    for methode, donnees in par_methode.items():
        xs, ms, ss = [], [], []
        for v, pertes in donnees.items():
            x = valeurs.index(v) + decalage[methode]
            ax.scatter(np.full(len(pertes), x), pertes, s=14, alpha=0.5,
                       color=COULEURS.get(methode, "gray"))
            xs.append(x), ms.append(np.mean(pertes)), ss.append(ecart_type(pertes))
        ax.errorbar(xs, ms, yerr=ss, fmt=MARQUEURS.get(methode, "o"), capsize=4,
                    color=COULEURS.get(methode, "gray"), label=LIBELLES.get(methode, methode))
    if pd_val:
        ax.axhline(pd_val, color="k", ls="--", lw=1, label="PD (référence)")
    ax.set_xticks(range(len(valeurs)))
    ax.set_xticklabels([f"{v:g}" for v in valeurs])
    ax.set_xlabel("Candidats par couche cachée (N)" if parametre == "n_cache"
                  else "tol_ratio des couches cachées")
    ax.set_ylabel("Perte de validation finale (log)")
    ax.set_yscale("log")
    ax.set_title("Ablation : " + ("nombre de candidats" if parametre == "n_cache" else "tolérance"))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fichier, dpi=200)
    plt.close(fig)
    return True


def figure_diagnostic_M(groupes, fichier):
    """Taille de M et part des recherches sans itération, par epoch (méthodes à circuit)."""
    cibles = [(c, r) for c, r in groupes.items() if c[0] in ("emulation", "grover")]
    if not cibles:
        return False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    vus = defaultdict(int)
    for cle, runs in cibles:
        c = couleur_groupe(cle[0], vus[cle[0]])
        vus[cle[0]] += 1
        n_lignes = min(len(r["lignes"]) for r in runs)
        ep = np.array([l["epoch"] for l in runs[0]["lignes"][1:n_lignes]])
        taille = np.array([[l["taille_M_moy"] for l in r["lignes"][1:n_lignes]] for r in runs])
        sans = np.array([[100 * l["part_sans_iteration"] for l in r["lignes"][1:n_lignes]]
                         for r in runs])
        ax1.plot(ep, taille.mean(axis=0), marker="o", ms=4, color=c, label=libelle(cle))
        ax2.plot(ep, sans.mean(axis=0), marker="o", ms=4, color=c, label=libelle(cle))
    ax1.set_ylabel("Taille moyenne de M\n(toutes couches)")
    ax2.set_ylabel("Recherches sans itération (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(-3, 103)
    ax1.set_title("Amplification de Grover : M est-il petit devant N ?")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fichier, dpi=200)
    plt.close(fig)
    return True


# ------------------------------------------------------------------
# Tableaux
# ------------------------------------------------------------------

def ecrire_tableau(entetes, lignes, base, intro=""):
    """Écrit base.csv et base.md ; retourne le texte markdown."""
    with open(f"{base}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(entetes)
        w.writerows(lignes)
    md = intro + "\n\n" if intro else ""
    md += "| " + " | ".join(entetes) + " |\n|" + "---|" * len(entetes) + "\n"
    for ligne in lignes:
        md += "| " + " | ".join(str(c) for c in ligne) + " |\n"
    Path(f"{base}.md").write_text(md, encoding="utf-8")
    return md


def tableau_racine_n(dossier_figures):
    lignes = []
    for n in (8, 16, 19, 24, 32, 48, 64, 128, 256, 1024, 4096):
        nb = q.nb_qubits_registre(n)
        k1 = q.iterations_optimales(1, nb)
        k3 = q.iterations_optimales(3, nb)
        lignes.append([
            n, nb, nb + 1, k1, f"{(n + 1) / 2:.1f}", f"{(n + 1) / 2 / k1:.1f}" if k1 else "-",
            k3, f"{(n + 1) / 4:.1f}", f"{(n + 1) / 4 / k3:.1f}" if k3 else "-",
        ])
    entetes = ["N", "qubits d'indice", "qubits avec ancilla", "requêtes Grover (M=1)",
               "essais classiques moyens (M=1)", "rapport (M=1)", "requêtes Grover (M=3)",
               "essais classiques moyens (M=3)", "rapport (M=3)"]
    intro = (
        "Requêtes à l'oracle de Grover (itérations optimales) contre nombre moyen d'essais "
        "d'une recherche classique, pour trouver un élément marqué parmi N.\n\n"
        "**Attention** : dans QEWO tel que décrit dans le papier, les N pertes sont évaluées "
        "classiquement (étape 3) AVANT le circuit. Ces rapports portent donc sur la seule "
        "étape 5 et ne sont pas un gain sur l'entraînement complet."
    )
    ecrire_tableau(entetes, lignes, str(Path(dossier_figures) / "tableau_racine_N"), intro)
    return entetes, lignes


def poids_par_epoch():
    n = cfg.N_SEGMENTS
    return 128 * n + 1024  # (32 x 3n) + (32 x 32) + (n x 32)


def resume_runs(groupes, test, dossier_figures):
    entetes = ["groupe", "runs", "epochs", "évaluations", "requêtes oracle", "requêtes/poids",
               "|M| moyen", "sans itération (%)", "mesures hors M", "poids non modifiés",
               "temps (min)", "val. finale (moy.)", "val. finale (écart-type)",
               "test final (moy.)", "taux de réussite (%)"]
    lignes = []
    for cle, runs in groupes.items():
        methode, n, tol, variante, duree, epochs = cle
        derniere = [r["lignes"][-1] for r in runs]
        val = [l["perte_val"] for l in derniere]
        circuit = methode in ("emulation", "grover")
        lignes_test = [t for t in test if t["methode"] == methode and t["n_cache"] == n
                       and abs(t["tol_cache"] - tol) < 1e-12 and t["variante"] == variante
                       and t["epochs"] == epochs]
        moy = lambda cle_: float(np.mean([l[cle_] for l in derniere]))  # noqa: E731
        lignes.append([
            libelle(cle), len(runs), epochs, f"{moy('evaluations'):.0f}",
            f"{moy('requetes_oracle'):.0f}" if circuit else "-",
            f"{moy('requetes_oracle') / (poids_par_epoch() * epochs):.2f}" if circuit else "-",
            f"{np.mean([l['taille_M_moy'] for r in runs for l in r['lignes'][1:]]):.1f}" if circuit else "-",
            f"{100 * np.mean([l['part_sans_iteration'] for r in runs for l in r['lignes'][1:]]):.0f}" if circuit else "-",
            f"{moy('mesures_hors_M'):.0f}" if circuit else "-",
            f"{moy('poids_sans_maj'):.0f}" if circuit else "-",
            f"{moy('temps_s') / 60:.1f}", f"{np.mean(val):.4f}", f"{ecart_type(val):.4f}",
            f"{np.mean([t['perte_moyenne'] for t in lignes_test]):.4f}" if lignes_test else "-",
            f"{100 * np.mean([t['taux_reussite'] for t in lignes_test]):.0f}" if lignes_test else "-",
        ])
    ecrire_tableau(entetes, lignes, str(Path(dossier_figures) / "resume_runs"))
    return entetes, lignes


# ------------------------------------------------------------------
# Programme principal
# ------------------------------------------------------------------

def analyser(dossier, tout=False):
    dossier = Path(dossier)
    figures = dossier / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    runs = charger_runs(dossier)
    if not runs:
        print(f"Aucun run terminé dans {dossier / 'courbes'}.")
        return []
    groupes = grouper(runs)
    produits = []

    chemin_test = dossier / "test_final.csv"
    test = ef.lire_csv(chemin_test) if chemin_test.is_file() else []
    if not test:
        print("[test final] test_final.csv absent : lancez evaluation_finale.py pour les figures de test.")

    variantes = sorted({c[3] for c in groupes})
    for variante in variantes:
        suffixe = "" if variante == "normale" else f"_{variante}"
        choisis = [(c, r) for c, r in groupes.items()
                   if c[3] == variante and (tout or est_defaut(c))]
        if not choisis:
            continue
        pd_val = perte_pd([r for _, rs in choisis for r in rs])
        titre = "Perte de validation" + ("" if variante == "normale" else " (sans moteur à la base)")

        f = figures / f"perte_val_vs_evaluations{suffixe}.png"
        figure_courbes(choisis, f, titre, pd_val)
        produits.append(f)

        f = figures / f"boites_perte_validation{suffixe}.png"
        figure_boites([(libelle(c), [r["lignes"][-1]["perte_val"] for r in rs]) for c, rs in choisis],
                      f, titre + " finale (une valeur par graine)", "Perte de validation finale (log)",
                      pd_val)
        produits.append(f)

        # test final : une valeur (perte moyenne sur les épisodes de test) par modèle
        donnees_test, succes = [], []
        for c, _ in choisis:
            lignes_t = [t for t in test if t["methode"] == c[0] and t["n_cache"] == c[1]
                        and abs(t["tol_cache"] - c[2]) < 1e-12 and t["variante"] == variante
                        and t["epochs"] == c[5]]
            if lignes_t:
                donnees_test.append((libelle(c), [t["perte_moyenne"] for t in lignes_t]))
                succes.append((libelle(c), [100 * t["taux_reussite"] for t in lignes_t]))
        ref_pd = [t for t in test if t["methode"] == "pd" and t["variante"] == variante]
        if donnees_test:
            f = figures / f"boites_perte_test{suffixe}.png"
            figure_boites(donnees_test, f, "Perte sur graines de test (jamais vues)",
                          "Perte de test (log)",
                          ref_pd[0]["perte_moyenne"] if ref_pd else None)
            produits.append(f)
            f = figures / f"taux_reussite{suffixe}.png"
            figure_boites(succes, f, "Taux de réussite sur graines de test",
                          "Épisodes réussis (%)", None, log=False)
            produits.append(f)

    pd_normal = perte_pd([r for r in runs if r["cle"][3] == "normale"])
    for parametre in ("n_cache", "tol_cache"):
        f = figures / f"ablation_{'N' if parametre == 'n_cache' else 'tol'}.png"
        if figure_ablation(groupes, parametre, f, pd_normal):
            produits.append(f)

    f = figures / "diagnostic_taille_M.png"
    if figure_diagnostic_M(groupes, f):
        produits.append(f)

    entetes, lignes = tableau_racine_n(figures)
    produits.append(figures / "tableau_racine_N.md")
    entetes_r, lignes_r = resume_runs(groupes, test, figures)
    produits.append(figures / "resume_runs.md")

    print(f"\n{len(runs)} runs, {len(groupes)} groupes.\n")
    for l in lignes_r:
        print(f"- {l[0]}: {l[1]} run(s), validation finale {l[11]} ± {l[12]}"
              + (f", test {l[13]} ({l[14]} % de réussite)" if l[13] != "-" else ""))
    print("\nFichiers écrits :")
    for p in produits:
        print(f"  {p}")
    return produits


def main(argv=None):
    p = argparse.ArgumentParser(description="Figures et tableaux pour le TIPE.")
    p.add_argument("--dossier", default=None, help="dossier des résultats")
    p.add_argument("--tout", action="store_true",
                   help="courbes de tous les groupes, pas seulement la configuration par défaut")
    a = p.parse_args(argv)
    dossier = Path(a.dossier) if a.dossier else cfg.DOSSIER_RESULTATS
    return 0 if analyser(dossier, a.tout) else 1


if __name__ == "__main__":
    sys.exit(main())
