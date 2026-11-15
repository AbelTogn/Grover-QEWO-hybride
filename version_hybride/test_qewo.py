"""
Tests : python version_hybride/test_qewo.py
(assertions simples, aucune dépendance au-delà de numpy ; le QDK n'est pas requis)
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import qewo
import quantique as q
import reseau
from pendule import Pendule


def test_physique_coriolis_et_masse():
    """Coriolis analytique == Christoffel par différences finies de la matrice de masse."""
    p = Pendule(3)
    rng = np.random.default_rng(0)
    eps = 1e-6
    for _ in range(50):
        th = rng.uniform(-np.pi, np.pi, 3)
        om = rng.uniform(-8, 8, 3)
        dM = [
            (p.matrice_masse(th + eps * np.eye(3)[k]) - p.matrice_masse(th - eps * np.eye(3)[k]))
            / (2 * eps)
            for k in range(3)
        ]
        coriolis = np.zeros(3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    gamma = 0.5 * (dM[k][i, j] + dM[j][i, k] - dM[i][j, k])
                    coriolis[i] += gamma * om[j] * om[k]
        analytique = (p.coef_masse * np.sin(th[:, None] - th[None, :])) @ om**2
        assert np.allclose(coriolis, analytique, atol=1e-6)


def test_iterations_optimales():
    assert q.iterations_optimales(1, 5) == 4   # N=32, M=1
    assert q.iterations_optimales(1, 6) == 6   # N=64, M=1
    assert q.iterations_optimales(16, 5) == 0  # M/N = 1/2
    assert q.iterations_optimales(20, 5) == 0  # M > N/2 : première pointe en k=0
    assert q.iterations_optimales(12, 5) == 1  # règle standard (1re pointe), pas une pointe lointaine
    assert q.iterations_optimales(0, 5) == 0
    assert q.nb_qubits_registre(32) == 5 and q.nb_qubits_registre(64) == 6
    assert q.nb_qubits_registre(19) == 5       # cas de l'article : 19 candidats -> 5 qubits d'indice


def test_probabilite_grover_sur_M():
    """Le circuit amplifie bien M, y compris quand N n'est pas une puissance de 2."""
    rng = np.random.default_rng(1)
    for n_cand in (19, 24, 32, 40, 64):
        nb = q.nb_qubits_registre(n_cand)
        for m in (1, 3, 8):
            marques = rng.choice(n_cand, size=m, replace=False)
            p = q.probabilites_grover(nb, marques, q.iterations_optimales(m, nb))
            assert p[marques].sum() > 0.94, (n_cand, m, p[marques].sum())
            assert np.isclose(p.sum(), 1.0)


def test_indices_marques():
    pertes = np.array([1.0, 1.04, 1.06, 2.0])
    assert list(q.indices_marques(pertes, 0.05)) == [0, 1]
    assert list(q.indices_marques(pertes, 0.10)) == [0, 1, 2]


def test_rechercher_grover_et_reessais():
    stats = q.StatsQuantiques()
    appels = []

    def capricieux(nb, marques, it):  # 1re mesure hors de M (comme du bruit matériel)
        appels.append(1)
        return 31 if len(appels) == 1 else marques[0]

    assert q.rechercher_grover(32, [5, 9], capricieux, stats) == 5
    assert stats.mesures == 2 and stats.mesures_invalides == 1

    stats = q.StatsQuantiques()
    assert q.rechercher_grover(32, [5], lambda nb, m, it: 0, stats) is None
    assert stats.mesures == cfg.MAX_ESSAIS_MESURE and stats.poids_sans_mise_a_jour == 1


def test_choisir_candidat():
    pertes = np.array([3.0, 1.0, 1.02, 5.0] + [9.0] * 28)
    stats = q.StatsQuantiques()
    assert qewo.choisir_candidat(pertes, 0.05, None, stats) == 1  # argmin
    ex = q.creer_executeur_numpy(7)
    tirages = {qewo.choisir_candidat(pertes, 0.05, ex, stats) for _ in range(200)}
    assert tirages == {1, 2}  # uniquement des indices de M, les deux apparaissent


def test_evaluation_en_lot_coherente():
    """Un lot de k candidats == k évaluations isolées (k=1)."""
    nn = reseau.ReseauQEWO([9, 32, 32, 3], graine=5)
    cands = nn.poids[1][4, 7] + np.linspace(-0.3, 0.3, 6)
    lot = reseau.evaluer_candidats(nn, 1, 4, 7, cands, [10, 20, 30], 1.0)
    seul = [
        reseau.evaluer_candidats(nn, 1, 4, 7, [c], [10, 20, 30], 1.0)[0] for c in cands
    ]
    assert np.allclose(lot, seul, rtol=1e-8)


def test_sauvegarde_sans_pickle():
    nn = reseau.ReseauQEWO([9, 32, 32, 3], graine=3)
    ancien = cfg.FICHIER_MODELE
    with tempfile.TemporaryDirectory() as dossier:
        cfg.FICHIER_MODELE = Path(dossier) / "m.npz"
        reseau.sauvegarder_modele(nn, [0.1, 0.2, 0.3], 4, 1.5, 2.5)
        nn2, epochs = reseau.charger_modele()
        assert epochs == 4 and nn2.alpha_couches == [0.1, 0.2, 0.3]
        assert all(np.array_equal(a, b) for a, b in zip(nn.poids, nn2.poids))
        np.load(cfg.FICHIER_MODELE, allow_pickle=False).close()
    cfg.FICHIER_MODELE = ancien


def test_boucle_complete_emulation():
    """Une epoch complète avec le circuit émulé (épisodes raccourcis pour la vitesse)."""
    ancien = (cfg.FICHIER_MODELE, cfg.EPOCHS, cfg.DUREE_EPISODE)
    with tempfile.TemporaryDirectory() as dossier:
        cfg.FICHIER_MODELE = Path(dossier) / "m.npz"
        cfg.EPOCHS, cfg.DUREE_EPISODE = 1, 0.2
        nn = qewo.entrainer(None, 0, q.creer_executeur_numpy(0))
        w0 = reseau.ReseauQEWO([9, 32, 32, 3]).poids
        assert any(not np.array_equal(a, b) for a, b in zip(nn.poids, w0))  # les poids ont bougé
        assert all(np.all(np.isfinite(W)) for W in nn.poids)
        _, epochs = reseau.charger_modele()
        assert epochs == 1
    cfg.FICHIER_MODELE, cfg.EPOCHS, cfg.DUREE_EPISODE = ancien


# ------------------------------------------------------------------
# Expériences : stratégies, journal, lanceur, évaluation finale, analyse
# ------------------------------------------------------------------

def test_strategie_aleatoire():
    from baselines import StrategieAleatoire
    s = StrategieAleatoire(3)
    c = s.candidats(1.0, 0.5, 64)
    assert len(c) == 64 and c.min() >= 0.5 and c.max() <= 1.5
    assert s.accepter(1.0, 2.0) and not s.accepter(2.0, 1.0) and not s.accepter(1.0, 1.0)
    assert np.array_equal(StrategieAleatoire(3).candidats(0, 1, 5),
                          StrategieAleatoire(3).candidats(0, 1, 5))
    assert qewo.StrategieQEWO().accepter(9.0, 1.0)  # QEWO : mise à jour inconditionnelle


def test_requetes_oracle_et_diagnostic_M():
    ex = q.creer_executeur_numpy(0)
    stats = q.StatsQuantiques()
    q.rechercher_grover(32, [3], ex, stats)            # M=1 sur 32 : 4 itérations / mesure
    assert stats.requetes_oracle == 4 * stats.mesures
    assert stats.recherches == 1 and stats.somme_marques == 1
    assert stats.recherches_sans_iteration == 0
    stats = q.StatsQuantiques()
    q.rechercher_grover(32, list(range(20)), ex, stats)  # |M| >= moitié : aucune itération
    assert stats.recherches_sans_iteration == 1 and stats.requetes_oracle == 0


def test_journal_csv():
    from journal import COLONNES, JournalCSV, lire_csv, run_termine
    fixes = dict(methode="argmin", graine=1, n_cache=32, n_sortie=64, tol_cache=0.05,
                 tol_sortie=0.1, variante="normale", duree_episode=5.0)
    ligne = dict(epoch=0, evaluations=0, requetes_oracle=0, mesures=0, mesures_hors_M=0,
                 poids_sans_maj=0, taille_M_moy=0.0, part_sans_iteration=0.0,
                 perte_train=1.5, perte_val=2.5, perte_pd_val=0.01, temps_s=0.0)
    with tempfile.TemporaryDirectory() as d:
        chemin = Path(d) / "sous" / "x.csv"
        j = JournalCSV(chemin, fixes)
        j(ligne)
        assert not run_termine(chemin, 2)
        j({**ligne, "epoch": 2, "perte_val": 0.5})
        try:
            j({k: v for k, v in ligne.items() if k != "perte_val"})
            raise AssertionError("une colonne manquante doit lever KeyError")
        except KeyError:
            pass
        j.fermer()
        lignes = lire_csv(chemin)
        assert len(lignes) == 2 and lignes[1]["epoch"] == 2 and lignes[1]["perte_val"] == 0.5
        assert isinstance(lignes[0]["graine"], int) and lignes[0]["methode"] == "argmin"
        assert run_termine(chemin, 2) and set(lignes[0]) == set(COLONNES)


def test_succes_et_variante_sans_moteur():
    from pendule import appliquer_actionneurs, controle_pd, simuler_details
    val = list(cfg.GRAINES_VALIDATION)
    cout, succes = simuler_details(controle_pd(), 1, val)
    assert succes.all() and abs(cout.mean() - 0.0065) < 1e-3
    _, sans = simuler_details(lambda th, om: np.zeros_like(th), 1, val)
    assert not sans.any()                                  # sans commande : le pendule tombe
    cfg.SANS_MOTEUR_BASE = True
    try:
        c = appliquer_actionneurs(np.full((2, 3), 100.0))
        assert np.all(c[:, 0] == 0) and np.all(c[:, 1:] == cfg.COUPLE_MAX)
        _, s = simuler_details(controle_pd(), 1, val)
        assert not s.any()                                 # le PD n'y réussit plus
    finally:
        cfg.SANS_MOTEUR_BASE = False


_RAPIDE = {}


def _jeu_de_donnees_rapide():
    """8 runs minuscules (1 epoch, épisodes de 0,1 s), construits une seule fois."""
    if "dossier" in _RAPIDE:
        return _RAPIDE["dossier"]
    import experiences
    _RAPIDE["tmp"] = tempfile.TemporaryDirectory()
    dossier = Path(_RAPIDE["tmp"].name)
    kw = dict(epochs=1, duree_episode=0.1, dossier=dossier)
    for graine in (0, 1):
        for methode in ("emulation", "argmin", "aleatoire"):
            assert experiences.executer_run(methode, graine, **kw) == "termine"
        experiences.executer_run("emulation", graine, n_cache=19, **kw)
    _RAPIDE["dossier"] = dossier
    return dossier


def test_experiences_lanceur():
    import experiences
    from journal import lire_csv
    from reseau import lire_meta
    dossier = _jeu_de_donnees_rapide()
    chemin = dossier / "courbes" / "emulation_s0_n32_t0.05_e1_d0.1.csv"
    lignes = lire_csv(chemin)
    assert [l["epoch"] for l in lignes] == [0, 1]
    assert lignes[1]["evaluations"] == 48128 and lignes[1]["requetes_oracle"] > 0
    assert lignes[0]["methode"] == "emulation" and lignes[1]["perte_pd_val"] > 0
    arg = lire_csv(dossier / "courbes" / "argmin_s0_n32_t0.05_e1_d0.1.csv")
    assert arg[1]["requetes_oracle"] == 0 and arg[1]["evaluations"] == 48128
    n19 = lire_csv(dossier / "courbes" / "emulation_s0_n19_t0.05_e1_d0.1.csv")
    assert n19[1]["evaluations"] == 19 * 1312 + 64 * 96   # 19 candidats cachés, 64 en sortie
    meta = lire_meta(dossier / "modeles" / "aleatoire_s1_n32_t0.05_e1_d0.1.npz")
    assert meta["methode"] == "aleatoire" and meta["graine"] == 1
    # configuration restaurée, run déjà fait sauté
    assert cfg.N_CANDIDATS_CACHES == 32 and cfg.EPOCHS == 10 and cfg.DUREE_EPISODE == 5.0
    assert experiences.executer_run("argmin", 0, epochs=1, duree_episode=0.1,
                                    dossier=dossier) == "deja_fait"
    assert (dossier / ".gitignore").read_text().strip() == "modeles/"


def test_evaluation_finale():
    import evaluation_finale as ef
    dossier = _jeu_de_donnees_rapide()
    lignes = ef.evaluer_dossier(dossier, duree=0.1)
    assert len(lignes) == 8 + 1                            # 8 modèles + PD
    pd = [l for l in lignes if l["methode"] == "pd"][0]
    assert pd["n_episodes"] == 50 and pd["perte_moyenne"] > 0
    ef.ecrire_csv(lignes, dossier / "test_final.csv")
    relues = ef.lire_csv(dossier / "test_final.csv")
    assert len(relues) == 9 and set(relues[0]) == set(ef.COLONNES)
    assert all(0.0 <= l["taux_reussite"] <= 1.0 for l in relues)


def test_analyse_produit_figures_et_tableaux():
    import analyse
    import evaluation_finale as ef
    dossier = _jeu_de_donnees_rapide()
    ef.ecrire_csv(ef.evaluer_dossier(dossier, duree=0.1), dossier / "test_final.csv")
    ancien = (cfg.EPOCHS, cfg.DUREE_EPISODE)
    cfg.EPOCHS, cfg.DUREE_EPISODE = 1, 0.1   # pour que ces runs comptent comme « par défaut »
    try:
        produits = analyse.analyser(dossier)
    finally:
        cfg.EPOCHS, cfg.DUREE_EPISODE = ancien
    noms = {p.name for p in produits}
    for attendu in ("perte_val_vs_evaluations.png", "boites_perte_validation.png",
                    "boites_perte_test.png", "taux_reussite.png", "ablation_N.png",
                    "diagnostic_taille_M.png", "tableau_racine_N.md", "resume_runs.md"):
        assert attendu in noms, attendu
        assert (dossier / "figures" / attendu).stat().st_size > 0
    md = (dossier / "figures" / "tableau_racine_N.md").read_text(encoding="utf-8")
    assert "| 32 | 5 | 6 | 4 |" in md                     # N=32 : 5 qubits, 4 requêtes (M=1)


if __name__ == "__main__":
    echecs = 0
    for nom, fonction in sorted(globals().items()):
        if nom.startswith("test_") and callable(fonction):
            try:
                fonction()
                print(f"OK   {nom}")
            except AssertionError as erreur:
                echecs += 1
                print(f"ÉCHEC {nom} : {erreur!r}")
    sys.exit(1 if echecs else 0)
