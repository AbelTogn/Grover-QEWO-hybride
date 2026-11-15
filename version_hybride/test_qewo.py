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
