import math
import os
import pickle
import time

import numpy as np
import pygame
from qdk import qsharp


# ============================================================
# CONFIGURATION
# ============================================================

N_SEGMENTS = 3
DT = 0.01
DUREE_EPISODE = 5.0

LONGUEUR = 1.0
MASSE = 1.0
GRAVITE = 9.81
AMORTISSEMENT = 0.08
COUPLE_MAX = 15.0

EPOCHS = 10 # 10 

N_CANDIDATS_CACHES = 32
N_CANDIDATS_SORTIE = 64
ALPHA_INITIAL = 0.1
GAMMA_DOWN = 0.95
GAMMA_UP = 1.05
TOL_RATIO_CACHE = 0.05
TOL_RATIO_SORTIE = 0.10

ALPHA_MIN = 0.01
ALPHA_MAX = 1.0

FICHIER_MODELE = "modele_2epochs.pkl"
FICHIER_QSHARP = os.path.join("version_hybride", "Main.qs")


# ============================================================
# INITIALISATION Q#
# ============================================================

def initialiser_qsharp():
    """Charge et compile l'opération Grover dans le runtime QDK."""
    if not os.path.isfile(FICHIER_QSHARP):
        raise FileNotFoundError(
            f"Fichier Q# introuvable : {FICHIER_QSHARP}\n"
            "Place Main.qs dans le dossier version_hybride/."
        )

    qsharp.init(target_profile=qsharp.TargetProfile.Unrestricted)

    with open(FICHIER_QSHARP, "r", encoding="utf-8") as fichier:
        qsharp.eval(fichier.read())


def appeler_grover(pertes, seuil):
    """
    Appelle réellement l'opération Q# Grover.
    Les pertes sont calculées classiquement, puis transmises à Q#.
    """
    valeurs = ", ".join(f"{float(x):.15f}" for x in pertes)
    seuil_qsharp = f"{float(seuil):.15f}"
    expression = f"Grover([{valeurs}], {seuil_qsharp})"

    resultat = qsharp.eval(expression)

    if isinstance(resultat, (int, np.integer)):
        return int(resultat)

    if isinstance(resultat, list) and resultat:
        return int(resultat[0])

    try:
        return int(resultat)
    except (TypeError, ValueError) as erreur:
        raise RuntimeError(
            f"Résultat inattendu de Grover : {resultat!r}"
        ) from erreur


# ============================================================
# RESEAU DE NEURONES
# ============================================================

class ReseauQEWO:
    """
    Contrôleur neuronal :
    entrées = cos(theta_i), sin(theta_i), omega_i / 10 ;
    sorties = commandes moteur normalisées entre -1 et 1.
    """

    def __init__(self, architecture, parametres=None):
        self.architecture = list(architecture)
        self.poids = []
        self.biais = []

        if parametres is None:
            for entree, sortie in zip(
                self.architecture[:-1], self.architecture[1:]
            ):
                # QEWO : initialisation uniforme des poids dans [-1, 1].
                self.poids.append(
                    np.random.uniform(-1.0, 1.0, size=(sortie, entree))
                )
                self.biais.append(np.zeros((sortie, 1)))
        else:
            self.poids = [w.copy() for w in parametres["poids"]]
            self.biais = [b.copy() for b in parametres["biais"]]

    def forward(self, X):
        A = X
        for i in range(len(self.poids) - 1):
            A = np.maximum(0.0, self.poids[i] @ A + self.biais[i])
        return np.tanh(self.poids[-1] @ A + self.biais[-1])

    def predire(self, etat):
        X = np.asarray(etat, dtype=float).reshape(-1, 1)
        return self.forward(X).ravel()

    def exporter_parametres(self):
        return {
            "poids": [w.copy() for w in self.poids],
            "biais": [b.copy() for b in self.biais],
        }

    def vecteur_parametres(self):
        morceaux = [w.ravel() for w in self.poids]
        morceaux.extend(b.ravel() for b in self.biais)
        return np.concatenate(morceaux)

    def definir_vecteur_parametres(self, vecteur):
        position = 0
        for w in self.poids:
            taille = w.size
            w[:] = vecteur[position:position + taille].reshape(w.shape)
            position += taille
        for b in self.biais:
            taille = b.size
            b[:] = vecteur[position:position + taille].reshape(b.shape)
            position += taille


# ============================================================
# SIMULATION MECANIQUE : PENDULE A N SEGMENTS
# ============================================================

class Pendule:
    """
    Pendule plan à tiges uniformes.
    Angles absolus mesurés depuis la verticale descendante :
    theta=0 correspond au bas, theta=pi au haut.
    """

    def __init__(self, n_segments):
        if n_segments < 1:
            raise ValueError("N_SEGMENTS doit être supérieur ou égal à 1.")

        self.n = n_segments
        self.longueurs = np.full(self.n, LONGUEUR, dtype=float)
        self.masses = np.full(self.n, MASSE, dtype=float)
        self.inerties = self.masses * self.longueurs**2 / 12.0
        self.theta = np.zeros(self.n)
        self.omega = np.zeros(self.n)

    def reinitialiser(self, aleatoire=True):
        if aleatoire:
            self.theta = np.pi + np.random.uniform(-0.25, 0.25, self.n)
            self.omega = np.random.uniform(-0.2, 0.2, self.n)
        else:
            self.theta = np.full(self.n, np.pi)
            self.omega = np.zeros(self.n)

    def matrice_masse(self, theta):
        n = self.n
        M = np.zeros((n, n))

        for k in range(n):
            J = np.zeros((2, n))
            for i in range(k + 1):
                facteur = self.longueurs[i] * (1.0 if i < k else 0.5)
                J[0, i] = facteur * np.cos(theta[i])
                J[1, i] = facteur * np.sin(theta[i])

            M += self.masses[k] * (J.T @ J)
            M[k, k] += self.inerties[k]

        return M

    def terme_gravite(self, theta):
        G = np.zeros(self.n)
        for k in range(self.n):
            for i in range(k + 1):
                facteur = self.longueurs[i] * (1.0 if i < k else 0.5)
                G[i] += (
                    self.masses[k] * GRAVITE
                    * facteur * np.sin(theta[i])
                )
        return G

    def terme_coriolis(self, theta, omega):
        n = self.n
        eps = 1e-5
        dM = np.zeros((n, n, n))

        for k in range(n):
            plus = theta.copy()
            moins = theta.copy()
            plus[k] += eps
            moins[k] -= eps
            dM[k] = (
                self.matrice_masse(plus) - self.matrice_masse(moins)
            ) / (2.0 * eps)

        C = np.zeros(n)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    gamma = 0.5 * (
                        dM[k, i, j] + dM[j, i, k] - dM[i, j, k]
                    )
                    C[i] += gamma * omega[j] * omega[k]
        return C

    def accelerations(self, theta, omega, couples):
        M = self.matrice_masse(theta)
        rhs = (
            couples
            - self.terme_coriolis(theta, omega)
            - self.terme_gravite(theta)
            - AMORTISSEMENT * omega
        )
        try:
            return np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return np.zeros(self.n)

    def avancer(self, couples, dt=DT):
        couples = np.clip(couples, -COUPLE_MAX, COUPLE_MAX)
        h = dt / 2.0

        for _ in range(2):
            acceleration = self.accelerations(
                self.theta, self.omega, couples
            )
            self.omega += acceleration * h
            self.omega = np.clip(self.omega, -30.0, 30.0)
            self.theta += self.omega * h
            self.theta = (self.theta + np.pi) % (2.0 * np.pi) - np.pi

    def etat(self):
        resultat = []
        for angle, vitesse in zip(self.theta, self.omega):
            resultat.extend([np.cos(angle), np.sin(angle), vitesse / 10.0])
        return np.asarray(resultat)

    def erreur_angulaire(self):
        return np.arctan2(
            np.sin(self.theta - np.pi),
            np.cos(self.theta - np.pi),
        )


# ============================================================
# FONCTION OBJECTIF : EVALUATION D'UN RESEAU
# ============================================================

def evaluer_reseau(nn, graines, duree=DUREE_EPISODE):
    scores = []
    nombre_pas = max(1, int(duree / DT))

    for graine in graines:
        rng_state = np.random.get_state()
        np.random.seed(graine)

        pendule = Pendule(N_SEGMENTS)
        pendule.reinitialiser(aleatoire=True)
        cout_total = 0.0

        for _ in range(nombre_pas):
            commande = nn.predire(pendule.etat())
            couples = COUPLE_MAX * commande
            pendule.avancer(couples)

            erreur = pendule.erreur_angulaire()
            cout = (
                np.mean(erreur**2)
                + 0.025 * np.mean(pendule.omega**2)
                + 0.0005 * np.mean(couples**2)
            )
            cout_total += cout * DT

        np.random.set_state(rng_state)
        scores.append(cout_total)

    return float(np.mean(scores))


# ============================================================
# QEWO : OPTIMISATION SÉQUENTIELLE DE CHAQUE POIDS
# ============================================================

def indice_grover_candidat(pertes, tol_ratio):
    """Sélection QEWO par Grover parmi les candidats proches du minimum.

    Les pertes sont positives. Le critère de marquage suit l'article :
        L_k <= min(L) + tol_ratio * min(L)

    Le circuit Q# doit exposer l'opération Grover(pertes, seuil) et renvoyer
    un indice entier. En cas de résultat non marqué/invalide, on utilise le
    meilleur candidat marqué comme repli robuste, sans changer le critère QEWO.
    """
    pertes = np.asarray(pertes, dtype=float)
    if pertes.ndim != 1 or len(pertes) == 0:
        raise ValueError("Il faut une liste non vide de pertes.")

    min_perte = float(np.min(pertes))
    seuil = min_perte + tol_ratio * min_perte
    indices_marques = np.flatnonzero(pertes <= seuil + 1e-12)
    indice = appeler_grover(pertes.tolist(), seuil)
    if 0 <= indice < len(pertes) and pertes[indice] <= seuil + 1e-12:
        return indice

    return int(indices_marques[np.argmin(pertes[indices_marques])])


def _liste_parametres_couches(nn):
    """Les matrices de poids sont optimisées couche par couche comme QEWO."""
    return nn.poids


def _formater_duree(secondes):
    """Convertit une durée en format lisible."""
    secondes = max(0, int(secondes))
    heures, reste = divmod(secondes, 3600)
    minutes, secondes = divmod(reste, 60)

    if heures:
        return f"{heures} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secondes:02d} s"
    return f"{secondes} s"


def entrainer(nn=None):
    """Entraînement QEWO avec calibration préalable et estimation du temps restant."""
    architecture = [3 * N_SEGMENTS, 32, 32, N_SEGMENTS]
    if nn is None:
        nn = ReseauQEWO(architecture)
    elif nn.architecture != architecture:
        raise ValueError(
            "L'architecture du modèle chargé ne correspond pas à la configuration actuelle."
        )

    alphas = getattr(nn, "alpha_couches", None)
    if alphas is None or len(alphas) != len(nn.poids):
        alphas = [ALPHA_INITIAL for _ in nn.poids]

    graines = [10, 20, 30]
    nombre_poids_par_couche = [W.size for W in nn.poids]
    candidats_par_couche = [
        N_CANDIDATS_SORTIE if t == len(nn.poids) - 1 else N_CANDIDATS_CACHES
        for t in range(len(nn.poids))
    ]
    evaluations_par_couche = [
        nombre_poids_par_couche[t] * candidats_par_couche[t]
        for t in range(len(nn.poids))
    ]
    evaluations_par_epoch = sum(evaluations_par_couche)
    nombre_poids_epoch = sum(nombre_poids_par_couche)
    evaluations_totales = evaluations_par_epoch * EPOCHS
    nombre_poids_total = nombre_poids_epoch * EPOCHS

    # Calibration avant l'entraînement : cette évaluation sert aussi de perte initiale.
    print("Calibration : mesure du temps d'une évaluation du réseau...")
    debut_calibration = time.perf_counter()
    perte_courante = evaluer_reseau(nn, graines)
    temps_evaluation_reference = time.perf_counter() - debut_calibration
    temps_evaluation_reference = max(temps_evaluation_reference, 1e-9)
    meilleur_score = -perte_courante

    print(f"Temps mesuré par évaluation (3 graines) : {temps_evaluation_reference:.3f} s")
    print(f"Perte initiale : {perte_courante:.6f}")
    print(
        f"Entraînement : {nombre_poids_epoch} poids/epoch, "
        f"{evaluations_par_epoch:,} évaluations/epoch".replace(",", " ")
    )

    evaluations_effectuees = 0
    poids_effectues_total = 0
    temps_evaluations = 0.0
    nombre_temps_evaluations = 0
    temps_grover = 0.0
    nombre_appels_grover = 0
    debut_entrainement = time.perf_counter()

    def estimer_temps_restant(evaluations_restantes, poids_restants):
        # Avant d'avoir des mesures réelles, utilise le temps de calibration.
        temps_moyen_eval = (
            temps_evaluations / nombre_temps_evaluations
            if nombre_temps_evaluations
            else temps_evaluation_reference
        )
        temps_moyen_grover = (
            temps_grover / nombre_appels_grover
            if nombre_appels_grover
            else 0.0
        )
        return (
            evaluations_restantes * temps_moyen_eval
            + poids_restants * temps_moyen_grover
        )

    for epoch in range(EPOCHS):
        ameliorations = 0

        for t, W in enumerate(_liste_parametres_couches(nn)):
            n_candidats = candidats_par_couche[t]
            tol_ratio = (
                TOL_RATIO_SORTIE
                if t == len(nn.poids) - 1
                else TOL_RATIO_CACHE
            )
            nom_couche = f"{t + 1}/{len(nn.poids)}"

            # Estimation affichée dès le début de chaque couche.
            evaluations_couche_restantes = sum(evaluations_par_couche[t:])
            evaluations_restantes = (
                (EPOCHS - epoch - 1) * evaluations_par_epoch
                + evaluations_couche_restantes
            )
            poids_couche_restants = sum(nombre_poids_par_couche[t:])
            poids_restants = (
                (EPOCHS - epoch - 1) * nombre_poids_epoch
                + poids_couche_restants
            )
            temps_restant = estimer_temps_restant(
                evaluations_restantes, poids_restants
            )
            print(
                f"\nEpoch {epoch + 1}/{EPOCHS} — couche {nom_couche} "
                f"({W.shape[1]} entrées → {W.shape[0]} neurones) | "
                f"temps restant estimé : {_formater_duree(temps_restant)}"
            )

            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    poids_effectues_total += 1
                    valeur_initiale = float(W[i, j])
                    sigma = float(np.std(W))
                    sigma_effectif = max(sigma, 1e-12)
                    perte_avant = perte_courante
                    alpha = alphas[t]

                    borne = alpha * sigma_effectif
                    candidats = np.linspace(
                        valeur_initiale - borne,
                        valeur_initiale + borne,
                        n_candidats,
                    )
                    pertes = np.empty(n_candidats, dtype=float)

                    for k, valeur in enumerate(candidats):
                        W[i, j] = valeur
                        debut_eval = time.perf_counter()
                        pertes[k] = evaluer_reseau(nn, graines)
                        temps_evaluations += time.perf_counter() - debut_eval
                        nombre_temps_evaluations += 1
                        evaluations_effectuees += 1
                    W[i, j] = valeur_initiale

                    debut_grover = time.perf_counter()
                    indice = indice_grover_candidat(pertes, tol_ratio)
                    temps_grover += time.perf_counter() - debut_grover
                    nombre_appels_grover += 1

                    W[i, j] = float(candidats[indice])
                    perte_courante = float(pertes[indice])

                    if perte_courante < perte_avant:
                        alphas[t] = max(GAMMA_DOWN * alphas[t], ALPHA_MIN)
                        ameliorations += 1
                    else:
                        alphas[t] = min(GAMMA_UP * alphas[t], ALPHA_MAX)

                    # Mise à jour discrète du terminal à chaque poids traité.
                    eval_restantes = evaluations_totales - evaluations_effectuees
                    poids_restants_total = nombre_poids_total - poids_effectues_total
                    temps_restant = estimer_temps_restant(
                        eval_restantes, poids_restants_total
                    )
                    temps_ecoule = time.perf_counter() - debut_entrainement
                    progression = 100 * poids_effectues_total / max(1, nombre_poids_total)
                    print(
                        f"\rEpoch {epoch + 1}/{EPOCHS} | couche {nom_couche} | "
                        f"poids {poids_effectues_total}/{nombre_poids_total} "
                        f"({progression:.1f} %) | restant ~{_formater_duree(temps_restant)}",
                        end="",
                        flush=True,
                    )

        print()
        nn.alpha_couches = list(alphas)
        score_mesure = -perte_courante
        meilleur_score = max(meilleur_score, score_mesure)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} terminée | "
            f"perte {perte_courante:.6f} | améliorations {ameliorations}"
        )

        with open(FICHIER_MODELE, "wb") as fichier:
            pickle.dump({
                "architecture": architecture,
                "parametres": nn.exporter_parametres(),
                "n_segments": N_SEGMENTS,
                "alpha_couches": list(alphas),
                "perte": perte_courante,
                "methode": "QEWO",
            }, fichier)

    nn.alpha_couches = list(alphas)
    print(f"Modèle sauvegardé : {os.path.abspath(FICHIER_MODELE)}")
    return nn


def charger_modele():
    with open(FICHIER_MODELE, "rb") as fichier:
        donnees = pickle.load(fichier)

    if donnees["n_segments"] != N_SEGMENTS:
        raise ValueError(
            f"Le modèle a {donnees['n_segments']} segments, "
            f"mais N_SEGMENTS vaut {N_SEGMENTS}."
        )

    nn = ReseauQEWO(
        donnees["architecture"],
        donnees["parametres"],
    )
    nn.alpha_couches = donnees.get("alpha_couches")
    return nn


# ============================================================
# VISUALISATION PYGAME
# ============================================================

def visualiser(nn):
    pygame.init()
    largeur, hauteur = 900, 700
    ecran = pygame.display.set_mode((largeur, hauteur))
    pygame.display.set_caption(
        f"Pendule à {N_SEGMENTS} segments — contrôle neuronal"
    )

    horloge = pygame.time.Clock()
    police = pygame.font.SysFont(None, 24)
    origine = np.array([largeur // 2, 150], dtype=float)
    echelle = min(100.0, 400.0 / max(1, N_SEGMENTS))

    pendule = Pendule(N_SEGMENTS)
    pendule.reinitialiser()
    pause = False
    actif = True

    while actif:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actif = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    actif = False
                elif event.key == pygame.K_SPACE:
                    pause = not pause
                elif event.key == pygame.K_r:
                    pendule.reinitialiser()

        if not pause:
            commande = nn.predire(pendule.etat())
            pendule.avancer(COUPLE_MAX * commande)

        ecran.fill((247, 247, 247))
        points = [origine.copy()]
        position = origine.copy()

        for angle in pendule.theta:
            position = position + np.array([
                echelle * np.sin(angle),
                echelle * np.cos(angle),
            ])
            points.append(position.copy())

        for i in range(N_SEGMENTS):
            a = tuple(points[i].astype(int))
            b = tuple(points[i + 1].astype(int))
            pygame.draw.line(ecran, (50, 85, 145), a, b, 7)
            pygame.draw.circle(ecran, (205, 65, 65), b, 10)

        pygame.draw.circle(ecran, (25, 25, 25), tuple(origine.astype(int)), 12)

        ecran.blit(
            police.render(
                "Espace : pause | R : réinitialiser | Échap : quitter",
                True, (20, 20, 20)
            ),
            (20, 20),
        )
        ecran.blit(
            police.render(
                "Angles : " + ", ".join(f"{x:.2f}" for x in pendule.theta),
                True, (20, 20, 20)
            ),
            (20, 50),
        )
        pygame.display.flip()
        horloge.tick(60)

    pygame.quit()


def main():
    modele_existe = os.path.isfile(FICHIER_MODELE)

    while True:
        if modele_existe:
            choix = input(
                "\\nChoisis une action :\\n"
                "  n — nouvel entraînement (poids aléatoires)\\n"
                "  c — continuer l'entraînement du modèle sauvegardé\\n"
                "  v — visualiser le modèle sauvegardé\\n"
                "Ton choix [n/c/v] : "
            ).strip().lower()
        else:
            choix = input(
                "\\nAucun modèle sauvegardé.\\n"
                "  n — nouvel entraînement\\n"
                "Ton choix [n] : "
            ).strip().lower()

        if choix in ("n", "c", "v"):
            break
        print("Choix invalide. Entre n, c ou v.")

    if choix == "v":
        if not modele_existe:
            print("Aucun modèle à visualiser. Lance d'abord un nouvel entraînement.")
            return
        nn = charger_modele()
        visualiser(nn)
        return

    if choix == "c":
        if not modele_existe:
            print("Aucun modèle à reprendre. Lance d'abord un nouvel entraînement.")
            return
        nn = charger_modele()
        print("Modèle chargé : reprise de l'entraînement à partir de ses poids.")
    else:
        nn = None
        print("Création d'un nouveau réseau avec des poids uniformes dans [-1, 1].")

    initialiser_qsharp()
    print("Début de l'entrainement.")
    nn = entrainer(nn)
    visualiser(nn)


if __name__ == "__main__":
    main()