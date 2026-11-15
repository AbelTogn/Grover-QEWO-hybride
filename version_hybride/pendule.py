"""Physique du pendule à n segments et simulation en lot (partie 100 % classique)."""
import numpy as np

import config as cfg


class Pendule:
    """
    Pendule plan à tiges uniformes, angles absolus mesurés depuis la
    verticale descendante (theta=0 : bas, theta=pi : haut).

    Chaque couple agit directement sur l'angle absolu d'un segment.
    Équations sous forme analytique :
        M_ij = c_ij cos(th_i - th_j) + I_i delta_ij
        C_i  = sum_j c_ij sin(th_i - th_j) omega_j^2
        G_i  = g_i sin(th_i)
    avec c_ij = sum_{k >= max(i,j)} m_k a_ki a_kj,  a_ki = l_i (1 si i<k, 1/2 si i=k).
    Les méthodes acceptent des tableaux de forme (..., n) : le même code
    simule un pendule ou un lot de pendules.
    """

    def __init__(self, n_segments):
        if n_segments < 1:
            raise ValueError("N_SEGMENTS doit être supérieur ou égal à 1.")

        self.n = n_segments
        self.longueurs = np.full(self.n, cfg.LONGUEUR, dtype=float)
        self.masses = np.full(self.n, cfg.MASSE, dtype=float)
        self.inerties = self.masses * self.longueurs**2 / 12.0

        n = self.n
        self.coef_masse = np.zeros((n, n))
        self.coef_gravite = np.zeros(n)
        for k in range(n):
            a = np.zeros(n)
            for i in range(k + 1):
                a[i] = self.longueurs[i] * (1.0 if i < k else 0.5)
            self.coef_masse += self.masses[k] * np.outer(a, a)
            self.coef_gravite += self.masses[k] * cfg.GRAVITE * a
        self.diag_inerties = np.diag(self.inerties)

        self.theta = np.zeros(n)
        self.omega = np.zeros(n)

    def reinitialiser(self, aleatoire=True):
        if aleatoire:
            self.theta = np.pi + np.random.uniform(-0.25, 0.25, self.n)
            self.omega = np.random.uniform(-0.2, 0.2, self.n)
        else:
            self.theta = np.full(self.n, np.pi)
            self.omega = np.zeros(self.n)

    def matrice_masse(self, theta):
        ecart = theta[..., :, None] - theta[..., None, :]
        return self.coef_masse * np.cos(ecart) + self.diag_inerties

    def accelerations(self, theta, omega, couples):
        ecart = theta[..., :, None] - theta[..., None, :]
        M = self.coef_masse * np.cos(ecart) + self.diag_inerties
        coriolis = np.einsum(
            "...ij,...j->...i", self.coef_masse * np.sin(ecart), omega**2
        )
        rhs = (
            couples - coriolis
            - self.coef_gravite * np.sin(theta)
            - cfg.AMORTISSEMENT * omega
        )
        try:
            return np.linalg.solve(M, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            return np.zeros_like(omega)

    def pas(self, theta, omega, couples, dt=None):
        """Un pas d'Euler semi-implicite en deux sous-pas. Retourne (theta, omega)."""
        dt = cfg.DT if dt is None else dt
        couples = appliquer_actionneurs(couples)
        h = dt / 2.0
        for _ in range(2):
            omega = omega + self.accelerations(theta, omega, couples) * h
            omega = np.clip(omega, -cfg.OMEGA_MAX, cfg.OMEGA_MAX)
            theta = theta + omega * h
            theta = (theta + np.pi) % (2.0 * np.pi) - np.pi
        return theta, omega

    def avancer(self, couples, dt=None):
        self.theta, self.omega = self.pas(self.theta, self.omega, couples, dt)

    def etat(self):
        return construire_etat(self.theta, self.omega)


def construire_etat(theta, omega):
    """(..., n) -> (..., 3n) : [cos, sin, omega/10] par segment."""
    return np.stack([np.cos(theta), np.sin(theta), omega / 10.0], axis=-1).reshape(
        *theta.shape[:-1], -1
    )


def erreur_angulaire(theta):
    return np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi))


def appliquer_actionneurs(couples):
    """Couples réellement appliqués : saturation, et segment 0 sans moteur
    si cfg.SANS_MOTEUR_BASE (variante sous-actionnée)."""
    couples = np.clip(couples, -cfg.COUPLE_MAX, cfg.COUPLE_MAX)
    if cfg.SANS_MOTEUR_BASE:
        couples[..., 0] = 0.0
    return couples


def conditions_initiales(graines):
    """Mêmes tirages que reinitialiser(aleatoire=True) avec np.random.seed(graine)."""
    theta0, omega0 = [], []
    for graine in graines:
        rng = np.random.RandomState(graine)
        theta0.append(np.pi + rng.uniform(-0.25, 0.25, cfg.N_SEGMENTS))
        omega0.append(rng.uniform(-0.2, 0.2, cfg.N_SEGMENTS))
    return np.array(theta0), np.array(omega0)


def simuler_details(controle, k, graines, duree=None):
    """
    Simule k contrôleurs x len(graines) épisodes en parallèle.
    controle(theta, omega) -> couples de forme (k, s, n).
    Retourne (cout, succes), tous deux de forme (k, s) :
      - cout   : coût de chaque épisode ;
      - succes : True si |erreur angulaire| < SEUIL_REUSSITE pour tous les
                 segments pendant toute la dernière FENETRE_REUSSITE secondes.
    """
    duree = cfg.DUREE_EPISODE if duree is None else duree
    pendule = Pendule(cfg.N_SEGMENTS)
    theta0, omega0 = conditions_initiales(graines)
    theta = np.broadcast_to(theta0, (k, *theta0.shape)).copy()
    omega = np.broadcast_to(omega0, (k, *omega0.shape)).copy()
    cout_total = np.zeros((k, len(graines)))
    succes = np.ones((k, len(graines)), dtype=bool)

    n_pas = max(1, int(duree / cfg.DT))
    debut_fenetre = n_pas - max(1, int(round(cfg.FENETRE_REUSSITE / cfg.DT)))

    for etape in range(n_pas):
        couples = appliquer_actionneurs(controle(theta, omega))
        theta, omega = pendule.pas(theta, omega, couples)
        erreur = erreur_angulaire(theta)
        cout = (
            np.mean(erreur**2, axis=-1)
            + 0.025 * np.mean(omega**2, axis=-1)
            + 0.0005 * np.mean(couples**2, axis=-1)
        )
        cout_total += cout * cfg.DT
        if etape >= debut_fenetre:
            succes &= np.max(np.abs(erreur), axis=-1) < cfg.SEUIL_REUSSITE

    return cout_total, succes


def simuler_lot(controle, k, graines, duree=None):
    """Coût moyen sur les graines pour chacun des k contrôleurs, de forme (k,)."""
    return simuler_details(controle, k, graines, duree)[0].mean(axis=1)


def controle_pd(kp=None, kd=None):
    """Contrôleur PD de référence (même coût, mêmes conditions initiales)."""
    kp = cfg.PD_KP if kp is None else kp
    kd = cfg.PD_KD if kd is None else kd

    def controle(theta, omega):
        return np.clip(
            -kp * erreur_angulaire(theta) - kd * omega,
            -cfg.COUPLE_MAX, cfg.COUPLE_MAX,
        )

    return controle


def evaluer_pd(graines, kp=None, kd=None, duree=None):
    """Perte moyenne du contrôleur PD de référence."""
    return float(simuler_lot(controle_pd(kp, kd), 1, graines, duree)[0])
