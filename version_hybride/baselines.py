"""
Baselines classiques de MÊME BUDGET d'évaluations que QEWO.

QEWO évalue N candidats par poids (étape 3). Les baselines utilisent le même
nombre d'évaluations par poids, la même adaptation de alpha et les mêmes
graines ; seule la façon de produire et d'accepter les candidats change.
"""
import numpy as np


class StrategieAleatoire:
    """
    Recherche aléatoire avec acceptation gloutonne (montée de colline) :
      - N candidats tirés uniformément dans [w - alpha*sigma, w + alpha*sigma]
        (QEWO : grille régulière sur le même intervalle) ;
      - le meilleur est retenu (argmin, sans circuit quantique) ;
      - il n'est accepté que s'il améliore la perte courante
        (QEWO : mise à jour inconditionnelle).
    Deux différences avec QEWO : tirage aléatoire au lieu de grille, et
    acceptation gloutonne. À présenter comme tel dans l'analyse.
    """

    nom = "aleatoire"

    def __init__(self, graine):
        self.rng = np.random.default_rng([int(graine), 7])

    def candidats(self, valeur, borne, n):
        return valeur + borne * self.rng.uniform(-1.0, 1.0, n)

    def accepter(self, perte_candidat, perte_avant):
        return perte_candidat < perte_avant
