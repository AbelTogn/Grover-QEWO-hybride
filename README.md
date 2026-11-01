# QEWO — Contrôle d'un pendule à n segments par apprentissage


## Objectifs du projet

- Modéliser et simuler le comportement d'un pendule triple.
- Utiliser un réseau de neurones pour déterminer les commandes de contrôle.
- Implémenter une méthode d'optimisation inspirée de QEWO.
- Intégrer l'algorithme de Grover dans le processus de recherche des paramètres.
- Comparer les performances du modèle au cours de l'entraînement.
- Sauvegarder et recharger les paramètres appris afin de poursuivre un entraînement.


## Utilisation

Depuis la racine du projet, activer l'environnement virtuel :

```bash
source .venv/bin/activate
```

Puis lancer le programme :

```bash
python version_hybride/main.py
```

Le programme propose plusieurs modes de fonctionnement :

| Choix | Fonctionnement |
|---|---|
| `n` | Créer un nouveau modèle avec des paramètres initialisés aléatoirement |
| `c` | Charger un modèle sauvegardé et poursuivre son entraînement |
| `v` | Visualiser un modèle sauvegardé |

