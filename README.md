# Grover-QEWO Hybride

Une implémentation hybride classique-quantique de l'optimisation des poids de réseaux de neurones basée sur l'algorithme de recherche de Grover (**QEWO** — *Quantum-Enhanced Weight Optimization*).

Ce projet explore l'utilisation de l'informatique quantique pour remplacer ou assister les algorithmes classiques basés sur le gradient (comme la rétropropagation) en exploitant l'accélération quadratique $\mathcal{O}(\sqrt{N})$ de l'algorithme de Grover dans des espaces de recherche non convexes.

## Fonctionnalités

- **Optimisation hybride (QEWO) :** Recherche quantique des poids optimaux pour contourner les minima locaux sans calculer de gradients.
- **Accélération quantique :** Utilisation de l'algorithme de Grover pour parcourir efficacement l'espace discret des paramètres du réseau.
- **Benchmarks & Comparaisons :** Évaluation des performances (perte, précision et vitesse de convergence) face aux optimiseurs classiques (SGD, Adam).
- **Passage à l'échelle :** Modélisation adaptable pour des architectures de réseaux de neurones (MLP) sur des jeux de données de classification.

## Architecture du Projet

| Composant | Description |
| :--- | :--- |
| **Moteur Quantique** | Circuits quantiques et oracles construits avec **Qiskit** / **PennyLane** |
| **Réseau de Neurones** | Architecture classique (PyTorch / Scikit-Learn) évaluée par l'oracle |
| **Boucle Hybride** | Interface assurant la mesure quantique et la mise à jour des poids |

## Installation

1. **Cloner le dépôt**
   ```bash
   git clone [https://github.com/AbelTogn/Grover-QEWO-hybride.git](https://github.com/AbelTogn/Grover-QEWO-hybride.git)
   cd Grover-QEWO-hybride