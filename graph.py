from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from qdk import qsharp

qsharp.init(target_profile=qsharp.TargetProfile.Unrestricted)

with open("./Main.qs", "r") as f:
    code_qsharp = f.read()
qsharp.eval(code_qsharp)

class ReseauQEWO:
    def __init__(self, architecture: list[int]):
        self.architecture = architecture
        self.poids = {}
        self.biais = {}
        self.nb_couches = len(architecture) - 1
        for l in range(self.nb_couches):
            self.poids[l] = np.random.randn(architecture[l+1], architecture[l]) * 0.1
            self.biais[l] = np.random.randn(architecture[l+1], 1) * 0.1

    def activation_relu(self, x): return np.maximum(0, x)
    def activation_sigmoid(self, x): return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        A = X
        for l in range(self.nb_couches - 1):
            A = self.activation_relu(np.dot(self.poids[l], A) + self.biais[l])
        return self.activation_sigmoid(np.dot(self.poids[self.nb_couches-1], A) + self.biais[self.nb_couches-1])

    def calculer_perte(self, X, Y):
        return np.mean((self.forward(X) - Y) ** 2)


def optimisation_poids(nn, couche, i, j, sigma, X, Y, tol_ratio=0.00, nb_shots=15):
    alpha = 0.5
    N_candidats = 16 
    
    w_actuel = nn.poids[couche][i, j]
    borne_inf = w_actuel - alpha * sigma
    borne_sup = w_actuel + alpha * sigma
    candidats = np.linspace(borne_inf, borne_sup, N_candidats)
    
    pertes = []
    for candidat in candidats:
        nn.poids[couche][i, j] = candidat
        pertes.append(float(nn.calculer_perte(X, Y)))
        
    nn.poids[couche][i, j] = w_actuel
    min_loss = min(pertes)
    seuil = min_loss + (tol_ratio * min_loss) + 1e-6
    
    # Correction : Appel de l'opération "Grover" définie dans le Q#
    expr = f"Grover({pertes}, {seuil})"
    
    tirages = []
    for _ in range(nb_shots):
        res = qsharp.eval(expr)
        idx = int(res[0]) if isinstance(res, list) else int(res)
        tirages.append(idx)
        
    index_gagnant = Counter(tirages).most_common(1)[0][0]
    return float(candidats[index_gagnant])


def main():
    # Correction : Entrées au format (3_caracteristiques, 3_echantillons) pour correspondre à l'entrée [3, 4, 1]
    X_train = np.array([
        [0.1, 0.2, 0.3],
        [0.9, 0.8, 0.7],
        [0.2, 0.3, 0.4]
    ])
    Y_train = np.array([[0.0, 1.0, 0.0]])

    nn = ReseauQEWO([3, 4, 1])
    
    nb_epoques = 120
    historique_perte = []
    
    perte_init = nn.calculer_perte(X_train, Y_train)
    historique_perte.append(perte_init)
    print(f"Perte initiale (Époque 0) : {perte_init:.6f}")

    print(f"\nLancement de l'optimisation sur {nb_epoques} époques...")
    for epoch in range(nb_epoques):
        for l in range(nn.nb_couches):
            sigma = np.std(nn.poids[l])
            for i in range(nn.poids[l].shape[0]):
                for j in range(nn.poids[l].shape[1]):
                    nouveau_poids = optimisation_poids(nn, l, i, j, sigma, X_train, Y_train)
                    nn.poids[l][i, j] = nouveau_poids

        perte_courante = nn.calculer_perte(X_train, Y_train)
        historique_perte.append(perte_courante)
        print(f"Époque {epoch + 1:02d}/{nb_epoques} — Perte : {perte_courante:.6f}")

    # --- Tracé du graphique ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(0, nb_epoques + 1), historique_perte, marker='o', color='b', linewidth=2, label="Grover-Opt")
    plt.title("Convergence de la fonction de perte (QEWO - Grover)")
    plt.xlabel("Époque")
    plt.ylabel("Erreur Quadratique Moyenne (MSE)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_grover.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()