from collections import Counter
import os
import pickle
from PIL import Image
import numpy as np
from qdk import qsharp

qsharp.init(target_profile=qsharp.TargetProfile.Unrestricted)

with open("./Main.qs", "r") as f:
    code_qsharp = f.read()
qsharp.eval(code_qsharp)

TAILLE_IMAGE = (16, 16)

# --- 1. Structure du Réseau (QEWO) ---

class ReseauQEWO:
    """Réseau de neurones dense optimisé par l'algorithme QEWO (Quantum-Enhanced Weight Optimization)."""
    
    def __init__(self, architecture: list[int]):
        """Initialise la structure du réseau, les poids et les biais de manière aléatoire."""
        self.architecture = architecture
        self.poids = {}
        self.biais = {}
        self.nb_couches = len(architecture) - 1
        
        for l in range(self.nb_couches):
            self.poids[l] = np.random.randn(architecture[l+1], architecture[l]) * 0.1
            self.biais[l] = np.random.randn(architecture[l+1], 1) * 0.1

    def activation_relu(self, x):
        """Applique la fonction d'activation ReLU (Rectified Linear Unit)."""
        return np.maximum(0, x)
        
    def activation_sigmoid(self, x):
        """Applique la fonction d'activation Sigmoïde."""
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        """Effectue la propagation avant (forward pass) dans l'ensemble du réseau."""
        A = X
        for l in range(self.nb_couches - 1):
            A = self.activation_relu(np.dot(self.poids[l], A) + self.biais[l])
        return self.activation_sigmoid(np.dot(self.poids[self.nb_couches-1], A) + self.biais[self.nb_couches-1])

    def calculer_perte(self, X, Y):
        """Calcule l'erreur quadratique moyenne (MSE) du réseau sur un jeu de données."""
        return np.mean((self.forward(X) - Y) ** 2)

    def predire(self, X):
        """Prédit la classe ayant la plus forte probabilité pour chaque échantillon."""
        return np.argmax(self.forward(X), axis=0)


def optimisation_poids(nn, couche, i, j, sigma, X, Y, tol_ratio=0.00, nb_shots=10):
    """Optimise un poids individuel via une grille de candidats évaluée par l'algorithme quantique de Grover."""
    alpha = 0.5
    N_candidats = 16 
    
    w_actuel = nn.poids[couche][i, j]
    inf = w_actuel - alpha * sigma
    sup = w_actuel + alpha * sigma
    candidats = np.linspace(inf, sup, N_candidats)
    
    pertes = []
    for candidat in candidats:
        nn.poids[couche][i, j] = candidat
        pertes.append(float(nn.calculer_perte(X, Y)))
        
    nn.poids[couche][i, j] = w_actuel
    min_loss = min(pertes)
    seuil = min_loss + (tol_ratio * min_loss) + 1e-6
    
    expr = f"Grover({pertes}, {seuil})"
    
    tirages = []
    for _ in range(nb_shots):
        res = qsharp.eval(expr)
        idx = int(res[0]) if isinstance(res, list) else int(res)
        tirages.append(idx)
        
    index_gagnant = Counter(tirages).most_common(1)[0][0]
    return float(candidats[index_gagnant])


# --- 2. Module d'importation des images ---

def charger_images(dossier_dataset, taille_image=TAILLE_IMAGE):
    """Parcourt un dossier structuré par classes, redimensionne et vectorise les images pour le réseau."""
    classes = sorted([d for d in os.listdir(dossier_dataset) if os.path.isdir(os.path.join(dossier_dataset, d))])
    nb_classes = len(classes)
    
    X_liste = []
    Y_liste = []

    print(f"Classes détectées ({nb_classes}) : {classes}")

    for idx_classe, nom_classe in enumerate(classes):
        chemin_classe = os.path.join(dossier_dataset, nom_classe)
        for fichier in os.listdir(chemin_classe):
            if fichier.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                chemin_img = os.path.join(chemin_classe, fichier)
                
                img = Image.open(chemin_img).convert('L')
                img = img.resize(taille_image)
                
                vecteur_img = np.array(img, dtype=np.float32).flatten() / 255.0
                X_liste.append(vecteur_img)
                
                y_onehot = np.zeros(nb_classes)
                y_onehot[idx_classe] = 1.0
                Y_liste.append(y_onehot)

    X = np.array(X_liste).T  
    Y = np.array(Y_liste).T  
    
    return X, Y, classes


# --- 3. Fonctions de Sauvegarde / Chargement ---

def sauvegarder_modele(nn, classes, chemin="modele.pkl"):
    """Exporte les poids, biais, architecture et labels du modèle dans un fichier pickle."""
    with open(chemin, "wb") as f:
        pickle.dump({"architecture": nn.architecture, "poids": nn.poids, "biais": nn.biais, "classes": classes}, f)
    print(f"\n[+] Modèle et classes sauvegardés dans '{chemin}'")

def charger_et_predire_image(chemin_image, chemin_modele="modele.pkl", taille_image=TAILLE_IMAGE):
    """Charge un modèle sauvegardé et réalise une inférence sur une image inconnue."""
    with open(chemin_modele, "rb") as f:
        donnees = pickle.load(f)
        
    nn = ReseauQEWO(donnees["architecture"])
    nn.poids = donnees["poids"]
    nn.biais = donnees["biais"]
    classes = donnees["classes"]
    
    # Traitement de la nouvelle image
    img = Image.open(chemin_image).convert('L').resize(taille_image)
    vecteur_img = (np.array(img, dtype=np.float32).flatten() / 255.0).reshape(-1, 1)
    
    index_classe = nn.predire(vecteur_img)[0]
    return classes[index_classe]


# --- 4. Boucle Principale ---

def main():
    """Fonction principale d'exécution : chargement des données, entraînement QEWO et sauvegarde."""
    DOSSIER_DATASET = "./dataset"  
    TAILLE_IMG = (4, 4)               
    
    if not os.path.exists(DOSSIER_DATASET):
        print(f"Erreur : Le dossier '{DOSSIER_DATASET}' n'existe pas.")
        return

    X_train, Y_train, noms_classes = charger_images(DOSSIER_DATASET, TAILLE_IMG)
    
    nb_pixels = X_train.shape[0]
    nb_classes = len(noms_classes)
    
    nn = ReseauQEWO([nb_pixels, 16, nb_classes])
    
    nb_epoques = 10
    print(f"Perte initiale : {nn.calculer_perte(X_train, Y_train):.6f}")

    print("\nLancement de l'entraînement quantique sur vos images...")
    for epoch in range(nb_epoques):
        for l in range(nn.nb_couches):
            sigma = np.std(nn.poids[l])
            for i in range(nn.poids[l].shape[0]):
                for j in range(nn.poids[l].shape[1]):
                    nouveau_poids = optimisation_poids(nn, l, i, j, sigma, X_train, Y_train)
                    nn.poids[l][i, j] = nouveau_poids

        perte_courante = nn.calculer_perte(X_train, Y_train)
        print(f"Époque {epoch + 1:02d}/{nb_epoques} — Perte MSE : {perte_courante:.6f}")

    sauvegarder_modele(nn, noms_classes, "modele.pkl")

if __name__ == "__main__":
    main()