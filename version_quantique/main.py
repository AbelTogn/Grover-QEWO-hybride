from collections import Counter
import os
import pickle
from PIL import Image
import numpy as np
from qdk import qsharp

qsharp.init(target_profile=qsharp.TargetProfile.Unrestricted)

with open("version_quantique/Main.qs", "r") as f:
    code_qsharp = f.read()
qsharp.eval(code_qsharp)

TAILLE_IMAGE = (4, 4)


class ReseauQEWOQuantique:
    """Réseau de neurones dont la structure et les données sont synchronisées avec Q#."""
    
    def __init__(self, architecture: list[int]):
        self.architecture = architecture
        self.nb_couches = len(architecture) - 1
        self.poids = []
        self.biais = []
        
        for l in range(self.nb_couches):
            p_L = (np.random.randn(architecture[l+1], architecture[l]) * 0.1).tolist()
            b_L = (np.random.randn(architecture[l+1], 1) * 0.1).flatten().tolist()
            self.poids.append(p_L)
            self.biais.append(b_L)

    def calculer_perte(self, X_list, Y_list):
        """Appelle la fonction de perte MSE implémentée en Q#."""
        return float(qsharp.eval(f"QuantumQEWO.CalculerPerte({self.architecture}, {self.poids}, {self.biais}, {X_list}, {Y_list})"))


def optimisation_poids_qpu(nn, couche, i, j, sigma, X_list, Y_list, tol_ratio=0.00, nb_shots=10):
    """Explore l'espace des candidats pour un poids et délague la sélection à Grover (Q#)."""
    alpha = 0.5
    N_candidats = 16 
    
    w_actuel = nn.poids[couche][i][j]
    inf = w_actuel - alpha * sigma
    sup = w_actuel + alpha * sigma
    candidats = np.linspace(inf, sup, N_candidats)
    
    pertes = []
    for candidat in candidats:
        nn.poids[couche][i][j] = float(candidat)
        pertes.append(nn.calculer_perte(X_list, Y_list))
        
    nn.poids[couche][i][j] = w_actuel
    min_loss = min(pertes)
    seuil = min_loss + (tol_ratio * min_loss) + 1e-6
    
    # Appel de l'opération Q# d'optimisation par Grover
    expr = f"QuantumQEWO.OptimiserPoidsGrover({pertes}, {seuil})"
    
    tirages = []
    for _ in range(nb_shots):
        res = qsharp.eval(expr)
        idx = int(res[0]) if isinstance(res, list) else int(res)
        tirages.append(idx)
        
    index_gagnant = Counter(tirages).most_common(1)[0][0]
    return float(candidats[index_gagnant])



def charger_images(dossier_dataset, taille_image=TAILLE_IMAGE):
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
                
                img = Image.open(chemin_img).convert('L').resize(taille_image)
                vecteur_img = (np.array(img, dtype=np.float32).flatten() / 255.0).tolist()
                X_liste.append(vecteur_img)
                
                y_onehot = np.zeros(nb_classes)
                y_onehot[idx_classe] = 1.0
                Y_liste.append(y_onehot.tolist())

    return X_liste, Y_liste, classes



def sauvegarder_modele(nn, classes, chemin="modele_quantique.pkl"):
    with open(chemin, "wb") as f:
        pickle.dump({"architecture": nn.architecture, "poids": nn.poids, "biais": nn.biais, "classes": classes}, f)
    print(f"\n[+] Modèle et classes sauvegardés dans '{chemin}'")



def main():
    DOSSIER_DATASET = "./dataset"  
    TAILLE_IMG = (4, 4)               
    
    if not os.path.exists(DOSSIER_DATASET):
        print(f"Erreur : Le dossier '{DOSSIER_DATASET}' n'existe pas.")
        return

    X_train, Y_train, noms_classes = charger_images(DOSSIER_DATASET, TAILLE_IMG)
    
    nb_pixels = len(X_train[0])
    nb_classes = len(noms_classes)
    
    nn = ReseauQEWOQuantique([nb_pixels, 16, nb_classes])
    
    nb_epoques = 10
    print(f"Perte initiale : {nn.calculer_perte(X_train, Y_train):.6f}")

    print("\nLancement de l'entraînement quantique (QPU)...")
    for epoch in range(nb_epoques):
        for l in range(nn.nb_couches):
            plat_poids = [p for ligne in nn.poids[l] for p in ligne]
            sigma = float(np.std(plat_poids))
            
            rows = len(nn.poids[l])
            cols = len(nn.poids[l][0])
            for i in range(rows):
                for j in range(cols):
                    nouveau_poids = optimisation_poids_qpu(nn, l, i, j, sigma, X_train, Y_train)
                    nn.poids[l][i][j] = nouveau_poids

        perte_courante = nn.calculer_perte(X_train, Y_train)
        print(f"Époque {epoch + 1}/{nb_epoques} — Perte MSE : {perte_courante:.6f}")

    sauvegarder_modele(nn, noms_classes, "modele_quantique.pkl")

if __name__ == "__main__":
    main()