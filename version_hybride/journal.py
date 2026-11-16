"""Journal CSV des expériences : une ligne par epoch (plus une ligne epoch=0 initiale)."""
import csv
from pathlib import Path

COLONNES = [
    # configuration du run (identique sur toutes les lignes)
    "methode", "graine", "n_cache", "n_sortie", "tol_cache", "tol_sortie",
    "variante", "duree_episode",
    # mesures
    "epoch", "evaluations", "requetes_oracle", "mesures", "mesures_hors_M",
    "poids_sans_maj", "taille_M_moy", "part_sans_iteration",
    "perte_train", "perte_val", "perte_pd_val", "temps_s",
]
_ENTIERS = {
    "graine", "n_cache", "n_sortie", "epoch", "evaluations",
    "requetes_oracle", "mesures", "mesures_hors_M", "poids_sans_maj",
}
_TEXTES = {"methode", "variante"}


class JournalCSV:
    """Écrit une ligne par appel et vide le tampon : un run interrompu garde ses lignes."""

    def __init__(self, chemin, champs_fixes):
        self.chemin = Path(chemin)
        self.fixes = dict(champs_fixes)
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        self._fichier = open(self.chemin, "w", newline="", encoding="utf-8")
        self._ecrivain = csv.DictWriter(self._fichier, fieldnames=COLONNES)
        self._ecrivain.writeheader()
        self._fichier.flush()

    def __call__(self, ligne):
        complet = {**self.fixes, **ligne}
        manquants = [c for c in COLONNES if c not in complet]
        if manquants:
            raise KeyError(f"Colonnes manquantes dans le journal : {manquants}")
        self._ecrivain.writerow({c: complet[c] for c in COLONNES})
        self._fichier.flush()

    def fermer(self):
        self._fichier.close()


def lire_csv(chemin):
    """Retourne la liste des lignes, avec les colonnes converties en int/float/str."""
    lignes = []
    with open(chemin, newline="", encoding="utf-8") as fichier:
        for brut in csv.DictReader(fichier):
            ligne = {}
            for cle, valeur in brut.items():
                if cle in _TEXTES:
                    ligne[cle] = valeur
                elif cle in _ENTIERS:
                    ligne[cle] = int(float(valeur))
                else:
                    ligne[cle] = float(valeur)
            lignes.append(ligne)
    return lignes


def run_termine(chemin, epochs):
    """True si le CSV existe et contient l'epoch finale attendue."""
    chemin = Path(chemin)
    if not chemin.is_file():
        return False
    try:
        lignes = lire_csv(chemin)
    except (ValueError, KeyError):
        return False
    return bool(lignes) and lignes[-1]["epoch"] >= epochs