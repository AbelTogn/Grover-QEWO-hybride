"""
Point d'entrée : QEWO appliqué au contrôle d'un pendule à n segments.

    python version_hybride/main.py                      # Grover (simulateur QDK)
    python version_hybride/main.py --selection argmin   # baseline classique
    python version_hybride/main.py --selection emulation  # circuit émulé en NumPy, sans QDK
"""
import argparse

import config as cfg
from qewo import entrainer
from reseau import charger_modele
from visualisation import visualiser


def creer_executeur(selection):
    if selection == "argmin":
        return None
    if selection == "emulation":
        from quantique import creer_executeur_numpy
        return creer_executeur_numpy()
    from quantique import creer_executeur_qdk
    return creer_executeur_qdk()


def main():
    analyseur = argparse.ArgumentParser(description="QEWO — pendule à n segments")
    analyseur.add_argument(
        "--selection", choices=("grover", "emulation", "argmin"), default="grover",
        help="grover : circuit Q# (QDK) ; emulation : même circuit en NumPy ; "
             "argmin : baseline classique",
    )
    selection = analyseur.parse_args().selection

    modele_existe = cfg.FICHIER_MODELE.is_file()

    while True:
        if modele_existe:
            choix = input(
                "\nChoisis une action :\n"
                "  n — nouvel entraînement (poids aléatoires)\n"
                "  c — continuer l'entraînement du modèle sauvegardé\n"
                "  v — visualiser le modèle sauvegardé\n"
                "Ton choix [n/c/v] : "
            ).strip().lower()
        else:
            choix = input(
                "\nAucun modèle sauvegardé.\n"
                "  n — nouvel entraînement\n"
                "Ton choix [n] : "
            ).strip().lower()

        if choix in (("n", "c", "v") if modele_existe else ("n",)):
            break
        print("Choix invalide.")

    if choix == "v":
        nn, _ = charger_modele()
        visualiser(nn)
        return

    if choix == "c":
        nn, deja = charger_modele()
        print(f"Modèle chargé ({deja} epochs déjà effectuées) : reprise de l'entraînement.")
    else:
        if modele_existe:
            confirmation = input(
                "Un modèle existe déjà et sera écrasé. Continuer ? [o/N] : "
            ).strip().lower()
            if confirmation != "o":
                print("Annulé.")
                return
        nn, deja = None, 0
        print("Création d'un nouveau réseau avec des poids uniformes dans [-1, 1].")

    executeur = creer_executeur(selection)
    print("Début de l'entraînement.")
    nn = entrainer(nn, deja, executeur)
    visualiser(nn)


if __name__ == "__main__":
    main()
