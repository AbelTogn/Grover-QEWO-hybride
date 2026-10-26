
import time
import statistics

import main_qewo_fidele as qewo

# Même architecture que le programme pour un segment
reseau = qewo.ReseauQEWO([3, 32, 32, 1])

# Même graines que pendant l'entraînement
graines = [10, 20, 30]

# Échauffement : évite de se fier à une toute première mesure
qewo.evaluer_reseau(reseau, graines)

temps = []

# Mesurer 5 évaluations complètes
for i in range(5):
    debut = time.perf_counter()

    perte = qewo.evaluer_reseau(reseau, graines)

    fin = time.perf_counter()
    duree = fin - debut
    temps.append(duree)

    print(
        f"Évaluation {i + 1} : "
        f"{duree:.3f} s — perte = {perte:.6f}"
    )

moyenne = statistics.mean(temps)

print(f"\nTemps moyen par évaluation : {moyenne:.3f} s")
print(f"Temps moyen par graine : {moyenne / len(graines):.3f} s")
