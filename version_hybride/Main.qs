import Std.Arrays.*;
import Std.Convert.*;
import Std.Math.*;

/// Recherche un indice parmi les candidats dont la perte est <= seuil.
/// Les pertes sont pré-calculées classiquement par Python.
/// Grover amplifie les indices marqués par cet oracle.
operation Grover(pertes : Double[], seuil : Double) : Int {
    let N = Length(pertes);

    if N == 0 {
        return 0;
    }

    if N == 1 {
        return 0;
    }

    mutable M = 0;
    for perte in pertes {
        if perte <= seuil {
            set M = M + 1;
        }
    }

    // Aucun état marqué : il n'y a rien à amplifier.
    if M == 0 {
        return 0;
    }

    let nbQubits = BitSizeI(N - 1);
    use registre = Qubit[nbQubits];

    // Superposition uniforme.
    ApplyToEachCA(H, registre);

    // Nombre d'itérations adapté au nombre d'états marqués.
    let iterations = IterationsOpti(N, M);

    for _ in 1..iterations {
        Oracle(pertes, seuil, registre);
        Diffuseur(registre);
    }

    let mesures = MResetEachZ(registre);
    let index = ResultArrayAsInt(mesures);

    // Les registres de taille puissance de deux peuvent coder
    // des indices qui ne correspondent à aucun candidat.
    if index >= N {
        return 0;
    }

    return index;
}

/// Oracle à déphasage par kickback de phase.
/// Les pertes sont des données classiques : la boucle décrit
/// les états à marquer dans le circuit.
operation Oracle(pertes : Double[], seuil : Double, registre : Qubit[]) : Unit {
    use ancilla = Qubit();

    within {
        X(ancilla);
        H(ancilla);
    } apply {
        for i in 0..(Length(pertes) - 1) {
            if pertes[i] <= seuil {
                ApplyControlledOnInt(i, X, registre, ancilla);
            }
        }
    }
}

/// Réflexion autour de la superposition uniforme.
operation Diffuseur(registre : Qubit[]) : Unit {
    ApplyToEachCA(H, registre);
    ApplyToEachCA(X, registre);

    // Après H-X, on inverse la phase de |11...1>.
    // Pour un seul qubit, Z suffit. Pour plusieurs qubits,
    // le dernier qubit est la cible et les précédents sont les contrôles.
    let n = Length(registre);
    if n == 1 {
        Z(registre[0]);
    } else {
        Controlled Z(registre[0..(n - 2)], registre[n - 1]);
    }

    ApplyToEachCA(X, registre);
    ApplyToEachCA(H, registre);
}

function IterationsOpti(NTotal : Int, MMarques : Int) : Int {
    if NTotal <= 1 or MMarques <= 0 {
        return 0;
    }

    let ratio = IntAsDouble(MMarques) / IntAsDouble(NTotal);

    // Quand presque tous les états sont marqués, une itération
    // de Grover n'est pas utile : on garde une mesure uniforme.
    if ratio >= 0.5 {
        return 0;
    }

    let theta = ArcSin(Sqrt(ratio));
    let iterations = Round(0.25 * PI() / theta - 0.5);

    return MaxI(0, iterations);
}