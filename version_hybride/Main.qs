import Std.Convert.*;

/// Étape 5 de QEWO : circuit de Grover sur le registre des indices de candidats.
///
/// Ressources : nbQubits qubits d'indice + 1 ancilla (comme dans l'article :
/// ceil(log2 N) + 1 qubits). Le circuit ne contient aucun calcul de perte :
/// les pertes sont évaluées classiquement (étape 3), l'ensemble M des indices
/// à marquer (étape 4) et le nombre d'itérations sont calculés côté Python
/// et passés en paramètres.
///
/// Retourne le résultat de mesure du registre (poids faible en premier).
/// C'est le point d'entrée à compiler pour du matériel : il ne renvoie que
/// des Result, sans calcul classique sur les mesures.
operation Grover(nbQubits : Int, marques : Int[], iterations : Int) : Result[] {
    use registre = Qubit[nbQubits];
    use ancilla = Qubit();

    // Superposition uniforme de tous les indices.
    ApplyToEach(H, registre);

    // L'ancilla est préparée une seule fois dans |-> : le contrôle sur
    // l'ancilla produit alors un déphasage (kickback de phase).
    within {
        X(ancilla);
        H(ancilla);
    } apply {
        for _ in 1..iterations {
            Oracle(marques, registre, ancilla);
            Diffuseur(registre);
        }
    }

    return MResetEachZ(registre);
}

/// Même circuit, mais la mesure est convertie en entier (simulateur uniquement :
/// cette conversion est un calcul classique sur les mesures).
operation GroverIndice(nbQubits : Int, marques : Int[], iterations : Int) : Int {
    return ResultArrayAsInt(Grover(nbQubits, marques, iterations));
}

/// Oracle : inverse la phase des états de base dont l'indice est dans M.
operation Oracle(marques : Int[], registre : Qubit[], ancilla : Qubit) : Unit {
    for indice in marques {
        ApplyControlledOnInt(indice, X, registre, ancilla);
    }
}

/// Opérateur de diffusion : réflexion autour de la superposition uniforme
/// (à une phase globale près).
operation Diffuseur(registre : Qubit[]) : Unit {
    ApplyToEach(H, registre);
    ApplyToEach(X, registre);

    // Après H-X, on inverse la phase de |11...1>.
    let n = Length(registre);
    if n == 1 {
        Z(registre[0]);
    } else {
        Controlled Z(registre[0..(n - 2)], registre[n - 1]);
    }

    ApplyToEach(X, registre);
    ApplyToEach(H, registre);
}
