import Std.Convert.IntAsDouble;
import Std.Math.*;
import Microsoft.Quantum.Diagnostics.*;
import Std.Arrays.*;
import Std.Convert.*;

operation Grover(pertes : Double[], seuil : Double) : Int {
    let N = Length(pertes);
    if N == 0 { return 0; }

    mutable M = 0;
    for p in pertes {
        if p <= seuil {
            set M = M + 1;
        }
    }
    
    if M == 0 { return 0; }

    let nQubits = BitSizeI(N - 1);
    use qubits = Qubit[nQubits];
    
    Init(qubits);

    let nbIterations = IterationsOpti(N, M);
    for i in 1..nbIterations {
        Oracle(pertes, seuil, qubits);
        Diffuseur(qubits);
    }
    
    let listeMesures = MResetEachZ(qubits);
    let index = ResultArrayAsInt(listeMesures);

    if index >= N { return 0; }
    return index;
}

operation Init(registre : Qubit[]) : Unit is Adj + Ctl {
    ApplyToEachCA(H, registre);
}

operation Oracle(pertes : Double[], seuil : Double, registre : Qubit[]) : Unit {
    use ancelle = Qubit();
    within {
        X(ancelle);
        H(ancelle);
    } apply {
        for i in 0 .. Length(pertes) - 1 {
            if pertes[i] <= seuil {
                ApplyControlledOnInt(i, X, registre, ancelle);
            }
        }
    }
}

operation Diffuseur(registre : Qubit[]) : Unit {
    use ancelle = Qubit();
    within {
        ApplyToEachCA(H, registre);
        ApplyToEachCA(X, registre);
        X(ancelle);
        H(ancelle);
    } apply {
        Controlled X(registre, ancelle);
    }
}

function IterationsOpti(N_total : Int, M_marques : Int) : Int {
    let N = IntAsDouble(N_total);
    let M = IntAsDouble(M_marques);
    let theta = ArcSin(Sqrt(M / N));
    let iterations = Round(0.25 * PI() / theta - 0.5);
    return MaxI(1, iterations);
}