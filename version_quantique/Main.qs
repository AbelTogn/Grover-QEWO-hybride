namespace QuantumQEWO {
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Math;

    // --- 1. FONCTIONS MATHÉMATIQUES & NEURALES EN Q# ---

    function DotProduct(v1 : Double[], v2 : Double[]) : Double {
        mutable sum = 0.0;
        for i in 0 .. Length(v1) - 1 {
            set sum += v1[i] * v2[i];
        }
        return sum;
    }

    function MatMulVec(mat : Double[][], vec : Double[]) : Double[] {
        let rows = Length(mat);
        mutable result = new Double[rows];
        for i in 0 .. rows - 1 {
            set result w/= i <- DotProduct(mat[i], vec);
        }
        return result;
    }

    function AddVectors(v1 : Double[], v2 : Double[]) : Double[] {
        let n = Length(v1);
        mutable result = new Double[n];
        for i in 0 .. n - 1 {
            set result w/= i <- v1[i] + v2[i];
        }
        return result;
    }

    function Relu(x : Double) : Double {
        return x > 0.0 ? x | 0.0;
    }

    function Sigmoid(x : Double) : Double {
        return 1.0 / (1.0 + Exp(-x));
    }

    // Propagation avant (Forward Pass) pour un échantillon unique
    function ForwardSample(
        architecture : Int[], 
        poids : Double[][][], 
        biais : Double[][], 
        xInput : Double[]
    ) : Double[] {
        let nbCouches = Length(architecture) - 1;
        mutable a = xInput;

        for l in 0 .. nbCouches - 2 {
            let z = AddVectors(MatMulVec(poids[l], a), biais[l]);
            mutable act = new Double[Length(z)];
            for i in 0 .. Length(z) - 1 {
                set act w/= i <- Relu(z[i]);
            }
            set a = act;
        }

        let zFinal = AddVectors(MatMulVec(poids[nbCouches - 1], a), biais[nbCouches - 1]);
        mutable output = new Double[Length(zFinal)];
        for i in 0 .. Length(zFinal) - 1 {
            set output w/= i <- Sigmoid(zFinal[i]);
        }
        return output;
    }

    // Calcul de la perte MSE sur tout le dataset
    function CalculerPerte(
        architecture : Int[], 
        poids : Double[][][], 
        biais : Double[][], 
        X : Double[][], 
        Y : Double[][]
    ) : Double {
        let nbSamples = Length(X);
        mutable totalLoss = 0.0;

        for s in 0 .. nbSamples - 1 {
            let pred = ForwardSample(architecture, poids, biais, X[s]);
            let target = Y[s];
            for i in 0 .. Length(pred) - 1 {
                let diff = pred[i] - target[i];
                set totalLoss += diff * diff;
            }
        }
        return totalLoss / IntAsDouble(nbSamples * Length(Y[0]));
    }

    // --- 2. PARTIE QUANTIQUE (GROVER) ---

    operation Diffuseur(qubits : Qubit[]) : Unit {
        within {
            ApplyToEachA(H, qubits);
            ApplyToEachA(X, qubits);
        } apply {
            Controlled Z(Rest(qubits), Head(qubits));
        }
    }

    operation EvaluateurPoids(
        indexCandidat : Int, 
        pertes : Double[], 
        seuil : Double, 
        registre : Qubit[], 
        ancelle : Qubit
    ) : Unit {
        for i in 0 .. Length(pertes) - 1 {
            if pertes[i] <= seuil {
                ApplyControlledOnInt(i, X, registre, ancelle);
            }
        }
    }

    operation OptimiserPoidsGrover(pertes : Double[], seuil : Double) : Int {
        let N = Length(pertes);
        if N == 0 { return 0; }

        mutable nbSolutions = 0;
        for p in pertes {
            if p <= seuil {
                set nbSolutions = nbSolutions + 1;
            }
        }
        if nbSolutions == 0 { return 0; }

        let nbQubits = BitSizeI(N - 1);
        use registre = Qubit[nbQubits];
        use ancelle = Qubit();

        ApplyToEach(H, registre);
        X(ancelle);
        H(ancelle);

        let theta = ArcSin(Sqrt(IntAsDouble(nbSolutions) / IntAsDouble(N)));
        let nbIterations = MaxI(1, Round(0.25 * PI() / theta - 0.5));

        for _ in 1 .. nbIterations {
            EvaluateurPoids(0, pertes, seuil, registre, ancelle);
            Diffuseur(registre);
        }

        let resultats = ForEach(M, registre);
        let index = ResultArrayAsInt(resultats);

        ResetAll(registre);
        Reset(ancelle);

        if index >= N { return 0; }
        return index;
    }
}