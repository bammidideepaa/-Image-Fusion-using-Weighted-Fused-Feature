import time

import numpy as np


# Define the objective function (Sphere Function)
def sphere(x):
    return np.sum(x ** 2)


def CSA(Crystals, fobj, LB, UB, MaxIteration):
    Cr_Number, Var_Number = Crystals.shape[0], Crystals.shape[1]

    Eval_Number = Cr_Number
    Fun_eval = fobj(Crystals[:])
    BestFitness = float('inf')
    Bestpos = np.zeros((1, Var_Number))
    Crb = Crystals
    Convergence_curve = np.zeros((MaxIteration, 1))

    t = 0
    ct = time.time()
    # Main loop
    for Iter in range(MaxIteration):
        for i in range(Cr_Number):
            # Generate new crystals
            Crmain = Crystals[np.random.choice(Cr_Number), :]
            RandSelectCrystal = np.random.choice(Cr_Number, np.random.randint(1, Cr_Number + 1), replace=False)
            Fc = np.mean(Crystals[RandSelectCrystal, :], axis=0) if len(RandSelectCrystal) > 1 else Crystals[
                                                                                                    RandSelectCrystal[
                                                                                                        0], :]

            r = 2 * np.random.rand() - 1
            r1 = 2 * np.random.rand() - 1
            r2 = 2 * np.random.rand() - 1
            r3 = 2 * np.random.rand() - 1

            # New crystals
            NewCrystals = np.array([
                Crystals[i, :] + r * Crmain,
                Crystals[i, :] + r1 * Crmain + r2 * Crb[:2],
                Crystals[i, :] + r1 * Crmain + r2 * Fc,
                Crystals[i, :] + r1 * Crmain + r2 * Crb[:2] + r3 * Fc
            ])

            # Evaluate and update crystals
            for i2 in range(4):
                NewCrystal = NewCrystals[i2, :]
                NewCrystal = np.clip(NewCrystal, LB, UB)  # Boundary handling
                Fun_evalNew = sphere(NewCrystal)

                if Fun_evalNew < Fun_eval[i]:
                    Fun_eval[i] = Fun_evalNew
                    Crystals[i, :] = NewCrystal
                Eval_Number += 1

        # Update best crystal
        BestFitness = np.min(Fun_eval)
        idbest = np.argmin(Fun_eval)
        Crb = Crystals[idbest, :]
        Convergence_curve[t] = BestFitness
        t = t + 1
    BestFitness = Convergence_curve[MaxIteration - 1][0]
    ct = time.time() - ct

    return BestFitness, Convergence_curve, Bestpos, ct
