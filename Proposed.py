import numpy as np
import time


# Example Levy function (you need to define or import it)
def levy(n, dim, beta):
    sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
               (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size=(n, dim))
    v = np.random.normal(0, 1, size=(n, dim))
    step = u / (np.abs(v) ** (1 / beta))
    return step


# Modified Hippopotamus Optimization Algorithm (MHOA) Starting Line No. 66
def Proposed(SearchAgents, fitness, lowerbound, upperbound, Max_iterations):
    dimension = SearchAgents.shape[1]

    # Lower and upper limits for variables
    lowerbound = np.ones((1, dimension)) * lowerbound
    upperbound = np.ones((1, dimension)) * upperbound

    X = SearchAgents
    fit = np.array([fitness(X[i, :]) for i in range(SearchAgents.shape[0])])
    Best_pos = np.zeros((dimension, 1))
    Best_score = float('inf')
    Convergence_curve = np.zeros((Max_iterations, 1))

    # Main loop
    t = 0
    ct = time.time()
    for t in range(Max_iterations):
        # Update the best candidate solution
        best = np.min(fit)
        fbest = best
        location = np.argmin(fit)
        if t == 0:
            Xbest = X[location, :]
            fbest = best
        elif best < fbest:
            fbest = best
            Xbest = X[location, :]
        else:
            Xbest = X

        # Phase 1: Hippopotamus position update in river or pond (Exploration)
        for i in range(SearchAgents.shape[0] // 2):
            print('i= ', i)
            Dominant_hippopotamus = Xbest
            I1 = np.random.randint(1, 3)
            I2 = np.random.randint(1, 3)
            Ip1 = np.random.randint(0, 2, 2)
            RandGroupNumber = np.random.randint(1, SearchAgents.shape[0], SearchAgents.shape[0])[0]
            RandGroup = np.random.randint(1, SearchAgents.shape[0], size=SearchAgents.shape[0])[:RandGroupNumber]
            MeanGroup = np.mean(X[RandGroup, :], axis=0) if len(RandGroup) > 1 else X[RandGroup, :]
            Alfa = [I2 * np.random.rand(dimension) + (1 - Ip1[0]),
                    2 * np.random.rand(dimension) - 1,
                    np.random.rand(dimension),
                    I1 * np.random.rand(dimension) + (1 - Ip1[1]),
                    np.random.rand()]
            A = Alfa[np.random.randint(0, 5)]
            B = Alfa[np.random.randint(0, 5)]
            X_P1 = X[i, :] + np.random.rand() * (Dominant_hippopotamus - I1 * X[i, :])
            T = np.exp(-t / Max_iterations)
            currentfit = fit[i]
            worstfit = np.max(fit)
            bestfit = np.min(fit)
            if T > ((currentfit + worstfit) / (2 * worstfit)):
                X_P2 = X_P1 + A * (Dominant_hippopotamus - I2 * MeanGroup)
            else:
                if np.random.random() > (currentfit + bestfit) / (2 * worstfit):
                    X_P2 = X_P1 + B * (MeanGroup - Dominant_hippopotamus)
                else:
                    X_P2 = lowerbound[i, :] + (np.random.uniform(lowerbound.shape[1], upperbound.shape[1], dimension) * (upperbound[i, :] - lowerbound[i, :]))

        # Phase 2: Hippopotamus defense against predators (Exploration)
        for j in range(SearchAgents.shape[0] // 2, SearchAgents.shape[0]):
            print('j= ', j)
            predator = np.random.rand(dimension) * (upperbound - lowerbound) + lowerbound
            F_HL = fitness(predator)
            distance2Leader = np.abs(predator - X[j, :])
            b = np.random.uniform(2, 4)
            c = np.random.uniform(1, 1.5)
            d = np.random.uniform(2, 3)
            l = np.random.uniform(-2 * np.pi, 2 * np.pi)
            RL = X_P2 * levy(SearchAgents.shape[0], dimension, 1.5)
            if (fit[j] > F_HL).all():
                X_P3 = RL[j, :] * predator + (b / (c - d * np.cos(l))) * (1 / distance2Leader)
            else:
                X_P3 = RL[j, :] * predator + (b / (c - d * np.cos(l))) * (
                        1 / (2 * distance2Leader + np.random.rand(dimension)))
            X_P3 = np.clip(X_P3, lowerbound, upperbound)

            F_P3 = fitness(X_P3)
            if (F_P3 < fit[j]).all():
                X[j, :] = X_P3[j, :]
                fit[j] = F_P3[j]

        # Phase 3: Hippopotamus escaping from the predator (Exploitation)
        for k in range(SearchAgents.shape[0]):
            print('k= ', k)
            LO_LOCAL = lowerbound / (t + 1)
            HI_LOCAL = upperbound / (t + 1)
            Alfa = [2 * np.random.rand(dimension) - 1, np.random.rand(), np.random.randn()]
            D = Alfa[0][np.random.randint(0, 1)]
            X_P4 = X + np.random.rand() * (LO_LOCAL + D * (HI_LOCAL - LO_LOCAL))
            X_P4 = np.clip(X_P4, lowerbound, upperbound)

            F_P4 = fitness(X_P4)
            if (F_P4 < fit[k]).all():
                X = X_P4
                fit = F_P4

    ct = time.time() - ct
    Best_score = fbest
    Best_pos = Xbest
    Convergence_curve[t] = Best_score
    Best_score = Convergence_curve[Max_iterations - 1][0]

    return Best_score, Convergence_curve, Best_pos, ct
