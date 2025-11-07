import time

import numpy as np
import matplotlib.pyplot as plt

# Northern Goshawk Optimization (NGO)
def NGO(Search_Agents, objective, Lowerbound, Upperbound, Max_iterations):
    N, dimensions = Search_Agents.shape[0], Search_Agents.shape[1]

    X = Search_Agents
    fit = objective(X[:])
    Best_pos = np.zeros((1, dimensions))
    NGO_curve = np.zeros(Max_iterations)

    # Store best solution
    fbest = np.min(fit)
    blocation = np.argmin(fit)
    xbest = X[blocation, :]
    Score = float('inf')

    ct = time.time()
    for t in range(Max_iterations):  # algorithm iteration

        # Update: BEST proposed solution
        best = np.min(fit)
        blocation = np.argmin(fit)


        if t == 0:
            xbest = X[blocation, :]  # Optimal location
            fbest = best  # The optimization objective function
        elif best < fbest:
            fbest = best
            xbest = X[blocation, :]

        # Update Northern goshawks based on PHASE1 and PHASE2
        for i in range(Search_Agents.shape[0]):
            # Phase 1: Exploration
            I = round(1 + np.random.rand())
            k = np.random.choice(Search_Agents.shape[0])
            P = X[k, :]  # Eq. (3)
            F_P = fit[k]

            if fit[i] > F_P:
                X_new = X[i, :] + np.random.rand(dimensions) * (P - I * X[i, :])  # Eq. (4)
            else:
                X_new = X[i, :] + np.random.rand(dimensions) * (X[i, :] - P)  # Eq. (4)

            X_new = np.clip(X_new, Lowerbound, Upperbound)  # Boundary handling

            # Update position based on Eq (5)
            fit_new = objective(X_new)
            if fit_new[i] < fit[i]:
                X[i, :] = X_new[i]
                fit[i] = fit_new[i]

            # Phase 2: Exploitation
            R = 0.02 * (1 - t / Max_iterations)  # Eq. (6)
            X_new = X[i, :] + (-R + 2 * R * np.random.rand(dimensions)) * X[i, :]  # Eq. (7)

            X_new = np.clip(X_new, Lowerbound, Upperbound)  # Boundary handling

            # Update position based on Eq (8)
            fit_new = objective(X_new)
            if fit_new[i] < fit[i]:
                X[i, :] = X_new[i]
                fit[i] = fit_new[i]

        # Save best score
        best_so_far = fbest  # Save best solution so far
        average = np.mean(fit)
        Score = fbest
        Best_pos = xbest
        NGO_curve[t] = Score
    ct = time.time() - ct
    return Score, NGO_curve, Best_pos, ct

