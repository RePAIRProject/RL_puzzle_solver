import numpy as np
import scipy
import cv2
import warnings

def fix_anchors(P, num_anchors: int, threshold: float):

    N = P.shape[-2]

    grid_sol = extract_grid_sol_from_P(P)

    if threshold <= 1:
        threshold  = threshold * 100

    anchor_mask = (grid_sol[:,-2:-1] > threshold).astype(int)

    new_anc = np.array(grid_sol * anchor_mask)
    num_anchors_new = np.sum(anchor_mask)

    # if we have more anchors than before, we fix those, otherwise we keep running
    if num_anchors_new > num_anchors:
        num_anchors = num_anchors_new
        # uniform distribution
        P = np.ones_like(P) / (P.size()/N)

        for i in range(N):
            # if new_anc[i, 0] != 0:
            if anchor_mask[i, 0] == 1:
                y, x, theta = new_anc[i, :3]

                P[:, :, :, i] = 0
                P[y, x, :, :] = 0
                P[y, x, theta, i] = 1

    return P, grid_sol, num_anchors_new

def extract_grid_sol_from_P(P):

    N = P.shape[-1]
     
    I = np.zeros(N)
    score = np.zeros(N)

    for j in range(N):
        pj_final = P[:, :, :, j]
        # TODO: what about multiple maxima?
        score[j], I[j] = np.max(pj_final), np.argmax(pj_final)

    i_x, i_y, i_theta = np.unravel_index(I.astype(int), P[:, :, :, 1].shape)

    # TODO: Check this works as expected
    sol = np.transpose(np.stack((i_x, i_y, i_theta, np.round(score * 100))).astype(int))

    return sol



def solver_rot_puzzle(R, P, T, verbosity=1, decimals=8):
    """
    Solves the puzzle using Relaxation Labelling adapted to puzzle solving

    R : Compatibility Matrix (num_x_r,num_y_r,num_rot,N,N)
    p : Probability Matrix (num_x_p,num_y_p,num_rot,N)
    T : number iterations
    verbosity : logging verbosity
    decimals : precision of p
    """

    num_rot = R.shape[2]
    N = R.shape[3]
    payoff = np.zeros(T + 1)
    rot_step = 360 / num_rot
    rot_values = np.arange(0, 360 - rot_step + 1, rot_step)
    
    t = 0
    eps = np.inf
    
    while t < T and eps > 0:
        
        Q = np.zeros_like(P)
        
        # Compute support (q)
        for i in range(N):
            # 
            R_i = R[:, :, :, :, i]

            # alpha: rotation index of piece i
            for alpha_idx in range(num_rot):
                # apply a rotation of alpha to the matrix, is the same as applying a rotation -alpha to the input
                R_i_rotated = scipy.ndimage.rotate(R_i, rot_values[alpha_idx], reshape=False, mode='constant', order=0)
                # subtract -alpha from beta, but since the angle is periodic, we need to roll the matrix
                R_i_rotated = np.roll(R_i_rotated, alpha_idx, axis=2)
            
                Q_temp = np.zeros(P.shape)
                for j in range(N):

                    # maybe do not roll and do a 3D conv?

                    # This could be vectorized ?
                    # beta: rotation index of piece j
                    for beta_idx in range(num_rot):
                        R_ij_beta = R_i_rotated[:, :, beta_idx, j]
                        P_j_beta = P[:, :, beta_idx, j]
                        Q_temp[:, :, beta_idx, j] = cv2.filter2D(P_j_beta, -1, R_ij_beta)

                #Q_temp.shape = (num_x_p,num_y_p,num_rot,N)
                Q_i_alpha = np.sum(Q_temp, axis=(2, 3))

                
                Q[:, :, alpha_idx, i] = Q_i_alpha
        
        # Shift the support to get non-negative values
        Q += Q + N * 1

        PQ = P * np.exp(Q)  # e = 1e-11

        P_new = PQ / (np.sum(PQ, axis=(0, 1, 2))) # P(t+1)
        
        if np.isnan(P_new).sum() > 0: 
            warnings.warn("P has NaN values! Setting them to 0")
            P_new = np.where(np.isnan(P_new), 0, P_new)

        payoff[t] = np.sum(P_new * Q)

        eps = abs(payoff[t] - payoff[t - 1])
        if verbosity > 1:
            if verbosity == 2:
                print(f'Iteration {t}: pay = {payoff[t]:.08f}, eps = {eps:.08f}', end='\r')
            else:
                print(f'Iteration {t}: pay = {payoff[t]:.08f}, eps = {eps:.08f}')

        # Rounding P changes the dynamics
        P = np.round(P_new, decimals)
        
        t += 1

    return P, payoff, eps