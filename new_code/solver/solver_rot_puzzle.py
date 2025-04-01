import numpy as np
import scipy
import cv2
import warnings

def solver_rot_puzzle(R, R_orig, P, T, iter, visual, verbosity=1, decimals=8):
    """
    R : Compatibility Matrix (num_x_r,num_y_r,num_rot,N,N)
    R_orig : Not used!
    p : Probability Matrix (num_x_p,num_y_p,num_rot,N)
    T : number iterations
    iter : total iteration counter (to be removed)
    visual : to be removed
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
        iter += 1 # to be removed

    return P, payoff, eps, iter