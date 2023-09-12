
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from scipy.ndimage import rotate, shift
from PIL import Image


def initialization(R, anc, anc_rot):
    # Initialize reconstruction plan
    nh = 1.8
    nw = 1.8  # %% para to decide
    no_patches = R.shape[3]
    st = R.shape[0]
    Y = round(nh * (st - 1) + 1)
    X = round(nw * (st - 1) + 1)
    Z = R.shape[2]

    # initialize assigment matrix
    y0 = round(Y/2)
    x0 = round(X/2)  # position for anchored patch (center)
    z0 = anc_rot  # rotation for anchored patch
    init_pos = np.zeros((no_patches, 3)).astype(int)
    init_pos[anc, :] = ([y0, x0, z0])
    p = np.ones((Y, X, Z, no_patches)) / (Y * X)
    p[:, :, :, anc] = 0
    p[y0, x0, :, :] = 0
    p[y0, x0, z0, anc] = 1  # anchor selected patch

    return p, init_pos

def RePairPuzz(R, p, anc_fix_tresh, Tfirst, Tnext, Tmax):
    na = 1
    fase = 0
    new_anc = []
    na_new = na
    f = 0
    iter = 0
    eps = np.inf
    all_pay = []
    all_sol = []
    all_anc = []
    Y, X, Z, noPatches = p.shape

    while eps != 0 and iter < Tmax:
        if na_new > na:
            na = na_new
            fase += 1
            p = np.ones((Y, X, Z, noPatches)) / (Y * X)
            for jj in range(noPatches):
                if new_anc[jj, 0] != 0:
                    y = new_anc[jj, 0]
                    x = new_anc[jj, 1]
                    z = new_anc[jj, 2]
                    p[:, :, :, jj] = 0
                    p[y, x, :, :] = 0
                    p[y, x, z, jj] = 1
        if fase == 0:
            T = Tfirst
        else:
            T = Tnext
        p, payoff, eps, iter = solver_rot_puzzle(R, p, T, iter, 0)

        I = np.zeros((noPatches, 1))
        m = np.zeros((noPatches, 1))

        for j in range(noPatches):
            pj_final = p[:, :, :, j]
            m[j, 0], I[j, 0] = np.max(pj_final), np.argmax(pj_final)

        I = I.astype(int)
        i1, i2, i3 = np.unravel_index(I, pj_final.shape)

        fin_sol = np.concatenate((i1, i2, i3), axis=1)
        print(np.concatenate((fin_sol, np.round(m * 100)), axis=1))

        a = (m > anc_fix_tresh).astype(int)
        new_anc = np.array(fin_sol*a)
        na_new = np.sum(a)
        print(new_anc)

        f += 1
        all_pay.append(payoff)
        if na_new > na:
            all_sol.append(fin_sol)
            all_anc.append(new_anc)
        p_final = p

    #all_sol[fase]=fin_sol ##
    all_sol.append(fin_sol)
    return all_pay, all_sol, all_anc, p_final, eps, iter

def solver_rot_puzzle(R, p, T, iter, visual):
    Z = R.shape[2]
    no_patches = R.shape[3]
    #all_p = [None] * T
    payoff = np.zeros(T+1)
    z_st = 360 / Z
    z_rot = np.arange(0, 360 - z_st + 1, z_st)
    t = 0
    eps = np.inf
    while t < T and eps > 0:
        t += 1
        iter += 1
        q = np.zeros_like(p)
        for i in range(no_patches):
            ri = R[:, :, :, :, i]
            #ri = R[:, :, :, i, :] #FOR ORACLE SQUARE ONLY !!!!!
            for zi in range(Z):
                rr = rotate(ri, z_rot[zi], reshape=False, mode='constant') #CHECK ??? method?? senso antiorario!!!
                rr = np.roll(rr, zi, axis=2) # matlab: rr = circshift(rr,zi-1,3); z1=0!!!
                c1 = np.zeros(p.shape)
                for j in range(no_patches):
                    for zj in range(Z):
                        rj_z = rr[:, :, zj, j]
                        pj_z = p[:, :, zj, j]
                        #cc = cv.filter2D(pj_z,-1, np.rot90(rj_z, 2)) #solves inverse order ??? - wrong!!
                        cc = cv.filter2D(pj_z, -1, rj_z)
                        c1[:, :, zj, j] = cc;

                q1 = np.sum(c1, axis=(2, 3))
                q2 = (q1 != 0) * (q1 + no_patches * Z * 0.5) ##new_experiment
                #q2 = (q1 + no_patches * Z * 0.5)
                q[:, :, zi, i] = q2
        pq = p * q
        p_new = pq / np.sum(pq, axis=(0, 1, 2))
        pay = np.sum(p_new * q)
        #pay = np.sum(pq)
        payoff[t] = pay
        eps = abs(pay - payoff[t-1])
        print(t)
        print(pay)
        print(eps)
        p = np.round(p_new, 8)
        #if visual == 1:
            #all_p[t] = p

    #all_p = all_p[0:t, 0]
    #payoff = payoff[1:t, 0]
    return p, payoff, eps, iter #, all_p

def visualize_result(all_pay, all_sol, all_anc, init_pos, p_final, pieces):
    Y, X, Z, _ = p_final.shape
    # init_im = reconstruct_toy9(init_pos, Y, X)
    init_im = reconstruct_group28_9(init_pos, Y, X, Z, pieces)

    fase = len(all_sol)  # fase = all_sol.shape[1]
    col = 2
    row = fase
    t = 1

    plt.figure()
    plt.subplot(col, row, t)
    plt.imshow((init_im * 255).astype(np.uint8))

    for f in range(fase - 1):
        t += 1
        new_anc = all_anc[f]
        # fase_im = reconstruct_toy9(new_anc, Y, X)
        fase_im = reconstruct_group28_9(new_anc, Y, X, Z, pieces)
        plt.subplot(col, row, t)
        plt.imshow((fase_im * 255).astype(np.uint8))

    for f in range(fase):
        t += 1
        fin_sol = all_sol[f]
        if fin_sol.size != 0:
            # fase_im = reconstruct_toy9(fin_sol, Y, X)
            fase_im = reconstruct_group28_9(fin_sol, Y, X, Z, pieces)
            plt.subplot(col, row, t)
            plt.imshow((fase_im * 255).astype(np.uint8))
    plt.show()

    plt.figure()
    plt.imshow((fase_im * 255).astype(np.uint8))
    plt.show()

    f_pay = []
    for ff in range(fase):
        a = all_pay[ff]
        f_pay = np.append(f_pay, a)
    f_pay = np.array(f_pay)
    plt.figure()
    plt.plot(f_pay, 'r', linewidth=1)
    plt.show()

def reconstruct_group28_9(fin_sol, Y, X, n_rot, pieces):
    step = 38
    #pieces = [p for p in pieces if p not in pieces[pieces_excl]]
    ang = 360 / n_rot
    z_rot = np.arange(0, 360, ang)
    pos = fin_sol
    fin_im = np.zeros((Y * step + 1000, X * step + 1000, 3))

    for i in range(pos.shape[0]):
        im_num = pieces[i]
        in_file = f'C:/Users/Marina/PycharmProjects/WP3-PuzzleSolving/Compatibility/data/repair/group_28/ready/RPf_00{im_num}.png'
        Im0 = Image.open(in_file).convert('RGBA')
        Im = np.array(Im0) / 255.0
        Im1 = Image.open(in_file).convert('RGBA').split()
        alfa = np.array(Im1[3]) / 255.0
        Im = np.multiply(Im, alfa[:, :, np.newaxis])
        Im = Im[:,:,0:3]

        id = pos[i, :2] * step - step + 500
        if np.min(pos[i, :2]) > 0:
            if pos.shape[1] == 3:
                rot = z_rot[pos[i, 2]]
                #Im = np.array(Image.fromarray(Im).rotate(rot, expand=True))
                Im = rotate(Im, rot, reshape=False, mode='constant')

            fin_im[id[0] - 500:id[0] + 500, id[1] - 500:id[1] + 500, :] = Im + fin_im[id[0] - 500:id[0] + 500,
                                                                               id[1] - 500:id[1] + 500, :]
    return fin_im

## MAIN ##

RP_group = 28
Group_im_path = f'C:\\Users\\Marina\\PycharmProjects\\WP3-PuzzleSolving\\Compatibility\\data\\repair\\group_{RP_group}\\ready\\'
Rmat_in_file = 'C:\\Users\\Marina\\Toy_Puzzle_Matlab\\R_line51_45_verLAP_fake2.mat'

mat = scipy.io.loadmat('C:\\Users\\Marina\\Toy_Puzzle_Matlab\\R_line51_45_verLAP_fake2.mat')
R = mat['R_line2']

#mat = scipy.io.loadmat('C:\\Users\\Marina\\Toy_Puzzle_Matlab\\RR_rot2.mat')
#R = mat['R']

## select rotation
#R = R[:, :, [0, 2, 4, 6], :, :]
#R = R[:, :, [0, 4], :, :]

#pieces_excl = np.array([]);
pieces_excl = np.array([1,4]);
#pieces_excl = np.array([1,2,4,5,6]);
all_pieces = np.concatenate((np.arange(194, 199), np.arange(200, 204)))
pieces = [p for p in all_pieces if p not in all_pieces[pieces_excl]]

pieces_incl = [p for p in np.arange(0, len(all_pieces)) if p not in pieces_excl]
R = R[:, :, :, pieces_incl, :] ## re-arange Rmatrix
R = R[:, :, :, :, pieces_incl]

# PARA
anc = 5
anc_rot = 0
anc_fix_tresh = 0.5
Tfirst = 500
Tnext = 100
Tmax = 5000

p_initial, init_pos = initialization(R, anc, anc_rot)
all_pay, all_sol, all_anc, p_final, eps, iter = RePairPuzz(R, p_initial, anc_fix_tresh, Tfirst, Tnext, Tmax)

## visualize results (copied from function)
f = len(all_sol)
Y, X, Z, _ = p_final.shape
fin_sol = all_sol[f-1]
fase_im = reconstruct_group28_9(fin_sol, Y, X, Z, pieces)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.imshow((fase_im * 255).astype(np.uint8))
plt.show()

f_pay = []
for ff in range(f):
    a = all_pay[ff]
    f_pay = np.append(f_pay, a)
f_pay = np.array(f_pay)
plt.figure(figsize=(6, 6))
plt.plot(f_pay, 'r', linewidth=1)
plt.show()

####  TO DO  ####
#fin_sol, fase_im = visualize_result(all_pay, all_sol, all_anc, init_pos, p_final, pieces)
#plot_Lines_fin_sol(new_anc, Y, X, pieces, pieces_excl, Z, fase_im)
