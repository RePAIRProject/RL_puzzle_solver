
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from scipy.ndimage import rotate, shift
from PIL import Image
import os
import configs.unified_cfg as cfg
import configs.folder_names as fnames
import argparse


def initialization(R, anc):  # (R, anc, anc_rot, nh, nw):
    # Initialize reconstruction plan
    no_patches = R.shape[3]

    # # Re-Pair
    # st = R.shape[0]
    # Y = round(cfg.nh * (st - 1) + 1)
    # X = round(cfg.nw * (st - 1) + 1)
    # Z = R.shape[2]

    # # Toy Puzzle (with o without initial anchor)

    n_side = cfg.num_patches_side
    #n_side = np.round(R.shape[4]**(1/2))
    Y = n_side * 2 - 1 
    X = n_side * 2 - 1
    Z = R.shape[2]

    # initialize assigment matrix
    p = np.ones((Y, X, Z, no_patches)) / (Y * X) # uniform distributed p
    init_pos = np.zeros((no_patches, 3)).astype(int)

    # place initial anchor
    y0 = n_side - 1 # round(Y / 2)
    x0 = n_side - 1 # round(X / 2)  # position for anchored patch (center)
    z0 = cfg.init_anc_rot  # rotation for anchored patch
    p[:, :, :, anc] = 0
    p[y0, x0, :, :] = 0
    p[y0, x0, z0, anc] = 1  # anchor selected patch
    init_pos[anc, :] = ([y0, x0, z0])

    return p, init_pos, x0, y0, z0

def RePairPuzz(R, p, na, verbosity=1):  # (R, p, anc_fix_tresh, Tfirst, Tnext, Tmax):
    #na = 1
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

    while eps != 0 and iter < cfg.Tmax:
        if na_new > na:
            na = na_new
            fase += 1
            p = np.ones((Y, X, Z, noPatches)) / (Y * X)
            if iter>3000:
                p = p+cfg.pert_noise
            for jj in range(noPatches):
                if new_anc[jj, 0] != 0:
                    y = new_anc[jj, 0]
                    x = new_anc[jj, 1]
                    z = new_anc[jj, 2]
                    p[:, :, :, jj] = 0
                    p[y, x, :, :] = 0
                    p[y, x, z, jj] = 1
        if fase == 0:
            T = cfg.Tfirst
        else:
            T = cfg.Tnext
        p, payoff, eps, iter = solver_rot_puzzle(R, p, T, iter, 0, verbosity=verbosity)

        I = np.zeros((noPatches, 1))
        m = np.zeros((noPatches, 1))

        for j in range(noPatches):
            pj_final = p[:, :, :, j]
            m[j, 0], I[j, 0] = np.max(pj_final), np.argmax(pj_final)

        I = I.astype(int)
        i1, i2, i3 = np.unravel_index(I, pj_final.shape)

        fin_sol = np.concatenate((i1, i2, i3), axis=1)
        if verbosity > 0:
            print("#" * 70)
            print("ITERATION", iter)
            print("#" * 70)
            print(np.concatenate((fin_sol, np.round(m * 100)), axis=1))
        else:
            if iter % 1000 == 0:
                print("iteration", iter)
        a = (m > cfg.anc_fix_tresh).astype(int)
        new_anc = np.array(fin_sol*a)
        na_new = np.sum(a)
        if verbosity > 0:
            print(new_anc)

        f += 1
        all_pay.append(payoff)
        all_sol.append(fin_sol)
        all_anc.append(new_anc)
        # if na_new > na:
        #     all_sol.append(fin_sol)
        #     all_anc.append(new_anc)
        p_final = p

    #all_sol[fase]=fin_sol ##
    all_sol.append(fin_sol)
    return all_pay, all_sol, all_anc, p_final, eps, iter, na_new

def solver_rot_puzzle(R, p, T, iter, visual, verbosity=1):
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
            for zi in [0]: #range([Z]):
                rr = rotate(ri, z_rot[zi], reshape=False, mode='constant') #CHECK ??? method?? senso antiorario!!!
                rr = np.roll(rr, zi, axis=2) # matlab: rr = circshift(rr,zi-1,3); z1=0!!!
                c1 = np.zeros(p.shape)
                for j in range(no_patches):
                    for zj in range(Z):
                        rj_z = rr[:, :, zj, j]
                        pj_z = p[:, :, zj, j]
                        # cc = cv.filter2D(pj_z,-1, np.rot90(rj_z, 2)) # solves inverse order ??? - wrong!!
                        cc = cv.filter2D(pj_z, -1, rj_z)
                        c1[:, :, zj, j] = cc;

                q1 = np.sum(c1, axis=(2, 3))
                # q2 = (q1 != 0) * (q1 + no_patches * Z * 0.5) ## new_experiment
                q2 = (q1 + no_patches * Z * 1)  # *0.5
                q[:, :, zi, i] = q2
        pq = p * q
        e = 1e-11
        p_new = pq / (np.sum(pq, axis=(0, 1, 2)))
        p_new = np.where(np.isnan(p_new), 0, p_new)
        pay = np.sum(p_new * q)
        #pay = np.sum(pq)
        payoff[t] = pay
        eps = abs(pay - payoff[t-1])
        if verbosity > 0:
            if verbosity == 1:
                print(f'Iteration {t}: pay = {pay:.05f}, eps = {eps:.05f}', end='\r')
            else:
                print(f'Iteration {t}: pay = {pay:.05f}, eps = {eps:.05f}')
        p = np.round(p_new, 8)
        # if visual == 1:
        #     all_p[t] = p
        # all_p = all_p[0:t, 0]
        # payoff = payoff[1:t, 0]
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

def reconstruct_puzzle(fin_sol, Y, X, pieces, pieces_files, pieces_folder):
    step = np.ceil(cfg.xy_step)
    ang = 360 / cfg.theta_grid_points
    z_rot = np.arange(0, 360, ang)

    pos = fin_sol
    fin_im = np.zeros(((Y * step + cfg.p_hs).astype(int), (X * step + cfg.p_hs).astype(int), 3))

    for i in range(len(pieces)):
        image = pieces_files[pieces[i]]  # read image 1
        im_file = os.path.join(pieces_folder, image)

        Im0 = Image.open(im_file).convert('RGBA')
        Im = np.array(Im0) / 255.0
        Im1 = Image.open(im_file).convert('RGBA').split()
        alfa = np.array(Im1[3]) / 255.0
        Im = np.multiply(Im, alfa[:, :, np.newaxis])
        Im = Im[:, :, 0:3]

        cc = cfg.p_hs
        ids = (pos[i, :2] * step + cc).astype(int)
        if pos.shape[1] == 3:
            rot = z_rot[pos[i, 2]]
            Im = rotate(Im, rot, reshape=False, mode='constant')
        fin_im[ids[0] - cc:ids[0] + cc + 1, ids[1] - cc:ids[1] + cc + 1, :] = Im + fin_im[
                                                                                   ids[0] - cc:ids[0] + cc + 1,
                                                                                   ids[1] - cc:ids[1] + cc + 1, :]
                                                                           
        # if np.min(pos[i, :2]) > 0:
        #     if pos.shape[1] == 3:
        #         rot = z_rot[pos[i, 2]]
        #         Im = rotate(Im, rot, reshape=False, mode='constant')
        #     fin_im[ids[0] - cc:ids[0] + cc + 1, ids[1] - cc:ids[1] + cc + 1, :] = Im + fin_im[
        #                                                                                ids[0] - cc:ids[0] + cc + 1,
        #                                                                                ids[1] - cc:ids[1] + cc + 1, :]

    return fin_im


## MAIN ##
def main(args):
    ## MAIN ##

    dataset_name = args.dataset
    puzzle_name = args.puzzle
    method = args.method
    num_pieces = args.pieces

    mat = scipy.io.loadmat(os.path.join(f"{fnames.output_dir}_{cfg.num_patches_side}x{cfg.num_patches_side}", dataset_name, puzzle_name,fnames.cm_output_name, f'CM_lines_{method}_p{args.penalty}.mat'))
    #mat = scipy.io.loadmat(f'C:\\Users\Marina\PycharmProjects\RL_puzzle_solver\output\\{dataset_name}\\{puzzle_name}\compatibility_matrix\\CM_lines_deeplsd_p0.mat')
    pieces_folder = os.path.join(f"{fnames.output_dir}_{cfg.num_patches_side}x{cfg.num_patches_side}", dataset_name, puzzle_name, f"{fnames.pieces_folder}")
    only_lines_pieces_folder = os.path.join(f"{fnames.output_dir}_{cfg.num_patches_side}x{cfg.num_patches_side}", dataset_name, puzzle_name, f"{fnames.lines_output_name}", method, 'lines_only')
    #pieces_folder = os.path.join(f'C:\\Users\Marina\PycharmProjects\RL_puzzle_solver\output\\{dataset_name}\\{puzzle_name}\pieces')
    R = mat['R_line']

    pieces_files = os.listdir(pieces_folder)
    pieces_files.sort()
    pieces_excl = []
    # pieces_excl = np.array([3,4,7,8,11,15]);
    all_pieces = np.arange(len(pieces_files))
    pieces = [p for p in all_pieces if p not in all_pieces[pieces_excl]]

    pieces_incl = [p for p in np.arange(0, len(all_pieces)) if p not in pieces_excl]
    R = R[:, :, :, pieces_incl, :] ## re-arange Rmatrix
    R = R[:, :, :, :, pieces_incl]

    R = R[:, :, [0,1], :, :]  # select rotation

    if args.anchor < 0:
        anc = cfg.init_anc
    else:
        anc = args.anchor

    p_initial, init_pos, x0, y0, z0 = initialization(R, anc)  #(R, anc, anc_rot, nh, nw)
    na = 1
    all_pay, all_sol, all_anc, p_final, eps, iter, na = RePairPuzz(R, p_initial, na, verbosity=args.verbosity) #(R, p_initial, anc_fix_tresh, Tfirst, Tnext, Tmax)

    ##

    ## visualize results
    f = len(all_sol)
    Y, X, Z, _ = p_final.shape
    fin_sol = all_sol[f-1]
    fin_im1 = reconstruct_puzzle(fin_sol, Y, X, pieces, pieces_files, pieces_folder)

    #solution_folder = os.path.join(f'C:\\Users\Marina\PycharmProjects\RL_puzzle_solver\output_8x8\\{dataset_name}\\{puzzle_name}\solution')
    #os.makedirs(solution_folder, exist_ok=True)
    solution_folder = os.path.join(f"{fnames.output_dir}_{cfg.num_patches_side}x{cfg.num_patches_side}", dataset_name, puzzle_name, f'{fnames.solution_folder_name}_anchor{args.anchor}')
    # _pen{args.penalty}')
    os.makedirs(solution_folder, exist_ok=True)
    final_solution = os.path.join(solution_folder, f'final_using_anchor{args.anchor}.png')
    plt.figure(figsize=(16, 16))
    plt.title("Final solution including all piece")
    plt.imshow((fin_im1 * 255).astype(np.uint8))
    plt.tight_layout()
    plt.savefig(final_solution)
    plt.close()

    f = len(all_anc)
    fin_sol = all_anc[f-1]
    fin_im2 = reconstruct_puzzle(fin_sol, Y, X, pieces, pieces_files, pieces_folder)

    final_solution_anchor = os.path.join(solution_folder, f'final_only_anchor_using_anchor{args.anchor}.png')
    plt.figure(figsize=(16,16))
    plt.title("Final solution including ONLY solved pieces")
    plt.imshow((fin_im2 * 255).astype(np.uint8))
    plt.tight_layout()
    plt.savefig(final_solution_anchor)
    plt.close()

    alc_path = os.path.join(solution_folder, 'alc_plot.png')
    f = len(all_pay)
    f_pay = []
    for ff in range(f):
        a = all_pay[ff]
        f_pay = np.append(f_pay, a)
    f_pay = np.array(f_pay)
    plt.figure(figsize=(6, 6))
    plt.plot(f_pay, 'r', linewidth=1)
    plt.tight_layout()
    plt.savefig(alc_path)

    filename = os.path.join(solution_folder, 'p_final')
    mdic = {"p_final": p_final, "label": "label", "anchor": anc, "anc_position": [x0, y0, z0]}
    scipy.io.savemat(f'{filename}.mat', mdic)
    np.save(filename, mdic)

    if args.save_frames is True:
        # intermediate steps
        frames_folders = os.path.join(solution_folder, 'frames_all')
        os.makedirs(frames_folders, exist_ok=True)

        for ff in range(f):
            frame_path = os.path.join(frames_folders, f"frame_{ff:05d}.png")
            cur_sol = all_sol[ff]
            im_rec = reconstruct_puzzle(cur_sol, Y, X, pieces, pieces_files, pieces_folder)
            im_rec = np.clip(im_rec,0,1)
            plt.imsave(frame_path, im_rec)


        frames_folders = os.path.join(solution_folder, 'frames_anc')
        os.makedirs(frames_folders, exist_ok=True)

        for ff in range(f):
            frame_path = os.path.join(frames_folders, f"frame_{ff:05d}.png")
            cur_sol = all_anc[ff]
            im_rec = reconstruct_puzzle(cur_sol, Y, X, pieces, pieces_files, pieces_folder)
            im_rec = np.clip(im_rec,0,1)
            plt.imsave(frame_path, im_rec)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='........ ')  # add some discription
    parser.add_argument('--dataset', type=str, default='manual_lines', help='dataset folder')   # repair, wikiart, manual_lines, architecture
    parser.add_argument('--puzzle', type=str, default='lines1', help='puzzle folder')           # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('--penalty', type=int, default=20, help='penalty value used')                 # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('--method', type=str, default='deeplsd', help='method used for compatibility')                 # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('--pieces', type=int, default=4, help='number of pieces (per side)')                 # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('--anchor', type=int, default=-1, help='anchor piece (index)')                 # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('--save_frames', default=False, action='store_true', help='use to save all frames of the reconstructions')
    parser.add_argument('--verbosity', type=int, default=1, help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')                 # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life

    args = parser.parse_args()

    main(args)
