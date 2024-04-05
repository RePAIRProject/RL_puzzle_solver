
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from scipy.ndimage import rotate
from PIL import Image
import os
import configs.folder_names as fnames
import argparse
from compatibility.line_matching_NEW_segments import read_info
#import configs.solver_cfg as cfg
from puzzle_utils.pieces_utils import calc_parameters_v2, crop_to_content
from puzzle_utils.shape_utils import prepare_pieces_v2, create_grid, place_on_canvas
import datetime
import pdb 
import time 
import json 

class CfgParameters(dict):
    __getattr__ = dict.__getitem__

def initialization(R, anc, p_size=0):
    z0 = 0  # rotation for anchored patch
    # Initialize reconstruction plan
    no_grid_points = R.shape[0]
    no_patches = R.shape[3]
    no_rotations = R.shape[2]

    if p_size > 0:
        Y = p_size 
        X = Y 
    else:
        Y = round(no_grid_points * 2 + 1) 
        # Y = round(0.5 * (no_grid_points - 1) * (no_patches + 1) + 1)
        X = Y
    Z = no_rotations

    # initialize assignment matrix
    p = np.ones((Y, X, Z, no_patches)) / (Y * X)  # uniform
    init_pos = np.zeros((no_patches, 3)).astype(int)

    # place anchored patch (center)
    y0 = round(Y / 2)
    x0 = round(X / 2)

    p[:, :, :, anc] = 0
    p[y0, x0, :, :] = 0
    p[y0, x0, z0, anc] = 1
    init_pos[anc, :] = ([y0, x0, z0])

    print("P:", p.shape)
    return p, init_pos, x0, y0, z0


def RePairPuzz(R, p, na, cfg, verbosity=1, decimals=8):
    R = np.maximum(R, -1)
    R_new = R
    faze = 0
    new_anc = []
    na_new = na
    f = 0
    iter = 0
    eps = np.inf

    all_pay = []
    all_sol = []
    all_anc = []
    Y, X, Z, noPatches = p.shape

    # while not np.isclose(eps, 0)
    print("started solving..")
    while eps != 0 and iter < cfg.Tmax:
        if na_new > na:
            na = na_new
            faze += 1
            p = np.ones((Y, X, Z, noPatches)) / (Y * X)

            for jj in range(noPatches):
                if new_anc[jj, 0] != 0:
                    y = new_anc[jj, 0]
                    x = new_anc[jj, 1]
                    z = new_anc[jj, 2]
                    p[:, :, :, jj] = 0
                    p[y, x, :, :] = 0
                    p[y, x, z, jj] = 1

                    ## NEW: Re-normalization of R after anchoring
        #             for jj_anc in range(noPatches):
        #                 if new_anc[jj_anc, 0] != 0:
        #                     R_new[:, :, : , jj_anc, jj] = 0
        #
        # R_renorm = R_new / np.max(R_new)
        # R_new = np.where((R_new > 0), R_renorm*1.5, R_new)

        if faze == 0:
            T = cfg.Tfirst
        else:
            T = cfg.Tnext

        #pdb.set_trace()
        p, payoff, eps, iter = solver_rot_puzzle(R_new, R, p, T, iter, 0, verbosity=verbosity, decimals=decimals)
        
        I = np.zeros((noPatches, 1))
        m = np.zeros((noPatches, 1))

        for j in range(noPatches):
            pj_final = p[:, :, :, j]
            m[j, 0], I[j, 0] = np.max(pj_final), np.argmax(pj_final)

        I = I.astype(int)
        i1, i2, i3 = np.unravel_index(I, p[:, :, :, 1].shape)

        fin_sol = np.concatenate((i1, i2, i3), axis=1)
        if verbosity > 0:
            print("#" * 70)
            print("ITERATION", iter)
            print("#" * 70)
            print(np.concatenate((fin_sol, np.round(m * 100)), axis=1))

        #pdb.set_trace()
        if na < (noPatches-2):
            fix_tresh = cfg.anc_fix_tresh
        elif na > (noPatches-2):
            fix_tresh = 0.11   ## just fix last 2 pieces  !!!
        else:
            fix_tresh = 0.33   ## just fix last 2 pieces  !!!

        a = (m > fix_tresh).astype(int)
        new_anc = np.array(fin_sol*a)
        na_new = np.sum(a)
        # if verbosity > 0:
        #     print("#" * 70)
        #     print(f"fixed solution for a new piece (at iteration {iter}):")
        #     print(new_anc)
        f += 1
        all_pay.append(payoff[2:])
        all_sol.append(fin_sol)
        all_anc.append(new_anc)

    # if verbosity > 0:
    #     print("#" * 70)
    #     print("ITERATION", iter)
    #     print("#" * 70)
    #     print(np.concatenate((fin_sol, np.round(m * 100)), axis=1))
    # all_sol.append(fin_sol)
    p_final = p
    return all_pay, all_sol, all_anc, p_final, eps, iter, na_new


def solver_rot_puzzle(R, R_orig, p, T, iter, visual, verbosity=1, decimals=8):
    no_rotations = R.shape[2]
    no_patches = R.shape[3]
    payoff = np.zeros(T+1)
    z_st = 360 / no_rotations
    z_rot = np.arange(0, 360 - z_st + 1, z_st)
    t = 0
    eps = np.inf
    while t < T and eps > 0:
        t += 1
        iter += 1
        q = np.zeros_like(p)
        for i in range(no_patches):
            ri = R[:, :, :, :, i]
            #  ri = R[:, :, :, i, :]  # FOR ORACLE SQUARE ONLY
            for zi in range(no_rotations):
                rr = rotate(ri, z_rot[zi], reshape=False, mode='constant')
                rr = np.roll(rr, zi, axis=2)
                c1 = np.zeros(p.shape)
                for j in range(no_patches):
                    for zj in range(no_rotations):
                        rj_z = rr[:, :, zj, j]
                        pj_z = p[:, :, zj, j]
                        # cc = cv.filter2D(pj_z, -1, np.rot90(rj_z, 2)) # solves in inverse order !?!
                        cc = cv.filter2D(pj_z, -1, rj_z)
                        c1[:, :, zj, j] = cc

                q1 = np.sum(c1, axis=(2, 3))
                # q2 = (q1 != 0) * (q1 + no_patches * no_rotations * 0.5) ## new_experiment
                q2 = (q1 + no_patches * no_rotations * 1)
                q[:, :, zi, i] = q2

        pq = p * q  # e = 1e-11
        p_new = pq / (np.sum(pq, axis=(0, 1, 2)))
        p_new = np.where(np.isnan(p_new), 0, p_new)
        pay = np.sum(p_new * q)

        payoff[t] = pay
        eps = abs(pay - payoff[t-1])
        if verbosity > 1:
            if verbosity == 2:
                print(f'Iteration {t}: pay = {pay:.08f}, eps = {eps:.08f}', end='\r')
            else:
                print(f'Iteration {t}: pay = {pay:.08f}, eps = {eps:.08f}')
        p = np.round(p_new, decimals)
    return p, payoff, eps, iter

def reconstruct_puzzle_v2(solved_positions, Y, X, Z, pieces, ppars, use_RGB=True):

    if use_RGB:
        canvas_image = np.zeros((np.round(Y * ppars.xy_step + ppars.p_hs).astype(int), np.round(X * ppars.xy_step + ppars.p_hs).astype(int), 3))
    else:
        canvas_image = np.zeros((np.round(Y * ppars.xy_step + ppars.p_hs).astype(int), np.round(X * ppars.xy_step + ppars.p_hs).astype(int)))
    for i, piece in enumerate(pieces):
        target_pos = solved_positions[i,:2] * ppars.xy_step        
        #target_rot = solved_positions[i, 2] * ppars.theta_step ## ERR !!! - recalculate theta step in the case of few rotations
        theta_step = 360/Z
        target_rot = solved_positions[i, 2] * theta_step
        if (target_pos < ppars.p_hs).any() or (target_pos > canvas_image.shape).any() or (canvas_image.shape[0] - target_pos > ppars.p_hs).any():
            print("poorly placed piece, ignoring")
        else:
            placed_piece = place_on_canvas(piece, target_pos, canvas_image.shape[0], target_rot)

            if use_RGB:
                if len(placed_piece['img'].shape) > 2:
                    canvas_image += placed_piece['img']
                else:
                    canvas_image += np.repeat(placed_piece['img'], 3).reshape(canvas_image.shape)
            else:
                canvas_image += placed_piece['img']

    return canvas_image
                    
def reconstruct_puzzle(fin_sol, Y, X, Z, pieces, pieces_files, pieces_folder, ppars):
    step = np.ceil(ppars.xy_step)
    #ang = ppars.theta_step # 360 / Z    
    ang = 360 / Z
    z_rot = np.arange(0, 360, ang)
    pos = fin_sol
    fin_im = np.zeros(((Y * step + (ppars.p_hs+1) * 2).astype(int), (X * step + (ppars.p_hs+1) * 2).astype(int), 3))

    for i in range(len(pieces)):
        image = pieces_files[pieces[i]]  # read image 1
        im_file = os.path.join(pieces_folder, image)

        Im0 = Image.open(im_file).convert('RGBA')
        Im = np.array(Im0) / 255.0
        Im1 = Image.open(im_file).convert('RGBA').split()
        alfa = np.array(Im1[3]) / 255.0
        Im = np.multiply(Im, alfa[:, :, np.newaxis])
        Im = Im[:, :, 0:3]

        cc = ppars.p_hs
        ids = (pos[i, :2] * step + cc).astype(int)
        if pos.shape[1] == 3:
            rot = z_rot[pos[i, 2]]
            Im = rotate(Im, rot, reshape=False, mode='constant')
        fin_im[ids[0]-cc:ids[0]+cc, ids[1]-cc:ids[1]+cc, :] = Im+fin_im[ids[0]-cc:ids[0]+cc, ids[1]-cc:ids[1]+cc, :]
    return fin_im


def select_anchor(folder):
    pieces_files = os.listdir(folder)
    json_files = [piece_file for piece_file in pieces_files if piece_file[-4:] == 'json']
    json_files.sort()
    n = len(json_files)

    num_lines = np.zeros(n)
    for f in range(n):
        im = json_files[f]
        beta, R, s1, s2, b1, b2 = read_info(folder, im)
        num_lines[f] = len(beta)

    mean_num_lines = np.round(np.mean(num_lines))
    good_anchors = np.array(np.where(num_lines > mean_num_lines))
    new_anc = np.random.choice(good_anchors[0, :], 1)
    return new_anc[0]


#  MAIN
def main(args):

    print("Solver log\nSearch for `SOLVER_START_TIME` or `SOLVER_END_TIME` if you want to see which images are done")

    #print(os.getcwd())
    dataset_name = args.dataset
    puzzle_name = args.puzzle
    method = args.det_method

    print()
    print("-" * 50)
    print("-- SOLVER_START_TIME -- ")
    time_start_puzzle = time.time()
    # get the current date and time
    now = datetime.datetime.now()
    print(f"{now}\nStarted working on {puzzle_name}")
    print(f"Dataset: {args.dataset}")
    print("-" * 50)

    cfg = CfgParameters()
    # pieces
    cfg['Tfirst'] = args.tfirst
    cfg['Tnext'] = args.tnext
    cfg['Tmax'] = args.tmax
    cfg['anc_fix_tresh'] = args.thresh
    cfg['p_matrix_shape'] = args.p_pts
    cfg['cmp_cost'] = args.cmp_cost
    print('\tSOLVER PARAMETERS')
    for cfg_key in cfg.keys():
        print(f"{cfg_key}: {cfg[cfg_key]}")
    print("-" * 50)

    pieces_dict, img_parameters = prepare_pieces_v2(fnames, args.dataset, args.puzzle, verbose=True)
    puzzle_root_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset, args.puzzle)
    solver_patameters_path = os.path.join(puzzle_root_folder, 'solver_parameters.json')
    with open(solver_patameters_path, 'w') as spj:
        json.dump(cfg, spj, indent=3)
    print("saved json solver parameters file")

    cmp_parameter_path = os.path.join(puzzle_root_folder, 'compatibility_parameters.json')
    if os.path.exists(cmp_parameter_path):
        ppars = CfgParameters()
        with open(cmp_parameter_path, 'r') as cp:
            ppars_dict = json.load(cp)
        for ppk in ppars_dict.keys():
            ppars[ppk] = ppars_dict[ppk]
    else:
        print("\n" * 3)
        print("/" * 70)
        print("/\t***ERROR***\n/ compatibility_parameters.json not found!")
        print("/" * 70)
        print("\n" * 3)
        ppars = calc_parameters_v2(img_parameters, args.xy_step, args.xy_grid_points, args.theta_step)
    # ppars = calc_parameters_v2(img_parameters, args.xy_step, args.xy_grid_points, args.theta_step)

    if args.cmp_cost == 'LAP':
        # mat = loadmat(os.path.join(puzzle_root_folder,fnames.cm_output_name, f'CM_lines_{method}.mat'))
        mat = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_linesdet_{method}_cost_{args.cmp_cost}'))
    else:
        mat = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_linesdet_{method}_cost_{args.cmp_cost}'))
    pieces_folder = os.path.join(puzzle_root_folder, f"{fnames.pieces_folder}")

    ### HERE THE LINES WERE USED
    #only_lines_pieces_folder = os.path.join(puzzle_root_folder, f"{fnames.lines_output_name}", method, 'lines_only')
    #detect_output = os.path.join(puzzle_root_folder, f"{fnames.lines_output_name}", method)
    #pdb.set_trace()
    R = mat['R_line']

    pieces_files = os.listdir(pieces_folder)
    pieces_files.sort()
    # print(pieces_files)
    pieces_excl = []
    # pieces_excl = np.array([3,4,7,8,11,15]);
    all_pieces = np.arange(len(pieces_files))
    pieces = [p for p in all_pieces if p not in all_pieces[pieces_excl]]

    # pieces_incl = [p for p in np.arange(0, len(all_pieces)) if p not in pieces_excl]
    # R = R[:, :, :, pieces_incl, :]  # re-arrange R-matrix
    # R = R[:, :, :, :, pieces_incl]

    if args.few_rotations > 0:
        n_rot = R.shape[2]
        rot_incl = np.arange(0, n_rot, n_rot/args.few_rotations)
        rot_incl = rot_incl.astype(int)
        R = R[:, :, rot_incl, :, :]

    # HERE THE LINES WERE USED
    if args.anchor < 0:
        anc = np.random.choice(len(all_pieces))  # select_anchor(detect_output)
    else:
        anc = args.anchor
    print(f"Using anchor the piece with id: {anc}")

    p_initial, init_pos, x0, y0, z0 = initialization(R, anc, args.p_pts)  # (R, anc, anc_rot, nh, nw)
    # print(p_initial.shape)
    na = 1
    all_pay, all_sol, all_anc, p_final, eps, iter, na = RePairPuzz(R, p_initial, na, cfg, verbosity=args.verbosity, decimals=args.decimals)

    print("-" * 50)
    time_in_seconds = time.time()-time_start_puzzle
    if time_in_seconds > 100:
        time_in_minutes = (np.ceil(time_in_seconds / 60))
        if time_in_minutes < 60:
            print(f"Solving this puzzle took almost {time_in_minutes:.0f} minutes")
        else:
            time_in_hours = (np.ceil(time_in_minutes / 60))
            print(f"Solving this puzzle took almost {time_in_hours:.0f} hours")
    else:
        print(f"Solving this puzzle took {time_in_seconds:.0f} seconds")
    print("-" * 50)

    num_rot = p_initial.shape[2]
    solution_folder = os.path.join(puzzle_root_folder, f'{fnames.solution_folder_name}_anchor{anc}_{method}_cost_{args.cmp_cost}_rot{num_rot}')
    os.makedirs(solution_folder, exist_ok=True)
    print("Done! Saving in", solution_folder)

    # SAVE THE MATRIX BEFORE ANY VISUALIZATION
    filename = os.path.join(solution_folder, 'p_final')
    mdic = {"p_final": p_final, "label": "label", "anchor": anc, "anc_position": [x0, y0, z0]}
    savemat(f'{filename}.mat', mdic)
    np.save(filename, mdic)

    # VISUALIZATION
    f = len(all_sol)
    Y, X, Z, _ = p_final.shape
    fin_sol = all_sol[f-1]
    # pdb.set_trace()
    fin_im1 = reconstruct_puzzle(fin_sol, Y, X, Z, pieces, pieces_files, pieces_folder, ppars)
    
    # fin_im_v3 = reconstruct_puzzle_vis(fin_sol, pieces_folder, ppars, suffix='')
    # alternative method for reconstruction (with transparency on overlap because of b/w image)
    # fin_im_v2 = reconstruct_puzzle_v2(fin_sol, Y, X, pieces_dict, ppars, use_RGB=False)

    os.makedirs(solution_folder, exist_ok=True)
    final_solution = os.path.join(solution_folder, f'final_using_anchor{anc}.png')
    plt.figure(figsize=(16, 16))
    plt.title("Final solution including all piece")
    plt.imshow((fin_im1 * 255).astype(np.uint8))
    plt.tight_layout()
    plt.savefig(final_solution)
    plt.close()

    # fin_im_v2 = reconstruct_puzzle_v2(fin_sol, Y, X, Z, pieces_dict, ppars, use_RGB=True)
    # final_solution_v2 = os.path.join(solution_folder, f'final_using_anchor{anc}_overlap.png')
    # if np.max(fin_im_v2) > 1:
    #     fin_im_v2 = np.clip(fin_im_v2, 0, 1)
    # plt.imsave(final_solution_v2, fin_im_v2)
    # fin_im_cropped = crop_to_content(fin_im_v2)
    # final_solution_v2_cropped = os.path.join(solution_folder, f'final_using_anchor{anc}_overlap_cropped.png')
    # plt.imsave(final_solution_v2_cropped, fin_im_cropped)

    f = len(all_anc)
    fin_sol = all_anc[f-1]
    fin_im2 = reconstruct_puzzle(fin_sol, Y, X, Z, pieces, pieces_files, pieces_folder, ppars)

    final_solution_anchor = os.path.join(solution_folder, f'final_only_anchor_using_anchor{anc}.png')
    plt.figure(figsize=(16, 16))
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

    if args.save_frames is True:
        # intermediate steps
        frames_folders = os.path.join(solution_folder, 'frames_all')
        os.makedirs(frames_folders, exist_ok=True)

        for ff in range(f):
            frame_path = os.path.join(frames_folders, f"frame_{ff:05d}.png")
            cur_sol = all_sol[ff]
            im_rec = reconstruct_puzzle(cur_sol, Y, X, Z, pieces, pieces_files, pieces_folder)
            im_rec = np.clip(im_rec, 0, 1)
            plt.imsave(frame_path, im_rec)

        frames_folders = os.path.join(solution_folder, 'frames_anc')
        os.makedirs(frames_folders, exist_ok=True)

        for ff in range(f):
            frame_path = os.path.join(frames_folders, f"frame_{ff:05d}.png")
            cur_sol = all_anc[ff]
            im_rec = reconstruct_puzzle(cur_sol, Y, X, Z, pieces, pieces_files, pieces_folder)
            im_rec = np.clip(im_rec, 0, 1)
            plt.imsave(frame_path, im_rec)
    
    print("-" * 50)
    print("-- SOLVER_END_TIME -- ")
    # get the current date and time
    now = datetime.datetime.now()
    print(f"{now}")
    print(f'Done with {puzzle_name}\n')
    print("-" * 50)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='........ ')  # add some description
    parser.add_argument('--dataset', type=str, default='synthetic_pattern_pieces_from_DS_5_Dafne', help='dataset folder')
    parser.add_argument('--puzzle', type=str, default='image_00000_1', help='puzzle folder')
    parser.add_argument('--det_method', type=str, default='exact', help='method line detection')  # exact, manual, deeplsd
    parser.add_argument('--cmp_cost', type=str, default='LCI', help='cost computation')  # LAP, LCI
    parser.add_argument('--anchor', type=int, default=-1, help='anchor piece (index)')
    parser.add_argument('--save_frames', default=False, action='store_true', help='use to save all frames of the reconstructions')
    parser.add_argument('--verbosity', type=int, default=2, help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')
    parser.add_argument('--few_rotations', type=int, default=0, help='uses only few rotations to make it faster')
    parser.add_argument('--tfirst', type=int, default=1000, help='when to stop for multi-phase the first time (fix anchor, reset the rest)')
    parser.add_argument('--tnext', type=int, default=500, help='the step for multi-phase (each tnext reset)')
    parser.add_argument('--tmax', type=int, default=5000, help='the final number of iterations (it exits after tmax)')
    parser.add_argument('--thresh', type=float, default=0.75, help='a piece is fixed (considered solved) if the probability is above the thresh value (max .99)')
    parser.add_argument('--p_pts', type=int, default=20, help='the size of the p matrix (it will be p_pts x p_pts)')
    parser.add_argument('--decimals', type=int, default=8, help='decimal after comma when cutting payoff')
    args = parser.parse_args()

    main(args)



# def visualize_result(all_pay, all_sol, all_anc, init_pos, p_final, pieces, pieces_files, pieces_folder, ppars):
#     Y, X, Z, _ = p_final.shape
#     # init_im = reconstruct_toy9(init_pos, Y, X)
#     # init_im = reconstruct_group28_9(init_pos, Y, X, Z, pieces)
#     init_im = reconstruct_puzzle(init_pos, Y, X, pieces, pieces_files, pieces_folder, ppars)
#
#     faze = len(all_sol)
#     col = 2
#     row = faze
#     t = 1
#
#     plt.figure()
#     plt.subplot(col, row, t)
#     plt.imshow((init_im * 255).astype(np.uint8))
#
#     for f in range(faze - 1):
#         t += 1
#         new_anc = all_anc[f]
#         # faze_im = reconstruct_toy9(new_anc, Y, X)
#         # faze_im = reconstruct_group28_9(new_anc, Y, X, Z, pieces)
#         faze_im = reconstruct_puzzle(new_anc, Y, X, pieces, pieces_files, pieces_folder, ppars)
#         plt.subplot(col, row, t)
#         plt.imshow((faze_im * 255).astype(np.uint8))
#
#     for f in range(faze):
#         t += 1
#         fin_sol = all_sol[f]
#         if fin_sol.size != 0:
#             # faze_im = reconstruct_toy9(fin_sol, Y, X)
#             # faze_im = reconstruct_group28_9(fin_sol, Y, X, Z, pieces)
#             faze_im = reconstruct_puzzle(fin_sol, Y, X, pieces, pieces_files, pieces_folder, ppars)
#             plt.subplot(col, row, t)
#             plt.imshow((faze_im * 255).astype(np.uint8))
#     plt.show()
#
#     plt.figure()
#     plt.imshow((faze_im * 255).astype(np.uint8))
#     plt.show()
#
#     f_pay = []
#     for ff in range(faze):
#         a = all_pay[ff]
#         f_pay = np.append(f_pay, a)
#     f_pay = np.array(f_pay)
#     plt.figure()
#     plt.plot(f_pay, 'r', linewidth=1)
#     plt.show()

# def reconstruct_group28_9(fin_sol, Y, X, n_rot, pieces):
#     step = 38
#     #pieces = [p for p in pieces if p not in pieces[pieces_excl]]
#     ang = 360 / n_rot
#     z_rot = np.arange(0, 360, ang)
#     pos = fin_sol
#     fin_im = np.zeros((Y * step + 1000, X * step + 1000, 3))
#
#     for i in range(pos.shape[0]):
#         im_num = pieces[i]
#         in_file = f'C:/Users/Marina/PycharmProjects/WP3-PuzzleSolving/Compatibility/data/repair/group_28/ready/RPf_00{im_num}.png'
#         Im0 = Image.open(in_file).convert('RGBA')
#         Im = np.array(Im0) / 255.0
#         Im1 = Image.open(in_file).convert('RGBA').split()
#         alfa = np.array(Im1[3]) / 255.0
#         Im = np.multiply(Im, alfa[:, :, np.newaxis])
#         Im = Im[:,:,0:3]
#
#         id = pos[i, :2] * step - step + 500
#         if np.min(pos[i, :2]) > 0:
#             if pos.shape[1] == 3:
#                 rot = z_rot[pos[i, 2]]
#                 Im = rotate(Im, rot, reshape=False, mode='constant')
#
#             fin_im[id[0] - 500:id[0] + 500, id[1] - 500:id[1] + 500, :] = Im + fin_im[id[0] - 500:id[0] + 500,
#                                                                                id[1] - 500:id[1] + 500, :]
#     return fin_im
