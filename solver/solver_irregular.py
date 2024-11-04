
import numpy as np
import cv2 as cv
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from scipy.io import savemat, loadmat
from scipy.ndimage import rotate
from PIL import Image
import os
import configs.folder_names as fnames
import argparse
#from compatibility.line_matching_NEW_segments import read_info
#import configs.solver_cfg as cfg
from puzzle_utils.pieces_utils import calc_parameters_v2, crop_to_content
from puzzle_utils.shape_utils import prepare_pieces_v2, create_grid, place_on_canvas
import datetime
import pdb 
import time 
import json 
from puzzle_utils.visualization import save_vis
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
        Y = round(no_grid_points * np.sqrt(no_patches)) # + no_patches)
        #Y = round(no_grid_points * 2 + 1) 
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
            p = np.ones((Y, X, Z, noPatches)) / (Y*X*Z)

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
                    
def reconstruct_puzzle(fin_sol, Y, X, Z, pieces, pieces_files, pieces_folder, ppars, show_borders=False):
    step = np.ceil(ppars.xy_step)
    #ang = ppars.theta_step # 360 / Z    
    ang = 360 / Z
    z_rot = np.arange(0, 360, ang)
    pos = fin_sol
    fin_im = np.zeros(((Y * step + (ppars.p_hs+1) * 2).astype(int), (X * step + (ppars.p_hs+1) * 2).astype(int), 3))
    if show_borders == True:
        # plt.ion()
        borders_cmap = mpl.colormaps['jet'].resampled(len(pieces))
        # deprecated
        # borders_cmap = mpl.cm.get_cmap('jet').resampled(len(pieces))
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
            if show_borders == True:
                mask = (Im > 0.05).astype(np.uint8)
                em = cv2.erode(mask, np.ones((5,5)))
                bordered_im = Im * em + (mask-em) * borders_cmap(i)[:3]
                Im = bordered_im
        if ppars.p_hs*2 < ppars.piece_size:
            fin_im[ids[0] - cc:ids[0] + cc + 1, ids[1] - cc:ids[1] + cc + 1, :] = Im + fin_im[
                                                                                       ids[0] - cc:ids[0] + cc + 1,
                                                                                       ids[1] - cc:ids[1] + cc + 1, :]
        else:
            fin_im[ids[0]-cc:ids[0]+cc, ids[1]-cc:ids[1]+cc, :] = Im+fin_im[ids[0]-cc:ids[0]+cc, ids[1]-cc:ids[1]+cc, :]
        # if show_borders == True:
        #     plt.imshow(fin_im)
        #     breakpoint()
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
    lines_det_method = args.lines_det_method
    motif_det_method = args.motif_det_method

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
    cfg['cmp_type'] = args.cmp_type
    cfg['cmp_cost'] = args.cmp_cost
    cfg['combo_type'] = args.combo_type
    print('\tSOLVER PARAMETERS')
    for cfg_key in cfg.keys():
        print(f"{cfg_key}: {cfg[cfg_key]}")
    print("-" * 50)

    # pieces, img_parameters = prepare_pieces_v2(fnames, args.dataset, args.puzzle, verbose=True)  # commented for RePAIR test only !!!!
    puzzle_root_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset, args.puzzle)
    solver_patameters_path = os.path.join(puzzle_root_folder, 'solver_parameters.json')
    with open(solver_patameters_path, 'w') as spj:
        json.dump(cfg, spj, indent=3)
    print("saved json solver parameters file")

    cmp_parameter_path = os.path.join(puzzle_root_folder, 'compatibility_parameters_v2.json')
    if os.path.exists(cmp_parameter_path):
        ppars = CfgParameters()
        with open(cmp_parameter_path, 'r') as cp:
            ppars_dict = json.load(cp)
        for ppk in ppars_dict.keys():
            ppars[ppk] = ppars_dict[ppk]
    else:
        print("\n" * 3)
        print("/" * 70)
        print("/\t***ERROR***\n/ compatibility_parameters_v2.json not found!")
        print("/" * 70)
        print("\n" * 3)
        ppars = calc_parameters_v2(img_parameters, args.xy_step, args.xy_grid_points, args.theta_step)
    # ppars = calc_parameters_v2(img_parameters, args.xy_step, args.xy_grid_points, args.theta_step)

    if args.cmp_type == 'lines':
        cmp_name = f"linesdet_{args.lines_det_method}_cost_{args.cmp_cost}"
    elif args.cmp_type == 'shape':
        cmp_name = "shape"
    elif args.cmp_type == 'combo':
        cmp_name = f"cmp_combo{args.combo_type}"
    elif args.cmp_type == 'motifs':
        cmp_name = f"motifs_{args.motif_det_method}"
        # cmp_name = f"motifs_{args.det_method}_cost_{args.cmp_cost}"
    elif args.cmp_type == 'color':
        cmp_name = f"cmp_color"
        #cmp_name = f"color_border{args.border_len}"
    else:
        cmp_name = f"cmp_{args.cmp_type}"

    it_nums = f"{args.tmax}its"
    # if args.cmp_cost == 'LAP':
    #     # mat = loadmat(os.path.join(puzzle_root_folder,fnames.cm_output_name, f'CM_lines_{method}.mat'))
    #     mat = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_{cmp_name}'))
    # else:
    
    
    pieces_folder = os.path.join(puzzle_root_folder, f"{fnames.pieces_folder}")
    # check if we are combining!
    if args.cmp_type == 'combo':
        cmp_name = f"combo_{args.combo_type}"
        if args.combo_type == "SH-LIN":
            print("combining shape and lines..")
            mat_lines = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_linesdet_{args.lines_det_method}_cost_{args.cmp_cost}'))
            mat_shape = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_shape'))
            #breakpoint()
            R_lines = mat_lines['R']
            R_shape = mat_shape['R']
            negative_region_map = R_lines < 0

            # only positive values
            R = (np.clip(R_lines, 0, 1) * np.clip(R_shape, 0, 1))
            R /= np.max(R)
            # test
            R *= 2   ### !!!!! CHECK !?

            # negative values set to -1
            R[negative_region_map] = -1

        elif args.combo_type == 'SH-MOT':
            print("combining shape and motifs..")
            mat_motifs = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f"CM_motifs_{args.motif_det_method}"))
            mat_shape = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_shape'))
            # breakpoint()
            R_motif = mat_motifs['R']
            R_shape = mat_shape['R']
            negative_region_map = R_motif < 0

            # only positive values
            R = (np.clip(R_motif, 0, 1) * np.clip(R_shape, 0, 1))
            R /= np.max(R)
            # negative values set to -1
            R[negative_region_map] = -1

        elif args.combo_type == 'SH-SEG':
            print("combining shape and motifs..")
            mat_seg = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f"CM_cmp_seg"))
            mat_shape = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_shape'))
            # breakpoint()
            R_seg = mat_seg['R']
            R_shape = mat_shape['R']
            negative_region_map = R_seg < 0

            # only positive values
            R = (np.clip(R_seg, 0, 1) * np.clip(R_shape, 0, 1))
            R /= np.max(R)
            # negative values set to -1
            R[negative_region_map] = -1

        else:
            raise Exception(f"Please select another combo type, this ({args.combo_type}) has not been implemented yet")

    else:
        print("loading", os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_{cmp_name}'))
        mat = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_{cmp_name}'))
        # R = mat['R_line'] ## temp
        R = mat['R']

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
        ## optional
        # m, I = np.max(R), np.argmax(R)
        # ii = np.unravel_index(I, R.shape)
        # anc = ii[4]
    else:
        anc = args.anchor
    print(f"Using anchor the piece with id: {anc}")

    ## K-sparsification
    #k = 10
    k = args.k
    for i in range(np.shape(R)[4]):
        for j in range(np.shape(R)[4]):
            r_temp = R[:, :, :, j, i]
            m = np.max(r_temp)
            a = np.min(np.partition(np.ravel(r_temp), -k)[-k:])
            r_zer = np.where(r_temp > -1, 0, -1)
            r_val = np.where(r_temp < a, 0, r_temp)
            R[:, :, :, j, i] = r_zer + r_val

    ## esclude if incompatible
    if args.exclude == True:
        Rnew = R
        for i in range(np.shape(R)[4]):
            r_temp = R[:, :, :, :, i]
            m = np.max(r_temp)
            if m <= 0:
                Rnew = np.delete(Rnew, i, 4)
                Rnew = np.delete(Rnew, i, 3)
                anc=anc-1

        R = Rnew
        anc = np.max(anc,0)
    #####

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
    solution_folder = os.path.join(puzzle_root_folder, f'{fnames.solution_folder_name}_anchor{anc}_{cmp_name}_with{num_rot}rot_{it_nums}')
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
    fin_im1 = reconstruct_puzzle(fin_sol, Y, X, Z, pieces, pieces_files, pieces_folder, ppars, show_borders=False)
    fin_im1_brd = reconstruct_puzzle(fin_sol, Y, X, Z, pieces, pieces_files, pieces_folder, ppars, show_borders=True)
    
    # fin_im_v3 = reconstruct_puzzle_vis(fin_sol, pieces_folder, ppars, suffix='')
    # alternative method for reconstruction (with transparency on overlap because of b/w image)
    # fin_im_v2 = reconstruct_puzzle_v2(fin_sol, Y, X, pieces_dict, ppars, use_RGB=False)
    # breakpoint()
    os.makedirs(solution_folder, exist_ok=True)
    final_solution = os.path.join(solution_folder, f'final_using_anchor{anc}.png')
    #plt.figure(figsize=(16, 16))
    #plt.title("Final solution including all piece")
    plt.imshow((fin_im1 * 255).astype(np.uint8))
    plt.tight_layout()
    plt.savefig(final_solution)
    plt.close()
    plt.imsave(f"{final_solution[:-4]}_bordered.png", np.clip(fin_im1_brd, 0, 1))
    clean_img = fin_im1 * (fin_im1 > 0.1)
    plt.imsave(f"{final_solution[:-4]}_cropped.png", crop_to_content(clean_img * 255).astype(np.uint8))
    plt.imsave(f"{final_solution[:-4]}_bordered_cropped.png", crop_to_content(np.clip(fin_im1_brd, 0, 1) * 255).astype(np.uint8))
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
    parser.add_argument('--dataset', type=str, default='RePAIR_exp_batch2', help='dataset folder')
    parser.add_argument('--puzzle', type=str, default='RPobj_g1_o0001_gt_rot', help='puzzle folder')
    parser.add_argument('--lines_det_method', type=str, default='deeplsd', help='method line detection')  # exact, manual, deeplsd
    parser.add_argument('--motif_det_method', type=str, default='yolo-obb', help='method motif detection')  # exact, manual, deeplsd
    parser.add_argument('--cmp_cost', type=str, default='LCI', help='cost computation')  # LAP, LCI
    parser.add_argument('--anchor', type=int, default=2, help='anchor piece (index)')
    parser.add_argument('--save_frames', default=False, action='store_true', help='use to save all frames of the reconstructions')
    parser.add_argument('--exclude', default=False, action='store_true', help='use to exclude pieces without compatibility (used for some partial compatibilities, not fully tested!)')
    parser.add_argument('--verbosity', type=int, default=2, help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')
    parser.add_argument('--few_rotations', type=int, default=0, help='uses only few rotations to make it faster')
    parser.add_argument('--tfirst', type=int, default=750, help='when to stop for multi-phase the first time (fix anchor, reset the rest)')
    parser.add_argument('--tnext', type=int, default=250, help='the step for multi-phase (each tnext reset)')
    parser.add_argument('--tmax', type=int, default=1000, help='the final number of iterations (it exits after tmax)')
    parser.add_argument('--thresh', type=float, default=0.75, help='a piece is fixed (considered solved) if the probability is above the thresh value (max .99)')
    parser.add_argument('--p_pts', type=int, default=-1, help='the size of the p matrix (it will be p_pts x p_pts)')
    parser.add_argument('--decimals', type=int, default=10, help='decimal after comma when cutting payoff')
    parser.add_argument('--k', type=int, default=10, help='keep the best k values (for each pair) in the compatibility')
    parser.add_argument('--cmp_type', type=str, default='shape', help='which compatibility to use!', choices=['combo', 'lines', 'shape', 'color',  'motifs', 'seg'])
    parser.add_argument('--combo_type', type=str, default='SH-MOT',
        help='If `--cmp_type` is `combo`, it chooses which compatibility to use!\
            \nAbbreviations: (LIN=lines, MOT=motif, SH=shape, COL=color, SEG=segmentation)\
            \nFor example, SH-MOT is motif+shape, SH-SEG is shape+segmentation', 
        choices=['SH-SEG', 'SH-MOT', 'SH-LIN'])   
    parser.add_argument('--border_len', type=int, default=-1, help='length of border (if -1 [default] it will be set to xy_step)')

    args = parser.parse_args()

    main(args)

