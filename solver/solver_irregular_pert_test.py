
import numpy as np
import cv2 as cv
# import matplotlib
# matplotlib.use('TkAgg')
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from scipy.ndimage import rotate
from PIL import Image
import os
import configs.folder_names as fnames
import argparse
#from compatibility.line_matching_NEW_segments import read_info
#import configs.solver_cfg as cfg
from puzzle_utils.pieces_utils import calc_parameters_v2
from puzzle_utils.shape_utils import prepare_pieces_v2, place_on_canvas
import datetime
import time 
import json 

class CfgParameters(dict):
    __getattr__ = dict.__getitem__


#def initialization_from_gt(args.sigma, args.sigma, args.dataset, args.puzzle, args.anchor, args.p_pts):
def initialization_from_gt(args):
    ## inputs for probability distribution
    sigma_x = args.sigma
    sigma_y = args.sigma
    anchor_id = args.anchor    # default=5

    dataset_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset)
    print(dataset_folder)
    puzzle = args.puzzle

    puzzle_folder = os.path.join(dataset_folder, puzzle)
    general_files = os.listdir(puzzle_folder)
    with open(os.path.join(puzzle_folder, "ground_truth.json"), 'r') as gtj:
        ground_truth = json.load(gtj)
    with open(os.path.join(puzzle_folder, f"parameters_{puzzle}.json"), 'r') as gtj:
        img_parameters = json.load(gtj)
    with open(os.path.join(puzzle_folder, f"compatibility_parameters.json"), 'r') as gtj:
        cmp_parameters = json.load(gtj)

    num_pcs = img_parameters['num_pieces']
    theta_step = cmp_parameters['theta_step']
    xy_step = cmp_parameters['xy_step']
    no_rotations = cmp_parameters['theta_grid_points']

    gt_coord = np.zeros([num_pcs, 3])
    for j in range(num_pcs):
        gt_coord[j, 2] = (+1) * ground_truth[f"piece_{j:04d}"]['rotation']
        gt_coord[j, 0:2] = (-1) * np.asarray(ground_truth[f"piece_{j:04d}"]['translation'][::-1])

    # 1. GT - shifted and rotated in PX (anchor is in ref.point = [0,0,0])
    anc_coord = gt_coord[anchor_id, 0:2]
    anc_rot = gt_coord[anchor_id, 2]
    gt_shift = np.zeros([num_pcs, 3])
    gt_shift[:, 0:2] = gt_coord[:, 0:2] - anc_coord
    gt_shift[:, 2] = gt_coord[:, 2] - anc_rot

    all_rot = gt_shift[:, 2]
    all_rot = np.where(all_rot < 0, all_rot + 360, all_rot)
    gt_shift[:, 2] = all_rot

    # 2. Translate to yxz space
    gt_yxz = np.zeros([num_pcs, 3])
    gt_yxz[:, 0:2] = np.round(gt_shift[:, 0:2] / xy_step)
    gt_yxz[:, 2] = gt_shift[:, 2] / theta_step

    # 3. Rotate GT according anchor rotation - CHECK!!!
    gt_rot = np.zeros([num_pcs, 3])
    gt_rot[:, 2] = gt_yxz[:, 2]
    if anc_rot == 0:
        gt_rot[:, 0:2] = gt_yxz[:, 0:2]
    else:
        if anc_rot == 90 or anc_rot == -270:
            gt_rot[:, 0] = gt_yxz[:, 1]   # y_new = +x
            gt_rot[:, 1] = -gt_yxz[:, 0]  # x_new = -y
        elif anc_rot == -90 or anc_rot == 270:
            gt_rot[:, 0] = -gt_yxz[:, 1]  # y_new = -x
            gt_rot[:, 1] = gt_yxz[:, 0]   # x_new = +y##
        elif anc_rot == 180 or anc_rot == -180:
            gt_rot[:, 0] = -gt_yxz[:, 0]  # y_new = -x
            gt_rot[:, 1] = -gt_yxz[:, 1]  # x_new = -y

    ##### NEW PART STARTS HERE !!!!!!
    # Shift ALL coord to the center of the Reconstruction plane !!!!
    grid_size = args.p_pts
    p = np.zeros((grid_size, grid_size, no_rotations, num_pcs))
    p_zyx = np.zeros((grid_size, grid_size, no_rotations, num_pcs))

    center_grid = round(grid_size/2)
    anc_position = [center_grid, center_grid, 0]
    probability_centers = np.zeros([num_pcs, 3])
    probability_centers[:, 0:2] = gt_rot[:, 0:2] + anc_position[0:2]
    probability_centers[:, 2] = gt_rot[:, 2]

    yy, xx = np.mgrid[0:grid_size:1, 0:grid_size:1]
    pos = np.dstack((yy, xx))

    pos_all = np.reshape(pos, (grid_size * grid_size, -1))
    pos3 = []
    for t in range(no_rotations):
        tt = t
        if t == 3:  # must be changed !!! igf num_rotation > 4
            tt = 1  # hardcoded rotation -90 = 270, anchor is always rotated 0
        pos_t = np.ones([pos_all.shape[0], 3])*tt
        pos_t[:, 0:2] = pos_all
        if tt == 0:
            pos3 = pos_t
        else:
            pos3 = np.concatenate((pos3, pos_t), axis=0)

    cov3 = [[sigma_y, 0, 0], [0, sigma_x, 0], [0, 0, 1]]
    # cov2 = [[sigma_y, 0], [0, sigma_x]]

    for j in range(num_pcs):
        # mu2 = probability_centers[j, 0:2]
        # rv2 = multivariate_normal(mu2, cov2)
        # p_norm_j = rv2.pdf(pos)
        # for t in range(no_rotations):
        #     if probability_centers[j,2] == t:
        #         p[:, :, t, j] = p_norm_j

        mu3 = probability_centers[j, :]
        rv2 = multivariate_normal(mu3, cov3)
        p_norm3_j = rv2.pdf(pos3)
        p_zyx_j = np.reshape(p_norm3_j, (4, grid_size, grid_size))
        p_reshape_j = np.moveaxis(p_zyx_j, 0, -1)
        p[:, :, :, j] = p_reshape_j

    init_pos = np.zeros((num_pcs, 3)).astype(int)
    init_pos[anchor_id, :] = anc_position

    p[:, :, :, anchor_id] = 0
    p[anc_position[0], anc_position[1], :, :] = 0
    p[anc_position[0], anc_position[1], anc_position[2], anchor_id] = 1

    return p, init_pos, anc_position, probability_centers


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
    anc_position = [y0, x0, z0]

    print("P:", p.shape)
    return p, init_pos, anc_position


def RePairPuzz(R, p_initial, na, cfg, verbosity=1, decimals=8):
    R = np.maximum(R, -1)
    p = p_initial
    na_new = na
    faze = 0
    iter = 0
    eps = np.inf
    new_anc = []
    all_pay = []
    all_sol = []
    all_anc = []
    Y, X, Z, noPatches = p_initial.shape

    # while not np.isclose(eps, 0)
    print("started solving..")
    while eps != 0 and iter < cfg.Tmax:
        if na_new > na:
            na = na_new
            faze += 1
            #p = p_initial   ## Optional !!! - OR - not re-initialize
            for jj in range(noPatches):
                if new_anc[jj, 0] != 0:
                    y = new_anc[jj, 0]
                    x = new_anc[jj, 1]
                    z = new_anc[jj, 2]
                    p[:, :, :, jj] = 0
                    p[y, x, :, :] = 0
                    p[y, x, z, jj] = 1

        if faze == 0:
            T = cfg.Tfirst
        else:
            T = cfg.Tnext
        p, payoff, eps, iter = solver_rot_puzzle(R, p, T, iter, 0, verbosity=verbosity, decimals=decimals)
        
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
        all_pay.append(payoff[2:])
        all_sol.append(fin_sol)
        all_anc.append(new_anc)

    return all_pay, all_sol, all_anc, p, eps, iter, na_new

def solver_rot_puzzle(R, p, T, iter, visual, verbosity=1, decimals=8):
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
        if ppars.p_hs*2 < ppars.piece_size:
            fin_im[ids[0] - cc:ids[0] + cc + 1, ids[1] - cc:ids[1] + cc + 1, :] = Im + fin_im[
                                                                                       ids[0] - cc:ids[0] + cc + 1,
                                                                                       ids[1] - cc:ids[1] + cc + 1, :]
        else:
            fin_im[ids[0]-cc:ids[0]+cc, ids[1]-cc:ids[1]+cc, :] = Im+fin_im[ids[0]-cc:ids[0]+cc, ids[1]-cc:ids[1]+cc, :]

    return fin_im

#  MAIN
def main(args):

    print("Solver log\nSearch for `SOLVER_START_TIME` or `SOLVER_END_TIME` if you want to see which images are done")

    method = args.det_method
    print()
    print("-" * 50)
    print("-- SOLVER_START_TIME -- ")
    time_start_puzzle = time.time()
    # get the current date and time
    now = datetime.datetime.now()
    print(f"{now}\nStarted working on {args.puzzle}")
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

    pieces_dict, img_parameters = prepare_pieces_v2(fnames, args.dataset, args.puzzle, verbose=True)  # commented for RePAIR test only !!!!
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

    if args.cmp_cost == 'LAP':
        # mat = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_lines_{method}_cost_{args.cmp_cost}'))
        mat = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_linesdet_{method}_cost_{args.cmp_cost}'))
    else:
        mat = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_linesdet_{method}_cost_{args.cmp_cost}'))
        #mat = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_lines_{method}_cost_{args.cmp_cost}'))
    R = mat['R_line']

    pieces_folder = os.path.join(puzzle_root_folder, f"{fnames.pieces_folder}")
    pieces_files = os.listdir(pieces_folder)
    pieces_files.sort()
    pieces = np.arange(len(pieces_files))

    if args.few_rotations > 0:
        n_rot = R.shape[2]
        rot_incl = np.arange(0, n_rot, n_rot/args.few_rotations)
        rot_incl = rot_incl.astype(int)
        R = R[:, :, rot_incl, :, :]

    # HERE THE LINES WERE USED
    # only_lines_pieces_folder = os.path.join(puzzle_root_folder, f"{fnames.lines_output_name}", method, 'lines_only')
    # detect_output = os.path.join(puzzle_root_folder, f"{fnames.lines_output_name}", method)

    if args.anchor < 0:
        anc = np.random.choice(len(pieces))  # select_anchor(detect_output)
    else:
        anc = args.anchor
    print(f"Using anchor the piece with id: {anc}")
    #p_initial, init_pos, anc_position = initialization(R, anc, args.p_pts)

    p_initial, init_pos, anc_position, probability_centers = initialization_from_gt(args)

    num_rot = p_initial.shape[2]
    solution_folder = os.path.join(puzzle_root_folder,
                                   f'{fnames.solution_folder_name}_pert_test_rot{num_rot}')
    os.makedirs(solution_folder, exist_ok=True)
    ##########################
    Y, X, Z, _ = p_initial.shape
    image_from_gt = reconstruct_puzzle(probability_centers.astype(int), Y, X, Z, pieces, pieces_files, pieces_folder, ppars)
    image_from_gt = np.clip(image_from_gt, 0, 1)
    initial_solution = os.path.join(solution_folder, f'gt_recostruct{anc}.png')
    plt.imsave(initial_solution, image_from_gt)
    #################àà

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


    print("Done! Saving in", solution_folder)

    # SAVE THE MATRIX BEFORE ANY VISUALIZATION
    filename = os.path.join(solution_folder, f'p_final_sigma{args.sigma}')
    mdic = {"p_final": p_final, "label": "label", "anchor": anc, "anc_position": anc_position}
    savemat(f'{filename}.mat', mdic)
    np.save(filename, mdic)

    # VISUALIZATION
    f = len(all_sol)
    Y, X, Z, _ = p_final.shape
    fin_sol = all_sol[f-1]
    # pdb.set_trace()
    fin_im1 = reconstruct_puzzle(fin_sol, Y, X, Z, pieces, pieces_files, pieces_folder, ppars)

    os.makedirs(solution_folder, exist_ok=True)
    final_solution = os.path.join(solution_folder, f'final_using_anchor{anc}_sigma{args.sigma}.png')
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

    final_solution_anchor = os.path.join(solution_folder, f'final_only_anchor_using_anchor{anc}_sigma{args.sigma}.png')
    plt.figure(figsize=(16, 16))
    plt.title("Final solution including ONLY solved pieces")
    plt.imshow((fin_im2 * 255).astype(np.uint8))
    plt.tight_layout()
    plt.savefig(final_solution_anchor)
    plt.close()

    alc_path = os.path.join(solution_folder, f'alc_plot_sigma{args.sigma}.png')
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
        frames_folders = os.path.join(solution_folder, f'frames_all_sigma{args.sigma}')
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
    print(f'Done with {args.puzzle}\n')
    print("-" * 50)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='........ ')  # add some description
    parser.add_argument('--dataset', type=str, default='synthetic_pattern_pieces_from_DS_5_Dafne_10px', help='dataset folder')
    parser.add_argument('--puzzle', type=str, default='image_00000_1', help='puzzle folder')
    parser.add_argument('--det_method', type=str, default='exact', help='method line detection')  # exact, manual, deeplsd
    parser.add_argument('--cmp_cost', type=str, default='LAP', help='cost computation')  # LAP, LCI
    parser.add_argument('--save_frames', default=False, action='store_true', help='use to save all frames of the reconstructions')
    parser.add_argument('--verbosity', type=int, default=2, help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')
    parser.add_argument('--few_rotations', type=int, default=0, help='uses only few rotations to make it faster')

    parser.add_argument('--tfirst', type=int, default=500, help='when to stop for multi-phase the first time (fix anchor, reset the rest)')
    parser.add_argument('--tnext', type=int, default=100, help='the step for multi-phase (each tnext reset)')
    parser.add_argument('--tmax', type=int, default=1800, help='the final number of iterations (it exits after tmax)')
    parser.add_argument('--thresh', type=float, default=0.55, help='a piece is fixed (considered solved) if the probability is above the thresh value (max .99)')
    parser.add_argument('--p_pts', type=int, default=15, help='the size of the p matrix (it will be p_pts x p_pts)')
    parser.add_argument('--decimals', type=int, default=8, help='decimal after comma when cutting payoff')
    parser.add_argument('--anchor', type=int, default=1, help='anchor piece (index)')
    parser.add_argument('--sigma', type=int, default=2, help='norm_dist_sigma same for x and y, mu=GT; z assume to have uniform_dist')

    args = parser.parse_args()

    main(args)

