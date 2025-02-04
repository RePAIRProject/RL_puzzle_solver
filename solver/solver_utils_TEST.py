
import numpy as np
import cv2 as cv
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from scipy.io import savemat, loadmat
from scipy.ndimage import rotate
from PIL import Image
import os
import configs.folder_names as fnames
import argparse
# from compatibility.line_matching_NEW_segments import read_info
from compatibility.utils import normalize_CM
# import configs.solver_cfg as cfg
from puzzle_utils.pieces_utils import calc_parameters_v2, crop_to_content
from puzzle_utils.shape_utils import prepare_pieces_v2, create_grid, place_on_canvas
import datetime
import pdb
import time
import json
from puzzle_utils.regions import combine_region_masks
from puzzle_utils.visualization import save_vis
import copy


def initialization_from_GT(anc, puzzle_root_folder, all_pieces, pieces_incl, no_rotations):
    border_points = 5  # xy_grid_points//10 ???

    init_pos = np.zeros((len(all_pieces), 3)).astype(int)
    gt_grid  = np.zeros((len(all_pieces), 3))

    # 1. load GT_grid
    import pandas as pd
    df = pd.read_csv(os.path.join(puzzle_root_folder, f'GT/gt_grid3.txt'))
    gt_grid[:, 0] = (df.loc[:, 'x'].values).astype(int)
    gt_grid[:, 1] = (df.loc[:, 'y'].values).astype(int)
    # gt_grid[:, 2] = (df.loc[:, 'rot'].values)

    # 1.1 include only pieces used
    gt_grid = gt_grid[pieces_incl, :]

    # 2. calculate optimal grid - p_matrix
    X = (np.max(gt_grid[:, 0]) + 2 * border_points).astype(int)
    Y = (np.max(gt_grid[:, 1]) + 2 * border_points).astype(int)
    Z = no_rotations

    # 3. create p_matrix (optimal_grid+border_points)
    p = np.ones((Y, X, Z, len(pieces_incl))) / (Y * X * Z)

    # 3. anchor position in GT_position+0.5*border_points
    # for anc in range(0, len(all_pieces)):
    x0 = (gt_grid[anc, 0] + border_points).astype(int)
    y0 = (gt_grid[anc, 1] + border_points).astype(int)
    z0 = 0

    p[:, :, :, anc] = 0
    p[y0, x0, :, :] = 0
    p[y0, x0, z0, anc] = 1
    init_pos[anc, :] = ([y0, x0, z0])

    print("P:", p.shape)
    return p, init_pos, x0, y0, z0


def initialization(R, anc, p_size_y=0, p_size_x=0, anc_pos=0):
    z0 = 0  # rotation for anchored patch
    # Initialize reconstruction plan
    no_grid_points = R.shape[0]
    no_patches = R.shape[3]
    no_rotations = R.shape[2]

    if p_size_y > 0:
        Y = p_size_y
        if p_size_x == 0:
            X = Y
        else:
            X = p_size_x
    else:
        Y = round(no_grid_points * np.sqrt(no_patches))  # + no_patches)
        X = Y
    Z = no_rotations

    # initialize assignment matrix
    p = np.ones((Y, X, Z, no_patches)) / (Y * X * Z)  # uniform
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


def save_vis_puzzle(fin_sol, Y, X, Z, saving_stuff, iter_num, show_borders=False):
    anc, pieces, pieces_files, pieces_folder, ppars, solver_visualization_folder = saving_stuff
    img = reconstruct_puzzle(fin_sol, Y, X, Z, anc, pieces, pieces_files, pieces_folder, ppars,
                             show_borders=show_borders)
    name = os.path.join(solver_visualization_folder, f'sol_it{iter_num:05d}.png')
    plt.imsave(name, crop_to_content(np.clip(img, 0, 1) * 255).astype(np.uint8))


def RePairPuzz(R, p, na, cfg, verbosity=1, decimals=8, save_each_phase=False, saving_stuff=[]):
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
            p = np.ones((Y, X, Z, noPatches)) / (Y * X * Z)  # OPTIONAL !!!
            for jj in range(noPatches):
                # if new_anc[jj, 0] != 0:
                if a[jj, 0] == 1:
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

        p, payoff, eps, iter = solver_rot_puzzle(R_new, R, p, T, iter, 0, verbosity=verbosity, decimals=decimals)

        I = np.zeros((noPatches, 1))
        m = np.zeros((noPatches, 1))

        for j in range(noPatches):
            pj_final = p[:, :, :, j]
            m[j, 0], I[j, 0] = np.max(pj_final), np.argmax(pj_final)

        I = I.astype(int)
        i1, i2, i3 = np.unravel_index(I, p[:, :, :, 1].shape)

        fin_sol = np.concatenate((i1, i2, i3), axis=1)
        if save_each_phase == True:
            save_vis_puzzle(fin_sol, Y, X, Z, saving_stuff, iter, show_borders=False)
        if verbosity > 0:
            print("#" * 70)
            print("ITERATION", iter)
            print("#" * 70)
            print(np.concatenate((fin_sol, np.round(m * 100)), axis=1))

        # pdb.set_trace()
        if na < (noPatches - 2):
            fix_tresh = cfg.anc_fix_tresh
        elif na > (noPatches - 2):
            fix_tresh = 0.11  ## just fix last 2 pieces  !!!
        else:
            fix_tresh = 0.33  ## just fix last 2 pieces  !!!

        a = (m > fix_tresh).astype(int)
        new_anc = np.array(fin_sol * a)
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


def reconstruct_puzzle(fin_sol, Y, X, Z, anc, pieces, pieces_files, pieces_folder, ppars, show_borders=False,                       exclude_overlapping_pieces=False):
    step = np.ceil(ppars.xy_step)
    # ang = ppars.theta_step # 360 / Z
    ang = 360 / Z
    z_rot = np.arange(0, 360, ang)
    pos = fin_sol
    fin_im = np.zeros(((Y * step + (ppars.p_hs + 1) * 2).astype(int), (X * step + (ppars.p_hs + 1) * 2).astype(int), 3))
    borders_cmap = mpl.colormaps['jet'].resampled(len(pieces))
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

        if np.sum(pos[i, :2]) > 0:

            ids = (pos[i, :2] * step + cc).astype(int)
            if pos.shape[1] == 3:
                rot = z_rot[pos[i, 2]]
                Im = rotate(Im, rot, reshape=False, mode='constant', order=0)

                if i == anc:
                    mask = (Im > 0.05).astype(np.uint8)
                    em = cv2.erode(mask, np.ones((5, 5)))
                    bordered_im = Im * em + (mask - em) * borders_cmap(i)[:3]
                    Im = bordered_im

                if show_borders == True:
                    mask = (Im > 0.05).astype(np.uint8)
                    em = cv2.erode(mask, np.ones((5, 5)))
                    bordered_im = Im * em + (mask - em) * borders_cmap(i)[:3]
                    Im = bordered_im
            if ppars.p_hs * 2 < ppars.piece_size:
                fin_im[ids[0] - cc:ids[0] + cc + 1, ids[1] - cc:ids[1] + cc + 1, :] = Im + fin_im[
                                                                                           ids[0] - cc:ids[0] + cc + 1,
                                                                                           ids[1] - cc:ids[1] + cc + 1,
                                                                                           :]
            else:
                fin_im[ids[0] - cc:ids[0] + cc, ids[1] - cc:ids[1] + cc, :] = Im + fin_im[ids[0] - cc:ids[0] + cc,
                                                                                   ids[1] - cc:ids[1] + cc, :]

        # if show_borders == True:
        #     plt.imshow(fin_im)
        #     breakpoint()
    return fin_im


def reconstruct_puzzle_v2(solved_positions, Y, X, Z, pieces, ppars, use_RGB=True):
    if use_RGB:
        canvas_image = np.zeros((np.round(Y * ppars.xy_step + ppars.p_hs).astype(int),
                                 np.round(X * ppars.xy_step + ppars.p_hs).astype(int), 3))
    else:
        canvas_image = np.zeros((np.round(Y * ppars.xy_step + ppars.p_hs).astype(int),
                                 np.round(X * ppars.xy_step + ppars.p_hs).astype(int)))
    for i, piece in enumerate(pieces):
        target_pos = solved_positions[i, :2] * ppars.xy_step
        # target_rot = solved_positions[i, 2] * ppars.theta_step ## ERR !!! - recalculate theta step in the case of few rotations
        theta_step = 360 / Z
        target_rot = solved_positions[i, 2] * theta_step
        if (target_pos < ppars.p_hs).any() or (target_pos > canvas_image.shape).any() or (
                canvas_image.shape[0] - target_pos > ppars.p_hs).any():
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


def sparsify_compatibility_matrix(R, k):
    # K-sparsification
    # k = 5
    best_scores_0rot = np.zeros((np.shape(R)[4], np.shape(R)[4]))
    best_scores = np.zeros((np.shape(R)[4], np.shape(R)[4]))
    for i in range(np.shape(R)[4]):
        for j in range(np.shape(R)[4]):
            if i != j:
                r_temp = R[:, :, :, j, i]
                a = np.min(np.partition(np.ravel(r_temp), -k)[-k:])
                r_neg = np.where(r_temp > -1, 0, -1)
                # r_neg = np.where(r_temp < 0, r_temp, 0)
                r_val = np.where(r_temp < a, 0, r_temp)
                R[:, :, :, j, i] = r_neg + r_val

                best_scores_0rot[j, i] = np.max(r_temp[:, :, 0])
                best_scores[j, i] = np.max(r_temp)
    return R
