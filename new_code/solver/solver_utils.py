
import numpy as np
import cv2 as cv
import matplotlib

#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from scipy.io import savemat, loadmat
from scipy.ndimage import rotate
from PIL import Image
import os
import argparse
# from compatibility.line_matching_NEW_segments import read_info
# from compatibility.compatibiliy_utils import normalize_CM
# import configs.solver_cfg as cfg
# from puzzle_utils.pieces_utils import calc_parameters_v2
# from puzzle_utils.shape_utils import prepare_pieces_v2, create_grid, place_on_canvas, 
from utils.visualization_utils import crop_to_content
import datetime
import pdb
import time
import json
# from puzzle_utils.regions import combine_region_masks
# from puzzle_utils.visualization import save_vis
import copy

def compute_pixel_solution(grid_solution: np.ndarray, grid_xy_step : int, grid_theta_step : int):

    pixel_solution = np.zeros_like(grid_solution)
    # multiply x,y by xy_step
    pixel_solution[:,:2] = grid_solution[:,:2] * grid_xy_step
    # multiply theta by theta_step
    pixel_solution[:,2] = grid_solution[:,2] * grid_theta_step
    # leave the probabilities as they are
    pixel_solution[:,3] = grid_solution[:,3]
    # SWAP y and x before returning as "pixel"
    pixel_solution.T[[0, 1]] = pixel_solution.T[[1, 0]]
    return pixel_solution


def initialize_p_from_GT(anc, puzzle_root_folder, all_pieces, pieces_incl, no_rotations):
    border_points = 20  # xy_grid_points//10 ???

    init_pos = np.zeros((len(all_pieces), 3)).astype(int)
    gt_grid  = np.zeros((len(all_pieces), 3))

    # 1. load GT_grid
    import pandas as pd
    df = pd.read_csv(os.path.join(puzzle_root_folder, f'GT/gt_grid3.txt'))
    #df = pd.read_csv(os.path.join(puzzle_root_folder, f'GT/gt_px251.txt'))
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
    anchor_pos = [y0, x0, z0]
    print("P:", p.shape)
    return p, init_pos, anchor_pos

def get_pieces_id_list(anchor_idx:int, adjacency_matrix:np.ndarray, max_adjacency_degree:int):
    """
    Given the anchor index, the adjacency matrix and a maximum degrees, it creates a list of the pieces id which are "neighbours" of rank <= of the max degree.
    It is used to recover a subset of the puzzle formed by neighbours. The higher the max_degree, the more pieces it will select. 
    The anchor is included in the list.
    - max_adjacency_degree = 0 -> anchor alone 
    - max_adjacency_degree = 1 -> only direct neighbours  
    - max_adjacency_degree = 2 -> direct neighbours and their respective neighbours 
    and so on..
    """
    pieces_list = [anchor_idx]
    if max_adjacency_degree == 0:
        return pieces_list
    elif max_adjacency_degree == 1:
        for adj_pair in adjacency_matrix:
            if anchor_idx in adj_pair:
                pieces_list.append(adj_pair[0])
                pieces_list.append(adj_pair[1]) # should add only the "other" id, but it seems easier to add everything and remove duplicates afterwards
    else:
        raise NotImplementedError("We need to iteratively add the other pieces ids!\nSince it is not used in this experiment, it was not yet implemented")

    # Check https://stackoverflow.com/questions/57261950/how-does-set-remove-duplicates-from-a-list
    # pieces_list = set(pieces_list)                  # it orders the ids
    pieces_list = list(dict.fromkeys(pieces_list))  # leaves the same order, with anchor at the beginning. Is it better?

    return pieces_list 

def initialize_p_using_neighbours_with_occupancy(R, anchor_idx: int, pieces_occupancy_grid:np.ndarray, adjacency_matrix:np.ndarray, max_adjacency_degree:int):
    """
    Initializing the P matrix choosing only a subset of pieces. 
    This is designed to test a hierarchical/multi-step method, it will nNOTot solve the whole puzzle.
    It implements the occupancy grid variant, already removing the points occupied by the anchor piece in the P matrix.
    """
    pieces_subset_id_list = get_pieces_id_list(anchor_idx, adjacency_matrix, max_adjacency_degree)
    print("using only pieces:", pieces_subset_id_list)
    for k in range(R.shape[3]):
        if k not in pieces_subset_id_list:
            # set to zero since we will not be using these pieces!
            R[:,:,:,k,:] = 0
            R[:,:,:,:,k] = 0

    P, init_pieces_pos, anchor_pos = initialize_p_with_occupancy(R, anchor_idx, pieces_occupancy_grid=pieces_occupancy_grid)
    return P, init_pieces_pos, anchor_pos, pieces_subset_id_list

def initialize_p_with_occupancy(R, anchor_idx, pieces_occupancy_grid=None):
    """
    Initializing the P matrix and already removing the points occupied by the anchor piece
    """
    num_pieces = R.shape[3]
    X = Y = round(R.shape[0] * np.sqrt(num_pieces))  # + no_patches)
    Z = R.shape[2]

    P = np.ones((Y, X, Z, num_pieces)) / (Y * X * Z)  # uniform
    init_pieces_pos = np.zeros((num_pieces, 3)).astype(int)

    # place anchored patch (center)
    z0 = 0 # should we allow different rotations? This means the anchor is placed without rotation
    y0 = round(Y / 2)
    x0 = round(X / 2)
    P[:, :, :, anchor_idx] = 0
    P[y0, x0, :, :] = 0
    P[y0, x0, z0, anchor_idx] = 1
    # occupancy
    anchor_pos = [y0, x0, z0]
    # plt.subplot(121)
    # plt.imshow(P[:,:,0,anchor_idx])
    anchor_mask = np.zeros((num_pieces, 1), dtype=int)
    anchor_mask[anchor_idx] = 1
    P = remove_occupied_grid_points(P, piece_pos=anchor_pos, piece_id=anchor_idx, piece_occ=pieces_occupancy_grid[anchor_idx,:,:], anchor_mask=anchor_mask)
    
    # for j in range(num_pieces):
    #     plt.subplot(4,4,j+1)
    #     plt.title(f"P matrix for piece {j}")
    #     plt.imshow(P[:,:,0,j])
    # plt.show()
    # breakpoint()
    init_pieces_pos[anchor_idx, :] = anchor_pos
    anchor_pos = [y0, x0, z0]
    
    return P, init_pieces_pos, anchor_pos


def remove_occupied_grid_points(P: np.ndarray, piece_pos:np.array, piece_id:int, piece_occ: np.ndarray, anchor_mask: np.ndarray) -> np.ndarray:
    """
    When fixing a piece on the P matrix, we remove (=set to 0) the nearby points on the grid
    """
    # occ is on rotation 0
    rotation_idx = piece_pos[2]
    x = piece_pos[0]
    y = piece_pos[1]
    if rotation_idx > 0:
        # raise NotImplementedError("Need to fix the rotation step")
        rot_step = 360 // P.shape[3] # we could pass the parameters here
        rotated_occ = rotate(piece_occ, rotation_idx * rot_step, reshape=False, mode='constant', order=0)
    else:
        rotated_occ = piece_occ
    po_hs = piece_occ.shape[0] // 2
    # set to zero everywhere where the occ grid has 1
    # for all of the other pieces 
    # (so they cannot overlap with the anchor)
    x_offset_m = y_offset_m = x_offset_M = y_offset_M = 0 # These are to avoid going out of P when the piece is on some borders
    if np.min(np.asarray(piece_pos[:2]) - po_hs) < 0:
        x_offset_m = np.maximum(po_hs - piece_pos[0], 0)
        y_offset_m = np.maximum(po_hs - piece_pos[1], 0)
        rotated_occ = rotated_occ[y_offset_m:, x_offset_m:]
        # print("\nCASE 1")
        # print(f"Occ:{rotated_occ.shape}")
        # print(f"P:{P[piece_pos[1]-po_hs+y_offset_m:piece_pos[1]+po_hs+1-y_offset_M, piece_pos[0]-po_hs+x_offset_m:piece_pos[0]+po_hs+1-x_offset_M, rotation_idx, piece_id].shape}")
        
    if np.max(np.asarray(piece_pos[:2]) + po_hs) > np.min(P.shape[:2]):
        x_offset_M = np.maximum(piece_pos[0] + po_hs + 1 - P.shape[0], 0)
        y_offset_M = np.maximum(piece_pos[1] + po_hs + 1 - P.shape[1], 0)
        rotated_occ = rotated_occ[:rotated_occ.shape[1]-y_offset_M, :rotated_occ.shape[0]-x_offset_M]
        # print("\nCASE 2")
        # print(f"Occ:{rotated_occ.shape}")
        # print(f"P:{P[piece_pos[1]-po_hs+y_offset_m:piece_pos[1]+po_hs+1-y_offset_M, piece_pos[0]-po_hs+x_offset_m:piece_pos[0]+po_hs+1-x_offset_M, rotation_idx, piece_id].shape}")

    
    for p_id in range(P.shape[3]):
        if p_id != piece_id and anchor_mask[p_id] == 0:
            # print(f"setting P matrix for piece {p_id}")
            # plt.imshow(P[:,:,rotation_idx, p_id])
            # plt.title(f"P matrix for piece {p_id} BEFORE (fixing={piece_id})")
            # plt.show()
            P[piece_pos[1]-po_hs+y_offset_m:piece_pos[1]+po_hs+1-y_offset_M, piece_pos[0]-po_hs+x_offset_m:piece_pos[0]+po_hs+1-x_offset_M, rotation_idx, p_id] -= rotated_occ
            # plt.imshow(P[:,:,rotation_idx, p_id])
            # plt.title(f"P matrix for piece {p_id} AFTER (fixing={piece_id})")
            # plt.show()
    P = np.clip(P, 0, 1)
    # plt.figure()
    # plt.suptitle(anchor_mask)
    # for j in range(P.shape[3]):
    #     plt.subplot(4,4,j+1)
    #     plt.title(f"P matrix for piece {j}")
    #     plt.imshow(P[:,:,0,j])
    # plt.show()
    # breakpoint()
    # keep the center of the piece to 1
    # P[piece_pos[0], piece_pos[1], piece_pos[2]] = 1

    return P


def initialize_p(R, anc, p_size_x=0, p_size_y=0, anc_pos=0):
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
        #Y = round(no_grid_points * (no_patches+1)) # + no_patches)
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
    anchor_pos = [y0, x0, z0]
    print("P:", p.shape)
    return p, init_pos, anchor_pos


def save_vis_puzzle(fin_sol, Y, X, Z, saving_stuff, iter_num, show_borders=False):
    anc, pieces, pieces_files, pieces_folder, ppars, solver_visualization_folder = saving_stuff
    img = reconstruct_puzzle(fin_sol, Y, X, Z, anc, pieces, pieces_files, pieces_folder, ppars,
                             show_borders=show_borders)
    name = os.path.join(solver_visualization_folder, f'sol_it{iter_num:05d}.png')
    plt.imsave(name, crop_to_content(np.clip(img, 0, 1) * 255).astype(np.uint8))


def reconstruct_puzzle(fin_sol, Y, X, Z, anc, pieces, pieces_files, pieces_folder, ppars, show_borders=False, exclude_overlapping_pieces=False):
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
