import numpy as np
from solver.solverRotPuzzArgs import reconstruct_puzzle
import os 
import pdb 
import cv2 

def get_sol_from_p(p_final):
    noPatches = p_final.shape[3]
    I = np.zeros((noPatches, 1))
    m = np.zeros((noPatches, 1))
    for j in range(noPatches):
        p_tmp_final = p_final[:, :, :, j]
        m[j, 0], I[j, 0] = np.max(p_tmp_final), np.argmax(p_tmp_final)

    I = I.astype(int)
    i1, i2, i3 = np.unravel_index(I, p_tmp_final.shape)

    fin_sol = np.concatenate((i1, i2, i3), axis=1)
    return fin_sol 

def get_offset(anchor_idx, anchor_pos, num_pieces):
    anchor_in_puzzle_Y = anchor_idx % num_pieces
    anchor_in_puzzle_X = anchor_idx // num_pieces
    anchor_pos_in_puzzle = np.asarray([anchor_in_puzzle_Y, anchor_in_puzzle_X])
    offset_start = anchor_pos[:2] - anchor_pos_in_puzzle
    return offset_start

def get_visual_solution_from_p(p_final, pieces_folder, piece_size, offset_start, num_pieces_side, crop=True):
    
    # reconstruct visual solution
    Y, X, Z, _ = p_final.shape
    pieces_files = os.listdir(pieces_folder)
    pieces_files.sort()
    pieces_excl = []
    all_pieces = np.arange(len(pieces_files))
    pieces = [p for p in all_pieces if p not in all_pieces[pieces_excl]]
    sol = get_sol_from_p(p_final=p_final)
    solution_img = reconstruct_puzzle(sol, Y, X, pieces, pieces_files, pieces_folder)
    start_point = (offset_start)*piece_size
    end_point = start_point + (num_pieces_side)*piece_size
    if crop is True:
        solution_img = solution_img[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    return solution_img

def get_best_anchor(im_ref, evaluation, num_pieces, piece_size, best='min'):
    
    if best=='min':
        best_idx = np.argmin(evaluation)
        best_val = np.min(evaluation)
    elif best=='max':
        best_idx = np.argmax(evaluation)
        best_val = np.max(evaluation)
    else:
        print("What is best?")
        print('choose sorting order <min> or <max>')
        return 0
    
    best_pos = get_xy_position(best_idx, num_pieces)
    anc_center = (best_pos) * piece_size
    best_anchor = im_ref[anc_center[1]:anc_center[1] + piece_size, anc_center[0]:anc_center[0]+piece_size]

    return best_anchor, best_idx, best_pos, best_val

def get_true_solution_vector(num_pieces):

    num_labels = np.round(np.square(num_pieces)).astype(int)
    true_solutions = np.zeros((num_labels, 2))
    for k in range(num_labels):
        true_solutions[k, :] = get_xy_position(k, num_pieces, offset_start=0)
    return true_solutions

def get_pred_solution_vector(p_final, num_pieces):

    num_labels = np.round(np.square(num_pieces)).astype(int)
    pred_solutions = np.zeros((num_labels, 2))
    for k in range(num_labels):
        pred_solutions[k, :] = np.unravel_index(np.argmax(p_final[:,:,0,k]), p_final[:,:,0,k].shape)[::-1]
    return pred_solutions

def simple_evaluation_vector(pred_solutions, true_absolute_solution, anchor_pos, anchor_idx, num_pieces):

    shift = anchor_pos[:2] - get_xy_position(anchor_idx, num_pieces, offset_start=0)
    true_solution_anchor = true_absolute_solution + shift
    corrects = pred_solutions == true_solution_anchor
    # we multiply the two columns of corrects
    # each column is one axis, we get the correct on both axis
    # np.sum counts how many of the correctly placed pieces we had
    num_correct_pieces = np.sum(corrects[:,0] * corrects[:,1])
    return num_correct_pieces
    

def simple_evaluation(p_final, num_pieces_side, offset_start, anchor_idx, verbosity=1):

    drawing_correctness = np.zeros((num_pieces_side, num_pieces_side, 3), dtype=np.uint8)
    num_correct_pieces = 0
    
    for j in range(num_pieces_side*num_pieces_side):
        estimated_pos_piece = np.unravel_index(np.argmax(p_final[:,:,0,j]), p_final[:,:,0,j].shape)[::-1]
        correct_position_relative = get_xy_position(j, num_pieces_side, offset_start=0)
        correct_position = correct_position_relative + offset_start
        if np.isclose(np.sum(np.abs(np.subtract(estimated_pos_piece, correct_position))), 0):
            num_correct_pieces += 1
            drawing_correctness[correct_position_relative[1], correct_position_relative[0]] = (0, 255, 0)
            if j == anchor_idx:
                drawing_correctness[correct_position_relative[1], correct_position_relative[0]] = (0, 0, 255)
            if verbosity > 0:
                print(f"piece {j} = estimated: {estimated_pos_piece}, correct: {correct_position} [CORRECT ({correct_position_relative})]")
        else:
            drawing_correctness[correct_position_relative[1], correct_position_relative[0]] = (255, 0, 0)
            if verbosity > 0:
                print(f"piece {j} = estimated: {estimated_pos_piece}, correct: {correct_position} [WRONG ({correct_position_relative})]")
    return num_correct_pieces, drawing_correctness


def get_neighbours(piece_idx, num_pieces_side):

    nbs = {}
    left_nb_idx = piece_idx - 1
    if left_nb_idx > -1 and left_nb_idx < (np.square(num_pieces_side) - 1):
        nbs['left'] = left_nb_idx
    top_nb_idx = piece_idx - num_pieces_side
    if top_nb_idx > -1 and top_nb_idx < (np.square(num_pieces_side) - 1):
        nbs['top'] = top_nb_idx
    right_nb_idx = piece_idx + 1 
    if right_nb_idx > -1 and right_nb_idx < (np.square(num_pieces_side) - 1):
        nbs['right'] = right_nb_idx
    bottom_nb_idx = piece_idx + num_pieces_side
    if bottom_nb_idx > -1 and bottom_nb_idx < (np.square(num_pieces_side) - 1):
        nbs['bottom'] = bottom_nb_idx
    return nbs


def get_xy_position(piece_idx, num_pieces_side, offset_start=0):
    pos_y = piece_idx % num_pieces_side
    pos_x = piece_idx // num_pieces_side
    correct_position = offset_start + np.asarray([pos_y, pos_x])
    return correct_position


def neighbor_comparison(solution_mat, num_pieces_side, offset_start):

    num_correct_neighbours = 0
    num_total_neighbours = 0

    for j in range(num_pieces_side*num_pieces_side):
        # where is the piece
        pos_central_piece = solution_mat[j][:2] # we are ignoring rotation!
        nbs_j = get_neighbours(j, num_pieces_side)
        for nb_rel_pos, nb_idx in nbs_j.items():
            num_total_neighbours += 1
            xy_pos = get_xy_position(nb_idx, num_pieces_side, offset_start)
            if nb_rel_pos == 'left':
                correct_position = pos_central_piece + np.array([-1, 0])               
            elif nb_rel_pos == 'top':
                correct_position = pos_central_piece + np.array([0, -1])
            elif nb_rel_pos == 'right':
                correct_position = pos_central_piece + np.array([1, 0])
            else: # if nb_rel_pos == 'bottom':
                correct_position = pos_central_piece + np.array([0, 1])

            if np.isclose(np.sum(np.subtract(xy_pos, correct_position)), 0):
                num_correct_neighbours += 1
    
    neighbor_val = num_correct_neighbours / num_total_neighbours
    return neighbor_val


def pixel_difference(proposed_solution, gt_img, measure='rmse'):

    # grayscale
    if len(gt_img.shape) > 2:
        gt_img = gt_img[:,:,0]
    # grayscale solution
    if len(proposed_solution.shape) > 2:
        proposed_solution = proposed_solution[:,:,0]
    # resizing
    if np.abs(np.sum(np.subtract(proposed_solution.shape[:2], gt_img.shape[:2]))) > 0:
        gt_img = cv2.resize(gt_img, proposed_solution.shape[:2], interpolation= cv2.INTER_NEAREST)

    proposed_solution = np.clip(proposed_solution, 0, 1)
    if np.max(gt_img) > 1:
        gt_img = gt_img.astype(float) / 255.0
    # measures
    if measure == 'rmse':
        pdiff = np.sqrt(np.mean(np.square(gt_img - proposed_solution)))
    elif measure == 'mse':
        pdiff = np.mean(np.square(gt_img - proposed_solution))
    elif measure == 'mae':
        pdiff = np.mean(np.abs(gt_img - proposed_solution))
    else:
        print("choose something like rmse, mse or mae (or implement new measures)! Meanwhile, I will return -1 as difference")
        pdiff = -1

    return pdiff 