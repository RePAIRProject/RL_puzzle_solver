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


def get_visual_solution_from_p(p_final, pieces_folder, piece_size, offset_start, num_pieces_side):
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
    squared_solution_img = solution_img[start_point[0]:end_point[0], start_point[1]:end_point[1]]
    return squared_solution_img

def simple_evaluation(p_final, num_pieces_side, offset_start, verbosity=1):

    drawing_correctness = np.zeros((num_pieces_side, num_pieces_side), dtype=np.uint8)
    num_correct_pieces = 0
    for j in range(num_pieces_side*num_pieces_side):
        estimated_pos_piece = np.unravel_index(np.argmax(p_final[:,:,0,j]), p_final[:,:,0,j].shape)
        correct_position_relative = get_xy_position(j, num_pieces_side, offset_start=0)
        #print(correct_position)
        correct_position = correct_position_relative + offset_start
        #pdb.set_trace()
        if np.isclose(np.sum(np.abs(np.subtract(estimated_pos_piece, correct_position))), 0):
            num_correct_pieces += 1
            drawing_correctness[correct_position_relative[0], correct_position_relative[1]] = (255)
            if verbosity > 0:
                print(f"piece {j} = estimated: {estimated_pos_piece}, correct: {correct_position} [CORRECT ({correct_position_relative})]")
        else:
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


def get_xy_position(piece_idx, num_pieces_side, offset_start):
    pos_y = piece_idx % num_pieces_side
    pos_x = piece_idx // num_pieces_side
    correct_position = offset_start + np.asarray([pos_x, pos_y])
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


def pixel_difference(gt_img, proposed_solution, measure='rmse'):

    if len(gt_img.shape) > 2:
        gt_img = gt_img[:,:,0]
    if len(proposed_solution.shape) > 2:
        proposed_solution = proposed_solution[:,:,0]
    if np.abs(np.sum(np.subtract(gt_img.shape[:2], proposed_solution.shape[:2]))) > 0:
        gt_img = cv2.resize(gt_img, proposed_solution.shape[:2])
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