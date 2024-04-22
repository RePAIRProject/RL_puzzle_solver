import json
import os
import pdb

from scipy.io import loadmat
from utils import solve_puzzle

"""
This is the method which should be used from the HIL interface

- v1 - 2024/04/11
Now it actually assumes RM and CM are precomputed and just launches the solver
"""

def assemble(fragments_list, return_solution_as='dict'):
    """
    Assemble a small subset of the puzzle given one anchor and some neighbours:
    ---
    Input Parameters:
    - fragments_list: a dictionary with the following keys():
        - anchor: the id (RPf_XXXXX) of the anchor piece
        - neighbours: a list of ids of the neighbour pieces
        - puzzle: the puzzle they refer to (ex: repair_g28)
    ---
    Output:
    if return_solution_as == "dict":
        - solution: a dictionary with as key the pieces (RPf_XXXXX) and as value the correct 
                    position (x, y, theta)
    if return_solution_as == "list":
        - solution: a list (same length as fragments_list) with tuples (x, y, theta)
    if return_solution_as == "nparray":
        - solution: a numpy array (shape: [len(fragments_list), 3] ) with (x, y, theta) values  
    """
    anchor_piece = fragments_list['anchor']

    neighbours = fragments_list['neighbours']
    puzzle = fragments_list['puzzle']

    print("anchor_piece", anchor_piece)
    print("neighbours", neighbours)

    puzzle_root = os.path.join(os.getcwd(), 'output', puzzle)
    pieces_folder = os.path.join(puzzle_root, 'pieces')
    cmp_parameter_path = os.path.join(puzzle_root, 'compatibility_parameters.json')
    if os.path.exists(cmp_parameter_path):
        ppars = {}
        with open(cmp_parameter_path, 'r') as cp:
            ppars = json.load(cp)

    pieces_names = os.listdir(pieces_folder)
    pieces_names.sort()
    
    # get pieces as a list
    anchor = 0
    neighbours_as_list = []
    pieces_to_include = []
    for k, p_name in enumerate(pieces_names):
        to_include = False
        if p_name == anchor_piece or anchor_piece in p_name:
            anchor = k
            to_include = True
        if p_name in neighbours:
            neighbours_as_list.append(k)
            to_include = True
        if to_include is True:
            pieces_to_include.append(k)

    print("piece to include", pieces_to_include)
    # extract from R matrix
    # THIS IS HARDCODED WE NEED TO CHANGE LATER
    mat = loadmat(os.path.join(puzzle_root, 'compatibility_matrix', f'CM_linesdet_manual_cost_LAP'))
    R = mat['R_line']
    # R = R[:, :, :, pieces_to_include, :]  # re-arrange R-matrix
    # R = R[:, :, :, :, pieces_to_include]
    # if you want rotation which you shouldn't
    R = R[:, :, 0:2, pieces_to_include, :]  # re-arrange R-matrix
    R = R[:, :, 0:2, :, pieces_to_include]

    anchor = pieces_to_include.index(anchor)

    pieces_included = []

    for i in range(len(pieces_to_include)):
        pieces_included.append(pieces_names[pieces_to_include[i]])

    # print("piece names", pieces_included)

    solution = solve_puzzle(R, anchor, pieces_included, ppars, return_as=return_solution_as)

    return solution
    
    