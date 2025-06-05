import numpy as np
from scipy.io import savemat
import argparse
import pdb
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import json, os
from PIL import Image

# # from configs import repair_cfg as cfg
# from configs import folder_names as fnames

# from puzzle_utils.shape_utils import prepare_pieces_v2, create_grid_v2, create_grid_v3, get_outside_borders, \
#         place_on_canvas, get_borders_around, include_shape_info, dilate
# # from puzzle_utils.shape_utils import prepare_pieces, shape_pairwise_compatibility
# from puzzle_utils.pieces_utils import calc_parameters_v2
# from puzzle_utils.visualization import save_vis

"""
WIP: 
probably should just read the .yaml file, create a RegionMatrix object, call .compute and that's it
most likely it will not be that easy  
"""
from new_code.compatibility.region import RegionMatrixModule
from utils.puzzle_utils import Puzzle
from utils.parameters_utils import Configuration

def main():

    ##############
    # PSEUDOCODE # hopefully it will be that simple
    ##############

    cfg = Configuration() # this contains all IO operations plus the folder structure
    params = cfg.load('input_parameters.yaml')   # basic reading in this case

    puzzle = Puzzle()
    puzzle.load(cfg.get_puzzle_name()) #, features=True)

    rmm = RegionMatrixModule(puzzle, params, cfg)  # it is redundant, we know 
    rmm.prepare() # creates grid, adjust/compute parameters and so on
    rmm.compute(verbose=params['verbosity'])
    # rmm.save_candidate_alignments_to_file(verbose=params['verbosity'])
    rmm.save()

if __name__ == '__main__':
    main()
