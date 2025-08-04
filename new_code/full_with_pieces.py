import numpy as np
from scipy.io import savemat
import argparse
import pdb
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import json, os

from compatibility.region import RegionMatrixModule
from compatibility.compatibility import CompatibilityMatrixModule
from compatibility.aggregation import AggregationModule
from solver.solver_with_pieces import SolverWithPiecesModule
from utils.puzzle_utils import Puzzle
from utils.parameters_utils import Configuration
from utils.visualization_utils import reconstruct, reconstruct_pil, crop_to_content, build_suffix


def main():

    cfg = Configuration() # this contains all IO operations plus the folder structure
    params = cfg.load('input_parameters.yaml')   # basic reading in this case

    print("*" * 60)
    print(" Working on puzzle", cfg.get_puzzle_name())
    print("*" * 60)


    puzzle = Puzzle()
    puzzle.load(cfg.get_puzzle_name(), cfg.get_data_folder(), params['compatibility']['features']) #, features=True)

    rmm = RegionMatrixModule(puzzle, params, cfg)  # it is redundant, we know 
    rmm.prepare() # creates grid, adjust/compute parameters and so on
    rmm.compute(verbose=params['verbosity'])
    # rmm.save_candidate_alignments_to_file(verbose=params['verbosity'])
    rmm.save()

    cmm = CompatibilityMatrixModule(puzzle, params, cfg) 

    cmm.prepare() # creates grid, adjust/compute parameters and so on
    cmm.compute(verbose=params['verbosity'])
    # rmm.save_candidate_alignments_to_file(verbose=params['verbosity'])
    cmm.save()

    am = AggregationModule(puzzle, params, cfg) 

    # am.prepare() # read the data creates grid, adjust/compute parameters and so on
    am.compute(verbose=params['verbosity']) # merge the data
    am.save()

    swpm = SolverWithPiecesModule(puzzle, params, cfg)
    pixel_solution = swpm.solve(verbose=params['verbosity'])
    swpm.save()

    d1 = swpm.P.shape[0]*swpm.grid_params['xy_step']
    d2 = swpm.P.shape[1]*swpm.grid_params['xy_step']
    dimension = (d1, d2)

    image_solution = reconstruct_pil(pixel_solution, puzzle.pieces, dimension)
    if params['solver']['solution']['visualization']['add_suffix'] == True:
        suffix = build_suffix(params)
        saving_path = cfg.get_VIS_path(add_as_suffix=suffix)
    else:
        saving_path = cfg.get_VIS_path()
    plt.imsave(saving_path, image_solution)  # save final image in solution folder
   

if __name__ == '__main__':

    main()