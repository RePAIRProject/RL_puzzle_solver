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
from solver.solver import SolverModule
from utils.puzzle_utils import Puzzle
from utils.parameters_utils import Configuration
from utils.visualization_utils import reconstruct_pil

def main():

    cfg = Configuration() # this contains all IO operations plus the folder structure
    params = cfg.load('input_parameters.yaml')   # basic reading in this case
    cfg.set_puzzle_single_run_random_folder_name('exp_nqnefd')

    # Load pieces
    puzzle = Puzzle()
    puzzle.load(cfg.get_puzzle_name(), cfg.get_data_folder(), params['compatibility']['features']) #, features=True)

    # Read solution
    pixel_solution = np.loadtxt(cfg.get_solution_as_csv_path())
    
    # # Create image
    # d1 = sm.P.shape[0]*sm.grid_params['xy_step']
    # d2 = sm.P.shape[1]*sm.grid_params['xy_step']
    # dimension = (d1, d2)
   
    image_solution = reconstruct_pil(pixel_solution, puzzle.pieces) #, dimension)
    plt.imsave(cfg.get_VIS_path(add_as_suffix=params['aggregation']['method']), image_solution)  # save final image in solution folder


if __name__ == '__main__':
    main()