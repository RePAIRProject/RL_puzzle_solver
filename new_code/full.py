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
from utils.visualization_utils import reconstruct, reconstruct_pil, crop_to_content, get_path_to_save_image


def main():

    cfg = Configuration() # this contains all IO operations plus the folder structure
    params = cfg.load('input_parameters.yaml')   # basic reading in this case

    puzzle = Puzzle()
    puzzle.load(cfg.get_puzzle_name(), cfg.get_data_folder(), load_features=False) #, features=True)

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

    sm = SolverModule(params, cfg)

    pixel_solution = sm.solve(verbose=params['verbosity'])
    sm.save()


    image_solution = reconstruct_pil(pixel_solution, puzzle.pieces)
    # save final image
    #image_solution_pil.save(cfg.get_VIS_path())


    plt.imsave(cfg.get_VIS_path(), image_solution)

if __name__ == '__main__':

    main()