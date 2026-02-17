import numpy as np
from scipy.io import savemat
import argparse
import pdb
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import json, os
import natsort 

from compatibility.region import RegionMatrixModule
from compatibility.compatibility import CompatibilityMatrixModule
from utils.puzzle_utils import Puzzle
from utils.parameters_utils import Configuration

def main():

    cfg_dataset = Configuration() # this contains all IO operations plus the folder structure
    params = cfg_dataset.load('input_parameters_dataset.yaml')   # basic reading in this case    # 
    
    puzzle_folders = os.listdir(cfg_dataset.get_preprocessing_folder())
    sorted_puzzle_folders = natsort.natsorted(puzzle_folders)
    skip_done = params['skip_done']

    for puzzle_folder in sorted_puzzle_folders:

        cfg = Configuration() # this contains all IO operations plus the folder structure

        if skip_done == False or os.path.exists(os.path.join(cfg_dataset.get_experiments_folder(), puzzle_folder)) == False:

            params = cfg.load('input_parameters_dataset.yaml', read_name=False)   # basic reading in this case
            # set by hand the puzzle name
            cfg.set_puzzle_name(puzzle_folder)

            print("*" * 60)
            print(" Working on puzzle", cfg.get_puzzle_name())
            print("-" * 60)

            puzzle = Puzzle()
            # print(f"puzzle: {puzzle} with {puzzle.num_of_pieces} pieces")
            puzzle.load(cfg.get_puzzle_name(), cfg.get_data_folder(), params['compatibility']['features']) #, features=True)

            rmm = RegionMatrixModule(puzzle, params, cfg)  # it is redundant, we know 
            print("Output going in:\n", rmm.cfg.get_current_experiments_folder())
            print("-" * 60)
            rmm.prepare() # creates grid, adjust/compute parameters and so on
            rmm.compute(verbose=params['verbosity'])
            # rmm.save_candidate_alignments_to_file(verbose=params['verbosity'])
            rmm.save()

            cmm = CompatibilityMatrixModule(puzzle, params, rmm.cfg)

            cmm.prepare() # creates grid, adjust/compute parameters and so on
            cmm.compute(verbose=params['verbosity'])
            # breakpoint()
            # rmm.save_candidate_alignments_to_file(verbose=params['verbosity'])
            cmm.save()
            print("Done with ", cfg.get_puzzle_name())
            print("*" * 60)

if __name__ == '__main__':

    main()