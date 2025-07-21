import numpy as np
from scipy.io import savemat
import argparse
import matplotlib.pyplot as plt
import cv2
import json, os
from PIL import Image

from compatibility.region import RegionMatrixModule
from utils.puzzle_utils import Puzzle
from utils.parameters_utils import Configuration

def main():
    cfg = Configuration() # this contains all IO operations plus the folder structure
    params = cfg.load('input_parameters.yaml')   # basic reading in this case

    puzzle = Puzzle()
    puzzle.load(cfg.get_puzzle_name(), cfg.get_data_folder(), params['compatibility']['features']) #, features=True)

    rmm = RegionMatrixModule(puzzle, params, cfg)  # it is redundant, we know 
    rmm.prepare() # creates grid, adjust/compute parameters and so on
    rmm.compute(verbose=params['verbosity'])
    # rmm.save_candidate_alignments_to_file(verbose=params['verbosity'])
    rmm.save(verbose=params['verbosity'])

if __name__ == '__main__':
    main()
