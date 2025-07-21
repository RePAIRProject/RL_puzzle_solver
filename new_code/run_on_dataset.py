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
from compatibility.aggregation import AggregationModule
from solver.solver import SolverModule
from utils.puzzle_utils import Puzzle
from utils.parameters_utils import Configuration
from utils.visualization_utils import reconstruct, reconstruct_pil, crop_to_content, get_path_to_save_image


def main():

    cfg_dataset = Configuration() # this contains all IO operations plus the folder structure
    params = cfg_dataset.load('input_parameters_dataset.yaml')   # basic reading in this case    # 
    
    puzzle_folders = os.listdir(cfg_dataset.get_preprocessing_folder())
    sorted_puzzle_folders = natsort.natsorted(puzzle_folders)
    skip_done = params['skip_done']

    for puzzle_folder in sorted_puzzle_folders:

        cfg = Configuration() # this contains all IO operations plus the folder structure
                
        # cfg.set_data_folder(cfg_dataset.get_data_folder())
        if skip_done == False or os.path.exists(os.path.join(cfg_dataset.get_experiments_folder(), puzzle_folder)) == False:
            print("Start on", puzzle_folder)
            # print(cfg)
            # breakpoint()
            params = cfg.load('input_parameters_dataset.yaml', read_name=False)   # basic reading in this case
            # set by hand the puzzle name
            cfg.set_puzzle_name(puzzle_folder)
            puzzle = Puzzle()
            # print(f"puzzle: {puzzle} with {puzzle.num_of_pieces} pieces")
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
            # cmm.save()

            # am = AggregationModule(puzzle, params, cfg) 

            # # am.prepare() # read the data creates grid, adjust/compute parameters and so on
            # am.compute(verbose=params['verbosity']) # merge the data
            # am.save()

            # sm = SolverModule(params, cfg)

            # pixel_solution = sm.solve(verbose=params['verbosity'])
            # sm.save()

            # image_solution = reconstruct_pil(pixel_solution, puzzle.pieces)
            # plt.imsave(cfg.get_VIS_path(), image_solution)

            # print("Finished", puzzle_folder)
            # del puzzle 
            # del cfg 
            # del rmm
            # del cmm 
            # del am 
            # del sm 
            # breakpoint()
        else:
            print(f"skipping {puzzle_folder} as it is already there")
    # save final image
    #image_solution_pil.save(cfg.get_VIS_path())


    

if __name__ == '__main__':

    main()