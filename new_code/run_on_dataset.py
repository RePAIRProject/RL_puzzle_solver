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
from solver.solver_with_pieces import SolverWithPiecesModule
from solver.solver import SolverModule
from utils.puzzle_utils import Puzzle
from utils.parameters_utils import Configuration
from utils.visualization_utils import reconstruct, reconstruct_pil, crop_to_content, build_suffix


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

            cmm = CompatibilityMatrixModule(puzzle, params, rmm.cfg)

            cmm.prepare() # creates grid, adjust/compute parameters and so on
            cmm.compute(verbose=params['verbosity'])
            # breakpoint()
            # rmm.save_candidate_alignments_to_file(verbose=params['verbosity'])
            cmm.save()

            am = AggregationModule(puzzle, params, cmm.cfg)

            # # am.prepare() # read the data creates grid, adjust/compute parameters and so on
            am.compute(verbose=params['verbosity']) # merge the data
            am.save()

            swpm = SolverWithPiecesModule(puzzle, params, am.cfg)
            pixel_solution = swpm.solve(verbose=params['verbosity'])
            swpm.save()

            d1 = swpm.P.shape[0] * swpm.grid_params['xy_step']
            d2 = swpm.P.shape[1] * swpm.grid_params['xy_step']
            dimension = (d1, d2)

            image_solution = reconstruct_pil(pixel_solution, puzzle.pieces, dimension)
            if params['solver']['solution']['visualization']['add_suffix'] == True:
                suffix = build_suffix(params)
                saving_path = am.cfg.get_VIS_path(add_as_suffix=suffix)
            else:
                saving_path = am.cfg.get_VIS_path()
            plt.imsave(saving_path, image_solution)  # save final image in solution folder

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