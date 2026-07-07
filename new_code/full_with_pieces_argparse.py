import argparse
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import shutil

from compatibility.region import RegionMatrixModule
from compatibility.compatibility import CompatibilityMatrixModule
from compatibility.aggregation import AggregationModule
from solver.solver_with_pieces import SolverWithPiecesModule
from utils.puzzle_utils import Puzzle, load_from_file
from utils.parameters_utils import Configuration
from utils.visualization_utils import reconstruct_pil, build_suffix


def main(args):

    cfg = Configuration() # this contains all IO operations plus the folder structure
    params = cfg.load(args.yaml)   # basic reading in this case

    print("*" * 60)
    print(" Working on puzzle", cfg.get_puzzle_name())
    print("*" * 60)

    puzzle = Puzzle()
    puzzle.load(cfg.get_puzzle_name(), cfg.get_data_folder(), params['compatibility']['features']) #, features=True)
    
    # copy the original image to have a reference
    os.makedirs(cfg.get_puzzle_experiments_subfolder(), exist_ok=True)
    shutil.copy(cfg.get_original_image_path(), cfg.get_puzzle_experiments_subfolder())

    if os.path.exists(cfg.get_puzzle_info_path()) and params['preprocessing'].get('load_from_file', False):
        params['preprocessing'] = load_from_file(cfg.get_puzzle_info_path())

    rmm = RegionMatrixModule(puzzle, params, cfg)  # it is redundant, we know 
    print("Working on a new experiment, output going in:\n", rmm.cfg.get_current_experiments_folder())
    rmm.prepare() # creates grid, adjust/compute parameters and so on
    rmm.compute(verbose=params['verbosity'])
    # rmm.save_candidate_alignments_to_file(verbose=params['verbosity'])
    rmm.save()

    cmm = CompatibilityMatrixModule(puzzle, params, cfg) 

    cmm.prepare() # creates grid, adjust/compute parameters and so on
    cmm.compute(verbose=params['verbosity'])
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
    

    ## Placement
    if params['solver']['generate_placement_file'] == True:
        swpm.generate_placement_file(pixel_solution)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Full pipeline of the Relaxation Labeling Puzzle Solver. \n\
        Consists of:\n\t1. Region Matrix Computation\n\t2. Compatibility Computation\n\t3. Aggregation\n\t4. ReLab Solver \n\
        All parameters must be specified in the REQUIRED yaml file.')
    parser.add_argument('-Y', '--yaml', type=str, default='input_parameters.yaml', help='yaml file with all settings!')
    args = parser.parse_args()
    main(args)