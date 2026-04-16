import os

from compatibility.region import RegionMatrixModule
from compatibility.compatibility import CompatibilityMatrixModule
from utils.puzzle_utils import Puzzle, load_from_file
from utils.parameters_utils import Configuration

import time 
# from concurrent.futures import ProcessPoolExecutor

def main():

    cfg = Configuration() # this contains all IO operations plus the folder structure
    params = cfg.load('input_parameters_wikiart_P.yaml')   # basic reading in this case

    print("*" * 60)
    print(" Working on puzzle", cfg.get_puzzle_name())
    print("*" * 60)

    puzzle = Puzzle()
    puzzle.load(cfg.get_puzzle_name(), cfg.get_data_folder(), params['compatibility']['features']) #, features=True)
    
    if os.path.exists(cfg.get_puzzle_info_path()) and params['preprocessing'].get('load_from_file', False):
        params['preprocessing'] = load_from_file(cfg.get_puzzle_info_path())

    rmm = RegionMatrixModule(puzzle, params, cfg)  # it is redundant, we know 
    print("Working on a new experiment, output going in:\n", rmm.cfg.get_current_experiments_folder())
    rmm.prepare() # creates grid, adjust/compute parameters and so on
    time_rmc = time.time()
    rmm.compute(verbose=params['verbosity'])
    print(f"took {time.time() - time_rmc} seconds to compute RM")
    rmm.save()

    cmm = CompatibilityMatrixModule(puzzle, params, cfg) 

    cmm.prepare() # creates grid, adjust/compute parameters and so on
    time_cmc = time.time()
    cmm.compute(verbose=params['verbosity'])
    print(f"took {(time.time() - time_cmc):.02f} seconds to compute CM")
    cmm.save()

if __name__ == '__main__':

    main()