import os

from compatibility.region import RegionMatrixModule
from compatibility.compatibility import CompatibilityMatrixModule
from utils.puzzle_utils import Puzzle, load_from_file
from utils.parameters_utils import Configuration

def main():

    cfg = Configuration() # this contains all IO operations plus the folder structure
    params = cfg.load('input_parameters_escher.yaml')   # basic reading in this case

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
    rmm.compute(verbose=params['verbosity'])
    rmm.save()

    cmm = CompatibilityMatrixModule(puzzle, params, cfg) 

    cmm.prepare() # creates grid, adjust/compute parameters and so on
    cmm.compute(verbose=params['verbosity'])
    cmm.save()

if __name__ == '__main__':

    main()