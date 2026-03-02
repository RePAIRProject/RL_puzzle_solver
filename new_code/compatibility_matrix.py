from compatibility.compatibility import CompatibilityMatrixModule
from utils.puzzle_utils import Puzzle, load_from_file
from utils.parameters_utils import Configuration
import os, json 

def main():

    cfg = Configuration() # this contains all IO operations plus the folder structure
    
    params = cfg.load('input_parameters_escher.yaml')   # basic reading in this case
    # we are calculating CM based on RM previously computed, so we need to know `exp_name`
    cfg.set_puzzle_single_run_random_folder_name(params['exp_name'])
    
    puzzle = Puzzle()
    puzzle.load(cfg.get_puzzle_name(), cfg.get_data_folder(), params['compatibility']['features']) #, features=True)
    
    if os.path.exists(cfg.get_puzzle_info_path()) and params['preprocessing'].get('load_from_file', False) == True:
        params['preprocessing'] = load_from_file(cfg.get_puzzle_info_path())

    cmm = CompatibilityMatrixModule(puzzle, params, cfg) 

    cmm.prepare() # creates grid, adjust/compute parameters and so on
    cmm.compute(verbose=params['verbosity'])
    cmm.save(verbose=params['verbosity'])


if __name__ == '__main__':
    main()
