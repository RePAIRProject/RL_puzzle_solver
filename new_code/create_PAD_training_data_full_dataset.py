import os
import natsort 

from compatibility.region import RegionMatrixModule
from compatibility.compatibility import CompatibilityMatrixModule
from utils.puzzle_utils import Puzzle, load_from_file
from utils.parameters_utils import Configuration

def main():

    cfg_dataset = Configuration() # this contains all IO operations plus the folder structure
    cfg_name = 'input_parameters_wikiart_P_neg.yaml'
    params = cfg_dataset.load(cfg_name, read_name=False)   # basic reading in this case    # 
    
    puzzle_folders = os.listdir(cfg_dataset.get_preprocessing_folder())
    sorted_puzzle_folders = natsort.natsorted(puzzle_folders)
    skip_done = params['skip_done']
    skip_first_N_puzzles = params.get('skip_first_N_puzzles', 0)
    stop_after_N_puzzles = params.get('stop_after_N_puzzles', 0)
    if skip_first_N_puzzles > 0:
        sorted_puzzle_folders = sorted_puzzle_folders[skip_first_N_puzzles:]
        print(f"skipped {skip_first_N_puzzles} puzzles, we start from {sorted_puzzle_folders[0]}")
    if stop_after_N_puzzles > 0:
        sorted_puzzle_folders = sorted_puzzle_folders[:stop_after_N_puzzles]
        print(f"will run and stop after {stop_after_N_puzzles} puzzles, last one will be {sorted_puzzle_folders[-1]}")


    for puzzle_folder in sorted_puzzle_folders:

        cfg = Configuration() # this contains all IO operations plus the folder structure

        if not skip_done or not os.path.exists(os.path.join(cfg_dataset.get_experiments_folder(), puzzle_folder)):

            params = cfg.load(cfg_name, read_name=False)   # basic reading in this case
            # set by hand the puzzle name
            cfg.set_puzzle_name(puzzle_folder)

            print("*" * 60)
            print(" Working on puzzle", cfg.get_puzzle_name())
            print("-" * 60)

            puzzle = Puzzle()
            puzzle.load(cfg.get_puzzle_name(), cfg.get_data_folder(), params['compatibility']['features']) #, features=True)

            if os.path.exists(cfg.get_puzzle_info_path()) and params['preprocessing'].get('load_from_file', False):
                params['preprocessing'] = load_from_file(cfg.get_puzzle_info_path())

            rmm = RegionMatrixModule(puzzle, params, cfg)  # it is redundant, we know 
            print("Output going in:\n", rmm.cfg.get_current_experiments_folder())
            print("-" * 60)
            rmm.prepare() # creates grid, adjust/compute parameters and so on
            rmm.compute(verbose=params['verbosity'])
            rmm.save()

            try:
                cmm = CompatibilityMatrixModule(puzzle, params, rmm.cfg)

                cmm.prepare() # creates grid, adjust/compute parameters and so on
                cmm.compute(verbose=params['verbosity'])
                cmm.save()
            except Exception as e:
                print("Exception")
                print(e)
                print(f"\nHad this (above) error with puzzle {puzzle_folder} on experiment {rmm.cfg.get_current_experiments_folder()}, will continue with the next one!")
            print("Done with ", cfg.get_puzzle_name())
            print("*" * 60)


if __name__ == '__main__':

    main()