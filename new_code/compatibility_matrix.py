from new_code.compatibility.compatibility import CompatibilityMatrixModule
from utils.puzzle_utils import Puzzle
from utils.parameters_utils import Configuration

def main():

    ##############
    # PSEUDOCODE # hopefully it will be that simple
    ##############  
    cfg = Configuration() # this contains all IO operations plus the folder structure
    params = cfg.load('input_parameters.yaml')   # basic reading in this case
    # cfg.load already sets the puzzle name internally! otherwise we need # cfg.set_puzzle_name(params['puzzle_name'])
    cfg.set_puzzle_single_run_random_folder_name('exp_efpiri')

    puzzle = Puzzle()
    puzzle.load(cfg.get_puzzle_name()) #, features=True)

    cmm = CompatibilityMatrixModule(puzzle, params, cfg) 

    cmm.prepare() # creates grid, adjust/compute parameters and so on
    cmm.compute(verbose=params['verbosity'])
    # rmm.save_candidate_alignments_to_file(verbose=params['verbosity'])
    cmm.save(verbose=params['verbosity'])


if __name__ == '__main__':
    main()
