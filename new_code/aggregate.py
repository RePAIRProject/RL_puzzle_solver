from new_code.compatibility.compatibility import CompatibilityMatrixModule
from utils.puzzle_utils import Puzzle
from utils.parameters_utils import Configuration
import argparse

def main(args):

    cfg = Configuration() # this contains all IO operations plus the folder structure
    params = cfg.load('input_parameters.yaml')   # basic reading in this case
    # cfg.load already sets the puzzle name internally! otherwise we need # cfg.set_puzzle_name(params['puzzle_name'])
    cfg.set_puzzle_single_run_random_folder_name('exp_efpiri')

    puzzle = Puzzle()
    puzzle.load(cfg.get_puzzle_name()) #, features=True)

    am = AggregationModule(puzzle, params, cfg) 

    # am.prepare() # read the data creates grid, adjust/compute parameters and so on
    am.compute(verbose=params['verbosity']) # merge the data
    am.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregating various compatibility matrix')
    args = parser.parse_args()
    main(args)
