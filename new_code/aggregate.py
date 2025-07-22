from compatibility.aggregation import AggregationModule
from utils.puzzle_utils import Puzzle
from utils.parameters_utils import Configuration
import argparse

def main(args):

    cfg = Configuration() # this contains all IO operations plus the folder structure
    params = cfg.load('input_parameters.yaml')   # basic reading in this case
    # we are aggregating CM previously computed, so we need to know `exp_name`
    cfg.set_puzzle_single_run_random_folder_name('exp_jdhoig')

    puzzle = Puzzle()
    puzzle.load(cfg.get_puzzle_name(), cfg.get_data_folder(), params['compatibility']['features']) #, features=True)

    am = AggregationModule(puzzle, params, cfg) 

    am.compute(verbose=params['verbosity']) # merge the data
    am.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregating various compatibility matrix')
    args = parser.parse_args()
    main(args)
