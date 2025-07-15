from compatibility.compatibility import CompatibilityMatrixModule
from utils.puzzle_utils import Puzzle
from utils.parameters_utils import Configuration
from solver.solver import SolverModule
import argparse

def main():

    ##############
    # PSEUDOCODE # hopefully it will be that simple
    ##############
    
    #puzzle = Puzzle()
    #puzzle.load('RPobj_g3_o0003_gt_rot') #, features=True)
    cfg = Configuration() # this contains all IO operations plus the folder structure
    params = cfg.load('input_parameters.yaml')   # basic reading in this case
    # cfg.load already sets the puzzle name internally! otherwise we need # cfg.set_puzzle_name(params['puzzle_name'])
    cfg.set_puzzle_single_run_random_folder_name('exp_efpiri')

    sm = SolverModule(params, cfg)

    sm.solve(verbosity=params['verbosity'])
    sm.save()

if __name__ == '__main__':
    main()


