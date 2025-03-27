from compatibility.compatibility_utils import CompatibilityMatrixModule
from utils.puzzle_utils import Puzzle
from utils.parameters_utils import Configuration
from new_code.solver.solver import SolverModule
import argparse

def main(args):

    ##############
    # PSEUDOCODE # hopefully it will be that simple
    ##############
    
    #puzzle = Puzzle()
    #puzzle.load('RPobj_g3_o0003_gt_rot') #, features=True)

    cfg = Configuration('RPobj_g3_o0003_gt_rot') # this contains all IO operations plus the folder structure
    cfg.set_puzzle_single_run_random_folder_name('exp_efpiri')
    params = cfg.load('input_parameters.yaml')   # basic reading in this case

    sm = SolverModule(params, cfg)

    sm.solve(verbose=params['verbosity'])
    sm.save()



