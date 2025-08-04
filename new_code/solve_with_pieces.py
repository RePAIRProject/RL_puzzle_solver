from utils.parameters_utils import Configuration
from solver.solver_with_pieces import SolverWithPiecesModule
from utils.puzzle_utils import Puzzle

def main():

    cfg = Configuration() # this contains all IO operations plus the folder structure
    params = cfg.load('input_parameters_local.yaml')   
    # we are solving based on the CM previously computed, so we need to know `exp_name`
    cfg.set_puzzle_single_run_random_folder_name('exp_luytop')

    puzzle = Puzzle()
    puzzle.load(cfg.get_puzzle_name(), cfg.get_data_folder(), params['compatibility']['features']) #, features=True)

    swpm = SolverWithPiecesModule(puzzle, params, cfg)
    swpm.solve(verbose=params['verbosity'])
    swpm.save()

if __name__ == '__main__':
    main()


