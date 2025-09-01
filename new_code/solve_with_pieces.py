from utils.parameters_utils import Configuration
from solver.solver_with_pieces import SolverWithPiecesModule
from utils.puzzle_utils import Puzzle
from utils.visualization_utils import reconstruct_pil, build_suffix, reconstruct
import matplotlib.pyplot as plt

def main():

    cfg = Configuration() # this contains all IO operations plus the folder structure
    params = cfg.load('input_parameters_local.yaml')   
    # we are solving based on the CM previously computed, so we need to know `exp_name`
    cfg.set_puzzle_single_run_random_folder_name(params['exp_name'])

    print("Solving", params['puzzle_name'], "-", params['exp_name'])
    puzzle = Puzzle()
    puzzle.load(cfg.get_puzzle_name(), cfg.get_data_folder(), params['compatibility']['features']) #, features=True)

    swpm = SolverWithPiecesModule(puzzle, params, cfg)
    pixel_solution = swpm.solve(verbose=params['verbosity'])
    swpm.save()

    d1 = swpm.P.shape[0]*swpm.grid_params['xy_step']
    d2 = swpm.P.shape[1]*swpm.grid_params['xy_step']
    dimension = (d1, d2)

    image_solution = reconstruct_pil(pixel_solution, puzzle.pieces, dimension)

    if params['solver']['solution']['visualization']['add_suffix'] == True:
        suffix = build_suffix(params)
        saving_path = cfg.get_VIS_path(add_as_suffix=suffix)
    else:
        saving_path = cfg.get_VIS_path()
    plt.imsave(saving_path, image_solution)  # save final image in solution folder

if __name__ == '__main__':
    main()


