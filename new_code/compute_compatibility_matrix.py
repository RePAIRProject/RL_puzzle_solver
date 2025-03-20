from compatibility.compatibility_utils import CompatibilityMatrixModule
from utils.puzzle_utils import Puzzle
from utils.parameters_utils import Configuration
import argparse

def main(args):

    ##############
    # PSEUDOCODE # hopefully it will be that simple
    ##############
    
    puzzle = Puzzle()
    puzzle.load('RPobj_g3_o0003_gt_rot') #, features=True)

    cfg = Configuration('RPobj_g3_o0003_gt_rot') # this contains all IO operations plus the folder structure
    cfg.set_puzzle_single_run_random_folder_name('exp_efpiri')
    params = cfg.load('input_parameters.yaml')   # basic reading in this case

    cmm = CompatibilityMatrixModule(puzzle, params, cfg) 

    cmm.prepare() # creates grid, adjust/compute parameters and so on
    cmm.compute(verbose=params['verbosity'])
    # rmm.save_candidate_alignments_to_file(verbose=params['verbosity'])
    cmm.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computing compatibility matrix')
    # parser.add_argument('--dataset', type=str, default='RePAIR_exp_batch3_clean_TEST', help='dataset (name of the folders)')
    # parser.add_argument('--puzzle', type=str, default='',
    #                     help='puzzle to work on - leave empty to generate for the whole dataset')
    # parser.add_argument('--save_everything', type=bool, default=False, help='save also overlap and borders matrices')
    # parser.add_argument('--lines', type=int, default=0, help='use line-based regions')
    # parser.add_argument('--lines_det_method', type=str, default='deeplsd', help='method line detection', choices=['exact', 'deeplsd', 'manual']) # exact, manual, deeplsd
    # parser.add_argument('--motif', type=int, default=0, help='use motif-based regions')
    # parser.add_argument('--motif_det_method', type=str, default='yolo-obb', help='method motif detection', choices=['yolo-obb', 'yolo-bbox', 'yolo-seg']) # exact', 'deeplsd', 'manual']
    # parser.add_argument('--irregular', type=int, default=0, help='use irregular parameters')
    # parser.add_argument('--save_visualization', type=bool, default=True,
    #                     help='save an image that showes the matrices color-coded')
    # parser.add_argument('-np', '--num_pieces', type=int, default=0,
    #                     help='number of pieces (per side) - use 0 (default value) for synthetic pieces')  # 8
    # parser.add_argument('--xy_step', type=int, default=3, help='the step (in pixels) between each grid point')
    # parser.add_argument('--xy_grid_points', type=int, default=121,
    #                     help='the number of points in the grid (for each axis, total number will be the square of what is given)')
    # parser.add_argument('--theta_step', type=int, default=90, help='degrees of each rotation')
    # parser.add_argument('--DEBUG', action='store_true', default=False,
    #                     help='WARNING: will use debugger! It stops and show the matrices!')
    args = parser.parse_args()
    main(args)
