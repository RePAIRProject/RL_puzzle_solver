from utils.parameters_utils import Configuration, CustomYAMLEncoder
from compatibility.grid import PuzzleGrid, PieceOnCanvas

from utils.puzzle_utils import Puzzle, PuzzlePiece
from .solver_rot_puzzle import solver_rot_puzzle, fix_anchors_with_occ
from .solver_utils import compute_pixel_solution,initialize_p, initialize_p_from_GT, initialize_p_with_occupancy
from utils.human_readable_duration import format_duration
from utils.visualization_utils import reconstruct

from typing import List
import numpy as np
import time
import yaml
import matplotlib.pyplot as plt


class SolverWithPiecesModule:

    """
    solver:
        T_first: 1000
        T_next: 500
        T_max: 3000
        no_rotations: True
        # TODO: switch to anchor ID
        anchor_index: -1 # TODO: null, None?
        # threshold
        accept_threshold: 0.75
        p_precision: 10
        grid:
            # ['gt','auto','manual']
            method: gt
            # this is used only when method is set to 'manual'
            manual_params:
            size: [151,151]
  
    """

    def __init__(self, puzzle: Puzzle, params: dict, cfg: Configuration):

        self.puzzle = puzzle
        self.params = params
        # here we need to load stuff from the yaml file     
        self.cfg = cfg #Configuration(puzzle.name) 
        self.solver_params = self.params['solver']
        self.T_first = self.solver_params['T_first']
        self.T_next = self.solver_params['T_next']
        self.T_max = self.solver_params['T_max']
        self.threshold = self.solver_params['accept_threshold']
        self.PQ_mode = self.solver_params['PQ_mode']

        self._init()

    def _init(self):
        """
        All the parameters are set here
        It should be self-explanatory as it's just setting values (with predefined factors we hard-coded)

        """
        # load compatibility matrix
        self.CM_dict = np.load(self.cfg.get_CM_path(), allow_pickle=True).item()
        # We use `R` as the aggregated matrix
        if 'R' not in self.CM_dict.keys():
            print("\nWARNING:")
            print("In the CM file we did not find the `R` key (the aggregated compatibility). \nDid you forget to run the aggregation?")
            print("We found the following keys:")
            for cmk in self.CM_dict.keys():
                print("\t -", cmk)
            print("\nPlease run the aggregation step on this CM\n")
            raise Exception("Missing the aggregation")
        else:
            self.R = self.CM_dict['R']
        self.grid_params = self.CM_dict['__context']['grid_params']

        if self.solver_params['no_rotations']:
            # Keep only the 0-th rotation but do not change R.ndim (= 5)
            self.R = self.R[:, :, 0:1, :, :]
        
        assert self.R.ndim == 5, f"R should have 5 dimensions: expecting (x,y,theta,N,N), got R.shape = {R.shape}"

        # numper of pieces
        self.N = self.R.shape[-1]
        # number of rotations
        num_rot = self.R.shape[2]

        # prepare their "occupancy grid" 
        # this is a grid of the "space" that a piece occupies in the P matrix.
        # for the moment we do not account rotation, it can be rotated when "placed"
        # the general idea is that if we fix a piece in the center, the grid points which are very close
        # are "occupied" by the piece itself, and they need to be zeroed in the P matrix
        self.occupancy_grid_pieces = self._compute_occupancy_grid()            

        # !!! Anchor number must be changed if some pieces were excluded
        if self.solver_params['anchor_index'] < 0:
            self.anchor_index = N//2  # TEMPORAL SOLUTION - central anchor for sanity check
            self.solver_params['anchor_index'] = self.anchor_index # this way it will be saved to the json file!
            #self.anchor_index = np.random.choice(N)  # select_anchor(detect_output)
        else:
            self.anchor_index = self.solver_params['anchor_index']


        print(f"Using anchor the piece with id: {self.anchor_index}")

        grid_method = self.solver_params['grid']['method']

        # initialize the probability / assignement matrix
        # this could?/should? be split into 2 phases parts: 1) compute the grid 2) initialize p  
        if grid_method == 'manual':
            p_xy_size = self.solver_params['grid']['manual_params']['p_xy_size']
            self.P, self.init_pos, self.anchor_pos = initialize_p(self.R, self.anchor_index, p_xy_size[0], p_xy_size[1])
            print(f'Using a grid of size {self.P.shape} points [manual]')
        elif grid_method == 'auto':
            self.P, self.init_pos, self.anchor_pos = initialize_p(self.R, self.anchor_index)
            print(f'Using a grid of size {self.P.shape} points [auto]')
        elif grid_method == 'gt':
            raise NotImplementedError()
            self.P, self.init_pos, self.anchor_pos = initialize_p_from_GT(anc, puzzle_root_folder, all_pieces, pieces, num_rot)
            print(f'Using a grid of size {self.P.shape} points [gt]')
        elif grid_method == 'occ':
            self.P, self.init_pos, self.anchor_pos = initialize_p_with_occupancy(self.R, self.anchor_index, self.occupancy_grid_pieces)
            print(f'Using a grid of size {self.P.shape} points [auto with occupancy]')
        else:
            raise ValueError(f'Unknown method {grid_method}')
    
    def _compute_occupancy_grid(self, show_results:bool=False):
            
        occ_grid = np.zeros((self.N, self.grid_params['xy_num_points'], self.grid_params['xy_num_points']))
        grid = PuzzleGrid(self.grid_params, self.params['preprocessing']['piece_size'])
        xy = grid.xy_values
        for n, piece in enumerate(self.puzzle.pieces):
            piece_on_canvas = PieceOnCanvas( piece=piece, grid=grid, x=grid.canvas_center, y=grid.canvas_center, theta=0)
            
            if show_results == True:
                plt.subplot(131)
                plt.imshow(piece_on_canvas.image)
                plt.subplot(132)
                plt.imshow(piece_on_canvas.image)
                green_points = []
                red_points = []

            for j in range(xy.shape[0]):
                for k in range(xy.shape[1]):
                    
                    if piece_on_canvas.mask[xy[k, j, 0], xy[k, j, 1]] > 0:
                        occ_grid[n, k, j] = 1
                        if show_results == True:
                            green_points.append([xy[k, j, 0], xy[k, j, 1]])
                    else:
                        occ_grid[n, k, j] = 0
                        if show_results == True:
                            red_points.append([xy[k, j, 0], xy[k, j, 1]])

            if show_results == True:
                green_points = np.asarray(green_points)
                red_points = np.asarray(red_points)
                plt.scatter(green_points[:,1], green_points[:,0], color='green') 
                plt.scatter(red_points[:,1], red_points[:,0], color='red') 
                plt.subplot(122)
                plt.imshow(occ_grid[n,:,:])
                plt.show()
                breakpoint()

        return occ_grid

    def solve(self, verbose:int=1):
        time_start = time.monotonic()

        self.payoffs = []
        self.grid_solutions = []
        self.anchors = []

        self.P_initial = self.P

        self._solve(verbosity=verbose)

        self.final_grid_solution = self.grid_solutions[-1]

        self.final_pixel_solution = compute_pixel_solution(self.final_grid_solution, self.grid_params['xy_step'], self.grid_params['theta_step'])

        print("-" * 50)
        time_in_seconds = time.monotonic() - time_start
        print(f"Solving this puzzle took {format_duration(time_in_seconds)}")
        print("-" * 50)

        return self.final_pixel_solution

    def _solve(self, verbosity=1, decimals=8, save_each_phase=False, saving_stuff=[]):

        phase = 0
        f = 0
        iter = 0
        eps = np.inf
        num_anchors = 1

        # while not np.isclose(eps, 0)
        print("started solving..")
        while eps != 0 and iter < self.T_max:
            if phase == 0:
                T = self.T_first
            else:
                T = self.T_next

            self.P, payoff, eps = solver_rot_puzzle(self.R, self.P, T, self.PQ_mode, verbosity=verbosity, decimals=decimals)

            self.P, sol, num_anchors = fix_anchors_with_occ(self.P, num_anchors, self.threshold, self.occupancy_grid_pieces)

            # if save_each_phase == True:
            #     save_vis_puzzle(sol, P, saving_stuff, iter, show_borders=False)

            phase += 1
            iter += T

            self.payoffs.append(payoff[2:])
            self.grid_solutions.append(sol)
            #self.anchors.append(new_anc)

            if verbosity > 0:
                print("#" * 70)
                print("ITERATION", iter)
                print("#" * 70)
                print(sol)

        # if verbosity > 0:
        #     print("#" * 70)
        #     print("ITERATION", iter)
        #     print("#" * 70)
        #     print(np.concatenate((fin_sol, np.round(m * 100)), axis=1))
        # all_sol.append(fin_sol)
    
    def _ensemble_solver(self, verbosity:int=1, decimals:int=8, save_each_phase:bool=False, saving_stuff:List=[]):
        """
        Solves the puzzle iteratively, launching one RL process per piece, using all of them as anchor.
        Then it fuses them (based on `consensus`) and fix 
        """
        iter = 0
        eps = np.inf 
        status = 0
        
    
    def save(self):
        """ save """
        context_params = self.CM_dict['__context']
        # context_params['input_params'] = self.params
        # context_params['grid_params'] = {'xy_num_points': self.grid.xy_num_points, 'theta_num_points': self.grid.theta_num_points, 'xy_step':self.grid.xy_step, 'theta_step':self.grid.theta_step, 
        #     'canvas_size': self.grid.canvas_size, 'pairwise_comp_range': self.grid.pairwise_comp_range}
        # context_params['features'] = self.features_status
        # context_params['puzzle'] = {'puzzle_name': self.puzzle.name, 'num_pieces': self.puzzle.num_of_pieces, 'piece_size': self.piece_size}
        context_params['solver'] = self.params['solver']
        # better to save all of them, instead of manually adding
        # {'T_first': self.T_first, 'T_next': self.T_next, 'T_max': self.T_max, 'anchor_idx':self.anchor_index, \
        #     'P_shape':self.P.shape}
        
        # values of the matrix  
        self.solver_dict = {}
        self.solver_dict['__context'] = context_params
        self.solver_dict['solver_data'] = {}
        data = self.solver_dict['solver_data']
        data['grid_solutions'] = self.grid_solutions
        data['grid_solution'] = self.final_grid_solution
        data['payoffs'] = self.payoffs

        self.solver_dict['solution'] = self.final_pixel_solution

        self.cfg.new_puzzle_solution_folder_name()
        np.save(self.cfg.get_solution_path(), self.solver_dict)
        if self.params['solver']['solution']['include_probs'] == False:
            self.final_pixel_solution = self.final_pixel_solution[:,:-1]
        if self.params['solver']['solution']['include_names'] == True:
            filenames = self.cfg.get_puzzle_pieces_filenames()
            names_as_columns = np.expand_dims(np.transpose(filenames), axis=1)
            pixel_solution_with_names = np.concatenate([names_as_columns, self.final_pixel_solution], axis=1)
            np.savetxt(self.cfg.get_solution_as_csv_path(), pixel_solution_with_names, fmt='%s')
        else:
            np.savetxt(self.cfg.get_solution_as_csv_path(), self.final_pixel_solution, fmt='%d')
        # input parameters
        input_params_path = self.cfg.get_solution_input_parameters_path() 
        with open(input_params_path, 'w') as f:
            yaml.dump(self.params, f, Dumper=CustomYAMLEncoder, default_flow_style=False)
        # output parameters
        context_params_path = self.cfg.get_solution_output_parameters_path() 
        with open(context_params_path, 'w') as f:
            yaml.dump(context_params, f, Dumper=CustomYAMLEncoder, default_flow_style=False)


    # def save(self):
        
    #     solution_folder = os.path.join(puzzle_root_folder,
    #                                 f'{fnames.solution_folder_name}_anchor{anc}_{cmp_name}_with{num_rot}rot_{it_nums}_gt{args.use_GT}_k{args.k}')
    #     os.makedirs(solution_folder, exist_ok=True)
    #     print("Done! Saving in", solution_folder)

    #     # SAVE THE MATRIX BEFORE ANY VISUALIZATION
    #     filename = os.path.join(solution_folder, 'p_final')
    #     mdic = {"p_final": p_final, "label": "label", "anchor": anc, "anc_position": [x0, y0, z0]}
    #     savemat(f'{filename}.mat', mdic)
    #     np.save(filename, mdic)
