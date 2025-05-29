from utils.parameters_utils import Configuration
from compatibility.grid_utils import PuzzleGrid

from .solver_rot_puzzle import solver_rot_puzzle, fix_anchors
from .solver_utils import initialize_p, initialize_p_from_GT

import numpy as np
import time
from ..utils.human_readable_duration import format_duration


class SolverModule:

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

    def __init__(self, params: dict, cfg: Configuration):


        # self.pieces = pcs_uts.load_pieces(puzzle_name=puzzle_name)
        # if use_feats == True:
        #     self.pieces = fts_uts.load_features(pieces=self.pieces, puzzle_name=puzzle_name)
        self.params = params
        # here we need to load stuff from the yaml file     
        self.cfg = cfg #Configuration(puzzle.name) 
        # self.exp_folder is already set when we create the object
        # self.exp_folder = self.cfg.new_puzzle_single_run_random_folder_name()

        self.T_first = self.params['T_first']
        self.T_next = self.params['T_next']
        self.T_max = self.params['T_max']


        self._init()

    def _init(self):
        """
        All the parameters are set here
        It should be self-explanatory as it's just setting values (with predefined factors we hard-coded)

        """
        # load compatibility matrix
        self.CM_dict = np.load(self.cfg.get_CM_path(), allow_pickle=True).item()

        R = self.CM_dict['R']


        if self.params['no_rotations']:
            # Keep only the 0-th rotation but do not change R.ndim (= 5)
            R = R[:, :, 0:1, :, :]
        
        assert R.ndim == 5, f"R should have 5 dimensions: expecting (x,y,theta,N,N), got R.shape = {R.shape}"

        # numper of pieces
        N = R.shape[-1]
        # number of rotations
        num_rot = R.shape[2]


        # !!! Anchor number must be changed if some pieces were excluded
        if self.params['anchor_index'] < 0:
            self.anchor_index = np.random.choice(N)  # select_anchor(detect_output)
        else:
            self.anchor_index = self.params['anchor_index']

        print(f"Using anchor the piece with id: {self.anchor_index}")


        grid_method = self.params['solver']['grid']['method']

        # initialize the probability / assignement matrix
        # this could?/should? be split into 2 phases parts: 1) compute the grid 2) initialize p  
        if grid_method == 'manual':
            p_xy_size = self.params['solver']['grid']['manual_params']['p_xy_size']
            self.P, self.init_pos, self.anchor_pos = initialize_p(R, self.anchor_index, p_xy_size[0], p_xy_size[1])
            print(f'Using a grid of size {self.P.shape} points [manual]')
        elif grid_method == 'auto':
            self.P, self.init_pos, self.anchor_pos = initialize_p(R, self.anchor_index)
            print(f'Using a grid of size {self.P.shape} points [auto]')
        elif grid_method == 'gt':
            raise NotImplementedError()
            self.P, self.init_pos, self.anchor_pos = initialize_p_from_GT(anc, puzzle_root_folder, all_pieces, pieces, num_rot)
            print(f'Using a grid of size {self.P.shape} points [gt]')
        else:
            raise ValueError(f'Unknown method {grid_method}')
        

        

        # # print(p_initial.shape)
        # solver_visualization_folder = os.path.join(puzzle_root_folder,
        #                                         f'{fnames.solution_folder_name}_anchor{anc}_{cmp_name}_with{num_rot}rot_{it_nums}_gt{args.use_GT}_k{args.k}',
        #                                         'phase_frames')
        # os.makedirs(solver_visualization_folder, exist_ok=True)

        # save_each_phase = True
        # saving_stuff = (anc, pieces, pieces_files, pieces_folder, ppars, solver_visualization_folder)


    def solve(self):
        time_start = time.monotonic()

        self.payoffs = []
        self.solutions = []
        self.anchors = []

        self.P_initial = self.P

        all_pay, all_sol, all_anc, p_final, eps, iter, na = self._solve()

        print("-" * 50)
        time_in_seconds = time.monotonic() - time_start
        print(f"Solving this puzzle took {format_duration(time_in_seconds)}")
        print("-" * 50)

    def _solve(self, verbosity=1, decimals=8, save_each_phase=False, saving_stuff=[]):

        phase = 0
        f = 0
        iter = 0
        eps = np.inf

        # while not np.isclose(eps, 0)
        print("started solving..")
        while eps != 0 and iter < self.T_max:
            if phase == 0:
                T = self.T_first
            else:
                T = self.T_next

            self.P, payoff, eps = solver_rot_puzzle(self.R, self.P, T, verbosity=verbosity, decimals=decimals)

            self.P, sol  = fix_anchors(self.P)

            # if save_each_phase == True:
            #     save_vis_puzzle(sol, P, saving_stuff, iter, show_borders=False)

            phase += 1
            iter += T

            self.payoffs.append(payoff[2:])
            self.solutions.append(sol)
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
