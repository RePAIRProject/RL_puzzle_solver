from utils.parameters_utils import Configuration
from compatibility.grid_utils import PuzzleGrid

from .solver_rot_puzzle import solver_rot_puzzle
from .solver_utils_TEST import initialize_p, initialize_p_from_GT

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

        self.CM_dict = np.load(self.cfg.get_CM_path(), allow_pickle=True).item()

        R = self.CM_dict['R']

        
        N = R.shape[-1]
        
        if self.params.no_rotations:
            R = R[:, :, 0, :, :]

        # !!! Anchor number must be changed if some pieces were excluded
        if self.params.anchor_index < 0:
            anchor_index = np.random.choice(N)  # select_anchor(detect_output)
        else:
            anchor_index = self.params.anchor_index

        print(f"Using anchor the piece with id: {anchor_index}")

        num_rot = R.shape[2]

        grid_method = self.params['solver']['grid']['method']

        # initialize the probability / assignement matrix
        # this could?/should? be split into 2 phases parts: 1) compute the grid 2) initialize p  
        if grid_method == 'manual':
            p_xy_size = self.params.grid.manual_params.p_xy_size
            print(f'Using a grid of size {p_xy_size} points')
            self.p_initial, self.init_pos, self.anchor_pos = initialize_p(R, anchor_index, p_xy_size[0], p_xy_size[1])
        elif grid_method == 'gt':
            raise NotImplementedError()
            print('Using ground truth to calculate the grid')
            self.p_initial, self.init_pos, self.anchor_pos = initialize_p_from_GT(anc, puzzle_root_folder, all_pieces, pieces, num_rot)
        elif grid_method == 'auto':
            raise NotImplementedError()
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

        all_pay, all_sol, all_anc, p_final, eps, iter, na = self._solve()

        print("-" * 50)
        time_in_seconds = time.monotonic() - time_start
        print(f"Solving this puzzle took {format_duration(time_in_seconds)}")
        print("-" * 50)

    def _solve(self):
        R = np.maximum(R, -1)

        R_new = R
        faze = 0
        new_anc = []
        num_anchors = 1
        f = 0
        iter = 0
        eps = np.inf

        all_pay = []
        all_sol = []
        all_anc = []
        Y, X, Z, noPatches = p.shape

        # while not np.isclose(eps, 0)
        print("started solving..")
        while eps != 0 and iter < self.T_max:
            if num_anchors > num_anchors:
                num_anchors = num_anchors
                faze += 1
                p = np.ones((Y, X, Z, noPatches)) / (Y * X * Z)  # OPTIONAL !!!
                for jj in range(noPatches):
                    # if new_anc[jj, 0] != 0:
                    if a[jj, 0] == 1:
                        y = new_anc[jj, 0]
                        x = new_anc[jj, 1]
                        z = new_anc[jj, 2]
                        p[:, :, :, jj] = 0
                        p[y, x, :, :] = 0
                        p[y, x, z, jj] = 1

            if faze == 0:
                T = self.T_first
            else:
                T = self.T_next

            p, payoff, eps, iter = solver_rot_puzzle(R_new, R, p, T, iter, 0, verbosity=verbosity, decimals=decimals)

            I = np.zeros((noPatches, 1))
            m = np.zeros((noPatches, 1))

            for j in range(noPatches):
                pj_final = p[:, :, :, j]
                m[j, 0], I[j, 0] = np.max(pj_final), np.argmax(pj_final)

            I = I.astype(int)
            i1, i2, i3 = np.unravel_index(I, p[:, :, :, 1].shape)

            fin_sol = np.concatenate((i1, i2, i3), axis=1)
            if save_each_phase == True:
                save_vis_puzzle(fin_sol, Y, X, Z, saving_stuff, iter, show_borders=False)
            if verbosity > 0:
                print("#" * 70)
                print("ITERATION", iter)
                print("#" * 70)
                print(np.concatenate((fin_sol, np.round(m * 100)), axis=1))

            # pdb.set_trace()
            if num_anchors < (noPatches - 2):
                fix_tresh = cfg.anc_fix_tresh
            elif num_anchors > (noPatches - 2):
                fix_tresh = 0.11  ## just fix last 2 pieces  !!!
            else:
                fix_tresh = 0.33  ## just fix last 2 pieces  !!!

            a = (m > fix_tresh).astype(int)
            new_anc = np.array(fin_sol * a)
            num_anchors = np.sum(a)
            # if verbosity > 0:
            #     print("#" * 70)
            #     print(f"fixed solution for a new piece (at iteration {iter}):")
            #     print(new_anc)
            f += 1
            all_pay.append(payoff[2:])
            all_sol.append(fin_sol)
            all_anc.append(new_anc)

        # if verbosity > 0:
        #     print("#" * 70)
        #     print("ITERATION", iter)
        #     print("#" * 70)
        #     print(np.concatenate((fin_sol, np.round(m * 100)), axis=1))
        # all_sol.append(fin_sol)
        p_final = p
        return all_pay, all_sol, all_anc, p_final, eps, iter, num_anchors

    def save(self):
        
        solution_folder = os.path.join(puzzle_root_folder,
                                    f'{fnames.solution_folder_name}_anchor{anc}_{cmp_name}_with{num_rot}rot_{it_nums}_gt{args.use_GT}_k{args.k}')
        os.makedirs(solution_folder, exist_ok=True)
        print("Done! Saving in", solution_folder)

        # SAVE THE MATRIX BEFORE ANY VISUALIZATION
        filename = os.path.join(solution_folder, 'p_final')
        mdic = {"p_final": p_final, "label": "label", "anchor": anc, "anc_position": [x0, y0, z0]}
        savemat(f'{filename}.mat', mdic)
        np.save(filename, mdic)
