from utils.parameters_utils import Configuration
from compatibility.grid_utils import PuzzleGrid

import numpy as np

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

        ## we could call here
        self._init()

    def _init(self):
        """
        All the parameters are set here
        It should be self-explanatory as it's just setting values (with predefined factors we hard-coded)

        """

        self.CM_dict = np.load(self.cfg.get_CM_path(), allow_pickle=True).item()


        grid_method = self.params['solver']['grid']['method']
        if grid_method != 'manual':
            raise Exception("Not implemeneted")
        
        self.grid = PuzzleGrid(self.params['solver']['grid'], self.piece_size)
        
        

    def solve():
        
       

        # ADD GT Oracle-Compatibility values
        mat2 = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_oracle_GT'))
        R_oracle = mat2['R']
        R = R + R_oracle * 1
        R = np.clip(R, -1, R)
        #R = sparsify_compatibility_matrix(R, args.k)    ## K-sparsification

        pieces_files = os.listdir(pieces_folder)
        pieces_files.sort()
        all_pieces = np.arange(len(pieces_files))
        pieces = np.arange(len(pieces_files))

        # # Few_Pieces
        # if args.few_pieces > 0:
        #     pieces_excl = np.array([0, 1, 2, 3, 4, 5, 6])
        #     pieces = [p for p in all_pieces if p not in all_pieces[pieces_excl]]
        #     R = R[:, :, :, pieces, :]  # re-arrange R-matrix
        #     R = R[:, :, :, :, pieces]

        # Few_Rotations
        if args.few_rotations > 0:
            n_rot = R.shape[2]
            rot_incl = np.arange(0, n_rot, n_rot / args.few_rotations)
            rot_incl = rot_incl.astype(int)
            R = R[:, :, rot_incl, :, :]

        # !!! Anchor number must be changed if some pieces were excluded
        if args.anchor < 0:
            anc = np.random.choice(len(pieces))  # select_anchor(detect_output)
        else:
            #anc = np.where(pieces[]
            anc = args.anchor
        print(f"Using anchor the piece with id: {anc}")

        na = 1
        num_rot = R.shape[2]

        ## INITIALIZATION
        if args.use_GT == True:
            print('Using ground truth to calculate the grid')
            p_initial, init_pos, anchor_pos = initialization_from_GT(anc, puzzle_root_folder, all_pieces, pieces, num_rot)
        else:
            print(f'Using a grid of {args.p_pts_x}x{args.p_pts_y} points!')
            p_initial, init_pos, anchor_pos = initialization(R, anc, args.p_pts_y, args.p_pts_x)

        # print(p_initial.shape)
        solver_visualization_folder = os.path.join(puzzle_root_folder,
                                                f'{fnames.solution_folder_name}_anchor{anc}_{cmp_name}_with{num_rot}rot_{it_nums}_gt{args.use_GT}_k{args.k}',
                                                'phase_frames')
        os.makedirs(solver_visualization_folder, exist_ok=True)

        save_each_phase = True
        saving_stuff = (anc, pieces, pieces_files, pieces_folder, ppars, solver_visualization_folder)

        all_pay, all_sol, all_anc, p_final, eps, iter, na = RePairPuzz(R, p_initial, na, cfg, verbosity=args.verbosity,
                                                                    decimals=args.decimals, \
                                                                    save_each_phase=save_each_phase,
                                                                    saving_stuff=saving_stuff)

        print("-" * 50)
        time_in_seconds = time.time() - time_start_puzzle
        if time_in_seconds > 100:
            time_in_minutes = (np.ceil(time_in_seconds / 60))
            if time_in_minutes < 60:
                print(f"Solving this puzzle took almost {time_in_minutes:.0f} minutes")
            else:
                time_in_hours = (np.ceil(time_in_minutes / 60))
                print(f"Solving this puzzle took almost {time_in_hours:.0f} hours")
        else:
            print(f"Solving this puzzle took {time_in_seconds:.0f} seconds")
        print("-" * 50)

        solution_folder = os.path.join(puzzle_root_folder,
                                    f'{fnames.solution_folder_name}_anchor{anc}_{cmp_name}_with{num_rot}rot_{it_nums}_gt{args.use_GT}_k{args.k}')
        os.makedirs(solution_folder, exist_ok=True)
        print("Done! Saving in", solution_folder)

        # SAVE THE MATRIX BEFORE ANY VISUALIZATION
        filename = os.path.join(solution_folder, 'p_final')
        mdic = {"p_final": p_final, "label": "label", "anchor": anc, "anc_position": [x0, y0, z0]}
        savemat(f'{filename}.mat', mdic)
        np.save(filename, mdic)
