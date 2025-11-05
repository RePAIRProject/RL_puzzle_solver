import numpy as np
import cv2 as cv
import os
import json
import time
import scipy
import natsort

from threading import Lock

# from GUI.ReinforcementLearning import ReinforcementLearning

class CfgParameters(dict):
    __getattr__ = dict.__getitem__

class PuzzleSolver:
    def __init__(self, *args, **kwargs):
        self.probability_matrix = None
        self.compatibility_matrix = None
        self.maximum_probability = None
        self.delta_probs = None
        self.cfg = None
        self.cache_path = None
        # self.reinforcement_learning = ReinforcementLearning(None, None)
        self.final_solution = None
        self.ppars = args[0] if len(args) > 0 else None

        self.pieces_names = args[1] if len(args) > 1 else None

        if len(args) > 2:
            self.path_dic = args[2]
            self.cache_path = self.path_dic['cache_path']
        else:
            self.path_dic = None

        self.running = True

        self.alive_flag = True
        self.process = 0.0

        self.iteration = 0

        self.repair_lock = Lock()

        self.p_matrix_lock = Lock()
        self.cm_matrix_lock = Lock()

        self.locked_pieces = {}
        self.auto_locked = {}
        self.manual_locked = {}

    def get_iteration(self):
        return self.iteration

    def default_cfg(self, path_dic):
        self.cfg = CfgParameters()
        solver_parameters = path_dic['solver_parameters']
        # solver_parameter = {}
        # if os.path.exists(solver_parameters):
        #     solver_parameter = {}
        #     with open(solver_parameters, 'r') as cp:
        #         solver_parameter = json.load(cp)
        solver_parameter = solver_parameters
        self.cfg['Tfirst'] = solver_parameter['T_first']
        self.cfg['Tnext'] = solver_parameter['T_next']
        self.cfg['Tmax'] = solver_parameter['T_max']
        self.cfg['anc_fix_tresh'] = solver_parameter['accept_threshold']
        self.cfg['p_matrix_shape_x'] = solver_parameter['grid']['manual_params']['p_xy_size'][0]
        self.cfg['p_matrix_shape_y'] = solver_parameter['grid']['manual_params']['p_xy_size'][1]

        string = ("Tfirst: " + str(self.cfg['Tfirst']) + "   Tfirst: " + str(self.cfg['Tnext']) + "   Tmax: " + str(self.cfg['Tmax']) + "   anc_fix_tresh: " +
                  str(self.cfg['anc_fix_tresh']) + "   p_matrix_shape_x: " + str(self.cfg['p_matrix_shape_x']) + "   p_matrix_shape_y: " + str(self.cfg['p_matrix_shape_y']))

        self.logger(string)

    def set_running(self, running):
        self.running = running

    def get_dict(self):
        sol_dict = None
        probability_dict = None
        final_solution_bank = None
        if (self.maximum_probability is not None and self.final_solution is not None and
                self.ppars is not None and self.pieces_names is not None and self.running and
                self.running is not None and self.maximum_probability is not None):
            final_solution_bank = self.final_solution.copy()
            final_solution_bank[:, :2] = final_solution_bank[:, :2] * self.ppars['xy_step']
            final_solution_bank[:, 2] = final_solution_bank[:, 2] * self.ppars['theta_step']
            sol_dict = {}
            probability_dict = {}
            for j in range(final_solution_bank.shape[0]):
                sol_dict[self.pieces_names[j]] = final_solution_bank[j, :]
                # probability_dict[pieces_names[j]] = np.round(highest_values[j], 3)
                probability_dict[self.pieces_names[j]] = self.maximum_probability[j]
            # highest_values = []
            #
            # # Iterate over the `j` dimension
            # for i in range(self.probability_matrix.shape[3]):  # p_final.shape[3] gives the size of the `j` dimension
            #     # Extract the slice for the current `i`
            #     current_slice = self.probability_matrix[:, :, 0, i]  # Slice along (243, 243)
            #
            #     # Find the maximum value in the current slice
            #     max_value = np.max(current_slice)
            #
            #     # Append the result to the list
            #     highest_values.append(max_value)
            #
            # # Convert the list to a numpy array for easier manipulation (if needed)
            # highest_values = np.array(highest_values)
            #
            # # Display the results
            # # print("Highest values for each slice along the j dimension:")
            # # print(highest_values)
        return sol_dict, probability_dict, self.process, self.iteration

    def set_p_matrix(self, p_matrix):
        with self.p_matrix_lock:
            self.probability_matrix = p_matrix

    def set_cm_matrix(self, cm_matrix):
        with self.cm_matrix_lock:
            self.compatibility_matrix = cm_matrix

    def set_p_matrix_element(self, x, y, r, piece_name, prob, value = True):

        # self.reinforcement_enforcement_learning.update_probability_matrix(x, y, r, self.extract_piece_number(piece_name))

        piece_number = self.extract_piece_number(piece_name)

        if not value:
            if piece_number in self.locked_pieces.keys():
                self.locked_pieces.pop(piece_number)
                Y, X, Z, noPatches = self.probability_matrix.shape
                self.probability_matrix[:, :, :, piece_number] = 1 / (Y * X * Z)
            return

        # Shape of the probability matrix for the current piece
        shape_x, shape_y, shape_r = self.probability_matrix.shape[:3]

        # Generate coordinate grids for the matrix
        X, Y, R = np.meshgrid(np.arange(shape_x), np.arange(shape_y), np.arange(shape_r), indexing="ij")

        # # Compute Gaussian adjustment
        # sigma = prob  # Spread of the Gaussian
        # gaussian_adjustment = np.exp(-((X - x) ** 2 + (Y - y) ** 2 + (R - r) ** 2) / (2 * sigma ** 2))
        #
        # # Update probabilities for the specific piece
        # # with self.p_matrix_lock:
        # self.probability_matrix[:, :, :, piece_number] *= (1 - gaussian_adjustment)
        # self.probability_matrix[x, y, r, piece_number] += gaussian_adjustment[x, y, r]
        #
        # # Normalize probabilities to ensure they sum to 1
        # self.probability_matrix[:, :, :, piece_number] /= np.sum(self.probability_matrix[:, :, :, piece_number])

        self.probability_matrix[:, :, :, piece_number] = 0
        self.probability_matrix[x, y, r, piece_number] = 1

        pos = [x,y,r]

        self.lock_piece(piece_number, pos, True)

    def lock_piece(self, piece_number, pos, value = True):
        print("piece_number", piece_number)
        self.locked_pieces.update({piece_number: pos})
        print(self.locked_pieces)
        if value:
            self.reinit_p_matrix()

    def reinit_p_matrix(self):
        print("RESETTING THE PROBABILITY MATRIX")
        Y, X, Z, noPatches = self.probability_matrix.shape
        for piece in range(noPatches):
            if piece not in self.locked_pieces.keys():
                print("piece_number", piece)
                # Reset to uniform distribution
                self.probability_matrix[:, :, :, piece] = 1 / (Y * X * Z)
            # else:
            #     self.probability_matrix[:, :, :, piece] = 0
            #     pos = self.locked_pieces[piece]
            #     self.probability_matrix[pos[0], pos[1], pos[2], piece] = 1


    def repair_lock_toggle(self, value):
        if value:  # If value is True, lock the program
            if not self.repair_lock.locked():  # Ensure it's not already locked
                self.repair_lock.acquire()
                print("locked")
                # self.program_lock()
        else:  # If value is False, unlock the program
            if self.repair_lock.locked():  # Ensure it's actually locked before unlocking
                self.repair_lock.release()
                print("unlocked")

    def remove_from_locked(self, piece_name):
        piece_id = self.pieces_names.index(piece_name)
        if piece_id in self.locked_pieces.keys():
            self.locked_pieces.pop(piece_id)
            print(piece_id, "has been removed")
        self.reinit_p_matrix()

    def set_cm_element(self, main, neighbour, relative_position, value):
        x = int(relative_position[0])
        y = int(relative_position[1])
        z = int(relative_position[2])
        main_index = self.pieces_names.index(main)
        self.remove_from_locked(main)
        neighbour_index = self.pieces_names.index(neighbour)

        shape = self.compatibility_matrix.shape
        x_center = int(shape[0] // 2)
        y_center = int(shape[1] // 2)

        x_new = x_center + x
        y_new = y_center + y

        x_new_prime = x_center - x
        y_new_prime = y_center - y

        current_value_main_neighbour = self.compatibility_matrix[x_new, y_new, z, main_index, neighbour_index]
        current_value_neighbour_main = self.compatibility_matrix[x_new_prime, y_new_prime, z, neighbour_index, main_index]

        # self.compatibility_matrix[x_new, y_new, z, main_index, neighbour_index] = value
        # self.compatibility_matrix[-x, -y, z, neighbour_index, main_index] = value

        if value:
            self.compatibility_matrix[x_new, y_new, z, main_index, neighbour_index] = (current_value_main_neighbour + 1) * 2
            self.compatibility_matrix[x_new_prime, y_new_prime, z, neighbour_index, main_index] = (current_value_neighbour_main + 1) * 2
        else:
            self.compatibility_matrix[x_new, y_new, z, main_index, neighbour_index] = 0
            self.compatibility_matrix[x_new_prime, y_new_prime, z, neighbour_index, main_index] = 0
        self.reinit_p_matrix()
        pass

    def get_p_matrix(self):
        with self.p_matrix_lock:
            return self.probability_matrix

    def extract_piece_number(self, piece_name):
        return self.pieces_names.index(piece_name)

    def set_alive(self, alive_flag):
        self.alive_flag = alive_flag

    def solve_puzzle(self, R, anchor, pieces_names, ppars, path_dic, return_as='dict', solved_pieces=None):
        self.ppars = ppars
        self.pieces_names = pieces_names

        if solved_pieces is None:
            solved_pieces = []

        init_pos, x0, y0, z0 = self.initialization(R, anchor, solved_pieces, pieces_names, path_dic) # we do not pass p_size so it chooses automatically
        num_anchors = 1
        self.default_cfg(path_dic)
        all_pay, all_sol, all_anc, eps, iter, num_anchors, m = self.RePairPuzz(num_anchors)
        p_final = self.probability_matrix

        fin_sol = all_sol[len(all_sol)-1]
        fin_sol[:,:2] = fin_sol[:,:2] * ppars['xy_step']
        fin_sol[:,2] = fin_sol[:,2] * ppars['theta_step']

        # highest_values = []
        #
        # # Iterate over the `j` dimension
        # for i in range(p_final.shape[3]):  # p_final.shape[3] gives the size of the `j` dimension
        #     # Extract the slice for the current `i`
        #     current_slice = p_final[:, :, 0, i]  # Slice along (243, 243)
        #
        #     # Find the maximum value in the current slice
        #     max_value = np.max(current_slice)
        #
        #     # Append the result to the list
        #     highest_values.append(max_value)
        #
        # # Convert the list to a numpy array for easier manipulation (if needed)
        # highest_values = np.array(highest_values)
        #
        # # Display the results

        if return_as == 'list':
            return fin_sol.tolist()
        elif return_as == 'dict':
            sol_dict = {}
            probability_dict = {}
            for j in range(fin_sol.shape[0]):
                sol_dict[pieces_names[j]] = fin_sol[j, :]
                # probability_dict[pieces_names[j]] = np.round(highest_values[j], 3)
                probability_dict[pieces_names[j]] = m[j]
            return sol_dict, probability_dict
        elif return_as == 'nparray':
            return np.asarray(fin_sol)
        else:
            print(f"Return type {return_as} not implemented - returning as a list")
            return fin_sol

    def old_init(self, R, anc, solved_pieces, pieces_names, path_dic, p_size=0):
        z0 = 0  # rotation for anchored patch
        # Initialize reconstruction plan
        self.set_cm_matrix(R)
        no_grid_points = R.shape[0]

        no_patches = R.shape[3]
        no_rotations = R.shape[2]

        if p_size > 0:
            Y = p_size
            X = Y
        else:
            Y = round(no_grid_points * 2 + 1)
            # Y = round(0.5 * (no_grid_points - 1) * (no_patches + 1) + 1)
            X = Y
        Z = no_rotations

        # initialize assignment matrix
        p = np.ones((Y, X, Z, no_patches)) / (Y * X)  # uniform
        init_pos = np.zeros((no_patches, 3)).astype(int)

        # place anchored patch (center)
        y0 = round(Y / 2)
        x0 = round(X / 2)

        p[:, :, :, anc] = 0
        p[y0, x0, :, :] = 0
        p[y0, x0, z0, anc] = 1
        pos = [y0, x0, z0]
        self.locked_pieces.update({anc: pos})
        for piece in solved_pieces:
            b = self.extract_piece_number(piece[0])
            b = pieces_names.index(piece[0])
            pos = (piece[1][0], piece[1][1], piece[1][2])
            self.locked_pieces.update({b: pos})

            pos = (round(pos[0]), round(pos[1]), round(pos[2]))
            p[:, :, :, b] = 0
            p[pos[0], pos[1], pos[2], b] = 1
            init_pos[b, :] = pos

        self.set_p_matrix(p)

        init_pos[anc, :] = ([y0, x0, z0])

        return init_pos, x0, y0, z0

    def initialization(self, R, anc, solved_pieces, pieces_names, path_dic, p_size=0):
        # elif grid_method == 'extern':
        self.cfg = path_dic['yaml']
        if not os.path.exists(self.cfg.get_puzzle_external_solution_subfolder_path()):
            init_pos, x0, y0, z0 = self.old_init(R, anc, solved_pieces, pieces_names, path_dic)
            return init_pos, x0, y0, z0
            # raise Exception(
            #     "Missing external solution folder! Maybe you want to change the init method? \nYou can find it in:\ninput_parameters.yaml: solver --> grid --> method\n")
        else:
            ext_solutions_files_list = os.listdir(self.cfg.get_puzzle_external_solution_subfolder_path())
            ext_solutions_files_list = natsort.natsorted(ext_solutions_files_list)
            self.ext_solutions = [
                np.genfromtxt(os.path.join(self.cfg.get_puzzle_external_solution_subfolder_path(), file_name), dtype=None)
                for file_name in ext_solutions_files_list]
            for j in range(len(self.ext_solutions)):
                self.ext_solutions[j] = np.asarray(self.ext_solutions[j]).tolist()

            with open(self.cfg.get_puzzle_info_path(), 'r') as pijf:
                self.puzzle_info = json.load(pijf)
            p_xy_size = self.solver_params['grid']['manual_params']['p_xy_size']
            P_adeela = self.initialize_p_from_external_solution(self.ext_solutions,
                                                         self.puzzle_info['rescaling_factor'],
                                                         self.anchor_index,
                                                         self.params['compatibility']['grid'], p_xy_size,
                                                         spars_p=self.params['solver']['reassembleNet']['sparsify_p'],
                                                         vis=self.params['solver']['reassembleNet']['visualization'])
            self.set_p_matrix(P_adeela)

    def initialize_p_from_external_solution(all_solutions, rescaling_factor, anchor_idx: int, grid, p_xy_size=(0, 0),
                                            sparsify_p=0, vis=1, var=1):
        import heapq
        xy_step = grid['xy_step']
        theta_num_points = grid['theta_num_points']
        p_size_x = p_xy_size[0]
        p_size_y = p_xy_size[1]

        # initialize assignment matrix
        grid_size = (p_size_y, p_size_x, theta_num_points)  ## p_size
        p = np.zeros((grid_size[0], grid_size[1], grid_size[2], len(all_solutions[0])))

        for sol in all_solutions[0:1]:
            if np.array(sol)[:, 1:].shape[1] > 3:
                solution = np.array(sol)[:, 1:-1].astype(float)
                input_vars = np.array(sol)[:, -1].astype(float)
                input_vars = input_vars ** (1 / 2) / rescaling_factor  # sqrt and rescale
                variance = np.ones_like(solution, dtype=np.float64) * input_vars[:, np.newaxis]
            else:
                solution = np.array(sol)[:, 1:].astype(float)
                variance = np.ones_like(solution, dtype=np.float64) * 11

            # Rescale
            solution[:, :2] = solution[:, :2] / rescaling_factor

            # rotate and translate to origin [0,0,0]
            norm_solutions = normalize_solutions(solution, anchor_idx)  # output is in pixels and grades

            # adapt solutions to grid (translations)
            center = np.array([p_size_y // 2, p_size_x // 2], dtype=np.int64)  # shift to center
            norm_solutions[:, :2] = norm_solutions[:, :2] / xy_step + center
            norm_solutions[:, 2] = (norm_solutions[:, 2] + 360) % 360

            for i in range(len(norm_solutions)):
                if i == anchor_idx:
                    p[center[0], center[1], 0, i] = 1
                else:
                    mean = norm_solutions[i, :]  ## solution for the piece, t° !
                    std_devs = variance[i, :]  # std_devs = (10.0, 10.0, 0.5)  # st. deviation, t° !
                    prob = probability_for_single_fragment(grid_size, mean, std_devs)

                    if sparsify_p > 0:
                        n = 9  ## TODO  - load from input_params.yaml !!! That can be val of spars_p [1,3,5,7 ... ]
                        # n = sparsify_p**2 #OPTION
                        top_val = heapq.nlargest(n, prob.flatten().tolist())
                        prob[prob < np.min(top_val)] = 0
                    p[:, :, :, i] += prob

        #######################################################
        for i in range(len(solution)):
            prob_i = p[:, :, :, i]
            if vis == 1:
                # Show distribution for (theta = 0, 1, ... , n_of_slice)
                import matplotlib.pyplot as plt
                n_of_slice = 4
                vmin = prob_i.min()  # global limits of the scale
                vmax = prob_i.max()
                fig, axes = plt.subplots(1, n_of_slice, figsize=(30, 10))

                for j in range(n_of_slice):
                    ax = axes[j]  # map 0–5 in (row, col)
                    im = ax.imshow(prob_i[:, :, j], cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
                    ax.set_title(f"θ = {j} fragment{i} anc {anchor_idx}")
                    ax.set_xlabel("y")
                    ax.set_ylabel("x")
                # colorbar comune a tutti i subplot
                cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.07,
                                    fraction=0.05, )  # shrink=0.8, orientation='horizontal',fraction=0.05,
                cbar.set_label("Probability Density")
                plt.suptitle("distribution for different rotations θ (Uniform Colors)", fontsize=18)
                plt.show()

        # TODO normalizzation
        return p

    def extract_info(self, p):
        Y, X, Z, noPatches = p.shape
        I = np.zeros((noPatches, 1))
        m = np.zeros((noPatches, 1))

        for j in range(noPatches):
            pj_final = p[:, :, :, j]
            m[j, 0], I[j, 0] = np.max(pj_final), np.argmax(pj_final)

        I = I.astype(int)
        i1, i2, i3 = np.unravel_index(I, p[:, :, :, 0].shape)

        fin_sol = np.concatenate((i1, i2, i3), axis=1)
        self.maximum_probability = m
        self.final_solution = fin_sol
        return fin_sol, m


    def RePairPuzz(self, na, verbosity=1, decimals=8):
        R = np.maximum(self.compatibility_matrix, -1)
        R_new = R

        faze = 0
        new_anc = []
        na_new = na
        f = 0
        total_iter = 0
        iter = 0
        eps = np.inf

        all_pay = []
        all_sol = []
        all_anc = []

        p = self.get_p_matrix()

        Y, X, Z, noPatches = p.shape

        # while not np.isclose(eps, 0)
        while iter < self.cfg.Tmax and self.alive_flag:
            na_new = len(self.locked_pieces.keys())
            #     p = np.ones((Y, X, Z, noPatches)) / (Y * X)
            #     for piece in self.locked_pieces.keys():
            # check this
            print("na_new", na_new)
            na = na_new
            faze += 1
            self.reinit_p_matrix()
            if na_new > na:
                for piece_1 in self.locked_pieces.keys():
                    for piece_2 in self.locked_pieces.keys():
                        R_new[:,:,:, piece_1, piece_2] = 0

                # for jj in range(noPatches):
                #     if new_anc[jj, 0] != 0:
                #         y = new_anc[jj, 0]
                #         x = new_anc[jj, 1]
                #         z = new_anc[jj, 2]
                #         p[:, :, :, jj] = 0
                #         p[y, x, :, :] = 0
                #         p[y, x, z, jj] = 1
                #
                #         ## NEW: Re-normalization of R after anchoring
                #         for jj_anc in range(noPatches):
                #             if new_anc[jj_anc, 0] != 0:
                #                 R_new[:, :, : , jj_anc, jj] = 0

            # self.set_p_matrix(p)

            R_renorm = R_new / np.max(R_new)
            R_new = np.where((R_new > 0), R_renorm*1.5, R_new)

            self.set_cm_matrix(R_new)

            Tmax = self.cfg.Tmax

            if faze == 0:
                T = self.cfg.Tfirst
            else:
                T = self.cfg.Tnext

            #pdb.set_trace()
            payoff, eps, iter, total_iter = self.solver_rot_puzzle(T, iter, total_iter, Tmax, 0, verbosity=3, decimals=decimals, )
            fin_sol, m = self.extract_info(self.probability_matrix)

            if verbosity > 0:
                print("#" * 70)
                print("ITERATION", iter)
                print("#" * 70)
                print(np.concatenate((fin_sol, np.round(m * 100)), axis=1))



            if na < (noPatches-2):
                fix_tresh = self.cfg.anc_fix_tresh
            elif na > (noPatches-2):
                fix_tresh = 0.11   ## just fix last 2 pieces  !!!
            else:
                fix_tresh = 0.33   ## just fix last 2 pieces  !!!

            a = (m > fix_tresh).astype(int)
            new_anc = np.array(fin_sol * a)
            locked_piece_indices = np.where(a == 1)[0]
            for i in range(len(locked_piece_indices)):
                piece = int(locked_piece_indices[i])
                if piece not in self.locked_pieces.keys():
                    self.locked_pieces.update({piece: new_anc[piece]})
                else:
                    print("piece", piece, "is already locked")
            na_new = np.sum(a)


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
        return all_pay, all_sol, all_anc, eps, iter, na_new, m


    def solver_rot_puzzle(self, T, iter, total_iter, Tmax, visual, verbosity=1, decimals=8):

        # no_rotations = 4

        with self.cm_matrix_lock: #gui
            no_rotations = self.compatibility_matrix.shape[2]
            no_patches = self.compatibility_matrix.shape[3]

        payoff = np.zeros(T+1)
        z_st = 360 / no_rotations
        z_rot = np.arange(0, 360 - z_st + 1, z_st)
        # z_rot = np.arange(0., 4.)

        t = 0
        eps = np.inf

        p = self.get_p_matrix().copy() #gui

        while t < T and eps > 0 and self.alive_flag:
            t += 1
            iter += 1
            total_iter += 1
            self.process = float(total_iter)/float(Tmax)
            self.iteration = int(total_iter)

            with self.cm_matrix_lock:
                no_rotations = self.compatibility_matrix.shape[2]
                no_patches = self.compatibility_matrix.shape[3]

            q = np.zeros_like(p)
            for i in range(no_patches):
                with self.cm_matrix_lock:

                    ri = self.compatibility_matrix[:, :, :, :, i]
                #  ri = R[:, :, :, i, :]  # FOR ORACLE SQUARE ONLY
                for zi in range(no_rotations):
                    rr = scipy.ndimage.rotate(ri, z_rot[zi], reshape=False, mode='constant', order=0)
                    rr = np.roll(rr, zi, axis=2)
                    c1 = np.zeros(p.shape)
                    for j in range(no_patches):
                        for zj in range(no_rotations):
                            rj_z = rr[:, :, zj, j]
                            pj_z = p[:, :, zj, j]
                            # cc = cv.filter2D(pj_z, -1, np.rot90(rj_z, 2)) # solves in inverse order !?!
                            cc = cv.filter2D(pj_z, -1, rj_z)
                            c1[:, :, zj, j] = cc

                    q1 = np.sum(c1, axis=(2, 3))
                    # q2 = (q1 != 0) * (q1 + no_patches * no_rotations * 0.5) ## new_experiment
                    q2 = (q1 + no_patches * 1) # with removing no_rotations it is faster
                    q[:, :, zi, i] = q2
            q += q + no_patches * 1
            with self.p_matrix_lock:
                heat = 1
                pq = self.probability_matrix * np.exp(heat * q)
                self.delta_probs = pq - self.probability_matrix
                self.probability_matrix = pq / (np.sum(pq, axis=(0, 1, 2)))
                self.probability_matrix = np.where(np.isnan(self.probability_matrix), 0, self.probability_matrix)


                pay = np.sum(self.probability_matrix * q)

                payoff[t] = pay
                eps = abs(pay - payoff[t-1])
                with self.repair_lock:
                    if verbosity > 1:
                        if verbosity == 2:
                            print(f'Iteration {t}: pay = {pay:.08f}, eps = {eps:.08f}', end='\r')
                        else:
                            print(f'Iteration {t}: pay = {pay:.08f}, eps = {eps:.08f}')
                self.probability_matrix = np.round(self.probability_matrix, decimals)
                p = self.probability_matrix
                fin_sol, m = self.extract_info(p)
                # fix_tresh = 0.8
                # a = (m > fix_tresh).astype(int)
                # locked_piece_indices = np.where(a == 1)[0]
                # for i in range(len(locked_piece_indices)):
                #     if locked_piece_indices[i] not in self.locked_pieces:
                #         self.locked_pieces.append(int(locked_piece_indices[i]))
                #         print("LOCKED")
                #         self.reinit_p_matrix()
        return payoff, eps, iter, total_iter

    def logger(self, string):
        timestamp = str(time.time())  # seconds since epoch (as float, converted to string)
        with open(self.cache_path + "solver_log.txt", "a") as f:
            f.write(timestamp + " " + string + "\n")
