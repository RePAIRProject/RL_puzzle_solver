import numpy as np
from scipy.ndimage import rotate
import cv2 as cv
import os
import json

class CfgParameters(dict):
    __getattr__ = dict.__getitem__

class PuzzleSolver:
    def __init__(self, *args, **kwargs):
        self.probability_matrix = None
        self.final_solution = None
        self.ppars = args[0] if len(args) > 0 else None
        self.pieces_names = args[1] if len(args) > 1 else None
        self.running = True

    def default_cfg(self, path_dic):
        cfg = CfgParameters()
        solver_parameters = path_dic['solver_parameters']
        solver_parameter = {}
        if os.path.exists(solver_parameters):
            solver_parameter = {}
            with open(solver_parameters, 'r') as cp:
                solver_parameter = json.load(cp)
        cfg['Tfirst'] = solver_parameter['Tfirst']
        cfg['Tnext'] = solver_parameter['Tnext']
        cfg['Tmax'] = solver_parameter['Tmax']
        cfg['anc_fix_tresh'] = solver_parameter['anc_fix_tresh']
        cfg['p_matrix_shape_x'] = solver_parameter['p_matrix_shape_x']
        cfg['p_matrix_shape_y'] = solver_parameter['p_matrix_shape_y']
        return cfg

    def set_running(self, running):
        self.running = running

    def get_probability_matrix(self):
        sol_dict = None
        probability_dict = None
        final_solution_bank = None
        if (self.probability_matrix is not None and self.final_solution is not None and
                self.ppars is not None and self.pieces_names is not None and self.running and
                self.running is not None and self.probability_matrix is not None):
            final_solution_bank = self.final_solution.copy()
            final_solution_bank[:, :2] = final_solution_bank[:, :2] * self.ppars['xy_step']
            final_solution_bank[:, 2] = final_solution_bank[:, 2] * self.ppars['theta_step']
            sol_dict = {}
            probability_dict = {}
            for j in range(final_solution_bank.shape[0]):
                sol_dict[self.pieces_names[j]] = final_solution_bank[j, :]
                # probability_dict[pieces_names[j]] = np.round(highest_values[j], 3)
                probability_dict[self.pieces_names[j]] = self.probability_matrix[j]
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
        return sol_dict, probability_dict


    def solve_puzzle(self, R, anchor, pieces_names, ppars, path_dic, return_as='dict', solved_pieces=None):
        self.ppars = ppars
        self.pieces_names = pieces_names

        if solved_pieces is None:
            solved_pieces = []

        p_initial, init_pos, x0, y0, z0 = self.initialization(R, anchor, solved_pieces, pieces_names) # we do not pass p_size so it chooses automatically
        num_anchors = 1
        cfg = self.default_cfg(path_dic)
        all_pay, all_sol, all_anc, p_final, eps, iter, num_anchors, m = self.RePairPuzz(R, p_initial, num_anchors, cfg)

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
        # print("Highest values for each slice along the j dimension:")
        # print(highest_values)

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


    def initialization(self, R, anc, solved_pieces, pieces_names, p_size=0):
        z0 = 0  # rotation for anchored patch
        # Initialize reconstruction plan
        no_grid_points = R.shape[0]
        print("no_grid_points", no_grid_points)
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
        for piece in solved_pieces:
            b = pieces_names.index(piece[0])
            pos = (piece[1][0], piece[1][1], piece[1][2])

            pos = (round(pos[0]), round(pos[1]), round(pos[2]))
            p[:, :, :, b] = 0
            p[pos[0], pos[1], pos[2], b] = 1
            init_pos[b, :] = pos

        init_pos[anc, :] = ([y0, x0, z0])

        return p, init_pos, x0, y0, z0

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
        self.probability_matrix = m
        self.final_solution = fin_sol
        return fin_sol, m


    def RePairPuzz(self, R, p, na, cfg, verbosity=1, decimals=8):
        R = np.maximum(R, -1)
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
        Y, X, Z, noPatches = p.shape

        # while not np.isclose(eps, 0)
        print("started solving..")
        while eps != 0 and iter < cfg.Tmax:
            if na_new > na:
                na = na_new
                faze += 1
                p = np.ones((Y, X, Z, noPatches)) / (Y * X)

                for jj in range(noPatches):
                    if new_anc[jj, 0] != 0:
                        y = new_anc[jj, 0]
                        x = new_anc[jj, 1]
                        z = new_anc[jj, 2]
                        p[:, :, :, jj] = 0
                        p[y, x, :, :] = 0
                        p[y, x, z, jj] = 1

                        ## NEW: Re-normalization of R after anchoring
                        for jj_anc in range(noPatches):
                            if new_anc[jj_anc, 0] != 0:
                                R_new[:, :, : , jj_anc, jj] = 0

            R_renorm = R_new / np.max(R_new)
            R_new = np.where((R_new > 0), R_renorm*1.5, R_new)

            if faze == 0:
                T = cfg.Tfirst
            else:
                T = cfg.Tnext

            #pdb.set_trace()
            p, payoff, eps, iter, total_iter = self.solver_rot_puzzle(R_new, R, p, T, iter, total_iter, 0, verbosity=3, decimals=decimals, )

            fin_sol, m = self.extract_info(p)
            if verbosity > 0:
                print("#" * 70)
                print("ITERATION", iter)
                print("#" * 70)
                print(np.concatenate((fin_sol, np.round(m * 100)), axis=1))



            if na < (noPatches-2):
                fix_tresh = cfg.anc_fix_tresh
            elif na > (noPatches-2):
                fix_tresh = 0.11   ## just fix last 2 pieces  !!!
            else:
                fix_tresh = 0.33   ## just fix last 2 pieces  !!!

            a = (m > fix_tresh).astype(int)
            new_anc = np.array(fin_sol*a)
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
        p_final = p
        return all_pay, all_sol, all_anc, p_final, eps, iter, na_new, m


    def solver_rot_puzzle(self, R, R_orig, p, T, iter, total_iter, visual, verbosity=1, decimals=8):
        no_rotations = R.shape[2]
        # no_rotations = 4
        print("No_Rotations", no_rotations)
        no_patches = R.shape[3]
        payoff = np.zeros(T+1)
        z_st = 360 / no_rotations
        z_rot = np.arange(0, 360 - z_st + 1, z_st)
        # z_rot = np.arange(0., 4.)
        print("z_rot", z_rot)
        t = 0
        eps = np.inf
        while t < T and eps > 0:
            t += 1
            iter += 1
            total_iter += 1

            q = np.zeros_like(p)
            for i in range(no_patches):
                ri = R[:, :, :, :, i]
                #  ri = R[:, :, :, i, :]  # FOR ORACLE SQUARE ONLY
                for zi in range(no_rotations):
                    rr = rotate(ri, z_rot[zi], reshape=False, mode='constant')
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
                    q2 = (q1 + no_patches * no_rotations * 1)
                    q[:, :, zi, i] = q2

            pq = p * q  # e = 1e-11
            p_new = pq / (np.sum(pq, axis=(0, 1, 2)))
            p_new = np.where(np.isnan(p_new), 0, p_new)


            pay = np.sum(p_new * q)

            payoff[t] = pay
            eps = abs(pay - payoff[t-1])
            if verbosity > 1:
                if verbosity == 2:
                    print(f'Iteration {t}: pay = {pay:.08f}, eps = {eps:.08f}', end='\r')
                else:
                    print(f'Iteration {t}: pay = {pay:.08f}, eps = {eps:.08f}')
            p = np.round(p_new, decimals)
            fin_sol, m = self.extract_info(p)
        return p, payoff, eps, iter, total_iter