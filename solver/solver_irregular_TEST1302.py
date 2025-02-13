import numpy as np
import cv2 as cv
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from scipy.io import savemat, loadmat
from scipy.ndimage import rotate
from PIL import Image
import os
import configs.folder_names as fnames
import argparse

from compatibility.utils import normalize_CM
from solver.solver_utils_TEST import initialization_from_GT, initialization, select_anchor, save_vis_puzzle
from solver.solver_utils_TEST import RePairPuzz, reconstruct_puzzle, sparsify_compatibility_matrix
from solver.aggregation_CM import aggregate_CM_matrices
from puzzle_utils.pieces_utils import calc_parameters_v2, crop_to_content
from puzzle_utils.visualization import save_vis
from puzzle_utils.shape_utils import prepare_pieces_v2
# from compatibility.line_matching_NEW_segments import read_info
# from puzzle_utils.shape_utils import prepare_pieces_v2, create_grid, place_on_canvas
import datetime
import pdb
import time
import json
from puzzle_utils.regions import combine_region_masks

class CfgParameters(dict):
    __getattr__ = dict.__getitem__


def solver_rot_puzzle(R, R_orig, p, T, iter, visual, verbosity=1, decimals=8):
    no_rotations = R.shape[2]
    no_patches = R.shape[3]
    payoff = np.zeros(T + 1)
    z_st = 360 / no_rotations
    z_rot = np.arange(0, 360 - z_st + 1, z_st)
    t = 0
    eps = np.inf
    while t < T and eps > 0:
        t += 1
        iter += 1
        q = np.zeros_like(p)
        for i in range(no_patches):
            ri = R[:, :, :, :, i]
            for zi in range(no_rotations):
                rr = rotate(ri, z_rot[zi], reshape=False, mode='constant', order=0)
                rr = np.roll(rr, zi, axis=2)
                c1 = np.zeros(p.shape)
                for j in range(no_patches):
                    for zj in range(no_rotations):
                        rj_z = rr[:, :, zj, j]
                        pj_z = p[:, :, zj, j]
                        cc = cv.filter2D(pj_z, -1, rj_z)
                        c1[:, :, zj, j] = cc

                q1 = np.sum(c1, axis=(2, 3))
                # q2 = (q1 + no_patches * no_rotations * 1) ### un dubbio !!!
                q2 = (q1 + no_patches * 1)
                q[:, :, zi, i] = q2

        pq = p * np.exp(q)  # e = 1e-11
        p_new = pq / (np.sum(pq, axis=(0, 1, 2)))
        p_new = np.where(np.isnan(p_new), 0, p_new)
        pay = np.sum(p_new * q)

        payoff[t] = pay
        eps = abs(pay - payoff[t - 1])
        if verbosity > 1:
            if verbosity == 2:
                print(f'Iteration {t}: pay = {pay:.08f}, eps = {eps:.08f}', end='\r')
            else:
                print(f'Iteration {t}: pay = {pay:.08f}, eps = {eps:.08f}')
        p = np.round(p_new, decimals)
    return p, payoff, eps, iter


#  MAIN
def main(args, pieces=None):
    print("Solver log\nSearch for `SOLVER_START_TIME` or `SOLVER_END_TIME` if you want to see which images are done")
    puzzle_name = args.puzzle

    print()
    print("-" * 50)
    print("-- SOLVER_START_TIME -- ")
    time_start_puzzle = time.time()
    # get the current date and time
    now = datetime.datetime.now()
    print(f"{now}\nStarted working on {puzzle_name}")
    print(f"Dataset: {args.dataset}")
    print("-" * 50)

    cfg = CfgParameters()
    cfg['Tfirst'] = args.tfirst
    cfg['Tnext'] = args.tnext
    cfg['Tmax'] = args.tmax
    cfg['anc_fix_tresh'] = args.thresh
    cfg['p_matrix_shape'] = args.p_pts_x
    cfg['cmp_type'] = args.cmp_type
    cfg['cmp_cost'] = args.cmp_cost
    cfg['combo_type'] = args.combo_type
    print('\tSOLVER PARAMETERS')
    for cfg_key in cfg.keys():
        print(f"{cfg_key}: {cfg[cfg_key]}")
    print("-" * 50)

    # pieces, img_parameters = prepare_pieces_v2(fnames, args.dataset, args.puzzle, verbose=True)  #commented for RePAIR
    puzzle_root_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset, args.puzzle)
    solver_patameters_path = os.path.join(puzzle_root_folder, 'solver_parameters.json')
    with open(solver_patameters_path, 'w') as spj:
        json.dump(cfg, spj, indent=3)
    print("saved json solver parameters file")

    cmp_parameter_path = os.path.join(puzzle_root_folder, 'compatibility_parameters_v2.json')
    if os.path.exists(cmp_parameter_path):
        ppars = CfgParameters()
        with open(cmp_parameter_path, 'r') as cp:
            ppars_dict = json.load(cp)
        for ppk in ppars_dict.keys():
            ppars[ppk] = ppars_dict[ppk]
    else:
        print("\n" * 3)
        print("/" * 70)
        print("/\t***ERROR***\n/ compatibility_parameters_v2.json not found!")
        print("/" * 70)
        print("\n" * 3)
        #ppars = calc_parameters_v2(img_parameters, args.xy_step, args.xy_grid_points, args.theta_step)

    if args.cmp_type == 'lines':
        cmp_name = f"linesdet_{args.lines_det_method}_cost_{args.cmp_cost}"
    elif args.cmp_type == 'shape':
        cmp_name = "shape"
    elif args.cmp_type == 'motifs':
        cmp_name = f"motifs_{args.motif_det_method}"
    elif args.cmp_type == 'color':
        cmp_name = f"cmp_color"
    elif args.cmp_type == 'combo':
        cmp_name = f"cmp_combo{args.combo_type}"
    else:
        cmp_name = f"cmp_{args.cmp_type}"

    it_nums = f"{args.tmax}its"
    pieces_folder = os.path.join(puzzle_root_folder, f"{fnames.pieces_folder}")

    ### AGGREGATION - moved to aggregate_CM
    # aggregate motifs
    if args.cmp_type == 'motifs' or args.combo_type == 'SH-AggMOT':
        print("loading motifs-CM for aggregation")
        R = aggregate_Motif_matrices(args, puzzle_root_folder)

    # combo
    if args.cmp_type == 'combo':
        cmp_name = f"combo_{args.combo_type}"
        R = aggregate_CM_matrices(args, puzzle_root_folder)
    else:
        print("loading", os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_{cmp_name}'))
        mat = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_{cmp_name}'))
        R = mat['R']

    ## LOAD CM matrices aggregate if combo o single if CM_type
    if args.cmp_type == 'combo':
        cmp_name = f"combo_{args.combo_type}"  # change folder_name to load (combo_)

    print("loading", os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_{cmp_name}'))
    mat = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_{cmp_name}'))
    R = mat['R']

    R = normalize_CM(R)

    # ADD GT Oracle-Compatibility values
    mat2 = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_cmp_Oracle_GT'))
    R_oracle = mat2['R']
    R = R + R_oracle * 1
    R = np.clip(R, -1, R)
    #R = sparsify_compatibility_matrix(R, args.k)    ## K-sparsification

    pieces_files = os.listdir(pieces_folder)
    pieces_files.sort()
    pieces = np.arange(len(pieces_files))

    # Few_Pieces
    pieces_excl = np.array([0, 1,2, 3,4, 5,6])
    all_pieces = np.arange(len(pieces_files))
    pieces = [p for p in all_pieces if p not in all_pieces[pieces_excl]]
    R = R[:, :, :, pieces, :]  # re-arrange R-matrix
    R = R[:, :, :, :, pieces]

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
        p_initial, init_pos, x0, y0, z0 = initialization_from_GT(anc, puzzle_root_folder, all_pieces, pieces, num_rot)
    else:
        print(f'Using a grid of {args.p_pts_x}x{args.p_pts_y} points!')
        p_initial, init_pos, x0, y0, z0 = initialization(R, anc, args.p_pts_y, args.p_pts_x)

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

    # VISUALIZATION
    f = len(all_sol)
    Y, X, Z, _ = p_final.shape
    fin_sol = all_sol[f - 1]
    # pdb.set_trace()
    fin_im1 = reconstruct_puzzle(fin_sol, Y, X, Z, anc, pieces, pieces_files, pieces_folder, ppars, show_borders=False)
    fin_im1_brd = reconstruct_puzzle(fin_sol, Y, X, Z, anc, pieces, pieces_files, pieces_folder, ppars,
                                     show_borders=True)
    os.makedirs(solution_folder, exist_ok=True)
    final_solution = os.path.join(solution_folder, f'final_using_anchor{anc}.png')
    plt.imshow((fin_im1 * 255).astype(np.uint8))
    plt.tight_layout()
    plt.savefig(final_solution)
    plt.close()
    plt.imsave(f"{final_solution[:-4]}_bordered.png", np.clip(fin_im1_brd, 0, 1))
    clean_img = fin_im1 * (fin_im1 > 0.1)
    plt.imsave(f"{final_solution[:-4]}_cropped.png", crop_to_content(clean_img * 255).astype(np.uint8))
    plt.imsave(f"{final_solution[:-4]}_bordered_cropped.png",
               crop_to_content(np.clip(fin_im1_brd, 0, 1) * 255).astype(np.uint8))
    # fin_im_v2 = reconstruct_puzzle_v2(fin_sol, Y, X, Z, pieces_dict, ppars, use_RGB=True)
    # final_solution_v2 = os.path.join(solution_folder, f'final_using_anchor{anc}_overlap.png')
    # if np.max(fin_im_v2) > 1:
    #     fin_im_v2 = np.clip(fin_im_v2, 0, 1)
    # plt.imsave(final_solution_v2, fin_im_v2)
    # fin_im_cropped = crop_to_content(fin_im_v2)
    # final_solution_v2_cropped = os.path.join(solution_folder, f'final_using_anchor{anc}_overlap_cropped.png')
    # plt.imsave(final_solution_v2_cropped, fin_im_cropped)

    f = len(all_anc)
    fin_sol = all_anc[f - 1]
    fin_im2 = reconstruct_puzzle(fin_sol, Y, X, Z, anc, pieces, pieces_files, pieces_folder, ppars)

    final_solution_anchor = os.path.join(solution_folder, f'final_only_anchor_using_anchor{anc}.png')
    plt.figure(figsize=(16, 16))
    plt.title("Final solution including ONLY solved pieces")
    plt.imshow((fin_im2 * 255).astype(np.uint8))
    plt.tight_layout()
    plt.savefig(final_solution_anchor)
    plt.close()

    alc_path = os.path.join(solution_folder, 'alc_plot.png')
    f = len(all_pay)
    f_pay = []
    for ff in range(f):
        a = all_pay[ff]
        f_pay = np.append(f_pay, a)
    f_pay = np.array(f_pay)
    plt.figure(figsize=(6, 6))
    plt.plot(f_pay, 'r', linewidth=1)
    plt.tight_layout()
    plt.savefig(alc_path)

    if args.save_frames is True:
        # intermediate steps
        frames_folders = os.path.join(solution_folder, 'frames_all')
        os.makedirs(frames_folders, exist_ok=True)

        for ff in range(f):
            frame_path = os.path.join(frames_folders, f"frame_{ff:05d}.png")
            cur_sol = all_sol[ff]
            im_rec = reconstruct_puzzle(cur_sol, Y, X, Z, anc, pieces, pieces_files, pieces_folder, ppars,
                                        show_borders=False)
            im_rec = np.clip(im_rec, 0, 1)
            plt.imsave(frame_path, im_rec)

        frames_folders = os.path.join(solution_folder, 'frames_anc')
        os.makedirs(frames_folders, exist_ok=True)

        for ff in range(f):
            frame_path = os.path.join(frames_folders, f"frame_{ff:05d}.png")
            cur_sol = all_anc[ff]
            im_rec = reconstruct_puzzle(cur_sol, Y, X, Z, anc, pieces, pieces_files, pieces_folder)
            im_rec = np.clip(im_rec, 0, 1)
            plt.imsave(frame_path, im_rec)

    print("-" * 50)
    print("-- SOLVER_END_TIME -- ")
    # get the current date and time
    now = datetime.datetime.now()
    print(f"{now}")
    print(f'Done with {puzzle_name}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='........ ')  # add some description
    parser.add_argument('--dataset', type=str, default='RePAIR_exp_batch3_clean', help='dataset folder')
    parser.add_argument('--puzzle', type=str, default='RPobj_g1_o0001_gt_rot', help='puzzle folder')
    parser.add_argument('--lines_det_method', type=str, default='deeplsd',
                        help='method line detection')  # exact, manual, deeplsd
    parser.add_argument('--motif_det_method', type=str, default='yolo-obb',
                        help='method motif detection')  # exact, manual, deeplsd
    parser.add_argument('--cmp_cost', type=str, default='LAP', help='cost computation')  # LAP, LCI
    parser.add_argument('--use_GT', default=False, action='store_true',
                        help='uses the ground truth (requires gt txt file!)')
    parser.add_argument('--anchor', type=int, default=2, help='anchor piece (index)')
    parser.add_argument('--save_frames', default=False, action='store_true',
                        help='use to save all frames of the reconstructions')
    parser.add_argument('--exclude', default=False, action='store_true',
                        help='use to exclude pieces without compatibility (used for some partial compatibilities, not fully tested!)')
    parser.add_argument('--verbosity', type=int, default=2,
                        help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')
    parser.add_argument('--few_rotations', type=int, default=0, help='uses only few rotations to make it faster')
    parser.add_argument('--tfirst', type=int, default=250,
                        help='when to stop for multi-phase the first time (fix anchor, reset the rest)')
    parser.add_argument('--tnext', type=int, default=250, help='the step for multi-phase (each tnext reset)')
    parser.add_argument('--tmax', type=int, default=1000, help='the final number of iterations (it exits after tmax)')
    parser.add_argument('--thresh', type=float, default=0.75,
                        help='a piece is fixed (considered solved) if the probability is above the thresh value (max .99)')
    parser.add_argument('--p_pts_y', type=int, default=-1, help='the size of the p matrix (it will be p_pts x p_pts)')
    parser.add_argument('--p_pts_x', type=int, default=0, help='the size of the p matrix (it will be p_pts x p_pts)')
    parser.add_argument('--decimals', type=int, default=10, help='decimal after comma when cutting payoff')
    parser.add_argument('--k', type=int, default=10, help='keep the best k values (for each pair) in the compatibility')
    parser.add_argument('--cmp_type', type=str, default='shape', help='which compatibility to use!',
                        choices=['combo', 'lines', 'shape', 'color', 'motifs', 'seg'])
    parser.add_argument('--combo_type', type=str, default='SLM_v1',
                        help='If `--cmp_type` is `combo`, it chooses which compatibility to use!\
            \nAbbreviations: (LIN=lines, MOT=motif, SH=shape, COL=color, SEG=segmentation)\
            \nFor example, SH-MOT is motif+shape, SH-SEG is shape+segmentation',
                        choices=['SH-SEG', 'SH-MOT', 'SH-LIN', 'SLM_v1', 'SLMS_v2', 'SLMS_version3', 'SH-AggMOT'])
    parser.add_argument('--border_len', type=int, default=-1,
                        help='length of border (if -1 [default] it will be set to xy_step)')

    args = parser.parse_args()

    main(args)

