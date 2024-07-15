import argparse
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pdb, os, json
from scipy.io import loadmat, savemat
import datetime
import matplotlib.pyplot as plt
import time

from PIL import Image
from ultralytics import YOLO

# internal
from configs import folder_names as fnames
from puzzle_utils.shape_utils import prepare_pieces_v2, create_grid, include_shape_info, \
    encode_boundary_segments
from puzzle_utils.pieces_utils import calc_parameters_v2, CfgParameters
from puzzle_utils.visualization import save_vis
#from puzzle_utils.lines_ops import compute_cost_wrapper, calc_line_matching_parameters

from compatibility_Motifs import compute_cost_wrapper_for_Motifs_compatibility
from compatibility_MGC import compute_cost_wrapper_for_Colors_compatibility


def reshape_list2mat_and_normalize(comp_as_list, n, norm_value):
    first_element = comp_as_list[0]
    cost_matrix = np.zeros((first_element.shape[0], first_element.shape[1], first_element.shape[2], n, n))
    norm_cost_matrix = np.zeros((first_element.shape[0], first_element.shape[1], first_element.shape[2], n, n))
    for i in range(n):
        for j in range(n):
            ji_mat = comp_as_list[i * n + j]
            cost_matrix[:, :, :, j, i] = ji_mat
            # norm_cost_matrix[:,:,:,j,i] = np.maximum(1 - ji_mat / norm_value, 0)
    return cost_matrix, norm_cost_matrix


def main(args):
    print("Compatibility log\nSearch for `CMP_START_TIME` or `CMP_END_TIME` if you want to see which images are done")

    ## IF to add
    yolov8_model_path = '/home/marina/PycharmProjects/RL_puzzle_solver/yolov5/best.pt'
    yolov8_obb_detector = YOLO(yolov8_model_path)

    if args.puzzle == '':
        puzzles = os.listdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset))
        puzzles = [puz for puz in puzzles if
                   os.path.isdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset, puz)) is True]
    else:
        puzzles = [args.puzzle]

    print(f"\nWill calculate compatibility matrices for: {puzzles}\n")
    for puzzle in puzzles:

        time_start_puzzle = time.time()
        ######
        # PREPARE PIECES AND GRIDS
        #
        # pieces is a list of dictionaries with the pieces (and mask, cm, id)
        # img_parameters contains the size of the image and of the pieces
        # ppars contains all the values needed for computing stuff (p_hs, comp_range..)
        # ppars is a dict but can be accessed by pieces_paramters.property!
        print()
        print("-" * 50)
        print("-- CMP_START_TIME -- ")
        # get the current date and time
        now = datetime.datetime.now()
        print(f"{now}\nStarted working on {puzzle}")
        print(f"Dataset: {args.dataset}")
        print("-" * 50)
        print("\tPIECES")
        pieces, img_parameters = prepare_pieces_v2(fnames, args.dataset, puzzle, verbose=True)
        print("-" * 50)
        print('\tIMAGE PARAMETERS')
        for cfg_key in img_parameters.keys():
            print(f"{cfg_key}: {img_parameters[cfg_key]}")

        puzzle_root_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset, puzzle)
        cmp_parameter_path = os.path.join(puzzle_root_folder, 'compatibility_parameters.json')
        if os.path.exists(cmp_parameter_path):
            ppars = CfgParameters()
            with open(cmp_parameter_path, 'r') as cp:
                ppars_dict = json.load(cp)
            print("-" * 50)
            print('\tCOMPATIBILITY PARAMETERS')
            for ppk in ppars_dict.keys():
                ppars[ppk] = ppars_dict[ppk]
                print(f"{ppk}: {ppars[ppk]}")
        else:
            print("\n" * 3)
            print("/" * 70)
            print("/\t***ERROR***\n/ compatibility_parameters.json not found!")
            print("/" * 70)
            print("\n" * 3)
            ppars = calc_parameters_v2(img_parameters, args.xy_step, args.xy_grid_points, args.theta_step)

        # line_matching_parameters = calc_line_matching_parameters(ppars, args.cmp_cost)
        # print("-" * 50)
        # print('\tLINE MATCHING PARAMETERS')
        # for cfg_key in line_matching_parameters.keys():
        #     print(f"{cfg_key}: {line_matching_parameters[cfg_key]}")
        # print("-" * 50)
        # line_matching_parameters_path = os.path.join(puzzle_root_folder, 'line_matching_parameters.json')
        # with open(line_matching_parameters_path, 'w') as lmpj:
        #     json.dump(line_matching_parameters, lmpj, indent=3)
        # print("saved json line matching file")

        if args.border_len < 0:
            seg_len = ppars.xy_step
        else:
            seg_len = args.border_len
        print("Using border length of {seg_len} pixels")
        pieces = include_shape_info(fnames, pieces, args.dataset, puzzle, args.det_method, line_based=False)
        #pieces = encode_boundary_segments(pieces, fnames, args.dataset, puzzle, boundary_seg_len=seg_len,
        #                                  boundary_thickness=2)

        region_mask_mat = loadmat(
            os.path.join(os.getcwd(), fnames.output_dir, args.dataset, puzzle, fnames.rm_output_name,
                         f'RM_{puzzle}.mat'))
        region_mask = region_mask_mat['RM']

        # parameters and grid
        p = [ppars.p_hs, ppars.p_hs]  # center of piece [125,125] - ref.point for lines
        m_size = ppars.xy_grid_points  # 101X101 grid
        m = np.zeros((m_size, m_size, 2))
        m2, m1 = np.meshgrid(np.linspace(-1, 1, m_size), np.linspace(-1, 1, m_size))
        m[:, :, 0] = m1
        m[:, :, 1] = m2
        z_rad = ppars.pairwise_comp_range // 2
        z_id = m * z_rad
        ang = ppars.theta_step
        if ang == 0:
            rot = [0]
        else:
            rot = np.arange(0, 360 - ang + 1, ang)
        #cmp_parameters = (p, z_id, m, rot, line_matching_parameters) #for lines
        cmp_parameters = (p, z_id, m, rot)
        n = len(pieces)

        # COST MATRICES
        All_cost = np.zeros((m.shape[1], m.shape[1], len(rot), n, n))
        All_norm_cost = np.zeros((m.shape[1], m.shape[1], len(rot), n, n))

        # check sizes
        if region_mask.shape[2] != All_norm_cost.shape[2]:
            step = region_mask.shape[2] / All_norm_cost.shape[2]
            if np.abs(step - int(step)) > 0:
                print('WRONG THETA STEP')
                print("SKIPPING")
                return 0
            else:
                step = int(step)
                print(
                    f"Seems compatibility has few values of rotation, using only a part of the region mask, each {step} values")
                region_mask = region_mask[:, :, ::step, :, :]
                print("now region mask shape is:", region_mask.shape)

        # TO BE PARALLELIZED
        if args.jobs > 1:
            print(f'running {args.jobs} parallel jobs with multiprocessing')
            # pool = multiprocessing.Pool(args.jobs)
            # costs_list = zip(*pool.map(compute_cost_matrix_LAP, [(i, j, pieces, region_mask, cmp_parameters, ppars) for j in range(n) for i in range(n)]))
            # with parallel_backend('threading', n_jobs=args.jobs):
            costs_list = Parallel(n_jobs=args.jobs, prefer="threads")(
                delayed(compute_cost_wrapper)(i, j, pieces, region_mask, cmp_parameters, ppars,
                                              verbosity=args.verbosity) for i in range(n) for j in
                range(n))  ## is something change replacing j and i ???
            #All_cost, All_norm_cost = reshape_list2mat_and_normalize(costs_list, n=n,
             #                                                        norm_value=line_matching_parameters.rmax)
        else:
            for i in range(n):  # select fixed fragment
                for j in range(n):
                    if args.verbosity == 1:
                        print(f"Computing compatibility between piece {i:04d} and piece {j:04d}..", end='\r')
                    # ji_mat = compute_cost_wrapper(i, j, pieces, region_mask, cmp_parameters, ppars, verbosity=args.verbosity)
                    # FOR TEST ONLY ####
                    # ji_mat = compute_cost_wrapper_for_Colors_compatibility(i, j, pieces, region_mask, cmp_parameters,
                    #                                                        ppars, seg_len,
                    #                                                        verbosity=args.verbosity)

                    ji_mat = compute_cost_wrapper_for_Motifs_compatibility(i, j, pieces, region_mask, cmp_parameters, ppars, yolov8_obb_detector, verbosity=1)

                    if i != j and args.DEBUG is True:
                        rotation_idx = 0
                        plt.suptitle(f"COST WITH {args.cmp_cost}", fontsize=45)
                        plt.subplot(541)
                        plt.imshow(pieces[i]['img'])
                        plt.title(f"piece {i}")
                        plt.subplot(542)
                        plt.imshow(pieces[j]['img'])
                        plt.title(f"piece {j}")
                        plt.subplot(545)
                        plt.imshow(region_mask[:, :, 0, j, i], vmin=-1, vmax=1, cmap='RdYlGn')
                        plt.title("region map 0")
                        plt.subplot(546)
                        plt.imshow(region_mask[:, :, 1, j, i], vmin=-1, vmax=1, cmap='RdYlGn')
                        plt.title("region map 1")
                        plt.subplot(547)
                        plt.imshow(region_mask[:, :, 2, j, i], vmin=-1, vmax=1, cmap='RdYlGn')
                        plt.title("region map 2")
                        plt.subplot(548)
                        plt.imshow(region_mask[:, :, 3, j, i], vmin=-1, vmax=1, cmap='RdYlGn')
                        plt.title("region map 3")
                        # plt.subplot(546); plt.imshow(ji_mat[:,:,rotation_idx], cmap='RdYlGn'); plt.title("cost")
                        # if args.cmp_cost == 'LCI':
                        #     norm_cmp = ji_mat[:,:,0] / np.max(ji_mat[:,:,0]) #np.maximum(1 - ji_mat[:,:,0] / line_matching_parameters.rmax, 0)
                        #     plt.subplot(547); plt.imshow(norm_cmp, vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("compatibility")
                        #     plt.subplot(548); plt.imshow(norm_cmp + np.minimum(region_mask[:,:,rotation_idx,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("final cmp")
                        # else:
                        #     norm_cmp = np.maximum(1 - ji_mat[:,:,0] / line_matching_parameters.rmax, 0)
                        #     plt.subplot(547); plt.imshow(norm_cmp, vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("compatibility")
                        #     plt.subplot(548); plt.imshow(norm_cmp + np.minimum(region_mask[:,:,rotation_idx,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("final cmp")

                        plt.subplot(549);
                        plt.title("COST ROTATION 0")
                        plt.imshow(ji_mat[:, :, 0], cmap='RdYlGn');
                        plt.subplot(5, 4, 10);
                        plt.title("COST ROTATION 1")
                        plt.imshow(ji_mat[:, :, 1], cmap='RdYlGn');
                        plt.subplot(5, 4, 11);
                        plt.title("COST ROTATION 2")
                        plt.imshow(ji_mat[:, :, 2], cmap='RdYlGn');
                        plt.subplot(5, 4, 12);
                        plt.title("COST ROTATION 3")
                        plt.imshow(ji_mat[:, :, 3], cmap='RdYlGn');

                        ji_mat[
                            ji_mat > line_matching_parameters.badmatch_penalty] = line_matching_parameters.badmatch_penalty
                        ji_unique_values = np.unique(ji_mat)
                        k = min(line_matching_parameters.k, len(ji_unique_values))
                        kmin_cut_val = np.sort(ji_unique_values)[-k]
                        if args.cmp_cost == 'LAP':
                            plt.subplot(5, 4, 13)
                            plt.title("COST ROTATION KMINCUT 0")
                            plt.imshow(np.maximum(1 - ji_mat[:, :, 0] / kmin_cut_val, 0), cmap='RdYlGn');
                            plt.subplot(5, 4, 14)
                            plt.title("COST ROTATION KMINCUT 1")
                            plt.imshow(np.maximum(1 - ji_mat[:, :, 1] / kmin_cut_val, 0), cmap='RdYlGn')
                            plt.subplot(5, 4, 15)
                            plt.title("COST ROTATION KMINCUT 2")
                            plt.imshow(np.maximum(1 - ji_mat[:, :, 2] / kmin_cut_val, 0), cmap='RdYlGn')
                            plt.subplot(5, 4, 16)
                            plt.title("COST ROTATION KMINCUT 3")
                            plt.imshow(np.maximum(1 - ji_mat[:, :, 3] / kmin_cut_val, 0), cmap='RdYlGn')

                            plt.subplot(5, 4, 17)
                            plt.title("EXP ROTATION 0")
                            plt.imshow(np.exp(-ji_mat[:, :, 0] / 76), cmap='RdYlGn')
                            # plt.imshow(norm_cmp + np.minimum(region_mask[:,:,0,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn');
                            plt.subplot(5, 4, 18)
                            plt.title("EXP ROTATION 1")
                            plt.imshow(np.exp(-ji_mat[:, :, 1] / 76), cmap='RdYlGn')
                            # plt.imshow(norm_cmp + np.minimum(region_mask[:,:,1,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn');
                            plt.subplot(5, 4, 19)
                            plt.title("EXP ROTATION 2")
                            plt.imshow(np.exp(-ji_mat[:, :, 2] / 76), cmap='RdYlGn')
                            # plt.imshow(norm_cmp + np.minimum(region_mask[:,:,2,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn');
                            plt.subplot(5, 4, 20)
                            plt.title("EXP ROTATION 3")
                            plt.imshow(np.exp(-ji_mat[:, :, 3] / 76), cmap='RdYlGn')
                            # plt.imshow(norm_cmp + np.minimum(region_mask[:,:,3,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn');
                        plt.show()
                        pdb.set_trace()
                    All_cost[:, :, :, j, i] = ji_mat

            def index(array, item):
                for idx, val in np.ndenumerate(array):
                    if val == item:
                        return idx

            k = args.k
            All_cost_cut = np.zeros((All_cost.shape))
            a_ks = np.zeros((region_mask.shape[0], region_mask.shape[1], n))
            a_min = np.zeros((region_mask.shape[0], region_mask.shape[1], n))

            for i in range(n):
                a_cost_i = All_cost[:, :, :, :, i]
                for x in range(a_cost_i.shape[0]):
                    for y in range(a_cost_i.shape[1]):
                        a_xy = a_cost_i[x, y, :, :]
                        a_all = np.array(np.unique(a_xy))
                        a = a_all[np.minimum(k, len(a_all) - 1)]
                        a_xy = np.where(a_xy > a, -1, a_xy)
                        a_cost_i[x, y, :, :] = a_xy
                        a_ks[x, y, i] = a
                        if len(a_all) > 1:
                            a_min[x, y, i] = a_all[1]
                print(a_ks[:, :, i])
                All_cost_cut[:, :, :, :, i] = a_cost_i

            norm_term = np.max(a_ks) / (2 * k)
            All_norm_cost = 2 - All_cost_cut / norm_term  # only for colors
            All_norm_cost = np.where(All_norm_cost > 2, 0, All_norm_cost)  # only for colors
            if args.norm_type == 'negative':
                All_norm_cost = np.where(All_norm_cost <= 0, -1, All_norm_cost)  ## NEW idea di Prof.Pelillo
            else:
                All_norm_cost = np.where(All_norm_cost < 0, 0, All_norm_cost)  # only for colors


            ## BEST BUDDIES
            All_cost_BB = np.zeros((All_cost.shape))
            bb_bonus = 3
            for i in range(n-1):
                for j in range(i+1,n,1):
                    a_cost_i = All_cost[:, :, :, j, i]
                    amin_1 = np.array(np.unique(a_cost_i))[1]  ## min = -1  second_min is best

                    a_cost_j = All_cost[:, :, :, i, j]
                    amin_2 = np.array(np.unique(a_cost_j))[1]
                    idx_1 = index(a_cost_i, amin_1)
                    all_val_1 = All_cost[idx_1[0], idx_1[1], idx_1[2], :, i]
                    v1 = np.min(all_val_1)
                    if v1 == -1:
                        v1 = np.array(np.unique(all_val_1))[1]

                    idx_2 = index(a_cost_j, amin_2)
                    all_val_2 = All_cost[idx_2[0], idx_2[1], idx_2[2], :, j]
                    v2 = np.min(all_val_2)
                    if v2 == -1:
                        v2 = np.array(np.unique(all_val_2))[1]

                    if v1 == amin_1 and v2 == amin_2:
                        if v1 >= norm_term and v2 >= norm_term:
                            All_cost_BB[idx_1[0], idx_1[1], idx_1[2], j, i] = bb_bonus
                            All_cost_BB[idx_2[0], idx_2[1], idx_2[2], i, j] = bb_bonus

        # NEW OPTION with Negative compatibilities
        only_zero_region = np.abs(region_mask)  # recover zero aria in region matrix
        R_line_initial = All_norm_cost * only_zero_region
        R_line = R_line_initial + All_cost_BB

        print("-" * 50)
        time_in_seconds = time.time() - time_start_puzzle
        if time_in_seconds > 60:
            time_in_minutes = (np.ceil(time_in_seconds / 60))
            if time_in_minutes < 60:
                print(f"Compatibility for this puzzle took almost {time_in_minutes:.0f} minutes")
            else:
                time_in_hours = (np.ceil(time_in_minutes / 60))
                print(f"Compatibility for this puzzle took almost {time_in_hours:.0f} hours")
        else:
            print(f"Compatibility for this puzzle took {time_in_seconds:.0f} seconds")
        print("-" * 50)
        # save output
        output_folder = os.path.join(fnames.output_dir, args.dataset, puzzle, fnames.cm_output_name)
        os.makedirs(output_folder, exist_ok=True)
        filename = os.path.join(output_folder, f'CM_{args.det_method}_{args.norm_type}_colors_BB')
        #filename = os.path.join(output_folder, f'CM_linesdet_{args.det_method}_cost_{args.cmp_cost}')
        mdic = {
            "R_line": R_line,
            "label": "label",
            "method": args.det_method,
            "cost": args.cmp_cost,
            "xy_step": ppars.xy_step,
            "xy_grid_points": ppars.xy_grid_points,
            "theta_step": ppars.theta_step
        }
        savemat(f'{filename}.mat', mdic)
        np.save(filename, R_line)

        if args.save_everything is True:
            filename = os.path.join(output_folder, f'CM_all_cost_lines_{args.det_method}_cost_{args.cmp_cost}')
            mdic = {
                "All_cost": All_cost,
                "label": "label",
                "method": args.det_method,
                "cost": args.cmp_cost,
                "xy_step": ppars.xy_step,
                "xy_grid_points": ppars.xy_grid_points,
                "theta_step": ppars.theta_step
            }
            savemat(f'{filename}.mat', mdic)
            np.save(filename, All_cost)

        vis_folder = os.path.join(output_folder, fnames.visualization_folder_name)
        os.makedirs(vis_folder, exist_ok=True)
        if args.save_visualization is True:
            print('Creating visualization')
            save_vis(R_line, pieces, ppars.theta_step, os.path.join(vis_folder,
                                                                    f'visualization_{puzzle}_linesdet_{args.det_method}_cost_{args.cmp_cost}_{m.shape[1]}x{m.shape[1]}x{len(rot)}x{n}x{n}'),
                     f"compatibility matrix {puzzle}", all_rotation=True, vmin=-2, vmax=10)
            if args.save_everything:
                save_vis(All_cost, pieces, ppars.theta_step, os.path.join(vis_folder,
                                                                          f'visualization_overlap_{puzzle}_linesdet_{args.det_method}_cost_{args.cmp_cost}_{m.shape[1]}x{m.shape[1]}x{len(rot)}x{n}x{n}'),
                         f"cost matrix {puzzle}", all_rotation=True, vmin=-2, vmax=2)

        print("-" * 50)
        print("-- CMP_END_TIME -- ")
        # get the current date and time
        now = datetime.datetime.now()
        print(f"{now}")
        print(f'Done with {puzzle}\n')
        print("-" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computing compatibility matrix')  # add some discription
    parser.add_argument('--dataset', type=str, default='repair',
                        help='dataset folder')  # repair
    parser.add_argument('--puzzle', type=str, default='repair_g28',
                        help='puzzle folder (if empty will do all folders inside the dataset folder)')  # repair_g97, repair_g28, decor_1_lines
    parser.add_argument('--det_method', type=str, default='exact',
                        help='method line detection')  # exact, manual, deeplsd
    parser.add_argument('--penalty', type=int, default=-1,
                        help='penalty (leave -1 to use the one from the config file)')
    parser.add_argument('--jobs', type=int, default=0, help='how many jobs (if you want to parallelize the execution')
    parser.add_argument('--save_visualization', type=bool, default=True,
                        help='save an image that showes the matrices color-coded')
    parser.add_argument('--save_everything', default=False, action='store_true',
                        help='use to save debug matrices (may require up to ~8 GB per solution, use with care!)')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')
    parser.add_argument('--cmp_cost', type=str, default='colors_BB', help='cost computation')  # LAP, LCI, colors, colors_BB
    parser.add_argument('--norm_type', type=str, default='simple', help='cost computation')  # negative, simple
    parser.add_argument('--border_len', type=int, default=-1,
                        help='length of border (if -1 [default] it will be set to xy_step)')
    parser.add_argument('-k', type=int, default=11,
                        help='keep the best k values (for given gamma transformation) in the compatibility')
    # parser.add_argument('--xy_step', type=int, default=30, help='the step (in pixels) between each grid point')
    # parser.add_argument('--xy_grid_points', type=int, default=7,
    #     help='the number of points in the grid (for each axis, total number will be the square of what is given)')
    # parser.add_argument('--theta_step', type=int, default=90, help='degrees of each rotation')
    parser.add_argument('--DEBUG', action='store_true', default=False,
                        help='WARNING: will use debugger! It stops and show the matrices!')

    args = parser.parse_args()
    main(args)
