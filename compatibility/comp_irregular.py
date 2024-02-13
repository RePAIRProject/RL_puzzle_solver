import argparse
from joblib import Parallel, delayed
import multiprocessing
import numpy as np 
import pdb, os, json
from scipy.io import loadmat, savemat
import datetime
import matplotlib.pyplot as plt 
import time 

# internal
from configs import folder_names as fnames
from puzzle_utils.shape_utils import prepare_pieces_v2, create_grid, include_shape_info
from puzzle_utils.pieces_utils import calc_parameters_v2, CfgParameters
from puzzle_utils.visualization import save_vis
from puzzle_utils.lines_ops import compute_cost_wrapper, calc_line_matching_parameters

def reshape_list2mat_and_normalize(comp_as_list, n, norm_value):
    first_element = comp_as_list[0]
    cost_matrix = np.zeros((first_element.shape[0], first_element.shape[1], first_element.shape[2], n, n))
    norm_cost_matrix = np.zeros((first_element.shape[0], first_element.shape[1], first_element.shape[2], n, n))
    for i in range(n):
        for j in range(n):
            ji_mat = comp_as_list[i*n + j]
            cost_matrix[:,:,:,j,i] = ji_mat
            norm_cost_matrix[:,:,:,j,i] = np.maximum(1 - ji_mat / norm_value, 0)
    return cost_matrix, norm_cost_matrix

def main(args):

    print("Compatibility log\nSearch for `CMP_START_TIME` or `CMP_END_TIME` if you want to see which images are done")

    if args.puzzle == '':  
        puzzles = os.listdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset))
        puzzles = [puz for puz in puzzles if os.path.isdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset,puz)) is True]
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
            print('\COMPATIBILITY PARAMETERS')
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


        line_matching_parameters = calc_line_matching_parameters(ppars, args.cmp_cost)
        print("-" * 50)
        print('\tLINE MATCHING PARAMETERS')
        for cfg_key in line_matching_parameters.keys():
            print(f"{cfg_key}: {line_matching_parameters[cfg_key]}")
        print("-" * 50)
        line_matching_parameters_path = os.path.join(puzzle_root_folder, 'line_matching_parameters.json')
        with open(line_matching_parameters_path, 'w') as lmpj:
            json.dump(line_matching_parameters, lmpj, indent=3)
        print("saved json line matching  file")

        pieces = include_shape_info(fnames, pieces, args.dataset, puzzle, args.det_method)

        region_mask_mat = loadmat(os.path.join(os.getcwd(), fnames.output_dir, args.dataset, puzzle, fnames.rm_output_name, f'RM_{puzzle}.mat'))
        region_mask = region_mask_mat['RM']
        
        # parameters and grid
        p = [ppars.p_hs, ppars.p_hs]    # center of piece [125,125] - ref.point for lines
        m_size = ppars.xy_grid_points   # 101X101 grid
        m = np.zeros((m_size, m_size, 2))
        m2, m1 = np.meshgrid(np.linspace(-1, 1, m_size), np.linspace(-1, 1, m_size))
        m[:, :, 0] = m1
        m[:, :, 1] = m2
        z_rad = ppars.pairwise_comp_range // 2
        z_id = m * z_rad
        ang = ppars.theta_step
        rot = np.arange(0, 360 - ang + 1, ang)
        cmp_parameters = (p, z_id, m, rot, line_matching_parameters)
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
                print(f"Seems compatibility has few values of rotation, using only a part of the region mask, each {step} values")
                region_mask = region_mask[:,:,::step,:,:]
                print("now region mask shape is:", region_mask.shape)
                

        # TO BE PARALLELIZED
        if args.jobs > 1:
            print(f'running {args.jobs} parallel jobs with multiprocessing')
            #pool = multiprocessing.Pool(args.jobs)
            #costs_list = zip(*pool.map(compute_cost_matrix_LAP, [(i, j, pieces, region_mask, cmp_parameters, ppars) for j in range(n) for i in range(n)]))
            #with parallel_backend('threading', n_jobs=args.jobs):
            costs_list = Parallel(n_jobs=args.jobs, prefer="threads")(delayed(compute_cost_wrapper)(i, j, pieces, region_mask, cmp_parameters, ppars, verbosity=args.verbosity, use_colors=args.use_colors) for i in range(n) for j in range(n)) ## is something change replacing j and i ???
            #costs_list = Parallel(n_jobs=args.jobs)(delayed(compute_cost_matrix_LAP)(i, j, pieces, region_mask, cmp_parameters, ppars) for j in range(n) for i in range(n))
            All_cost, All_norm_cost = reshape_list2mat_and_normalize(costs_list, n=n, norm_value=line_matching_parameters.rmax)
        else:
            for i in range(n):  # select fixed fragment
                for j in range(n):
                    if args.verbosity == 1:
                        print(f"Computing compatibility between piece {i:04d} and piece {j:04d}..", end='\r')
                    ji_mat = compute_cost_wrapper(i, j, pieces, region_mask, cmp_parameters, ppars, verbosity=args.verbosity, use_colors=args.use_colors)
                    if i != j and args.DEBUG is True and j == 2 and i == 0:
                        plt.subplot(321); plt.imshow(pieces[i]['img']); plt.title(f"piece {i}")
                        plt.subplot(322); plt.imshow(pieces[j]['img']); plt.title(f"piece {j}")
                        plt.subplot(323); plt.imshow(region_mask[:,:,0,i,j], vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("region map")
                        plt.subplot(324); plt.imshow(ji_mat[:,:,0], vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("cost")
                        if args.cmp_cost == 'LCI':
                            norm_cmp = ji_mat[:,:,0] / np.max(ji_mat[:,:,0]) #np.maximum(1 - ji_mat[:,:,0] / line_matching_parameters.rmax, 0)
                            plt.subplot(325); plt.imshow(norm_cmp, vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("compatibility")
                            plt.subplot(326); plt.imshow(norm_cmp + np.minimum(region_mask[:,:,0,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("final cmp")
                        else:
                            norm_cmp = np.maximum(1 - ji_mat[:,:,0] / line_matching_parameters.rmax, 0)
                            plt.subplot(325); plt.imshow(norm_cmp, vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("compatibility")
                            plt.subplot(326); plt.imshow(norm_cmp + np.minimum(region_mask[:,:,0,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("final cmp")
                        plt.show()
                        pdb.set_trace()
                    All_cost[:, :, :, j, i] = ji_mat


        if args.cmp_cost == 'LCI':
            All_norm_cost = All_cost/np.max(All_cost)  # normalize to max value TODO !!!
        else:  # args.cmp_cost == 'LAP':
            All_norm_cost = np.maximum(1 - All_cost / line_matching_parameters.rmax, 0)

        only_negative_region = np.minimum(region_mask, 0)  # recover overlap (negative) areas
        R_line = All_norm_cost + only_negative_region  # insert negative regions to cost matrix

        # it should not be needed
        # R_line = (All_norm_cost * region_mask) * 2
        # R_line[R_line < 0] = -1
        # for jj in range(n):
        #     R_line[:, :, :, jj, jj] = -1
        print("-" * 50)
        print(f"Compatibility for this puzzle took {(time.time()-time_start_puzzle):.02f} seconds")
        print("-" * 50)
        # save output
        output_folder = os.path.join(fnames.output_dir, args.dataset, puzzle, fnames.cm_output_name)
        os.makedirs(output_folder, exist_ok=True)
        filename = os.path.join(output_folder, f'CM_linesdet_{args.det_method}_cost_{args.cmp_cost}')
        mdic = {
                    "R_line": R_line, 
                    "label": "label", 
                    "method":args.det_method, 
                    "cost":args.cmp_cost, 
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
                        "method":args.det_method, 
                        "cost":args.cmp_cost, 
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
            save_vis(R_line, pieces, ppars.theta_step, os.path.join(vis_folder, f'visualization_{puzzle}_linesdet_{args.det_method}_cost_{args.cmp_cost}_{m.shape[1]}x{m.shape[1]}x{len(rot)}x{n}x{n}'), f"compatibility matrix {puzzle}", all_rotation=True)
            if args.save_everything:
                save_vis(All_cost, pieces, ppars.theta_step, os.path.join(vis_folder, f'visualization_overlap_{puzzle}_linesdet_{args.det_method}_cost_{args.cmp_cost}_{m.shape[1]}x{m.shape[1]}x{len(rot)}x{n}x{n}'), f"cost matrix {puzzle}", all_rotation=True)
        
        print("-" * 50)
        print("-- CMP_END_TIME -- ")
        # get the current date and time
        now = datetime.datetime.now()
        print(f"{now}")
        print(f'Done with {puzzle}\n')
        print("-" * 50)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Computing compatibility matrix')  # add some discription
    parser.add_argument('--dataset', type=str, default='synthetic_irregular_9_pieces_by_drawing_coloured_lines_peynrh', help='dataset folder')  # repair
    parser.add_argument('--puzzle', type=str, default='image_00000', help='puzzle folder (if empty will do all folders inside the dataset folder)')  # repair_g97, repair_g28, decor_1_lines
    parser.add_argument('--det_method', type=str, default='exact', help='method line detection')  # exact, manual, deeplsd
    parser.add_argument('--penalty', type=int, default=-1, help='penalty (leave -1 to use the one from the config file)')
    parser.add_argument('--jobs', type=int, default=0, help='how many jobs (if you want to parallelize the execution')
    parser.add_argument('--save_visualization', type=bool, default=True, help='save an image that showes the matrices color-coded')
    parser.add_argument('--save_everything', default=False, action='store_true',
        help='use to save debug matrices (may require up to ~8 GB per solution, use with care!)')
    parser.add_argument('--verbosity', type=int, default=1, help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')
    parser.add_argument('--cmp_cost', type=str, default='LCI', help='cost computation')   
    # parser.add_argument('--xy_step', type=int, default=30, help='the step (in pixels) between each grid point')
    # parser.add_argument('--xy_grid_points', type=int, default=7, 
    #     help='the number of points in the grid (for each axis, total number will be the square of what is given)')
    # parser.add_argument('--theta_step', type=int, default=90, help='degrees of each rotation')
    parser.add_argument('--use_colors', type=bool, default=True, help='use colors of lines')
    parser.add_argument('--DEBUG', action='store_true', default=False, help='WARNING: will use debugger! It stops and show the matrices!')

    args = parser.parse_args()
    main(args)
