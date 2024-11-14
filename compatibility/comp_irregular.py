import argparse
from joblib import Parallel, delayed
import multiprocessing
import numpy as np 
import pdb, os, json
from scipy.io import loadmat, savemat
import datetime
import matplotlib.pyplot as plt 
import time
from ultralytics import YOLO


# internal
from configs import folder_names as fnames
from puzzle_utils.shape_utils import prepare_pieces_v2, create_grid, include_shape_info, encode_boundary_segments
from puzzle_utils.pieces_utils import calc_parameters_v2, CfgParameters
from puzzle_utils.visualization import save_vis
from puzzle_utils.regions import combine_region_masks
from utils import compute_cost_wrapper, calc_computation_parameters, normalize_CM, reshape_list2mat,\
    show_debug_visualization


def main(args):

    print("Compatibility log\nSearch for `CMP_START_TIME` or `CMP_END_TIME` if you want to see which images are done")

    ###########################
    #   ONE PUZZLE OR MULTIPLE
    ########################### 
    if args.puzzle == '':  
        puzzles = os.listdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset))
        puzzles.sort()
        puzzles = [puz for puz in puzzles if os.path.isdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset,puz)) is True]
    else:
        puzzles = [args.puzzle]

    print(f"\nWill calculate compatibility matrices for: {puzzles}\n")
    for puzzle in puzzles:

        time_start_puzzle = time.time()
        #################################
        #   PREPARE PIECES AND GRIDS
        # pieces is a list of dictionaries with the pieces (and mask, cm, id)
        # img_parameters contains the size of the image and of the pieces
        # ppars contains all the values needed for computing stuff (p_hs, comp_range..)
        # ppars is a dict but can be accessed by pieces_paramters.property!
        ################################# 
        print()
        print("-" * 60)
        print("-- CMP_START_TIME -- ")
        # get the current date and time
        now = datetime.datetime.now()
        print(f"{now}\nStarted working on {puzzle}")
        print(f"Dataset: {args.dataset}")
        print("-" * 60)
        print("\tPIECES")
        pieces, img_parameters = prepare_pieces_v2(fnames, args.dataset, puzzle, verbose=True)
        print("-" * 60)
        print('\tIMAGE PARAMETERS')
        for cfg_key in img_parameters.keys():
            print(f"{cfg_key}: {img_parameters[cfg_key]}")
        
        puzzle_root_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset, puzzle)
        cmp_parameter_path = os.path.join(puzzle_root_folder, 'compatibility_parameters_v2.json')
        #################################
        #   PARAMETERS (from .json file)
        #################################
        if os.path.exists(cmp_parameter_path):
            ppars = CfgParameters()
            with open(cmp_parameter_path, 'r') as cp:
                ppars_dict = json.load(cp)
            ppars_dict['cmp_type'] = args.cmp_type
            print("-" * 60)
            print('\tCOMPATIBILITY PARAMETERS')
            for ppk in ppars_dict.keys():
                ppars[ppk] = ppars_dict[ppk]
                print(f"{ppk}: {ppars[ppk]}")
        else:
            print("\n" * 3)
            print("/" * 60)
            print("/\t***ERROR***\n/ compatibility_parameters.json not found!")
            print("/" * 60)
            print("\n" * 3)
            ppars = calc_parameters_v2(img_parameters, args.xy_step, args.xy_grid_points, args.theta_step)

        ###########################
        #   ADDITIONAL PARAMETERS
        ########################### 
        additional_cmp_pars = calc_computation_parameters(ppars, cmp_type=args.cmp_type, \
            cmp_cost=args.cmp_cost, lines_det_method=args.lines_det_method, motif_det_method=args.motif_det_method)

        for parkey in additional_cmp_pars.keys():
            ppars[parkey] = additional_cmp_pars[parkey]
        ppars['cmp_type'] = args.cmp_type
        calc_sdf = False
        if args.cmp_type == 'shape':
            calc_sdf = True
        ppars['calc_sdf'] = calc_sdf
        line_based = False
        if args.cmp_type == 'lines':
            line_based = True
        ppars['line_based'] = line_based
        motif_based = False
        if args.cmp_type == 'motifs':
            motif_based = True
            if args.yolo_path == '':
                raise Exception("You are trying to use yolo-based motif compatibility without specifying the yolo model to be used.\
                    \nPlease set the path with `--yolo_path path_to_the_pt_model` and relaunch")
            yolov8_model_path = args.yolo_path
            ppars['yolo_path'] = yolov8_model_path
            yolov8_obb_detector = YOLO(yolov8_model_path)
        else:
            yolov8_obb_detector = None
        ppars['motif_based'] = motif_based
        color_based = False
        seg_len = 0
        if args.cmp_type == 'color':
            color_based = True
            if args.border_len < 0:
                seg_len = ppars.xy_step
            else:
                seg_len = args.border_len
        if color_based == True:
            print(f"Using border length of {seg_len} pixels")
        ppars['color_based'] = color_based
        ppars['seg_len'] = seg_len
        ppars['k'] = args.k

        print("-" * 60)
        print('\tUPDATED COMPATIBILITY PARAMETERS')
        for cfg_key in ppars.keys():
            print(f"{cfg_key}: {ppars[cfg_key]}")
        print("-" * 60)
        computation_parameters_path = os.path.join(puzzle_root_folder, 'compatibility_parameters_v2.json')
        with open(computation_parameters_path, 'w') as lmpj:
            json.dump(ppars, lmpj, indent=3)
        print("saved json compatibility parameters file")
        
        ################################
        #   SHAPE INFORMATION (reading)
        ################################
        pieces = include_shape_info(fnames, pieces, args.dataset, puzzle, lines_det_method=args.lines_det_method, \
            motif_det_method=args.motif_det_method, line_based=line_based, sdf=calc_sdf, motif_based=motif_based)
        if color_based == True:
            pieces = encode_boundary_segments(pieces, fnames, args.dataset, puzzle, boundary_seg_len=seg_len,
                                         boundary_thickness=2)
        ###########################
        #   REGION MASK (reading)
        ########################### 
        region_mask_mat = loadmat(os.path.join(os.getcwd(), fnames.output_dir, args.dataset, puzzle, fnames.rm_output_name, f'RM_{puzzle}.mat'))
        shape_RM = region_mask_mat['RM_shapes']
        if line_based == True and motif_based == False:
            lines_RM = region_mask_mat['RM_lines']
            region_mask = combine_region_masks([shape_RM, lines_RM])
        elif line_based == False and motif_based == True:
            motif_RM = region_mask_mat['RM_motifs']
            region_mask = combine_region_masks([shape_RM, motif_RM])
        elif line_based == True and motif_based == True:
            lines_RM = region_mask_mat['RM_lines']
            motif_RM = region_mask_mat['RM_motifs']
            region_mask = combine_region_masks([shape_RM, motif_RM, lines_RM])
        else: # line_based == False and motif_based == False:
            region_mask = shape_RM        
        
        ###########################
        #   PARAMETERS AND GRID
        ############################ 
        p = [ppars.p_hs, ppars.p_hs]    
        m_size = ppars.xy_grid_points  
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
        ppars['p'] = p
        ppars['z_id'] = z_id
        ppars['m'] = m
        ppars['rot'] = rot
        n = len(pieces)

        ###########################
        #   COST MATRICES INIT
        ########################### 
        All_cost = np.zeros((m.shape[1], m.shape[1], len(rot), n, n))
        
        # check sizes
        if region_mask.shape[2] != All_cost.shape[2]:
            step = region_mask.shape[2] / All_cost.shape[2] 
            if np.abs(step - int(step)) > 0:
                print('WRONG THETA STEP')
                print("SKIPPING")
                return 0
            else:
                step = int(step)
                print(f"Seems compatibility has few values of rotation, using only a part of the region mask, each {step} values")
                region_mask = region_mask[:,:,::step,:,:]
                print("now region mask shape is:", region_mask.shape)
                
        ################################
        #   COMPATIBILITY COMPUTATION
        ################################
        if args.jobs > 1: # parallelized version!
            print("### WARNING ###\nIn case of issues, re-run with `jobs 0` (default) to avoid parallel jobs!")
            print(f'running {args.jobs} parallel jobs with multiprocessing')
            costs_list = Parallel(n_jobs=args.jobs, prefer="threads")(delayed(compute_cost_wrapper)(i, j, pieces, region_mask, ppars, compatibility_type=args.cmp_type, verbosity=args.verbosity) for i in range(n) for j in range(n)) ## is something change replacing j and i ???
            All_cost = reshape_list2mat(costs_list, n=n)
        else:
            # standard (for-loop) version
            for i in range(n):  # select fixed fragment
                for j in range(n):
                    if args.verbosity == 1:
                        print(f"Computing compatibility between piece {i:04d} and piece {j:04d}..", end='\r')
                    ji_mat = compute_cost_wrapper(i, j, pieces, region_mask, ppars, \
                        detector=yolov8_obb_detector, seg_len=seg_len,
                        verbosity=args.verbosity)
                    print(np.max(ji_mat))
                    print(np.unique(ji_mat))
                    All_cost[:, :, :, j, i] = ji_mat

                    if i > 1 and i != j and args.DEBUG == True:
                        show_debug_visualization(pieces, i, j, args, ji_mat, region_mask, ppars)

        ###########################
        #   NORMALIZATION
        ###########################
        # here we have the full matrix and we normalize it
        R = normalize_CM(All_cost, ppars, region_mask)

        ###########################
        #   TIMING
        ###########################                    
        print("-" * 60)
        time_in_seconds = time.time()-time_start_puzzle
        if time_in_seconds > 60:
            time_in_minutes = (np.ceil(time_in_seconds / 60))
            if time_in_minutes < 60:
                print(f"Compatibility for this puzzle took almost {time_in_minutes:.0f} minutes")
            else:
                time_in_hours = (np.ceil(time_in_minutes / 60))
                print(f"Compatibility for this puzzle took almost {time_in_hours:.0f} hours")
        else:
            print(f"Compatibility for this puzzle took {time_in_seconds:.0f} seconds")
        print("-" * 60)
        
        ###########################
        #   OUTPUT
        ###########################
        output_folder = os.path.join(fnames.output_dir, args.dataset, puzzle, fnames.cm_output_name)
        os.makedirs(output_folder, exist_ok=True)
        if args.cmp_type == 'lines':
            cmp_name = f"linesdet_{args.lines_det_method}_cost_{args.cmp_cost}"
        elif args.cmp_type == 'shape':
            cmp_name = "shape"
        elif args.cmp_type == 'motifs':
            cmp_name = f"motifs_{args.motif_det_method}"
        elif args.cmp_type == 'color':
            cmp_name = f"color_border{seg_len}"
        else:
            cmp_name = f"cmp_{args.cmp_type}"
        filename = os.path.join(output_folder, f"CM_{cmp_name}")
        mdic = {
                    "R": R, 
                    "label": "label", 
                    "cmp_type":args.cmp_type, 
                    "cmp_cost":args.cmp_cost, 
                    "lines_det_method":args.lines_det_method, 
                    "motif_det_method":args.motif_det_method, 
                    "xy_step": ppars.xy_step, 
                    "xy_grid_points": ppars.xy_grid_points, 
                    "theta_step": ppars.theta_step
                }
        if color_based == True:
            mdic['seg_len'] = seg_len 
        savemat(f'{filename}.mat', mdic)
        np.save(filename, R)

        if args.save_everything is True:
            filename = os.path.join(output_folder, f'CM_all_cost_{cmp_name}')
            mdic = {
                        "All_cost": All_cost, 
                        "label": "label", 
                        "motif_det_method":args.motif_det_method, 
                        "lines_det_method":args.lines_det_method, 
                        "cost":args.cmp_cost, 
                        "xy_step": ppars.xy_step, 
                        "xy_grid_points": ppars.xy_grid_points, 
                        "theta_step": ppars.theta_step
                    }
            savemat(f'{filename}.mat', mdic)
            np.save(filename, All_cost)
        
        ###########################
        #   VISUALIZATION
        ###########################
        vis_folder = os.path.join(output_folder, fnames.visualization_folder_name)
        os.makedirs(vis_folder, exist_ok=True)
        if args.save_visualization is True:
            print('Creating visualization')
            save_vis(R, pieces, ppars.theta_step, os.path.join(vis_folder, f'visualization_{puzzle}_{cmp_name}_{m.shape[1]}x{m.shape[1]}x{len(rot)}x{n}x{n}'), f"compatibility matrix {puzzle}", all_rotation=True)
            if args.save_everything:
                save_vis(All_cost, pieces, ppars.theta_step, os.path.join(vis_folder, f'visualization_overlap_{puzzle}_{cmp_name}_{m.shape[1]}x{m.shape[1]}x{len(rot)}x{n}x{n}'), f"cost matrix {puzzle}", all_rotation=True, vmin=-2, vmax=2)
        
        print("-" * 60)
        print("-- CMP_END_TIME -- ")
        # get the current date and time
        now = datetime.datetime.now()
        print(f"{now}")
        print(f'Done with {puzzle}\n')
        print("-" * 60)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Computing compatibility matrix')  # add some discription
    parser.add_argument('--dataset', type=str, default='RePAIR_exp_batch3', help='dataset folder')  # repair
    parser.add_argument('--puzzle', type=str, default='', help='puzzle folder (if empty will do all folders inside the dataset folder)')  # repair_g97, repair_g28, decor_1_lines
    parser.add_argument('--penalty', type=int, default=-1, help='penalty (leave -1 to use the one from the config file)')
    parser.add_argument('--jobs', type=int, default=0, help='how many jobs (if you want to parallelize the execution')
    parser.add_argument('--save_visualization', type=bool, default=True, help='save an image that showes the matrices color-coded')
    parser.add_argument('--save_everything', default=False, action='store_true',
        help='use to save debug matrices (may require up to ~8 GB per solution, use with care!)')
    parser.add_argument('--verbosity', type=int, default=1, help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')

    # COMPATIBILITY PARAMETERS
    parser.add_argument('--cmp_type', type=str, default='motifs',
        help='Chooses the compaitbility to use.\nIf more than one should be used, select `combo`\
            \nIt is connected with `--cmp_cost` and `--det_method`!', 
        choices=['lines', 'shape', 'color', 'motifs'])
    parser.add_argument('--cmp_cost', type=str, default='LAP',
        help='Chooses the cost used to compute compatibility - it depends on the `--cmp_type`\
            \nUse LAP or LCI for lines, YOLO or overlap for motif, SDF for shape, MGC for color',
        choices=[
            'LAP', 'LAPvis', 'LCI', # line-based
            'YOLO_conf', 'overlap', # motif-based
            'SDF', # shape-based
            'MGC' # color-based
        ])
    parser.add_argument('--lines_det_method', type=str, default='deeplsd', 
        help='method for the feature detection (usually lines or motif)',
        choices=['exact', 'deeplsd', 'manual'])  
    parser.add_argument('--motif_det_method', type=str, default='yolo-obb',
        help='method for the feature detection (usually lines or motif)',
        choices=['yolo-obb', 'yolo-bbox', 'yolo-seg'])  
    parser.add_argument('--yolo_path', type=str, default='/home/marina/PycharmProjects/RL_puzzle_solver/yolov5/best.pt', help='yolo path (.pt model)')
    parser.add_argument('--border_len', type=int, default=-1, help='length of border (if -1 [default] it will be set to xy_step)')   
    parser.add_argument('--k', type=int, default=5, help='keep the best k values (for given gamma transformation) in the compatibility')   

        # exact, manual, deeplsd
    # parser.add_argument('--xy_step', type=int, default=30, help='the step (in pixels) between each grid point')
    # parser.add_argument('--xy_grid_points', type=int, default=7, 
    #     help='the number of points in the grid (for each axis, total number will be the square of what is given)')
    # parser.add_argument('--theta_step', type=int, default=90, help='degrees of each rotation')
    parser.add_argument('--DEBUG', action='store_true', default=False, help='WARNING: will use debugger! It stops and show the matrices!')

    args = parser.parse_args()
    main(args)
