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
from puzzle_utils.shape_utils import prepare_pieces_v2, create_grid, include_shape_info, encode_boundary_segments
from puzzle_utils.pieces_utils import calc_parameters_v2, CfgParameters
from puzzle_utils.visualization import save_vis
from utils import compute_cost_wrapper, calc_computation_parameters

from compatibility_Segmentation import Segmentator

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

        additional_cmp_pars = calc_computation_parameters(ppars, cmp_type=args.cmp_type, \
            cmp_cost=args.cmp_cost, det_method=args.det_method)
        
        ############################## Saving puzzle root folder #############################
        ppars['puzzle_root_folder'] = puzzle_root_folder

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
            if args.det_method == 'yolo-bbox':
                from ultralytics import YOLOv10 as YOLO
            elif args.det_method == 'yolo-obb':
                from ultralytics import YOLO
            if args.yolo_path == '':
                raise Exception("You are trying to use yolo-based motif compatibility without specifying the yolo model to be used.\
                    \nPlease set the path with `--yolo_path path_to_the_pt_model` and relaunch")
            yolov8_model_path = args.yolo_path
            yolov8_obb_detector = YOLO(yolov8_model_path)
            ppars['yolo_path'] = yolov8_model_path
        else:
            yolov8_obb_detector = None
        ppars['motif_based'] = motif_based
        ####### Segmentation #######
        ppars['seg_based'] = False
        if args.cmp_type == 'seg':
            ppars['seg_based'] = True
            segmentator = Segmentator(ppars,args)
        ###### Color ######
        color_based = False
        seg_len = 0
        if args.cmp_type == 'color':
            color_based = True
            if args.border_len < 0:
                seg_len = ppars.xy_step
            else:
                seg_len = args.border_len
        print("Using border length of {seg_len} pixels")
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

        
        pieces = include_shape_info(fnames, pieces, args.dataset, puzzle, args.det_method, \
            line_based=line_based, sdf=calc_sdf, motif_based=motif_based)
        if color_based == True:
            pieces = encode_boundary_segments(pieces, fnames, args.dataset, puzzle, boundary_seg_len=seg_len,
                                         boundary_thickness=2)
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
        if ang == 0:
            rot = [0]
        else:
            rot = np.arange(0, 360 - ang + 1, ang)
        ppars['p'] = p
        ppars['z_id'] = z_id
        ppars['m'] = m
        ppars['rot'] = rot
        # cmp_parameters = (p, z_id, m, rot, ppars)
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
            print("##" * 50)
            print("##" * 50)
            print("PROBABLY NOT WORKING NOW WITH NEW COMPATIBILITY, RE-RUN with jobs = 0!")
            print("##" * 50)
            print("##" * 50)

            print(f'running {args.jobs} parallel jobs with multiprocessing')
            #pool = multiprocessing.Pool(args.jobs)
            #costs_list = zip(*pool.map(compute_cost_matrix_LAP, [(i, j, pieces, region_mask, cmp_parameters, ppars) for j in range(n) for i in range(n)]))
            #with parallel_backend('threading', n_jobs=args.jobs):
            costs_list = Parallel(n_jobs=args.jobs, prefer="threads")(delayed(compute_cost_wrapper)(i, j, pieces, region_mask, ppars, compatibility_type=args.cmp_type, verbosity=args.verbosity) for i in range(n) for j in range(n)) ## is something change replacing j and i ???
            #costs_list = Parallel(n_jobs=args.jobs)(delayed(compute_cost_matrix_LAP)(i, j, pieces, region_mask, cmp_parameters, ppars) for j in range(n) for i in range(n))
            All_cost, All_norm_cost = reshape_list2mat_and_normalize(costs_list, n=n, norm_value=computation_parameters.rmax)
        else:
            for i in range(n):  # select fixed fragment
                for j in range(n):
                    if args.verbosity == 1:
                        print(f"Computing compatibility between piece {i:04d} and piece {j:04d}..", end='\r')
                    ji_mat = compute_cost_wrapper(i, j, pieces, region_mask, ppars, \
                        detector=yolov8_obb_detector, segmentator=segmentator, seg_len=seg_len,
                        verbosity=args.verbosity)
                    
                    All_cost[:, :, :, j, i] = ji_mat
                    
                    # DEBUG
                    if i == 0 and j == 1 and args.DEBUG is True:
                        rotation_idx = 0
                        plt.suptitle(f"COST WITH {args.cmp_cost}", fontsize=45)
                        plt.subplot(541); plt.imshow(pieces[i]['img']); plt.title(f"piece {i}")
                        plt.subplot(542); plt.imshow(pieces[j]['img']); plt.title(f"piece {j}")
                        plt.subplot(545); plt.imshow(region_mask[:,:,0,j,i], vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("region map 0")
                        if region_mask.shape[2] > 1:
                            plt.subplot(546); plt.imshow(region_mask[:,:,1,j,i], vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("region map 1")
                            plt.subplot(547); plt.imshow(region_mask[:,:,2,j,i], vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("region map 2")
                            plt.subplot(548); plt.imshow(region_mask[:,:,3,j,i], vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("region map 3")
                        # plt.subplot(546); plt.imshow(ji_mat[:,:,rotation_idx], cmap='RdYlGn'); plt.title("cost")
                        # if args.cmp_cost == 'LCI':
                        #     norm_cmp = ji_mat[:,:,0] / np.max(ji_mat[:,:,0]) #np.maximum(1 - ji_mat[:,:,0] / computation_parameters.rmax, 0)
                        #     plt.subplot(547); plt.imshow(norm_cmp, vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("compatibility")
                        #     plt.subplot(548); plt.imshow(norm_cmp + np.minimum(region_mask[:,:,rotation_idx,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("final cmp")
                        # else:
                        #     norm_cmp = np.maximum(1 - ji_mat[:,:,0] / computation_parameters.rmax, 0)
                        #     plt.subplot(547); plt.imshow(norm_cmp, vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("compatibility")
                        #     plt.subplot(548); plt.imshow(norm_cmp + np.minimum(region_mask[:,:,rotation_idx,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("final cmp")
                        
                        plt.subplot(549); plt.title("COST ROTATION 0")
                        plt.imshow(ji_mat[:,:,0], cmap='RdYlGn'); 
                        if ji_mat.shape[2] > 1:
                            plt.subplot(5,4,10); plt.title("COST ROTATION 1")
                            plt.imshow(ji_mat[:,:,1], cmap='RdYlGn'); 
                            plt.subplot(5,4,11); plt.title("COST ROTATION 2")
                            plt.imshow(ji_mat[:,:,2], cmap='RdYlGn'); 
                            plt.subplot(5,4,12); plt.title("COST ROTATION 3")
                            plt.imshow(ji_mat[:,:,3], cmap='RdYlGn'); 

                        
                        if args.cmp_cost == 'LAP':
                            ji_mat[ji_mat > computation_parameters.badmatch_penalty] = computation_parameters.badmatch_penalty
                            ji_unique_values = np.unique(ji_mat)
                            k = min(computation_parameters.k, len(ji_unique_values))
                            kmin_cut_val = np.sort(ji_unique_values)[-k]
                            plt.subplot(5,4,13); plt.title("COST ROTATION KMINCUT 0")
                            plt.imshow(np.maximum(1 - ji_mat[:,:,0] / kmin_cut_val, 0), cmap='RdYlGn'); 
                            plt.subplot(5,4,17); plt.title("EXP ROTATION 0")
                            plt.imshow(np.exp(-ji_mat[:,:,0]/76), cmap='RdYlGn'); 
                            if ji_mat.shape[2] > 1:
                                plt.subplot(5,4,14); plt.title("COST ROTATION KMINCUT 1")
                                plt.imshow(np.maximum(1 - ji_mat[:,:,1] / kmin_cut_val, 0), cmap='RdYlGn')
                                plt.subplot(5,4,15); plt.title("COST ROTATION KMINCUT 2")
                                plt.imshow(np.maximum(1 - ji_mat[:,:,2] / kmin_cut_val, 0), cmap='RdYlGn') 
                                plt.subplot(5,4,16); plt.title("COST ROTATION KMINCUT 3")
                                plt.imshow(np.maximum(1 - ji_mat[:,:,3] / kmin_cut_val, 0), cmap='RdYlGn')

                                #plt.imshow(norm_cmp + np.minimum(region_mask[:,:,0,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); 
                                plt.subplot(det_method5,4,18); plt.title("EXP ROTATION 1")
                                plt.imshow(np.exp(-ji_mat[:,:,1]/76), cmap='RdYlGn'); 
                                #plt.imshow(norm_cmp + np.minimum(region_mask[:,:,1,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); 
                                plt.subplot(5,4,19); plt.title("EXP ROTATION 2")
                                plt.imshow(np.exp(-ji_mat[:,:,2]/76), cmap='RdYlGn'); 
                                #plt.imshow(norm_cmp + np.minimum(region_mask[:,:,2,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); 
                                plt.subplot(5,4,20); plt.title("EXP ROTATION 3")
                                plt.imshow(np.exp(-ji_mat[:,:,3]/76), cmap='RdYlGn'); 
                            #plt.imshow(norm_cmp + np.minimum(region_mask[:,:,3,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); 
                        if args.cmp_cost == 'LAP2':
                            clipping_val = computation_parameters.max_dist + (computation_parameters.badmatch_penalty - computation_parameters.max_dist) / 3
                            ji_mat = np.clip(ji_mat, 0, clipping_val)
                            ji_mat_normalized = 1 - ji_mat / clipping_val
                            plt.subplot(5,4,13); plt.title("compatibility normalized")
                            plt.imshow(ji_mat_normalized, cmap='RdYlGn'); 
                        
                        plt.show()
                        breakpoint()

        if args.cmp_type == 'lines':
            if args.cmp_cost == 'LCI':
                print("WARNING: normalized over each piece!")
                #All_norm_cost = All_cost/np.max(All_cost)  # normalize to max value TODO !!!
            elif args.cmp_cost == 'LAP3':
                min_vals = []
                for j in range(All_cost.shape[3]):
                    for i in range(All_cost.shape[4]):
                        min_val = np.min(All_cost[:, :, :, j, i])
                        min_vals.append(min_val)
                kmin_cut_val = np.max(min_vals) + 1
                All_norm_cost = np.maximum(1 - All_cost/ kmin_cut_val, 0)
            elif args.cmp_cost == 'LAP2':
                clipping_val = computation_parameters.max_dist + (computation_parameters.badmatch_penalty - computation_parameters.max_dist) / 3
                All_cost = np.clip(All_cost, 0, clipping_val)
                All_norm_cost = 1 - All_cost / clipping_val
            else: 
                #max_cost = np.max(All_cost)
                #All_norm_cost = np.maximum(1 - All_cost / computation_parameters.rmax, 0)
                All_norm_cost = All_cost # / np.max(All_cost) #
        elif args.cmp_type == 'color':
            # breakpoint()
            # normalization
            k = ppars['k']
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
                breakpoint()
                print(a_ks[:, :, i])
                All_cost_cut[:, :, :, :, i] = a_cost_i

            # norm_term = 100
            # norm_term = np.max(a_min)/(3*k)
            norm_term = np.max(a_ks)/(2*k)

            All_norm_cost = 2 - All_cost_cut / norm_term  # only for colors

            All_norm_cost = np.where(All_norm_cost > 2, 0, All_norm_cost)    # only for colors
            #All_norm_cost = np.where(All_norm_cost < 0, 0, All_norm_cost)   # only for colors
            All_norm_cost = np.where(All_norm_cost <= 0, -1, All_norm_cost)  ## NEW idea di Prof.Pelillo
            #All_norm_cost /= np.max(All_norm_cost)
        else:
            All_norm_cost = All_cost

                    
            # only_negative_region = np.clip(region_mask, -1, 0)
            # All_cost = (np.clip(All_cost, 0, max_cost))/max_cost

        if args.cmp_type == 'motifs':
            max_cost = np.max(All_cost)
            print(max_cost)
            if max_cost < 0.1:
                breakpoint()
            only_negative_region = np.clip(region_mask, -1, 0)
            All_norm_cost = (np.clip(All_cost, 0, max_cost))/max_cost
        else:
            only_negative_region =  np.minimum(region_mask, 0)  # recover overlap (negative) areas

        R = All_norm_cost + only_negative_region  # insert negative regions to cost matrix

        # it should not be needed
        # R_line = (All_norm_cost * region_mask) * 2
        # R_line[R_line < 0] = -1
        # for jj in range(n):
        #     R_line[:, :, :, jj, jj] = -1
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
        # save output
        output_folder = os.path.join(fnames.output_dir, args.dataset, puzzle, fnames.cm_output_name)
        os.makedirs(output_folder, exist_ok=True)
        if args.cmp_type == 'lines':
            cmp_name = f"linesdet_{args.det_method}_cost_{args.cmp_cost}"
        elif args.cmp_type == 'shape':
            cmp_name = "shape"
        elif args.cmp_type == 'motifs':
            cmp_name = f"motifs_{args.det_method}"
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
                    "det_method":args.det_method, 
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
    parser.add_argument('--dataset', type=str, default='synthetic_irregular_9_pieces_by_drawing_coloured_lines_peynrh', help='dataset folder')  # repair
    parser.add_argument('--puzzle', type=str, default='', help='puzzle folder (if empty will do all folders inside the dataset folder)')  # repair_g97, repair_g28, decor_1_lines
    parser.add_argument('--penalty', type=int, default=-1, help='penalty (leave -1 to use the one from the config file)')
    parser.add_argument('--jobs', type=int, default=0, help='how many jobs (if you want to parallelize the execution')
    parser.add_argument('--save_visualization', type=bool, default=True, help='save an image that showes the matrices color-coded')
    parser.add_argument('--save_everything', default=False, action='store_true',
        help='use to save debug matrices (may require up to ~8 GB per solution, use with care!)')
    parser.add_argument('--verbosity', type=int, default=1, help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')
    # COMPATIBILITY PARAMETERS
    parser.add_argument('--cmp_type', type=str, default='lines', 
        help='Chooses the compaitbility to use.\nIf more than one should be used, select `combo`\
            \nIt is connected with `--cmp_cost` and `--det_method`!', 
        choices=['lines', 'shape', 'color', 'motifs', 'combo','seg'])   
    parser.add_argument('--cmp_cost', type=str, default='LCI', 
        help='Chooses the cost used to compute compatibility - it depends on the `--cmp_type`\
            \nUse LAP, LAP3 or LCI for lines, YOLO or overlap for motif, SDF for shape, MGC for color', 
        choices=[
            'LAP', 'LAP3', 'LCI', # line-based 
            'YOLO_conf', 'overlap', # motif-based
            'SDF', # shape-based
            'MGC' # color-based
        ])    
    parser.add_argument('--cmp_combo', type=str, default='LS', 
        help='If `--cmp_type` is `combo`, it chooses which compatibility to use!\
            \nThe capital letters are used (L=lines, M=motif, S=shape, C=color)\
            \nFor example, MS is motif+shape, LS is lines+shape', 
        choices=['LS', 'MS', 'CS', 'CLMS'])   
    parser.add_argument('--det_method', type=str, default='exact', 
        help='method for the feature detection (usually lines or motif)',
        choices=['exact', 'deeplsd', 'manual', 'yolo-obb', 'yolo-bbox', 'yolo-seg']) 
    parser.add_argument('--yolo_path', type=str, default='/home/marina/PycharmProjects/RL_puzzle_solver/yolov5/best.pt', help='yolo path (.pt model)')
    parser.add_argument('--border_len', type=int, default=-1, help='length of border (if -1 [default] it will be set to xy_step)')   
    parser.add_argument('--k', type=int, default=5, help='keep the best k values (for given gamma transformation) in the compatibility')
    
    Segmentator.add_args(parser)  

        # exact, manual, deeplsd
    # parser.add_argument('--xy_step', type=int, default=30, help='the step (in pixels) between each grid point')
    # parser.add_argument('--xy_grid_points', type=int, default=7, 
    #     help='the number of points in the grid (for each axis, total number will be the square of what is given)')
    # parser.add_argument('--theta_step', type=int, default=90, help='degrees of each rotation')
    parser.add_argument('--DEBUG', action='store_true', default=False, help='WARNING: will use debugger! It stops and show the matrices!')

    args = parser.parse_args()
    main(args)
