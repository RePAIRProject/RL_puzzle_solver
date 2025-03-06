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
from puzzle_utils.regions import combine_region_masks, combine_region_masks_V2
from compatibility.compatibility_Motifs import compute_CM_using_motifs, compute_CM_using_motifs_vis
from utils import compute_cost_wrapper, calc_computation_parameters, normalize_CM, reshape_list2mat, \
    show_debug_visualization


def motifs_TEST(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, \
                                       yolo_obj_detector, det_type='yolo-obb', detect_on_crop=True, area_ratio=0.1,
                                       verbosity=1):

    n_motifs = np.shape(mask_ij)[-1]
    R_cost_conf = np.zeros((m.shape[1], m.shape[1], len(rot), n_motifs))
    R_cost_overlap = np.zeros((m.shape[1], m.shape[1], len(rot), n_motifs))

    for mt in range(n_motifs):
        for t in range(len(rot)):# theta_rad = theta * np.pi / 180
            for ix in range(m.shape[1]):
                for iy in range(m.shape[1]):
                    valid_point = mask_ij[iy, ix, t, mt]
                    if valid_point > 0:
                        canv_cnt = ppars.canvas_size // 2
                        grid = z_id + canv_cnt
                        x_j_pixel, y_j_pixel = grid[iy, ix]

                        # Place on canvas pairs of pieces given position
                        center_pos = ppars.canvas_size // 2
                        piece_i_on_canvas = place_on_canvas(pieces[idx1], (center_pos, center_pos), ppars.canvas_size,
                                                            0)
                        piece_j_on_canvas = place_on_canvas(pieces[idx2], (x_j_pixel, y_j_pixel), ppars.canvas_size,
                                                            t * ppars.theta_step)
                        overlap_area = piece_i_on_canvas['mask'] + piece_j_on_canvas['mask']
                        pieces_ij_on_canvas = piece_i_on_canvas['img'] + piece_j_on_canvas['img'] * (
                            np.dstack(((overlap_area < 2), (overlap_area < 2), (overlap_area < 2)))).astype(int)

                        if detect_on_crop == True:
                            cropped_img, x0, x1, y0, y1 = crop_to_content(pieces_ij_on_canvas, return_vals=True)
                            img_pil = Image.fromarray(np.uint8(cropped_img))
                        else:
                            x0 = 0
                            y0 = 0
                            img_pil = Image.fromarray(np.uint8(pieces_ij_on_canvas))

                        if mt == 8:    # mt - motif type
                            plt.imshow(img_pil)
                            print([idx1, idx2])

                        detected = yolo_obj_detector(img_pil, verbose=False)[0]

                        if det_type == 'yolo-obb':
                            det_objs = detected.obb
                        elif det_type == 'yolo-bbox':
                            det_objs = detected.boxes

                        motif_conf_score = np.array([])
                        motif_overlap_score = np.array([])

                        for det_obb in det_objs:
                            class_label = int(det_obb.cpu().cls.numpy()[0])

                            if class_label == mt:        #only for objects with expected class label !!!
                                if det_type == 'yolo-obb':
                                    do_pts = det_obb.cpu().xyxyxyxy.numpy()[0]
                                elif det_type == 'yolo-bbox':
                                    ps = det_obb.cpu().xyxy[0]
                                    p1 = np.asarray(ps[:2])
                                    p2 = np.asarray([ps[0], ps[3]])
                                    p3 = np.asarray(ps[2:])
                                    p4 = np.asarray([ps[2], ps[1]])
                                    do_pts = np.asarray([p1, p2, p3, p4])

                                if detect_on_crop == True:
                                    obb_shapely_points = [(point[0] + x0, point[1] + y0) for point in do_pts]
                                else:
                                    obb_shapely_points = [(point[0], point[1]) for point in do_pts]
                                det_obb_poly = shapely.Polygon(obb_shapely_points)

                                inters_poly_i = shapely.intersection(det_obb_poly, piece_i_on_canvas['polygon'])
                                inters_poly_j = shapely.intersection(det_obb_poly, piece_j_on_canvas['polygon'])

                                if not(inters_poly_j.is_empty) and not(inters_poly_i.is_empty):
                                    score = det_obb.conf.item()
                                    motif_conf_score = np.append(motif_conf_score, score)

                        if len(motif_conf_score) != 0:
                            print(mt, motif_conf_score)
                            motif_conf_score = np.max(motif_conf_score)
                            motif_overlap_score = 1
                        else:
                            motif_conf_score = 0.1
                            motif_overlap_score = 0.1

                        R_cost_conf[iy, ix, t, mt] = motif_conf_score
                        R_cost_overlap[iy, ix, t, mt] = motif_overlap_score

    return R_cost_conf, R_cost_overlap



def create_motif_matching_pairs(idx1, idx2, pieces, mask_ij, ppars, yolo_obj_detector, det_type='yolo-obb',
                                verbosity=1):
    p = ppars['p']
    z_id = ppars['z_id']
    m = ppars['m']
    rot = ppars['rot']

    if verbosity > 1:
        print(f"Computing cost for pieces {idx1:>2} and {idx2:>2}")
    if idx1 == idx2:
        # print('idx == ')
        R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) - 1
    else:
        print(f"computing cost matrix for piece {idx1} vs piece {idx2}")
        candidate_values = np.sum(mask_ij > 0)
        #image1 = pieces[idx1]['img']
        #image2 = pieces[idx2]['img']
        #poly1 = pieces[idx1]['polygon']
        #poly2 = pieces[idx2]['polygon']
        ## OLD VERSION
        # R_cost_conf, R_cost_overlap = motifs_compatibility_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, yolo_obj_detector, det_type=det_type, verbosity=1)
        R_cost_conf, R_cost_overlap = motifs_TEST(p, z_id, m, rot, pieces, mask_ij, ppars,
                                                                              idx1,
                                                                              idx2, yolo_obj_detector,
                                                                              det_type=det_type,
                                                                              verbosity=1)
        print(f"computed cost matrix for piece {idx1} vs piece {idx2}")

        R_cost = R_cost_conf
        # R_cost = R_cost_overlap

    return R_cost


def compatibility_wrapper(idx1, idx2, pieces, regions_mask, ppars, puzzle_root_folder, detector=None, seg_len=0,
                          verbosity=1):
    """
    Wrapper for creating fragment pairs, based on motif-mask intersection, (line-based -??)
    """

    #p = ppars['p']
    #m = ppars['m']
    #z_id = ppars['z_id']
    #rot = ppars['rot']
    #n = len(pieces)
    compatibility_type = ppars['cmp_type']
    #compatibility_cost = ppars['cmp_cost']
    lines_det_method = ppars['lines_det_method']
    motif_det_method = ppars['motif_det_method']

    mask_ij = regions_mask[:, :, :, idx2, idx1]
    compatibility_matrix = np.zeros(mask_ij.shape)  ## check???
    #compatibility_matrix = np.zeros((m.shape[1], m.shape[1], len(rot)))

    if compatibility_type == 'motifs':
        compatibility_matrix = np.zeros(mask_ij.shape) ## check???
        #compatibility_matrix = np.zeros((mask_ij.shape[1], mask_ij.shape[1], len(rot), mask_ij.shape[-1]))

    if idx1 != idx2:
        poly1 = pieces[idx1]['polygon']
        poly2 = pieces[idx2]['polygon']
        candidate_values = np.sum(mask_ij > 0)

        if compatibility_type == 'shape':
            ids_to_score = np.where(mask_ij > 0)
            compatibility_matrix = compute_SDF_CM_matrix(pieces[idx1], pieces[idx2], ids_to_score, ppars,
                                                         verbosity=verbosity)
        elif compatibility_type == 'motifs':
            assert ((motif_det_method == "yolo-obb") | (motif_det_method == "yolo-bbox")), f"Unkown detection method for motifs!\nWe know `yolo-obb` and `yolo-bbox`, given `{motif_det_method}`\nRe-run specifying `--det_method`"
            compatibility_matrix = create_motif_matching_pairs(idx1, idx2, pieces, mask_ij, ppars,
                                                           yolo_obj_detector=detector, det_type=motif_det_method,
                                                           verbosity=verbosity)
        elif compatibility_type == 'Oracle_GT':
            compatibility_matrix = compute_oracle_compatibility(idx1, idx2, pieces, mask_ij, ppars, puzzle_root_folder,
                                                                verbosity=1)
        else:
            print("=" * 50)
            print("WARNING: NOT IMPLEMENTED METHOD, RETURNING EMPTY MATRIX")
            print("=" * 50)
            compatibility_matrix = np.zeros((m.shape[1], m.shape[1], len(rot)))
    return compatibility_matrix


def main(args):
    global range
    print("Compatibility log\nSearch for `CMP_START_TIME` or `CMP_END_TIME` if you want to see which images are done")

    ###########################
    #   ONE PUZZLE OR MULTIPLE
    ###########################
    if args.puzzle == '':
        puzzles = os.listdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset))
        puzzles.sort()
        puzzles = [puz for puz in puzzles if
                   os.path.isdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset, puz)) is True]
    else:
        puzzles = [args.puzzle]

    print(f"\nWill calculate compatibility matrices for: {puzzles}\n")
    for puzzle in puzzles:

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
                                                          cmp_cost=args.cmp_cost,
                                                          lines_det_method=args.lines_det_method,
                                                          motif_det_method=args.motif_det_method)

        for parkey in additional_cmp_pars.keys():
            ppars[parkey] = additional_cmp_pars[parkey]
        ppars['cmp_type'] = args.cmp_type
        calc_sdf = False
        if 'shape' in args.cmp_type:
            calc_sdf = True
        ppars['calc_sdf'] = calc_sdf
        line_based = False
        if args.cmp_type == 'lines':
            line_based = True
        ppars['line_based'] = line_based
        motif_based = False
        if 'motifs' in args.cmp_type:
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
                                    motif_det_method=args.motif_det_method, line_based=line_based, sdf=calc_sdf,
                                    motif_based=motif_based)

        ###########################
        #   REGION MASK (reading)
        ###########################
        region_mask_mat = loadmat(
            os.path.join(os.getcwd(), fnames.output_dir, args.dataset, puzzle, fnames.rm_output_name,
                         f'RM_{puzzle}.mat'))
        shape_RM = region_mask_mat['RM_shapes']

        if motif_based:
            motif_RM = region_mask_mat['RM_motifs']
            poly_motif_RM = region_mask_mat['RM_poly_motifs']  # new part
            # region_mask_0 = combine_region_masks([shape_RM, motif_RM])
            # NEW VERSION con poly-motif intersection:
            region_mask = combine_region_masks_V2([shape_RM, poly_motif_RM, motif_RM])
        else:
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
                print(
                    f"Seems compatibility has few values of rotation, using only a part of the region mask, each {step} values")
                region_mask = region_mask[:, :, ::step, :, :]
                print("now region mask shape is:", region_mask.shape)

        ################################
        #   COMPATIBILITY COMPUTATION
        ################################
        if 'motifs' in args.cmp_type:
            if len(np.shape(region_mask)) == 6:
                n_motifs = np.shape(region_mask)[-1]
                All_cost = np.zeros((m.shape[1], m.shape[1], len(rot), n, n, n_motifs))

        for i in range(n):
            for j in range(n):
                if args.verbosity == 1:
                    print(f"Computing compatibility between piece {i:04d} and piece {j:04d}..", end='\r')
                ji_mat = compatibility_wrapper(i, j, pieces, region_mask, ppars, puzzle_root_folder, detector=yolov8_obb_detector, seg_len=seg_len, verbosity=args.verbosity)
                All_cost[:, :, :, j, i] = ji_mat

        R = normalize_CM(All_cost, ppars, region_mask)


        ###########################
        #   VISUALIZATION
        ###########################
        vis_folder = os.path.join(output_folder, fnames.visualization_folder_name)
        os.makedirs(vis_folder, exist_ok=True)
        if args.save_visualization is True:
            print('Creating visualization')

            file_partial_name = f'{puzzle}_{cmp_name}_{m.shape[1]}x{m.shape[1]}x{len(rot)}x{n}x{n}'
            if len(R.shape) == 6:  # for motif-wised compativility
                n_motifs = R.shape[5]
                us_motifs = [2, 6, 8, 6, 10, 11]
                for mt in (us_motifs):
                    # for mt in range(2, n_motifs, 1):
                    R_mt = R[:, :, :, :, :, mt]
                    save_vis(R_mt, pieces, ppars.theta_step,
                             os.path.join(vis_folder, f'visualization_motif_{mt}_{file_partial_name}'),
                             f"compatibility matrix {puzzle}", all_rotation=True)
            else:
                save_vis(R, pieces, ppars.theta_step, os.path.join(vis_folder, f'visualization_{file_partial_name}'),
                         f"compatibility matrix {puzzle}", all_rotation=True)

            if args.save_everything:
                save_vis(All_cost, pieces, ppars.theta_step,
                         os.path.join(vis_folder, f'visualization_overlap_{file_partial_name}'),
                         f"cost matrix {puzzle}", all_rotation=True, vmin=-2, vmax=2)

        print("-" * 60)
        print("-- CMP_END_TIME -- ")
        # get the current date and time
        now = datetime.datetime.now()
        print(f"{now}")
        print(f'Done with {puzzle}\n')
        print("-" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computing compatibility matrix')  # add some discription
    parser.add_argument('--dataset', type=str, default='RePAIR_exp_batch3_clean_TEST', help='dataset folder')  # repair
    parser.add_argument('--puzzle', type=str, default='RPobj_g39_o0039_gt_rot_ANNOT',
                        help='puzzle folder (if empty will do all folders inside the dataset folder)')  # repair_g97, repair_g28, decor_1_lines
    parser.add_argument('--penalty', type=int, default=-1,
                        help='penalty (leave -1 to use the one from the config file)')
    parser.add_argument('--jobs', type=int, default=0, help='how many jobs (if you want to parallelize the execution')
    parser.add_argument('--save_visualization', type=bool, default=True,
                        help='save an image that showes the matrices color-coded')
    parser.add_argument('--save_everything', default=False, action='store_true',
                        help='use to save debug matrices (may require up to ~8 GB per solution, use with care!)')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')

    # COMPATIBILITY PARAMETERS
    parser.add_argument('--cmp_type', type=str, default='motifs',
                        help='Chooses the compaitbility to use.\nIf more than one should be used, select `combo`\
            \nIt is connected with `--cmp_cost` and `--det_method`!',
                        choices=['lines', 'shape', 'color', 'motifs', 'Oracle_GT', 'motifs_vis', 'shape_vis'])
    parser.add_argument('--cmp_cost', type=str, default='LAP',
                        help='Chooses the cost used to compute compatibility - it depends on the `--cmp_type`\
            \nUse LAP or LCI for lines, YOLO or overlap for motif, SDF for shape, MGC for color',
                        choices=[
                            'LAP', 'LAPvis', 'LCI',  # line-based
                            'YOLO_conf', 'overlap',  # motif-based
                            'SDF',  # shape-based
                            'MGC'  # color-based
                        ])
    parser.add_argument('--lines_det_method', type=str, default='deeplsd',
                        help='method for the feature detection (usually lines or motif)',
                        choices=['exact', 'deeplsd', 'manual'])
    parser.add_argument('--motif_det_method', type=str, default='yolo-obb',
                        help='method for the feature detection (usually lines or motif)',
                        choices=['yolo-obb', 'yolo-bbox', 'yolo-seg'])
    parser.add_argument('--yolo_path', type=str, default='/Users/Marina/PycharmProjects/RL_puzzle_solver/yolov5/best.pt',
                        help='yolo path (.pt model)')
    parser.add_argument('--border_len', type=int, default=-1,
                        help='length of border (if -1 [default] it will be set to xy_step)')
    parser.add_argument('--k', type=int, default=5,
                        help='keep the best k values (for given gamma transformation) in the compatibility')

    # exact, manual, deeplsd
    # parser.add_argument('--xy_step', type=int, default=30, help='the step (in pixels) between each grid point')
    # parser.add_argument('--xy_grid_points', type=int, default=7,
    #     help='the number of points in the grid (for each axis, total number will be the square of what is given)')
    # parser.add_argument('--theta_step', type=int, default=90, help='degrees of each rotation')
    parser.add_argument('--DEBUG', action='store_true', default=False,
                        help='WARNING: will use debugger! It stops and show the matrices!')

    args = parser.parse_args()
    main(args)
