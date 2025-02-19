import numpy as np
import cv2 as cv
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from scipy.io import savemat, loadmat
import os
import configs.folder_names as fnames
import argparse

from compatibility.utils import normalize_CM
from puzzle_utils.visualization import save_vis
from puzzle_utils.shape_utils import prepare_pieces_v2
import json
from puzzle_utils.regions import combine_region_masks
import copy


def aggregate_motif_matrices (args, puzzle_root_folder):
    mat = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_motifs_{args.motif_det_method}'))
    R = mat['R']
    if len(R.shape) == 6:
        # n_motifs = R.shape[5]
        a = np.where((R != 0), R, 100)
        R_new = np.min(a, axis=5)
        R = np.where((R_new == 100), 0, R_new)
    return R


def aggregate_cm_matrices (args, puzzle_root_folder):
    cmp_name = f"combo_{args.combo_type}"

    if args.combo_type == "SH-LIN":
        print("combining shape and lines..")
        mat_shape = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_shape'))
        mat_lines = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name,
                                         f'CM_linesdet_{args.lines_det_method}_cost_{args.cmp_cost}'))
        region_mask_mat = loadmat(os.path.join(puzzle_root_folder, fnames.rm_output_name, f'RM_{args.puzzle}.mat'))
        R_shape = mat_shape['R']
        R_lines = mat_lines['R']
        lines_RM = region_mask_mat['RM_lines']
        shape_RM = region_mask_mat['RM_shapes']
        norm_R_shape = normalize_CM(R_shape)
        norm_R_lines = normalize_CM(R_lines)
        negative_region_map = R_shape < 0
        region_lines = combine_region_masks([shape_RM, lines_RM])
        prm_lines = (region_lines > 0).astype(int)  ## positive in RM
        prm_shape = (shape_RM > 0).astype(int)
        shape_basis = norm_R_shape * prm_shape
        lines_avg_val = 0.5  # fix level ???  # lines_avg_val1 = np.mean(norm_R_lines > 0)
        lines_contrib = prm_lines * ((norm_R_lines / lines_avg_val) - 1)
        R = np.zeros_like(R_shape)
        total_contrib = shape_basis * (lines_contrib)
        R = shape_basis + total_contrib
        R += -1 * negative_region_map.astype(int)
        R = normalize_CM(R)
        R = np.maximum(-1, R)

    elif args.combo_type == 'SH-MOT' or args.combo_type == 'SH-AggMOT':
        print("combining shape and motifs..")

        if args.combo_type == 'SH-AggMOT':
            mat_motifs = loadmat(
                os.path.join(puzzle_root_folder, fnames.cm_output_name, f"CM_Agg_motifs_{args.motif_det_method}"))
        else:
            mat_motifs = loadmat(
            os.path.join(puzzle_root_folder, fnames.cm_output_name, f"CM_motifs_{args.motif_det_method}"))

        mat_shape = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_shape'))

        # breakpoint()
        R_motif = mat_motifs['R']
        R_shape = mat_shape['R']

        # only positive values
        R = copy.deepcopy(R_shape)
        negative_region_map = R_motif < 0
        positive_motif_ids = np.where(R_motif > 0)
        zero_motif_ids = np.where(R_motif == 0)

        shape_imp = 0.3
        #R[positive_motif_ids] = (np.clip(R_motif[positive_motif_ids], 0, 1) + np.clip(R_shape[positive_motif_ids], 0, 1)) / 2
        R[positive_motif_ids] = R_motif[positive_motif_ids] + shape_imp*((R_shape[positive_motif_ids]))  ## (+/-) shape_comp. val
        R[zero_motif_ids] = R_shape[zero_motif_ids]
        R[negative_region_map] = -1

    elif args.combo_type == 'SH-SEG':
        print("combining shape and motifs..")
        mat_seg = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f"CM_cmp_seg"))
        mat_shape = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_shape'))
        # breakpoint()
        R_seg = mat_seg['R']
        R_shape = mat_shape['R']
        negative_region_map = R_seg < 0

        # only positive values
        R = (np.clip(R_seg, 0, 1) * np.clip(R_shape, 0, 1))
        R /= np.max(R)
        # negative values set to -1
        R[negative_region_map] = -1

    elif args.combo_type == 'SLM_v1':
        print("trying to combine three compatibilities (ShapeLinesMotifs)")
        mat_motif = loadmat(
            os.path.join(puzzle_root_folder, fnames.cm_output_name, f"CM_motifs_{args.motif_det_method}"))
        mat_shape = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_shape'))
        mat_lines = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name,
                                         f'CM_linesdet_{args.lines_det_method}_cost_{args.cmp_cost}'))
        region_mask_mat = loadmat(os.path.join(puzzle_root_folder, fnames.rm_output_name, f'RM_{args.puzzle}.mat'))

        R_shape = mat_shape['R']
        R_lines = mat_lines['R']
        R_motif = mat_motif['R']
        lines_RM = region_mask_mat['RM_lines']
        motif_RM = region_mask_mat['RM_motifs']
        shape_RM = region_mask_mat['RM_shapes']

        norm_R_shape = normalize_CM(R_shape)
        norm_R_lines = normalize_CM(R_lines)
        norm_R_motif = normalize_CM(R_motif)

        negative_region_map = R_shape < 0
        region_motif = combine_region_masks([shape_RM, motif_RM])
        region_lines = combine_region_masks([shape_RM, lines_RM])

        prm_motif = (region_motif > 0).astype(int)  ## positive in RM
        prm_lines = (region_lines > 0).astype(int)  ## positive in RM
        prm_shape = (shape_RM > 0).astype(int)
        shape_basis = norm_R_shape * prm_shape

        lines_avg_val = 0.5  # fix level ???  # lines_avg_val1 = np.mean(norm_R_lines > 0)
        motif_avg_val = 0.5  # motif_avg_val1 = np.mean(norm_R_motif > 0)
        motif_contrib = prm_motif * ((norm_R_motif / motif_avg_val) - 1)
        lines_contrib = prm_lines * ((norm_R_lines / lines_avg_val) - 1)

        R = np.zeros_like(R_shape)
        total_contrib = shape_basis * (motif_contrib + lines_contrib)
        R = shape_basis + total_contrib
        R += -1 * negative_region_map.astype(int)
        R = normalize_CM(R)
        R = np.maximum(-1, R)

    elif args.combo_type == 'SLMS_v2':
        print("trying to combine three compatibilities (ShapeLinesMotifs)")
        mat_motif = loadmat(
            os.path.join(puzzle_root_folder, fnames.cm_output_name, f"CM_motifs_{args.motif_det_method}"))
        mat_shape = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_shape'))
        mat_lines = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name,
                                         f'CM_linesdet_{args.lines_det_method}_cost_{args.cmp_cost}'))
        mat_seg = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_cmp_seg'))
        region_mask_mat = loadmat(os.path.join(puzzle_root_folder, fnames.rm_output_name, f'RM_{args.puzzle}.mat'))

        R_shape = mat_shape['R']
        R_lines = mat_lines['R']
        R_motif = mat_motif['R']
        R_seg = mat_seg['R']
        lines_RM = region_mask_mat['RM_lines']
        motif_RM = region_mask_mat['RM_motifs']
        seg_RM = region_mask_mat['RM_motifs']
        shape_RM = region_mask_mat['RM_shapes']

        norm_R_shape = normalize_CM(R_shape)
        norm_R_lines = normalize_CM(R_lines)
        norm_R_motif = normalize_CM(R_motif)
        norm_R_seg = normalize_CM(R_seg)

        negative_region_map = R_shape < 0
        region_motif = combine_region_masks([shape_RM, motif_RM])
        region_lines = combine_region_masks([shape_RM, lines_RM])
        region_seg = combine_region_masks([shape_RM, seg_RM])

        prm_motif = (region_motif > 0).astype(int)  ## positive in RM
        prm_lines = (region_lines > 0).astype(int)  ## positive in RM
        prm_seg = (region_seg > 0).astype(int)  ## positive in RM
        prm_shape = (shape_RM > 0).astype(int)
        shape_basis = norm_R_shape * prm_shape

        lines_avg_val = 0.5  # fix level ???  # lines_avg_val1 = np.mean(norm_R_lines > 0)
        motif_avg_val = 0.5  # motif_avg_val1 = np.mean(norm_R_motif > 0)
        seg_avg_val = 0.5
        motif_contrib = prm_motif * ((norm_R_motif / motif_avg_val) - 1)
        lines_contrib = prm_lines * ((norm_R_lines / lines_avg_val) - 1)
        seg_contrib = prm_seg * ((norm_R_seg / seg_avg_val) - 1)

        R = np.zeros_like(R_shape)
        total_contrib = shape_basis * (motif_contrib + lines_contrib + seg_contrib)
        # total_contrib = shape_basis * (lines_contrib + np.max(seg_contrib, motif_contrib))  # option
        R = shape_basis + total_contrib
        R += -1 * negative_region_map.astype(int)
        R = normalize_CM(R)
        R = np.maximum(-1, R)

        # import matplotlib.pyplot as plt
        # plt.subplot(331)
        # plt.imshow(prm[:, :, 0, 1, 2])
        # plt.subplot(332)
        # plt.imshow(prm_motif[:, :, 0, 1, 2])
        # plt.subplot(333)
        # plt.imshow(prm_lines[:, :, 0, 1, 2])
        # plt.subplot(334)
        # plt.imshow(shape_basis[:, :, 0, 1, 2])
        # plt.subplot(335)
        # plt.imshow(motif_contrib[:, :, 0, 1, 2])
        # plt.subplot(336)
        # plt.imshow(lines_contrib[:, :, 0, 1, 2])
        # plt.subplot(338)
        # plt.imshow(R2[:, :, 0, 1, 2])
        # plt.subplot(339)
        # plt.imshow(R[:, :, 0, 1, 2])
        # plt.show()
        # breakpoint()

    elif args.combo_type == 'SLMS_version3':
        print("trying to combine three compatibilities (ShapeLinesMotifs)")
        mat_motif = loadmat(
            os.path.join(puzzle_root_folder, fnames.cm_output_name, f"CM_motifs_{args.motif_det_method}"))
        mat_shape = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_shape'))
        mat_lines = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name,
                                         f'CM_linesdet_{args.lines_det_method}_cost_{args.cmp_cost}'))
        mat_seg = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_cmp_seg'))
        region_mask_mat = loadmat(os.path.join(puzzle_root_folder, fnames.rm_output_name, f'RM_{args.puzzle}.mat'))

        R_shape = mat_shape['R']
        R_lines = mat_lines['R']
        R_motif = mat_motif['R']
        R_seg = mat_seg['R']
        lines_RM = region_mask_mat['RM_lines']
        motif_RM = region_mask_mat['RM_motifs']
        seg_RM = region_mask_mat['RM_motifs']
        shape_RM = region_mask_mat['RM_shapes']

        norm_R_shape = normalize_CM(R_shape)
        norm_R_lines = normalize_CM(R_lines)
        norm_R_motif = normalize_CM(R_motif)
        norm_R_seg = normalize_CM(R_seg)

        negative_region_map = R_shape < 0
        region_motif = combine_region_masks([shape_RM, motif_RM]).astype(int)
        region_lines = combine_region_masks([shape_RM, lines_RM]).astype(int)
        region_seg = combine_region_masks([shape_RM, seg_RM]).astype(int)

        prm_motif = (region_motif > 0).astype(int)  ## positive in RM
        prm_lines = (region_lines > 0).astype(int)  ## positive in RM
        prm_seg = (region_seg > 0).astype(int)  ## positive in RM
        prm_shape = (shape_RM > 0).astype(int)
        shape_basis = norm_R_shape * prm_shape

        lines_acc_lev = 0.01  # fix level ???  # lines_avg_val1 = np.mean(norm_R_lines > 0)
        motif_acc_lev = 0.3  # motif_avg_val1 = np.mean(norm_R_motif > 0)
        seg_acc_lec = 0.3

        lines_contrib = prm_lines * (np.where(norm_R_lines < lines_acc_lev, norm_R_lines - 0.5, norm_R_lines))
        motif_contrib = prm_motif * (np.where(norm_R_motif < motif_acc_lev, norm_R_motif - 0.5, norm_R_motif))
        seg_contrib = prm_seg * (np.where(norm_R_seg < seg_acc_lec, norm_R_seg - 0.5, norm_R_seg))

        R = np.zeros_like(R_shape)
        total_contrib = shape_basis * (motif_contrib + lines_contrib + seg_contrib)
        # total_contrib = shape_basis * (lines_contrib + np.max(seg_contrib, motif_contrib))  # option
        R = shape_basis + total_contrib
        R += -1 * negative_region_map.astype(int)
        R = normalize_CM(R)
        R = np.maximum(-1, R)

    elif args.combo_type == 'SLM_new_TEST':
        print("trying to combine three compatibilities (ShapeLinesMotifs)")
        mat_motif = loadmat(
            os.path.join(puzzle_root_folder, fnames.cm_output_name, f"CM_motifs_{args.motif_det_method}"))
        mat_shape = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_shape'))
        mat_lines = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name,
                                         f'CM_linesdet_{args.lines_det_method}_cost_{args.cmp_cost}'))
        # mat_seg = loadmat(os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_cmp_seg'))
        region_mask_mat = loadmat(os.path.join(puzzle_root_folder, fnames.rm_output_name, f'RM_{args.puzzle}.mat'))

        R_shape = mat_shape['R']
        R_lines = mat_lines['R']
        R_motif = mat_motif['R']
        # R_seg = mat_seg['R']
        shape_RM = region_mask_mat['RM_shapes']
        lines_RM = region_mask_mat['RM_lines']
        motif_RM = region_mask_mat['RM_motifs']
        poly_motif_RM = region_mask_mat['RM_poly_motifs']  # new part
        # seg_RM = region_mask_mat['RM_motifs']

        norm_R_shape = normalize_CM(R_shape)
        norm_R_lines = normalize_CM(R_lines)
        norm_R_motif = normalize_CM(R_motif)  # only motif-motif-intersection values
        # norm_R_seg = normalize_CM(R_seg)

        region_motif = combine_region_masks([shape_RM, motif_RM]).astype(int)
        region_lines = combine_region_masks([shape_RM, lines_RM]).astype(int)
        # region_seg = combine_region_masks([shape_RM, seg_RM]).astype(int)

        prm_shape = (shape_RM > 0).astype(int)
        prm_motif = (region_motif > 0).astype(int)  ## positive in RM
        prm_lines = (region_lines > 0).astype(int)  ## positive in RM
        # prm_seg = (region_seg > 0).astype(int)  ## positive in RM

        shape_basis = norm_R_shape * prm_shape
        ### TEMP FOR DEBUG !!!!
        shape_basis = prm_shape

        lines_contrib = prm_lines * norm_R_lines
        motif_contrib = prm_motif * norm_R_motif
        # seg_contrib  = prm_seg   * norm_R_seg

        # total_contrib = min(motif_contrib, lines_contrib) # positive contribution
        total_contrib = np.where((lines_contrib > 0 and motif_contrib > 0), min(motif_contrib, lines_contrib),
                                 max(motif_contrib, lines_contrib))

        # version 1 - shape compatibility if no other contributions
        R = np.where(total_contrib > 0, total_contrib, shape_basis)
        # version 2 - add to shpe basis
        # R = shape_basis + total_contrib

        # ADD negative shape-values(overlap) and penalised motif UN-MATCH
        negative_region_map = R_shape < 0
        negative_motif_map = poly_motif_RM < 0
        R += -1 * negative_region_map.astype(int)
        R += -1 * negative_motif_map.astype(int)
        R = normalize_CM(R)
        R = np.maximum(-1, R)

    return R


class CfgParameters(dict):
    __getattr__ = dict.__getitem__


def main(args, pieces=None):
    puzzle_name = args.puzzle

    print("-" * 50)
    print(f"Started working on {puzzle_name}")
    print(f"Dataset: {args.dataset}")
    print("-" * 50)

    pieces, img_parameters = prepare_pieces_v2(fnames, args.dataset, args.puzzle, verbose=True)
    cfg = CfgParameters()
    cfg['cmp_type'] = args.cmp_type
    cfg['cmp_cost'] = args.cmp_cost
    cfg['combo_type'] = args.combo_type
    puzzle_root_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset, args.puzzle)

    cmp_parameter_path = os.path.join(puzzle_root_folder, 'compatibility_parameters_v2.json')
    if os.path.exists(cmp_parameter_path):
        ppars = CfgParameters()
        with open(cmp_parameter_path, 'r') as cp:
            ppars_dict = json.load(cp)
        for ppk in ppars_dict.keys():
            ppars[ppk] = ppars_dict[ppk]

    ## Aggregate motifs
    if args.cmp_type == 'motifs' or args.combo_type == 'SH-AggMOT':
        print("loading motifs-CM for aggregation")
        R = aggregate_motif_matrices(args, puzzle_root_folder)

    ## Save aggregated Motif matrix
    filename = os.path.join(puzzle_root_folder, fnames.cm_output_name, f'CM_Agg_motifs_{args.motif_det_method}')
    np.save(filename, R)
    mdic = {
        "R": R,
        "label": "label",
        "cmp_type": args.cmp_type,
        "cmp_cost": args.cmp_cost,
        "lines_det_method": args.lines_det_method,
        "motif_det_method": args.motif_det_method,
        "xy_step": ppars.xy_step,
        "xy_grid_points": ppars.xy_grid_points,
        "theta_step": ppars.theta_step
    }
    savemat(f'{filename}.mat', mdic)
    vis_folder = os.path.join(puzzle_root_folder, fnames.cm_output_name, f'visualization')
    save_vis(R, pieces, ppars.theta_step, os.path.join(vis_folder, f'CM_Agg_motifs_{args.motif_det_method}'),
             f"compatibility matrix {puzzle_name}", all_rotation=True)


    ## Combine CM and Save Combo-compatibility matrix
    R = aggregate_cm_matrices(args, puzzle_root_folder)
    filename = os.path.join(puzzle_root_folder, fnames.cm_output_name,
                            f'CM_combo_{args.combo_type}')
    np.save(filename, R)
    mdic = {
        "R": R,
        "label": "label",
        "cmp_type": args.cmp_type,
        "cmp_cost": args.cmp_cost,
        "lines_det_method": args.lines_det_method,
        "motif_det_method": args.motif_det_method,
        "xy_step": ppars.xy_step,
        "xy_grid_points": ppars.xy_grid_points,
        "theta_step": ppars.theta_step
    }
    savemat(f'{filename}.mat', mdic)
    vis_folder = os.path.join(puzzle_root_folder, fnames.cm_output_name, f'visualization')
    save_vis(R, pieces, ppars.theta_step, os.path.join(vis_folder, f'CM_combo_{args.combo_type}'),
             f'compatibility matrix {puzzle_name}', all_rotation=True)

    print("-" * 50)
    print(f'Done with aggregation CM for {puzzle_name}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='........ ')  # add some description
    parser.add_argument('--dataset', type=str, default='RePAIR_exp_batch3_clean_TEST', help='dataset folder')
    parser.add_argument('--puzzle', type=str, default='RPobj_g39_o0039_gt_rot_ANNOT', help='puzzle folder')
    parser.add_argument('--lines_det_method', type=str, default='deeplsd',
                        help='method line detection')  # exact, manual, deeplsd
    parser.add_argument('--motif_det_method', type=str, default='yolo-obb',
                        help='method motif detection')  # exact, manual, deeplsd
    parser.add_argument('--cmp_cost', type=str, default='LAP', help='cost computation')  # LAP, LCI
    parser.add_argument('--exclude', default=False, action='store_true',
                        help='use to exclude pieces without compatibility (used for some partial compatibilities, not fully tested!)')
    parser.add_argument('--verbosity', type=int, default=2,
                        help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')
    parser.add_argument('--few_rotations', type=int, default=0, help='uses only few rotations to make it faster')
    parser.add_argument('--cmp_type', type=str, default='combo', help='which compatibility to use!',
                        choices=['combo', 'lines', 'shape', 'color', 'motifs', 'seg'])
    parser.add_argument('--combo_type', type=str, default='SH-AggMOT',
                        help='If `--cmp_type` is `combo`, it chooses which compatibility to use!\
            \nAbbreviations: (LIN=lines, MOT=motif, SH=shape, COL=color, SEG=segmentation)\
            \nFor example, SH-MOT is motif+shape, SH-SEG is shape+segmentation',
                        choices=['SH-SEG', 'SH-MOT', 'SH-LIN', 'SLM_v1', 'SLMS_v2', 'SLMS_version3', 'SH-AggMOT'])

    args = parser.parse_args()
    main(args)

