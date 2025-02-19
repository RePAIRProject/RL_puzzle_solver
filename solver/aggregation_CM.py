
import numpy as np
import matplotlib.colors
import os
import configs.folder_names as fnames
from PIL import Image
import time
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
# from compatibility.line_matching_NEW_segments import read_info
from compatibility.utils import normalize_CM
# import configs.solver_cfg as cfg
from puzzle_utils.pieces_utils import calc_parameters_v2
from puzzle_utils.shape_utils import prepare_pieces_v2, create_grid, place_on_canvas, crop_to_content
import datetime
import pdb
import time
import json
from puzzle_utils.regions import combine_region_masks
from puzzle_utils.visualization import save_vis
import copy

def aggregate_CM_matrices (args, puzzle_root_folder):

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
                os.path.join(puzzle_root_folder, fnames.cm_output_name, f"CM_Aggregated_motifs_{args.motif_det_method}"))
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