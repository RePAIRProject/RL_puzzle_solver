import numpy as np
from scipy.io import savemat
import argparse
import pdb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import json, os
from PIL import Image

# from configs import repair_cfg as cfg
from configs import folder_names as fnames

from puzzle_utils.shape_utils import prepare_pieces_v2, create_grid, get_outside_borders, place_on_canvas, get_borders_around, include_shape_info
# from puzzle_utils.shape_utils import prepare_pieces, shape_pairwise_compatibility
from puzzle_utils.pieces_utils import calc_parameters_v2
from puzzle_utils.visualization import save_vis


def main(args):
    if args.puzzle == '':
        puzzles = os.listdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset))
        puzzles = [puz for puz in puzzles if
                   os.path.isdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset, puz)) is True]
    else:
        puzzles = [args.puzzle]

    print(f"\nWill calculate regions masks for: {puzzles}\n")
    for puzzle in puzzles:

        ######
        # PREPARE PIECES AND GRIDS
        #
        # pieces is a list of dictionaries with the pieces (and mask, cm, id)
        # img_parameters contains the size of the image and of the pieces
        # ppars contains all the values needed for computing stuff (p_hs, comp_range..)
        # ppars is a dict but can be accessed by pieces_paramters.property!
        print()
        print("-" * 50)
        print(puzzle)
        # old version
        pieces, img_parameters = prepare_pieces_v2(fnames, args.dataset, puzzle, args.num_pieces, verbose=True)

        # PARAMETERS
        puzzle_root_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset, puzzle)
        cmp_parameter_path = os.path.join(puzzle_root_folder, 'regions_parameters.json')
        # if os.path.exists(cmp_parameter_path):
        #     print("never tested! remove this comment afterwars (line 53 of comp_irregular.py)")
        #     with open(cmp_parameter_path, 'r') as cp:
        #         ppars = json.load(cp)
        # else:
        ppars = calc_parameters_v2(img_parameters, args.xy_step, args.xy_grid_points, args.theta_step)
        # for ppk in ppars.keys():
        #     if type(ppars[ppk])== np.uint8:
        #         ppars[ppk] = int(ppars[ppk])
        #     print(ppk, ":", type(ppars[ppk]))
        # pdb.set_trace()
        with open(cmp_parameter_path, 'w') as cpj:
            json.dump(ppars, cpj, indent=3)
        print("saved json regions file")

        # INCLUDE SHAPE
        pieces = include_shape_info(fnames, pieces, args.dataset, puzzle, args.method, line_based=False, line_thickness=3, motif_based=False)

        grid_size_xy = ppars.comp_matrix_shape[0]
        grid_size_rot = ppars.comp_matrix_shape[2]
        grid, grid_step_size = create_grid(grid_size_xy, ppars.p_hs, ppars.canvas_size)

        print()
        print('#' * 50)
        print('SETTINGS')
        print(f"The puzzle (maybe rescaled) has size {ppars.img_size[0]}x{ppars.img_size[1]} pixels")
        print(f'Pieces are squared images of {ppars.piece_size}x{ppars.piece_size} pixels (p_hs={ppars.p_hs})')
        print(f"This puzzle has {ppars.num_pieces} pieces")
        print(
            f'The region matrix has shape: [{grid_size_xy}, {grid_size_xy}, {grid_size_rot}, {len(pieces)}, {len(pieces)}]')
        print(f'Using a grid on xy and {grid_size_rot} rotations on {len(pieces)} pieces')
        print(f'\txy_step: {ppars.xy_step}, rot_step: {ppars.theta_step}')
        print(f'Canvas size: {ppars.canvas_size}x{ppars.canvas_size}')
        print('#' * 50)
        print()

        ## CREATE MATRIX
        RM_combo = np.zeros((grid_size_xy, grid_size_xy, grid_size_rot, len(pieces), len(pieces)))
        RM_motifs = np.zeros((grid_size_xy, grid_size_xy, grid_size_rot, len(pieces), len(pieces)))
        RM_lines = np.zeros((grid_size_xy, grid_size_xy, grid_size_rot, len(pieces), len(pieces)))
        RM_shapes = np.zeros((grid_size_xy, grid_size_xy, grid_size_rot, len(pieces), len(pieces)))
        for i in range(len(pieces)):
            for j in range(len(pieces)):
                #j = 3
                print(f"regions for pieces {i:>2} and {j:>2}", end='\r')
                if i == j:
                    RM_shapes[:, :, :, j, i] = -1
                    RM_lines[:, :, :, j, i] = -1
                    RM_motifs[:, :, :, j, i] = -1
                    RM_combo[:, :, :, j, i] = -1
                else:
                    center_pos = ppars.canvas_size // 2
                    piece_i_on_canvas = place_on_canvas(pieces[i], (center_pos, center_pos), ppars.canvas_size, 0)

                    if 'lines_mask' in pieces[i].keys():
                        mask_i = cv2.resize(piece_i_on_canvas['lines_mask'], (ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]))
                        outside_borders_i = get_outside_borders(mask_i, borders_width=1)
                    if 'motif_mask' in pieces[i].keys():
                        mask_i = cv2.resize(piece_i_on_canvas['motif_mask'], (ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]))
                        outside_borders_i = get_outside_borders(mask_i, borders_width=1)

                    for t in range(grid_size_rot):
                        piece_j_on_canvas = place_on_canvas(pieces[j], (center_pos, center_pos), ppars.canvas_size, t * ppars.theta_step)

                        # SHAPE case - BASIC
                        overlap_shapes = cv2.filter2D(piece_i_on_canvas['mask'], -1, piece_j_on_canvas['mask'])
                        thresholded_regions_map = (overlap_shapes > ppars.threshold_overlap).astype(np.int32)
                        around_borders_trm = get_borders_around(thresholded_regions_map.astype(np.uint8),
                                                                border_dilation=int(
                                                                    ppars.borders_regions_width_outside * ppars.xy_step),
                                                                border_erosion=int(
                                                                    ppars.borders_regions_width_inside * ppars.xy_step))
                        thresholded_regions_map *= -1
                        thresholded_regions_map += 2 * (around_borders_trm > 0)
                        thresholded_regions_map = np.clip(thresholded_regions_map, -1, 1)

                        #  LINES case
                        if 'lines_mask' in pieces[i].keys():
                            mask_j = cv2.resize(piece_j_on_canvas['lines_mask'], (ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]))
                            outside_borders_j = get_outside_borders(mask_j, borders_width=1)
                            overlap_lines = cv2.filter2D(piece_i_on_canvas['lines_mask'], -1, piece_j_on_canvas['lines_mask'])
                            binary_overlap_lines = (overlap_lines > ppars.threshold_overlap_lines).astype(np.int32)
                            resized_lines = np.array(Image.fromarray(binary_overlap_lines).resize((ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]), Image.Resampling.NEAREST))

                            # COMBO for LINES-case
                            combo = thresholded_regions_map * binary_overlap_lines
                            combo[thresholded_regions_map < 0] = -1  # enforce -1 in the overlapping areas
                            # ??? overlap lines is calculate independently from line-category ???
                            # IF we calculate separately overlap of different category, will it improve results ??
                        else:
                            resized_lines = np.zeros((ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]))

                        #  MOTIFS case
                        if 'motif_mask' in pieces[i].keys():
                            mask_j = cv2.resize(piece_j_on_canvas['motif_mask'], (ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]))
                            outside_borders_j = get_outside_borders(mask_j, borders_width=1)  # is not used ???

                            # check FILTER !!! - probably IS the same as ???:
                            # overlap_motifs = cv2.filter2D(piece_i_on_canvas['motif_mask'], -1, piece_j_on_canvas['motif_mask'])

                            n_motifs = piece_i_on_canvas['motif_mask'].shape[2]
                            mask_mt = np.zeros((piece_i_on_canvas['motif_mask'].shape[0], piece_i_on_canvas['motif_mask'].shape[1], n_motifs))
                            for mt in range(n_motifs):
                                a = piece_i_on_canvas['motif_mask'][:, :, mt]
                                b = piece_j_on_canvas['motif_mask'][:, :, mt]
                                if np.sum(a) > 0 and np.sum(b) > 0:
                                    mask_mt[:, :, mt] = cv2.filter2D(a, -1, b)
                                    #plt.subplot(1,3,1); plt.imshow(a)
                                    #plt.subplot(1,3,2); plt.imshow(b)
                                    #plt.subplot(1,3,3); plt.imshow(mask_mt[:, :, mt])
                                    #plt.show()

                            overlap_motifs = np.sum(mask_mt, 2)
                            binary_overlap_motifs = (overlap_motifs > ppars.threshold_overlap_motifs).astype(np.int32)  # CHECK !!!
                            resized_motifs = np.array(Image.fromarray(binary_overlap_motifs).resize((ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]), Image.Resampling.NEAREST))

                            # COMBO for MOTIFS case
                            combo = thresholded_regions_map * binary_overlap_motifs
                            combo[thresholded_regions_map < 0] = -1  # enforce -1 in the overlapping areas
                        else:
                            resized_motifs = np.zeros((ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]))

                        # we convert the matrix to resize the image without losing the values
                        thr_reg_map_shape_uint = (thresholded_regions_map + 1).astype(np.uint8)
                        thr_reg_map_comp_range = thr_reg_map_shape_uint[ppars.p_hs + 1:-(ppars.p_hs + 1), ppars.p_hs + 1:-(ppars.p_hs + 1)]
                        resized_shape = np.array(Image.fromarray(thr_reg_map_comp_range).resize((ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]), Image.Resampling.NEAREST))

                        if 'motif_mask' in pieces[i].keys() or 'lines_mask' in pieces[i].keys():
                            combo_uint = (combo + 1).astype(np.uint8)
                            combo_comp_range = combo_uint[ppars.p_hs + 1:-(ppars.p_hs + 1), ppars.p_hs + 1:-(ppars.p_hs + 1)]
                            resized_combo = np.array(Image.fromarray(combo_comp_range).resize((ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]), Image.Resampling.NEAREST))
                        else:
                            resized_combo = resized_shape
                        #resized_lines2 = cv2.resize(binary_overlap_lines.astype(np.uint8), (ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]), cv2.INTER_NEAREST)
                        #resized_motifs2 = cv2.resize(binary_overlap_motifs.astype(np.uint8), (ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]), cv2.INTER_NEAREST)
                        #resized_shape2 = cv2.resize(thr_reg_map_comp_range, (ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]), cv2.INTER_NEAREST)
                        #resized_combo2 = cv2.resize(combo_comp_range, (ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]))

                        # PAIR OVERLAP VISUALIZATION
                        # plt.subplot(321); plt.imshow(piece_i_on_canvas['img'])
                        # plt.subplot(322); plt.imshow(piece_j_on_canvas['img'])
                        # plt.subplot(323); plt.imshow(binary_overlap_motifs)
                        # plt.subplot(324); plt.imshow(overlap_shapes)
                        # plt.subplot(325); plt.imshow(resized_motifs)
                        # plt.subplot(326); plt.imshow(resized_combo)
                        # plt.show()

                        # These are the matrices
                        RM_combo[:, :, t, j, i] = (resized_combo.astype(np.int32) - 1)
                        RM_motifs[:, :, t, j, i] = resized_motifs
                        RM_lines[:, :, t, j, i] = resized_lines
                        RM_shapes[:, :, t, j, i] = (resized_shape.astype(np.int32) - 1)

                        # if args.DEBUG is True:
                        #     pdb.set_trace()
                        #     plt.suptitle(f"Piece {i} against piece {j} on rotation {t}")
                        #     plt.subplot(631);
                        #     plt.imshow(piece_i_on_canvas['img']);
                        #     plt.title("Fixed in the center")
                        #     plt.subplot(632);
                        #     plt.imshow(piece_j_on_canvas['img']);
                        #     plt.title("Moving around")
                        #     coords = (center_pos + 3 * ppars.xy_step, center_pos - 1 * ppars.xy_step)
                        #     print(coords)
                        #     # piece_j_correct = place_on_canvas(pieces[j], coords, ppars.canvas_size, 0)
                        #     # plt.subplot(633); plt.imshow(piece_i_on_canvas['img'] + piece_j_correct['img'])
                        #     # plt.subplot(333); plt.imshow(around_borders_trm); plt.title("Borders")
                        #     # shapes
                        #     plt.subplot(634)
                        #     plt.imshow(overlap_shapes)
                        #     plt.title("Overlap Shapes")
                        #     # plt.subplot(435); plt.imshow(around_borders_trm); plt.title("Borders")
                        #     # plt.subplot(435); plt.imshow(thresholded_regions_map); plt.title("Region Map Shapes")
                        #     plt.subplot(635)
                        #     plt.imshow(thr_reg_map_comp_range)
                        #     plt.title("Region Map Shapes (uint8)")
                        #     plt.subplot(636)
                        #     plt.imshow(resized_shape)
                        #     plt.title("Region Map Shapes (resized)")
                        #     # lines
                        #     plt.subplot(637)
                        #     plt.imshow(piece_i_on_canvas['lines_mask'])
                        #     plt.title(f"Lines Mask {i}")
                        #     plt.subplot(638)
                        #     plt.imshow(piece_j_on_canvas['lines_mask'])
                        #     plt.title(f"Lines Mask {j}")
                        #     # plt.subplot(639); plt.imshow(resized_lines); plt.title("Overlap Lines (resized)")
                        #     # lines
                        #     plt.subplot(6, 3, 10)
                        #     plt.imshow(overlap_lines)
                        #     plt.title("Overlap Lines (values)")
                        #     plt.subplot(6, 3, 11)
                        #     plt.imshow(binary_overlap_lines)
                        #     plt.title("Overlap Lines (mask)")
                        #     plt.subplot(6, 3, 12)
                        #     plt.imshow(resized_lines)
                        #     plt.title("Overlap Lines (resized)")
                        #     # combo
                        #     plt.subplot(6, 3, 13)
                        #     plt.imshow(combo)
                        #     plt.title("Overlap Combo (mask)")
                        #     plt.subplot(6, 3, 14)
                        #     plt.imshow(combo_comp_range)
                        #     plt.title("Overlap Combo (uint8)")
                        #     plt.subplot(6, 3, 15)
                        #     plt.imshow(resized_combo)
                        #     plt.title("Overlap Combo (resized)")
                        #     # results
                        #     plt.subplot(6, 3, 16);
                        #     plt.imshow(RM_lines[:, :, t, j, i]);
                        #     plt.title("Lines")
                        #     plt.subplot(6, 3, 17);
                        #     plt.imshow(RM_shapes[:, :, t, j, i]);
                        #     plt.title("Shapes")
                        #     plt.subplot(6, 3, 18);
                        #     plt.imshow(RM_combo[:, :, t, j, i]);
                        #     plt.title("Combo")
                        #     plt.show()
                        #     pdb.set_trace()
        print("\n")
        print('Done calculating')
        print('#' * 50)
        print('Saving the matrix..')
        output_root_dir = fnames.output_dir
        output_folder = os.path.join(output_root_dir, args.dataset, puzzle, fnames.rm_output_name)
        # should we add this to the folder? it will create a subfolder that we may not need
        # f"{ppars.rm_output_dir}_{puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}")
        vis_folder = os.path.join(output_folder, fnames.visualization_folder_name)
        os.makedirs(vis_folder, exist_ok=True)
        RM_D = {}
        RM_D['RM'] = RM_combo
        RM_D['RM_motifs'] = RM_motifs
        RM_D['RM_lines'] = RM_lines
        RM_D['RM_shapes'] = RM_shapes

        filename = f'{output_folder}/RM_{puzzle}'
        savemat(f'{filename}.mat', RM_D)
        if args.save_visualization is True:
            print('Creating visualization')
            file_partial_name = f'{puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}'
            save_vis(RM_combo, pieces, ppars.theta_step, os.path.join(vis_folder, f'visualization_combo_{file_partial_name}'), f"regions matrix {puzzle}", save_every=4, all_rotation=False)
            save_vis(RM_lines, pieces, ppars.theta_step, os.path.join(vis_folder, f'visualization_lines_{file_partial_name}'), f"overlap {puzzle}", save_every=4, all_rotation=False)
            save_vis(RM_motifs, pieces, ppars.theta_step, os.path.join(vis_folder, f'visualization_motifs_{file_partial_name}'), f"overlap {puzzle}", save_every=4, all_rotation=False)
            save_vis(RM_shapes, pieces, ppars.theta_step, os.path.join(vis_folder, f'visualization_shapes_{file_partial_name}'), f"borders {puzzle}", save_every=4, all_rotation=False)
        print(f'Done with {puzzle}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computing compatibility matrix')
    parser.add_argument('--dataset', type=str, default='repair', help='dataset (name of the folders)')
    parser.add_argument('--puzzle', type=str, default='',
                        help='puzzle to work on - leave empty to generate for the whole dataset')
    parser.add_argument('--method', type=str, default='exact', help='method line detection')  # exact, manual, deeplsd
    parser.add_argument('--save_everything', type=bool, default=False, help='save also overlap and borders matrices')
    parser.add_argument('--save_visualization', type=bool, default=True,
                        help='save an image that showes the matrices color-coded')
    parser.add_argument('-np', '--num_pieces', type=int, default=0,
                        help='number of pieces (per side) - use 0 (default value) for synthetic pieces')  # 8
    parser.add_argument('--xy_step', type=int, default=10, help='the step (in pixels) between each grid point')
    parser.add_argument('--xy_grid_points', type=int, default=51,
                        help='the number of points in the grid (for each axis, total number will be the square of what is given)')
    parser.add_argument('--theta_step', type=int, default=90, help='degrees of each rotation')
    parser.add_argument('--DEBUG', action='store_true', default=False,
                        help='WARNING: will use debugger! It stops and show the matrices!')
    args = parser.parse_args()
    main(args)