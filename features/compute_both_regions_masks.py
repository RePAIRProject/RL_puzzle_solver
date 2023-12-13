import numpy as np
import scipy
import argparse 
import pdb
import matplotlib.pyplot as plt 
import cv2
import json, os 

#from configs import repair_cfg as cfg
from configs import folder_names as fnames

from puzzle_utils.shape_utils import prepare_pieces_v2, create_grid, shape_pairwise_compatibility, \
    get_outside_borders, place_on_canvas, get_borders_around, include_shape_info
from puzzle_utils.pieces_utils import calc_parameters
from puzzle_utils.visualization import save_vis

def main(args):

    if args.puzzle == '':  
        puzzles = os.listdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset))
        puzzles = [puz for puz in puzzles if os.path.isdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset,puz)) is True]
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
        pieces, img_parameters = prepare_pieces_v2(fnames, args.dataset, puzzle, args.num_pieces, verbose=True)
        # PARAMETERS
        puzzle_root_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset, puzzle)
        cmp_parameter_path = os.path.join(puzzle_root_folder, 'compatibility_parameters.json')
        if os.path.exists(cmp_parameter_path):
            print("never tested! remove this comment afterwars (line 53 of comp_irregular.py)")
            with open(cmp_parameter_path, 'r') as cp:
                ppars = json.load(cmp_parameter_path)
        else:
            ppars = calc_parameters(img_parameters)

        # INCLUDE SHAPE
        pieces = include_shape_info(fnames, pieces, args.dataset, puzzle, args.method)

        #pdb.set_trace()
        grid_size_xy = ppars.comp_matrix_shape[0]
        grid_size_rot = ppars.comp_matrix_shape[2]
        grid, grid_step_size = create_grid(grid_size_xy, ppars.p_hs, ppars.canvas_size)

        print()
        print('#' * 50)
        print('SETTINGS')
        print(f"The puzzle (maybe rescaled) has size {ppars.img_size[0]}x{ppars.img_size[1]} pixels")
        print(f'Pieces are squared images of {ppars.piece_size}x{ppars.piece_size} pixels (p_hs={ppars.p_hs})')
        print(f"This puzzle has {ppars.num_pieces} pieces")
        print(f'The region matrix has shape: [{grid_size_xy}, {grid_size_xy}, {grid_size_rot}, {len(pieces)}, {len(pieces)}]')
        print(f'Using a grid on xy and {grid_size_rot} rotations on {len(pieces)} pieces')
        print(f'\txy_step: {ppars.xy_step}, rot_step: {ppars.theta_step}')
        print(f'Canvas size: {ppars.canvas_size}x{ppars.canvas_size}')
        print('#' * 50)
        print()

        ## CREATE MATRIX                      
        RM_combo = np.zeros((grid_size_xy, grid_size_xy, grid_size_rot, len(pieces), len(pieces)))
        RM_lines = np.zeros((grid_size_xy, grid_size_xy, grid_size_rot, len(pieces), len(pieces)))
        RM_shapes = np.zeros((grid_size_xy, grid_size_xy, grid_size_rot, len(pieces), len(pieces)))
        for i in range(len(pieces)):
            for j in range(len(pieces)):
                print(f"regions for pieces {i:>2} and {j:>2}", end='\r')
                if i == j:
                    RM_combo[:,:,:,j,i] = -1
                    RM_lines[:,:,:,j,i] = -1
                    RM_shapes[:,:,:,j,i] = -1
                else:
                    #pdb.set_trace()
                    center_pos = ppars.canvas_size // 2
                    piece_i_on_canvas = place_on_canvas(pieces[i], (center_pos, center_pos), ppars.canvas_size, 0)
                    mask_i = cv2.resize(piece_i_on_canvas['lines_mask'], (ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]))
                    #outside_borders_i = get_outside_borders(mask_i, borders_width=1)
                    for t in range(grid_size_rot):
                        piece_j_on_canvas = place_on_canvas(pieces[j], (center_pos, center_pos), ppars.canvas_size, t * ppars.theta_step)
                        mask_j = cv2.resize(piece_j_on_canvas['lines_mask'], (ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]))
                        #outside_borders_j = get_outside_borders(mask_j, bordpers_width=1)
                        overlap_lines = cv2.filter2D(piece_i_on_canvas['lines_mask'], -1, piece_j_on_canvas['lines_mask'])
                        overlap_shapes = cv2.filter2D(piece_i_on_canvas['mask'], -1, piece_j_on_canvas['mask'])
                        thresholded_regions_map = (overlap_shapes > ppars.threshold_overlap).astype(np.int32)
                        thresholded_regions_map *= -1
                        around_borders_trm = get_borders_around(thresholded_regions_map.astype(np.uint8), border_dilation=int(ppars.borders_regions_width_outside*ppars.xy_step), border_erosion=int(ppars.borders_regions_width_inside*ppars.xy_step))
                        thresholded_regions_map += 2*(around_borders_trm > 0)
                        thresholded_regions_map = np.clip(thresholded_regions_map, -1, 1)
                        binary_overlap_lines = (overlap_lines > ppars.threshold_overlap_lines).astype(np.int32)
                        combo = thresholded_regions_map * binary_overlap_lines
                        combo[thresholded_regions_map < 0] = -1 #enforce -1 in the overlapping areas

                        # we convert the matrix to resize the image without losing the values
                        converted_shape = (thresholded_regions_map+1).astype(np.uint8)
                        resized_shape = cv2.resize(converted_shape, (ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]), cv2.INTER_NEAREST)

                        converted_combo = (combo+1).astype(np.uint8)
                        resized_combo = cv2.resize(converted_combo, (ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]), cv2.INTER_NEAREST)

                        # These are the matrices
                        RM_combo[:,:,t,j,i] = (resized_combo.astype(np.int32) - 1)
                        RM_lines[:,:,t,j,i] = cv2.resize(binary_overlap_lines.astype(np.uint8), (ppars.comp_matrix_shape[0], ppars.comp_matrix_shape[1]))
                        RM_shapes[:,:,t,j,i] = (resized_shape.astype(np.int32) - 1)
                        

        print("\n")
        print('Done calculating')
        print('#' * 50)
        print('Saving the matrix..')     
        if args.num_pieces == 8:
            output_root_dir = f"{fnames.output_dir}_8x8"
        else:
            output_root_dir = fnames.output_dir
        output_folder = os.path.join(output_root_dir, args.dataset, puzzle, fnames.rm_output_name)
        # should we add this to the folder? it will create a subfolder that we may not need
        # f"{ppars.rm_output_dir}_{puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}")
        vis_folder = os.path.join(output_folder, fnames.visualization_folder_name)
        os.makedirs(vis_folder, exist_ok=True)
        RM_D = {}
        RM_D['RM'] = RM_combo
        RM_D['RM_lines'] = RM_lines
        RM_D['RM_shapes'] = RM_shapes

        filename = f'{output_folder}/RM_{puzzle}'
        scipy.io.savemat(f'{filename}.mat', RM_D)
        if args.save_visualization is True:
            print('Creating visualization')
            save_vis(RM_combo, pieces, os.path.join(vis_folder, f'visualization_combo_{puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}'), f"regions matrix {puzzle}", all_rotation=False)
            save_vis(RM_lines, pieces, os.path.join(vis_folder, f'visualization_lines_{puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}'), f"overlap {puzzle}", all_rotation=False)
            save_vis(RM_shapes, pieces, os.path.join(vis_folder, f'visualization_shapes_{puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}'), f"borders {puzzle}", all_rotation=False)
        print(f'Done with {puzzle}\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Computing compatibility matrix')
    parser.add_argument('--dataset', type=str, default='repair', help='dataset (name of the folders)')
    parser.add_argument('--puzzle', type=str, default='', help='puzzle to work on - leave empty to generate for the whole dataset')
    parser.add_argument('--method', type=str, default='deeplsd', help='method line detection')  # exact, manual, deeplsd
    parser.add_argument('--save_everything', type=bool, default=False, help='save also overlap and borders matrices')
    parser.add_argument('--save_visualization', type=bool, default=True, help='save an image that showes the matrices color-coded')
    parser.add_argument('-np', '--num_pieces', type=int, default=0, help='number of pieces (per side) - use 0 (default value) for synthetic pieces')  # 8
    args = parser.parse_args()
    main(args)


# % MASKs of pairs of fragments

# pieces = [194:198, 200:203];

# ang = 45;              % rotation stem in gradi
# rot = 0:ang:360-ang;

# ni = 9;
# nj = 9;

# zerCR = zeros(51,51,8);
# CR    = zeros(51,51,8,9,9);
# CRm   = CR;
# CRneg = CR;
# CRcont= CR;
# %CR_new = CR;

# for i=1:ni
#     im_num  = pieces(i);
#     in_file = (sprintf('%s%s%s','C:\Users\Marina\PycharmProjects\WP3-PuzzleSolving\Compatibility\data\repair\group_28\ready\RPf_00',num2str(im_num),'.png'));
    
#     [Im,~,alfa] = imread(in_file);
#     A = im2double(alfa); 
#     A = ceil(A);
#     %figure; imshow(A);
    
#     for j=1:nj
#         if eq(i,j)
#             CR(:,:,:,j,i) = zerCR-1;
#         else
#         im_num  = pieces(j);
#         in_file = (sprintf('%s%s%s','C:\Users\Marina\PycharmProjects\WP3-PuzzleSolving\Compatibility\data\repair\group_28\ready\RPf_00',num2str(im_num),'.png'));
#         [Im,~,alfa] = imread(in_file);
#         B = im2double(alfa);
#         B = ceil(B);

#         for t=1:size(rot,2)            
#             Br = imrotate(B,rot(t),'crop'); % figure; imshow(B); %figure; imshow(Br);
#             C = conv2(A,rot90(Br,2));               
            
#             C1 = imresize(C,[51,51],'nearest');
#             CR(:,:,t,j,i) = C1;
            
#             C0 = C1; 
#             C0(C0>0)=-1; 
#             CRneg(:,:,t,j,i) = C0;
            
#             cc = contourc(double(C1),[1 1]); % will be silent and faster%
#             ix = round(cc(1,2:end)'); 
#             iy = round(cc(2,2:end)'); 
            
#             C2 = zeros(size(C1));
#             for ii=1:size(ix,1), C2(iy(ii),ix(ii))=10; end              
#             CRcont(:,:,t,j,i) = C2;
            
#             CRm(:,:,t,j,i) = C2+C0;
#             %figure; imshow(C2);            
#         end
#         end
#     end
# end

# %% compute and plot overlap area of each pair of pieces

# CRm(CRm>0) = 1;
# % CR_new = CR;
# % CR_new(CR_new>5) = -1;
# % CR_new(CR_new>0) = 1;

# %% Plot contour matrices
# t=2;
# % figure;
# % jj=0;
# % nii = 9;
# % njj = 9;
# % for i=1:nii
# %     for j=1:njj        
# %         Rr = CR(:,:,t,j,i);
# %         jj=jj+1;
# %         subplot(nii,njj,jj); image(Rr,'CDataMapping','scaled'); colorbar;       
# %     end
# % end
# % 
# % % Plot neg-matrices
# % figure;
# % jj=0;
# % nii = 9;
# % njj = 9;
# % for i=1:nii
# %     for j=1:njj
# %         t=1;
# %         Rr = CRneg(:,:,t,j,i);
# %         jj=jj+1;
# %         subplot(nii,njj,jj); image(Rr,'CDataMapping','scaled'); colorbar;       
# %     end
# % end
# % 
# % % Plot count-matrices
# % figure;
# % jj=0;
# % nii = 9;
# % njj = 9;
# % for i=1:nii
# %     for j=1:njj
# %         t=1;
# %         Rr = CRcont(:,:,t,j,i);
# %         jj=jj+1;
# %         subplot(nii,njj,jj); image(Rr,'CDataMapping','scaled'); colorbar;       
# %     end
# % end

# % Plot mask-matrices
# t=7;

# figure;
# jj=0;
# nii = 9;
# njj = 9;
# for i=1:nii
#     for j=1:njj
#         Rr = CRm(:,:,t,j,i);
#         jj=jj+1;
#         subplot(nii,njj,jj); image(Rr,'CDataMapping','scaled'); colorbar;       
#     end
# end


# R_mask = CRm;
# save('R_mask51_45cont2.mat', 'R_mask')
