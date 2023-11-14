
from puzzle_utils.shape_utils import prepare_pieces, create_grid, shape_pairwise_compatibility, \
    get_outside_borders, place_on_canvas
import numpy as np
import scipy
import argparse 
import pdb
import matplotlib.pyplot as plt 
import cv2
import json, os 
from configs import repair_cfg as cfg
from configs import folder_names as fnames

from puzzle_utils.visualization import save_vis

def main(args):

    ## PREPARE PIECES AND GRIDS
    #pdb.set_trace()
    pieces = prepare_pieces(cfg, fnames, args.dataset, args.puzzle)
    grid_size_xy = cfg.comp_matrix_shape[0]
    grid_size_rot = cfg.comp_matrix_shape[2]
    grid, grid_step_size = create_grid(grid_size_xy, cfg.p_hs, cfg.canvas_size)
    print('#' * 50)
    print('SETTINGS')
    print(f'RM has shape: [{grid_size_xy}, {grid_size_xy}, {grid_size_rot}, {len(pieces)}, {len(pieces)}]')
    print(f'Using a grid  on xy and {grid_size_rot} rotations on {len(pieces)} pieces')
    print(f'Pieces are squared images of {cfg.piece_size}x{cfg.piece_size} pixels (p_hs={cfg.p_hs})')
    print(f'xy_step: {cfg.xy_step}, rot_step: {cfg.theta_step}')
    print(f'Canvas size: {cfg.canvas_size}x{cfg.canvas_size}')
    print('#' * 50)
    #pdb.set_trace()
    ## CREATE MATRIX
    RM = np.zeros((grid_size_xy, grid_size_xy, grid_size_rot, len(pieces), len(pieces)))
    for i in range(len(pieces)):
        for j in range(len(pieces)):
            if i == j:
                RM[:,:,:,i,j] = -1
            else:
                print(f"regions for pieces {i:>2} and {j:>2}", end='\r')
                #pdb.set_trace()
                center_pos = cfg.canvas_size // 2
                piece_i_on_canvas = place_on_canvas(pieces[i], (center_pos, center_pos), cfg.canvas_size, 0)
                mask_i = cv2.resize(piece_i_on_canvas['mask'], (cfg.comp_matrix_shape[0], cfg.comp_matrix_shape[1]))
                #outside_borders_i = get_outside_borders(mask_i, borders_width=1)
                for t in range(grid_size_rot):
                    piece_j_on_canvas = place_on_canvas(pieces[j], (center_pos, center_pos), cfg.canvas_size, t * cfg.theta_step)
                    mask_j = cv2.resize(piece_j_on_canvas['mask'], (cfg.comp_matrix_shape[0], cfg.comp_matrix_shape[1]))
                    #outside_borders_j = get_outside_borders(mask_j, borders_width=1)
                    overlap_conv = cv2.filter2D(mask_i, -1, mask_j)
                    thresholded_regions_map = (overlap_conv > cfg.threshold_overlap).astype(np.int32)
                    thresholded_regions_map *= -1
                    outside_borders_trm = get_outside_borders(thresholded_regions_map.astype(np.uint8), cfg.borders_regions_width)
                    thresholded_regions_map += outside_borders_trm > 0
                    #candidate_region = cv2.filter2D(outside_borders_i, -1, outside_borders_j)
                    RM[:,:,t,i,j] = (thresholded_regions_map)

    print()
    print('Done calculating')
    print('#' * 50)
    print('Saving the matrix..')     
    output_folder = os.path.join(f"{fnames.output_dir}_8x8", args.dataset, args.puzzle, fnames.rm_output_name)
    # should we add this to the folder? it will create a subfolder that we may not need
    # f"{cfg.rm_output_dir}_{args.puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}")
    vis_folder = os.path.join(output_folder, fnames.visualization_folder_name)
    os.makedirs(vis_folder, exist_ok=True)
    RM_D = {}
    RM_D['RM'] = RM
    filename = f'{output_folder}/RM'
    scipy.io.savemat(f'{filename}.mat', RM_D)
    if cfg.save_visualization is True:
        print('Creating visualization')
        save_vis(RM, pieces, os.path.join(vis_folder, f'visualization_{args.puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}'), f"regions matrix {args.puzzle}", all_rotation=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Computing compatibility matrix')
    parser.add_argument('--dataset', type=str, default='repair', help='dataset (name of the folders)')
    parser.add_argument('--puzzle', type=str, default='decor_1_lines', help='puzzle to work on')
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
