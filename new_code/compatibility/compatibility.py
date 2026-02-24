import scipy 
import numpy as np 
import cv2 
import shapely 
from transformers import ViTForImageClassification, AutoImageProcessor
import torch 
from PIL import Image 

from utils.puzzle_utils import Puzzle, PuzzlePiece
from utils.visualization_utils import save_pairwise_matrix_visualization_to_file
from utils.parameters_utils import Configuration, CustomYAMLEncoder
from compatibility.grid import PuzzleGrid, PieceOnCanvas, recalculate_position_after_rotation
from compatibility.alignment_scorer import AlignmentScorer
import matplotlib.pyplot as plt
from utils.visualization_utils import crop_to_content

import yaml
import os
import random
import json

############################################################################
#                                                                          #
#   ██████╗███╗   ███╗    ███╗   ███╗ █████╗ ████████╗██████╗ ██╗██╗  ██╗  #
#  ██╔════╝████╗ ████║    ████╗ ████║██╔══██╗╚══██╔══╝██╔══██╗██║╚██╗██╔╝  #
#  ██║     ██╔████╔██║    ██╔████╔██║███████║   ██║   ██████╔╝██║ ╚███╔╝   #
#  ██║     ██║╚██╔╝██║    ██║╚██╔╝██║██╔══██║   ██║   ██╔══██╗██║ ██╔██╗   #
#  ╚██████╗██║ ╚═╝ ██║    ██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║██║██╔╝ ██╗  #
#   ╚═════╝╚═╝     ╚═╝    ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝  #
#                                                                          #
#  ███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗██╗     ███████╗                   #
#  ████╗ ████║██╔═══██╗██╔══██╗██║   ██║██║     ██╔════╝                   #
#  ██╔████╔██║██║   ██║██║  ██║██║   ██║██║     █████╗                     #
#  ██║╚██╔╝██║██║   ██║██║  ██║██║   ██║██║     ██╔══╝                     #
#  ██║ ╚═╝ ██║╚██████╔╝██████╔╝╚██████╔╝███████╗███████╗                   #
#  ╚═╝     ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝ ╚══════╝╚══════╝                   #
#                                                                          #
############################################################################
class CompatibilityMatrixModule:

    def __init__(self, puzzle: Puzzle, params: dict, cfg: Configuration):

        self.puzzle = puzzle
        print(puzzle.name)
        # self.pieces = pcs_uts.load_pieces(puzzle_name=puzzle_name)
        # if use_feats == True:
        #     self.pieces = fts_uts.load_features(pieces=self.pieces, puzzle_name=puzzle_name)
        self.params = params
        # here we need to load stuff from the yaml file
        self.features_status = {}
        self.features = []
        self.check_features_status()      
        self.cfg = cfg #Configuration(puzzle.name) 
        # self.exp_folder is already set when we create the object
        # self.exp_folder = self.cfg.new_puzzle_single_run_random_folder_name()
        self.vis_params = params['compatibility']['save_visualization']
        self.save_vis = self.vis_params['save_CM']
        
        ## we could call here
        # self.prepare()


    # TODO: could be static!
    def check_features_status(self):
        features = self.params['compatibility']['features']
        for feature in features:
            self.features.append(feature)
            self.features_status[feature] = features[feature]['enabled']    

    def prepare(self):
        """
        All the parameters are set here
        It should be self-explanatory as it's just setting values (with predefined factors we hard-coded)
        """
        self.piece_size = self.params['preprocessing']['piece_size']
        self.p_hs = self.piece_size ## 2
        self.grid = PuzzleGrid(self.params['compatibility']['grid'], self.piece_size)
        self.RM_dict = np.load(self.cfg.get_RM_path(), allow_pickle=True).item()
        self.CM_size = (self.grid.xy_num_points, self.grid.xy_num_points, self.grid.theta_num_points, self.puzzle.num_of_pieces, self.puzzle.num_of_pieces)
        self.CM = {}

    def compute(self, verbose:int = 1):
        """
        Computes the region matrix for all pairs of pieces 
        for all features we have extracted
        This works calling the "pairwise" RM, can be good for debug
        """
        for feature in self.features:
            if verbose > 1:
                print("-" * 50)
            if self.features_status[feature] == True:
                self.CM[feature] = self._compute_feature_based_CM_wrapper(feature, verbose=verbose)
                if verbose > 1:
                    print(f"{feature}-based CM computed!")
            else:
                if verbose > 1:
                    print(f"{feature}-based CM skipped, as {feature} is disabled!")

    def _compute_feature_based_CM_wrapper(self, feature: str, verbose: int = 0):
        """
        Just a wrapper, will decide which method to call depending on the feature
        """
        if feature == 'shape':
            if verbose > 1:
                print("shape-based CM Computation")
            CM = self._compute_shape_based_CM(verbose=verbose)
        elif feature == 'lines':
            if verbose > 1:
                print("lines-based CM Computation")
            CM = self._compute_line_based_CM(verbose=verbose)
        elif feature == 'motives':
            if verbose > 1:
                print("motives-based CM Computation")
            CM = self._compute_motives_based_CM(verbose=verbose)
        elif feature == 'oracle':
            if verbose > 1:
                print("oracle CM computation")
            CM = self._compute_oracle_CM(verbose=verbose)
        elif feature == 'pairwise_alignment_discriminator' or feature == 'geometry':
            if verbose > 1:
                print("PAD CM computation")
            CM = self._compute_pad_CM(verbose=verbose)
        elif feature == 'geometry':
            if verbose > 1:
                print("PAD CM computation with Geometry alignment step")
            ## TODO - check dictionary !!!
            CM = self._compute_pad_CM_gemetric(verbose=verbose)
        elif feature == 'alignment_scorer':
            if verbose > 1:
                print("Alignment Scorer Compatibility")
            CM = self._compute_alignment_scorer_CM(verbose=verbose)
        else:
            raise Exception(f"{feature}-based CM not implemented yet!")

        return CM 

    ##################################
    #               SAVE             #
    ##################################
    def save(self, verbose=0):
        """ save """
        context_params = {}
        context_params['input_params'] = self.params
        context_params['grid_params'] = {'xy_num_points': self.grid.xy_num_points, 'theta_num_points': self.grid.theta_num_points, 'xy_step':self.grid.xy_step, 'theta_step':self.grid.theta_step, 
            'canvas_size': self.grid.canvas_size, 'pairwise_comp_range': self.grid.pairwise_comp_range}
        context_params['features'] = self.features_status
        context_params['puzzle'] = {'puzzle_name': self.puzzle.name, 'num_pieces': self.puzzle.num_of_pieces, 'piece_size': self.piece_size}
        # values of the matrix  
        self.CM['__context'] = context_params
        #breakpoint()
        np.save(self.cfg.get_CM_path(), self.CM)
        if self.save_vis == True:
            for feature in self.features:
                if self.features_status[feature] == True:
                    if verbose > 1:
                        print("-" * 50)
                        print(f"Saving {feature}-based CM")
                    if self.features_status[feature] == True:
                        save_pairwise_matrix_visualization_to_file(self.CM[feature], pieces=self.puzzle.pieces, rot_step=self.params['compatibility']['grid']['theta_step'], based_on=feature,
                                                                        output_folder=os.path.join(self.cfg.get_current_experiments_folder(), 'CM_vis'), visualization_params=self.vis_params)
        # input parameters
        input_params_path = self.cfg.get_CM_input_parameters_path() 
        with open(input_params_path, 'w') as f:
            yaml.dump(self.params, f, Dumper=CustomYAMLEncoder, default_flow_style=False)
        # output parameters
        context_params_path = self.cfg.get_CM_output_parameters_path() 
        with open(context_params_path, 'w') as f:
            yaml.dump(context_params, f, Dumper=CustomYAMLEncoder, default_flow_style=False)



    #####################################################
    #                                                   #
    #  ██████╗ ██████╗  █████╗  ██████╗██╗     ███████╗ #
    # ██╔═══██╗██╔══██╗██╔══██╗██╔════╝██║     ██╔════╝ #
    # ██║   ██║██████╔╝███████║██║     ██║     █████╗   #
    # ██║   ██║██╔══██╗██╔══██║██║     ██║     ██╔══╝   #
    # ╚██████╔╝██║  ██║██║  ██║╚██████╗███████╗███████╗ #
    #  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚══════╝ #
    #                                                   #
    #####################################################
    def _compute_oracle_CM(self, verbose: int = 3):
        
        """Loops over pairs of pieces - not symmetric yet"""
        self.oracle_params = self.params['compatibility']['features']['oracle']
        with open(self.cfg.get_GT_path(), 'r') as gtjf:
            self.gt = json.load(gtjf)
        with open(self.cfg.get_puzzle_info_path(), 'r') as pijf:
            self.puzzle_info = json.load(pijf)
        # gt_data = pd.read_csv(os.path.join(oracle_info['gt_root_folder'], f"{oracle_info['gt_puzzle_name']}.{oracle_info['gt_puzzle_name_extension']}"))

        if self.oracle_params['create_pairwise_alignments_dataset'] == True:
            print("\nCreating pairwise alignment datasets..\n\n")
            self.oracle_params['pairwise_alignments_dataset']['correct_alignment_folder'] = os.path.join(self.cfg.data_folder, self.oracle_params['pairwise_alignments_dataset']['data_folder'], 'positive', 'images')  #, oracle_info['gt_puzzle_name']) 
            os.makedirs(self.oracle_params['pairwise_alignments_dataset']['correct_alignment_folder'], exist_ok=True)
            self.oracle_params['pairwise_alignments_dataset']['wrong_alignment_folder'] = os.path.join(self.cfg.data_folder, self.oracle_params['pairwise_alignments_dataset']['data_folder'], 'negative', 'images')      #, oracle_info['gt_puzzle_name']) 
            os.makedirs(self.oracle_params['pairwise_alignments_dataset']['wrong_alignment_folder'], exist_ok=True)
            self.oracle_params['pairwise_alignments_dataset']['correct_alignment_masks_folder'] = os.path.join(self.cfg.data_folder, self.oracle_params['pairwise_alignments_dataset']['data_folder'], 'positive', 'masks')  #, oracle_info['gt_puzzle_name']) 
            os.makedirs(self.oracle_params['pairwise_alignments_dataset']['correct_alignment_masks_folder'], exist_ok=True)
            self.oracle_params['pairwise_alignments_dataset']['wrong_alignment_masks_folder'] = os.path.join(self.cfg.data_folder, self.oracle_params['pairwise_alignments_dataset']['data_folder'], 'negative', 'masks')      #, oracle_info['gt_puzzle_name']) 
            os.makedirs(self.oracle_params['pairwise_alignments_dataset']['wrong_alignment_masks_folder'], exist_ok=True)

        CM_oracle = np.zeros(self.CM_size)
        if verbose > 1:
            print()
        if 'adjacency' not in self.gt.keys():
            print("WARNING:\nno adjacency matrix, will compute oracle for all pairs!\n")
        for i in range(self.puzzle.num_of_pieces):
            for j in range(self.puzzle.num_of_pieces):
                if i != j:
                    adjacent_pieces = False
                    if 'adjacency' in self.gt.keys():
                        if [i,j] in self.gt['adjacency'] or [j,i] in self.gt['adjacency']: 
                            adjacent_pieces = True 
                    else:
                        adjacent_pieces = True
                    if adjacent_pieces == True:
                        if verbose > 1:
                            print(f'computing oracle CM[:, :, :, {i:02d}, {j:02d}]', end='\r')
                        RM_ij = np.ones((CM_oracle.shape[0], CM_oracle.shape[1], CM_oracle.shape[2]))
                        # breakpoint()
                        if self.oracle_params['create_pairwise_alignments_dataset'] == True:
                            RM_ij = self.RM_dict['oracle'][:, :, :, j, i]
                        #if np.sum(RM_ij > 0) > 0:
                        # breakpoint()
                        gt_piece_i = self.gt['pieces'][f'{i}']
                        gt_piece_j = self.gt['pieces'][f'{j}']
                        gt_pos_i = np.asarray([gt_piece_i['x'], gt_piece_i['y']]) #/ self.puzzle_info['pieces_image_size'][0] * self.puzzle.img_piece_size[0] / 0.166
                        gt_pos_j = np.asarray([gt_piece_j['x'], gt_piece_j['y']]) #/self.puzzle_info['pieces_image_size'][0] * self.puzzle.img_piece_size[0] / 0.166
                        gt_rel_j_vs_i = np.round((gt_pos_j - gt_pos_i)).astype(int) #* 1.5 # / self.grid.xy_step).astype(int)
                        # gt_pos_i_theta = -gt_piece_i['theta']
                        # gt_pos_j_theta = -gt_piece_j['theta']
                        if verbose > 2:
                            print("\nrelative GT:", gt_rel_j_vs_i)
                        CM_oracle[:, :, :, j, i] = self._compute_pairwise_oracle_CM(self.puzzle.pieces[i], self.puzzle.pieces[j], RM_ij=RM_ij, gt_rel_pos=gt_rel_j_vs_i, verbose=verbose)
                    # else: # it should already be zero!
                    #     CM_oracle[:, :, :, j, i] = np.zeros_like(CM_oracle[:, :, :, j, i])
        if verbose > 1:
            print()
        return CM_oracle
    
    # def _prepare_gt_data(self):
    #     """ just reorganizes the gt as a list with the index to be more `in line` with the rest of the data """
    def _pick_plausible_wrong_position(self, RM_ij: np.ndarray, xc: int, yc: int, max_attempts: int = 10):
        """ 
        Picks a plausible alignment position (positive value in the RM matrix) which is NOT the correct one (used for dataset) 
            - for now not using rotation
        ---
        Note: for small pieces with small grid (ex. polyomino) sometimes it fails (it is just random uniform after all)
        and leaves after max_attemtps. TODO: think of a smarter way to find a solution?
        """
        plausible_pos = np.where(RM_ij > 0)
        found = False
        attempts = 0
        while not found:
            rnd_idx = int(random.uniform(0, len(plausible_pos[0])))
            x_idx = plausible_pos[1][rnd_idx]
            y_idx = plausible_pos[0][rnd_idx] 
            if x_idx != xc and y_idx != yc:
                found = True
            attempts += 1
            if attempts > max_attempts:
                break

        xj_w, yj_w = self.grid.xy_values[x_idx, y_idx]
        return xj_w, yj_w, 0, found

    def _compute_pairwise_oracle_CM_R(self, piece_i: PuzzlePiece, piece_j: PuzzlePiece, RM_ij: np.ndarray, gt_rel_pos: np.ndarray, thetai: int, thetaj: int, verbose: int = 0):
        """
        For each pair of pieces, it places them on the canvas in the position `accepted` by RM_ij 
        and calls the scoring function to fill the pairwise compatibility matrix CM_ij 
        """
        kernel_size = 10*2+1
        kernel = np.ones((kernel_size, kernel_size))
        
        CM_ij = np.zeros((RM_ij.shape[0], RM_ij.shape[1], RM_ij.shape[2]))
        #CM_ij = np.zeros(tuple(self.RM_size[:3]))
        xj, yj = (np.asarray([self.grid.canvas_center, self.grid.canvas_center]) + np.asarray([gt_rel_pos[0], gt_rel_pos[1]])).tolist()
        x_idx = np.round(self.grid.xy_num_points / 2 + gt_rel_pos[0] / self.grid.xy_step).astype(int)
        y_idx = np.round(self.grid.xy_num_points / 2 + gt_rel_pos[1] / self.grid.xy_step).astype(int)
        x_i_idx = np.round(self.grid.xy_num_points / 2)
        y_i_idx = np.round(self.grid.xy_num_points / 2)
        #theta_idx = gt_rel_rot % self.grid.theta_num_points
        gt_rel_rot = np.round((thetaj - thetai)).astype(int)
        theta_i_idx = np.round(thetai // self.grid.theta_step).astype(int)
        theta_j_idx = np.round(thetaj // self.grid.theta_step).astype(int)

        new_xj, new_yj, new_thetaj = recalculate_position_after_rotation(self.grid.canvas_center, self.grid.canvas_center, xj, yj, thetai, thetaj)
        new_xj_idx, new_yj_idx, new_thetaj_idx = recalculate_position_after_rotation(x_i_idx, y_i_idx, x_idx, y_idx, theta_i_idx, theta_j_idx)
        # print(new_xj_idx, new_yj_idx, new_thetaj_idx)
        delta_theta_idx = np.round(gt_rel_rot // self.grid.theta_step).astype(int)
        # print(f"idx: {x_idx}, {y_idx}, pix: {xj}, {yj}\n")
        
        if 1 > 0: #np.max(abs(gt_rel_pos)) < (self.grid.p_hs): 
            # y_idx = np.round(yj / self.grid.xy_step).astype(int)
            
            # thetaj = self.grid.theta_values[0]
            # thetaj = gt_rel_rot % 360
            theta_idx = np.round(thetaj // self.grid.theta_step).astype(int)
            if verbose > 2:
                print(f"CM[{x_idx}, {y_idx}, {theta_idx}] = 1")
            
            y_c0 = np.ceil(yj-self.grid.p_hs).astype(int)
            y_c1 = np.ceil(yj+self.grid.p_hs+1).astype(int)
            x_c0 = np.ceil(xj-self.grid.p_hs).astype(int)
            x_c1 = np.ceil(xj+self.grid.p_hs+1).astype(int)
            if x_c0 < 0 or y_c0 < 0 or y_c1 > self.grid.canvas_size or x_c1 > self.grid.canvas_size:
                print("Out of the canvas! Error, skipping for now to see successive results.")
            else:
                # piece_i_on_canvas = PieceOnCanvas(piece=piece_i, grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=thetai, enabled_features=self.features_status)
                # piece_j_on_canvas = PieceOnCanvas(piece=piece_j, grid=self.grid, x=xj, y=yj, theta=thetaj, enabled_features=self.features_status)
                
                # correct after rotation!
                piece_j_on_canvas = PieceOnCanvas(piece=piece_j, grid=self.grid, x=new_xj, y=new_yj, theta=gt_rel_rot, enabled_features=self.features_status)
                piece_i_on_canvas = PieceOnCanvas(piece=piece_i, grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=0, enabled_features=self.features_status)
                piece_j_on_canvas_full_rot = PieceOnCanvas(piece=piece_j, grid=self.grid, x=new_xj, y=new_yj, theta=new_thetaj, enabled_features=self.features_status)
                piece_i_on_canvas_full_rot = PieceOnCanvas(piece=piece_i, grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=thetai, enabled_features=self.features_status)
                
                # import matplotlib.pyplot as plt 
                
                # plt.subplot(121)
                # plt.title(f"GT_rel: {gt_rel_pos}, xj: {new_xj:.1f}, yj:, {new_yj:.1f}, thetaj: {gt_rel_rot}, thetai: 0, pieces_size: {self.puzzle.img_piece_size}")
                # plt.imshow(piece_i_on_canvas.image  + piece_j_on_canvas.image)
                # plt.subplot(122)
                # plt.title(f"GT_rel: {gt_rel_pos}, xj: {xj}, yj:, {yj}, thetaj: {thetaj}, thetai: {thetai}, theta_idx: {theta_idx} pieces_size: {self.puzzle.img_piece_size}")
                # plt.imshow(piece_j_on_canvas_full_rot.image  + piece_i_on_canvas_full_rot.image)
                # plt.suptitle(f"i={piece_i.name}, j={piece_j.name}")
                # plt.scatter(self.grid.xy_values[0][0][1]+new_yj_idx*self.grid.xy_step, self.grid.xy_values[0][0][0]+new_xj_idx*self.grid.xy_step, color='blue')
                # plt.scatter(self.grid.xy_values[0][0][0]+new_xj_idx*self.grid.xy_step, self.grid.xy_values[0][0][1]+new_yj_idx*self.grid.xy_step, color='green')
                # plt.show()
                # breakpoint()
                if np.sum(cv2.dilate(piece_i_on_canvas.mask, kernel) * cv2.dilate(piece_j_on_canvas.mask, kernel) > 0): 
                    # CM_ij[y_idx, x_idx, theta_idx] = 1
                    # with rotation we need to fill the matrix in the new values!
                    new_yj_idx = np.round(new_yj_idx).astype(int)
                    new_xj_idx = np.round(new_xj_idx).astype(int)
                    new_thetaj_idx = np.round(delta_theta_idx).astype(int) # % self.grid.theta_num_points
                    CM_ij[new_yj_idx, new_xj_idx, new_thetaj_idx] = 1

                    if self.oracle_params['create_pairwise_alignments_dataset'] == True:
                        import matplotlib.pyplot as plt 
                        
                        # This is the PAIRWISE GROUND TRUTH
                        alignment_basename = f'{self.puzzle.name}_vis_{piece_i.name}_{piece_j.name}_{new_xj}_{new_yj}_{new_thetaj}'
                        if self.oracle_params['pairwise_alignments_dataset']['save_true_alignment'] == True:
                            true_alignment_name = f'{alignment_basename}_gt.png'
                            correctly_aligned_image, correctly_aligned_mask = piece_i_on_canvas.blend_with(piece_j_on_canvas, return_mask=True)
                            plt.subplot(221); plt.imshow(correctly_aligned_image)
                            plt.subplot(222); plt.imshow(crop_to_content(correctly_aligned_image, padding=self.oracle_params['pairwise_alignments_dataset']['padding']))
                            plt.subplot(223); plt.imshow(correctly_aligned_mask)
                            plt.subplot(224); plt.imshow(crop_to_content(correctly_aligned_mask, padding=self.oracle_params['pairwise_alignments_dataset']['padding']))
                            plt.show()
                            breakpoint()
                            if self.oracle_params['pairwise_alignments_dataset']['crop_images'] == True:
                                correctly_aligned_image = crop_to_content(correctly_aligned_image, padding=self.oracle_params['pairwise_alignments_dataset']['padding'])
                                correctly_aligned_mask = crop_to_content(correctly_aligned_mask, padding=self.oracle_params['pairwise_alignments_dataset']['padding'])
                            plt.imsave(os.path.join(self.oracle_params['pairwise_alignments_dataset']['correct_alignment_folder'], true_alignment_name), np.clip(correctly_aligned_image, 0, 1))
                            if self.oracle_params['pairwise_alignments_dataset']['save_masks'] == True:
                                cv2.imwrite(os.path.join(self.oracle_params['pairwise_alignments_dataset']['correct_alignment_masks_folder'], true_alignment_name), np.clip(correctly_aligned_mask, 0, 2))

                        if self.oracle_params['pairwise_alignments_dataset']['save_grid_alignment'] == True:
                            # This is the PAIRWISE "BEST" given the grid step that we have
                            grid_alignment_name = f'{alignment_basename}_grid.png'
                            xj_grid, yj_grid = self.grid.xy_values[x_idx, y_idx]
                            thetaj_grid = self.grid.theta_values[theta_idx]
                            piece_j_on_canvas_on_grid = PieceOnCanvas(piece=piece_j, grid=self.grid, x=xj_grid, y=yj_grid, theta=thetaj_grid, enabled_features=self.features_status)
                            grid_aligned_image, grid_aligned_mask = piece_i_on_canvas.blend_with(piece_j_on_canvas_on_grid, return_mask=True)
                            if self.oracle_params['pairwise_alignments_dataset']['crop_images'] == True:
                                grid_aligned_image = crop_to_content(grid_aligned_image, padding=self.oracle_params['pairwise_alignments_dataset']['padding'])
                                grid_aligned_mask = crop_to_content(grid_aligned_mask, padding=self.oracle_params['pairwise_alignments_dataset']['padding'])
                            plt.imsave(os.path.join(self.oracle_params['pairwise_alignments_dataset']['correct_alignment_folder'], grid_alignment_name), np.clip(grid_aligned_image, 0, 1))
                            if self.oracle_params['pairwise_alignments_dataset']['save_masks'] == True:
                                cv2.imwrite(os.path.join(self.oracle_params['pairwise_alignments_dataset']['correct_alignment_masks_folder'], grid_alignment_name), np.clip(grid_aligned_mask, 0, 2))

                        # These are randomly chosen "plausible" (but not correct!) alignment of the two pieces
                        if self.oracle_params['pairwise_alignments_dataset']['save_wrong_alignments'] == True:
                            for wk in range(self.oracle_params['pairwise_alignments_dataset']['wrong_alignments_num']):
                                plausible_alignment_name = f'{alignment_basename}_wrong_{wk}.png'
                                xj_w, yj_w, thetaj_w, found = self._pick_plausible_wrong_position(RM_ij=RM_ij, xc=x_idx, yc=y_idx)
                                if found == True:
                                    piece_j_on_canvas_plausible1 = PieceOnCanvas(piece=piece_j, grid=self.grid, x=xj_w, y=yj_w, theta=thetaj_w, enabled_features=self.features_status)
                                    wrong_alignment1, wrong_alignment1_mask = piece_i_on_canvas.blend_with(piece_j_on_canvas_plausible1, return_mask=True)
                                    if self.oracle_params['pairwise_alignments_dataset']['crop_images'] == True:
                                        wrong_alignment1 = crop_to_content(wrong_alignment1, padding=self.oracle_params['pairwise_alignments_dataset']['padding'])
                                        wrong_alignment1_mask = crop_to_content(wrong_alignment1_mask, padding=self.oracle_params['pairwise_alignments_dataset']['padding'])
                                    plt.imsave(os.path.join(self.oracle_params['pairwise_alignments_dataset']['wrong_alignment_folder'], plausible_alignment_name), np.clip(wrong_alignment1, 0, 1))
                                    if self.oracle_params['pairwise_alignments_dataset']['save_masks'] == True:
                                        cv2.imwrite(os.path.join(self.oracle_params['pairwise_alignments_dataset']['wrong_alignment_masks_folder'], plausible_alignment_name), np.clip(wrong_alignment1_mask, 0, 2))
                                else: 
                                    print("could not find plausible wrong alingment, skip this one!")
                else:            
                    # we consider these two as "not neighbours"
                    if verbose > 2:
                        print("we have values but they are not considered neighbours, we do not write")
            
        # oracle visualization! 
        # will be removed at some point (I hope)
        debug_and_show = False
        if debug_and_show == True:
            import matplotlib.pyplot as plt 
            # for x_idx, y_idx, theta_idx, motif_idx in zip(ids_to_score[0], ids_to_score[1], ids_to_score[2], ids_to_score[3]):
            
            piece_i_on_canvas = PieceOnCanvas(piece=piece_i, grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=0, enabled_features=self.features_status)
            
            thetaj = self.grid.theta_values[0]
            # print(xj, yj)
            piece_j_on_canvas = PieceOnCanvas(piece=piece_j, grid=self.grid, x=xj, y=yj, theta=thetaj, enabled_features=self.features_status)
            plt.imshow(piece_i_on_canvas.image  + piece_j_on_canvas.image)
            plt.title(f"GT_rel: {gt_rel_pos}, pieces_size: {self.puzzle.img_piece_size}")
            plt.show()
            breakpoint()
        # touching_region = self._compute_touching_region(piece_i_on_canvas, piece_j_on_canvas, dil_kernel)
        # CM_ij[x_idx, y_idx, theta_idx] = 1
        # import matplotlib.pyplot as plt 
        # plt.subplot(221); plt.imshow(piece_i.data.image); plt.title("piece i")
        # plt.subplot(222); plt.imshow(piece_j.data.image); plt.title("piece j")
        # plt.subplot(223); plt.imshow(RM_ij); plt.title("RM")
        # plt.subplot(224); plt.imshow(CM_ij); plt.title("CM")
        # plt.show()
        # breakpoint()
        return CM_ij

    def _compute_pairwise_oracle_CM(self, piece_i: PuzzlePiece, piece_j: PuzzlePiece, RM_ij: np.ndarray, gt_rel_pos: np.ndarray, verbose: int = 0):
        """
        For each pair of pieces, it places them on the canvas in the position `accepted` by RM_ij 
        and calls the scoring function to fill the pairwise compatibility matrix CM_ij 
        """
        kernel_size = 10*2+1
        kernel = np.ones((kernel_size, kernel_size))
        
        CM_ij = np.zeros((RM_ij.shape[0], RM_ij.shape[1], RM_ij.shape[2]))
        #CM_ij = np.zeros(tuple(self.RM_size[:3]))
        xj, yj = (np.asarray([self.grid.canvas_center, self.grid.canvas_center]) + np.asarray([gt_rel_pos[0], gt_rel_pos[1]])).tolist()
        x_idx = np.round(self.grid.xy_num_points // 2 + gt_rel_pos[0] / self.grid.xy_step).astype(int)
        y_idx = np.round(self.grid.xy_num_points // 2 + gt_rel_pos[1] / self.grid.xy_step).astype(int)
        x_i_idx = np.round(self.grid.xy_num_points // 2)
        y_i_idx = np.round(self.grid.xy_num_points // 2)
        
        import time 

        if 1 > 0: #np.max(abs(gt_rel_pos)) < (self.grid.p_hs): 
            # y_idx = np.round(yj / self.grid.xy_step).astype(int) 
            theta_idx = 0
            thetaj = self.grid.theta_values[theta_idx]
            if verbose > 2:
                print(f"CM[{x_idx}, {y_idx}, 0] = 1")
            
            y_c0 = np.ceil(yj-self.grid.p_hs).astype(int)
            y_c1 = np.ceil(yj+self.grid.p_hs+1).astype(int)
            x_c0 = np.ceil(xj-self.grid.p_hs).astype(int)
            x_c1 = np.ceil(xj+self.grid.p_hs+1).astype(int)
            if x_c0 < 0 or y_c0 < 0 or y_c1 > self.grid.canvas_size or x_c1 > self.grid.canvas_size:
                print("Out of the canvas! Error, skipping for now to see successive results.")
            else:
                piece_i_on_canvas = PieceOnCanvas(piece=piece_i, grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=0, enabled_features=self.features_status)
                piece_j_on_canvas = PieceOnCanvas(piece=piece_j, grid=self.grid, x=xj, y=yj, theta=thetaj, enabled_features=self.features_status)
          
                if np.sum(cv2.dilate(piece_i_on_canvas.mask, kernel) * cv2.dilate(piece_j_on_canvas.mask, kernel) > 0): 
                    CM_ij[y_idx, x_idx, theta_idx] = 1

                    if self.oracle_params['create_pairwise_alignments_dataset'] == True:
                        import matplotlib.pyplot as plt 
                        from utils.visualization_utils import crop_to_content
                        
                        # This is the PAIRWISE GROUND TRUTH
                        
                        alignment_basename = f'{self.puzzle.name}_vis_{piece_i.name}_{piece_j.name}_{xj}_{yj}_{0}'                        
                        if self.oracle_params['pairwise_alignments_dataset']['save_true_alignment'] == True:
                            time_gt = time.time()
                            true_alignment_name = f'{alignment_basename}_gt.png'
                            correctly_aligned_image, correctly_aligned_mask = piece_i_on_canvas.blend_with(piece_j_on_canvas, mask_type=self.oracle_params['pairwise_alignments_dataset']['mask_type'], return_mask=True)
                            if self.oracle_params['pairwise_alignments_dataset']['crop_images'] == True:
                                # plt.subplot(221); plt.imshow(correctly_aligned_image)
                                # plt.subplot(222); plt.imshow(crop_to_content(correctly_aligned_image, padding=self.oracle_params['pairwise_alignments_dataset']['padding']))
                                # plt.subplot(223); plt.imshow(correctly_aligned_mask)
                                # plt.subplot(224); plt.imshow(crop_to_content(correctly_aligned_mask, padding=self.oracle_params['pairwise_alignments_dataset']['padding']))
                                # plt.show()
                                # breakpoint()
                                correctly_aligned_image = crop_to_content(correctly_aligned_image, padding=self.oracle_params['pairwise_alignments_dataset']['padding'])
                                correctly_aligned_mask = crop_to_content(correctly_aligned_mask, padding=self.oracle_params['pairwise_alignments_dataset']['padding'])
                            plt.imsave(os.path.join(self.oracle_params['pairwise_alignments_dataset']['correct_alignment_folder'], true_alignment_name), np.clip(correctly_aligned_image, 0, 1))
                            if self.oracle_params['pairwise_alignments_dataset']['save_masks'] == True:
                                cv2.imwrite(os.path.join(self.oracle_params['pairwise_alignments_dataset']['correct_alignment_masks_folder'], true_alignment_name), np.clip(correctly_aligned_mask, 0, 2).astype(np.uint8))
                            print(f"Took {time.time()-time_gt} to create and save pairwise GT")

                        if self.oracle_params['pairwise_alignments_dataset']['save_grid_alignment'] == True:
                            time_ggt = time.time()
                            # This is the PAIRWISE "BEST" given the grid step that we have
                            grid_alignment_name = f'{alignment_basename}_grid.png'
                            xj_grid, yj_grid = self.grid.xy_values[x_idx, y_idx]
                            thetaj_grid = self.grid.theta_values[theta_idx]
                            piece_j_on_canvas_on_grid = PieceOnCanvas(piece=piece_j, grid=self.grid, x=xj_grid, y=yj_grid, theta=thetaj_grid, enabled_features=self.features_status)
                            grid_aligned_image, grid_aligned_mask = piece_i_on_canvas.blend_with(piece_j_on_canvas_on_grid, mask_type=self.oracle_params['pairwise_alignments_dataset']['mask_type'], return_mask=self.oracle_params['pairwise_alignments_dataset']['save_masks'])
                            if self.oracle_params['pairwise_alignments_dataset']['crop_images'] == True:
                                grid_aligned_image = crop_to_content(grid_aligned_image, padding=self.oracle_params['pairwise_alignments_dataset']['padding'])
                                grid_aligned_mask = crop_to_content(grid_aligned_mask, padding=self.oracle_params['pairwise_alignments_dataset']['padding'])
                            plt.imsave(os.path.join(self.oracle_params['pairwise_alignments_dataset']['correct_alignment_folder'], grid_alignment_name), np.clip(grid_aligned_image, 0, 1))
                            if self.oracle_params['pairwise_alignments_dataset']['save_masks'] == True:
                                cv2.imwrite(os.path.join(self.oracle_params['pairwise_alignments_dataset']['correct_alignment_masks_folder'], grid_alignment_name), np.clip(grid_aligned_mask, 0, 2).astype(np.uint8))
                            print(f"Took {time.time()-time_ggt} to create and save pairwise grid - GT")

                        # These are randomly chosen "plausible" (but not correct!) alignment of the two pieces
                        if self.oracle_params['pairwise_alignments_dataset']['save_wrong_alignments'] == True:
                            time_pwa = time.time()
                            for wk in range(self.oracle_params['pairwise_alignments_dataset']['wrong_alignments_num']):
                                plausible_alignment_name = f'{alignment_basename}_wrong_{wk}.png'
                                xj_w, yj_w, thetaj_w, found = self._pick_plausible_wrong_position(RM_ij=RM_ij, xc=x_idx, yc=y_idx)
                                if found == True:
                                    piece_j_on_canvas_plausible1 = PieceOnCanvas(piece=piece_j, grid=self.grid, x=xj_w, y=yj_w, theta=thetaj_w, enabled_features=self.features_status)
                                    wrong_alignment1, wrong_alignment1_mask = piece_i_on_canvas.blend_with(piece_j_on_canvas_plausible1, mask_type=self.oracle_params['pairwise_alignments_dataset']['mask_type'], return_mask=self.oracle_params['pairwise_alignments_dataset']['save_masks'])
                                    if self.oracle_params['pairwise_alignments_dataset']['crop_images'] == True:
                                        wrong_alignment1 = crop_to_content(wrong_alignment1, padding=self.oracle_params['pairwise_alignments_dataset']['padding'])
                                        wrong_alignment1_mask = crop_to_content(wrong_alignment1_mask, padding=self.oracle_params['pairwise_alignments_dataset']['padding'])
                                    plt.imsave(os.path.join(self.oracle_params['pairwise_alignments_dataset']['wrong_alignment_folder'], plausible_alignment_name), np.clip(wrong_alignment1, 0, 1))
                                    if self.oracle_params['pairwise_alignments_dataset']['save_masks'] == True:
                                        cv2.imwrite(os.path.join(self.oracle_params['pairwise_alignments_dataset']['wrong_alignment_masks_folder'], plausible_alignment_name), np.clip(wrong_alignment1_mask, 0, 2).astype(np.uint8))
                                else: 
                                    print("could not find plausible wrong alingment, skip this one!")
                                print(f"Took {time.time()-time_pwa} to create and save plausible wrong alignments" )
                else:            
                    # we consider these two as "not neighbours"
                    if verbose > 2:
                        print("we have values but they are not considered neighbours, we do not write")
            
        # oracle visualization! 
        # will be removed at some point (I hope)
        debug_and_show = False
        if debug_and_show == True:
            import matplotlib.pyplot as plt 
            # for x_idx, y_idx, theta_idx, motif_idx in zip(ids_to_score[0], ids_to_score[1], ids_to_score[2], ids_to_score[3]):
            
            piece_i_on_canvas = PieceOnCanvas(piece=piece_i, grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=0, enabled_features=self.features_status)
            
            thetaj = self.grid.theta_values[0]
            # print(xj, yj)
            print(piece_i.name, piece_j.name)
            piece_j_on_canvas = PieceOnCanvas(piece=piece_j, grid=self.grid, x=xj, y=yj, theta=thetaj, enabled_features=self.features_status)
            plt.subplot(121)
            plt.imshow(piece_i_on_canvas.image  + piece_j_on_canvas.image)
            plt.title(f"GT_rel: {gt_rel_pos}, pieces_size: {self.puzzle.img_piece_size}")
            plt.subplot(122)
            plt.imshow(CM_ij[:,:,0])
            plt.show()
            breakpoint()
        # touching_region = self._compute_touching_region(piece_i_on_canvas, piece_j_on_canvas, dil_kernel)
        # CM_ij[x_idx, y_idx, theta_idx] = 1
        # import matplotlib.pyplot as plt 
        # plt.subplot(221); plt.imshow(piece_i.data.image); plt.title("piece i")
        # plt.subplot(222); plt.imshow(piece_j.data.image); plt.title("piece j")
        # plt.subplot(223); plt.imshow(RM_ij); plt.title("RM")
        # plt.subplot(224); plt.imshow(CM_ij); plt.title("CM")
        # plt.show()
        # breakpoint()
        return CM_ij

    ###############################################################
    #                                                             #
    #  ███╗   ███╗ ██████╗ ████████╗██╗██╗   ██╗███████╗███████╗  #
    #  ████╗ ████║██╔═══██╗╚══██╔══╝██║██║   ██║██╔════╝██╔════╝  #
    #  ██╔████╔██║██║   ██║   ██║   ██║██║   ██║█████╗  ███████╗  #
    #  ██║╚██╔╝██║██║   ██║   ██║   ██║╚██╗ ██╔╝██╔══╝  ╚════██║  #
    #  ██║ ╚═╝ ██║╚██████╔╝   ██║   ██║ ╚████╔╝ ███████╗███████║  #
    #  ╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═══╝  ╚══════╝╚══════╝  #
    #                                                             #
    ###############################################################
    def _compute_motives_based_CM(self, verbose: int = 0):
        """Loops over pairs of pieces - not symmetric yet"""
        CM_motives = np.zeros(self.CM_size)
        if verbose > 1:
            print()
        for i in range(self.puzzle.num_of_pieces):
            for j in range(self.puzzle.num_of_pieces):
                if i != j:
                    if verbose > 1:
                        print(f'computing motives-based CM[:, :, :, {i:02d}, {j:02d}]', end='\r')
                    RM_ij = self.RM_dict['motives'][:, :, :, j, i]    
                    if np.sum(RM_ij > 0) > 0:
                        CM_motives[:, :, :, j, i] = self._compute_pairwise_motives_based_CM(self.puzzle.pieces[i], self.puzzle.pieces[j], RM_ij)
        if verbose > 1:
            print()
        return CM_motives

    def _compute_pairwise_motives_based_CM(self, piece_i: PuzzlePiece, piece_j: PuzzlePiece, RM_ij: np.ndarray):
        """
        For each pair of pieces, it places them on the canvas in the position `accepted` by RM_ij 
        and calls the scoring function to fill the pairwise compatibility matrix CM_ij 
        """
        CM_ij = np.zeros((RM_ij.shape[0], RM_ij.shape[1], RM_ij.shape[2]))
        ids_to_score = np.where(RM_ij > 0)

        for x_idx, y_idx, theta_idx, motif_idx in zip(ids_to_score[0], ids_to_score[1], ids_to_score[2], ids_to_score[3]):
            piece_i_on_canvas = PieceOnCanvas(piece=piece_i, grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=0, enabled_features=self.features_status)
            xj, yj = self.grid.xy_values[x_idx, y_idx]
            thetaj = self.grid.theta_values[theta_idx]
            piece_j_on_canvas = PieceOnCanvas(piece=piece_j, grid=self.grid, x=xj, y=yj, theta=thetaj, enabled_features=self.features_status)

            motives_based_score = self._compute_motives_score(piece_i_on_canvas, piece_j_on_canvas)
            # touching_region = self._compute_touching_region(piece_i_on_canvas, piece_j_on_canvas, dil_kernel)
            CM_ij[x_idx, y_idx, theta_idx] = motives_based_score

        return CM_ij

    def _compute_motives_score(self, piece_i: PieceOnCanvas, piece_j: PieceOnCanvas):
        """
        It receives two pieces already aligned, and it needs to give a score of how `good` this alignment is
        There are multiple scoring functions, see the commented code below
        """
        scoring_func = self.params['compatibility']['features']['motives']['scoring_func']
        score = -1
        if scoring_func == 'simple':
            # simple is the number of "touching" pixels, referring to the pixel shared by the two (slightly dilated) motives
            # (we already avoided overlap at this point)
            dilation_size = self.params['compatibility']['features']['motives']['simple_params']['dilation']
            dil_kernel = np.ones((dilation_size, dilation_size))
            num_pixels_kernel = np.sum(dil_kernel)
            # for each motif class
            simple_motives_score = 0
            for m in range(piece_i.motives_cube.shape[2]):
                touching_region_m = (cv2.dilate(piece_i.motives_cube[:,:,m].astype(np.uint8), dil_kernel) + cv2.dilate(piece_j.motives_cube[:,:,m].astype(np.uint8), dil_kernel)) > 0
                num_pixels_touching_region =  np.sum(touching_region_m > 0)
                if num_pixels_touching_region > (num_pixels_kernel / 4):
                    simple_motives_score += 0.2
            simple_motives_score = np.clip(simple_motives_score, -1, 1)
            if simple_motives_score > 0:
                score = simple_motives_score

            # touching_region = self._compute_touching_region(piece_i_on_canvas, piece_j_on_canvas, dil_kernel)
            # 
            # if num_pixels_touching_region < 1:
            #     score = -1
            # elif num_pixels_touching_region < (num_pixels_kernel / 4):
            #     score = 0
            # elif num_pixels_touching_region < (num_pixels_kernel / 2):
            #     score = 0.5
            # else: 
            #     score = 1
        elif scoring_func == 'detector':
            # in this case we use a detector on the aligned visualization of the two images 
            # and we assign a score if the detector detects a single object across the two pieces
            score = 0
        elif scoring_func == 'geometric':
            # in this case we use `geometric` motives (such as lines/curves) and we use their orientation
            # to compute a good continuation score
            score = 0
        return score


    ##############################################
    #                                            #
    #  ███████╗██╗  ██╗ █████╗ ██████╗ ███████╗  #
    #  ██╔════╝██║  ██║██╔══██╗██╔══██╗██╔════╝  #
    #  ███████╗███████║███████║██████╔╝█████╗    #
    #  ╚════██║██╔══██║██╔══██║██╔═══╝ ██╔══╝    #
    #  ███████║██║  ██║██║  ██║██║     ███████╗  #
    #  ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝  #
    #                                            #
    ##############################################
    def _compute_shape_based_CM(self, verbose: int = 0):
        """Loops over pairs of pieces - not symmetric yet"""
        
        #### TMP

        if self.params['compatibility']['features']['shape']['save_best_images']['enabled'] == True:
            print("\nWARNING: 'save_best_images' enabled!\n")
            print("It uses a combination of parameters across `shape` and `oracle`, and requires the ground truth. \
                It is used to create the pairwise alignment dataset.")
            print("It may be a bit complex to track the parameters, refer to `default.yaml` for more info or \
                just disable the dataset creation (setting `save_best_images/enabled` to No) if you are interested in the compatibility only.\n")
            with open(self.cfg.get_GT_path(), 'r') as gtjf:
                self.gt = json.load(gtjf)
        
        CM_shape = np.zeros(self.CM_size)
        if verbose > 1:
            print()
        for i in range(self.puzzle.num_of_pieces):
            for j in range(self.puzzle.num_of_pieces):
                if i != j:
                    # breakpoint()
                # if i > j:
                    if verbose > 1:
                        print(f'computing shape-based CM[:, :, :, {i:02d}, {j:02d}]', end='\r')
                    RM_ij = self.RM_dict['shape'][:, :, :, j, i]  
                    if self.params['compatibility']['features']['shape']['save_best_images']['enabled'] == True:  
                        gt_piece_i = self.gt['pieces'][f'{i}']
                        gt_piece_j = self.gt['pieces'][f'{j}']
                        gt_pos_i = np.asarray([gt_piece_i['x'], gt_piece_i['y']]) #/ self.puzzle_info['pieces_image_size'][0] * self.puzzle.img_piece_size[0] / 0.166
                        gt_pos_j = np.asarray([gt_piece_j['x'], gt_piece_j['y']]) #/self.puzzle_info['pieces_image_size'][0] * self.puzzle.img_piece_size[0] / 0.166
                        gt_rel_j_vs_i = np.round((gt_pos_j - gt_pos_i)).astype(int) #* 1.5 # / self.grid.xy_step).astype(int)
                        CM_shape[:, :, :, j, i] = self._compute_pairwise_shape_based_CM(self.puzzle.pieces[i], self.puzzle.pieces[j], RM_ij, gt_rel_j_vs_i)
                    else:
                        CM_shape[:, :, :, j, i] = self._compute_pairwise_shape_based_CM(self.puzzle.pieces[i], self.puzzle.pieces[j], RM_ij)
                    
        if verbose > 1:
            print()
        return CM_shape

    def _compute_pairwise_shape_based_CM(self, piece_i: PuzzlePiece, piece_j: PuzzlePiece, RM_ij: np.ndarray, gt_rel=None):
        """ 
        It computes SDF-based cost matrix between piece_i and piece_j
        """
        CM_ij = np.zeros_like(RM_ij)
        ids_to_score = np.where(RM_ij > 0)
        # TODO: move these to parameters?   improve?
        dilation_size = self.params['compatibility']['features']['shape']['SDF_dilation']
        dil_kernel = np.ones((dilation_size, dilation_size))
        sigma = self.grid.p_hs
        
        if self.params['compatibility']['features']['shape']['save_best_images']['enabled'] == True:
            os.makedirs(self.params['compatibility']['features']['shape']['save_best_images']['data_folder'], exist_ok=True)

        for x_idx, y_idx, theta_idx in zip(ids_to_score[0], ids_to_score[1], ids_to_score[2]):
            piece_i_on_canvas = PieceOnCanvas(piece=piece_i, grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=0, enabled_features=self.features_status)
            xj, yj = self.grid.xy_values[x_idx, y_idx]
            thetaj = self.grid.theta_values[theta_idx]
            piece_j_on_canvas = PieceOnCanvas(piece=piece_j, grid=self.grid, x=xj, y=yj, theta=thetaj, enabled_features=self.features_status)

            touching_region = self._compute_touching_region(piece_i_on_canvas, piece_j_on_canvas, dil_kernel)
            size_touching_region = np.sum(touching_region > 0)
            # print(f"We have {size_touching_region} pixels in the touching region")
            if size_touching_region < 2*self.grid.p_hs:
                shape_score = 0
            else:
                shape_score = self._compute_shape_score(piece_i_on_canvas, piece_j_on_canvas, touching_region, sigma=sigma)
            #
            CM_ij[x_idx, y_idx, theta_idx] = shape_score

            # this is only for saving the image (and mask) when creating the `hard negatives`
            if self.params['compatibility']['features']['shape']['save_best_images']['enabled'] == True:
                
                # create folders if they do not exist
                imgs_folder = os.path.join(self.params['compatibility']['features']['shape']['save_best_images']['data_folder'], 'images')
                os.makedirs(imgs_folder, exist_ok=True)
                if self.params['compatibility']['features']['oracle']['pairwise_alignments_dataset']['save_masks'] == True:
                    masks_folder = os.path.join(self.params['compatibility']['features']['shape']['save_best_images']['data_folder'], 'masks')
                    os.makedirs(masks_folder, exist_ok=True)
                
                # print(f'shape score[{x_idx}, {y_idx}, {theta_idx}]={shape_score}')
                if shape_score > self.params['compatibility']['features']['shape']['save_best_images']['score_threshold']:
                    cm_rel_j_vs_i = [x_idx - self.grid.xy_values.shape[0]//2, y_idx - self.grid.xy_values.shape[1]//2]
                    cm_rel_shift = np.asarray(cm_rel_j_vs_i) * self.grid.xy_step
                    dist_from_correct = np.linalg.norm(cm_rel_shift - gt_rel)
                    if dist_from_correct > self.params['compatibility']['features']['shape']['save_best_images']['dist2GT_threshold']:                    
                        # best_pos_idx = np.argmax(CM_ij)
                        # best_pos_xy_idx = [best_pos_idx % CM_ij.shape[0], best_pos_idx // CM_ij.shape[0]]
                        # best_pos_xy = self.grid.xy_values[best_pos_xy_idx[0], best_pos_xy_idx[1]]
                        # piece_j_on_canvas = PieceOnCanvas(piece=piece_j, grid=self.grid, x=best_pos_xy[0], y=best_pos_xy[1], theta=thetaj, enabled_features=self.features_status)
                        hard_negative_img, hard_negative_mask = piece_i_on_canvas.blend_with(piece_j_on_canvas, mask_type=self.params['compatibility']['features']['oracle']['pairwise_alignments_dataset']['mask_type'], return_mask=self.params['compatibility']['features']['oracle']['pairwise_alignments_dataset']['save_masks'])
                        alignment_name = f"{self.puzzle.name}_{piece_i.name}_vs_{piece_j.name}_score{int(shape_score*100):d}.png"
                        img_path = os.path.join(imgs_folder, alignment_name)
                        if self.params['compatibility']['features']['oracle']['pairwise_alignments_dataset']['crop_images'] == True:
                            hard_negative_img = crop_to_content(hard_negative_img, padding=self.params['compatibility']['features']['oracle']['pairwise_alignments_dataset']['padding'])
                            hard_negative_mask = crop_to_content(hard_negative_mask, padding=self.params['compatibility']['features']['oracle']['pairwise_alignments_dataset']['padding'])     
                        plt.imsave(img_path, np.clip(hard_negative_img, 0, 1))
                        if self.params['compatibility']['features']['oracle']['pairwise_alignments_dataset']['save_masks'] == True:
                            mask_path = os.path.join(masks_folder, alignment_name)
                            cv2.imwrite(mask_path, hard_negative_mask.astype(np.uint8))
                        # breakpoint()
                        
                        # plt.imshow(img_to_discriminate_mpl)
                        # plt.imshow(touching_region, cmap='gray', alpha=0.35)
                        #print(f"best pos: {best_pos_xy}")

        return CM_ij

    def _compute_touching_region(self, piece_i_on_canvas: PieceOnCanvas, piece_j_on_canvas: PieceOnCanvas, dil_kernel: np.ndarray):
        """ Computes the touching region between the binary masks of two pieces on the virtual canvas """
        dilated_pi_mask = cv2.dilate(piece_i_on_canvas.mask, dil_kernel)
        dilated_pj_mask = cv2.dilate(piece_j_on_canvas.mask, dil_kernel)
        inters_dilated_pi_mask_pj = ((dilated_pi_mask + piece_j_on_canvas.mask) > 1).astype(np.uint8)      
        inters_dilated_pj_mask_pi = ((dilated_pj_mask + piece_i_on_canvas.mask) > 1).astype(np.uint8)      
        touching_region = ((inters_dilated_pi_mask_pj + inters_dilated_pj_mask_pi) > 0).astype(np.uint8)
        touching_region = cv2.morphologyEx(touching_region, cv2.MORPH_CLOSE, dil_kernel)
        return touching_region
        
    def _compute_shape_score(self, piece_i: PieceOnCanvas, piece_j: PieceOnCanvas, mregion_mask: np.ndarray, sigma:float = 1.0):
        # get ellipsoidal region
        normalization_factor = np.sum(mregion_mask > 0)
        # sdf sum 
        sdf_sum = np.square(piece_i.sdf + piece_j.sdf)
        dissim_score = np.sum(sdf_sum * mregion_mask.astype(float) / normalization_factor)
        comp_score = np.exp(-(dissim_score / sigma))
        # print(f"d: {dissim_score:.03f}, c: {comp_score:.03f}")
        return comp_score


    
    ########################################################################################################
    #                                                                                                      #
    #   ██████╗  █████╗ ██╗██████╗ ██╗    ██╗██╗███████╗███████╗                                           #
    #   ██╔══██╗██╔══██╗██║██╔══██╗██║    ██║██║██╔════╝██╔════╝                                           #
    #   ██████╔╝███████║██║██████╔╝██║ █╗ ██║██║███████╗█████╗                                             #
    #   ██╔═══╝ ██╔══██║██║██╔══██╗██║███╗██║██║╚════██║██╔══╝                                             #
    #   ██║     ██║  ██║██║██║  ██║╚███╔███╔╝██║███████║███████╗                                           #
    #   ╚═╝     ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝╚══════╝╚══════╝                                           #
    #                                                                                                      #
    #    █████╗ ██╗     ██╗ ██████╗ ███╗   ██╗███╗   ███╗███████╗███╗   ██╗████████╗                       #
    #   ██╔══██╗██║     ██║██╔════╝ ████╗  ██║████╗ ████║██╔════╝████╗  ██║╚══██╔══╝                       #
    #   ███████║██║     ██║██║  ███╗██╔██╗ ██║██╔████╔██║█████╗  ██╔██╗ ██║   ██║                          #
    #   ██╔══██║██║     ██║██║   ██║██║╚██╗██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║                          #
    #   ██║  ██║███████╗██║╚██████╔╝██║ ╚████║██║ ╚═╝ ██║███████╗██║ ╚████║   ██║                          #
    #   ╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝                          #
    #                                                                                                      #
    #   ██████╗ ██╗███████╗ ██████╗██████╗ ██╗███╗   ███╗██╗███╗   ██╗ █████╗ ████████╗ ██████╗ ██████╗    #
    #   ██╔══██╗██║██╔════╝██╔════╝██╔══██╗██║████╗ ████║██║████╗  ██║██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗   #
    #   ██║  ██║██║███████╗██║     ██████╔╝██║██╔████╔██║██║██╔██╗ ██║███████║   ██║   ██║   ██║██████╔╝   #
    #   ██║  ██║██║╚════██║██║     ██╔══██╗██║██║╚██╔╝██║██║██║╚██╗██║██╔══██║   ██║   ██║   ██║██╔══██╗   #
    #   ██████╔╝██║███████║╚██████╗██║  ██║██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║   ██║   ╚██████╔╝██║  ██║   #
    #   ╚═════╝ ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝   #
    #                                                                                                      #
    ########################################################################################################

    def _compute_pad_CM_gemetric(self, verbose: int = 0):
        """Loops over pairs of pieces - not symmetric yet"""
        CM_pad = np.zeros(self.CM_size)
        if verbose > 1:
            print()
        # prepare model
        PAD_params = self.params['compatibility']['features']['pairwise_alignment_discriminator']

        model = ViTForImageClassification.from_pretrained(PAD_params['trained_model_folder'])
        processor = AutoImageProcessor.from_pretrained(
            os.path.join(PAD_params['trained_model_folder'], "config.json"),
            do_center_crop=PAD_params['do_center_crop'],
            crop_size={"height": PAD_params['crop_height'], "width": PAD_params['crop_width']},
            use_fast=PAD_params['use_fast'],
            trust_remote_code=True  # Required for local models
        )

        for i in range(self.puzzle.num_of_pieces):
            for j in range(self.puzzle.num_of_pieces):
                if i != j:
                    if verbose > 1:
                        print(f'computing PAD CM[:, :, :, {i:02d}, {j:02d}]', end='\r')

                    ## TODO - check !!!!!
                    RM_ij = self.RM_dict['geometry'][:, :, :, j, i]

                    if np.sum(RM_ij > 0) > 0:
                        if PAD_params['use_batch'] == True:
                            CM_pad[:, :, :, j, i] = self._batch_compute_pairwise_discriminator_CM(self.puzzle.pieces[i],
                                                                                                  self.puzzle.pieces[j],
                                                                                                  RM_ij, model=model,
                                                                                                  processor=processor,
                                                                                                  PAD_params=PAD_params)
                        else:
                            CM_pad[:, :, :, j, i] = self._compute_pairwise_discriminator_CM(self.puzzle.pieces[i],
                                                                                            self.puzzle.pieces[j],
                                                                                            RM_ij, model=model,
                                                                                            processor=processor,
                                                                                            PAD_params=PAD_params)

        if verbose > 1:
            print()

        return CM_pad



    def _compute_pad_CM(self, verbose: int = 0):
        """Loops over pairs of pieces - not symmetric yet"""
        CM_pad = np.zeros(self.CM_size)
        if verbose > 1:
            print()
        # prepare model 
        PAD_params = self.params['compatibility']['features']['pairwise_alignment_discriminator']

        model = ViTForImageClassification.from_pretrained(PAD_params['trained_model_folder'])
        processor = AutoImageProcessor.from_pretrained(
            os.path.join(PAD_params['trained_model_folder'], "config.json"),
            do_center_crop=PAD_params['do_center_crop'], 
            crop_size={"height": PAD_params['crop_height'], "width": PAD_params['crop_width']},
            use_fast=PAD_params['use_fast'],
            trust_remote_code=True  # Required for local models
        )

        for i in range(self.puzzle.num_of_pieces):
            for j in range(self.puzzle.num_of_pieces):
                if i != j:
                    if verbose > 1:
                        print(f'computing PAD CM[:, :, :, {i:02d}, {j:02d}]', end='\r')
                    RM_ij = self.RM_dict['pairwise_alignment_discriminator'][:, :, :, j, i]    
                    if np.sum(RM_ij > 0) > 0:
                        if PAD_params['use_batch'] == True:
                            CM_pad[:, :, :, j, i] = self._batch_compute_pairwise_discriminator_CM(self.puzzle.pieces[i], self.puzzle.pieces[j], RM_ij, model=model, processor=processor, PAD_params=PAD_params)
                        else:
                            CM_pad[:, :, :, j, i] = self._compute_pairwise_discriminator_CM(self.puzzle.pieces[i], self.puzzle.pieces[j], RM_ij, model=model, processor=processor, PAD_params=PAD_params)
                        
                            
                        # import time 
                        # time2 = time.time()
                        # CM_batch = self._batch_compute_pairwise_discriminator_CM(self.puzzle.pieces[i], self.puzzle.pieces[j], RM_ij, model=model, processor=processor, PAD_params=PAD_params)
                        # time_passed = time.time() - time2 
                        # print(f"CM computation using batch took {time_passed} seconds")
                        # time1 = time.time()
                        # CM_classic = self._compute_pairwise_discriminator_CM(self.puzzle.pieces[i], self.puzzle.pieces[j], RM_ij, model=model, processor=processor, PAD_params=PAD_params)
                        # time_passed = time.time() - time1 
                        # print(f"CM computation took {time_passed} seconds")
                        # breakpoint()
                        # print(f"CM_batch == CM: {CM_batch == CM}")
                        
                    # import matplotlib.pyplot as plt 
                    # plt.subplot(131); plt.imshow(self.puzzle.pieces[i].data.image)
                    # plt.subplot(132); plt.imshow(self.puzzle.pieces[j].data.image)
                    # plt.subplot(133); plt.imshow(CM_pad[:, :, :, j, i])
                    # plt.show()
                    # breakpoint()
        if verbose > 1:
            print()
        
        return CM_pad

    def _batch_compute_pairwise_discriminator_CM(self, piece_i: PuzzlePiece, piece_j: PuzzlePiece, RM_ij: np.ndarray, model: ViTForImageClassification, processor: AutoImageProcessor, PAD_params:dict):
        """ 
        It computes SDF-based cost matrix between piece_i and piece_j by loading all "candidate" images on a batch and 
        feeding it to the model. The *batch* version is useful for powerful GPUs
        """
        CM_ij = np.zeros_like(RM_ij)
        neg_region = RM_ij < 0
        ids_to_score = np.where(RM_ij > 0)
        import matplotlib.pyplot as plt 
        images = []
        for y_idx, x_idx, theta_idx in zip(ids_to_score[0], ids_to_score[1], ids_to_score[2]):
            piece_i_on_canvas = PieceOnCanvas(piece=piece_i, grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=0, enabled_features=self.features_status)
            yj, xj = self.grid.xy_values[y_idx, x_idx]
            thetaj = self.grid.theta_values[theta_idx]
            piece_j_on_canvas = PieceOnCanvas(piece=piece_j, grid=self.grid, x=xj, y=yj, theta=thetaj, enabled_features=self.features_status)
            
            img_to_discriminate_mpl = piece_i_on_canvas.blend_with(piece_j_on_canvas, return_mask=False)
            img_to_discriminate_PIL = Image.fromarray(np.uint8(img_to_discriminate_mpl * 255))
            img_to_discriminate_PIL = img_to_discriminate_PIL.convert('RGB')
            # add image to images
            images.append(img_to_discriminate_PIL)
        
        # process all images at once
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=False)

        # fill the matrix 
        pred_scores = outputs['logits']  # (it has 2 values, one for each `class`)
        for pred_score, y_idx, x_idx, theta_idx in zip(pred_scores, ids_to_score[0], ids_to_score[1], ids_to_score[2]):
            # breakpoint()
            pred_probs = torch.softmax(pred_score, dim=0)
            pred_class = torch.argmax(pred_probs).item() 
            if PAD_params['scoring_method'] == 'softmax':
                cmp_score = pred_probs[1].item()
            elif PAD_params['scoring_method'] == 'thresh':
                if pred_score[1].item() > PAD_params['thresholds']['positive']:
                    cmp_score = pred_score[0][1].item()
                elif pred_score[0].item() > PAD_params['thresholds']['negative']:
                    cmp_score = -1    
                else:
                    cmp_score = 0
            elif PAD_params['scoring_method'] == prob_diff:
                cmp_score = pred_probs[1].item() if pred_class == 1 else -1*pred_probs[0].item()
                cmp_score *= torch.abs(torch.diff(pred_score)).item()
            else:
                cmp_score = pred_probs[1].item() if pred_class == 1 else -1*pred_probs[0].item()
            
            CM_ij[y_idx, x_idx, theta_idx] = cmp_score
        
        # import matplotlib.pyplot as plt 
        # plt.subplot(231)
        # plt.imshow(piece_i.data.image)
        # plt.subplot(232)
        # plt.imshow(piece_j.data.image)

        # # best_pos_idx = np.argmax(CM_ij)
        # # best_pos_xy_idx = [best_pos_idx % CM_ij.shape[0], best_pos_idx // CM_ij.shape[0]]
        # # best_pos_xy = self.grid.xy_values[best_pos_xy_idx[0], best_pos_xy_idx[1]]
        # # print(f"best pos before: {best_pos_xy}")
                        
        # # cm_rel_j_vs_i = [best_pos_xy_idx[0] - self.grid.xy_values.shape[0]//2, best_pos_xy_idx[1] - self.grid.xy_values.shape[1]//2]
        # # print(f"estimated: {np.asarray(cm_rel_j_vs_i) * self.grid.xy_step}")
        # plt.title(f"score: {cmp_score:.02f}")
        # # plt.show()
        # plt.subplot(234)
        # plt.imshow(np.transpose(CM_ij[:,:,0]))
        # plt.title("CM raw")

        ###################################################
        ###################################################
        ###################################################
        # CM post processing
        if np.max(CM_ij) > 0:
            CM_ij /= np.max(CM_ij)
        else:
            values = np.sort(np.unique(CM_ij))[::-1]
            CM_ij[CM_ij<0] -= values[PAD_params['push_k_values']]
        # plt.subplot(235)
        # plt.title("CM after pushing")
        # plt.imshow(np.transpose(CM_ij[:,:,0]))
        ###################################################
        ###################################################
        ###################################################
        #         cut values
        CM_ij[CM_ij < PAD_params['cutoff_value']] = 0
        CM_ij -= neg_region

        
        # plt.subplot(236)
        # plt.imshow(CM_ij)
        # plt.title("CM final")
        # plt.imshow(np.transpose(CM_ij[:,:,0]))
        
        # # plt.figure()
        # # plt.subplot(131); plt.imshow(RM_ij)
        # # plt.subplot(132); plt.imshow(CM_ij)
        # plt.show()
        # # plt.subplot(133)
        # breakpoint()

        # best_pos_idx = np.argmax(CM_ij)
        # best_pos_xy_idx = [best_pos_idx % CM_ij.shape[0], best_pos_idx // CM_ij.shape[0]]
        # best_pos_xy = self.grid.xy_values[best_pos_xy_idx[0], best_pos_xy_idx[1]]
        # print(f"best pos after: {best_pos_xy}")
        # piece_j_on_canvas = PieceOnCanvas(piece=piece_j, grid=self.grid, x=best_pos_xy[1], y=best_pos_xy[0], theta=thetaj, enabled_features=self.features_status)
        # img_to_discriminate_mpl = piece_i_on_canvas.blend_with(piece_j_on_canvas, return_mask=False)
        # plt.imshow(img_to_discriminate_mpl)

        # plt.show()
        # breakpoint()
        return CM_ij

    def _compute_pairwise_discriminator_CM(self, piece_i: PuzzlePiece, piece_j: PuzzlePiece, RM_ij: np.ndarray, model: ViTForImageClassification, processor: AutoImageProcessor, PAD_params:dict):
        """ 
        It computes SDF-based cost matrix between piece_i and piece_j
        """
        CM_ij = np.zeros_like(RM_ij)
        ids_to_score = np.where(RM_ij > 0)
        # import matplotlib.pyplot as plt 
        for y_idx, x_idx, theta_idx in zip(ids_to_score[0], ids_to_score[1], ids_to_score[2]):
            piece_i_on_canvas = PieceOnCanvas(piece=piece_i, grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=0, enabled_features=self.features_status)
            yj, xj = self.grid.xy_values[y_idx, x_idx]
            thetaj = self.grid.theta_values[theta_idx]
            piece_j_on_canvas = PieceOnCanvas(piece=piece_j, grid=self.grid, x=xj, y=yj, theta=thetaj, enabled_features=self.features_status)
            
            img_to_discriminate_mpl = piece_i_on_canvas.blend_with(piece_j_on_canvas, return_mask=False)
            img_to_discriminate_PIL = Image.fromarray(np.uint8(img_to_discriminate_mpl * 255))
            img_to_discriminate_PIL = img_to_discriminate_PIL.convert('RGB')
            inputs = processor(images=img_to_discriminate_PIL, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=False)
            # most likely we can use some post-processing on the scores
            pred_score = outputs['logits']  # (it has 2 values, one for each `class`)
            pred_class = torch.argmax(pred_score).item() 
            if PAD_params['use_thresh'] == True:
                if pred_score[0][1].item() > PAD_params['thresholds']['positive']:
                    cmp_score = pred_score[0][1].item()
                elif pred_score[0][0].item() > PAD_params['thresholds']['negative']:
                    cmp_score = -1    
                else:
                    cmp_score = 0
            else:
                cmp_score = pred_score[0][1].item() if pred_class == 1 else -1*pred_score[0][0].item()
            CM_ij[y_idx, x_idx, theta_idx] = cmp_score
            # if pred_class == 0:
            #     CM_ij[x_idx, y_idx, theta_idx] *= -1
            
            # # if CM_ij[x_idx, y_idx, theta_idx] > -1:
            # # if pred_score[0][1] > -1:
            # proc_img = inputs['pixel_values'].squeeze(0).permute(1, 2, 0)
            # import matplotlib.pyplot as plt 
            # plt.imshow(proc_img)
            # plt.title(f"xj:{xj}, yj:{yj}, center:{self.grid.canvas_center}\nwrong: {pred_score[0][0].item():.2f},correct: {pred_score[0][1].item():.2f}")
            # plt.show()
            # breakpoint()

        # # import matplotlib.pyplot as plt 
        # plt.subplot(131)
        # plt.imshow(CM_ij)
        # plt.title("CM raw")
        # plt.show()
        # breakpoint()
        if np.max(CM_ij) > 0:
            CM_ij /= np.max(CM_ij)
        else:
            values = np.sort(np.unique(CM_ij))[::-1]
            CM_ij[CM_ij<0] -= values[PAD_params['push_k_values']]
        # plt.subplot(132)
        # plt.title("CM after pushing")
        # plt.imshow(CM_ij)

        # cut values
        CM_ij[CM_ij < PAD_params['cutoff_value']] = 0
        # plt.subplot(133)
        # # plt.imshow(CM_ij)
        # plt.title("CM final")
        # plt.imshow(CM_ij)
        neg_region = RM_ij < 0
        CM_ij -= neg_region
        # plt.show()
        # breakpoint()
        return CM_ij


##################################################################################
#                                                                                #
#   █████╗ ██╗     ██╗ ██████╗ ███╗   ██╗███╗   ███╗███████╗███╗   ██╗████████╗  #
#  ██╔══██╗██║     ██║██╔════╝ ████╗  ██║████╗ ████║██╔════╝████╗  ██║╚══██╔══╝  #
#  ███████║██║     ██║██║  ███╗██╔██╗ ██║██╔████╔██║█████╗  ██╔██╗ ██║   ██║     #
#  ██╔══██║██║     ██║██║   ██║██║╚██╗██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║     #
#  ██║  ██║███████╗██║╚██████╔╝██║ ╚████║██║ ╚═╝ ██║███████╗██║ ╚████║   ██║     #
#  ╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝     #
#                                                                                #
#              ███████╗ ██████╗ ██████╗ ██████╗ ███████╗██████╗                  #
#              ██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗                 #
#              ███████╗██║     ██║   ██║██████╔╝█████╗  ██████╔╝                 #
#              ╚════██║██║     ██║   ██║██╔══██╗██╔══╝  ██╔══██╗                 #
#              ███████║╚██████╗╚██████╔╝██║  ██║███████╗██║  ██║                 #
#              ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝                 #
#                                                                                #
##################################################################################

    def _compute_alignment_scorer_CM(self, verbose: int = 0):
        """Loops over pairs of pieces - not symmetric yet"""
        CM_as = np.zeros(self.CM_size)
        if verbose > 1:
            print()

        # Load model
        scorer = AlignmentScorer(
            checkpoint_path=self.params['compatibility']['features']['alignment_scorer']['checkpoint_path'],
            device='cuda',
            radius=self.params['compatibility']['features']['alignment_scorer']['radius'],
            threshold=self.params['compatibility']['features']['alignment_scorer']['threshold']
        )

        for i in range(self.puzzle.num_of_pieces):
            for j in range(self.puzzle.num_of_pieces):
                if i != j:
                    if verbose > 1:
                        print(f'computing Alignment Scorer CM[:, :, :, {i:02d}, {j:02d}]', end='\r')
                    RM_ij = self.RM_dict['alignment_scorer'][:, :, :, j, i]    
                    if np.sum(RM_ij > 0) > 0:
                            CM_as[:, :, :, j, i] = self._batch_compute_alignment_scorer_CM(self.puzzle.pieces[i], self.puzzle.pieces[j], RM_ij, model=scorer, params=self.params['compatibility']['features']['alignment_scorer'])
        if verbose > 1:
            print()
        
        return CM_as


    def _batch_compute_alignment_scorer_CM(self, piece_i: PuzzlePiece, piece_j: PuzzlePiece, RM_ij: np.ndarray, model: AlignmentScorer, params:dict):
        """ 
        It ranks the possible candidates alignment. It was trained to rank the scores of N possible alignments. 
        """
        CM_ij = np.zeros_like(RM_ij)
        neg_region = RM_ij < 0
        ids_to_score = np.where(RM_ij > 0)
        
        # load all the images on a list
        rgb_images = []
        mask_images = []
        # plt.figure()
        for y_idx, x_idx, theta_idx in zip(ids_to_score[0], ids_to_score[1], ids_to_score[2]):
            piece_i_on_canvas = PieceOnCanvas(piece=piece_i, grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=0, enabled_features=self.features_status)
            yj, xj = self.grid.xy_values[y_idx, x_idx]
            thetaj = self.grid.theta_values[theta_idx]
            piece_j_on_canvas = PieceOnCanvas(piece=piece_j, grid=self.grid, x=xj, y=yj, theta=thetaj, enabled_features=self.features_status)
                        
            img_to_discriminate_mpl, mask = piece_i_on_canvas.blend_with(piece_j_on_canvas, blend_mode='average', 
                                        mask_type='pieces_center_first', return_mask=True)
            img_to_discriminate_PIL = Image.fromarray(np.uint8(img_to_discriminate_mpl * 255))
            img_to_discriminate_PIL = img_to_discriminate_PIL.convert('RGB')

            mask_to_discriminate_PIL = Image.fromarray(np.uint8(mask * 255))
            mask_to_discriminate_PIL = mask_to_discriminate_PIL.convert('L')

            # add image to images
            rgb_images.append(img_to_discriminate_PIL)
            mask_images.append(mask_to_discriminate_PIL)
        
        # rank the images at once
        scores = model.score(rgb_images, mask_images)
        comps = torch.sigmoid(torch.from_numpy(scores))                 # these are the compatibilities (0 - 1)
        ranks = torch.argsort(torch.argsort(comps, descending=True))    # these are the ranking (order descending)
        topK = params.get('topK', 0)

        # fill the matrix 
        for comp, rank_order, y_idx, x_idx, theta_idx in zip(comps, ranks, ids_to_score[0], ids_to_score[1], ids_to_score[2]):
            if topK > 0:
                if rank_order < topK:
                    CM_ij[y_idx, x_idx, theta_idx] = comp
                else:
                    CM_ij[y_idx, x_idx, theta_idx] = 0          # cutting the values after the top K
            else:                   # if topK is 0 or -1, we use all values
                CM_ij[y_idx, x_idx, theta_idx] = comp
            
        CM_ij -= neg_region    
        
        # import matplotlib.pyplot as plt 
        # plt.subplot(231)
        # plt.imshow(piece_i.data.image)
        # plt.subplot(232)
        # plt.imshow(piece_j.data.image)

        # # best_pos_idx = np.argmax(CM_ij)
        # # best_pos_xy_idx = [best_pos_idx % CM_ij.shape[0], best_pos_idx // CM_ij.shape[0]]
        # # best_pos_xy = self.grid.xy_values[best_pos_xy_idx[0], best_pos_xy_idx[1]]
        # # print(f"best pos before: {best_pos_xy}")
                        
        # # cm_rel_j_vs_i = [best_pos_xy_idx[0] - self.grid.xy_values.shape[0]//2, best_pos_xy_idx[1] - self.grid.xy_values.shape[1]//2]
        # # print(f"estimated: {np.asarray(cm_rel_j_vs_i) * self.grid.xy_step}")
        # plt.title(f"score: {cmp_score:.02f}")
        # # plt.show()
        # plt.subplot(234)
        # plt.imshow(np.transpose(CM_ij[:,:,0]))
        # plt.title("CM raw")



        
        # plt.subplot(236)
        # plt.imshow(CM_ij)
        # plt.title("CM final")
        # plt.imshow(np.transpose(CM_ij[:,:,0]))
        
        # # plt.figure()
        # # plt.subplot(131); plt.imshow(RM_ij)
        # # plt.subplot(132); plt.imshow(CM_ij)
        # plt.show()
        # # plt.subplot(133)
        # breakpoint()

        # best_pos_idx = np.argmax(CM_ij)
        # best_pos_xy_idx = [best_pos_idx % CM_ij.shape[0], best_pos_idx // CM_ij.shape[0]]
        # best_pos_xy = self.grid.xy_values[best_pos_xy_idx[0], best_pos_xy_idx[1]]
        # print(f"best pos after: {best_pos_xy}")
        # piece_j_on_canvas = PieceOnCanvas(piece=piece_j, grid=self.grid, x=best_pos_xy[1], y=best_pos_xy[0], theta=thetaj, enabled_features=self.features_status)
        # img_to_discriminate_mpl = piece_i_on_canvas.blend_with(piece_j_on_canvas, return_mask=False)
        # plt.imshow(img_to_discriminate_mpl)

        # plt.show()
        # breakpoint()
        return CM_ij
