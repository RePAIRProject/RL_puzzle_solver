import numpy as np 
import os
# import features_utils as fts_uts
from utils.puzzle_utils import Puzzle 
from compatibility.grid import PuzzleGrid, PieceOnCanvas
from utils.parameters_utils import Configuration, CustomYAMLEncoder
from utils.visualization_utils import crop_to_content, save_pairwise_matrix_visualization_to_file
import cv2 
from PIL import Image
import matplotlib.pyplot as plt
import yaml
import torch

from compatibility.polex_refactored import (
    to_torch_bgra,
    extract_potential_alignments,
    score_alignment,
    warp_single_image,
    tensor_to_rgba,
    pad_to_same_size,
)


###########################################################
#                                                         #
#  ██████╗ ███████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗        #
#  ██╔══██╗██╔════╝██╔════╝ ██║██╔═══██╗████╗  ██║        #
#  ██████╔╝█████╗  ██║  ███╗██║██║   ██║██╔██╗ ██║        #
#  ██╔══██╗██╔══╝  ██║   ██║██║██║   ██║██║╚██╗██║        #
#  ██║  ██║███████╗╚██████╔╝██║╚██████╔╝██║ ╚████║        #
#  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝ ╚═════╝ ╚═╝  ╚═══╝        #
#                                                         #
#  ███╗   ███╗ █████╗ ████████╗██████╗ ██╗██╗  ██╗        #
#  ████╗ ████║██╔══██╗╚══██╔══╝██╔══██╗██║╚██╗██╔╝        #
#  ██╔████╔██║███████║   ██║   ██████╔╝██║ ╚███╔╝         #
#  ██║╚██╔╝██║██╔══██║   ██║   ██╔══██╗██║ ██╔██╗         #
#  ██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║██║██╔╝ ██╗        #
#  ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝        #
#                                                         #
#  ███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗██╗     ███████╗  #
#  ████╗ ████║██╔═══██╗██╔══██╗██║   ██║██║     ██╔════╝  #
#  ██╔████╔██║██║   ██║██║  ██║██║   ██║██║     █████╗    #
#  ██║╚██╔╝██║██║   ██║██║  ██║██║   ██║██║     ██╔══╝    #
#  ██║ ╚═╝ ██║╚██████╔╝██████╔╝╚██████╔╝███████╗███████╗  #
#  ╚═╝     ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝ ╚══════╝╚══════╝  #
#                                                         #
###########################################################
class RegionMatrixModule:
    """
    The class for all the operation (initialization, computation, loading, writing)
    related to the region matrix (also called partial payoff matrix in the paper)
    
    It contains a dictionary which contains the different region matrices 
    - shape based (self.RM['shape'])
    - motif based (self.RM['motif'])
    - and so on..
    """
    def __init__(self, puzzle: Puzzle, params: dict, cfg: Configuration):
        
        self.puzzle = puzzle
        # self.pieces = pcs_uts.load_pieces(puzzle_name=puzzle_name)
        # if use_feats == True:
        #     self.pieces = fts_uts.load_features(pieces=self.pieces, puzzle_name=puzzle_name)
        self.params = params
        # here we need to load stuff from the yaml file
        self.features_status = {}
        self.features = []
        self.check_features_status()      
        self.cfg = cfg #Configuration(puzzle.name) 
        # self.exp_folder = self.cfg.new_puzzle_single_run_random_folder_name()
        self.cfg.new_puzzle_single_run_random_folder_name()
        self.vis_params = params['compatibility']['save_visualization']
        self.save_vis = self.vis_params['save_RM']
        ## we could call here
        # self.prepare()

    def check_features_status(self):
        features = self.params['compatibility']['features']
        for feature in features:
            self.features.append(feature)
            self.features_status[feature] = features[feature]['enabled']                

    def print_features(self):
        """
        If I want to check which features are loaded (min/max helpful for seeing if it's empty still)
        """
        print(f"Working with {len(self.RM.keys())} features")
        for rmk in self.features:
            print(f"{rmk}: shape ({self.RM[rmk].shape}), max: {np.max(self.RM[rmk])}, min: {np.min(self.RM[rmk])}")

    def add_features(self, features: list):
        """
        Adding features? Not sure if we actually need this
        """
        for feat in features:
            if feat not in self.features:
                self.features.append(feat)
                self.RM[feat] = np.zeros(self.RM_size)

    def prepare(self):
        """
        All the parameters are set here
        It should be self-explanatory as it's just setting values (with predefined factors we hard-coded)
        It creates the shape-based RM as it is the baseline we decided to _always_ use
        The `features_status` dict keeps track of which features we are using
        """
        self.piece_size = self.params['preprocessing']['piece_size']
        self.p_hs = self.piece_size // 2
        self.canvas_buffer = 0
        self.grid = PuzzleGrid(self.params['compatibility']['grid'], self.piece_size)
        self.regions_dilation = self.params['compatibility']['regions']['borders_dilation']
        self.regions_erosion = self.params['compatibility']['regions']['borders_erosion']
        self.threshold_overlap_shapes = self.piece_size / 6 # it was /2 !
        self.threshold_overlap_lines = self.piece_size / 8
        self.threshold_overlap_motifs = self.piece_size / 5
        self.RM_size = (
            self.grid.xy_num_points,
            self.grid.xy_num_points,
            self.grid.theta_num_points,
            self.puzzle.num_of_pieces,
            self.puzzle.num_of_pieces,
        )
        self.RM_computed = False
        self.RM = {
            'shape': np.zeros(self.RM_size)
        }
        # Not true, different features may have different RM sizes! 
        # motives for example has one more channel
        # for feature in self.features:
        #     if self.features_status[feature] == True:
        #         self.RM[feature] = np.zeros(self.RM_size)
        

    def load_from_file(self, file_path: str):
        """
        Loading the RM matrix from file if it already exists
        """
        print("WIP: specify if more matrices are in the same file!")
        # assert os.path.exists(file_path), f"{file_path} non existing! please check"
        # if file_path.endswith('.mat'):
        #     import scipy
        #     RM_dict = scipy.io.loadmat(file_path)
        #     for feat in features:
        #         if feat in RM_dict.keys():
        #             self.RM[f'{feat}'] = RM_dict[f'R_{feat}']
        #             self.features.append(feat)
        #         else:
        #             print(f"could not find {feat}-based RM")
        #             # raise Exception(f"When reading {file_path} I expected as key R_{RM_type}, but was not found\nPlease pass RM_type such that R_`RM_type` is a key of the dictionary read.") 
        # else:
        #     print("WIP: probably not working")
        #     self.RM = np.load(file_path)
        #     self.features = []

    def save(self, verbose:int=0):
        """
        Save a .npy dictionary with the values of the RM computed and some contextual parameters (useful to use the data)
        """
        rm_path = self.cfg.get_RM_path() 
        context_params = {}
        context_params['input_params'] = self.params
        context_params['grid_params'] = {'xy_num_points': self.grid.xy_num_points, 'theta_num_points': self.grid.theta_num_points, 'xy_step':self.grid.xy_step, 'theta_step':self.grid.theta_step, 
            'canvas_size': self.grid.canvas_size, 'pairwise_comp_range': self.grid.pairwise_comp_range}
        context_params['features'] = self.features_status
        context_params['puzzle'] = {'puzzle_name': self.puzzle.name, 'num_pieces': self.puzzle.num_of_pieces, 'piece_size': self.piece_size}
        # values of the matrix
        self.RM['__context'] = context_params
        np.save(rm_path, self.RM)
        if self.save_vis == True:
            for feature in self.features:
                if self.features_status[feature] == True:
                    if verbose > 1:
                        print("-" * 50)
                        print(f"Saving {feature}-based RM")
                    if self.features_status[feature] == True:
                        save_pairwise_matrix_visualization_to_file(self.RM[feature], pieces=self.puzzle.pieces, rot_step=self.params['compatibility']['grid']['theta_step'], based_on=feature,
                                                                        output_folder=os.path.join(self.cfg.get_current_experiments_folder(), 'RM_vis'), visualization_params=self.vis_params, 
                                                                        matrix_type='RM')

        # input parameters
        input_params_path = self.cfg.get_RM_input_parameters_path() 
        #os.path.join(self.cfg.current_experiment_folder, 'RM_input_params.yaml')
        with open(input_params_path, 'w') as f:
            yaml.dump(self.params, f, Dumper=CustomYAMLEncoder, default_flow_style=False)
        # output parameters
        context_params_path = self.cfg.get_RM_output_parameters_path() 
        # os.path.join(self.cfg.current_experiment_folder, 'RM_output_params.yaml')
        with open(context_params_path, 'w') as f:
            yaml.dump(context_params, f, Dumper=CustomYAMLEncoder, default_flow_style=False)

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
                self.RM[feature] = self._compute_feature_based_RM_wrapper(feature, verbose=verbose)
                if verbose > 1:
                    print(f"{feature}-based RM computed!")
            else:
                if verbose > 1:
                    print(f"{feature}-based RM skipped, as {feature} is disabled!")

    def _compute_feature_based_RM_wrapper(self, feature: str, verbose: int = 0):
        """
        Just a wrapper, will decide which method to call depending on the feature
        """
        if feature == 'shape':
            if verbose > 1:
                print("shape-based RM Computation")
            RM = self.compute_shape_based_RM(verbose=verbose)
        elif feature == 'lines':
            if verbose > 1:
                print("lines-based RM Computation")
            RM = self.compute_line_based_RM(verbose=verbose)
        elif feature == 'motives':
            if verbose > 1:
                print("motives-based RM Computation")
            RM = self.compute_motif_based_RM(verbose=verbose)
        elif feature == 'oracle':
            if verbose > 1:
                print("WARNING:\nfor the oracle compatibility, we still use shape-based RM Computation")
            # TODO: check if we already computed the shape!
            if 'shape' in self.RM.keys():
                RM = self.RM['shape']
            else:
                RM = self.compute_shape_based_RM(verbose=verbose)
        elif feature == 'PAD' or feature == 'pairwise_alignment_discriminator':
            if verbose > 1:
                print("WARNING:\nfor the PAD compatibility, we still use shape-based RM Computation")
            RM = self.compute_shape_based_RM(verbose=verbose)
        elif feature == 'geometry':
            if verbose > 1:
                print("geometry-based RM Computation")
            RM = self.compute_geometry_based_RM(verbose=verbose)
        elif feature == 'alignment_scorer' :
            if verbose > 1:
                print("WARNING:\nfor the AS compatibility, we still use shape-based RM Computation")
            RM = self.compute_shape_based_RM(verbose=verbose)
        else:
            raise Exception(f"{feature}-based RM not implemented yet!")

        return RM 

    ##########################################
    #####   TODO  NEW   ########################
    ##########################################

    def compute_geometry_based_RM(self, verbose: int = 0):
        """Loops over pairs of pieces - not symmetric yet"""
        self.RM_size = (self.grid.xy_num_points, self.grid.xy_num_points, self.grid.theta_num_points,
                        self.puzzle.num_of_pieces, self.puzzle.num_of_pieces)
        RM_geometric = np.zeros(self.RM_size)
        if verbose > 1:
            print()
        for i in range(self.puzzle.num_of_pieces):
            for j in range(self.puzzle.num_of_pieces):
                if i != j:
                    if verbose > 1:
                        print(f'computing shape-based RM[:, :, :, {i:02d}, {j:02d}]', end='\r')
                    RM_geometric[:, :, :, j, i] = self.compute_pairwise_geometry_based_RM(i, j)
        if verbose > 1:
            print()
        return RM_geometric

    def compute_pairwise_geometry_based_RM(self, i: int, j: int, dilate: bool = True, erode: bool = True):
        """
        Geometry based RM with 1, 0 and -1 regions
        """

        # 1) Load BGRA images (with alpha) as numpy arrays

        ###piece_img = pieces[i].data.image
        tgt_np = self.puzzle.pieces[i].data.image
        src_np = self.puzzle.pieces[j].data.image
        #tgt_np = cv2.imread("target2.png", cv2.IMREAD_UNCHANGED)
        #src_np = cv2.imread("source.png", cv2.IMREAD_UNCHANGED)

        # 2) Convert to [1,4,H,W] CUDA tensors
        dev = torch.device("cuda")
        tgt_t = to_torch_bgra(tgt_np, pad_by=200, device=dev)
        src_t = to_torch_bgra(src_np, pad_by=200, device=dev)

        # 3) Extract all geometric candidates
        cands = extract_potential_alignments(
            tgt_t, src_t,
            gap=2.0,
            min_edge_length=20.0,
            min_length_ratio=0.8,
            epsilon_ratio=0.005,
            angle_threshold_deg=10.0,
            smoothing_kernel_size=3,
            pad_by=200
        )

        RM_ij = np.zeros(tuple(self.RM_size[:3]))
        t = self.RM_size[2]
        all_theta =  np.array([i * 360 / t for i in range(t)] )   # [0, 90, ....]

        t_center = (self.grid.xy_num_points-1)/2
        for candidate in cands:
            tx = candidate["translation_x"]
            ty = candidate["translation_y"]
            rotation = candidate["rotation"]%360
            print(f"tx = {tx}, ty = {ty}, t = {rotation}")

            x_grid = np.round(t_center+(tx / self.grid.xy_step)).astype(int)
            y_grid = np.round(t_center-(ty / self.grid.xy_step)).astype(int)
            t_grid = np.argmin(abs(rotation - all_theta))
            print(f"x = {x_grid}, y = {y_grid}, t = {t_grid}")

            #RM_ij[x_grid, y_grid, t_grid] = 1
            RM_ij[y_grid, x_grid, t_grid] = 1

        return RM_ij

    ##########################################
    #######  FINE TODO      #################
    ##########################################




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
    def compute_shape_based_RM(self, verbose: int = 0):
        """Loops over pairs of pieces - not symmetric yet"""
        RM_shape = np.zeros((self.RM_size[0], self.RM_size[1], self.RM_size[2], self.puzzle.num_of_pieces, self.puzzle.num_of_pieces))
        if verbose > 1:
            print()
        for i in range(self.puzzle.num_of_pieces):
            for j in range(self.puzzle.num_of_pieces):
                if i != j:
                    if verbose > 1:
                        print(f'computing shape-based RM[:, :, :, {i:02d}, {j:02d}]', end='\r')
                    RM_shape[:, :, :, j, i] = self.compute_pairwise_shape_based_RM(i, j)
        if verbose > 1:
            print()
        return RM_shape

    def compute_pairwise_shape_based_RM(self, i: int, j: int, dilate: bool = True, erode: bool = True):
        """
        Shape based RM with 1, 0 and -1 regions
        """
        RM_ij = np.zeros((self.RM_size[0], self.RM_size[1], self.RM_size[2]))
        piece_i_on_canvas = PieceOnCanvas(piece=self.puzzle.pieces[i], grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=0, enabled_features=self.features_status)
        # piece_i_on_canvas = pcs_uts.place_on_canvas(piece_i, (center_pos, center_pos), self.canvas_size, 0)
        for theta_idx in range(self.RM_size[2]):
            theta = theta_idx * self.grid.theta_step
            piece_j_on_canvas = PieceOnCanvas(piece=self.puzzle.pieces[j], grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=theta, enabled_features=self.features_status)
            # piece_j_on_canvas = pcs_uts.place_on_canvas(piece_j, (center_pos, center_pos), self.canvas_size, t * self.theta_step)
            # SHAPE case - BASIC
            overlap_shapes = cv2.filter2D(piece_i_on_canvas.mask, -1, piece_j_on_canvas.mask)
            thresholded_regions_map = (overlap_shapes > self.threshold_overlap_shapes).astype(np.int32)

            if dilate == True:
                border_dilation = int(self.regions_dilation * self.grid.xy_step)
            else:
                border_dilation = 1
            if erode == True:
                border_erosion = int(self.regions_erosion * self.grid.xy_step)
            else:
                border_erosion = 1

            around_borders_trm = self.get_borders_around(thresholded_regions_map.astype(np.uint8),
                                                border_dilation=border_dilation, border_erosion=border_erosion)
            thresholded_regions_map *= -1
            thresholded_regions_map += 2 * (around_borders_trm > 0)
            thresholded_regions_map = np.clip(thresholded_regions_map, -1, 1)

            # we convert the matrix to resize the image without losing the values
            thr_reg_map_shape_uint = (thresholded_regions_map + 1).astype(np.uint8)
            thr_reg_map_comp_range = thr_reg_map_shape_uint[self.p_hs + 1:-(self.p_hs + 1), self.p_hs + 1:-(self.p_hs + 1)]
            resized_shape = np.array(Image.fromarray(thr_reg_map_comp_range).resize((self.RM_size[0], self.RM_size[1]), Image.Resampling.NEAREST))
            RM_ij[:,:,theta_idx] = (resized_shape.astype(np.int32) - 1)

            # # write a nice visualization of the "maps" defined above
            # plt.subplot(231); plt.imshow(piece_i_on_canvas.image)
            # plt.subplot(232); plt.imshow(piece_j_on_canvas.image)
            # plt.subplot(233); plt.imshow(RM_ij[:,:,theta_idx])
            # plt.show()
            # breakpoint()

        return RM_ij

    
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
    def compute_motif_based_RM(self, verbose: int = 0):
        """Loops over pairs of pieces - not symmetric yet"""
        RM_motives = np.zeros((self.RM_size[0], self.RM_size[1], self.RM_size[2], self.puzzle.num_of_pieces, self.puzzle.num_of_pieces, self.puzzle.pieces[0].features.motives.num_of_classes))
        if verbose > 1:
            print()
        for i in range(self.puzzle.num_of_pieces):
            for j in range(self.puzzle.num_of_pieces):
                if i != j:
                    if verbose > 1:
                        print(f'computing motives-based RM[:, :, :, {i:02d}, {j:02d}]', end='\r')
                    RM_motives[:, :, :, j, i, :] = self.compute_pairwise_motif_based_RM(i, j)
        if verbose > 1:
            print()
        return RM_motives

    def compute_pairwise_motif_based_RM(self, i: int, j: int, skip_first_n_classes=2):

        num_of_motives = self.puzzle.pieces[i].features.motives.num_of_classes # the number of classes used in the segmentation
        RM_ij_motives = np.zeros((self.RM_size[0], self.RM_size[1], self.RM_size[2], num_of_motives))
        piece_i_on_canvas = PieceOnCanvas(piece=self.puzzle.pieces[i], grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=0, enabled_features=self.features_status)
        for theta_idx in range(self.RM_size[2]):
            theta = theta_idx * self.grid.theta_step
            piece_j_on_canvas = PieceOnCanvas(piece=self.puzzle.pieces[j], grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=theta, enabled_features=self.features_status)
            RM_motives_motives = np.zeros_like(piece_i_on_canvas.motives_cube)         # motives masks
            RM_poly_motives = np.zeros_like(piece_i_on_canvas.motives_cube)             # shapes of pieces masks
            # mask_mt = (piece_i_on_canvas['motif_mask'].shape[0], piece_i_on_canvas['motif_mask'].shape[1], num_of_motives))        
            # poly_mask_mt = np.zeros((piece_i_on_canvas['motif_mask'].shape[0], piece_i_on_canvas['motif_mask'].shape[1], num_of_motives))   
            # mask_i = piece_i_on_canvas.mask
            for motif_class in range(skip_first_n_classes, num_of_motives, 1):
                motif_i_mask = piece_i_on_canvas.motives_cube[:, :, motif_class]
                motif_j_mask = piece_j_on_canvas.motives_cube[:, :, motif_class]
                motif_i_mask = self.dilate(motif_i_mask.astype(np.uint8), width=np.floor(1 * self.grid.xy_step).astype(int))
                motif_j_mask = self.dilate(motif_j_mask.astype(np.uint8), width=np.floor(1 * self.grid.xy_step).astype(int))
                # if there are some values
                if np.sum(motif_i_mask) > 0 and np.sum(motif_j_mask) > 0:
                                           RM_motives_motives[:, :, motif_class] = cv2.filter2D(motif_i_mask, -1, motif_j_mask)
                #mask_i = dilate(mask_i.astype(np.uint8), width=np.floor(1 * ppars.xy_step).astype(int))
                if np.sum(motif_i_mask) > 0:
                    poly_j_vs_motif_i = cv2.filter2D(piece_j_on_canvas.mask, -1, motif_i_mask) 
                else:
                    poly_j_vs_motif_i = np.zeros_like(piece_j_on_canvas.mask)
                if np.sum(motif_j_mask) > 0:
                    poly_i_vs_motif_j = cv2.filter2D(piece_i_on_canvas.mask, -1, motif_j_mask) 
                else:
                    poly_i_vs_motif_j = np.zeros_like(piece_i_on_canvas.mask)
                
                # This combines the `negative` part of the motives-based RM
                # the negative part is where the motives of piece i touch the polygon j without motif (poly_j_vs_motif_i)
                # or where the motives of piece j touch the polygon i without motif (poly_i_vs_motif_j)
                # the bitwise_or means that if any of these two case is true, we set the true value
                RM_poly_motives[:, :, motif_class] = cv2.bitwise_or(poly_j_vs_motif_i, poly_i_vs_motif_j)

            binary_overlap_motifs = (RM_motives_motives > self.threshold_overlap_motifs).astype(np.int32)
            binary_overlap_motifs_no_pad = binary_overlap_motifs[self.p_hs + 1:-(self.p_hs + 1), self.p_hs + 1:-(self.p_hs + 1),:]
            resized_motives = cv2.resize(binary_overlap_motifs_no_pad, dsize=(self.grid.xy_num_points, self.grid.xy_num_points),
                        interpolation=cv2.INTER_NEAREST)

            binary_overlap_poly_motifs = (RM_poly_motives > self.threshold_overlap_motifs).astype(np.int32)
            binary_overlap_poly_motifs_no_pad = binary_overlap_poly_motifs[self.p_hs + 1:-(self.p_hs + 1),
                                            self.p_hs + 1:-(self.p_hs + 1), :]
            resized_poly_motives = cv2.resize(binary_overlap_poly_motifs_no_pad,
                        dsize=(self.grid.xy_num_points, self.grid.xy_num_points), interpolation=cv2.INTER_NEAREST)
                                        
            # combine the two temporary RMs
            # the matrix has positive values (1) where the two motives touch, 
            # and has negative values (-1) where one motif touches an empty part (not continued!)
            RM_ij_motives[:, :, theta_idx, :] = 1 * resized_motives - 1 * resized_poly_motives 
        
        return RM_ij_motives

    ###########################################
    #                                         #
    #  ██╗     ██╗███╗   ██╗███████╗███████╗  #
    #  ██║     ██║████╗  ██║██╔════╝██╔════╝  #
    #  ██║     ██║██╔██╗ ██║█████╗  ███████╗  #
    #  ██║     ██║██║╚██╗██║██╔══╝  ╚════██║  #
    #  ███████╗██║██║ ╚████║███████╗███████║  #
    #  ╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝  #
    #                                         #
    ###########################################
    def compute_line_based_RM(self, verbose: int = 0):
        """Loops over pairs of pieces - not symmetric yet"""
        RM_lines = np.zeros((self.RM_size[0], self.RM_size[1], self.RM_size[2], self.num_of_pieces, self.num_of_pieces))
        if verbose > 1:
            print()
        for i in range(self.puzzle.num_of_pieces):
            for j in range(self.puzzle.num_of_pieces):
                if i != j:
                    if verbose > 1:
                        print(f'computing line-based RM[:, :, :, {i:02d}, {j:02d}]', end='\r')
                    RM_lines[:, :, :, j, i] = self.compute_pairwise_line_based_RM(i, j)
        if verbose > 1:
            print()
        return RM_lines

    def compute_pairwise_line_based_RM(self, i: int, j: int):
        """
        Line based RM with "positive" 1 regions and "empty" 0 regions
        It should be used in combination with shape-based, as it does not have the -1 values!
        """
        RM_ij = np.zeros((self.RM_size[0], self.RM_size[1], self.RM_size[2]))
        piece_i_on_canvas = PieceOnCanvas(piece=self.puzzle.pieces[i], grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=0, enabled_features=self.features_status)
        for theta_idx in range(self.RM_size[2]):
            theta = theta_idx * self.theta_step
            piece_j_on_canvas = PieceOnCanvas(piece=self.puzzle.pieces[j], grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=theta, enabled_features=self.features_status)
            #  LINES case
            overlap_lines = cv2.filter2D(piece_i_on_canvas.lines_mask, -1, piece_j_on_canvas['lines_mask'])
            dilated_overlap_lines = dilate(overlap_lines, width=np.floor(self.borders_regions_width_outside * self.xy_step).astype(int))
            binary_overlap_lines = (dilated_overlap_lines > self.threshold_overlap_lines).astype(np.int32)
            binary_overlap_lines_no_pad = binary_overlap_lines[self.p_hs + 1:-(self.p_hs + 1), self.p_hs + 1:-(self.p_hs + 1)]
            resized_lines = np.array(Image.fromarray(binary_overlap_lines_no_pad).resize((self.RM_size[0], self.RM_size[1]), Image.Resampling.NEAREST))
            RM_ij[:,:,t] = (resized_lines.astype(np.int32) - 1)
        return RM_ij

    # def compute_pairwise_feature_RM_wrapper(self, i: int, j: int, feature: str):
    def dilate(self, mask, width=3):
        """dilate wrapper for opencv-python dilate method"""
        kernel_size = width*2+1
        kernel = np.ones((kernel_size, kernel_size))
        dilated_mask = cv2.dilate(mask, kernel)
        return dilated_mask 

    def get_borders_around(self, mask, border_dilation=3, border_erosion=3):
        """
        Get the borders around the mask contour (border_erosion outside, border_dilation inside) 
        """
        kernel_dilation = np.ones((border_dilation, border_dilation))
        kernel_erosion = np.ones((border_erosion, border_erosion))
        dilated_mask = cv2.dilate(mask, kernel_dilation)
        eroded_mask = cv2.erode(mask, kernel_erosion)
        return dilated_mask - eroded_mask

    def save_candidate_alignments_to_file(self, folder_name: str ='relative_transformations_images', verbose: int = 1):
        """
        Save to files (images) the image-version of the points in the RM which are positive!
        It is used to "see" the candidate alignments of each pairs which are consiedered "good candidates" from the region_matrix computation
        """
        # assert self.RM_computed == True, "Please run RMM.compute() first! Saving candidate alignments requires values on the RM matrix!"
        if verbose > 1:
            print("-" * 50)
            print("Combining RMs")
        self.combine_RMs()
        for i in range(self.puzzle.num_of_pieces):
            for j in range(self.puzzle.num_of_pieces):
                if i != j:
                    relative_transformations_mat = self.combined_RM[:,:,:,j,i]
                    # path
                    relative_transformations_folder_pair = os.path.join(self.exp_folder, folder_name, f"pieces_{i}_vs_{j}")
                    os.makedirs(relative_transformations_folder_pair, exist_ok=True)
                    valid_rel_t_values = np.where(relative_transformations_mat > 0)
                    y_ids, x_ids, theta_ids = valid_rel_t_values
                    assert len(x_ids) == len(y_ids) == len(theta_ids), "something went wrong during the extraction of the values from the combined matrix!"
                    if verbose > 1:
                        print(f"Found {len(x_ids)} possible alignment candidate relative transformations for piece {i} vs piece {j}!")
                    for k in range(len(x_ids)):
                        xj, yj = self.grid.xy_values[x_ids[k], y_ids[k]]
                        thetaj = self.grid.theta_values[theta_ids[k]]
                        print(xj, yj, thetaj, self.puzzle.pieces[i].data.image.shape)
                        image_relative_transf = self.render_pair_at(i, j, xj, yj, thetaj)
                        img_path = os.path.join(relative_transformations_folder_pair, f'candidate_assembly_{k}_x{xj}_y{yj}_theta{thetaj}.png')
                        cv2.imwrite(img_path, image_relative_transf)
                        # breakpoint()
                        #plt.imsave(img_path, image_relative_transf)
                        
                    # rm_path = os.path.join(self.exp_folder, 'RM.npy')
                    # np.save(rm_path, self.RM)

    def render_pair_at(self, i: int, j: int, xj: int, yj: int, thetaj: int, crop: bool = True, padding: int = 3, max_noise: int = 0):
        """
        It places the two pieces on a virtual canvas with piece_i at the center and piece_j at xj, yj, thetaj and returns the image
        """
        piece_i_on_canvas = PieceOnCanvas(piece=self.puzzle.pieces[i], grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=0, enabled_features=self.features_status)
        piece_j_on_canvas = PieceOnCanvas(piece=self.puzzle.pieces[j], grid=self.grid, x=xj, y=yj, theta=thetaj, enabled_features=self.features_status)
        rendered_image = self.create_aligned_image(piece_i_on_canvas, piece_j_on_canvas)
        if crop == True:
            rendered_image = crop_to_content(rendered_image, padding=padding, max_noise=max_noise)
        #rendered_image = cv2.cvtColor(rendered_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return rendered_image

    def create_aligned_image(self, piece_i: PieceOnCanvas, piece_j: PieceOnCanvas):
        """
        Creates the combined image. If there is no overlap, is just piece_i + piece_j. 
        This method should take care of small overlaps and also small holes (fill them/align the images)
        """
        combo_mask = (piece_i.mask + piece_j.mask)
        combo_image = (piece_i.image + piece_j.image)
        if np.sum(combo_mask) > 1:
            # OVERLAP
            mask = combo_mask == 2
            combo_image[mask] /= 2
        return combo_image

    def combine_RMs(self):
        """
        Combine the region matrices, knowing that:
        - shape has positive, zero and negative values
        - feature-based have positive and zero values
        """
        negative_regions_shape_based = (self.RM['shape'] < 0).astype(int)
        positive_regions_shape_based = (self.RM['shape'] > 0).astype(int)
        positive_regions = np.zeros_like(positive_regions_shape_based)
        negative_regions = np.zeros_like(negative_regions_shape_based)
        for feature in self.features:
            if self.features_status[feature] == True and feature != 'shape':
                if feature == 'motives':
                    positive_regions_motives_based, negative_regions_motives_based = MotivesRMM.extract_pos_neg_motives_RM(self.RM[feature])
                    # we combine positive regions (they should be 1 only if 
                    # it's 1 in both the feature-based RM and the shape-based one)
                    positive_regions = cv2.bitwise_and(positive_regions_shape_based, positive_regions_motives_based)
                    # we combine also negative regions (they should be -1 if there is a -1 in any of the two RM)
                    # keep in mind that we always use `True/1` values and then we subtract these at the end to get negative part
                    # because it's easier to use `bitwise_and` and `bitwise_or` operations
                    negative_regions = cv2.bitwise_or(negative_regions_shape_based, negative_regions_motives_based)
        self.combined_RM = positive_regions - negative_regions



###############################################################
#                                                             #
#  ███╗   ███╗ ██████╗ ████████╗██╗██╗   ██╗███████╗███████╗  #
#  ████╗ ████║██╔═══██╗╚══██╔══╝██║██║   ██║██╔════╝██╔════╝  #
#  ██╔████╔██║██║   ██║   ██║   ██║██║   ██║█████╗  ███████╗  #
#  ██║╚██╔╝██║██║   ██║   ██║   ██║╚██╗ ██╔╝██╔══╝  ╚════██║  #
#  ██║ ╚═╝ ██║╚██████╔╝   ██║   ██║ ╚████╔╝ ███████╗███████║  #
#  ╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═══╝  ╚══════╝╚══════╝  #
#                                                             #
#  ██████╗ ███╗   ███╗███╗   ███╗                             #
#  ██╔══██╗████╗ ████║████╗ ████║                             #
#  ██████╔╝██╔████╔██║██╔████╔██║                             #
#  ██╔══██╗██║╚██╔╝██║██║╚██╔╝██║                             #
#  ██║  ██║██║ ╚═╝ ██║██║ ╚═╝ ██║                             #
#  ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝                             #
#                                                             #
###############################################################
class MotivesRMM():

    @staticmethod
    def extract_pos_neg_motives_RM(RM):
        # for k in range(RM.shape[5]):
        # positive values to 1 (enough to have it on one layer)
        RM_all_motives_pos = np.sum(RM > 0, axis=5)
        # negative values to -1 (one layer -1 implies -1 even if other layers have +1!)
        RM_all_motives_neg = np.sum(RM < 0, axis=5)
        # here we apply it
        RM_all_motives = np.clip(RM_all_motives_pos - 2 * RM_all_motives_neg, -1, 1)
        positives = (RM_all_motives > 0).astype(int)
        negatives = (RM_all_motives < 0).astype(int)
        return positives, negatives

    