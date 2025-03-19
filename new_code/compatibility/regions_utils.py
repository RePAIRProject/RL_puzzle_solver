import numpy as np 
import os
# import features_utils as fts_uts
from utils.puzzle_utils import Puzzle 
from compatibility.compatibility_utils import PuzzleGrid, PieceOnCanvas
from utils.parameters_utils import Configuration
from utils.visualization_utils import crop_to_content
import cv2 
from PIL import Image
import matplotlib.pyplot as plt

class RegionMatrixModule:
    """
    The class for all the operation (initialization, computation, loading, writing)
    related to the region matrix (also called partial payoff matrix in the paper)
    
    It contains a dictionary which contains the different region matrices 
    - shape based (self.RM['shape'])
    - motif based (self.RM['motif'])
    - and so on..
    """
    def __init__(self, puzzle: Puzzle, params: dict):
        
        self.puzzle = puzzle
        # self.pieces = pcs_uts.load_pieces(puzzle_name=puzzle_name)
        # if use_feats == True:
        #     self.pieces = fts_uts.load_features(pieces=self.pieces, puzzle_name=puzzle_name)
        self.params = params
        # here we need to load stuff from the yaml file
        self.features_status = {}
        self.features = []
        self.check_features_status()      
        self.cfg = Configuration(puzzle.name) 
        self.exp_folder = self.cfg.get_puzzle_single_run_random_folder_name()

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
        self.grid = PuzzleGrid(self.params['compatibility']['grid'], self.piece_size)
        self.regions_dilation = self.params['compatibility']['regions']['borders_dilation']
        self.regions_erosion = self.params['compatibility']['regions']['borders_erosion']
        self.threshold_overlap_shapes = self.piece_size / 6 # it was /2 !
        self.threshold_overlap_lines = self.piece_size / 8
        self.threshold_overlap_motifs = self.piece_size / 5
        self.RM_size = (self.grid.xy_num_points, self.grid.xy_num_points, self.grid.theta_num_points, self.puzzle.num_of_pieces, self.puzzle.num_of_pieces)
        self.RM_computed = False
        self.RM = {
            'shape': np.zeros(self.RM_size)
        }
        for feature in self.features:
            if self.features_status[feature] == True:
                self.RM[feature] = np.zeros(self.RM_size)
        

    def load_from_file(self, file_path: str):
        """
        Loading the RM matrix from file if it already exists
        """
        print("WIP: specify if more matrices are in the same file!")
        assert os.path.exists(file_path), f"{file_path} non existing! please check"
        if file_path.endswith('.mat'):
            import scipy
            RM_dict = scipy.io.loadmat(file_path)
            for feat in features:
                if feat in RM_dict.keys():
                    self.RM[f'{feat}'] = RM_dict[f'R_{feat}']
                    self.features.append(feat)
                else:
                    print(f"could not find {feat}-based RM")
                    # raise Exception(f"When reading {file_path} I expected as key R_{RM_type}, but was not found\nPlease pass RM_type such that R_`RM_type` is a key of the dictionary read.") 
        else:
            print("WIP: probably not working")
            self.RM = np.load(file_path)
            self.features = []

    def save(self):
        breakpoint()
        os.makedirs(self.exp_folder, exist_ok=True)
        rm_path = os.path.join(self.exp_folder, 'RM.npy')
        np.save(rm_path, self.RM)

    def compute(self, verbose=1):
        """
        Computes the region matrix for all pairs of pieces 
        for all features we have extracted
        This works calling the "pairwise" RM, can be good for debug
        """
        for i in range(self.puzzle.num_of_pieces):
            for j in range(self.puzzle.num_of_pieces):
                # here we compute the "basic" shape-based RM
                # self.RM['shape'][:, :, :, j, i] = self.compute_pairwise_shape_based_RM(i, j)
                for feature in self.features:
                    if self.features_status[feature] == True:
                        self.RM[feature][:, :, :, j, i] = self.compute_pairwise_feature_RM_wrapper(i, j, feature)
                    else:
                        if verbose > 1:
                            print(f"{feature} is disabled, skipping.")
        self.RM_computed = True
        if verbose > 1:
            print("RM computed!")

    def compute_pairwise_feature_RM_wrapper(self, i: int, j: int, feature: str):
        """
        Just a wrapper, will decide which method to call depending on the feature
        """
        if feature == 'shape':
            RM = self.compute_pairwise_shape_based_RM(i, j)
        elif feature == 'lines':
            RM = self.compute_pairwise_line_based_RM(i, j)
        elif feature == 'motif':
            RM = self.compute_pairwise_motif_based_RM(i, j)
        else:
            raise Exception(f"{feature}-based RM not implemented yet!")

        return RM 

    def compute_pairwise_line_based_RM(self, i: int, j: int):
        """
        Line based RM with "positive" 1 regions and "empty" 0 regions
        It should be used in combination with shape-based, as it does not have the -1 values!
        """
        RM = np.zeros((self.RM_size[0], self.RM_size[1], self.RM_size[2]))
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
            RM[:,:,t] = (resized_lines.astype(np.int32) - 1)
        return RM

    def compute_pairwise_shape_based_RM(self, i: int, j: int, dilate: bool = True, erode: bool = True):
        """
        Shape based RM with 1, 0 and -1 regions
        """
        RM = np.zeros((self.RM_size[0], self.RM_size[1], self.RM_size[2]))
        piece_i_on_canvas = PieceOnCanvas(piece=self.puzzle.pieces[i], grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=0, enabled_features=self.features_status)
        # piece_i_on_canvas = pcs_uts.place_on_canvas(piece_i, (center_pos, center_pos), self.canvas_size, 0)
        for theta_idx in range(self.RM_size[2]):
            theta = theta_idx * self.grid.theta_step
            piece_j_on_canvas = PieceOnCanvas(piece=self.puzzle.pieces[j], grid=self.grid, x=self.grid.canvas_center, y=self.grid.canvas_center, theta=theta, enabled_features=self.features_status)
            # piece_j_on_canvas = pcs_uts.place_on_canvas(piece_j, (center_pos, center_pos), self.canvas_size, t * self.theta_step)
            # SHAPE case - BASIC
            overlap_shapes = cv2.filter2D(piece_i_on_canvas.mask, -1, piece_j_on_canvas.mask)
            thresholded_regions_map = (overlap_shapes > self.threshold_overlap_shapes).astype(np.int32)
            breakpoint()
            plt.imshow(overlap_shapes)
            plt.show()
            breakpoint()

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
            RM[:,:,theta_idx] = (resized_shape.astype(np.int32) - 1)

        return RM

    def get_borders_around(self, mask, border_dilation=3, border_erosion=3):
        """
        Get the borders around the mask contour (border_erosion outside, border_dilation inside) 
        """
        kernel_dilation = np.ones((border_dilation, border_dilation))
        kernel_erosion = np.ones((border_erosion, border_erosion))
        dilated_mask = cv2.dilate(mask, kernel_dilation)
        eroded_mask = cv2.erode(mask, kernel_erosion)
        return dilated_mask - eroded_mask

    def save_candidate_alignments_to_file(self, folder_name='relative_transformations_images'):
        """
        Save to files (images) the image-version of the points in the RM which are positive!
        It is used to "see" the candidate alignments of each pairs which are consiedered "good candidates" from the region_matrix computation
        """
        assert self.RM_computed == True, "Please run RMM.compute() first! Saving candidate alignments requires values on the RM matrix!"
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
                    for k in range(len(x_ids)):
                        xj, yj = self.grid.xy_values[x_ids[k], y_ids[k]]
                        thetaj = self.grid.theta_values[theta_ids[k]]
                        print(xj, yj, thetaj, self.puzzle.pieces[i].data.image.shape)
                        image_relative_transf = self.render_pair_at(i, j, xj, yj, thetaj)
                        img_path = os.path.join(relative_transformations_folder_pair, f'candidate_assembly_{k}_x{xj}_y{yj}_theta{thetaj}.png')
                        cv2.imwrite(img_path, image_relative_transf)
                        breakpoint()
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
        negative_region = self.RM['shape'] < 0
        combined_positive_region = self.RM['shape'] * (self.RM['shape'] > 0).astype(int)
        for feature in self.features:
            if self.features_status[feature] == True:
                combined_positive_region *= self.RM[feature] * (self.RM[feature] > 0).astype(int)
        self.combined_RM = combined_positive_region - negative_region
