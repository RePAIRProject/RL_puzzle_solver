import numpy as np 
import pieces_utils as pcs_uts
import features_utils as fts_uts

class RegionMatrix():
    """
    The class for all the operation (initialization, computation, loading, writing)
    related to the region matrix (also called partial payoff matrix in the paper)
    
    It contains a dictionary which contains the different region matrices 
    - shape based (self.RM['shape'])
    - motif based (self.RM['motif'])
    - and so on..
    """
    def __init__(self, RM_size, puzzle_name: str, use_feats: str = True):
        assert len(shape) == 3, "shape should be a tuple/list with three elements"
        self.RM_size = RM_size
        self.RM = {
            'shape': np.zeros(RM_size)
        }
        self.features = [
            'shape'
        ]
        self.pieces = pcs_uts.load_pieces(puzzle_name=puzzle_name)
        if use_feats == True:
            self.pieces = fts_uts.load_features(pieces=self.pieces, puzzle_name=puzzle_name)
        self.num_pieces = len(self.pieces)
        # here we need to load stuff from the yaml file
        self.canvas_size = 1 #
        self.xy_step = 1 #
        self.theta_step = 1 #
        self.threshold_overlap = 1 #
        self.borders_regions_width_outside = 1 #
        self.borders_regions_width_inside = 1 #
        self.p_hs = 1 #

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

    def compute(self, verbose=1):
        """
        Computes the region matrix for all pairs of pieces 
        for all features we have extracted
        """
        for i in self.num_pieces:
            for j in self.num_pieces:
                # here we compute the "basic" shape-based RM
                self.RM['shape'][:, :, :, j, i] = self.compute_pairwise_shape_based_RM(i, j)
                for feature in self.features:
                    self.RM[feature][:, :, :, j, i] = self.compute_pairwise_feature_RM_wrapper(i, j, feature)

    def compute_pairwise_feature_RM_wrapper(self, i: int, j: int, feature: str):
        """
        Just a wrapper, will decide which method to call depending on the feature
        """
        if feature == 'lines':
            RM = self.compute_pairwise_line_based_RM()
        elif feature == 'motif':
            RM = self.compute_pairwise_motif_based_RM()
        else:
            raise Exception(f"{feature}-based RM not implemented yet!")

        return RM 

    def compute_pairwise_line_based_RM(self, i: int, j: int):
        """
        Line based RM with "positive" 1 regions and "empty" 0 regions
        It should be used in combination with shape-based, as it does not have the -1 values!
        """
        RM = np.zeros((self.RM_size[0], self.RM_size[1], self.RM_size[2]))
        center_pos = self.canvas_size // 2
        piece_i_on_canvas = pcs_uts.place_on_canvas(piece_i, (center_pos, center_pos), self.canvas_size, 0)
        for t in range(self.RM_size[2]):
            piece_j_on_canvas = pcs_uts.place_on_canvas(piece_j, (center_pos, center_pos), self.canvas_size, t * self.theta_step)
            #  LINES case
            overlap_lines = cv2.filter2D(piece_i_on_canvas['lines_mask'], -1, piece_j_on_canvas['lines_mask'])
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
        center_pos = self.canvas_size // 2
        piece_i_on_canvas = pcs_uts.place_on_canvas(piece_i, (center_pos, center_pos), self.canvas_size, 0)
        for t in range(self.RM_size[2]):
            piece_j_on_canvas = pcs_uts.place_on_canvas(piece_j, (center_pos, center_pos), self.canvas_size, t * self.theta_step)
            # SHAPE case - BASIC
            overlap_shapes = cv2.filter2D(piece_i_on_canvas['mask'], -1, piece_j_on_canvas['mask'])
            thresholded_regions_map = (overlap_shapes > self.threshold_overlap).astype(np.int32)
            
            if dilate == True:
                border_dilation = int(self.borders_regions_width_outside * self.xy_step)
            else:
                border_dilation = 1
            if erode == True:
                border_erosion = int(self.borders_regions_width_inside * self.xy_step)
            else:
                border_erosion = 1

            around_borders_trm = pcs_uts.get_borders_around(thresholded_regions_map.astype(np.uint8),
                                                border_dilation=border_dilation, border_erosion=border_erosion)
            thresholded_regions_map *= -1
            thresholded_regions_map += 2 * (around_borders_trm > 0)
            thresholded_regions_map = np.clip(thresholded_regions_map, -1, 1)

            # we convert the matrix to resize the image without losing the values
            thr_reg_map_shape_uint = (thresholded_regions_map + 1).astype(np.uint8)
            thr_reg_map_comp_range = thr_reg_map_shape_uint[self.p_hs + 1:-(self.p_hs + 1), self.p_hs + 1:-(self.p_hs + 1)]
            resized_shape = np.array(Image.fromarray(thr_reg_map_comp_range).resize((self.RM_size[0], self.RM_size[1]), Image.Resampling.NEAREST))
            RM[:,:,t] = (resized_shape.astype(np.int32) - 1)

        return RM