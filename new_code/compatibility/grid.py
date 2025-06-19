import scipy 
import numpy as np 
import cv2 
from utils.puzzle_utils import PuzzlePiece
import shapely 



##################################
#                                #
#   ██████╗ ██████╗ ██╗██████╗   #
#  ██╔════╝ ██╔══██╗██║██╔══██╗  #
#  ██║  ███╗██████╔╝██║██║  ██║  #
#  ██║   ██║██╔══██╗██║██║  ██║  #
#  ╚██████╔╝██║  ██║██║██████╔╝  #
#   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝   #
#                                #
##################################
class PuzzleGrid():

    def __init__(self, grid_parameters, piece_size):
        self.p_hs = piece_size // 2
        self.xy_step = grid_parameters['xy_step']
        self.xy_num_points = grid_parameters['xy_num_points']
        self.theta_step = grid_parameters['theta_step']
        self.theta_num_points = grid_parameters['theta_num_points']
        self.pairwise_comp_range = self.xy_step * (self.xy_num_points - 1)
        self.canvas_size = self.pairwise_comp_range + 2 * (self.p_hs + 1)
        self.canvas_center = self.canvas_size // 2


        # we can create using the `largest_val` or using the `step` and `points`
        # largest_val = self.piece_size * 2 #step*pts
        largest_val = self.xy_step * self.xy_num_points
        # create a regularly spaced grid (the center value should be the center of th epiece)
        axis_grid = np.arange(0, largest_val, self.xy_step)
        zero_aligned_axis_grid = axis_grid - axis_grid[np.floor(len(axis_grid) // 2).astype(int)]
        # align to the canvas
        canvas_alignment = self.canvas_size // 2 # - largest_val - step) / 2
        pieces_grid = np.zeros((len(axis_grid), len(axis_grid), 2))
        for b in range(len(axis_grid)):
            for g in range(len(axis_grid)):
                pieces_grid[g, b] = (zero_aligned_axis_grid[g]+canvas_alignment, zero_aligned_axis_grid[b]+canvas_alignment)
        self.xy_values = pieces_grid.astype(int)
        self.theta_values = np.arange(0, 360, self.theta_step)            




#########################################################
#                                                       #
#   ██████╗ █████╗ ███╗   ██╗██╗   ██╗ █████╗ ███████╗  #
#  ██╔════╝██╔══██╗████╗  ██║██║   ██║██╔══██╗██╔════╝  #
#  ██║     ███████║██╔██╗ ██║██║   ██║███████║███████╗  #
#  ██║     ██╔══██║██║╚██╗██║╚██╗ ██╔╝██╔══██║╚════██║  #
#  ╚██████╗██║  ██║██║ ╚████║ ╚████╔╝ ██║  ██║███████║  #
#   ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝  #
#                                                       #
#########################################################
class PieceOnCanvas:
    """
    Not sure if having it as a class is needed, but it might help with autocompletion and knowing what is inside.
    It just creates a slightly modified version of the PuzzlePiece object placed on a canvas, 
    which is then used in the RM or CM calculation
    """
    def __init__(self, piece: PuzzlePiece, grid: PuzzleGrid, x: float, y: float, theta: float, enabled_features: dict):
        
        # placement of the piece
        y_c0 = np.ceil(y-grid.p_hs).astype(int)
        y_c1 = np.ceil(y+grid.p_hs+1).astype(int)
        x_c0 = np.ceil(x-grid.p_hs).astype(int)
        x_c1 = np.ceil(x+grid.p_hs+1).astype(int)
        # maybe not the best, but a quick fix ?
        # if y_c1 - y_c0 == 2*grid.p_hs + 1:
        #     y_c1 -= 1
        # elif y_c1 - y_c0 == 2*grid.p_hs - 1:
        #     y_c1 += 1
        # if x_c1 - x_c0 == 2*grid.p_hs + 1:
        #     x_c1 -= 1
        # elif x_c1 - x_c0 == 2*grid.p_hs - 1:
        #     x_c1 += 1

        self.image = np.zeros((grid.canvas_size, grid.canvas_size, piece.data.image.shape[2]))
        self.mask = np.zeros((grid.canvas_size, grid.canvas_size))
        # self.polygon = transform(piece.data.polygon, lambda f: f - piece.data.img_center)
        if enabled_features['shape'] == True:
            self.sdf = np.zeros((grid.canvas_size, grid.canvas_size)) + np.min(piece.features.sdf.data)
        if enabled_features['lines'] == True:
            self.lines_mask = np.zeros((grid.canvas_size, grid.canvas_size))
        if enabled_features['motives'] == True:
            self.motives_cube = np.zeros((grid.canvas_size, grid.canvas_size, piece.features.motives.num_of_classes))
        # self.centroid = np.zeros((2,1))

        # each features should have their own abstract method
        # lines_mask = piece.features.lines.rotate()
        # lines_mask = piece.features.lines.transform() ?

        # ROTATION
        # first we rotate the components
        # if theta > 0:
        closing_kernel = np.ones((9, 9))
        image = scipy.ndimage.rotate(piece.data.image, theta, reshape=False, mode='constant', order=0)
        mask = scipy.ndimage.rotate(piece.data.mask, theta, reshape=False, mode='constant', prefilter=False, order=0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
        polygon = shapely.affinity.rotate(piece.data.polygon, -theta, origin=tuple(piece.data.img_center))
        #piece_mask = (piece_mask > eps_mh).astype(np.uint8)
        if enabled_features['shape'] == True:
            sdf = scipy.ndimage.rotate(piece.features.sdf.data, theta, reshape=False, mode='constant', order=0)
        if enabled_features['lines'] == True:
            lines_mask = scipy.ndimage.rotate(piece.features.lines_mask, theta, reshape=False, mode='constant', order=0, prefilter=False)
            lines_mask = cv2.morphologyEx(lines_mask, cv2.MORPH_CLOSE, closing_kernel)
        ## NEW MOTIF-BASED
        if enabled_features['motives'] == True:
            motives_cube = scipy.ndimage.rotate(piece.features.motives.motives_cube, theta, reshape=False, mode='constant', order=0)
            
        # PLACEMENT
        # then we place them into their canvas version
        if x_c0 < 0 or y_c0 < 0 or y_c1 > grid.canvas_size or x_c1 > grid.canvas_size:
            print("#" * 50)
            print("WARNING: seems like the piece is being placed outside of the canvas! Check the code")
            print(f"we are trying:\n\tself.image[{y_c0}:{y_c1}, {x_c0}:{x_c1}, :] = image\nwhere")
            print(f"\timage.shape = {image.shape}\n\tgrid.canvas_size = {grid.canvas_size}")
            print("#" * 50)
        self.image[y_c0:y_c1, x_c0:x_c1, :] = image
        self.mask[y_c0:y_c1, x_c0:x_c1] = mask
        for ch in range(self.image.shape[2]):
            self.image[:,:,ch] *= (self.mask > 0)
        self.polygon = shapely.transform(polygon, lambda f: f + [x,y] - piece.data.img_center)
        if enabled_features['shape'] == True:
            self.sdf[y_c0:y_c1, x_c0:x_c1] = sdf
        if enabled_features['lines'] == True:
            self.lines_mask[y_c0:y_c1, x_c0:x_c1, :] = lines_mask
        if enabled_features['motives'] == True:
            self.motives_cube[y_c0:y_c1, x_c0:x_c1, :] = motives_cube
        self.centroid = np.asarray([x, y])