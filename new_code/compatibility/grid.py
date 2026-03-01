import scipy 
import numpy as np 
import cv2 
from utils.puzzle_utils import PuzzlePiece
import shapely 

# @staticmethod
def recalculate_position_after_rotation(x1, y1, x2, y2, rotation_O1_deg, rotation_O2_deg):
    """
    Recalculate the position of O2 after rotating O1.
    
    Parameters:
    -----------
    x1, y1 : float
        Position of object O1
    x2, y2 : float
        Initial position of object O2
    rotation_O1_deg : float
        Rotation angle for O1 in degrees
    rotation_O2_deg : float, optional
        Rotation angle for O2 in degrees. If None, assumes O2 rotates by the same amount as O1
    
    Returns:
    --------
    tuple : (new_x2, new_y2, rotation_O2_deg)
        New position of O2 and its rotation angle
    """        
    # Convert degrees to radians
    theta = np.radians(rotation_O1_deg)
    
    # Calculate the offset vector from O1 to O2
    dx = x2 - x1
    dy = y2 - y1
    
    # Create rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Rotate the offset vector
    dx_new = dx * cos_theta - dy * sin_theta
    dy_new = dx * sin_theta + dy * cos_theta
    
    # Calculate new position of O2
    new_x2 = x1 + dx_new
    new_y2 = y1 + dy_new
    
    return new_x2, new_y2, rotation_O2_deg

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
        self.canvas_size = self.pairwise_comp_range + 2 * (self.p_hs + 1) + grid_parameters.get('canvas_buffer', 0)
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
        if self.theta_step == 0:
            self.theta_values = np.asarray([0])
        else:
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
    def __init__(self, piece: PuzzlePiece, grid: PuzzleGrid, x: float, y: float, theta: float, enabled_features: dict=None):
        
        # placement of the piece
        y_c0 = np.ceil(y-grid.p_hs).astype(int)
        y_c1 = np.ceil(y+grid.p_hs).astype(int)
        # if piece.data.image.shape[0] % 2 == 1:
        #     y_c1 +=1 
        x_c0 = np.ceil(x-grid.p_hs).astype(int)
        x_c1 = np.ceil(x+grid.p_hs).astype(int)
        # if piece.data.image.shape[1] % 2 == 1:
        #     x_c1 +=1
        # maybe not the best, but a quick fix ?
        if y_c1 - y_c0 == 2*grid.p_hs + 1:
            y_c1 -= 1
        elif y_c1 - y_c0 == 2*grid.p_hs - 1:
            y_c1 += 1
        if x_c1 - x_c0 == 2*grid.p_hs + 1:
            x_c1 -= 1
        elif x_c1 - x_c0 == 2*grid.p_hs - 1:
            x_c1 += 1

        self.image = np.zeros((grid.canvas_size, grid.canvas_size, piece.data.image.shape[2]))
        self.mask = np.zeros((grid.canvas_size, grid.canvas_size))
        # self.polygon = transform(piece.data.polygon, lambda f: f - piece.data.img_center)
        if enabled_features is not None:
            if enabled_features['shape']:
                self.sdf = np.zeros((grid.canvas_size, grid.canvas_size)) + np.min(piece.features.sdf.data)
            if enabled_features['lines']:
                self.lines_mask = np.zeros((grid.canvas_size, grid.canvas_size))
            if enabled_features['motives']:
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
        if enabled_features is not None:
            if enabled_features['shape']:
                sdf = scipy.ndimage.rotate(piece.features.sdf.data, theta, reshape=False, mode='constant', order=0)
            if enabled_features['lines']:
                lines_mask = scipy.ndimage.rotate(piece.features.lines_mask, theta, reshape=False, mode='constant', order=0, prefilter=False)
                lines_mask = cv2.morphologyEx(lines_mask, cv2.MORPH_CLOSE, closing_kernel)
            ## NEW MOTIF-BASED
            if enabled_features['motives']:
                motives_cube = scipy.ndimage.rotate(piece.features.motives.motives_cube, theta, reshape=False, mode='constant', order=0)
                
        # PLACEMENT
        # then we place them into their canvas version
        if x_c0 < 0 or y_c0 < 0 or y_c1 > grid.canvas_size or x_c1 > grid.canvas_size:
            print("#" * 50)
            print("WARNING: seems like the piece is being placed outside of the canvas! Check the code")
            print(f"we are trying:\n\tself.image[{y_c0}:{y_c1}, {x_c0}:{x_c1}, :] = image\nwhere")
            print(f"\timage.shape = {image.shape}\n\tgrid.canvas_size = {grid.canvas_size}")
            print("#" * 50)
        # print(f"we are trying:\n\tself.image[{y_c0}:{y_c1}, {x_c0}:{x_c1}, :] = image\nwhere")
        self.image[y_c0:y_c1, x_c0:x_c1, :] = image
        self.mask[y_c0:y_c1, x_c0:x_c1] = mask
        for ch in range(self.image.shape[2]):
            self.image[:,:,ch] *= (self.mask > 0)
        self.polygon = shapely.transform(polygon, lambda f: f + [x,y] - piece.data.img_center)
        if enabled_features is not None:
            if enabled_features['shape'] == True:
                self.sdf[y_c0:y_c1, x_c0:x_c1] = sdf
            if enabled_features['lines'] == True:
                self.lines_mask[y_c0:y_c1, x_c0:x_c1, :] = lines_mask
            if enabled_features['motives'] == True:
                self.motives_cube[y_c0:y_c1, x_c0:x_c1, :] = motives_cube
        self.centroid = np.asarray([x, y])

    def blend_with(self, piece: "PieceOnCanvas", blend_mode:str='average', mask_type:str='pieces_center_first', return_mask:bool=False):
        """ 
        blend image and masks from two pieces on canvas (self, and the one given).
        
        - blend_mode controls how the two images are merged (in case of overlap)
            'average' means averaging the two colors to get a slightly blurred transition
        - mask_type controls how the two masks are saved (if return_mask==True)
            'binary' means 1 for pixels belonging to any piece, 0 for background
            'pieces_center_first' means 1 for central piece (self), 2 for other piece, and in case of overlap, 1
            'pieces_other_first' means 1 for central piece (self), 2 for other piece, and in case of overlap, 2

        it raises NotImplementedError() in case of other mode or types which are not considered. 
        Possible to edit the code below to include more possibilities
        """ 
        aligned_image = self.image + piece.image
        aligned_mask = self.mask + piece.mask
        if np.max(aligned_mask) > 1:
            if blend_mode == 'average':
                average_mask = np.clip(aligned_mask, 1, 2)
                aligned_image /= np.dstack((average_mask, average_mask, average_mask, (average_mask>-1)))
            else:
                raise NotImplementedError()

        if return_mask == True:
            if mask_type == 'binary':
                aligned_mask = np.clip(aligned_mask, 0, 1)
            elif mask_type.startswith('pieces'):
                aligned_mask = self.mask + 2 * piece.mask
                if mask_type == 'pieces_center_first':
                    aligned_mask[aligned_mask == 3] = 1
                elif mask_type == 'pieces_other_first':
                    aligned_mask[aligned_mask == 3] = 2 # or aligned_mask = np.clip(aligned_mask, 0, 2)
            else:
                raise NotImplementedError()
            return aligned_image, aligned_mask
        
        return aligned_image