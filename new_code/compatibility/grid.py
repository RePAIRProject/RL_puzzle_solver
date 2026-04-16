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

@staticmethod
def weirdly_working_rotation(img, degrees:int, method:str='CV'):
    ################################################################################################################################################
    #   NOTE: this should not be done like this!
    #   why do we rotate `img[2:, 2:]` ? 
    #       cannot explain, really. There is an issue with the center "value" (we have odd size images guaranteed, so floating value)
    #       which never aligns with any rotation method, and empirically I found out that this gentle nudge (+2) before rotation is needed for the 
    #       correct rotation. It seems simple (just move the "center" + 1!) but after losing a lot of time trying to find an explainable solution, 
    #       I gave up. If you find the solution and can explain, please fix the code and reach out, I will be grateful.
    #       The debug visualization parts are here to help "visualize" the issue if needed.
    #   also, I think now polygons are screwed up (of course, because of this push), and to correct, there should be an offset (dependent on the 
    #   angle). But they are not used, so we probably leave here this bomb ready to explode
    ################################################################################################################################################
    if method == 'CV' or method == 'OPENCV-ROTATE':
        if degrees == 0 or degrees == 360:
            rotated_img = img
        elif degrees == 90 or degrees == -270:
            rotated_img = cv2.rotate(img[2:,2:], cv2.ROTATE_90_COUNTERCLOCKWISE) 
        elif degrees == 180 or degrees == -180:
            rotated_img = cv2.rotate(img[2:,2:], cv2.ROTATE_180) 
        elif degrees == 270 or degrees == -90:
            rotated_img = cv2.rotate(img[2:,2:], cv2.ROTATE_90_CLOCKWISE) 
        else:
            print(f"ERROR: this method handles only 90deg rotations! (and degrees={degrees})\nChoose 'WARP' or 'scipy' for other rotations.")
        final_img = np.zeros_like(img)
        final_img[2:, 2:] = rotated_img
    elif method == 'ND' or method == 'scipy': # this handles also non-90 degrees rotations
        rotated_img = scipy.ndimage.rotate(img[2:,2:], degrees, reshape=False, mode='constant')
        final_img = np.zeros_like(img)
        final_img[2:, 2:] = rotated_img
    return final_img
    # if we want to include a third method..
    # elif method == 'OPENCV-WARP' or method == 'WARP':
    #         # region_rot = ndimage.rotate(region_pad, degree, reshape=False, cval=bg_color)
    #         if center_of_rotation is None:
    #             center_of_rotation = (np.array(squared_img.shape[:2])) / 2.0  # match scipy's center convention
    #         rotation_mat = cv2.getRotationMatrix2D(center_of_rotation, degrees, 1)
    #         rotated_square_img = cv2.warpAffine(squared_img, rotation_mat, (squared_img.shape[1], squared_img.shape[0]),  # keep original size
    #                         flags=cv2.INTER_LINEAR,
    #                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    #         rotated_square_mask = cv2.warpAffine(squared_mask, rotation_mat, (squared_img.shape[1], squared_img.shape[0]),  # keep original size
    #                         flags=cv2.INTER_LINEAR,
    #                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)



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
    which is then used in the RM or CM calculation (mostly to create visually the image of a pairwise alignment between two pieces)
    """
    def __init__(self, piece: PuzzlePiece, grid: PuzzleGrid, x: float, y: float, theta: float, enabled_features: dict=None):
        
        # for now actually using integers.
        # this avoid some issues with .000000005 numbers
        x = int(x)
        y = int(y)
        theta = int(theta)

        # placement of the piece
        y_c0 = np.ceil(y-grid.p_hs).astype(int)
        y_c1 = np.ceil(y+grid.p_hs+1).astype(int)
        # if piece.data.image.shape[0] % 2 == 1:
        #     y_c1 +=1 
        x_c0 = np.ceil(x-grid.p_hs).astype(int)
        x_c1 = np.ceil(x+grid.p_hs+1).astype(int)

        # print(f"{piece.name} --> x:{x}, y:{y}, p_hs: {grid.p_hs} // yc[{y_c0}:{y_c1}], xc[{x_c0}:{x_c1}] // shape: ({y_c1-y_c0}, {x_c1-x_c0})")
        #### THIS SHOULD NOT BE NECESSARY
        # if piece.data.image.shape[1] % 2 == 1:
        #     x_c1 +=1
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
        if enabled_features is not None:
            if enabled_features['shape']:
                self.sdf = np.zeros((grid.canvas_size, grid.canvas_size)) + np.min(piece.features.sdf.data)
            if enabled_features['lines']:
                self.lines_mask = np.zeros((grid.canvas_size, grid.canvas_size))
            if enabled_features['motives']:
                self.motives_cube = np.zeros((grid.canvas_size, grid.canvas_size, piece.features.motives.num_of_classes))

        # each features should have their own abstract method
        # lines_mask = piece.features.lines.rotate()
        # lines_mask = piece.features.lines.transform() ?

        ################################################
        #   DEBUG VISUALIZATION 1 (continues below)
        ################################################
        # import matplotlib.pyplot as plt
        # coords = np.argwhere(piece.data.mask)
        # y0, x0 = coords.min(axis=0)
        # y1, x1 = coords.max(axis=0)
        # plt.subplot(131); plt.imshow(piece.data.image); plt.title("Before rotation"); 
        # plt.scatter([x0, x0, x1, x1], [y0, y1, y0, y1], marker='x', c='orange')
        # plt.plot([x0, x1], [y0, y1], c='orange')
        # plt.plot([x1, x0], [y0, y1], c='orange')
        # plt.scatter(piece.data.img_center[0], piece.data.img_center[1], marker='x', c='red')
        # # plt.plot(*piece.data.polygon.boundary.xy)
        # plt.scatter(piece.data.image.shape[0] / 2, piece.data.image.shape[1] / 2, marker='x', c='green')

        image = weirdly_working_rotation(piece.data.image, degrees=theta, method='scipy')
        mask = weirdly_working_rotation(piece.data.mask, degrees=theta, method='scipy')
        # OLD CODE (not aligned)
        # image = scipy.ndimage.rotate(piece.data.image, theta, reshape=False, mode='constant', order=0)
        # mask = scipy.ndimage.rotate(piece.data.mask, theta, reshape=False, mode='constant', prefilter=False, order=0)
        closing_kernel = np.ones((9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)

        ################################################
        #   DEBUG VISUALIZATION 1 (continuing)
        ################################################
        # coords = np.argwhere(mask)
        # y0, x0 = coords.min(axis=0)
        # y1, x1 = coords.max(axis=0)
        # plt.subplot(132); plt.imshow(image); plt.title(f"After rotation {theta}")
        # plt.scatter([x0, x0, x1, x1], [y0, y1, y0, y1], marker='x', c='orange')
        # plt.plot([x0, x1], [y0, y1], c='orange')
        # plt.plot([x1, x0], [y0, y1], c='orange')
        # plt.scatter(piece.data.img_center[0], piece.data.img_center[1], marker='x', c='red')
        # plt.scatter(image.shape[0] / 2, image.shape[1] / 2, marker='x', c='green')
        # plt.subplot(132); plt.imshow(image+piece.data.image); plt.title(f"Overlapping the two")
        # # plt.scatter([x0, x0, x1, x1], [y0, y1, y0, y1], marker='x', c='orange')
        # # plt.plot([x0, x1], [y0, y1], c='orange')
        # # plt.plot([x1, x0], [y0, y1], c='orange')
        # plt.scatter(piece.data.img_center[0], piece.data.img_center[1], marker='x', c='red')
        # plt.scatter(image.shape[0] / 2, image.shape[1] / 2, marker='x', c='green')
        # plt.show()
        # breakpoint()

        if piece.data.polygon is not None:
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
        if piece.data.polygon is not None:
            self.polygon = shapely.transform(polygon, lambda f: f + [x,y] - piece.data.img_center)
        if enabled_features is not None:
            if enabled_features['shape']:
                self.sdf[y_c0:y_c1, x_c0:x_c1] = sdf
            if enabled_features['lines']:
                self.lines_mask[y_c0:y_c1, x_c0:x_c1, :] = lines_mask
            if enabled_features['motives']:
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

        if return_mask:
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

    def check_overlap(self, piece: "PieceOnCanvas"):
        aligned_mask = self.mask + piece.mask
        if np.max(aligned_mask) > 1:
            return True 
        else:
            return False

    def touches(self, piece: "PieceOnCanvas"):
        combined_mask = np.clip(self.mask + piece.mask, 0, 1)
        num_labels, labels = cv2.connectedComponents(combined_mask.astype(np.uint8))
        if num_labels == 2:
            return True 
        else:
            return False 
        #num_blobs = num_labels - 1  # label 0 is background

    def touches_without_overlapping(self, piece: "PieceOnCanvas"):
        combined_mask = np.clip(self.mask + piece.mask, 0, 1)
        num_labels, labels = cv2.connectedComponents(combined_mask.astype(np.uint8))
        # print(np.unique(labels))
        # import matplotlib.pyplot as plt 
        # plt.subplot(1,2,1); plt.title(f"Mask, max: {np.max(combined_mask)}, n_labels: {num_labels}")
        # plt.imshow(combined_mask)
        # plt.subplot(1,2,2)
        # plt.imshow(self.image + piece.image)
        # plt.show()
        # breakpoint()
        if np.max(combined_mask) == 1 and num_labels == 2:
            return True, "touching without overlap"
        elif np.max(combined_mask) > 1:
            return False, "overlapping"
        elif num_labels == 3:
            return False, "not touching"
        else:
            return False, f"something else? (max: {np.max(combined_mask)}, num_labels: {num_labels})"

