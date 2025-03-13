import scipy 
import numpy as np 
import cv2 
from utils.puzzle_utils import PuzzlePiece
import shapely 

class PuzzleGrid():

    def __init__(self, grid_parameters, piece_size):
        # repetition? # may be needed in place_on_canvas
        self.piece_size = piece_size
        self.p_hs = self.piece_size // 2
        self.xy_step = grid_parameters['xy_step']
        self.xy_points = grid_parameters['xy_points']
        self.theta_step = grid_parameters['theta_step']
        self.theta_points = grid_parameters['theta_points']
        self.pairwise_comp_range = self.xy_step * (self.xy_points - 1)
        self.canvas_size = self.pairwise_comp_range + 2 * (self.p_hs + 1)
        self.canvas_center = self.canvas_size // 2
        self.create_grid_data()

    def create_grid_data(self):
        # we can create using the `largest_val` or using the `step` and `points`
        # largest_val = self.piece_size * 2 #step*pts
        largest_val = self.xy_step * self.xy_points
        # create a regularly spaced grid (the center value should be the center of th epiece)
        axis_grid = np.arange(0, largest_val, self.xy_step)
        zero_aligned_axis_grid = axis_grid - axis_grid[np.floor(len(axis_grid) // 2).astype(int)]
        # align to the canvas
        canvas_alignment = self.canvas_size // 2 # - largest_val - step) / 2
        pieces_grid = np.zeros((len(axis_grid), len(axis_grid), 2))
        for b in range(len(axis_grid)):
            for g in range(len(axis_grid)):
                pieces_grid[g, b] = (axis_grid[g]+canvas_alignment, axis_grid[b]+canvas_alignment)
        self.data = pieces_grid.astype(int)


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
            self.motives_map = np.zeros((grid.canvas_size, grid.canvas_size, piece.features.motives.shape[2]))
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
        polygon = shapely.affinity.rotate(piece.data.polygon, -theta, origin=piece.data.img_center)
        #piece_mask = (piece_mask > eps_mh).astype(np.uint8)
        if enabled_features['shape'] == True:
            breakpoint()
            sdf = scipy.ndimage.rotate(piece.features.sdf, theta, reshape=False, mode='constant', order=0)
        if enabled_features['lines'] == True:
            lines_mask = scipy.ndimage.rotate(piece.features.lines_mask, theta, reshape=False, mode='constant', order=0, prefilter=False)
            lines_mask = cv2.morphologyEx(lines_mask, cv2.MORPH_CLOSE, closing_kernel)
        ## NEW MOTIF-BASED
        if enabled_features['motives'] == True:
            motives_map = scipy.ndimage.rotate(piece.features.motives_map, theta, reshape=False, mode='constant', order=0)
            
        # PLACEMENT
        # then we place them into their canvas version
        self.image[y_c0:y_c1, x_c0:x_c1, :] = image
        self.mask[y_c0:y_c1, x_c0:x_c1] = mask
        self.polygon = shapely.affinity.transform(polygon, lambda f: f + [x,y] - piece.data.img_center)
        if enabled_features['shape'] == True:
            self.sdf[y_c0:y_c1, x_c0:x_c1, :] = sdf
        if enabled_features['lines'] == True:
            self.lines_mask[y_c0:y_c1, x_c0:x_c1, :] = lines_mask
        if enabled_features['motives'] == True:
            self.motives_map[y_c0:y_c1, x_c0:x_c1, :] = motives_map
        self.centroid = np.asarray([x, y])


def place_on_canvas(piece: PuzzlePiece, grid: PuzzleGrid, x: float, y: float, theta: float):
    """
    OLD METHOD, DO NOT USE NOW
    """
    # to fix "holes" due to interpolation when rotating masks
    eps_mh = 0.0005
    closing_kernel = np.ones((9, 9))

    # handling Grayscale and RGB images
    # if len(piece['img'].shape) > 2:
    #     img_with_channels = True
    #     channels = piece['img'].shape[2]
    # else:
    #     img_with_channels = False
    # if img_with_channels is True:
    #     img_on_canvas = np.zeros((canvas_size, canvas_size, channels))
    # else:
    #     img_on_canvas = np.zeros((canvas_size, canvas_size))
    # can we assume RGB?
    

    piece_img = piece.data.image
    piece_mask = piece.data.mask
    # piece.data.img_center
    #half_piece_shift = [piece_center_pixel, piece_center_pixel]
    
    ## TODO:
    # check keys in piece because we may forget something half-way
    # for example `lines_mask` ?

    # we should have a more elegant for loop on the keys
    # not a hard-coded part
  
    if 'sdf' in piece.keys():
    # if piece.features.sdf is not None: ?
        piece_sdf = piece.features.sdf
        sdf_on_canvas = np.zeros((grid.canvas_size, grid.canvas_size))
        sdf_on_canvas += np.min(piece_sdf)
    if 'lines_mask' in piece.keys():
        lines_on_canvas = np.zeros((grid.canvas_size, grid.canvas_size))
        piece_lines_mask = piece.features.lines
    ## NEW MOTIF-BASED
    if 'motif_mask' in piece.keys():
        piece_motif_mask = piece.features.motives
        n_motifs = piece_motif_mask.shape[2]
        motif_on_canvas = np.zeros((grid.canvas_size, grid.canvas_size, n_motifs))

    

    # def place_on_canvas(piece: PuzzlePiece, coords: tuple, canvas_size, theta=0):       
    if 'polygon' in piece.keys():
        poly_on_canvas = transform(rotated_poly, lambda f: f + [x,y] - half_piece_shift)

    if piece['img'].shape[0] % 2 == 0:
        msk_on_canvas[y_c0:y_c1, x_c0:x_c1] = piece_mask
        if img_with_channels is True:
            img_on_canvas[y_c0:y_c1, x_c0:x_c1, :] = piece_img
        else:
            img_on_canvas[y_c0:y_c1, x_c0:x_c1] = piece_img
        if 'sdf' in piece.keys():
            sdf_on_canvas[y_c0:y_c1, x_c0:x_c1] = piece_sdf
        if 'lines_mask' in piece.keys():
            lines_on_canvas[y_c0:y_c1, x_c0:x_c1] = piece_lines_mask
        # NEW MOTIF-BASED !!!!
        if 'motif_mask' in piece.keys():
            motif_on_canvas[y_c0:y_c1, x_c0:x_c1, :] = piece_motif_mask

    else:
        msk_on_canvas[y_c0:y_c1 + 1, x_c0:x_c1 + 1] = piece_mask
        if img_with_channels is True:
            img_on_canvas[y_c0:y_c1+1, x_c0:x_c1+1, :] = piece_img
        else:
            img_on_canvas[y_c0:y_c1+1, x_c0:x_c1+1] = piece_img
        if 'sdf' in piece.keys():
            sdf_on_canvas[y_c0:y_c1+1, x_c0:x_c1+1] = piece_sdf
        if 'lines_mask' in piece.keys():
            lines_on_canvas[y_c0:y_c1+1, x_c0:x_c1+1] = piece_lines_mask
        ## NEW MOTIF-BASED
        if 'motif_mask' in piece.keys():
            motif_on_canvas[y_c0:y_c1+1, x_c0:x_c1+1, :] = piece_motif_mask
    
    shift_y = y - piece['cm'][1]
    shift_x = x - piece['cm'][0]
    cm_on_canvas = [piece['cm'][0] + shift_x, piece['cm'][1] + shift_y]
    piece_on_canvas = {
        'img': img_on_canvas.astype(int),
        'mask': msk_on_canvas,
        'cm': cm_on_canvas,
    }
    if 'sdf' in piece.keys():
        piece_on_canvas['sdf'] = sdf_on_canvas
    if 'polygon' in piece.keys():
        piece_on_canvas['polygon'] = poly_on_canvas
    if 'lines_mask' in piece.keys():
        piece_on_canvas['lines_mask'] = lines_on_canvas
    ## NEW MOTIF-BASED
    if 'motif_mask' in piece.keys():
        piece_on_canvas['motif_mask'] = motif_on_canvas

    # plt.imshow(piece_on_canvas['img'])
    # plt.plot(*piece_on_canvas['polygon'].boundary.xy)
    # breakpoint()
    return piece_on_canvas
