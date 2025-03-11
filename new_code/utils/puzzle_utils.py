"""
All other `_utils.py` files can import these methods, so here we should not import anything from them to avoid circular imports
We can import parameters_utils.py as it contains what we need to handle our .yaml files
It is a kind of "basic" utility functions for loading the pieces
"""
import os 
import numpy as np
import cv2 
from parameters_utils import PuzzleDaedalus
import scipy
from typing import List
import yaml

# Preprocessor
#
# for file in folder
#
#     img = cv2.imread(file)
#     piece = PuzzlePiece(img)
#     piece.mask =
#     piece.dadad
#     cm = get_center_of_mass(piece.mask)
#
#
#     piece.save_to_files(name)

# p = PuzzlePiece()
# p.id
# p.name
# p.data.image
# p.data.mask
# p.data.polygon
# p.features.lines
# p.features.motives
# p.features.sdf
class Puzzle:
    def __init__(self, pieces: List[PuzzlePiece] = []):
        self.pieces = pieces
        self.num_of_pieces = len(self.pieces)

    def load_data_from_files(self, puzzle_name: str):
        """
        Loads the data (images, masks and polygon) and fill the properties of the Puzzle object
        """
        images_subfolder = PuzzleDaedalus.get_puzzle_images_subfolder(puzzle_name=puzzle_name)
        masks_subfolder = PuzzleDaedalus.get_puzzle_masks_subfolder(puzzle_name=puzzle_name)
        polygons_subfolder = PuzzleDaedalus.get_puzzle_polygons_subfolder(puzzle_name=puzzle_name)
        pieces_names = os.listdir(images_subfolder)
        pieces_names.sort()
        for piece_name in pieces_names:
            piece = PuzzlePiece()
            piece.name = piece_name
            piece.id = piece.name[:10]  # piece_XXXXX.png
            sepiecelf.data.image = cv2.imread(os.path.join(images_subfolder, f"{piece_name}".png))
            piece.data.mask = plt.imread(os.path.join(masks_subfolder, f"{piece_name}".png), cv2.IMREAD_GRAYSCALE)
            piece.data.polygon = np.load(os.path.join(polygons_subfolder, piece.name), allow_pickle=True).tolist()
            self.pieces.append(piece)
        self.num_of_pieces = len(self.pieces)

    def load_features_from_files(self, puzzle_name: str):
        """
        TODO: Loads the features (whatever it finds in the `features` folder) already extracted!
        """
        # features_extracted = PuzzleDaedalus.get_features_extracted(puzzle_name=puzzle_name)
        # puzzle_feats = PuzzleFeatures(puzzle_features_root_folder = PuzzleDaedalus.get_puzzle_features_subfolder(puzzle_name=puzzle_name))
        # for piece in pieces:
        #     for feature in features_extracted:
        #         piece[feature] = puzzle_feats.extract_feature(piece, feature)
        # return pieces

    def load_parameters_from_file(self, yaml_file_path: str):
        """
        Loads the parameters from the .yaml file 
        Some parameters are "consequences" of the loaded one (calculated from)
        """
        with open(yaml_file_path, 'r') as file:
            parameters = yaml.safe_load(file)
        self.piece_size = parameters['preprocessing']['piece_size']
        self.p_hs = self.piece_size // 2
        self.grid_pars = PuzzleGrid(parameters['compatibility']['grid'], self.piece_size)
        self.regions_dilation = parameters['compatibility']['regions']['borders_dilation']
        self.regions_erosion = parameters['compatibility']['regions']['borders_erosion']
    
    def save_parameters_to_file(self, yaml_file_path: str):
        """
        Save (the parameters) to the .yaml file 
        """

class PuzzleGrid():
    def __init__(self, grid_parameters, piece_size):
        # repetition?
        # self.piece_size = piece_size
        self.p_hs = self.piece_size // 2
        self.xy_step = grid_parameters['xy_step']
        self.xy_points = grid_parameters['xy_points']
        self.theta_step = grid_parameters['theta_step']
        self.theta_points = grid_parameters['theta_points']
        self.pairwise_comp_range = self.xy_step * (self.xy_points - 1)
        self.canvas_size = self.pairwise_comp_range + 2 * (self.p_hs + 1)
        self.create_grid()

    def create_grid(self):
        # we can create using the `largest_val` or using the `step` and `points`
        # largest_val = self.piece_size * 2 #step*pts
        largest_val = self.xy_step * self.xy_points
        # create a regularly spaced grid (the center value should be the center of th epiece)
        axis_grid = np.arange(0, largest_val, self.xy_step)
        zero_aligned_axis_grid = axis_grid - axis_grid[np.floor(len(axis_grid) // 2).astype(int)]
        # align to the canvas
        canvas_alignment = canvas_size // 2 # - largest_val - step) / 2
        pieces_grid = np.zeros((len(axis_grid), len(axis_grid), 2))
        for b in range(len(axis_grid)):
            for g in range(len(axis_grid)):
                pieces_grid[g, b] = (axis_grid[g]+canvas_alignment, axis_grid[b]+canvas_alignment)
        self.grid = pieces_grid.astype(int)


class Lines():
    def __init__(self):
        self.detection_method = None

class Motives():
    def __init__(self):
        self.segmentation_method = None

class SDF():
    def __init__(self):
        self.method = None

class Features():
    def __init__(self):
        self.lines = Lines()
        self.motives = Motives()
        self.sdf = SDF()

class PuzzlePiece:
    def __init__(self, *args, **kwargs):
        self.id = None
        self.name = None
        self.centroid_preproc = None      # centroid
        self.data.image = None
        self.data.mask = None
        self.data.polygon = None
        self.features = Features()

    # save preprocessed data
    def save_to_files(self):
        # TODO: save all the data to files
        return True



def get_center_of_mass(mask):
    """
    Calculates center of mass
    """
    mass_y, mass_x = np.where(mask >= 0.5)
    cent_x = np.average(mass_x)
    cent_y = np.average(mass_y)
    # center = [ np.average(indices) for indices in np.where(th1 >= 255) ]
    return [cent_x, cent_y]

def load_single_piece(path, background=0, verbose=False):
    """
    The path is related to the image (.png) of the piece. 
    We assume the rest (mask/etc) are in the standard folders
    -----------
    Params:
        - path: str = path to image
        - background: int = if the image has a unique value as background
        - verbose: bool = if we want to print out more stuff
    -----------
    Returns:
        - dict: piece_d = a dictionary with all the relative information
    """
    piece_d = {}
    img = cv2.imread(path)
    piece_d['img'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if mask_path == '':
        piece_d['mask'] = get_mask(piece_d['img'], background=background, noisy=True)
    else:
        mask = plt.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
    piece_d['cm'] = get_cm(piece_d['mask'])
    piece_name = path.split('/')[-1]
    piece_d['id'] = piece_name[:10]  # piece_XXXXX.png
    piece_d['name'] = piece_name[:-4]  # piece_XXXXX.png    
    return piece_d

def load_pieces(puzzle_name, background=0, verbose=False):
    """
    The path is related to the image (.png) of the piece. 
    We assume the rest (mask/etc) are in the standard folders
    -----------
    Params:
        - puzzle_name: str = puzzle folder name
        - background: int = if the image has a unique value as background
        - verbose: bool = if we want to print out more stuff
    -----------
    Returns:
        - pieces: list = a list of dictionaries of the single pieces with all the relative information
    """
    pieces = []
    images_folder = PuzzleDaedalus.get_puzzle_images_subfolder(puzzle_name=puzzle_name)
    masks_folder = PuzzleDaedalus.get_puzzle_masks_subfolder(puzzle_name=puzzle_name)
    polygons_folder = PuzzleDaedalus.get_puzzle_polygons_subfolder(puzzle_name=puzzle_name)

    pieces_names = os.listdir(images_folder)
    pieces_names.sort()

    if verbose is True:
        print(f"Found {len(pieces_names)} pieces:")
    # needed?
    closing_kernel = np.ones((9, 9))
    
    for piece_name in pieces_names:
        if verbose is True:
            print(f'- {piece_name}')
        piece_full_path = os.path.join(data_folder, piece_name)
        piece_d = {}
        # IMAGE
        img = cv2.imread(piece_full_path)
        piece_d['img'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # MASK
        mask_full_path = os.path.join(masks_folder, piece_name)
        mask = plt.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        if len(mask.shape) > 2:
            print("WARNING:Mask has multiple channels, using the first..")
            mask = mask[:,:,0]
        piece_d['mask'] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
        # POLYGON
        piece_d['polygon'] = np.load(os.path.join(polygons_folder, piece_name), allow_pickle=True).tolist()
        # CENTER OF MASS
        piece_d['cm'] = get_center_of_mass(piece_d['mask'])
        # NAME / ID
        piece_d['name'] = piece_name[:-4]  # piece_XXXXX.png
        if 'repair' in dataset:
            piece_d['id'] = piece_name[:9]   # RPf_00194.png
        else:
            piece_d['id'] = piece_name[:10]  # piece_XXXXX.png

        pieces.append(piece_d)

    return pieces

def load_features(pieces: list, puzzle_name: str):
    """
    defined as a method to be called without the need to initialize extra objects
    """
    features_extracted = PuzzleDaedalus.get_features_extracted(puzzle_name=puzzle_name)
    puzzle_feats = PuzzleFeatures(puzzle_features_root_folder = PuzzleDaedalus.get_puzzle_features_subfolder(puzzle_name=puzzle_name))
    for piece in pieces:
        for feature in features_extracted:
            piece[feature] = puzzle_feats.extract_feature(piece, feature)
    return pieces


def get_borders_around(mask, border_dilation=3, border_erosion=3):
    """
    Get the borders around the mask contour (border_erosion outside, border_dilation inside) 
    """
    kernel_dilation = np.ones((border_dilation, border_dilation))
    kernel_erosion = np.ones((border_erosion, border_erosion))
    dilated_mask = cv2.dilate(mask, kernel_dilation)
    eroded_mask = cv2.erode(mask, kernel_erosion)
    return dilated_mask - eroded_mask


def place_on_canvas(piece, coords, canvas_size, theta=0):
    ## TODO:
    # check keys in piece because we may forget something half-way
    # for example `lines_mask` ?

    # we should have a more elegant for loop on the keys
    # not a hard-coded part

    # to fix "holes" due to interpolation when rotating masks
    eps_mh = 0.0005
    closing_kernel = np.ones((9, 9))

    y, x = coords
    hs = piece['img'].shape[0] // 2
    # TODO: ceil, floor, int? does it matter?
    y_c0 = np.ceil(y-hs).astype(int)
    y_c1 = np.ceil(y+hs).astype(int)
    x_c0 = np.ceil(x-hs).astype(int)
    x_c1 = np.ceil(x+hs).astype(int)
    # maybe not the best, but a quick fix ?
    if y_c1 - y_c0 == 2*hs + 1:
        y_c1 -= 1
    elif y_c1 - y_c0 == 2*hs - 1:
        y_c1 += 1
    if x_c1 - x_c0 == 2*hs + 1:
        x_c1 -= 1
    elif x_c1 - x_c0 == 2*hs - 1:
        x_c1 += 1

    if len(piece['img'].shape) > 2:
        img_with_channels = True
        channels = piece['img'].shape[2]
    else:
        img_with_channels = False
    if img_with_channels is True:
        img_on_canvas = np.zeros((canvas_size, canvas_size, channels))
    else:
        img_on_canvas = np.zeros((canvas_size, canvas_size))

    msk_on_canvas = np.zeros((canvas_size, canvas_size))
    if 'sdf' in piece.keys():
        sdf_on_canvas = np.zeros((canvas_size, canvas_size))
        sdf_on_canvas += np.min(piece['sdf'])
        piece_sdf = piece['sdf']
    if 'lines_mask' in piece.keys():
        lines_on_canvas = np.zeros((canvas_size, canvas_size))
        piece_lines_mask = piece['lines_mask']
    ## NEW MOTIF-BASED
    if 'motif_mask' in piece.keys():
        piece_motif_mask = piece['motif_mask']
        n_motifs = piece_motif_mask.shape[2]
        motif_on_canvas = np.zeros((canvas_size, canvas_size, n_motifs))


    piece_img = piece['img']
    piece_mask = piece['mask']
    if 'polygon' in piece.keys():
        piece_center_pixel = piece_img.shape[0] // 2
        half_piece_shift = [piece_center_pixel, piece_center_pixel]

    ## ROTATE
    if theta > 0:
        piece_img = scipy.ndimage.rotate(piece_img, theta, reshape=False, mode='constant', order=0)
        piece_mask = scipy.ndimage.rotate(piece_mask, theta, reshape=False, mode='constant', prefilter=False, order=0)
        piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_CLOSE, closing_kernel)
        #piece_mask = (piece_mask > eps_mh).astype(np.uint8)
        if 'sdf' in piece.keys():
            piece_sdf = scipy.ndimage.rotate(piece_sdf, theta, reshape=False, mode='constant', order=0)
        if 'lines_mask' in piece.keys():
            piece_lines_mask = scipy.ndimage.rotate(piece_lines_mask, theta, reshape=False, mode='constant', order=0, prefilter=False)
            piece_lines_mask = cv2.morphologyEx(piece_lines_mask, cv2.MORPH_CLOSE, closing_kernel)
            #piece_lines_mask = (piece_lines_mask > eps_mh).astype(np.uint8)
        piece['cm'] = get_cm(piece_mask)
        ## NEW MOTIF-BASED
        if 'motif_mask' in piece.keys():
            piece_motif_mask = scipy.ndimage.rotate(piece_motif_mask, theta, reshape=False, mode='constant', order=0)
        #piece['cm'] = get_cm(piece_mask)
        if 'polygon' in piece.keys():
            rotated_poly = rotate(piece['polygon'], -theta, origin=half_piece_shift)
    else:
        if 'polygon' in piece.keys():
            rotated_poly = piece['polygon']
        
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

def crop_to_content(image, padding=1, return_vals=False, max_noise=0):

    if len(image.shape) > 2:
        x0 = np.clip(np.min(np.where(np.sum(image, axis=2) > max_noise)[1]) - padding, 0, image.shape[1])
        x1 = np.clip(np.max(np.where(np.sum(image, axis=2) > max_noise)[1]) + padding, 0, image.shape[1])
        y0 = np.clip(np.min(np.where(np.sum(image, axis=2) > max_noise)[0]) - padding, 0, image.shape[0])
        y1 = np.clip(np.max(np.where(np.sum(image, axis=2) > max_noise)[0]) + padding, 0, image.shape[0])
    else:
        x0 = np.min(np.where(image > max_noise)[1]) - padding
        x1 = np.max(np.where(image > max_noise)[1]) + padding
        y0 = np.min(np.where(image > max_noise)[0]) - padding
        y1 = np.max(np.where(image > max_noise)[0]) + padding

    if return_vals == True:
        return image[y0:y1, x0:x1, :], x0, x1, y0, y1
    return image[y0:y1, x0:x1, :]