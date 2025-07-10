"""
All other `_utils.py` files can import these methods, so here we should not import anything from them to avoid circular imports
We can import parameters_utils.py as it contains what we need to handle our .yaml files
It is a kind of "basic" utility functions for loading the pieces
"""
import os 
import numpy as np
import cv2 
from utils.parameters_utils import Configuration
import scipy
from typing import List
import yaml
import matplotlib.pyplot as plt 
import skfmm
import natsort

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
class Lines():
    def __init__(self):
        self.detection_method = None


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
class Motives():
    def __init__(self):
        self.segmentation_method = 'yolo-seg'

    def load(self, path: str):
        self.motives_cube = np.load(path)
        self.num_of_classes = self.motives_cube.shape[2]

    def load_RM(self, path: str):
        """
        Loads and returns the RM for motives
        Which has one more dimension (stores each motif as a layer) 
        """
        return True 
    
    def aggregate_RM(self, baseline_RM):
        """
        Aggregates it to the baseline_RM given, which should have positive, zero and negative values (between 1 and -1)
        """
        return False


##############################
#                            #
#  ███████╗██████╗ ███████╗  #
#  ██╔════╝██╔══██╗██╔════╝  #
#  ███████╗██║  ██║█████╗    #
#  ╚════██║██║  ██║██╔══╝    #
#  ███████║██████╔╝██║       #
#  ╚══════╝╚═════╝ ╚═╝       #
#                            #
##############################
class SDF():
    def __init__(self):
        self.method = None
    
    def compute(self, mask, q=1):
        phi = np.int64(mask[:, :])
        phi = np.where(phi, 0, -1) + 0.5
        sdf = skfmm.distance(phi, dx = 1)
        if q > 1: #quantize (stepwise sdf)
            sdf = (sdf // q) * q
        self.data = sdf


########################################################################
#                                                                      #
#  ███████╗███████╗ █████╗ ████████╗██╗   ██╗██████╗ ███████╗███████╗  #
#  ██╔════╝██╔════╝██╔══██╗╚══██╔══╝██║   ██║██╔══██╗██╔════╝██╔════╝  #
#  █████╗  █████╗  ███████║   ██║   ██║   ██║██████╔╝█████╗  ███████╗  #
#  ██╔══╝  ██╔══╝  ██╔══██║   ██║   ██║   ██║██╔══██╗██╔══╝  ╚════██║  #
#  ██║     ███████╗██║  ██║   ██║   ╚██████╔╝██║  ██║███████╗███████║  #
#  ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝  #
#                                                                      #
########################################################################
class Features():
    def __init__(self):
        self.lines = Lines()
        self.motives = Motives()
        self.sdf = SDF()


#######################################
#                                     #
#  ██████╗  █████╗ ████████╗ █████╗   #
#  ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗  #
#  ██║  ██║███████║   ██║   ███████║  #
#  ██║  ██║██╔══██║   ██║   ██╔══██║  #
#  ██████╔╝██║  ██║   ██║   ██║  ██║  #
#  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝  #
#                                     #
#######################################
class Data():
    def __init__(self):
        self.image = None
        self.img_center = None 
        self.mask = None 
        self.polygon = None 


#########################################
#                                       #
#  ██████╗ ██╗███████╗ ██████╗███████╗  #
#  ██╔══██╗██║██╔════╝██╔════╝██╔════╝  #
#  ██████╔╝██║█████╗  ██║     █████╗    #
#  ██╔═══╝ ██║██╔══╝  ██║     ██╔══╝    #
#  ██║     ██║███████╗╚██████╗███████╗  #
#  ╚═╝     ╚═╝╚══════╝ ╚═════╝╚══════╝  #
#                                       #
#########################################
class PuzzlePiece:
    def __init__(self, *args, **kwargs):
        self.id = None
        self.name = None
        self.centroid_preproc = None      # centroid
        self.data = Data()
        self.features = Features()

    # save preprocessed data
    def save_to_files(self):
        # TODO: save all the data to files
        return True



#######################################################
#                                                     #
#  ██████╗ ██╗   ██╗███████╗███████╗██╗     ███████╗  #
#  ██╔══██╗██║   ██║╚══███╔╝╚══███╔╝██║     ██╔════╝  #
#  ██████╔╝██║   ██║  ███╔╝   ███╔╝ ██║     █████╗    #
#  ██╔═══╝ ██║   ██║ ███╔╝   ███╔╝  ██║     ██╔══╝    #
#  ██║     ╚██████╔╝███████╗███████╗███████╗███████╗  #
#  ╚═╝      ╚═════╝ ╚══════╝╚══════╝╚══════╝╚══════╝  #
#                                                     #
#######################################################
class Puzzle:

    def __init__(self): #, pieces: List[PuzzlePiece] = []):
        self.pieces = [] # somehow using pieces "kept" the old pieces when running on a dataset over multiple puzzles! Cannot understand why  
        self.num_of_pieces = len(self.pieces)

    def load(self, puzzle_name: str, data_folder: str, load_features: bool = True):
        """
        Loads the data (images, masks and polygon) and fill the properties of the Puzzle object
        """
        self.name = puzzle_name
        self.cfg = Configuration()
        ##
        self.cfg.set_data_folder(data_folder)
        ##
        self.cfg.set_puzzle_name(puzzle_name)
        images_subfolder = self.cfg.get_puzzle_images_subfolder()
        masks_subfolder = self.cfg.get_puzzle_masks_subfolder()
        polygons_subfolder = self.cfg.get_puzzle_polygons_subfolder()
        self.computer_ordered_pieces_names = os.listdir(images_subfolder)
        self.pieces_names = natsort.natsorted(self.computer_ordered_pieces_names)
        
        # breakpoint()
        for j, piece_name in enumerate(self.pieces_names):
            piece = PuzzlePiece()
            piece.name = piece_name[:-4]
            piece.id = j  
            piece.repair_id = piece.name[:10]  # piece_XXXXX.png
            piece.data.image = plt.imread(os.path.join(images_subfolder, f"{piece.name}.png"))
            piece.data.img_center = np.asarray(piece.data.image.shape[:2]) // 2
            piece.data.mask = cv2.imread(os.path.join(masks_subfolder, f"{piece.name}.png"), cv2.IMREAD_GRAYSCALE)
            piece.data.polygon = np.load(os.path.join(polygons_subfolder, f"{piece.name}.npy"), allow_pickle=True).tolist()
            if load_features == True:
                piece.features.sdf.compute(piece.data.mask)
                piece.features.motives.load(os.path.join(self.cfg.get_puzzle_features_subfolder(), 'motifs_segmentation', f"motifs_cube_{piece_name[:-4]}.npy"))
            #piece.data.features = load_features(self)
            self.pieces.append(piece)

        self.num_of_pieces = len(self.pieces)
        self.img_piece_size = self.pieces[0].data.image.shape

    # write code to check which features folders exist?
    # def check_extracted_features(self):
    #     features = self.params['compatibility']['features']
    #     for feature in features:
    #         self.features.append(feature)
    #         self.features_status[feature] = features[feature]['enabled']       

    def load_features(self):
        """
        TODO: Loads the features (whatever it finds in the `features` folder) already extracted!
        """
        return True
        # features_extracted = cfg.get_features_extracted(puzzle_name=puzzle_name)
        # puzzle_feats = PuzzleFeatures(puzzle_features_root_folder = cfg.get_puzzle_features_subfolder(puzzle_name=puzzle_name))
        # for piece in pieces:
        #     for feature in features_extracted:
        #         piece[feature] = puzzle_feats.extract_feature(piece, feature)
        # return pieces




####################################################################
#                                                                  #
#   ██████╗ ██╗     ██████╗      ██████╗ ██████╗ ██████╗ ███████╗  #
#  ██╔═══██╗██║     ██╔══██╗    ██╔════╝██╔═══██╗██╔══██╗██╔════╝  #
#  ██║   ██║██║     ██║  ██║    ██║     ██║   ██║██║  ██║█████╗    #
#  ██║   ██║██║     ██║  ██║    ██║     ██║   ██║██║  ██║██╔══╝    #
#  ╚██████╔╝███████╗██████╔╝    ╚██████╗╚██████╔╝██████╔╝███████╗  #
#   ╚═════╝ ╚══════╝╚═════╝      ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝  #
#                                                                  #
####################################################################

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
    images_folder = cfg.get_puzzle_images_subfolder(puzzle_name=puzzle_name)
    masks_folder = cfg.get_puzzle_masks_subfolder(puzzle_name=puzzle_name)
    polygons_folder = cfg.get_puzzle_polygons_subfolder(puzzle_name=puzzle_name)

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
    features_extracted = cfg.get_features_extracted(puzzle_name=puzzle_name)
    puzzle_feats = PuzzleFeatures(puzzle_features_root_folder = cfg.get_puzzle_features_subfolder(puzzle_name=puzzle_name))
    for piece in pieces:
        for feature in features_extracted:
            piece[feature] = puzzle_feats.extract_feature(piece, feature)
    return pieces

