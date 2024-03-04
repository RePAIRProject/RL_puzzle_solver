import os, pdb, json
import cv2 
import shapely 
from shapely import transform
from shapely.affinity import rotate as rotate_poly
from puzzle_utils.puzzle_gen.generator import PuzzleGenerator
from puzzle_utils.shape_utils import get_sd, get_mask, get_cm, shift_img
import numpy as np
from scipy.ndimage import rotate as rotate_img
import math
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt

"""
Thanks Friedrich -- Слава Україні 
- https://stackoverflow.com/questions/60684269/accessing-dict-values-like-an-attribute
We can access the CfgParameters dict like a module!
"""
class CfgParameters(dict):
    __getattr__ = dict.__getitem__

def calc_parameters(parameters, xy_grid_points=101, theta_grid_points=24):

    ppars = CfgParameters()
    # pieces
    ppars['piece_size'] = parameters['piece_size']
    ppars['num_pieces'] = parameters['num_pieces']
    ppars['img_size'] = parameters['size']
    ppars['p_hs'] = ppars.piece_size // 2

    # grid
    ppars['xy_grid_points'] = xy_grid_points
    ppars['theta_grid_points'] = theta_grid_points
    ppars['comp_matrix_shape'] = [ppars.xy_grid_points, ppars.xy_grid_points, ppars.theta_grid_points]
    ppars['pairwise_comp_range'] = 4 * (ppars.p_hs) + 1
    ppars['canvas_size'] = ppars.pairwise_comp_range #+ 2 * p_hs + 1 
    ppars['xy_step'] = ppars.pairwise_comp_range / (ppars.comp_matrix_shape[0] - 1)
    ppars['theta_step'] = (360 / ppars.comp_matrix_shape[2])

    # region
    ppars['threshold_overlap'] = ppars.piece_size / 2
    ppars['threshold_overlap_lines'] = ppars.piece_size / 4
    ppars['borders_regions_width_outside'] = 2
    ppars['borders_regions_width_inside'] = 5
    ppars['border_tolerance'] = ppars.piece_size // 60

    return ppars

def calc_parameters_v2(parameters, xy_step=3, xy_grid_points=101, theta_step=45):
    
    ppars = CfgParameters()
    # pieces
    ppars['piece_size'] = parameters['piece_size']
    ppars['num_pieces'] = parameters['num_pieces']
    ppars['img_size'] = parameters['size']
    ppars['p_hs'] = ppars.piece_size // 2

    # create grid starting from the step
    ppars['xy_step'] = xy_step
    ppars['xy_grid_points'] = xy_grid_points
    ppars['theta_step'] = theta_step
    if theta_step == 0 or theta_step == 360:
        ppars['theta_grid_points'] = 1
    else:
        ppars['theta_grid_points'] = int(np.round(360 / theta_step))
    ppars['pairwise_comp_range'] = xy_step * (xy_grid_points - 1)
    ppars['canvas_size'] = ppars.pairwise_comp_range + 2 * (ppars.p_hs + 1)
    ppars['comp_matrix_shape'] = [ppars.xy_grid_points, ppars.xy_grid_points, ppars.theta_grid_points]

    # region
    ppars['threshold_overlap'] = ppars.piece_size / 2
    ppars['threshold_overlap_lines'] = ppars.piece_size / 8
    ppars['borders_regions_width_outside'] = 3
    ppars['borders_regions_width_inside'] = 2  # changed from 5 +++++++
    ppars['border_tolerance'] = ppars.piece_size // 30

    return ppars

def rescale_image(img, size, lines=None):
    """
    Rescale the image (while preserving proportions) so that the largest of the two axis 
    is equal to `size`
    """
    # if max(img.shape[:2]) < size:
    #     return img 
    if img.shape[0] > img.shape[1]:
        rescaling_ratio = (size / img.shape[0])
        other_axis_size = np.round(rescaling_ratio * img.shape[1]).astype(int)
        img = cv2.resize(img, (other_axis_size, size))  # opencv use these inverted :/
    else:
        rescaling_ratio = (size / img.shape[1])
        other_axis_size = np.round(rescaling_ratio * img.shape[0]).astype(int)
        img = cv2.resize(img, (size, other_axis_size))  # opencv use these inverted :/

    if lines is not None:
        rescaled_lines = rescale_lines(lines, rescaling_ratio)
        return img, np.asarray(rescaled_lines)
    return img 

def rescale_lines(lines, ratio):
    scaled_lines = []
    for line in lines:
        new_values = (np.array(line) * ratio).tolist()
        scaled_lines.append(new_values)
    return scaled_lines

def cut_into_pieces(image, shape, num_pieces, output_path, puzzle_name, patterns_map=None, rotate_pieces=True, save_extrapolated_regions=False):

    pieces = []
    if shape == 'square':
        patch_size = image.shape[0] // num_pieces
        x0_all = np.arange(0, image.shape[0], patch_size, dtype=int)
        y0_all = np.arange(0, image.shape[1], patch_size, dtype=int)
        for iy in range(num_pieces):
            for ix in range(num_pieces):
                x0 = x0_all[ix]
                y0 = y0_all[iy]
                x1 = x0 + patch_size
                y1 = y0 + patch_size
                box = shapely.box(x0, y0, x1, y1)  # patche box (xmin, ymin, xmax, ymax)
                ## create patch
                patch = image[y0:y1, x0:x1]
                piece_in_full_image = np.zeros_like(image)
                piece_in_full_image[y0:y1, x0:x1] = patch
                mask_full_image = np.zeros((image.shape[:2]))
                mask_full_image[y0:y1, x0:x1] = 1 #np.ones((patch[:2]))
                centered_patch = patch
                centered_mask = np.ones(patch.shape[:2])
                shift2square = np.asarray([0, 0])
                # cm_image = np.asarray(image.shape[:2])/2
                # cm_patch = np.asarray([y0 + patch_size/2, x0 + patch_size/2])
                shift2center_frag = np.asarray([-x0, -y0])
                center_of_mass = np.asarray([(x1-x0)/2, (y1-y0)/2])
                squared_polygon = transform(box, lambda x: x + shift2center_frag)
                # add here the shifted version 
                piece_dict = {
                    'mask': mask_full_image,
                    'centered_mask': centered_mask,
                    'squared_mask': centered_mask,
                    'image': piece_in_full_image,
                    'centered_image': centered_patch,
                    'squared_image': centered_patch,
                    'polygon': box,
                    'centered_polygon': squared_polygon,
                    'squared_polygon': squared_polygon,
                    'center_of_mass': center_of_mass,
                    'height': patch_size,
                    'width': patch_size,
                    'shift2square': shift2square,
                    'shift2center': shift2center_frag
                }
                # print("box:", box)
                # for kk in piece_dict.keys():
                #     if not type(piece_dict[kk]) == np.ndarray:
                #         print(f"{kk}: {piece_dict[kk]}")
                # print(piece_dict['shift2center'])
                pieces.append(piece_dict)

    if shape == 'irregular':
        generator = PuzzleGenerator(image, puzzle_name)
        generator.run(num_pieces, offset_rate_h=0.2, offset_rate_w=0.2, small_region_area_ratio=0.25, rot_range=0,
            smooth_flag=True, alpha_channel=True, perc_missing_fragments=0, erosion=0, borders=False)
        generator.save_jpg_regions(output_path, skip_bg=False)
        pieces, patch_size = generator.get_pieces_from_puzzle_v2(start_from=0)
        
    if shape == 'pattern' and patterns_map is not None:
        generator = PuzzleGenerator(image, puzzle_name)
        generator.region_cnt = num_pieces + 1
        generator.region_mat = patterns_map 
        generator.save_jpg_regions(output_path, skip_bg=True)
        pieces, patch_size = generator.get_pieces_from_puzzle_v2(start_from=1)

    if (shape == 'irregular' or shape == 'pattern') and save_extrapolated_regions is True:
        if shape == 'pattern':
            generator.extrapolate_regions(start_from=1)
        if shape == 'irregular':
            generator.extrapolate_regions(start_from=0)
        extr_folder = os.path.join(output_path, 'extrapolated')
        os.makedirs(extr_folder, exist_ok=True)
        generator.save_extrapolated_regions(extrap_folder=extr_folder)
        # for j in range(len(frags)):
        #     centered_fragment, _m, _s = center_fragment(frags[j])
        #     cv2.imwrite(os.path.join(extr_folder, f"cmass_piece_{j:04d}.png"), centered_fragment)
        #     centered_extr_fragment, _m, _s = center_fragment(extr_frags[j])
        #     cv2.imwrite(os.path.join(extr_folder, f"cmass_piece_{j:04d}_ext.png"), centered_extr_fragment)

    # this is in shapely coordinates (x,y) and if we use in opencv/matplotlib we should invert order again
    for piece in pieces:
        piece['shift_global2square'] = piece['shift2center'][::-1] + piece['shift2square'][::-1]
        piece['rotation'] = 0

    ### get optimal xy_grid from GT ###########
    # pos_mat = []
    # for j in range(len(pieces)):
    #     t = pieces[j]['shift_global2square'].tolist()
    #     pos_mat.append((t))
    # position_matrix = np.array(pos_mat)
    # save_grid_info(position_matrix, output_path)
    ############################################

    if rotate_pieces == True:
        # plt.subplot(121)
        # rand_num = 2
        # plt.imshow(pieces[rand_num]['squared_image'])
        # plt.plot(*(pieces[rand_num]['squared_polygon'].boundary.xy))
        rotated_pieces, rotation_info_unused = randomly_rotate_pieces(pieces, chances_to_be_rotated=0.4, possible_rotation=4)

        # plt.subplot(122)
        # print(rotation_info)
        # plt.title(f'rotation: {rotation_info[f"piece_{rand_num:04d}"]} degrees')
        # plt.plot(*(rotated_pieces[rand_num]['squared_polygon'].boundary.xy))
        # plt.imshow(rotated_pieces[rand_num]['squared_image'])
        # plt.show()
        # pdb.set_trace()
        return rotated_pieces, patch_size

    # if we do not use rotation, we return the original pieces
    return pieces, patch_size

def save_grid_info(position_matrix, output_path):

    D = pdist(position_matrix, metric='euclidean')
    D5 = (np.round(D / 5) * 5).astype(int)
    xy_step = math.gcd(*D5)

    xy_grid_points = (np.round(np.max(D) / xy_step) * 2 + 1).astype(int)
    print(f"optimal xy_step = {xy_step} px.")
    print(f"max distance = {xy_grid_points} px.")

    xy_grid = {
        "xy_step": int(xy_step),
        "xy_grid_points": int(xy_grid_points)
    }
    with open(os.path.join(output_path, 'xy_grid.json'), 'w') as rij:
        json.dump(xy_grid, rij, indent=2)

def save_transformation_info(pieces, output_path):

    transformation_dict = {}
    for j in range(len(pieces)):
        t = pieces[j]['shift_global2square'].tolist()
        r = -int(pieces[j]['rotation'])
        transformation_dict[f"piece_{j:04d}"] = { 
            "translation": t,
            "rotation": r
            }
    with open(output_path, 'w') as rij:
        json.dump(transformation_dict, rij, indent=3)

def rotate_piece(piece, rot_ang_deg):
    """
    Piece is a dict with keys (at least):
    - 'mask': centered_mask,
    - 'centered_mask': centered_mask,
    - 'image': centered_patch,
    - 'centered_image': centered_patch,
    - 'polygon': box,
    - 'centered_polygon': box,

    We rotate only everything that start with "squared_" 
    // polygon is rotate with `shapely.affinity.rotate`
    // image with `scipy.ndimage.rotate`
    """
    rot_origin = [piece['squared_image'].shape[0] // 2, piece['squared_image'].shape[1] // 2]
    # scipy uses angle in degree for images
    # print(f"\tSize before:\nimg={piece['centered_image'].shape}\nmask={piece['centered_mask'].shape}\n")
    piece['squared_image'] = rotate_img(piece['squared_image'], rot_ang_deg, reshape=False, mode='constant')
    piece['squared_mask'] = rotate_img(piece['squared_mask'], rot_ang_deg, reshape=False, mode='constant')
    # shapely (for polygons) uses the negative angle and needs origin of rotation!
    piece['squared_polygon'] = rotate_poly(piece['squared_polygon'], -rot_ang_deg, origin=rot_origin)
    # print(f"\tSize after:\nimg={piece['centered_image'].shape}\nmask={piece['centered_mask'].shape}\n")
    return piece


def randomly_rotate_pieces(pieces, chances_to_be_rotated=0.3, possible_rotation=4):
    """Pieces is a list of dictionary containing the pieces"""
    rot_info = {}
    for j in range(len(pieces)):
        if chances_to_be_rotated > 0.99:
            rotate_this_one = True
        else:
            chance_num = np.round(1 / chances_to_be_rotated) - 1
            rotate_this_one = np.round(np.random.uniform() * chance_num).astype(np.uint8)
        if rotate_this_one == 1:
            # plt.subplot(121)
            # plt.imshow(pieces[j]['centered_image'])
            rot_num = possible_rotation - 1
            random_rotation_deg = np.round(np.random.uniform() * rot_num).astype(np.uint8) * 90
            rot_info[f"piece_{j:04d}"] = float(random_rotation_deg)
            rotated_piece_dict = rotate_piece(pieces[j], random_rotation_deg)
            pieces[j] = rotated_piece_dict
            pieces[j]['rotation'] = random_rotation_deg
            # plt.subplot(122)
            # plt.title(f'rotation: {random_rotation_deg} degrees (@randomly_rotate_pieces)')
            # plt.imshow(pieces[j]['centered_image'])
            # plt.show()
            # pdb.set_trace()
        else:
            rot_info[f"piece_{j:04d}"] = 0
            pieces[j]['rotation'] = 0
        
    return pieces, rot_info

def center_fragment(image):
    sd, mask = get_sd(image)
    cm = get_cm(mask)
    center_pos = [np.round(image.shape[1]/2).astype(int), np.round(image.shape[0]/2).astype(int)]
    shift = np.round(np.array(cm) - center_pos).astype(int)
    centered_image = shift_img(image, -shift[0], -shift[1])    
    centered_mask = shift_img(mask, -shift[0], -shift[1])    
    return centered_image, centered_mask, shift

def read_pieces(fnames, puzzle_name):
    """
    Read the pieces and return   
    """
    pieces = []
    data_folder = os.path.join(fnames.data_path, puzzle_name, fnames.imgs_folder)
    pieces_names = os.listdir(data_folder)
    for piece_name in pieces_names:
        piece_full_path = os.path.join(data_folder, piece_name)
        piece_d = {}
        img = cv2.imread(piece_full_path)
        if background != 0:
            img[img[:,:,0] == background] = 0
        if img.shape[0] != img.shape[1]:
            squared_img = np.zeros((cfg.piece_size, cfg.piece_size, 3))
            mx = cfg.piece_size - img.shape[0]
            bx = 0
            if mx % 2 > 0:
                bx = 1
            mx += bx
            by = 0
            my = cfg.piece_size - img.shape[1]
            if my % 2 > 0:
                by = 1    
            my += by
            squared_img[mx//2:-mx//2, my//2:-my//2] = img[:img.shape[0]-bx, :img.shape[1]-by]
            piece_d['img'] = squared_img
        else:
            piece_d['img'] = img
        piece_d['img'] = img
        piece_d['id'] = piece_name[:9]
        pieces.append(piece_d)
    return pieces


def place_at(piece, canvas, location):
    
    assert(np.max(location) < np.max(canvas.shape)), f"location ({location}) is out of the canvas ({canvas.shape})"
    assert(np.min(location) >= 0), f"location ({location}) has negative values"
    assert(piece.shape[0] == piece.shape[1]), "we were using squared fragments! the piece is not squared (shape {piece.shape}). What happened?"
    p_hs = piece.shape[0] // 2
    x0 = location[0] - p_hs 
    x1 = location[0] + p_hs
    y0 = location[1] - p_hs 
    y1 = location[1] + p_hs
    all_fine = True
    if x0 < 0 or x1 > canvas.shape[0]:
        print(f"error on the x axis (trying to place between {x0} and {x1}, with canvas going from 0 to {canvas.shape[0]})")
        all_fine = False
    if y0 < 0 or y1 > canvas.shape[1]:
        print(f"error on the y axis (trying to place between {y0} and {y1}, with canvas going from 0 to {canvas.shape[1]})")
        all_fine = False
    if all_fine == True:
        # this inverted because opencv use y,x
        canvas[y0:y1, x0:x1, :] += piece
    return canvas

def crop_to_content(image, padding=1, return_vals=False):

    x0 = np.min(np.where(image > 0)[1]) - padding
    x1 = np.max(np.where(image > 0)[1]) + padding
    y0 = np.min(np.where(image > 0)[0]) - padding
    y1 = np.max(np.where(image > 0)[0]) + padding

    if return_vals == True:
        return image[y0:y1, x0:x1, :], x0, x1, y0, y1
    return image[y0:y1, x0:x1, :]
