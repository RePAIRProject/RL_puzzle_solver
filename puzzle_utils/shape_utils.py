import numpy as np 
import matplotlib.pyplot as plt 
import numpy as np 
import cv2 
import skfmm
import time 
import pickle
import scipy
import pdb
import os

def create_grid(grid_size, padding, canvas_size):
    axis_grid = np.linspace(padding, canvas_size - padding - 1, grid_size)
    grid_step_size = axis_grid[1] - axis_grid[0]
    pieces_grid = np.zeros((grid_size, grid_size, 2))
    for b in range(len(axis_grid)):
        for g in range(len(axis_grid)):
            pieces_grid[g, b] = (axis_grid[g], axis_grid[b])
    return pieces_grid, grid_step_size

def place_on_canvas(piece, coords, canvas_size, theta=0):
    y, x = coords
    hs = piece['img'].shape[0] // 2
    y_c0 = int(y-hs)
    y_c1 = int(y+hs)
    x_c0 = int(x-hs)
    x_c1 = int(x+hs)
    channels = piece['img'].shape[2]
    img_on_canvas = np.zeros((canvas_size, canvas_size, channels))
    msk_on_canvas = np.zeros((canvas_size, canvas_size))
    sdf_on_canvas = np.zeros((canvas_size, canvas_size))
    sdf_on_canvas += np.min(piece['sdf'])
    piece_img = piece['img']
    piece_mask = piece['mask']
    piece_sdf = piece['sdf']

    if theta > 0:
        piece_img = scipy.ndimage.rotate(piece_img, theta, reshape=False)
        piece_mask = scipy.ndimage.rotate(piece_mask, theta, reshape=False)
        piece_sdf = scipy.ndimage.rotate(piece_sdf, theta, reshape=False)
        piece['cm'] = get_cm(piece_mask)

    #print(y_c0, y_c1+1, x_c0, x_c1+1)
    img_on_canvas[y_c0:y_c1+1, x_c0:x_c1+1, :] = piece_img
    msk_on_canvas[y_c0:y_c1+1, x_c0:x_c1+1] = piece_mask
    sdf_on_canvas[y_c0:y_c1+1, x_c0:x_c1+1] = piece_sdf
    shift_y = y - piece['cm'][1]
    shift_x = x - piece['cm'][0]
    cm_on_canvas = [piece['cm'][0] + shift_x, piece['cm'][1] + shift_y]
    piece_on_canvas = {
        'img': img_on_canvas.astype(int),
        'mask': msk_on_canvas,
        'sdf': sdf_on_canvas,
        'cm': cm_on_canvas 
    }
    return piece_on_canvas

def get_sd(img, background=0):
    if img.shape[2] == 4:
        mask = 1 - (img[:,:,3] == background).astype(np.uint8)
    else:
        mask = 1 - (img[:,:,0] == background).astype(np.uint8)
    phi = np.int64(mask[:, :])
    phi = np.where(phi, 0, -1) + 0.5
    sd = skfmm.distance(phi, dx = 1)
    return sd, mask 

def get_outside_borders(mask, borders_width=3):
    kernel_size = borders_width*2+1
    kernel = np.ones((kernel_size, kernel_size))
    dilated_mask = cv2.dilate(mask, kernel)
    return dilated_mask - mask 

def shift_img(img, x, y):
    new_img = np.zeros_like(img)
    if x == 0 and y == 0:
        new_img = img
    if x >= 0 and y >= 0:
        new_img[y:, x:] = img[:img.shape[0]-y, :img.shape[1]-x]
    elif x >= 0 and y < 0:
        new_img[:y, x:] = img[-y:, :img.shape[1]-x]
    elif x < 0 and y >= 0:
        new_img[y:, :x] = img[:img.shape[0]-y, -x:]
    elif x < 0 and y < 0:
        new_img[:y, :x] = img[-y:, -x:]
    return new_img 

def get_cm(mask):

    mass_y, mass_x = np.where(mask >= 0.5)
    cent_x = np.average(mass_x)
    cent_y = np.average(mass_y)
    # center = [ np.average(indices) for indices in np.where(th1 >= 255) ]
    return [cent_x, cent_y]

def get_ellipsoid(cm0, cm1, min_axis, img_shape, show=False):
    
    #min_axis -= 50 # DEBUG
    #print()
    #pdb.set_trace()
    diff_x = (cm1[0] - cm0[0])
    diff_y = (cm1[1] - cm0[1])
    #print('diff', diff_x, diff_y)
    center_coordinates = np.round((cm1[0] - diff_x / 2, cm1[1] - diff_y / 2)).astype(int)
    ###
    angle = np.rad2deg(np.arctan2(diff_y, diff_x))
    if diff_x < 0:
        angle += 180
    ###
    d_x = np.abs(diff_x)
    d_y = np.abs(diff_y)
    if d_x == d_y == 0:
        axesLength = np.round((min_axis, min_axis)).astype(int)
    else: 
        axesLength = np.round((np.maximum(d_x, d_y) / 2, min_axis)).astype(int)
    # if d_x > d_y:
    #     axesLength = np.round((d_x / 2, np.maximum(d_y / 2, min_axis))).astype(int)
    # elif d_y > d_x:
    #     axesLength = np.round((d_y / 2, np.maximum(d_x / 2, min_axis))).astype(int)
    # else: # same value 
    #     axesLength = np.round((d_x / 2, np.maximum(d_x / 4, min_axis))).astype(int)
    # print(d_x, d_y, min_axis, axesLength)
    startAngle = 0
    endAngle = 360
    color = (1, 0, 0)
    thickness = cv2.FILLED

    img_to_draw = np.zeros((img_shape, img_shape)).astype(np.uint8)
    ellip = cv2.ellipse(img_to_draw, center_coordinates, axesLength,
           angle, startAngle, endAngle, color, thickness)
    # this for the notebook (otherwise let it be false)
    if show:
        plt.imshow(ellip)
    return ellip, (ellip > 0).astype(float) 

def compute_shape_score(piece_i, piece_j, mregion_mask, sigma=1):

    # get ellipsoidal region
    normalization_factor = np.sum(mregion_mask > 0)
    # sdf sum 
    sdf_sum = np.abs(piece_i['sdf'] + piece_j['sdf'])
    dissim_score = np.sum(sdf_sum * mregion_mask.astype(float) / normalization_factor)
    comp_score = np.exp(-(dissim_score / sigma))
    return comp_score

def get_borders(piece, width=5):
    mask = piece['mask']
    kernel = np.ones((width*2+1, width*2+1))
    eroded_mask = cv2.erode(mask, kernel)
    borders = mask - eroded_mask
    return borders 

def prepare_pieces(cfg, fnames, puzzle_name, background=0):
    pieces = []
    data_folder = os.path.join(fnames.data_path, puzzle_name, fnames.imgs_folder)
    pieces_names = os.listdir(data_folder)
    #pieces_full_path = [os.path.join(data_folder, piece_name) for piece_name in pieces_names]
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
        piece_d['sdf'], piece_d['mask'] = get_sd(piece_d['img'])
        piece_d['cm'] = get_cm(piece_d['mask'])
        piece_d['id'] = piece_name[:9]
        pieces.append(piece_d)
    return pieces

def shape_pairwise_compatibility(piece_i, piece_j, x_j, y_j, theta_j, puzzle_cfg, grid, sigma=1):

    assert type(piece_i) == dict and type(piece_j) == dict, 'pieces should be dict with (img, sd, mask, cm) as keys'
    assert type(x_j) == int and type(y_j) == int, 'x_j and y_j should be integer position (relative to the grid)'

    # pixel coordinates of the two pieces (i is in the center)
    center_pos = (len(grid) - 1 ) // 2
    x_c_pixel, y_c_pixel = grid[center_pos, center_pos]
    x_j_pixel, y_j_pixel = grid[x_j, y_j]
    theta_degrees = theta_j * puzzle_cfg.theta_step

    # place the pieces on the canvas (for visualization purpose)
    piece_i_canvas = place_on_canvas(piece_i, (y_c_pixel, x_c_pixel), puzzle_cfg.canvas_size, 0)
    piece_j_canvas = place_on_canvas(piece_j, (y_j_pixel, x_j_pixel), puzzle_cfg.canvas_size, theta_degrees)
    
    # Get matching region
    min_axis = puzzle_cfg.min_axis_factor * np.minimum(np.sqrt(np.sum(piece_i['mask'])), np.sqrt(np.sum(piece_j['mask']))) # MAGIC NUMBER :/
    drawn_matching_region, mregion_mask = get_ellipsoid(piece_i_canvas['cm'], piece_j_canvas['cm'], min_axis, puzzle_cfg.canvas_size)

    # check if we have an 'easy' solution or we actually need to calculate
    # piece_j in the center means both at the same position, impossible
    if x_j == center_pos and y_j == center_pos:
        return puzzle_cfg.CENTER
    
    # if they are far apart, we save calculations and return fixed value
    if np.abs(x_j_pixel - x_c_pixel) > puzzle_cfg.max_dist_between_pieces + 1 or np.abs(y_j_pixel - y_c_pixel) > puzzle_cfg.max_dist_between_pieces + 1:
        #print(np.abs(x_j_pixel - x_c_pixel), ">", puzzle_cfg.max_dist_between_pieces+1)
        #print(np.abs(y_j_pixel - y_c_pixel), ">", puzzle_cfg.max_dist_between_pieces+1)
        return puzzle_cfg.FAR_AWAY

    # if they are far apart, we save calculations and return fixed value
    if np.sum(((1-piece_i_canvas['mask']) * (1-piece_j_canvas['mask'])) * mregion_mask) > puzzle_cfg.empty_space_tolerance * np.sum(mregion_mask):
        return puzzle_cfg.FAR_AWAY

    # if overlap exceeds the tolerance (see config)
    if np.sum(piece_i_canvas['mask'] + piece_j_canvas['mask'] > 1) > puzzle_cfg.overlap_tolerance * np.sum(mregion_mask):
        return puzzle_cfg.OVERLAP

    #pdb.set_trace()
    comp_shape = compute_shape_score(piece_i_canvas, piece_j_canvas, mregion_mask, sigma=60) #components[1]['sigma'])

    return comp_shape