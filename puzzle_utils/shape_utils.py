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
import shapely
from shapely import transform
from shapely.affinity import rotate
import json
from puzzle_utils.lines_ops import draw_lines

def get_polygon(binary_image):
    bin_img = binary_image.copy()
    bin_img = cv2.dilate(bin_img.astype(np.uint8), np.ones((2,2)), iterations=1)
    contours, _ = cv2.findContours(bin_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_points = contours[0]
    shapely_points = [(point[0][0], point[0][1]) for point in contour_points]  # Shapely expects points in the format (x, y)
    if len(shapely_points) < 4:
        print('we have a problem, too few points', shapely_points)
        raise ValueError('\nWe have fewer than 4 points on the polygon, so we cannot create a Shapely polygon out of this points! Maybe something went wrong with the mask?')
    polygon = shapely.Polygon(shapely_points)
    return polygon

def create_grid(grid_size, padding, canvas_size):
    axis_grid = np.linspace(padding, canvas_size - padding - 1, grid_size)
    grid_step_size = axis_grid[1] - axis_grid[0]
    pieces_grid = np.zeros((grid_size, grid_size, 2))
    for b in range(len(axis_grid)):
        for g in range(len(axis_grid)):
            pieces_grid[g, b] = (axis_grid[g], axis_grid[b])
    return pieces_grid, grid_step_size

def place_on_canvas(piece, coords, canvas_size, theta=0):
    ## TODO:
    # check keys in piece because we may forget something half-way
    # for example `lines_mask` ?

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
        piece_img = scipy.ndimage.rotate(piece_img, theta, reshape=False, mode='constant')
        piece_mask = scipy.ndimage.rotate(piece_mask, theta, reshape=False, mode='constant', prefilter=False)
        piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_CLOSE, closing_kernel)
        #piece_mask = (piece_mask > eps_mh).astype(np.uint8)
        if 'sdf' in piece.keys():
            piece_sdf = scipy.ndimage.rotate(piece_sdf, theta, reshape=False, mode='constant')
        if 'lines_mask' in piece.keys():
            piece_lines_mask = scipy.ndimage.rotate(piece_lines_mask, theta, reshape=False, mode='constant', prefilter=False)
            piece_lines_mask = cv2.morphologyEx(piece_lines_mask, cv2.MORPH_CLOSE, closing_kernel)
            #piece_lines_mask = (piece_lines_mask > eps_mh).astype(np.uint8)
        piece['cm'] = get_cm(piece_mask)
        ## NEW MOTIF-BASED
        if 'motif_mask' in piece.keys():
            piece_motif_mask = scipy.ndimage.rotate(piece_motif_mask, theta, reshape=False, mode='constant')
        #piece['cm'] = get_cm(piece_mask)
        if 'polygon' in piece.keys():
            piece['polygon'] = rotate(piece['polygon'], -theta, origin=half_piece_shift)
    
    if 'polygon' in piece.keys():
        poly_on_canvas = transform(piece['polygon'], lambda f: f + [x,y] - half_piece_shift)

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


def get_mask(img, background=0, noisy=False, epsilon=0.1):

    if img.shape[2] == 4:
        img = img[:,:,3]
    else:
        img = img[:,:,0]
    if noisy == True:
        mask = img > epsilon*np.max(img)
    else:
        mask = 1 - (img == background).astype(np.uint8)
    return mask

def get_sd(img, background=0):
    if img.shape[2] == 4:
        mask = 1 - (img[:,:,3] == background).astype(np.uint8)
    else:
        mask = 1 - (img[:,:,0] == background).astype(np.uint8)
    phi = np.int64(mask[:, :])
    phi = np.where(phi, 0, -1) + 0.5
    sd = skfmm.distance(phi, dx = 1)
    return sd, mask 

def mask2sdf(mask, q=1):
    phi = np.int64(mask[:, :])
    phi = np.where(phi, 0, -1) + 0.5
    sdf = skfmm.distance(phi, dx = 1)
    if q > 1: #quantize (stepwise sdf)
        sdf = (sdf // q) * q
    return sdf

def get_outside_borders(mask, borders_width=3):
    """
    Get the borders outside of the mask contour (borders_width) 
    """
    kernel_size = borders_width*2+1
    kernel = np.ones((kernel_size, kernel_size))
    dilated_mask = cv2.dilate(mask, kernel)
    return dilated_mask - mask 

def get_borders_around(mask, border_dilation=3, border_erosion=3):
    """
    Get the borders around the mask contour (border_erosion outside, border_dilation inside) 
    """
    kernel_dilation = np.ones((border_dilation, border_dilation))
    kernel_erosion = np.ones((border_erosion, border_erosion))
    dilated_mask = cv2.dilate(mask, kernel_dilation)
    eroded_mask = cv2.erode(mask, kernel_erosion)
    return dilated_mask - eroded_mask

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
    sdf_sum = np.square(piece_i['sdf'] + piece_j['sdf'])
    dissim_score = np.sum(sdf_sum * mregion_mask.astype(float) / normalization_factor)
    comp_score = np.exp(-(dissim_score / sigma))
    return comp_score

def get_borders(piece, width=5):
    mask = piece['mask']
    kernel = np.ones((width*2+1, width*2+1))
    eroded_mask = cv2.erode(mask, kernel)
    borders = mask - eroded_mask
    return borders   

def approximate(xy_pt, step, offset):
    app_pt = np.zeros((2))
    for j in range(2):
        ptj = xy_pt[j]-offset[j]
        if ptj % step == 0:
            app_pt[j] = np.round(xy_pt[j]).astype(int)
        else:
            nnn = np.round(ptj / step)
            app_pt[j] = (nnn * step + offset[j]).astype(int)
    return app_pt

def divide_segment(start, end, seg_len):
    subsegments = []
    #pdb.set_trace()
    if start[0] == end[0] and start[1] == end[1]:
        print("skip")
    if start[0] == end[0]:
        num_steps = np.abs(start[1] - end[1]) // seg_len + 1
        int_steps = np.linspace(start[1], end[1], num_steps.astype(int))
        for k in range(len(int_steps)-1):# in int_steps:
            sta_pt = np.asarray([start[0], int_steps[k]])
            end_pt = np.asarray([start[0], int_steps[k+1]])
            mid_pt = (sta_pt + end_pt) / 2
            subsegments.append({
                'start':sta_pt.tolist(), 
                'end':end_pt.tolist(),
                'middle':mid_pt.tolist()
            })
    elif start[1] == end[1]:
        num_steps = np.abs(start[0] - end[0]) // seg_len + 1
        int_steps = np.linspace(start[0], end[0], num_steps.astype(int))
        for k in range(len(int_steps)-1):# in int_steps:
            sta_pt = np.asarray([int_steps[k], start[1]])
            end_pt = np.asarray([int_steps[k+1], start[1]])
            mid_pt = (sta_pt + end_pt) / 2
            subsegments.append({
                'start':sta_pt.tolist(), 
                'end':end_pt.tolist(),
                'middle':mid_pt.tolist()
            })
    else:
        print("wrong!")
    # print("subsegments")
    # for subs in subsegments: print(subs['start'], subs['end'])
    return subsegments

def divide_boundaries_in_segments(poly, seg_len):

    xs, ys = poly.boundary.xy
    # outlier detecxtion by guglielmo
    offset = [xs[0] % seg_len, ys[0] % seg_len]
    segments = []
    for i in range(len(xs)-1):
        start = approximate([xs[i], ys[i]], seg_len, offset)
        end = approximate([xs[i+1], ys[i+1]], seg_len, offset)
        if np.abs(end[0] - start[0]) > seg_len or np.abs(end[1] - start[1]) > seg_len:
            subsegments = divide_segment(start, end, seg_len)
            for subseg in subsegments:
                #print(subseg)
                segments.append(subseg)
        # elif end[0] == start[0] and end[1] == start[1]:
        #     print("point, skip")
        elif end[0] != start[0] or end[1] != start[1]:
            middle_pt = (start + end) / 2
            subseg = {
                'start':start.tolist(), 
                'end':end.tolist(),
                'middle':middle_pt.tolist()
            }
            #print(subseg)
            segments.append(subseg)
    return segments

def add_colors(image, borders_segments, thickness):
    
    colorful_borders_segments = []
    for bs in borders_segments:
        #plt.subplot(131); plt.imshow(image)
        stt = np.asarray(bs['start']).astype(int)
        end = np.asarray(bs['end']).astype(int)
        #print("start", stt, "end", end)
        #pdb.set_trace()
        if np.isclose(stt[0], end[0]):
            if end[1] - stt[1] < 0:
                st = end[1]
                en = stt[1]
            else:
                st = stt[1]
                en = end[1]
            left_part = image[st:en, stt[0]-thickness:end[0]]
            # horizontal
            if np.sum(left_part) > 10:
                # good one
                 #scipy.ndimage.rotate(, 90)
                colors = scipy.ndimage.rotate(left_part, 180)
                # plt.subplot(132); plt.imshow(left_part); plt.title(f"left part ({st},{stt[0]} to {en})")
                # plt.subplot(133); plt.imshow(colors); plt.title("no rotation needed")
                # plt.show()
            else:
                right_part = image[st:en, stt[0]:thickness+end[0]]
                colors = right_part
                # plt.subplot(132); plt.imshow(right_part); plt.title(f"right part ({st},{stt[0]} to {en})")
                # plt.subplot(133); plt.imshow(colors); plt.title("rotated 180")
                # plt.show()
        else:
            if end[0] - stt[0] < 0:
                st = end[0]
                en = stt[0]
            else:
                st = stt[0]
                en = end[0]
            top_part = image[stt[1]-thickness:end[1], st:en]
            # vertical   
            if np.sum(top_part) > 10:
                colors = scipy.ndimage.rotate(top_part, -90)
                # plt.subplot(132); plt.imshow(top_part); plt.title(f"top part ({st} to {en})")
                # plt.subplot(133); plt.imshow(colors); plt.title("rotated -90")
                # plt.show()
            else:
                bottom_part = image[stt[1]:thickness+end[1], st:en] 
                colors = scipy.ndimage.rotate(bottom_part, 90)
                # plt.subplot(132); plt.imshow(bottom_part); plt.title(f"bottom part ({st} to {en})")
                # plt.subplot(133); plt.imshow(colors); plt.title("rotated 90")
                # plt.show()
        bs['colors'] = colors
        bs['mean_border_colors'] = np.mean(colors[:, 0, :], axis=0)
        colorful_borders_segments.append(bs)
    return colorful_borders_segments

def encode_boundary_segments(pieces, fnames, dataset, puzzle, boundary_seg_len, boundary_thickness=2):
    for piece in pieces:
        #pdb.set_trace()
        borders_segments = divide_boundaries_in_segments(piece['polygon'], seg_len=boundary_seg_len)
        #borders_segments = piece['polygon'].tolist().segmentize(boundary_seg_len)
        borders_segments = add_colors(piece['img'], borders_segments, boundary_thickness)
        coords = [bs['start'] for bs in borders_segments]
        coords.append(borders_segments[-1]['end'])
        segmented_poly = shapely.Polygon(coords)
        piece['segmented_poly'] = segmented_poly
        piece['boundary_seg'] = borders_segments
    return pieces

def include_shape_info(fnames, pieces, dataset, puzzle, method, line_thickness=1, line_based=True, sdf=False, motif_based=True):

    root_folder = os.path.join(fnames.output_dir, dataset, puzzle)
    polygons_folder = os.path.join(root_folder, fnames.polygons_folder)
    polygons = os.listdir(polygons_folder)
    if line_based == True:
        lines_folder = os.path.join(root_folder, fnames.lines_output_name, method)
        lines_files = os.listdir(lines_folder)
        lines = [line for line in lines_files if line.endswith('.json')]
        assert len(polygons) == len(lines), f'Error: have {len(polygons)} polygons files and {len(lines)} lines files, they should have the same length!'

    ## NEW MOTIVE PART
    if motif_based == True:
        motif_folder = os.path.join(root_folder, fnames.motifs_output_name)
        # motif_folder = os.path.join(root_folder, fnames.motifs_output_name, method)  #TODO - add method to detection path !!! Yolo5 ect...
        motif_files = os.listdir(motif_folder)
        motif = [line for line in motif_files if line.endswith('.npy')]
        assert len(polygons) == len(motif), f'Error: have {len(polygons)} polygons files and {len(motif)} motif files, they should have the same length!'

    for piece in pieces:
        piece_name = piece['name']
        polygon_path = os.path.join(polygons_folder, f"{piece_name}.npy")

        piece['polygon'] = np.load(polygon_path, allow_pickle=True).tolist()
        if type(piece['polygon']) != shapely.Polygon:
            shapely_points = [(point[0], point[1]) for point in piece['polygon'][0]]
            piece['polygon'] = shapely.Polygon(shapely_points)

        #assert(type(np.load(polygon_path, allow_pickle=True).tolist()) == shapely.Polygon), "The polygon is not a shapely.Polygon! Check the files!"
        if line_based == True:
            lines_path = os.path.join(lines_folder, f"{piece_ID}.json")
            with open(lines_path, 'r') as file:
                piece['extracted_lines'] = json.load(file)
            drawn_lines = draw_lines(piece['extracted_lines'], piece['img'].shape, line_thickness, use_color=False)
            piece['lines_mask'] = drawn_lines

        if sdf == True:
            piece['sdf'] = mask2sdf(piece['mask'], q=1)

        ## NEW MOTIVE BASED
        if motif_based == True:
            motif_path = os.path.join(motif_folder, f'motifs_cube_{piece_name}.npy')
            motif_cube = np.load(motif_path)
            piece['motif_mask'] = motif_cube

    return pieces

def prepare_pieces_v2(fnames, dataset, puzzle_name, background=0, verbose=False):
    pieces = []
    root_folder = os.path.join(fnames.output_dir, dataset, puzzle_name)
    data_folder = os.path.join(root_folder, fnames.pieces_folder)
    masks_folder = os.path.join(root_folder, fnames.masks_folder)
    pieces_names = os.listdir(data_folder)
    pieces_names.sort()
    closing_kernel = np.ones((9, 9))
    if verbose is True:
        print(f"Found {len(pieces_names)} pieces:")           
    for piece_name in pieces_names:
        if verbose is True:
            print(f'- {piece_name}')
        piece_full_path = os.path.join(data_folder, piece_name)
        piece_d = {}
        img = cv2.imread(piece_full_path)
        piece_d['img'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_full_path = os.path.join(masks_folder, piece_name)
        mask = plt.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        if len(mask.shape) > 2:
            print("WARNING:Mask has multiple channels, using the first..")
            mask = mask[:,:,0]
        piece_d['mask'] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
        piece_d['cm'] = get_cm(piece_d['mask'])
        piece_d['id'] = piece_name[:10]  # piece_XXXXX.png
        piece_d['name'] = piece_name[:-4]  # piece_XXXXX.png
        if dataset=='repair':
            piece_d['id'] = piece_name[:9]   # RPf_00194.png
        pieces.append(piece_d)
    
    with open(os.path.join(os.getcwd(), root_folder, f'parameters_{puzzle_name}.json'), 'r') as pf:
        parameters = json.load(pf)

    return pieces, parameters

def prepare_pieces(cfg, fnames, dataset, puzzle_name, background=0):
    pieces = []
    data_folder = os.path.join(fnames.data_path, dataset, puzzle_name, fnames.imgs_folder)
    pieces_names = os.listdir(data_folder)
    pieces_names.sort()
    #pieces_full_path = [os.path.join(data_folder, piece_name) for piece_name in pieces_names]
    for piece_name in pieces_names:
        piece_full_path = os.path.join(data_folder, piece_name)
        piece_d = {}
        img = plt.imread(piece_full_path)
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

def process_region_map(region_map, perc_min=0.01):
    """
    It eliminates small regions and keep only the one who are "big" enough 
    """
    uvals = np.unique(region_map)
    rmap = np.zeros_like(region_map)
    rc = 1
    min_pixels = region_map.shape[0] * region_map.shape[1] * perc_min
    for uval in np.unique(region_map): 
        # print(f"region with value:{uval} has {np.sum(region_map==uval)} pixels")
        # plt.imshow(region_map==uval)
        # plt.show()
        if np.sum(region_map==uval) > min_pixels and uval > 0:
            rmap += (region_map==uval).astype(np.uint8) * rc
            rc += 1
        elif uval > 0:
            print("region too small! check threshold")

    # plt.subplot(121); plt.imshow(region_map, vmin=0, vmax=255)
    # plt.subplot(122); plt.imshow(rmap, vmin=0, vmax=31)
    # plt.show()
    # pdb.set_trace()
    return rmap, rc-1

def compute_SDF_cost_matrix(piece_i, piece_j, ids_to_score, ppars, verbosity=1):
    """ 
    It computes SDF-based cost matrix between piece_i and piece_j
    """
    p = ppars['p']
    alignment_grid = ppars['z_id']
    m = ppars['m']
    rot = ppars['rot']    
    R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))

    # grid on the canvas
    canv_cnt = ppars.canvas_size // 2
    grid = alignment_grid + canv_cnt #alignment_grid has negative values

    # TODO: move these to parameters?   improve?
    dilation_size = ppars['dilation_size'] #35
    dil_kernel = np.ones((dilation_size, dilation_size))
    sigma = ppars.p_hs
    for x,y,t in zip(ids_to_score[0], ids_to_score[1], ids_to_score[2]):
        theta = rot[t]
        center_pos = (len(grid) - 1 ) // 2
        x_c_pixel, y_c_pixel = grid[center_pos, center_pos]
        x_j_pixel, y_j_pixel = grid[y, x]
        piece_i_on_canvas = place_on_canvas(piece_i, (y_c_pixel, x_c_pixel), ppars.canvas_size, 0)
        piece_j_on_canvas = place_on_canvas(piece_j, (y_j_pixel, x_j_pixel), ppars.canvas_size, theta)
        #piece_i_on_canvas['mask'] = (piece_i_on_canvas['mask'] > 0.0005).astype(np.uint8)
        #piece_j_on_canvas['mask'] = (piece_j_on_canvas['mask'] > 0.0005).astype(np.uint8)
        dilated_pi_mask = cv2.dilate(piece_i_on_canvas['mask'], dil_kernel)
        dilated_pj_mask = cv2.dilate(piece_j_on_canvas['mask'], dil_kernel)
        inters_dilated_pi_mask_pj = ((dilated_pi_mask + piece_j_on_canvas['mask']) > 1).astype(np.uint8)      
        inters_dilated_pj_mask_pi = ((dilated_pj_mask + piece_i_on_canvas['mask']) > 1).astype(np.uint8)      
        touching_region = ((inters_dilated_pi_mask_pj + inters_dilated_pj_mask_pi) > 0).astype(np.uint8)
        touching_region = cv2.morphologyEx(touching_region, cv2.MORPH_CLOSE, dil_kernel)
        size_touching_region = np.sum(touching_region > 0)
        #print(f"We have {size_touching_region} pixels in the touching region")
        if size_touching_region < 2*ppars.p_hs:
            shape_score = 0
            #print(f"skipping as number of pixels < {2*ppars.p_hs}!")
        #touching_region = ((dilated_pi_mask + dilated_pj_mask) > 1).astype(np.uint8)

        # plt.subplot(231); plt.imshow(dilated_pi_mask)
        # plt.subplot(234); plt.imshow(dilated_pj_mask)
        # plt.subplot(232); plt.imshow(inters_dilated_pi_mask_pj)
        # plt.subplot(235); plt.imshow(dilated_pj_mask)
        # plt.subplot(233); plt.imshow((inters_dilated_pi_mask_pj + inters_dilated_pj_mask_pi))
        # plt.subplot(236); plt.imshow(touching_region)
        # plt.show()
        # breakpoint()
        # min_axis = min_axis_factor * np.minimum(np.sqrt(np.sum(piece_i['mask'])), np.sqrt(np.sum(piece_j['mask']))) # MAGIC NUMBER :/
        # drawn_matching_region, mregion_mask = get_ellipsoid(piece_i_on_canvas['cm'], piece_j_on_canvas['cm'], min_axis, ppars.canvas_size)
        else:
            shape_score = compute_shape_score(piece_i_on_canvas, piece_j_on_canvas, touching_region, sigma=sigma)
            # plt.subplot(251); plt.imshow(piece_i_on_canvas['img']); plt.title("image piece i")
            # plt.subplot(256); plt.imshow(piece_j_on_canvas['img']); plt.title("image piece j")
            # plt.subplot(252); plt.imshow(piece_i_on_canvas['mask']); plt.title("mask piece i")
            # plt.subplot(257); plt.imshow(piece_j_on_canvas['mask']); plt.title("mask piece j")
            # plt.subplot(253); plt.imshow(piece_i_on_canvas['sdf']); plt.title("sdf piece i")
            # plt.subplot(258); plt.imshow(piece_j_on_canvas['sdf']); plt.title("sdf piece j")
            # sdf_sum = piece_i_on_canvas['sdf'] + piece_j_on_canvas['sdf']
            # plt.subplot(254); plt.imshow(touching_region); plt.title("touching_region")
            # touching_region_size = np.sum(touching_region > 0)
            # sdf_val = np.sum(touching_region*np.square(sdf_sum)) / touching_region_size
            # plt.subplot(259); plt.imshow(touching_region*np.square(sdf_sum), cmap='jet'); plt.title(f"SDF ellipsoid region (val={sdf_val:.03f})")
            # reassembled_image = piece_i_on_canvas['img']+piece_j_on_canvas['img']
            # plt.subplot(255); plt.imshow(reassembled_image); plt.title(f"reassembled image (score={shape_score:.03f})")
            # plt.subplot(2,5,10); plt.imshow(reassembled_image * np.dstack((touching_region, touching_region, touching_region))); plt.title("reassembled image ellipsoid region")
            # plt.suptitle(f"T: ({x},{y},{t}) # SDF-val: {sdf_val:.03f} # Score: {shape_score:.03f}")
            # plt.show()
        #
        R_cost[x,y,t] = shape_score
    return R_cost
