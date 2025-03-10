import numpy as np 
from puzzle_utils.shape_utils import get_borders_around, place_on_canvas
import cv2 
from PIL import Image

def compute_pairwise_shape_based_RM(piece_i, piece_j, ppars, dilate=True, erode=True):

    RM = np.zeros((ppars.xy_grid_points, ppars.xy_grid_points, ppars.theta_grid_points))
    center_pos = ppars.canvas_size // 2
    piece_i_on_canvas = place_on_canvas(piece_i, (center_pos, center_pos), ppars.canvas_size, 0)
    for t in range(ppars.theta_grid_points):
        piece_j_on_canvas = place_on_canvas(piece_j, (center_pos, center_pos), ppars.canvas_size, t * ppars.theta_step)
        # SHAPE case - BASIC
        overlap_shapes = cv2.filter2D(piece_i_on_canvas['mask'], -1, piece_j_on_canvas['mask'])
        thresholded_regions_map = (overlap_shapes > ppars.threshold_overlap).astype(np.int32)
        
        if dilate == True:
            border_dilation = int(ppars.borders_regions_width_outside * ppars.xy_step)
        else:
            border_dilation = 1
        if erode == True:
            border_erosion = int(ppars.borders_regions_width_inside * ppars.xy_step)
        else:
            border_erosion = 1

        around_borders_trm = get_borders_around(thresholded_regions_map.astype(np.uint8),
                                            border_dilation=border_dilation, border_erosion=border_erosion)
        thresholded_regions_map *= -1
        thresholded_regions_map += 2 * (around_borders_trm > 0)
        thresholded_regions_map = np.clip(thresholded_regions_map, -1, 1)

        # we convert the matrix to resize the image without losing the values
        thr_reg_map_shape_uint = (thresholded_regions_map + 1).astype(np.uint8)
        thr_reg_map_comp_range = thr_reg_map_shape_uint[ppars.p_hs + 1:-(ppars.p_hs + 1), ppars.p_hs + 1:-(ppars.p_hs + 1)]
        resized_shape = np.array(Image.fromarray(thr_reg_map_comp_range).resize((ppars.xy_grid_points, ppars.xy_grid_points), Image.Resampling.NEAREST))
        RM[:,:,t] = (resized_shape.astype(np.int32) - 1)
    return RM