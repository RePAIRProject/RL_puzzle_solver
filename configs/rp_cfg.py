import numpy as np 

### PREPARATION 
# pieces and grid
piece_size = 251
p_hs = piece_size // 2
xy_grid_points = 101
theta_grid_points = 24
comp_matrix_shape = [xy_grid_points, xy_grid_points, theta_grid_points]
pairwise_comp_range = 4 * (p_hs) + 1
canvas_size = pairwise_comp_range + 2 * p_hs + 1 
xy_step = pairwise_comp_range / (comp_matrix_shape[0] - 1)
theta_step = (360 / comp_matrix_shape[2])

### LINE DETECTION
# HOUGH
k = 0.8 # accumulator ratio (80%)
hough_angular_range = 180
# FLD 
length_threshold=25
distance_threshold=1.4
do_merge=True

### FOLDERS
import os
data_path = 'data'
output_dir = 'C:\\Users\\Marina\\PycharmProjects\\RL_puzzle_solver\\output'
cm_output_name ='CompatibilityMatrix'
rm_output_name = 'RegionsMatrix'
segm_output_name = 'MotifSegmentation'
lines_segm_name = 'Lines'
motifs_segm_name = 'Motif'
lines_output_name = 'LinesDetection'
visualization_folder_name = 'visualization'
imgs_folder = 'images'
masks_folder = 'masks'

### VISUALIZATION
save_visualization = True

### COMPATIBILITY 
max_dist_between_pieces = p_hs
overlap_tolerance = 0.05
empty_space_tolerance = 0.35
threshold_overlap = 10
borders_regions_width = 3
min_axis_factor = 0.35 # magic number :( for ellipsoid 
sigma = xy_step / 2 # for the shape based compatibility (sigma of the exponential)
dist = 'bd'

### RETURN VALUES 
CENTER = -2
OVERLAP = -1
FAR_AWAY = 0
NOT_MATCHING = 0.001

### LINE MATCHING
rmax = 50
tr_coef = 0.15
max_dist = 1000


