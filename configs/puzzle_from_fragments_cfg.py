import numpy as np 

### PREPARATION 
# pieces and grid
piece_size = 1501
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
thr_coef = 0.08
max_dist = 3
badmatch_penalty = 30
mismatch_penalty = 20 #10
rmax = 15 #11
border_tolerance = piece_size // 60


