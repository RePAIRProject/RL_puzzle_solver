import numpy as np 

# pieces and grid
piece_size = 251
p_hs = piece_size // 2
xy_grid_points = 101
theta_grid_points = 24
comp_matrix_shape = [xy_grid_points, xy_grid_points, theta_grid_points]
pairwise_comp_range = 2 * (p_hs) + 1 
canvas_size = pairwise_comp_range + 2 * p_hs + 1 
xy_step = pairwise_comp_range / (comp_matrix_shape[0] - 1)
theta_step = (360 / comp_matrix_shape[2])

# folders
data_path = 'data'
output_dir = 'output'
cm_output_dir = 'cm'
imgs_folder = 'images'

# output
save_visualization = True

# for compatibility
max_dist_between_pieces = p_hs
overlap_tolerance = 0.05
empty_space_tolerance = 0.35
min_axis_factor = 0.35 # magic number :( for ellipsoid 
sigma = xy_step / 2 # for the shape based compatibility (sigma of the exponential)
dist = 'bd'

# RETURN VALUES 
CENTER = -2
OVERLAP = -1
FAR_AWAY = 0
NOT_MATCHING = 0.001
