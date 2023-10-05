import numpy as np
scaling_method = 'crop+resize'

### PREPARATION
# pieces and grid
piece_size = 151 # 301
p_hs = piece_size // 2
xy_grid_points = 3
theta_grid_points = 4
comp_matrix_shape = [xy_grid_points, xy_grid_points, theta_grid_points]
pairwise_comp_range = 4 * (p_hs) + 1
canvas_size = pairwise_comp_range + 2 * p_hs + 1
xy_step = pairwise_comp_range / (comp_matrix_shape[0] - 1)
theta_step = (360 / comp_matrix_shape[2])

# preprocess
num_patches_side = 8 # 4
img_size = num_patches_side*piece_size

## lines
line_detection_method = 'fld'
# hough
k = 0.8
hough_angular_range = 180
## fld
blur_kernel_size = 7
length_threshold = piece_size // 3
distance_threshold = 2
# segments
border_tolerance = piece_size // 60

### LINE MATCHING
thr_coef = 0.08
max_dist = 3
badmatch_penalty = 30
mismatch_penalty = 20
rmax = 11

### initialization P

#init_anc = []
init_anc = ((np.ceil(num_patches_side/2) - 1)*(num_patches_side+1)).astype(int)  # anchor central patch
init_anc_rot = 0
nh = 3  # 2 for repair
nw = 3  # 2 for repair

### SOLVER
Tfirst = 200
Tnext = 100
Tmax = 7000
anc_fix_tresh = 0.55



