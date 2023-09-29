
scaling_method = 'crop+resize'

### PREPARATION
# pieces and grid
piece_size = 301
p_hs = piece_size // 2
xy_grid_points = 3
theta_grid_points = 4
comp_matrix_shape = [xy_grid_points, xy_grid_points, theta_grid_points]
pairwise_comp_range = 4 * (p_hs) + 1
canvas_size = pairwise_comp_range + 2 * p_hs + 1
xy_step = pairwise_comp_range / (comp_matrix_shape[0] - 1)
theta_step = (360 / comp_matrix_shape[2])

# preprocess
num_patches_side = 4
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

### LINE MATCHING                  ## OLD VALUES
thr_coef = 0.16
max_dist = 10
badmatch_penalty = 30    # piece_size/1
mismatch_penalty = 20
#mismatch_penalty_exp = 1
rmax = 11  # piece_size/30  # Norm

### initialization P
init_anc = 6
init_anc_rot = 0
nh = 4  # 2 for repair
nw = 4  # 2  # %% para to decide

### SOLVER
Tfirst = 200
Tnext = 200
Tmax = 7000
anc_fix_tresh = 0.75


# ### LINE MATCHING                  ## OLD VALUES
# thr_coef = 0.08                    ##   0.5
# max_dist = 30    # piece_size/1     ##   10
# mismatch_penalty = 10               ##   0
# mismatch_penalty_exp = 1
# rmax = 20   # piece_size/30  # Norm       ##  20 best
#
# ### initialization P
# init_anc = 6
# init_anc_rot = 0
# nh = 4  # 2 for repair
# nw = 4  # 2  # %% para to decide
#
# ### SOLVER
# Tfirst = 200
# Tnext = 200
# Tmax = 7000
# anc_fix_tresh = 0.55



