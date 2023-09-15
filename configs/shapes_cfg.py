
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
border_tolerance = piece_size // 10