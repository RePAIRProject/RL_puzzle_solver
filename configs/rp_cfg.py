import numpy as np 

piece_size = 251
p_hs = piece_size // 2
comp_matrix_shape = [101, 101, 24]
pairwise_comp_range = 2 * (piece_size - 1) 
canvas_size = pairwise_comp_range + piece_size
xy_step = pairwise_comp_range / (comp_matrix_shape[0] - 1)
theta_step = np.deg2rad(360 / comp_matrix_shape[2])
sigma = 60
data_path = 'data/repair'
cm_output_dir = 'data/cm'
imgs_folder = 'images'