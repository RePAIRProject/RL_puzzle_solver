piece_size = 151
# preprocess
num_patches_side = 16
img_size = num_patches_side*piece_size

# segments
border_tolerance = piece_size // 60
scaling_method = 'crop+resize'