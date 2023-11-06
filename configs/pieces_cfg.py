
scaling_method = 'crop+resize'

### PREPARATION
# pieces and grid
piece_size = 151

# preprocess
num_patches_side = 16
img_size = num_patches_side*piece_size

# segments
border_tolerance = piece_size // 60
