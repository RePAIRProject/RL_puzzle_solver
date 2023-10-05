""" 
This file just merges the various configuration files so that you can import only one thing
"""
# should folder names be here or we leave it as separate?
# from configs.folder_names import *

# matching lines (to calculate compatibility)
from configs.line_matching_cfg import * 

# pieces (size and how mnay)
from configs.pieces_cfg import *

# compatibility (ranges, values, grids, .. )
from configs.compatibility_cfg import * 

# solver (anchors, iterations, ..)
from configs.solver_cfg import *