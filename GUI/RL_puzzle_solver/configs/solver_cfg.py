
### initialization P
import numpy as np
from configs.pieces_cfg import num_patches_side
#init_anc = ((np.ceil(num_patches_side/2) - 1)*(num_patches_side+1)).astype(int)  # anchor central patch
#init_anc = 23
#init_anc_rot = 0
#nh = 3  # 2 for repair
#nw = 3  # 2 for repair

### SOLVER
Tfirst = 300
Tnext = 300
Tmax = 3000
anc_fix_tresh = 0.75
