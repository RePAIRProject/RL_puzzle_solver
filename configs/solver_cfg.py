### initialization P
init_anc = ((np.ceil(num_patches_side/2) - 1)*(num_patches_side+1)).astype(int)  # anchor central patch
init_anc_rot = 0
nh = 3  # 2 for repair
nw = 3  # 2 for repair

### SOLVER
Tfirst = 200
Tnext = 100
Tmax = 7000
pert_noise = 1e-11
anc_fix_tresh = 0.55
