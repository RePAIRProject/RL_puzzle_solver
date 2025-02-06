import os 
import numpy as np 
import scipy
import pdb 

def combine_region_masks(RMs):

    neg_reg = RMs[0] < 0
    combined_pos = RMs[0] * (RMs[0] > 0).astype(int)
    for i in range(1, len(RMs)):
        combined_pos *= RMs[i] * (RMs[i] > 0).astype(int)
    combined = combined_pos - neg_reg
    return combined

def combine_region_masks_V2(RMs):
    shape = RMs[0]      #[:,:,0,8,9]
    poly_motif = RMs[1] #[:,:,0,8,9,8]
    motifs = RMs[2]     #[:,:,0,8,9,8]

    shape_neg = shape < 0
    shape_pos = shape * (shape > 0).astype(int)

    poly_neg = ((motifs - poly_motif) < 0).astype(int)
    #poly_neg_combined  = shape_pos * poly_neg ## negative values to add
    motifs_pos = motifs * (motifs > 0).astype(int)

    neg_reg = np.zeros(np.shape(motifs))
    combined_motifs = np.zeros(np.shape(motifs))

    if len(np.shape(motifs)) != len(np.shape(shape)):
        n_motifs = np.shape(motifs)[-1]
        for mt in range(n_motifs):
            combined_motifs[:,:,:,:,:,mt] = shape_pos * motifs_pos[:,:,:,:,:,mt] ## store "useful" motif-motif intersection
            combined_poly_neg = shape_pos * poly_neg[:, :, :, :, :, mt]  ## negative values to add
            neg_reg[:, :, :, :, :, mt] = shape_neg + combined_poly_neg   ## store all negative (motif-poly and poly-poly overlap)
    else:
        neg_reg = shape_neg + poly_neg
        combined_motifs = shape_pos + motifs_pos

    combined = (combined_motifs - neg_reg).astype(int)  ## add negative to RM

    import matplotlib.pyplot as plt
    plt.subplot(241)
    plt.title("shape positive")
    plt.imshow(shape_pos[:,:,0,8,9])
    plt.subplot(242)
    plt.title("motif-motif positive")
    plt.imshow(motifs[:,:,0,8,9,8])
    plt.subplot(243)
    plt.title("motif+shape positive")
    plt.imshow(combined_motifs[:,:,0,8,9,8])
    plt.subplot(244)
    plt.title("poly-motif intersection")
    plt.imshow(poly_motif[:,:,0,8,9,8])
    plt.subplot(245)
    plt.title("shape negative")
    plt.imshow(shape_neg[:,:,0,8,9])
    plt.subplot(246)
    plt.title("poly-motif negative")
    plt.imshow(poly_neg[:,:,0,8,9,8])
    plt.subplot(247)
    plt.title("shape+motif negative")
    plt.imshow(neg_reg[:,:,0,8,9,8])
    plt.subplot(248)
    plt.title("final RM")
    plt.imshow(combined[:,:,0,8,9,8])

    return combined

def read_region_masks(pzl_cfg, pzl_name, mat_file_path=''):

    if len(mat_file_path) == 0:
        rm_folder = os.path.join(os.getcwd(), pzl_cfg.output_dir, pzl_name, pzl_cfg.rm_output_name)
        print('no mat file path provided, looking into the output folder:\n', rm_folder)
        if not os.path.exists(rm_folder):
            print("\nERROR: There are no regions matrix! Compute them with the compute_regions_masks.py script!\n")  
            return -3, None

        mat_files = [rm_file for rm_file in os.listdir(rm_folder) if rm_file.endswith('.mat')]
        if len(mat_files) == 0:
            print("\nERROR: There are no regions matrix! Compute them with the compute_regions_masks.py script!\n")  
            return -2, None  
        if len(mat_files) == 1:
            mat_file_path = os.path.join(rm_folder, mat_files[0])
            print('found', mat_file_path)
        else:
            print("\nWARNING: There are several matrices for regions! Which one should be used? Re-run specifying the path!\nThe matrices are:")
            for j, mat_file in enumerate(mat_files):
                print(f"{j:02d}: {mat_file}")
            return -1, None
    
    regions_matrix = scipy.io.loadmat(mat_file_path)
    return 0, regions_matrix['RM']