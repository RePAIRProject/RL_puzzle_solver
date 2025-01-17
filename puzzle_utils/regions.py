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
    neg_reg0 = RMs[0] < 0
    neg_reg1 = (RMs[2] - RMs[1]) < 0
    neg_reg = neg_reg0+neg_reg1

    m_m_all = RMs[2]
    m_m = m_m_all[:, :, 0, 1, 2]
    p_m_all = RMs[1]
    p_m = p_m_all[:, :, 0, 1, 2]

    neg_reg0_a = neg_reg0[:,:,0,1,2]
    neg_reg1_a = neg_reg1[:, :, 0, 1, 2]
    neg_reg_a = neg_reg[:, :, 0, 1, 2]

    combined_pos = RMs[0] * (RMs[0] > 0).astype(int)
    for i in range(2, len(RMs)):
        combined_pos *= RMs[i] * (RMs[i] > 0).astype(int)
    combined = combined_pos - neg_reg
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