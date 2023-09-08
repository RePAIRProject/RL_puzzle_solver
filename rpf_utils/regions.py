import os 
import numpy as np 
import scipy
import pdb 

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