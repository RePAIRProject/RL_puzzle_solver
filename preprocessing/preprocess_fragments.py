import os 
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from puzzle_utils.shape_utils import get_sd, get_cm, shift_img
import pdb
from configs import repair_cfg as cfg
from configs import folder_names as fnames
import argparse 

def center_fragment(image):
    #pdb.set_trace()
    sd, mask = get_sd(image)
    cm = get_cm(mask)
    center_pos = [np.round(image.shape[0]/2).astype(int), np.round(image.shape[1]/2).astype(int)]
    shift = np.round(np.array(cm) - center_pos).astype(int)
    centered_image = shift_img(image, -shift[0], -shift[1])    
    return centered_image, mask

def main(args):
    target_img_shape = 1501 #cfg.piece_size 
    groups = [97]
    db_folder = args.dataset
    groups_folders = [os.path.join(db_folder, f'group_{group}') for group in groups]
    output_root_folder = 'data'
    output_folders = [os.path.join(output_root_folder, f'repair_g{group}') for group in groups]
    for (group_folder, output_folder) in zip(groups_folders, output_folders):
        target_out_folder = os.path.join(output_folder, fnames.imgs_folder)
        target_out_folder_masks = os.path.join(output_folder, fnames.masks_folder)
        os.makedirs(target_out_folder, exist_ok=True)
        os.makedirs(target_out_folder_masks, exist_ok=True)
        for piece_path in os.listdir(group_folder):
            #if 'RPf' in piece_path:
            piece_full_path = os.path.join(group_folder, piece_path)
            imgcv = cv2.imread(piece_full_path, cv2.IMREAD_UNCHANGED)
            if imgcv.shape[0] != target_img_shape:
                imgcv = cv2.resize(imgcv, dsize=(target_img_shape, target_img_shape), interpolation=cv2.INTER_CUBIC)
            centered_img, img_mask = center_fragment(imgcv)
            target_path = os.path.join(target_out_folder, piece_path)
            target_path_mask = os.path.join(target_out_folder_masks, f"{piece_path[:-4]}_mask.png")
            #pdb.set_trace()
            cv2.imwrite(target_path_mask, img_mask*255)
            cv2.imwrite(target_path, centered_img)
            print('saved centered image in', target_path)
        #pdb.set_trace()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare the data')
    parser.add_argument('-d', '--dataset', type=str, default='/media/lucap/big_data/datasets/repair/puzzle2D', help='data folder')

    args = parser.parse_args()
    main(args)