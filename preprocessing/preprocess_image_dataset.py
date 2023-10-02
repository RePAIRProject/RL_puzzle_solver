import cv2 
import pdb 
from configs import folder_names as fnames 
from configs import puzzle_from_image_cfg_8 as cfg
import os
import argparse 
import numpy as np 

def main(args):

    images = os.path.join(fnames.data_path, args.dataset, fnames.images_folder)
    puzzle_image_folder = os.path.join(f"{fnames.output_dir}_{cfg.num_patches_side}x{cfg.num_patches_side}", args.dataset) #, fnames.pieces_folder)
    for image_name in os.listdir(images):
        image = cv2.imread(os.path.join(fnames.data_path, args.dataset, fnames.images_folder, image_name))
        if cfg.scaling_method == 'resize':
            image = cv2.resize(image, (cfg.img_size, cfg.img_size))
        elif cfg.scaling_method == 'crop':
            ihs = cfg.img_size // 2 
            if np.any(image.shape[:2] < cfg.img_size):
                print("ERROR! image is too small!")
                return
            center = image.shape
            xc = center[1]//2 
            yc = center[0]//2 
            image = image[yc-ihs:yc+ihs, xc-ihs:xc+ihs, :]
        elif cfg.scaling_method == 'crop+resize':
            min_dim = np.minimum(image.shape[0], image.shape[1])
            mdhs = min_dim // 2 
            center = image.shape
            xc = center[1]//2 
            yc = center[0]//2 
            image = image[yc-mdhs:yc+mdhs, xc-mdhs:xc+mdhs, :]
            image = cv2.resize(image, (cfg.img_size, cfg.img_size))

        num_patches_side = cfg.num_patches_side
        patch_size = image.shape[0] // num_patches_side
        pieces_single_folder = os.path.join(puzzle_image_folder, image_name[:-4], fnames.pieces_folder)
        os.makedirs(pieces_single_folder, exist_ok=True)
        k = 0
        for i in range(num_patches_side):
            for j in range(num_patches_side):
                patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size,:]
                cv2.imwrite(os.path.join(pieces_single_folder, f"piece_{k:05d}.png"), patch)
                k += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create patches from image')
    parser.add_argument('-d', '--dataset', type=str, default='architecture', help='dataset to work on', choices=['architecture', 'wikiart', 'shapes', 'manual_lines'])
    parser.add_argument('--e', '--edgemaps', action='store_true', default=False, help='use if edge maps are available')
    args = parser.parse_args()
    main(args)