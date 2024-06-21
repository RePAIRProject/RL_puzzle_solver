import os 
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from configs import folder_names as fnames
from puzzle_utils.shape_utils import get_sd, get_cm, shift_img, get_polygon
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

    folders_list = os.listdir(args.dataset)
    # temporary for repair dataset
    folders_list = [f for f in folders_list if f.startswith('RPobj')]
    folders_list.sort()
    
    # image szie
    target_img_shape = args.image_size

    # name 
    if args.name == '':
        args.name = args.dataset.split('/')[-1]

    output_dataset_folder = os.path.join(os.getcwd(), fnames.output_dir, args.name)
    os.makedirs(output_dataset_folder, exist_ok=True)
    for folder in folders_list:
        output_puzzle_folder = os.path.join(output_dataset_folder, folder)
        os.makedirs(output_puzzle_folder, exist_ok=True)
        # here the subfolders!
        pieces_subfolder = os.path.join(output_puzzle_folder, 'pieces')
        os.makedirs(pieces_subfolder, exist_ok=True)
        masks_subfolder = os.path.join(output_puzzle_folder, 'masks')
        os.makedirs(masks_subfolder, exist_ok=True)
        polygons_subfolder = os.path.join(output_puzzle_folder, 'polygon')
        os.makedirs(polygons_subfolder, exist_ok=True)
        # get for each piece
        pieces_folder_path = os.path.join(args.dataset, folder)
        pieces_paths = os.listdir(pieces_folder_path)
        for piece_path in pieces_paths:
            piece_full_path = os.path.join(pieces_folder_path, piece_path)
            print(piece_full_path)
            imgcv = cv2.imread(piece_full_path, cv2.IMREAD_UNCHANGED)
            # plt.subplot(131)
            # plt.title('original')
            # plt.imshow(cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB))
            if imgcv.shape[0] != target_img_shape:
                imgcv = cv2.resize(imgcv, dsize=(target_img_shape, target_img_shape), interpolation=cv2.INTER_CUBIC)
            centered_img, img_mask = center_fragment(imgcv)
            polygon = get_polygon(img_mask)

            piece_name = piece_path[:9]
            target_path = os.path.join(pieces_subfolder, f"{piece_name}.png")
            target_path_mask = os.path.join(masks_subfolder, f"{piece_name}.png")
            # np.save(os.path.join(polygons_subfolder, piece_name), polygon)

            # plt.subplot(132)
            # plt.title('centered')
            # plt.imshow(cv2.cvtColor(centered_img, cv2.COLOR_BGR2RGB))

            # plt.subplot(133)
            # plt.title('mask (green polygon shape)')
            # plt.imshow(img_mask, cmap='jet')
            # plt.plot(*polygon.boundary.xy, color='green', linewidth=5)
            
            # plt.show()
            # breakpoint()
            
            cv2.imwrite(target_path_mask, img_mask*255)
            cv2.imwrite(target_path, centered_img)
            np.save(os.path.join(polygons_subfolder, piece_name), polygon)
            print(f'saved piece {piece_name} in', pieces_subfolder)

   
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare the data')
    parser.add_argument('-d', '--dataset', type=str, default='/media/lucap/big_data/datasets/repair/2D_dataset', help='data folder')
    parser.add_argument('-n', '--name', type=str, default='repair2D_v2_512px', help='dataset name')
    parser.add_argument('-is', '--image_size', type=int, default=512)
    args = parser.parse_args()
    main(args)