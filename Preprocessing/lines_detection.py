import argparse
from ultralytics import YOLO
import os 
import cv2 
import numpy as np
import pdb 
import matplotlib.pyplot as plt
import configs.rp_cfg as cfg
import imutils
import json 

def main(args):

    # input
    images_folder = os.path.join(cfg.data_path, args.puzzle, cfg.imgs_folder)
    masks_folder = os.path.join(cfg.data_path, args.puzzle, cfg.masks_folder)

    # stripe 
    segmentation_folder = os.path.join(cfg.output_dir, args.puzzle, cfg.segm_output_name)
    output_folder = os.path.join(cfg.output_dir, args.puzzle, cfg.lines_output_name)
    os.makedirs(output_folder, exist_ok=True)

    imgs_names = [img_name for img_name in os.listdir(images_folder)]
    #lines_imgs = [img_name for img_name in os.listdir(segmentation_folder) if 'lines' in img_name]

    for img_name in imgs_names:
        
        seg_img_path = os.path.join(segmentation_folder, f"lines_{img_name}")
        seg_img = cv2.imread(seg_img_path, cv2.IMREAD_GRAYSCALE)
        mask_img_path = os.path.join(masks_folder, f"{img_name[:-4]}_mask.png")
        mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        # pdb.set_trace()
        # plt.subplot(121); plt.imshow(seg_img)
        # plt.subplot(122); plt.imshow(mask_img)
        # plt.show()
        # pdb.set_trace()

        lines = {}

        # if there are lines!
        if np.max(seg_img) > 0:
            
            seg_contours = cv2.findContours(seg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            scnts = imutils.grab_contours(seg_contours)
            seg_cnt_draw = cv2.drawContours(np.zeros((251, 251), dtype=np.uint8), scnts, -1, (255), 1)

            mask_contours = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mcnts = imutils.grab_contours(mask_contours)
            mask_cnt_draw = cv2.drawContours(np.zeros((251, 251), dtype=np.uint8), mcnts, -1, (255), 1)

            lines_draw = (seg_cnt_draw - mask_cnt_draw*seg_cnt_draw) > 0

            print("\n\n\n WARNING: THIS IS NOT FINISHED")
            plt.subplot(131)
            plt.imshow(seg_cnt_draw)
            plt.subplot(132)
            plt.imshow(mask_cnt_draw)
            plt.subplot(133)
            plt.imshow(lines_draw)
            plt.show()
            
            pdb.set_trace()

            #seg_contours = 

        # pdb.set_trace()
        with open(os.path.join(output_folder, f"{img_name[:-4]}.json"), 'w') as lj:
            json.dump(lines, lj, indent=3)

        print('saved lines from', img_name[:-4])

    return 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract lines from segmented motifs')
    parser.add_argument('--puzzle', type=str, default='', help='puzzle folder')

    args = parser.parse_args()
    main(args)