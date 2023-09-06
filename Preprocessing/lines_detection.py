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
from skimage.transform import hough_line
from matplotlib import cm
import math 
from rpf_utils.lines_ops import hough_line, cluster_lines_dbscan, draw_hough_lines, \
    draw_prob_hough_line, display_unprocessed_hough_result, line_cart2pol, polar2cartesian


def main(args):

    # input
    images_folder = os.path.join(cfg.data_path, args.puzzle, cfg.imgs_folder)
    ###
    # there was an idea to use the mask (see below) but not used at the moment
    # masks_folder = os.path.join(cfg.data_path, args.puzzle, cfg.masks_folder)

    # stripe 
    segmentation_folder = os.path.join(cfg.output_dir, args.puzzle, cfg.segm_output_name, cfg.lines_segm_name)
    output_folder = os.path.join(cfg.output_dir, args.puzzle, cfg.lines_output_name)
    hough_output = os.path.join(output_folder, 'Hough')
    fld_output = os.path.join(output_folder, 'FLD')
    vis_output = os.path.join(output_folder, 'visualization')
    os.makedirs(hough_output, exist_ok=True)
    os.makedirs(fld_output, exist_ok=True)
    os.makedirs(vis_output, exist_ok=True)

    imgs_names = [img_name for img_name in os.listdir(images_folder)]

    for img_name in imgs_names:
        
        rgb_img_path = os.path.join(images_folder, img_name)
        rgb_image = cv2.imread(rgb_img_path)
        seg_img_path = os.path.join(segmentation_folder, f"lines_{img_name}")
        seg_img = cv2.imread(seg_img_path, cv2.IMREAD_GRAYSCALE)
        
        ###
        # there was an idea to use the mask (see below) but not used at the moment
        # mask_img_path = os.path.join(masks_folder, f"{img_name[:-4]}_mask.png")
        # mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

        hough_final_lines = {}
        fld_final_lines = {}

        # if there are lines!
        if np.max(seg_img) > 0:
            
            seg_contours = cv2.findContours(seg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            scnts = imutils.grab_contours(seg_contours)
            seg_cnt_draw = cv2.drawContours(np.zeros((251, 251), dtype=np.uint8), scnts, -1, (255), 1)
            
            ###
            # The idea here was "removing" the contours from the line. But not really usefl
            ###
            # mask_contours = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # mcnts = imutils.grab_contours(mask_contours)
            # mask_cnt_draw = cv2.drawContours(np.zeros((251, 251), dtype=np.uint8), mcnts, -1, (255), 1)
            # lines_draw = (seg_cnt_draw - mask_cnt_draw*seg_cnt_draw) > 0
            ###

            ## SKLEARN HOUGH
            k = cfg.k
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, cfg.hough_angular_range)
            # Perform Hough Transformation to change x, y, to h, theta, dist space.
            h, theta, d = hough_line(seg_cnt_draw, tested_angles)
            accumulator_threshold = k * h.max()
            indices = np.argwhere(h > accumulator_threshold)
            angles, dists = theta[indices[:, 1]], d[indices[:, 0]]
            
            # FLD
            FLD = cv2.ximgproc.createFastLineDetector(length_threshold=cfg.length_threshold, \
                                distance_threshold=cfg.distance_threshold, do_merge=cfg.do_merge)
            lines_fld = FLD.detect((seg_img*255).astype(np.uint8)).tolist()

            # 1) cluster hough lines
            lines_h = cluster_lines_dbscan(image=seg_img, angles=angles, dists=dists, epsilon=0.01, min_samples=1)
            angles_rec = []
            dists_rec = []
            for line_h in lines_h:
                rhop, thetap = line_cart2pol(line_h[0:2], line_h[2:4])
                angles_rec.append(thetap)
                dists_rec.append(rhop)
            
            # 2) lines from FLD 
            angles_fld = []
            dists_fld = []
            for line_fld in lines_fld:
                line_fld = np.squeeze(line_fld[0])
                rhofld, thetafld = line_cart2pol(line_fld[0:2], line_fld[2:4])
                angles_fld.append(thetafld)
                dists_fld.append(rhofld)
                
            hough_final_lines['angles'] = angles_rec
            hough_final_lines['dists'] = dists_rec
            fld_final_lines['angles'] = angles_fld
            fld_final_lines['dists'] = dists_fld
            
        
        plt.figure(figsize=(32,12))
        plt.subplot(241)
        plt.title('RGB')
        plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))

        plt.subplot(245)
        plt.title('Segmentation')
        plt.imshow(seg_img, cmap='gray')
        
        if np.max(seg_img) > 0:

            # CARTESIAN 
            plt.subplot(243)
            plt.title(f'HOUGH CLUSTERED (CARTESIAN, {len(angles_rec)} lines)')
            plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        
            for line2draw in lines_h:
                plt.axline((line2draw[0], line2draw[1]), (line2draw[2], line2draw[3]))

            plt.subplot(244)
            plt.title(f'FLD (CARTESIAN, {len(angles_fld)} segments)')
            plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))

            for line2draw in lines_fld:
                line2draw = np.squeeze(line2draw)
                plt.axline((line2draw[0], line2draw[1]), (line2draw[2], line2draw[3]))


            # POLAR
            plt.subplot(246)
            plt.title(f'HOUGH PLAIN (POLAR, {len(angles)} lines)')
            plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))

            lines_hough_cart = polar2cartesian(rgb_image, angles, dists)
            for line2draw in lines_hough_cart:
                plt.axline((line2draw[0], line2draw[1]), (line2draw[2], line2draw[3]))

            plt.subplot(247)
            plt.title(f'HOUGH CLUSTERED (POLAR, {len(angles_rec)} lines)')
            plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        
            lines_hough_clustered_cart = polar2cartesian(rgb_image, angles_rec, dists_rec)
            for line2draw in lines_hough_clustered_cart:
                plt.axline((line2draw[0], line2draw[1]), (line2draw[2], line2draw[3]))

            plt.subplot(248)
            plt.title(f'FLD (POLAR, {len(angles_fld)} segments)')
            plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))

            lines_fld_cart = polar2cartesian(rgb_image, angles_fld, dists_fld)
            for line2draw in lines_fld_cart:
                plt.axline((line2draw[0], line2draw[1]), (line2draw[2], line2draw[3]))

        fig_path = os.path.join(vis_output, f"{img_name[:-4]}.png")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

        with open(os.path.join(hough_output, f"{img_name[:-4]}_hough.json"), 'w') as lj:
            json.dump(hough_final_lines, lj, indent=3)
        
        with open(os.path.join(fld_output, f"{img_name[:-4]}_fld.json"), 'w') as lj:
            json.dump(fld_final_lines, lj, indent=3)

        print('saved lines from', img_name[:-4])

    return 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract lines from segmented motifs')
    parser.add_argument('--puzzle', type=str, default='repair_g28', help='puzzle folder')

    args = parser.parse_args()
    main(args)