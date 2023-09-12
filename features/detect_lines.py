import cv2 
import os 
from configs import wikiart_cfg as cfg
from configs import folder_names as fnames
from puzzle_utils.lines_ops import polar2cartesian
import pdb 
import argparse
import numpy as np 
from skimage.transform import hough_line 
import matplotlib.pyplot as plt 

def main(args):

    # input
    images_folder = os.path.join(fnames.data_path, args.puzzle, fnames.imgs_folder)
    lines_output_folder = os.path.join(fnames.output_dir, args.puzzle, fnames.lines_output_name)
    vis_output = os.path.join(lines_output_folder, 'visualization')
    os.makedirs(vis_output, exist_ok=True)

    imgs_names = [img_name for img_name in os.listdir(images_folder)]

    for img_name in imgs_names:
        
        rgb_img_path = os.path.join(images_folder, img_name)
        rgb_image = cv2.imread(rgb_img_path)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        if cfg.line_detection_method == 'hough':
            k = cfg.k
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, cfg.hough_angular_range)
            # Perform Hough Transformation to change x, y, to h, theta, dist space.
            h, theta, d = hough_line(gray_image, tested_angles)
            accumulator_threshold = k * h.max()
            indices = np.argwhere(h > accumulator_threshold)
            angles, dists = theta[indices[:, 1]], d[indices[:, 0]]

        # 1) cluster hough lines
        #lines_h = cluster_lines_dbscan(image=seg_img, angles=angles, dists=dists, epsilon=0.01, min_samples=1)
        lines_h = polar2cartesian(rgb_image, angles, dists, show_image=False)

        plt.title(f'HOUGH ({len(angles)} lines)')
        plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        for line2draw in lines_h:
            plt.axline((line2draw[0], line2draw[1]), (line2draw[2], line2draw[3]))        
        plt.show()        
        pdb.set_trace()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract lines from segmented motifs')
    parser.add_argument('--puzzle', type=str, default='wikiart_kuroda_4x4', help='puzzle folder')

    args = parser.parse_args()
    main(args)

        