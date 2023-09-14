import cv2 
import os 
from configs import wikiart_cfg as cfg
from configs import folder_names as fnames
from puzzle_utils.lines_ops import polar2cartesian, line_cart2pol
import pdb 
import argparse
import numpy as np 
from skimage.transform import hough_line 
import matplotlib.pyplot as plt 
import json 

def main(args):

    # input
    images_folder = os.path.join(fnames.data_path, args.puzzle, 'edge_maps') #fnames.imgs_folder)
    lines_output_folder = os.path.join(fnames.output_dir, args.puzzle, fnames.lines_output_name)
    vis_output = os.path.join(lines_output_folder, 'visualization')
    os.makedirs(vis_output, exist_ok=True)

    imgs_names = [img_name for img_name in os.listdir(images_folder)]

    for img_name in imgs_names:
        
        rgb_img_path = os.path.join(images_folder, img_name)
        rgb_image = cv2.imread(rgb_img_path)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        # method can be loaded either via commanbd line parameter 
        # or via config file
        if len(args.method) > 0:
            method = args.method
        else:
            method = cfg.line_detection_method
        if method == 'hough':
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
            plt.savefig(os.path.join(vis_output, f"{img_name[:-4]}_hough.jpg"))
            plt.close()
        
        if method == 'fld':
            # preprocessing
            img = cv2.medianBlur(gray_image, 11)
            FLD = cv2.ximgproc.createFastLineDetector(length_threshold=cfg.length_threshold) #, distance_threshold=0.5)
            lines_fld = FLD.detect((img).astype(np.uint8)) 
            angles_fld = []
            dists_fld = []
            p1s_fld = []
            p2s_fld = []
            b1s_fld = []
            b2s_fld = []
        
            if lines_fld is not None:
                # convert to polar
                for line_fld in lines_fld:
                    line_fld = np.squeeze(line_fld[0])
                    p1 = line_fld[0:2]
                    p2 = line_fld[2:4]
                    if np.any(line_fld < cfg.border_tolerance) or np.any(line_fld > img.shape[1] - cfg.border_tolerance):
                        rhofld, thetafld = line_cart2pol(p1, p2)
                        angles_fld.append(thetafld)
                        dists_fld.append(rhofld)
                        p1s_fld.append(p1.tolist())
                        p2s_fld.append(p2.tolist())
                        if np.any(p1 < cfg.border_tolerance) or np.any(p1 > (img.shape[1] - cfg.border_tolerance)):
                            b1s_fld.append(0)
                        else:
                            b1s_fld.append(1)
                        if np.any(p2 < cfg.border_tolerance) or np.any(p2 > (img.shape[1] - cfg.border_tolerance)):
                            b2s_fld.append(0)
                        else:
                            b2s_fld.append(1)

                plt.title(f'{img_name}({len(lines_fld)} lines with FLD)')
                plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
                for line_fld in lines_fld:
                    line2draw = line_fld[0]
                    plt.axline((line2draw[0], line2draw[1]), (line2draw[2], line2draw[3]))        
                plt.savefig(os.path.join(vis_output, f"{img_name[:-4]}_em.jpg"))
                plt.close()
            else:
                plt.title(f'{img_name} (no lines with FLD)')
                plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))    
                plt.savefig(os.path.join(vis_output, f"{img_name[:-4]}_em.jpg"))
                plt.close()
            #pdb.set_trace()
            detected_lines = {
                'angles': angles_fld,
                'dists': dists_fld,
                'p1s': p1s_fld,
                'p2s': p2s_fld,
                'b1s': b1s_fld,
                'b2s': b2s_fld
            }
            with open(os.path.join(lines_output_folder, f"{img_name[:-4]}.json"), 'w') as lj:
                json.dump(detected_lines, lj, indent=3)
            print('saved', img_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract lines from segmented motifs')
    parser.add_argument('-p', '--puzzle', type=str, default='wikiart_kuroda_4x4', help='puzzle folder')
    parser.add_argument('-m', '--method', type=str, default='', choices=['', 'fld', 'hough'], help='The method used to detect the lines. Leave empty and it will be loaded from cfg file.')
    args = parser.parse_args()
    main(args)

        