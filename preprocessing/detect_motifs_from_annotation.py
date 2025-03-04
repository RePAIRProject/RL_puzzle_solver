import argparse
# from ultralytics import YOLO
import os
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import pdb
import json
from configs import folder_names as fnames
import yaml
import matplotlib as mpl
import copy


def read_PIL_image(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    return img


def main(args):
    gr_num = 39
    obj_num = 39

    if args.classes_names == '':
        det_classes_file = 'preprocessing/det_classes.yaml'
    else:
        det_classes_file = args.classes_names

    class_names = []
    if os.path.exists(det_classes_file):
        with open(det_classes_file, 'r') as yaml_file:
            classes_names = yaml.safe_load(yaml_file)
        for i in range(14):
            class_names.append(f"{classes_names[i]} (Class {i})")
    else:
        for i in range(14):
            class_names.append(f"Class {i}")

    motif_folder = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobj_g39_o0039_gt_rot/motif_annotation/motifs'
    imgs_folder = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobj_g39_o0039_gt_rot/motif_annotation/pieces'
    motifs_output = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobj_g39_o0039_gt_rot/motif_annotation'
    motifs_output = os.path.join(motifs_output, 'motif_cubs')
    os.makedirs(motifs_output, exist_ok=True)

    indent_spaces = 3
    class_names = []
    if os.path.exists(det_classes_file):
        with open(det_classes_file, 'r') as yaml_file:
            classes_names = yaml.safe_load(yaml_file)
        for i in range(14):
            class_names.append(f"{classes_names[i]} (Class {i})")
    else:
        for i in range(14):
            class_names.append(f"Class {i}")

    # Go through the images and extract features
    obb_colormap = mpl.colormaps['jet'].resampled(12)
    vis_output_dir = os.path.join(motifs_output, 'visualization')
    os.makedirs(vis_output_dir, exist_ok=True)
    list_names = os.listdir(imgs_folder)
    list_names.sort()
    for img_p in (list_names):
        img_name = img_p[:-4]

        # read the image
        img_cv = cv2.imread(os.path.join(imgs_folder, img_p))
        base_img_colored = copy.deepcopy(img_cv)

        # read motifs_mask
        motif_image = cv2.imread(os.path.join(motif_folder, img_p), cv2.IMREAD_GRAYSCALE)
        levels = np.unique(motif_image)
        cubo_image0 = np.zeros((np.shape(motif_image)[0], np.shape(motif_image)[1], 14), dtype='uint8')

        for j in enumerate(levels):
            # breakpoint()
            class_label = levels[j[0]]
            pts = (motif_image==class_label)*1
            print(int(class_label))
            plt.imshow(pts)
            cubo_image0[:, :, int(class_label)] = cubo_image0[:, :, int(class_label)] + pts

        # save motifs_CUBE for every image
        filename = os.path.join(motifs_output, f'motifs_cube_{img_name}')
        np.save(filename, cubo_image0)

        n_motifs = cubo_image0.shape[2]
        plt.figure(figsize=(32, 16))
        plt.suptitle(img_name, fontsize=38)
        plt.subplot(3, 7, 1)
        plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        plt.subplot(3, 7, 2)
        plt.imshow(cv2.cvtColor(base_img_colored, cv2.COLOR_BGR2RGB))
        for mt in range(n_motifs):
            motif_mask_mt = cubo_image0[:, :, mt]
            plt.subplot(3, 7, 7 + mt + 1)
            print(class_names[mt])
            plt.title(class_names[mt], fontsize=22)
            plt.imshow(motif_mask_mt, cmap='gray')

        # plt.title(f"Fragment {img_name}")
        fig_name = os.path.join(vis_output_dir, f'det_motifs{img_name}.png')
        plt.tight_layout()
        plt.savefig(fig_name)
        print('stop')

    print("Finished! Output in", motifs_output)
    return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect motifs')
    parser.add_argument('-ym', '--yolo_model', type=str, default='', help='yolo model path (.pt)')
    parser.add_argument('-i', '--images', type=str, default='', help='images input folder')
    parser.add_argument('-cn', '--classes_names', type=str, default='', help='images input folder')

    args = parser.parse_args()
    main(args)
