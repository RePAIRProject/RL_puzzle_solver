
import argparse
# from ultralytics import YOLO
import os
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib
# matplotlib.use('TkAgg')

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

    gr_num = 29
    obj_num = 29

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

    if args.yolo_model == '':
        yolov8_model_path = '/home/marina/PycharmProjects/RL_puzzle_solver/yolov5/best.pt'
        #yolov8_model_path = '/Users/Marina/PycharmProjects/RL_puzzle_solver/yolov5/best.pt'
    else:
        yolov8_model_path = args.yolo_model
    if args.images == "":
        imgs_folder = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePair_new/RPobj_g{gr_num}_o{obj_num:04d}_gt_rot/pieces'
        #imgs_folder = '/Users/Marina/PycharmProjects/RL_puzzle_solver/output/repair/repair_g28/pieces'
        #motifs_output = '/Users/Marina/PycharmProjects/RL_puzzle_solver/output/repair/repair_g28/motif_OBB'
        motifs_output = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePair_new/RPobj_g{obj_num}_o{obj_num:04d}_gt_rot'
    else:
        imgs_folder = os.path.join(args.images, 'pieces')
        motifs_output = args.images

    motifs_output = os.path.join(motifs_output, 'motifs_segmentation')
    #motifs_output = os.path.join(args.images, 'motifs_detection_OBB')
    os.makedirs(motifs_output, exist_ok=True)
    
    indent_spaces = 3

    # Get the yolo model
    yolov8_obb_detector = YOLO(yolov8_model_path)

    # Go through the images and extract features
    obb_colormap = mpl.colormaps['jet'].resampled(12)
    vis_output_dir = os.path.join(motifs_output, 'visualization')
    os.makedirs(vis_output_dir, exist_ok=True)
    list_names = os.listdir(imgs_folder)
    list_names.sort()
    for img_p in (list_names):
        img_name = img_p[:-4]

        img_fp = os.path.join(imgs_folder, img_p)
        # read the image
        img_cv = cv2.imread(img_fp)
        img_pil = read_PIL_image(img_fp)

        obbs = yolov8_obb_detector(img_pil)[0]
        base_img_colored = copy.deepcopy(img_cv)
        image0 = np.zeros(np.shape(img_pil)[0:2], dtype='uint8')
        jetcmap = mpl.colormaps['jet'].resampled(len(class_names))
        cubo_image0 = np.zeros((np.shape(img_pil)[0], np.shape(img_pil)[1], 14), dtype='uint8')
        # breakpoint()
        for mask, box in zip(obbs.masks, obbs.boxes):
            # class_label = mask.cpu().cls.numpy()[0]
            # do_pts = det_obb.cpu().xyxyxyxy.numpy()[0]
            # breakpoint()
            resized_mask = cv2.resize(mask.data.cpu().numpy()[0,:,:], (np.shape(img_pil)[0], np.shape(img_pil)[1]))
            class_label = int(box.data[0][-1].item())
            # # Polygon corner points coordinates
            # pts = np.array(do_pts, dtype='int64')
            # #color = 255*np.array(obb_colormap(class_label)[:3])
            # color = (255, 255, 255)
            # thickness = 2
            # isClosed = True
            # im0 = np.zeros(np.shape(img_pil)[0:2], dtype='uint8')
            # image0_new = cv2.fillPoly(im0, [pts], color)
            # base_img_colored = cv2.polylines(base_img_colored, [pts], isClosed=True, color=np.asarray(jetcmap(int(class_label))[:3]) * 255, thickness=3)
            # print(int(class_label))
            cubo_image0[:, :, class_label] += resized_mask.astype(np.uint8) #cubo_image0[:, :, int(class_label)] + image0_new
        cubo_image0 = np.clip(cubo_image0, 0, 1)
        # breakpoint()

        # save motifs_CUBE per ogni image
        filename = os.path.join(motifs_output, f'motifs_cube_{img_name}')
        np.save(filename, cubo_image0)

        n_motifs = cubo_image0.shape[2]
        plt.figure(figsize=(32,16))
        plt.suptitle(img_name, fontsize=38)
        plt.subplot(3, 7, 1)
        plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        plt.subplot(3, 7, 2)
        plt.imshow(cv2.cvtColor(base_img_colored, cv2.COLOR_BGR2RGB))
        for mt in range(n_motifs):
            motif_mask_mt = cubo_image0[:, :, mt]
            plt.subplot(3, 7, 7+mt+1)
            print(class_names[mt])
            plt.title(class_names[mt], fontsize=22)
            plt.imshow(motif_mask_mt, cmap='gray')

        #plt.title(f"Fragment {img_name}")
        fig_name = os.path.join(vis_output_dir, f'det_motifs{img_name}.png')
        plt.tight_layout()
        plt.savefig(fig_name)
        print('stop')
        
        # n_motifs = cubo_image0.shape[2]
        # for mt in range(n_motifs):
        #     motif_mask_mt = cubo_image0[:, :, mt]
        #     plt.subplot(2, 7, mt+1)
        #     plt.imshow(motif_mask_mt)
        # #plt.show()
        # plt.title(f"Fragment {img_name}")
        # fig_name = os.path.join(vis_output_dir, f'det_motifs{img_name}.png')
        # plt.savefig(fig_name)
        # print('stop')

    print("Finished! Output in", motifs_output)
    return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect motifs')
    parser.add_argument('-ym', '--yolo_model', type=str, default='', help='yolo model path (.pt)')
    parser.add_argument('-i', '--images', type=str, default='', help='images input folder')
    parser.add_argument('-cn', '--classes_names', type=str, default='', help='images input folder')

    args = parser.parse_args()
    main(args)
