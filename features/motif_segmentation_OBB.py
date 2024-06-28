
import argparse
from ultralytics import YOLO
import os
import cv2
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
import torch
from PIL import Image
from ultralytics import YOLO
import pdb
import json
from configs import folder_names as fnames

def read_PIL_image(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    return img

def main():

    yolov8_model_path = '/home/marina/PycharmProjects/RL_puzzle_solver/yolov5/best.pt'
    imgs_folder = '/home/marina/PycharmProjects/RL_puzzle_solver/output/repair/repair_g28/pieces'
    motifOBB_folder = '/home/marina/PycharmProjects/RL_puzzle_solver/output/repair/repair_g28/motif_OBB'

    #images_folder = os.path.join(fnames.data_path, args.puzzle, fnames.imgs_folder)
    #output_folder = os.path.join(os.getcwd(), fnames.output_dir, args.puzzle)
    #motifOBB_folder = os.path.join(output_folder, fnames.segm_output_name)

    patterns_v8_feats_folder = os.path.join(os.getcwd(), 'detected_obb')
    os.makedirs(patterns_v8_feats_folder, exist_ok=True)
    indent_spaces = 3

    # Get the yolo model
    yolov8_obb_detector = YOLO(yolov8_model_path)

    # Go through the images and extract features
    obb_colormap = mpl.colormaps['jet'].resampled(12)

    for img_p in os.listdir(imgs_folder):
        img_name = img_p[:-4]

        img_fp = os.path.join(imgs_folder, img_p)
        # read the image
        img_cv = cv2.imread(img_fp)
        img_pil = read_PIL_image(img_fp)

        obbs = yolov8_obb_detector(img_pil)[0]

        image0 = np.zeros(np.shape(img_pil)[0:2], dtype='uint8')
        cubo_image0 = np.zeros((np.shape(img_pil)[0],np.shape(img_pil)[1],12), dtype='uint8')
        for det_obb in obbs.obb:
            # breakpoint()
            class_label = det_obb.cpu().cls.numpy()[0]
            do_pts = det_obb.cpu().xyxyxyxy.numpy()[0]
            #do_xywhr = det_obb.cpu().xywhr.numpy()[0]
            # det_obb_dict = {
            #     'class': class_label,
            #     'center': do_xywhr[:2],
            #     'width': do_xywhr[2],
            #     'height': do_xywhr[3],
            #     'angle': do_xywhr[4],
            #     'coords': do_pts.tolist()
            # }

            # Polygon corner points coordinates
            pts = np.array(do_pts, dtype='int64')
            color = 255*np.array(obb_colormap(class_label)[:3])
            # color = (255, 0, 0)
            thickness = 2
            isClosed = True
            image0 = cv2.polylines(image0, [pts], isClosed, color, thickness)
            print(int(class_label))
            cubo_image0[:, :, int(class_label)-1] = cubo_image0[:, :, int(class_label)-1] + image0

        print(img_p)
        plt.imshow(image0)
        plt.show()
        print('done 1 image')

        ### TODO !!!!
        ### save OBB_CUBO per ogni image ###
        ###

    print("Finished! Output in", motifOBB_folder)
    return 1

    cv2.imwrite(os.path.join(motifOBB_folder, img), segmentation_mask)
    print('saved', img)

if __name__ == '__main__':
    main()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Segment with yolov8 (get masks without visualization)')
#     parser.add_argument('--model', type=str, default='', help='model (.pt file)')
#     parser.add_argument('--puzzle', type=str, default='', help='puzzle name (inside the data folder)')
#     parser.add_argument('--size', type=int, default=251, help='resizing of the images')
#
#     args = parser.parse_args()
#     main(args)
