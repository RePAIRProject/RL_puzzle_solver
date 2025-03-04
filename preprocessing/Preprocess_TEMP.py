import os
from distutils.dep_util import newer_group
import numpy as np
import cv2
import matplotlib.pyplot as plt
from puzzle_utils.shape_utils import get_sd, get_cm, shift_img, get_mask
import pdb
from configs import repair_cfg as cfg
from configs import folder_names as fnames
import argparse

def center_fragment(image):
    #pdb.set_trace()
    #sd, mask = get_sd(image)
    #mask = get_mask(image, black_bg=False)
    mask = get_mask(image[:,:,:2])
    cm = get_cm(mask)
    center_pos = [np.round(image.shape[0]/2).astype(int), np.round(image.shape[1]/2).astype(int)]
    shift = np.round(np.array(cm) - center_pos).astype(int)
    centered_image = shift_img(image, -shift[0], -shift[1])
    centered_mask = shift_img(mask, -shift[0], -shift[1])
    return centered_image, centered_mask

def center_motif_masks(image, motif_mask):
    #mask = get_mask(image, black_bg=False)
    mask = get_mask(image[:,:,:2])
    cm = get_cm(mask)
    center_pos = [np.round(image.shape[0]/2).astype(int), np.round(image.shape[1]/2).astype(int)]
    shift = np.round(np.array(cm) - center_pos).astype(int)
    centered_image = shift_img(image, -shift[0], -shift[1])
    centered_mask = shift_img(mask, -shift[0], -shift[1])
    centered_motif_masks = shift_img(motif_mask, -shift[0], -shift[1])
    return centered_image, centered_mask, centered_motif_masks

def main(args):
    target_img_shape = 251 #cfg.piece_size

    input_folder = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobj_g39_o0039_gt_rot/motif_annotation/input_RGB_images'
    input_motif_folder = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobj_g39_o0039_gt_rot/motif_annotation/input_motif_masks'
    output_folder_pieces = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobj_g39_o0039_gt_rot/motif_annotation/resized_images'
    output_folder_masks = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobj_g39_o0039_gt_rot/motif_annotation/resized_masks'
    output_folder_motif_masks = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobj_g39_o0039_gt_rot/motif_annotation/resized_motif_masks'

    for piece_path in os.listdir(input_folder):
        # if 'RPf' in piece_path:
        piece_full_path = os.path.join(input_folder, piece_path)
        motif_full_path = os.path.join(input_motif_folder, piece_path)
        imgcv = cv2.imread(piece_full_path, cv2.IMREAD_UNCHANGED)
        motif_masks = cv2.imread(motif_full_path, cv2.IMREAD_UNCHANGED)

        # put image onto black square
        imgcv_nero = np.zeros((1500, 1500, 4),dtype=np.uint8)
        imgcv_nero[:,:,3] = 255
        x, y, z = np.shape(imgcv)
        imgcv_nero[:x, :y, :z] = imgcv

        motif_nero = np.zeros((1500, 1500), dtype=np.uint8)
        motif_nero[:x, :y] = motif_masks
        centered_img, img_mask, centered_motif_masks = center_motif_masks(imgcv_nero, motif_nero)
        centered_img[:,:,3] = img_mask*255

        if imgcv.shape[0] != target_img_shape:
            centered_img = cv2.resize(centered_img, dsize=(target_img_shape, target_img_shape), interpolation=cv2.INTER_CUBIC)
            img_mask = cv2.resize(img_mask, dsize=(target_img_shape, target_img_shape),
                                      interpolation=cv2.INTER_NEAREST)
            centered_motif_masks = cv2.resize(centered_motif_masks, dsize=(target_img_shape, target_img_shape),
                                      interpolation=cv2.INTER_NEAREST)

        target_path = os.path.join(output_folder_pieces, piece_path)
        target_path_mask = os.path.join(output_folder_masks, f"{piece_path[:-4]}_mask.png")
        target_path_motif_masks = os.path.join(output_folder_motif_masks, piece_path)
        cv2.imwrite(target_path, centered_img)
        cv2.imwrite(target_path_mask, img_mask * 255)
        cv2.imwrite(target_path_motif_masks, centered_motif_masks)
        print('saved centered image in', target_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare the data')
    parser.add_argument('-d', '--dataset', type=str, default='/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobj_g39_o0039_gt_rot/motif_annotation/RGB_images', help='data folder')
    args = parser.parse_args()
    main(args)