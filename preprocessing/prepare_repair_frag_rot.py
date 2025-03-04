import os 
import numpy as np 
import scipy
import cv2
import pandas as pd 
from puzzle_utils.shape_utils import get_mask, get_polygon
import matplotlib.pyplot as plt 

target_shape = 251
gr_num = 3
obj_num = 3
#dataset = f'/media/lucap/big_data/datasets/repair/2D_dataset/RPobj_g{obj_num}_o{obj_num:04d}'
#output = f'/media/lucap/big_data/datasets/repair/2D_dataset/RPobj_g{obj_num}_o{obj_num:04d}_gt_rot'
#dataset = f'output/repair_rt/RPobj_g{gr_num}_o{obj_num:04d}'
#output = f'output/repair_rt/RPobj_g{gr_num}_o{obj_num:04d}_gt_rot'

gt_path = f'/home/marina/PycharmProjects/repair_ground_truth/RePAIR_dataset/gt_2d_txt/group_39.txt'
dataset = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobj_g39_o0039_gt_rot/motif_annotation/resized_images'
dataset_motifs_annotation = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobj_g39_o0039_gt_rot/motif_annotation/resized_motif_masks'
output = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobj_g39_o0039_gt_rot/motif_annotation'

output_pieces = os.path.join(output, 'pieces')
output_masks = os.path.join(output, 'masks')
output_poly = os.path.join(output, 'polygons')
output_motifs = os.path.join(output, 'motifs')

os.makedirs(output_pieces, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)
os.makedirs(output_poly, exist_ok=True)
os.makedirs(output_motifs, exist_ok=True)
files = os.listdir(dataset)
files.sort()

gt = pd.read_csv(gt_path)
for j, file in enumerate(files):
    print(file)
    img = cv2.imread(os.path.join(dataset, file), cv2.IMREAD_COLOR)
    scaled_img = cv2.resize((img).astype(np.uint8), (target_shape, target_shape), interpolation=cv2.INTER_NEAREST)
    gt_info = gt[gt['rpf'] == file[:-4]]
    rot_angle = gt_info['rot'].item()

    rotated_img = scipy.ndimage.rotate(scaled_img, rot_angle, reshape=False)
    cv2.imwrite(os.path.join(output_pieces, file), rotated_img)
    mask = get_mask(rotated_img, noisy=True)
    cv2.imwrite(os.path.join(output_masks, file), (mask*255).astype(np.uint8))
    polygon = get_polygon(mask)
    np.save(os.path.join(output_poly, f"{file[:-4]}"), polygon)

    img_motif = cv2.imread(os.path.join(dataset_motifs_annotation, file), cv2.IMREAD_GRAYSCALE)
    scaled_motif = cv2.resize(img_motif.astype(np.uint8), (target_shape, target_shape), interpolation=cv2.INTER_NEAREST)
    rotated_motif = scipy.ndimage.rotate(scaled_motif, rot_angle, reshape=False, order=0)
    cv2.imwrite(os.path.join(output_motifs, file), rotated_motif)

