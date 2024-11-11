import os 
import numpy as np 
import scipy
import cv2
import pandas as pd 
from puzzle_utils.shape_utils import get_mask, get_polygon
import matplotlib.pyplot as plt 

target_shape = 251
gr_num = 90
obj_num = 114

#dataset = f'/media/lucap/big_data/datasets/repair/2D_dataset/RPobj_g{obj_num}_o{obj_num:04d}'
#output = f'/media/lucap/big_data/datasets/repair/2D_dataset/RPobj_g{obj_num}_o{obj_num:04d}_gt_rot'

dataset = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePair_new/RPobj_g{gr_num}_o{obj_num:04d}'
output = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePair_new/RPobj_g{gr_num}_o{obj_num:04d}_gt_rot'

output_pieces = os.path.join(output, 'pieces')
output_masks = os.path.join(output, 'masks')
output_poly = os.path.join(output, 'polygons')
os.makedirs(output_pieces, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)
os.makedirs(output_poly, exist_ok=True)
files = os.listdir(dataset)

#gt_path = f'/home/lucap/Downloads/RPobj_g{obj_num}_o{obj_num:04d}.txt'
gt_path = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePair_new2/2D_Ground_Truth/RPobj_g{gr_num}_o{obj_num:04d}.txt'

gt = pd.read_csv(gt_path)

for file in files:
    print(file)
    img = cv2.imread(os.path.join(dataset, file), cv2.IMREAD_COLOR)
    scaled_img = cv2.resize((img).astype(np.uint8), (target_shape, target_shape))
    gt_info = gt[gt['rpf'] == file]
    rot_angle = gt_info['rot'].item()
    rotated_img = scipy.ndimage.rotate(scaled_img, rot_angle, reshape=False)
    cv2.imwrite(os.path.join(output_pieces, file), rotated_img)
    mask = get_mask(rotated_img, noisy=True)
    cv2.imwrite(os.path.join(output_masks, file), (mask*255).astype(np.uint8))
    polygon = get_polygon(mask)
    np.save(os.path.join(output_poly, f"{file[:-4]}"), polygon)

