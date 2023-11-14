import os 
from gpytoolbox import png2poly
import numpy as np
import matplotlib.pyplot as plt 
import pdb
import cv2 
from puzzle_utils.shape_utils import get_sd
import shutil 

def reflection(x0):
    return lambda x, y: (2*x0 - x, y)

dataset = 'repair'
puzzle = 'repair_g97'
root_folder = os.path.join('output_8x8', f'{dataset}', f'{puzzle}')

pieces_folder = os.path.join(root_folder, 'pieces')
tmp_mask_folder = os.path.join(root_folder, 'tmp_flipped_mask')
os.makedirs(tmp_mask_folder, exist_ok=True)
mask_folder = os.path.join(root_folder, 'masks2')
os.makedirs(mask_folder, exist_ok=True)
polygons_folder = os.path.join(root_folder, 'polygons')
os.makedirs(polygons_folder, exist_ok=True)


pieces_names = os.listdir(pieces_folder)
for piece_name in pieces_names: 
    image = plt.imread(os.path.join(pieces_folder, piece_name))
    _, mask = get_sd(image)
    # save mask!
    target_mask_path = os.path.join(mask_folder, piece_name)
    plt.imsave(target_mask_path, mask, cmap='gray')
    # flipped tmp mask
    flipped = cv2.flip(mask, 0)
    target_mask_path = os.path.join(tmp_mask_folder, piece_name)
    plt.imsave(target_mask_path, flipped)
    # polygon!
    poly = png2poly(target_mask_path) 
    poly = np.asarray(poly)
    np.save(os.path.join(polygons_folder, piece_name[:-4]), poly)
    print('done', piece_name)

shutil.rmtree(tmp_mask_folder)


# for img_num in range(194, 204):
#     img_name = f"output/repair/decor_1_lines/pieces/RPf_{img_num:05d}_mask.png"
#     image = plt.imread(img_name)
#     flipped = cv2.flip(image, 0)
#     img_namef = f"output/repair/decor_1_lines/masks/RPf_{img_num:05d}_mask_flipped.png"
#     plt.imsave(img_namef, flipped)
#     poly = png2poly(img_namef) 
#     poly = np.asarray(poly)
#     # plt.imshow(image)
#     # plt.plot(poly[0][:, 0], poly[0][:, 1], '-', linewidth=10)
#     # plt.show()
#     # pdb.set_trace()
#     np.save(f"RPf_{img_num:05d}", poly)
#     print('done', img_num)