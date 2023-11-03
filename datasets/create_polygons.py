import os 
from gpytoolbox import png2poly
import numpy as np
import matplotlib.pyplot as plt 
import pdb
import cv2 

def reflection(x0):
    return lambda x, y: (2*x0 - x, y)


for img_num in range(194, 204):
    img_name = f"data/repair/repair_g28/masks/RPf_{img_num:05d}_mask.png"
    image = plt.imread(img_name)
    flipped = cv2.flip(image, 0)
    img_namef = f"data/repair/repair_g28/masks/RPf_{img_num:05d}_mask_flipped.png"
    plt.imsave(img_namef, flipped)
    poly = png2poly(img_namef) 
    poly = np.asarray(poly)
    # plt.imshow(image)
    # plt.plot(poly[0][:, 0], poly[0][:, 1], '-', linewidth=10)
    # plt.show()
    # pdb.set_trace()
    np.save(f"RPf_{img_num:05d}", poly)
    print('done', img_num)