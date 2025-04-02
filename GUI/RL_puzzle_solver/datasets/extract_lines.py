import os 
import numpy as np
import matplotlib.pyplot as plt 
import pdb
import cv2 

for img_num in range(194, 204):
    img_name = f"/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/segmap14c/RPf_{img_num:05d}.png"
    image = plt.imread(img_name)
    colored_image = np.zeros_like(image)
    colored_image += (image == 10/255).astype(int)
    colored_image += (image == 9/255).astype(int)
    colored_image[:,:,3] = 1
    
    # plt.imshow(colored_image)
    # plt.show()
    # pdb.set_trace()
    plt.imsave(f'output_8x8/repair/g28seg/pieces/RPf_{img_num:05d}_bands.png', colored_image)

    #np.save(f"RPf_{img_num:05d}", poly)
    print('done', img_num)