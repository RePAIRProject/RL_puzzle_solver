
import numpy as np
import cv2 as cv2
from skimage import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

frag_num = 4

#im = io.imread(f'/home/marina/Desktop/pigmentmaps_image/Decor1_ex2/tiff/Decor1_ex2_p{frag_num}.tiff')
im = io.imread(f'/home/marina/Desktop/pigmentmaps_image/Decor1_ex4_all_colors.tiff')

channel_num = np.shape(im)[0]
plt.figure(figsize=(16, 8))
plt.suptitle(f'fragment_{frag_num}', fontsize=22)
image_0 = im[0,:,:,:]
plt.subplot(2, channel_num//2, 1)
plt.imshow(image_0)

pigment_cube = np.zeros((np.shape(im)[1], np.shape(im)[2],channel_num-1))  # 1-blue, 2-green, 3-red
kernel = np.ones((30,30), np.uint8)
for col_channel in range(1,channel_num):
    layer = np.sum(im[col_channel,:,:,:]>70, axis = 2)
    denoised_layer = cv2.morphologyEx(layer.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    pigment_cube[:, :, col_channel-1] = denoised_layer

    plt.subplot(2, channel_num//2, col_channel+1)
    plt.title(f'pigment_{col_channel}', fontsize=14)
    plt.imshow(denoised_layer, cmap='gray')

plt.show()
