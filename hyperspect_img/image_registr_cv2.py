
import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

frag_num = 4
url_image1  = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobi_g35_36_TEST/tiff/RPf_00278.tiff'
url_image2 = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobi_g35_36_TEST/rgb/RPf_00278.png'

# Open the image files.
im = io.imread(url_image1, plugin='tifffile')
img1_color = cv2.imread(url_image1)     # Image to be aligned. INPUT
img2_color = cv2.imread(url_image2)     # Reference image. !!!!

# Convert to grayscale.
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)  #
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
height, width = img2.shape  #size of reference image

img1 = cv2.resize(img1, 100, 100)

# Create ORB detector with 5000 features.
orb_detector = cv2.ORB_create(5000)

kp1, d1 = orb_detector.detectAndCompute(img1, None)
kp2, d2 = orb_detector.detectAndCompute(img2, None)

# Match features between the two images (Brute Force matcher with Hamming distance as measurement mode)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Match the two sets of descriptors.
matches = matcher.match(d1, d2)
# matches.sort(key = lambda x: x.distance)  # Sort matches on the basis of their Hamming distance.
# matches = matches[:int(len(matches)*0.9)] # Take the top 90 % matches forward.
no_of_matches = len(matches)

# Define empty matrices of shape no_of_matches * 2.
p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))
for i in range(len(matches)):
  p1[i, :] = kp1[matches[i].queryIdx].pt
  p2[i, :] = kp2[matches[i].trainIdx].pt

# Find the homography matrix.
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

transformed_image = cv2.warpPerspective(img1, homography, (width, height))
#plt.imshow(transformed_image)
#plt.show()

plt.figure(figsize=(16, 8))
plt.suptitle(f'fragment_{frag_num}', fontsize=22)
#image_0 = im[0,:,:,:]
plt.subplot(1, 3, 1)
plt.imshow(img1)
plt.subplot(1, 3, 2)
plt.imshow(img2)
plt.subplot(1, 3, 3)
plt.imshow(transformed_image)

##################################
##################################
##################################


# back to tiff_file
channel_num = np.shape(im)[0]
transformed_tiff =  np.zeros(np.shape(im))
pigment_cube = np.zeros((np.shape(im)[1], np.shape(im)[2],channel_num-1))  # 1-blue, 2-green, 3-red
kernel = np.ones((30,30), np.uint8)

plt.figure(figsize=(16, 8))
plt.suptitle(f'fragment_{frag_num}', fontsize=22)
#image_0 = im[0,:,:,:]
plt.subplot(2, channel_num//2, 1)
plt.imshow(img2_color)

for col_channel in range(1,channel_num):
    layer = im[col_channel, :, :, :]
    transformed_layer = cv2.warpPerspective(layer[:,:,0:3], homography, (width, height))
    transformed_tiff[col_channel, :, :, 0:3] = transformed_layer

    agg_layer = np.sum(transformed_layer > 70, axis=2)
    denoised_layer = cv2.morphologyEx(agg_layer.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    pigment_cube[:, :, col_channel-1] = denoised_layer

    plt.subplot(2, channel_num // 2, col_channel + 1)
    plt.title(f'pigment_{col_channel}', fontsize=14)
    plt.imshow(denoised_layer, cmap='gray')

# Use this matrix to transform the colored image wrt the reference image.
transformed_img = cv2.warpPerspective(img1_color, homography, (width, height))
transformed_img_new = cv2.resize(transformed_img, (251,251))
cv2.imwrite(f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobi_g35_36_TEST/output.jpg', transformed_img_new)

plt.show()
