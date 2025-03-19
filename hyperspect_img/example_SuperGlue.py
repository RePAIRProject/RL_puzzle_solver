
from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np
from PIL import Image
from skimage import io
from skimage.io import imsave, imread
from scipy.ndimage import rotate
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import json, os

#url_image1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
#image1 = Image.open(requests.get(url_image1, stream=True).raw)
#url_image2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
#image2 = Image.open(requests.get(url_image2, stream=True).raw)

denoise_input = True

folder_path = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobi_g35_36_TEST'
pieces_names = os.listdir(os.path.join(folder_path, 'tiff'))
pieces_names.sort()

rotation_list = [180, 120, 180+20, 120, 120, 260]

list_AB = []
for frag in range(len(pieces_names)):
    frag_id = pieces_names[frag][:9]
    angle = rotation_list[frag]

    url_image1 = f'{folder_path}/tiff/{frag_id}.tiff'
    url_image2 = f'{folder_path}/rgb/{frag_id}.png'

    img2_color = io.imread(url_image2)
    img_tiff = io.imread(url_image1, plugin='tifffile')

    img1_color = img_tiff[0, :, :, :]
    img1_color = rotate(img1_color, angle, reshape=True, mode='constant', order=0)

    channel_num = np.shape(img_tiff)[0]
    img_tiff_rot = np.zeros([channel_num, np.shape(img1_color)[0], np.shape(img1_color)[1], np.shape(img1_color)[2]])
    img_tiff_rot[0,:, :, :] = img1_color
    for col_channel in range(1, channel_num):
        layer = img_tiff[col_channel, :, :, :]
        rot_layer = rotate(layer, angle, reshape = True, mode = 'constant', order = 0)
        img_tiff_rot[col_channel,:, :, :] = rot_layer

    # plt.imshow(img1_color)
    # angle = rotation_list[frag]
    # img1_color = (rotate(img1_color, angle, resize=True) * 255).astype(np.uint8)

#################################################
    if denoise_input:
        plt.figure(figsize=(16, 8))
        plt.suptitle(f'fragment_{frag_id}', fontsize=22)
        plt.subplot(1, 3, 1)
        plt.imshow(img1_color)

        hsv = cv2.cvtColor(img1_color, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] += 80
        bright_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        alfa = img1_color[:, :, 3]>0
        denoised_image = img1_color
        denoised_image[:, :, 0:3] = bright_image * alfa[:, :, np.newaxis]

        plt.subplot(1, 3, 2)
        plt.imshow(denoised_image)
        plt.subplot(1, 3, 3)
        plt.imshow(img2_color)
        plt.show()

        img1_color = denoised_image

#############################################
    #cv2.imwrite(f'{folder_path}/in_sg/B_{frag_id}.jpg', cv2.cvtColor(img1_color, cv2.COLOR_RGB2BGR))
    #cv2.imwrite(f'{folder_path}/in_sg/A_{frag_id}.jpg', cv2.cvtColor(img2_color, cv2.COLOR_RGB2BGR))
    #list_AB.append(f'A_{frag_id}.jpg B_{frag_id}.jpg')

    cv2.imwrite(f'{folder_path}/in_sg/B_{frag_id}.png', img1_color)
    cv2.imwrite(f'{folder_path}/in_sg/A_{frag_id}.png', img2_color)
    # save img_tiff_rot !!!!!!!
    list_AB.append(f'A_{frag_id}.png B_{frag_id}.png')

    #cv2.imwrite(f'{folder_path}/in_sg/B_{frag_id}.png', cv2.cvtColor(img1_color, cv2.COLOR_RGB2BGR))
    #cv2.imwrite(f'{folder_path}/in_sg/A_{frag_id}.png', cv2.cvtColor(img2_color, cv2.COLOR_RGB2BGR))
    #list_AB.append(f'A_{frag_id}.png B_{frag_id}.png')

file = open('SuperGluePretrainedNetwork/assets/test_repair2.txt', 'w')
for item in list_AB:
    file.write(item + "\n")
file.close()

#####################
#####################

# python match_pairs.py --input_pairs assets/test_repair2.txt --input_dir /home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobi_g35_36_TEST/in_sg --output_dir /home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobi_g35_36_TEST/out_sg --viz --show_keypoints --resize 1000 1000

#####################
#####################

for frag in range(len(pieces_names)):
    frag_id = pieces_names[frag][:9]

    img2_color = cv2.imread(f'{folder_path}/in_sg/A_{frag_id}.png')
    img1_color = cv2.imread(f'{folder_path}/in_sg/B_{frag_id}.png')

    path = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobi_g35_36_TEST/out_sg/A_{frag_id}_B_{frag_id}_matches.npz'
    npz = np.load(path)  #npz.files
    matches =npz['matches']
    kp0=  npz['keypoints0']
    kp1 = npz['keypoints1']
    conf = npz['match_confidence']

    # Keep the matching keypoints.
    valid = matches > -1
    kp0 = kp0[valid]
    kp1 = kp1[matches[valid]]
    mconf = conf[valid]

    homography, mask = cv2.findHomography(kp1, kp0, cv2.RANSAC)
    height, width, z = img2_color.shape
    #transformed_img = cv2.warpPerspective(img1_color_rot, homography, (height, width))
    transformed_img = cv2.warpPerspective(cv2.resize(img1_color, (1000, 1000)), homography, (1000, 1000))

    plt.figure(figsize=(16, 8))
    plt.suptitle(f'fragment_{frag_id}', fontsize=22)
    #image_0 = im[0,:,:,:]
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.resize(img2_color, (1000, 1000)))
    plt.scatter(kp0[:, 0], kp0[:, 1], color='red', label='Origin points', zorder=5)
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.resize(img1_color, (1000, 1000)))
    plt.scatter(kp1[:, 0], kp1[:, 1], color='green', label='Destination points', zorder=5)
    plt.subplot(1, 3, 3)
    plt.imshow(transformed_img)

    channel_num = np.shape(img_tiff)[0]
    tiff_transformed = np.zeros([channel_num, np.shape(img1_color)[0], np.shape(img1_color)[1], np.shape(img1_color)[2]])
    tiff_transformed[0, :, :, :] = transformed_img
    for col_channel in range(1, channel_num):
        layer = img_tiff[col_channel, :, :, :]
        rot_layer = rotate(layer, angle, reshape=True, mode='constant', order=0)
        img_tiff_rot[col_channel, :, :, :] = rot_layer

    # apply transformation to every layer of tiff_image
    # save transformed_img

########################
# images = [img1_color, img2_color]
# image_sizes =[[img1_color.shape[:2], img2_color.shape[:2]]]
#
# processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
# model = AutoModel.from_pretrained("magic-leap-community/superglue_outdoor")
#
# inputs = processor(images, return_tensors="pt")
# with torch.no_grad():
#     outputs = model(**inputs)
#
# outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
#
# for i, output in enumerate(outputs):
#     print("For the image pair", i)
#     for keypoint0, keypoint1, matching_score in zip(
#             output["keypoints0"], output["keypoints1"], output["matching_scores"]
#     ):
#         print(
#             f"Keypoint at coordinate {keypoint0.numpy()} in the first image matches with keypoint at coordinate {keypoint1.numpy()} in the second image with a score of {matching_score}."
#         )
