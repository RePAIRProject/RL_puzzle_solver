
import numpy as np
from skimage import io
from scipy.ndimage import rotate
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import json, os


def prepare_img(pieces_names, folder_path, img_path, rotation_list, denoise_input = True, vis = False):
    list_ab = []
    for frag in range(len(pieces_names)):
        frag_id = pieces_names[frag][:9]
        angle = rotation_list[frag]

        url_image1 = f'{folder_path}/tiff/{frag_id}.tiff'
        url_image2 = f'{img_path}/pieces/{frag_id}.png'

        img2 = io.imread(url_image2)
        img_tiff = io.imread(url_image1, plugin='tifffile')

        img1 = img_tiff[0, :, :, :]
        img1 = rotate(img1, angle, reshape=True, mode='constant', order=0)

        channel_num = np.shape(img_tiff)[0]
        img_tiff_rot = np.zeros([channel_num, np.shape(img1)[0], np.shape(img1)[1], np.shape(img1)[2]])
        img_tiff_rot[0,:, :, :] = img1
        for color_channel in range(1, channel_num):
            layer = img_tiff[color_channel, :, :, :]
            rot_layer = rotate(layer, angle, reshape = True, mode = 'constant', order = 0)
            img_tiff_rot[color_channel,:, :, :] = rot_layer

        if denoise_input:
            if vis:
                plt.figure(figsize=(16, 8))
                plt.suptitle(f'fragment_{frag_id}', fontsize=22)
                plt.subplot(1, 3, 1)
                plt.imshow(img1)

            hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] += 80
            bright_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            alfa = img1[:, :, 3]>0
            denoised_image = img1
            denoised_image[:, :, 0:3] = bright_image * alfa[:, :, np.newaxis]
            img1 = denoised_image

            if vis:
                plt.subplot(1, 3, 2)
                plt.imshow(denoised_image)
                plt.subplot(1, 3, 3)
                plt.imshow(img2)
                plt.show()

        np.save(f'{folder_path}/pigment_cubes/pigment_cube_{frag_id}', img_tiff_rot)  # save img_tiff_rot as cube.npy!!!!!!!

        # save the input images for SuperGlue and .txt list of fragments (A - reference img, B - img to transform)
        cv2.imwrite(f'{folder_path}/in_sg/B_{frag_id}.png', img1)
        cv2.imwrite(f'{folder_path}/in_sg/A_{frag_id}.png', cv2.resize(img2, (1000, 1000),
               interpolation = cv2.INTER_LINEAR))
        list_ab.append(f'A_{frag_id}.png B_{frag_id}.png')

    file = open('SuperGluePretrainedNetwork/assets/test_repair2.txt', 'w')
    for item in list_ab:
        file.write(item + "\n")
    file.close()


def transform_pigment_fragments(pieces_names, folder_path):

    for frag in range(len(pieces_names)):
        frag_id = pieces_names[frag][:9]

        img2_color = cv2.imread(f'{folder_path}/in_sg/A_{frag_id}.png')
        img1_color = cv2.imread(f'{folder_path}/in_sg/B_{frag_id}.png')
        pigment_cube = np.load(f'{folder_path}/pigment_cubes/pigment_cube_{frag_id}.npy')

        path = f'{folder_path}/out_sg/A_{frag_id}_B_{frag_id}_matches.npz'
        npz = np.load(path)  #npz.files
        matches = npz['matches']
        kp0 = npz['keypoints0']
        kp1 = npz['keypoints1']
        conf = npz['match_confidence']

        # Keep the matching keypoints.
        valid = matches > -1
        kp0 = kp0[valid]
        kp1 = kp1[matches[valid]]
        m_conf = conf[valid] ## not used

        if len(kp0)>4:
            homography, mask = cv2.findHomography(kp1, kp0, cv2.RANSAC)
            transformed_img = cv2.warpPerspective(cv2.resize(img1_color, (1000, 1000)), homography, (1000, 1000))

            plt.figure(figsize=(16, 8))
            plt.suptitle(f'fragment_{frag_id}', fontsize=22)
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.resize(img2_color, (1000, 1000)))
            plt.scatter(kp0[:, 0], kp0[:, 1], color='red', label='Origin points', zorder=5)
            plt.subplot(1, 3, 2)
            plt.imshow(cv2.resize(img1_color, (1000, 1000)))
            plt.scatter(kp1[:, 0], kp1[:, 1], color='green', label='Destination points', zorder=5)
            plt.subplot(1, 3, 3)
            plt.imshow(transformed_img)

            channel_num = np.shape(pigment_cube)[0]
            pigment_cube_transformed = np.zeros(
                [channel_num, np.shape(transformed_img)[0], np.shape(transformed_img)[1], 4])

            plt.figure(figsize=(8, 16))
            for col_channel in range(channel_num):
                layer = pigment_cube[col_channel, :, :, :].astype(np.uint8)
                transformed_layer = cv2.warpPerspective(cv2.resize(layer, (1000, 1000)), homography, (1000, 1000))
                pigment_cube_transformed[col_channel, :, :, :] = transformed_layer

                # visualization of transformed pigment_layers
                plt.subplot(channel_num, 2, (col_channel+1)*2-1)
                plt.imshow(layer)
                plt.subplot(channel_num, 2, (col_channel+1)*2)
                plt.imshow(transformed_layer)
                # plt.show()
                vis_output_dir = os.path.join(f'{folder_path}/pigment_cubes/visualization')
                os.makedirs(vis_output_dir, exist_ok=True)
                fig_name = os.path.join(vis_output_dir, f'transformed_pigment_cube_{frag_id}.png')
                plt.tight_layout()
                plt.savefig(fig_name)

            transform_output_dir = os.path.join(f'{folder_path}/pigment_cubes/transformed_cubes')
            os.makedirs(transform_output_dir, exist_ok=True)
            np.save(os.path.join(transform_output_dir, f'transformed_pigment_cube_{frag_id}'), pigment_cube_transformed)  # save


def clean_pigment_cube(folder_path, pieces_names):

    cube_names = os.listdir(os.path.join(folder_path,'pigment_cubes/transformed_cubes'))
    cube_names.sort()

    for frag in range(len(cube_names)):
        #frag_id = pieces_names[frag][:9]
        frag_id = cube_names[frag][-13:-4]
###TODO
        input_cube = np.load(f'{folder_path}/pigment_cubes/transformed_cubes/transformed_pigment_cube_{frag_id}.npy')
        #input_cube = np.load(f'{folder_path}/pigment_cubes/transformed_pigment_cube_{frag_id}.npy')
        channel_num = np.shape(input_cube)[0]
        #pigment_cube_cleaned = np.zeros(np.shape(input_cube))  # 1-blue, 2-green, 3-red
        pigment_cube_cleaned = np.zeros((np.shape(input_cube)[1], np.shape(input_cube)[2], channel_num - 1)) # 1-blue, 2-green, 3-red

        plt.figure(figsize=(8, 12))

        kernel = np.ones((10,10), np.uint8)
        for col_channel in range(1, channel_num):
            layer = np.sum(input_cube[col_channel,:,:,:]>70, axis = 2)
            denoised_layer = np.clip(cv2.morphologyEx(layer.astype(np.uint8), cv2.MORPH_OPEN, kernel),0,1)
            pigment_cube_cleaned[:, :, col_channel-1] = denoised_layer

            # visualization of transformed pigment_layers
            plt.subplot(channel_num-1, 2, col_channel*2-1)
            plt.imshow(layer)
            plt.subplot(channel_num-1, 2, col_channel*2)
            plt.imshow(denoised_layer)

        # plt.show()
        vis_output_dir = os.path.join(f'{folder_path}/pigment_cubes/visualization')
        os.makedirs(vis_output_dir, exist_ok=True)
        fig_name = os.path.join(vis_output_dir, f'clean_pigment_cube_{frag_id}.png')
        plt.tight_layout()
        plt.savefig(fig_name)

        clean_output_dir = os.path.join(f'{folder_path}/pigment_cubes/cleaned_cubes')
        os.makedirs(clean_output_dir, exist_ok=True)
        np.save(os.path.join(clean_output_dir, f'clean_pigment_cube_{frag_id}'), pigment_cube_cleaned)


#  MAIN
group = 'RPobj_g35_g36'
folder_path = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/{group}//pigment_map'
img_path = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/{group}/'
pieces_names = os.listdir(os.path.join(folder_path, 'tiff'))
pieces_names.sort()
rotation_list = [180, 180, 180, 180, 0, 0]

# Step 1 - prepare pieces
prepare_img(pieces_names, folder_path, img_path, rotation_list, True, True)

#####################
# Step 2 - run SuperGlue model

# python match_pairs.py --input_pairs assets/test_repair2.txt --input_dir /home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobj_g35_g36/pigment_map/in_sg --output_dir /home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/RPobj_g35_g36/pigment_map/out_sg --viz --show_keypoints --resize 1000 1000

#####################

# Step 3 - transform fragments
transform_pigment_fragments(pieces_names, folder_path)

# Step 4 - clean fragments
clean_pigment_cube(folder_path, pieces_names)

