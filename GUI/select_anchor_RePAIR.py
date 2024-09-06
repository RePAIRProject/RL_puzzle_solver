import os
import re
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import argparse
import json
from scipy.io import loadmat

fg_folder = ""
images_folder = ""


def calculate_color_variation(img, mask, band_width):
    # Find contours of the foreground mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract a band around the borders
    band_mask = np.zeros_like(mask)
    for contour in contours:
        cv2.drawContours(band_mask, [contour], 0, 255, thickness=band_width)

    # Apply the band mask to the RGB image
    band_img = cv2.bitwise_and(img, img, mask=band_mask)

    # Convert the band image to the HSV color space
    hsv_band_img = cv2.cvtColor(band_img, cv2.COLOR_BGR2HSV)

    # Calculate the 2D histogram of the hue and saturation channels
    hist_2d, x_edges, y_edges = np.histogram2d(
        hsv_band_img[:,:,0].ravel(),
        hsv_band_img[:,:,1].ravel(),
        bins=[256, 256],
        range=[[0, 256], [0, 256]]
    )

    # Visualize the 2D histogram (optional)
    '''plt.imshow(np.log(hist_2d + 1), cmap='plasma', interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('2D Histogram of Hue and Saturation in the Band')
    plt.xlabel('Hue')
    plt.ylabel('Saturation')
    plt.show()'''

    # Count the number of non-zero bins in the histogram
    non_zero_bins = np.count_nonzero(hist_2d)

    # Normalize to be in the range [0, 1] based on the total number of bins
    total_bins = 256 * 256  # Assuming 256 bins for both hue and saturation channels
    normalized_color_variation = non_zero_bins / total_bins

    return normalized_color_variation


def detect_perpendicular_lines(img, fg_mask, band_width, existing_lines, min_line_distance):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection using Canny
    edges = cv2.Canny(gray, 30, 150, apertureSize=3)

    # Apply the foreground mask to exclude fragment borders
    edges = cv2.bitwise_and(edges, edges, mask=fg_mask)


    # Dilate the foreground mask to include the band region inside the fragment
    kernel = np.ones((band_width, band_width), np.uint8)
    eroded_fg_mask = cv2.erode(fg_mask, kernel, iterations=1)

    # Create a copy of the image for visualization
    img_vis = img.copy()

    # Create a copy of the Canny edge map for visualization
    #edge_map_vis = edges.copy()
    # Detect lines using the Hough Line Transform
    #lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Apply the dilated foreground mask to the Canny edge map
    edges_band = cv2.bitwise_and(edges, edges, mask=eroded_fg_mask)

    # Detect lines using the Hough Line Transform
    lines = cv2.HoughLines(edges_band, 1, np.pi / 180, threshold=100)
    #plt.imshow(img_vis)
    #plt.show()

    #plt.imshow(edges)
    #plt.show()

    # Check for perpendicular lines
    perpendicular_lines = 0
    if lines is not None and len(lines) >= 2:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                rho1, theta1 = lines[i][0]
                rho2, theta2 = lines[j][0]

                # Calculate the angle between the lines
                angle_diff = np.abs(np.degrees(theta1 - theta2))

                # Check if the lines are approximately perpendicular (you can adjust the threshold as needed)
                if 85 <= angle_diff <= 95:
                    # Check if the new lines are far from existing lines
                    if not are_lines_close(rho1, theta1, rho2, theta2, existing_lines, min_line_distance):
                        perpendicular_lines += 1

                    # Draw the lines on the visualization image
                    a1, b1 = np.cos(theta1), np.sin(theta1)
                    x1, y1 = a1 * rho1, b1 * rho1
                    pt1 = (int(x1 + 1000 * (-b1)), int(y1 + 1000 * (a1)))
                    pt2 = (int(x1 - 1000 * (-b1)), int(y1 - 1000 * (a1)))
                    cv2.line(img_vis, pt1, pt2, (0, 0, 255), 2)
                    #cv2.line(edge_map_vis, pt1, pt2, 255, 2)

                    a2, b2 = np.cos(theta2), np.sin(theta2)
                    x2, y2 = a2 * rho2, b2 * rho2
                    pt1 = (int(x2 + 1000 * (-b2)), int(y2 + 1000 * (a2)))
                    pt2 = (int(x2 - 1000 * (-b2)), int(y2 - 1000 * (a2)))
                    cv2.line(img_vis, pt1, pt2, (0, 0, 255), 2)
                    #cv2.line(edge_map_vis, pt1, pt2, 255, 2)


                    # Visualize the image with detected perpendicular lines
                    '''plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
                    plt.title('Perpendicular Lines')
                    plt.show()'''

                    # Visualize the Canny edge map with detected perpendicular lines
                    '''plt.imshow(edge_map_vis, cmap='gray')
                    plt.title('Canny Edge Map with Perpendicular Lines')
                    plt.show()'''

    if perpendicular_lines > 0:
        perpendicular_lines = 1
    return perpendicular_lines


def are_lines_close(rho1, theta1, rho2, theta2, existing_lines, min_distance):
    for (rho, theta) in existing_lines:
        distance = np.abs(rho - rho1) + np.abs(theta - theta1) + np.abs(rho - rho2) + np.abs(theta - theta2)
        if distance < min_distance:
            return True
    return False


def display_detected_key_fragments(sorted_image_scores, image_scores, images_with_borders, images_folder, imgs_names, nmb_frag):
    # Create a figure for subplots
    fig, axes = plt.subplots(2, nmb_frag, figsize=(15, 7))
    fig.suptitle('COLOR (band_width = 100) AND PERPENDICULAR LINES. Preprocessed with Median filtering.', fontsize=16)

    # Visualize highest ranked 5 images
    for i, (img_name, score) in enumerate(sorted_image_scores[:nmb_frag]):
        img_with_border = images_with_borders[imgs_names.index(img_name)]
        ax = axes[0, i]
        ax.imshow(cv2.cvtColor(img_with_border, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Score: {score:.3f}')
        ax.axis('off')

        # Sort image scores based on color variation count
    sorted_image_scores = sorted(image_scores, key=lambda x: x[1])
    # Visualize lowest ranked 5 images
    for i, (img_name, score) in enumerate(sorted_image_scores[:nmb_frag]):
        img_with_border = images_with_borders[imgs_names.index(img_name)]
        ax = axes[1, i]
        ax.imshow(cv2.cvtColor(img_with_border, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Score: {score:.3f}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Sort image scores based on color variation count

    # figure for subplots
    fig, axes = plt.subplots(2, nmb_frag, figsize=(15, 7))
    #fig.suptitle('ONLY PERPENDICULAR LINES. Preprocessed with Median filtering.', fontsize=16)
    #fig.suptitle('ONLY COLOR. band_width = 100. Preprocessed with Median filtering.', fontsize=16)
    fig.suptitle('COLOR (band_width = 100) AND PERPENDICULAR LINES. Preprocessed with Median filtering.', fontsize=16)

    # Visualize lowest ranked 5 images
    for i, (img_name, score) in enumerate(sorted_image_scores[:nmb_frag]):
        img_path = os.path.join(images_folder, img_name)
        img = cv2.imread(img_path)
        ax = axes[1, i]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Score: {score:.3f}')
        ax.axis('off')

    # Sort image scores based on color variation count
    sorted_image_scores = sorted(image_scores, key=lambda x: x[1], reverse=True)
    # Visualize highest ranked 5 images
    for i, (img_name, score) in enumerate(sorted_image_scores[:nmb_frag]):
        img_path = os.path.join(images_folder, img_name)
        img = cv2.imread(img_path)
        ax = axes[0, i]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Score: {score:.3f}')
        ax.axis('off')


    plt.tight_layout()
    plt.show()


def extract_number_from_image_name(image_name):
    match = re.search(r'\d+', image_name)
    return int(match.group()) if match else 0


def select_neighbour(path_dic, anchor_id):
    global images_folder
    global fg_folder

    back_path = path_dic['backend_path']
    path = path_dic['image_path']
    path_bw = path_dic['mask_path']
    comp_folder = path_dic['comp_folder']
    comp_name = path_dic['comp_name']

    images = []
    k = 0

    images_names = [img_name for img_name in os.listdir(images_folder)]
    sorted_images_names = sorted(images_names, key=extract_number_from_image_name)

    for img_name in sorted_images_names:
        if anchor_id == img_name:
            anchor_id = k
        images.append((img_name, k, -3))
        k += 1
    print(anchor_id)

    mat = loadmat(os.path.join(comp_folder, comp_name))

    print(mat.keys())
    # R = mat['R_line']
    R = mat['R']

    for i in range(R.shape[len(R.shape) - 1]):
        images[i] = (images[i][0], images[i][1], np.max(R[:, :, :, anchor_id, i]))

    neighbour_numbers = 3

    sorted_by_score = sorted(images, key=lambda x: x[2], reverse=True)

    top_k_images = sorted_by_score[:neighbour_numbers]

    print(top_k_images)

    set_backend_path(back_path, path, path_bw)

    neighbour_fragments = []

    for top in top_k_images:
        neighbour_fragments.append((top[0], top[2]))

    print(neighbour_fragments)

    return neighbour_fragments


def is_neighbour(img_name):
    if img_name == "piece_0010.png":
        return True
    elif img_name == "gr28_RPf_00200_intact_mesh.png":
        return True
    elif img_name == "gr28_RPf_00203_intact_mesh.png":
        return True
    elif img_name == "gr41_RPf_00334_intact_mesh.png":
        return True
    elif img_name == "RPf_00203.png":
        return True
    elif img_name == "RPf_00197.png":
        return True
    elif img_name == "RPf_00201.png":
        return True
    else:
        return False


def select_anchor(back_path, path, path_bw):
    global images_folder
    global fg_folder
    set_backend_path(back_path, path, path_bw)

    # args = get_args()
    plt.close('all')
    band_width = 100  # Adjust the width of the band as needed
    nmb_frag = 10
    kernel_size = 3
    check_color = True
    check_corner = True

    # images_folder = args.dataset

    # fg_folder = args.fg_mask

    existing_lines = []
    images_names = [img_name for img_name in os.listdir(images_folder)]

    image_scores = []
    images_with_borders = []

    for img_name in images_names:

        rgb_img_path = os.path.join(images_folder, img_name)
        rgb_image = cv2.imread(rgb_img_path)
        fg_img_path = os.path.join(fg_folder, img_name)
        fg_img = cv2.imread(fg_img_path, cv2.IMREAD_GRAYSCALE)

        rgb_image = cv2.medianBlur(rgb_image, kernel_size)

        #-----------------------------------------------------------------------------------
        # compute normalized color variation in a band on fragment border:
        color_variation_count = calculate_color_variation(rgb_image, fg_img, band_width)

        # -----------------------------------------------------------------------------------
        # Detect perpendicular lines and get the count
        min_line_distance = 0.01 * min(rgb_image.shape[0], rgb_image.shape[1])
        #perpendicular_lines_count = detect_perpendicular_lines(rgb_image, fg_img, band_width)
        perpendicular_lines_count = detect_perpendicular_lines(rgb_image, fg_img, band_width, existing_lines, min_line_distance)

        # COMPUTE THE FINAL SCORE
        # The final score is determined based on either the color variation score, the presence of perpendicular lines, or a combination of both factors
        if (check_color == 1 and check_corner == 1):
            # Combine scores with 0.5 weight each
            final_score = 0.5 * color_variation_count + 0.5 * perpendicular_lines_count
        elif (check_color==0 and check_corner == 1):
            final_score = perpendicular_lines_count
        elif (check_color == 1 and check_corner == 0):
            final_score = color_variation_count

        # Append image name and score to the list
        image_scores.append((img_name, final_score))

        # commented and added to a method for later use
        # init_visualization(band_width, fg_img, images_with_borders, rgb_image)

        # Sort image scores based on color variation count
    sorted_image_scores = sorted(image_scores, key=lambda x: x[1], reverse=True)

    # Select the top 10 fragments
    top_10_fragments = sorted_image_scores[:4]

    # Display user interface and get selected fragments
    # selected_fragments = create_user_interface(top_10_fragments, images_with_borders, imgs_names)


    #******** OMID, you can close this part and use the sorted scores directly **********************#
    ##-----------------------------------------------------------------------------------------------------------##
    # OUTPUT the final_scores of the top 10 fragments and corresponding image names into a JSON file
    output_data = [{'img_name': img_name, 'score': score} for img_name, score in top_10_fragments]
    output_file_path = 'top_10_fragments.json'
    with open(output_file_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=2)

    print(f"Top 10 fragments saved to: {output_file_path}")

    # Read the saved JSON file back into Python
    with open(output_file_path, 'r') as json_file:
        loaded_data = json.load(json_file)
    ##-----------------------------------------------------------------------------------------------------------##


    # Display user interface and get selected fragments
    #selected_fragments = create_user_interface(image_scores, images_with_borders, imgs_names)
    return top_10_fragments

    ##-----------------------------------------------------------------------------------------------------------##
    ## DISPLAY THE DETECTED KEY FRAGMENTS
    # display_detected_key_fragments(sorted_image_scores, image_scores, images_with_borders, images_folder, imgs_names, nmb_frag)


def init_visualization(band_width, fg_img, images_with_borders, rgb_image):
    ##-----------------------------------------------------------------------------------------------------------##
    ## FOLLOWING CODE IS FOR VISUALIZING THE BOUNDARY BAND OF THE FRAGMENT, WHERE COLOR VARIATION WAS COMPUTED FROM
    # Find contours of the foreground mask
    contours, _ = cv2.findContours(fg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create an empty mask for the border
    border_mask = np.zeros_like(fg_img)
    # Draw the border inside the fragment region
    for contour in contours:
        cv2.drawContours(border_mask, [contour], 0, 255, thickness=band_width)
    # Apply the border mask to the original image
    img_with_border = cv2.bitwise_and(rgb_image, rgb_image, mask=border_mask)
    # Append image with border to the list
    images_with_borders.append(img_with_border)
    ##-----------------------------------------------------------------------------------------------------------##


backend_path = os.getcwd() + "/GUI/DataBase/Images/RePAIR_plaque_2/"


def set_backend_path(back_path, path, path_bw):
    global backend_path
    global images_folder
    global fg_folder
    backend_path = back_path
    images_folder = path
    fg_folder = path_bw
