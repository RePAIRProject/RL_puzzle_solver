import cv2
import numpy as np
import random
import os
import shapely
import argparse
import matplotlib.pyplot as plt
import json
import configs.unified_cfg as cfg


def create_random_image2(line_type, num_lines, width, height, lines_length=10, isClosed = False, thickness=1, color=(0, 0, 0)):

    img2draw = np.ones(shape=(width, height), dtype=np.uint8) * 255
    img_shape = [width, height]

    all_lines = np.zeros((num_lines, 4))

    if line_type == 'polylines':
        n_lines = num_lines+1
        pts = np.zeros((n_lines, 2), dtype=np.int32)
        for k in range(n_lines):
            r_ax = np.round(np.clip(np.random.uniform(-4, 3.7), -1, 3)).astype(int)
            pts[k, :] = generate_random_point(img_shape, distribution='uniform', on_axis=r_ax)
        img2draw = cv2.polylines(img2draw, [pts], isClosed=isClosed, color=color, thickness=thickness)

        all_lines[:, 0:2] = pts[:-1, :]
        all_lines[:, 2:4] = pts[1:, :]

    else:
        for j in range(num_lines):
            if line_type == 'segments':
                p1 = generate_random_point(img_shape, distribution='uniform')
                p2 = generate_random_point(img_shape, distribution='uniform')
                img2draw = cv2.line(img2draw, p1, p2, color=color, thickness=thickness)

            else:  # line_type == 'lines':
                axis1, axis2 = random.sample(range(0, 4), 2)  # select two axis
                p1 = generate_random_point(img_shape, distribution='uniform', on_axis=axis1)
                p2 = generate_random_point(img_shape, distribution='uniform', on_axis=axis2)
                img2draw = cv2.line(img2draw, p1, p2, color=color, thickness=thickness)

            all_lines[j, 0:2] = p1
            all_lines[j, 2:4] = p2

    return img2draw, all_lines

def generate_random_point(ranges, distribution='uniform', on_axis=-1):

    if on_axis < 0:
        p1x = np.random.uniform(0, ranges[0])
        p1y = np.random.uniform(0, ranges[1])
    elif on_axis == 0:
        p1x = 0
        p1y = np.random.uniform(0, ranges[1])
    elif on_axis == 1:
        p1x = np.random.uniform(0, ranges[0])
        p1y = 0
    elif on_axis == 2:
        p1x = ranges[0]
        p1y = np.random.uniform(0, ranges[1])
    elif on_axis == 3:
        p1x = np.random.uniform(0, ranges[0])
        p1y = ranges[1]

    p1 = np.round(np.array([p1x, p1y])).astype(int)

    return p1

def line_cart2pol(pt1, pt2):

    x_diff = pt1[0] - pt2[0]
    y_diff = pt1[1] - pt2[1]
    theta = np.arctan(-(x_diff/(y_diff + 10**-5)))
    rho = pt1[0] * np.cos(theta) + pt1[1] * np.sin(theta)
    rho2 = pt2[0] * np.cos(theta) + pt2[1] * np.sin(theta)
    #print("checkcart2pol", theta, rho, rho2)
    #pdb.set_trace()
    return rho, theta


###############################
num_images = 10
num_lines = 40
line_type = 'lines'
# height = 1000
# width = 1000
height = cfg.img_size
width = cfg.img_size
patch_size = cfg.piece_size
num_patches_side = cfg.num_patches_side
n_patches = num_patches_side * num_patches_side
#################################

dataset_path = os.path.join(f'C:\\Users\Marina\OneDrive - unive.it\RL\data')
cur_folder = os.path.join(dataset_path, f'random_{num_lines}_{line_type}_exact_detection')
os.makedirs(cur_folder, exist_ok=True)


print(f'creating images with {num_lines} lines', end='\r')
for N in range(num_images):
    ## create images with lines
    img, all_lines = create_random_image2(line_type, num_lines, height, width)

    ## save created image
    cv2.imwrite(os.path.join(cur_folder, f'image_{N}.jpg'), img)
    plt.figure()
    plt.imshow(img)
    plt.show()

    ## where to save patches
    puzzle_image_folder = os.path.join(
        f'C:\\Users\Marina\OneDrive - unive.it\RL\output_8x8\\random_{num_lines}_{line_type}_exact_detection')
    pieces_single_folder = os.path.join(puzzle_image_folder, f'image_{N}', 'pieces')
    os.makedirs(pieces_single_folder, exist_ok=True)

    ## where to save detected lines
    lines_output_folder = os.path.join(puzzle_image_folder, f'image_{N}', 'lines_detection\exact')
    os.makedirs(lines_output_folder, exist_ok=True)

    ## cut image into the patches and compute internal lines
    image = img
    x0_all = np.arange(0, image.shape[0], patch_size, dtype=int)
    y0_all = np.arange(0, image.shape[1], patch_size, dtype=int)

    k = 0
    for iy in range(num_patches_side):
        for ix in range(num_patches_side):
            x0 = x0_all[ix]
            y0 = y0_all[iy]
            x1 = x0 + patch_size - 1
            y1 = y0 + patch_size - 1
            box = shapely.box(x0, y0, x1, y1)  # patche box (xmin, ymin, xmax, ymax)

            ## create patch
            patch = image[y0:y1 + 1, x0:x1 + 1]
            # plt.figure()
            # plt.imshow(patch)
            # plt.show()

            ## save patch
            cv2.imwrite(os.path.join(pieces_single_folder, f"piece_{k:05d}.png"), patch)

            ## compute intersection points (line-box)
            angles_lsd = []
            dists_lsd = []
            p1s_lsd = []
            p2s_lsd = []
            b1s_lsd = []
            b2s_lsd = []

            for i in range(num_lines):
                line = shapely.LineString([all_lines[i, 0:2], all_lines[i, 2:4]])
                # line = shapely.LineString([(p1[0], p1[1]), (p2[0], p2[1])])

                intersect = shapely.intersection(line, box)  # points of intersection line with patch
                print(intersect)

                if shapely.is_empty(intersect) == False:
                    if len(list(zip(*intersect.xy))) > 1:
                        [s1, s2] = list(zip(*intersect.xy))
                        # change reference point
                        s1 = np.array(np.round(s1)) - [x0, y0]  ## CHECK !!!
                        s2 = np.array(np.round(s2)) - [x0, y0]  ## CHECK !!!
                        #
                        rhofld, thetafld = line_cart2pol(s1, s2)
                        angles_lsd.append(thetafld)
                        dists_lsd.append(rhofld)
                        p1s_lsd.append(s1.tolist())
                        p2s_lsd.append(s2.tolist())

            detected_lines = {
                'angles': angles_lsd,
                'dists': dists_lsd,
                'p1s': p1s_lsd,
                'p2s': p2s_lsd,
                'b1s': b1s_lsd,
                'b2s': b2s_lsd
            }
            print(detected_lines)

            with open(os.path.join(lines_output_folder, f"piece_{k:05d}.json"), 'w') as lj:
                json.dump(detected_lines, lj, indent=3)
            k += 1

# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='It creates images with min, ..., max number of lines/segments (from min to max)')
#     parser.add_argument('-min', '--min_lines', type=int, default=1, help='min number of lines')
#     parser.add_argument('-max', '--max_lines', type=int, default=10, help='max number of lines')
#     parser.add_argument('-hh', '--height', type=int, default=1920, help='height of the images')
#     parser.add_argument('-ww', '--width', type=int, default=1920, help='width of the images')
#     parser.add_argument('-i', '--imgs_per_line', type=int, default=10, help='number of images for each number of line')
#     parser.add_argument('-o', '--output', type=str, default='', help='output folder')
#     parser.add_argument('-t', '--type', type=str, default='segments', choices=['segments', 'lines', 'polylines'], help='choose type of features')
#     parser.add_argument('-th', '--thickness', type=int, default=1, help='thickness of the drawings')
#
#     args = parser.parse_args()
#     main(args)