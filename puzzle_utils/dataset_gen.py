import numpy as np
import cv2 
import random

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

def create_random_image(line_type, num_lines, width, height, is_closed = False, thickness=1, color=(0, 0, 0)):

    img2draw = np.ones(shape=(width, height), dtype=np.uint8) * 255
    img_shape = [width, height]

    all_lines = np.zeros((num_lines, 4))

    if line_type == 'mix':
        for j in range(num_lines):
            if j == 0:
                axis1, axis2 = random.sample(range(0, 4), 2)  # select two axis
                p1 = generate_random_point(img_shape, distribution='uniform', on_axis=axis1)
                p2 = generate_random_point(img_shape, distribution='uniform', on_axis=axis2)
                img2draw = cv2.line(img2draw, p1, p2, color=color, thickness=thickness)
            
            else:
                choice = np.random.uniform(7)
                if choice < 2:
                    p1 = generate_random_point(img_shape, distribution='uniform')
                    p2 = generate_random_point(img_shape, distribution='uniform')
                    img2draw = cv2.line(img2draw, p1, p2, color=color, thickness=thickness)
                elif choice < 4:
                    axis1, axis2 = random.sample(range(0, 4), 2)  # select two axis
                    p1 = generate_random_point(img_shape, distribution='uniform')
                    p2 = generate_random_point(img_shape, distribution='uniform', on_axis=axis2)
                    img2draw = cv2.line(img2draw, p1, p2, color=color, thickness=thickness)
                elif choice < 6:
                    axis1, axis2 = random.sample(range(0, 4), 2)  # select two axis
                    p1 = generate_random_point(img_shape, distribution='uniform', on_axis=axis1)
                    p2 = generate_random_point(img_shape, distribution='uniform', on_axis=axis2)
                    img2draw = cv2.line(img2draw, p1, p2, color=color, thickness=thickness)
                else:
                    p1 = p2 # continue from last point (kind of polyline)
                    p2 = generate_random_point(img_shape, distribution='uniform')
                    img2draw = cv2.line(img2draw, p1, p2, color=color, thickness=thickness)

            all_lines[j, 0:2] = p1
            all_lines[j, 2:4] = p2


    elif line_type == 'polylines':
        n_lines = num_lines+1
        pts = np.zeros((n_lines, 2), dtype=np.int32)
        for k in range(n_lines):
            r_ax = np.round(np.clip(np.random.uniform(-4, 3.7), -1, 3)).astype(int)
            pts[k, :] = generate_random_point(img_shape, distribution='uniform', on_axis=r_ax)
        img2draw = cv2.polylines(img2draw, [pts], isClosed=is_closed, color=color, thickness=thickness)

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