import cv2 
import numpy as np 
import random 
import pdb 

def create_random_image(line_type, num_lines, width, height, lines_length=10, isClosed=False, thickness=1, color=(0,0,0)):

    img2draw = np.ones(shape=(width, height), dtype=np.uint8) * 255
    img_shape = [width, height]
    for j in range(num_lines):
        if line_type == 'segments':
            p1 = generate_random_point(img_shape, distribution='uniform')
            p2 = generate_random_point(img_shape, distribution='uniform')
            img2draw = cv2.line(img2draw, p1, p2, color=color, thickness=thickness)
        elif line_type == 'lines':
            axis1, axis2 = random.sample(range(0,4), 2) # select two axis 
            p1 = generate_random_point(img_shape, distribution='uniform', on_axis=axis1)
            p2 = generate_random_point(img_shape, distribution='uniform', on_axis=axis2)
            img2draw = cv2.line(img2draw, p1, p2, color=color, thickness=thickness)
        elif line_type == 'polylines':
            pts = np.zeros((lines_length, 2), dtype=np.int32)
            for k in range(lines_length):
                pts[k, :] = generate_random_point(img_shape, distribution='uniform')
            img2draw = cv2.polylines(img2draw, [pts], isClosed=isClosed, color=color, thickness=thickness)

    return img2draw

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