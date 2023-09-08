from skimage.transform import hough_line
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib import cm
import pdb 
import math 



# def polar_cartesian_check():
# print("check polar/cart")
# print("polar")
# print(angles, dists)
# print('\n')
# cartesians = polar2cartesian(seg_img, angles, dists)
# print('cartesian')
# print(cartesians)
# print('\n\n')
# for j, cartline in enumerate(cartesians):
#     #cartline = line_pol2cart(angle, dist)
#     print(f'\noriginal polar: theta={angles[j]:.3f}, rho={dists[j]:.3f}')
#     print(f'converted to cartesian: pt1 = ({cartline[0]:.3f}, {cartline[1]:.3f}), pt2 = ({cartline[2]:.3f}, {cartline[3]:.3f})')
#     rho, theta = line_cart2pol(cartline[0:2], cartline[2:4])
#     print(f'polar: theta={theta:.3f}, rho={rho:.3f}')

# print('\n\n')
# pdb.set_trace()

# Convert polar coordinates to Cartesian coordinates and display the result
def polar2cartesian(image, angles, dists, show_image=False):

    line_length = max(image.shape[0], image.shape[1])
    if show_image:
        plt.imshow(image, cmap=cm.gray)
    lines = []
    for angle, dist in zip(angles, dists):
        (x1, y1) = dist * np.array([np.cos(angle), np.sin(angle)])

        slope = np.tan(angle + np.pi / 2)
        # Calculate the y-intercept (b)
        intercept = y1 - slope * x1

        # Generate x-values for the line
        x = np.linspace(x1, x1 + 1, line_length)  # Adjust the range as needed

        # Calculate y-values using the slope-intercept form
        y = slope * x + intercept

        # Get the coordinates of the end point
        x2, y2 = x[-1], y[-1]

        lines.append((x1, y1, x2, y2))
        #plt.axline((x1, y1), (x2,y2))
    if show_image:
        plt.tight_layout()
        plt.show()

    return lines

def cluster_lines_dbscan(image, angles, dists, epsilon, min_samples):
    line_points = polar2cartesian(image, angles, dists)
    #line_points_ = np.expand_dims(np.array(line_points), axis=1)

    # Convert line endpoints to a suitable format for DBSCAN
    points = np.array(line_points)

    # Apply DBSCAN to cluster line endpoints
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    # Find the most representative line in each cluster
    unique_labels = np.unique(labels)
    centroid_lines = []

    for label in unique_labels:
        #cluster_lines = line_points[labels == label]

        cluster_indices = np.where(labels == label)[0]
        cluster_lines = [line_points[idx] for idx in cluster_indices]

        # Calculate the representative line as the median of the lines in the cluster
        mean_line = np.mean(cluster_lines, axis=0)
        centroid_lines.append(mean_line)
    return centroid_lines

def line_cart2pol(pt1, pt2):

    x_diff = pt1[0] - pt2[0]
    y_diff = pt1[1] - pt2[1]
    theta = np.arctan(-(x_diff/(y_diff + 10**-5)))
    rho = pt1[0] * np.cos(theta) + pt1[1] * np.sin(theta)
    rho2 = pt2[0] * np.cos(theta) + pt2[1] * np.sin(theta)
    #print("checkcart2pol", theta, rho, rho2)
    #pdb.set_trace()
    return rho, theta
    
def draw_hough_lines(img, lines):
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            pt1, pt2 = line_pol2cart(rho=rho, theta=theta)
            cv2.line(img, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)
    return img

def draw_prob_hough_line(img, linesP):
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(img, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 1, cv2.LINE_AA)
    return img 

def display_unprocessed_hough_result(image,theta,d,h):

    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
              np.rad2deg(theta[-1] + angle_step),
              d[-1] + d_step, d[0] - d_step]
    ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(image, cmap=cm.gray)
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    for i, (angle, dist) in enumerate(zip(theta, d)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))

    plt.tight_layout()
    plt.show()
