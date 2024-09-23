
import numpy as np
import matplotlib.colors
import os
import configs.folder_names as fnames
from PIL import Image
import time

import shapely
from shapely import transform
from shapely import intersection, segmentize
from shapely.affinity import rotate
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib import cm
import cv2

from puzzle_utils.shape_utils import place_on_canvas
from puzzle_utils.pieces_utils import crop_to_content


def compute_cost_using_segmentation_compatibility(idx1, idx2, pieces, mask_ij, ppars, sam_segmentator, verbosity=1):

    p = ppars['p']
    z_id = ppars['z_id']
    m = ppars['m']
    rot = ppars['rot']

    if verbosity > 1:
        print(f"Computing cost for pieces {idx1:>2} and {idx2:>2}")

    if idx1 == idx2:
        # set compatibility to -1:
        R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) - 1
    else:
        print(f"computing cost matrix for piece {idx1} vs piece {idx2}")
        candidate_values = np.sum(mask_ij > 0)

        R_cost_conf, R_cost_overlap  = segmentation_compatibility_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, sam_segmentator, verbosity=verbosity)
        print(f"computed cost matrix for piece {idx1} vs piece {idx2}")
      
        R_cost = R_cost_overlap

    return R_cost

#### NEW
## pairwise compatibility measure between two pieces with and without rotation
def segmentation_compatibility_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, \
        sam_segmentator, detect_on_crop=True, area_ratio=0.1, verbosity=1):

    R_cost_conf = np.zeros((m.shape[1], m.shape[1], len(rot)))
    R_cost_overlap = np.zeros((m.shape[1], m.shape[1], len(rot)))

    valid_points_mask = mask_ij > 0

    ### <hack>
    ### Remove the one that do not touch. This is an hack and should be removed
    neg_points_mask = mask_ij < 0

    zero_points_mask = mask_ij != 0

    valid_points_mask_eroded = np.zeros_like(valid_points_mask)

    for i in range(zero_points_mask.shape[2]):
        # plt.subplot(1,2,1)
        # plt.imshow(valid_points_mask[:,:,i])
        valid_points_mask_eroded[:,:,i] = cv2.erode(zero_points_mask[:,:,i].astype(np.uint8),np.ones((3,3))) - neg_points_mask[:,:,i]
        # plt.subplot(1,2,2)
        # plt.imshow(valid_points_mask_eroded[:,:,i])
        # plt.show()
    ### </hack>
    

    valid_points_idx = np.argwhere(valid_points_mask_eroded)
    

    # Generate images

    # One day this will be processed in batches, but that day is not today
    #images_batch = np.zeros((len(valid_points_idx),ppars.canvas_size,ppars.canvas_size,3),dtype=np.uint8)

    for k,idx in enumerate(valid_points_idx):

        iy,ix,t = tuple(idx)

        canv_cnt = ppars.canvas_size // 2
        grid = z_id + canv_cnt
        x_j_pixel, y_j_pixel = grid[iy, ix]

        # Place on canvas pairs of pieces given position
        center_pos = ppars.canvas_size // 2
        piece_i_on_canvas = place_on_canvas(pieces[idx1], (center_pos, center_pos), ppars.canvas_size, 0)
        piece_j_on_canvas = place_on_canvas(pieces[idx2], (x_j_pixel, y_j_pixel), ppars.canvas_size, t * ppars.theta_step)
        pieces_ij_on_canvas = piece_i_on_canvas['img'] + piece_j_on_canvas['img']
        #mask_ij_on_canvas = piece_i_on_canvas['mask'] + piece_j_on_canvas['mask']
        #pieces_ij_on_canvas/= np.clip(mask_ij_on_canvas,1,2).astype(float)
        #plt.imshow(pieces_ij_on_canvas)
        #plt.show()

        x0 = 0
        y0 = 0
        img = pieces_ij_on_canvas

        # if detect_on_crop == True:
        #     img, x0, x1, y0, y1  = crop_to_content(pieces_ij_on_canvas, return_vals=True)
            
        # plt.imshow(img)
        # plt.show()

        filename = f'./seg/mask_{idx1}vs{idx2}_y{iy}_x{ix}_r{t}.npy'

        if os.path.isfile(filename):
            print("loading segmentation from file...",end='',flush=True)
            masks = np.load(filename,allow_pickle=True)
            print('done')
        else:
            # Compute segmentation
            if verbosity > 1:
                print("segmenting...",end='',flush=True)
                start = time.time()

            #img_cv2 = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
            masks = sam_segmentator(np.uint8(img))

            if verbosity > 1:
                print(f"computed segmentation in {time.time()-start:.2f}s ({len(masks)} areas detected)")

            np.save(filename,masks)

        # Computing score

        score = 0
        threshold = 20

        for mask_dict in masks:
            mask = mask_dict['segmentation'].astype(np.uint8)
            # for every color channel
            for c in range(3):
                s1 = np.sum(mask * piece_i_on_canvas['img'][:,:,c])
                s2 = np.sum(mask * piece_j_on_canvas['img'][:,:,c])
                if s1 > threshold and s2 > threshold:
                    score += 1

        # Show the image with the segmentation superposed
        # plt.title(f"Masks: {len(masks)} Score: {score}")
        # plt.imshow(img)
        # show_anns(masks)
        # plt.axis('off')
        # plt.show()

        R_cost_conf[iy, ix, t] = score
        R_cost_overlap[iy, ix, t] = score

    return R_cost_conf, R_cost_overlap

# utility function to draw areas on top of image
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
