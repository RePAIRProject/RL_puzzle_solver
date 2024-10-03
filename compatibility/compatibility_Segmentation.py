
import numpy as np
import matplotlib.colors
import os
import configs.folder_names as fnames
from PIL import Image
import time
import torch

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


class Segmentator:
    def __init__():
        pass

    @staticmethod
    def add_args(parser):
        parser.add_argument('--seg_torch_device', type=str, help='torch device')
        parser.add_argument('--seg_sam_points_per_side', type=int, default=32, help='')
        #parser.add_argument('--seg_sam_model_path', type=str, help='SAM checkpoint path (.pt file)')
        #parser.add_argument('--seg_sam_model_type', type=str, choices=['default','vit_h','vit_l','vit_b'], help='SAM model type')

    def __init__(self,ppars,args):
        
        # if args.seg_sam_model_path is None or args.seg_sam_model_type is None:
        #     raise Exception("You are trying to use SAM-based compatibility without specifying model path")
        
        #ppars['seg_sam_model_type'] = args.seg_sam_model_type
        #ppars['seg_sam_model_path'] = args.seg_sam_model_path

        if args.seg_torch_device != "":
            self.device = args.seg_torch_device
        else:
            import torch
            # Automatically select best device if not specified
            if torch.backends.mps.is_available():
                # macOS specific
                self.device = "mps"
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        
        ppars['seg_torch_device'] = args.seg_torch_device

        ##### SAM
        ppars['seg_model'] = 'sam_vit_b'
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        self.sam = sam_model_registry["vit_b"](checkpoint="../PretrainedModels/SAM/sam_vit_b_01ec64.pth").to(self.device)

        ##### Mobile SAM
        # ppars['seg_model'] = 'mobile_sam'
        # from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry
        # self.sam = sam_model_registry["vit_h"](checkpoint="../PretrainedModels/MobileSAM/mobile_sam.pt").to(self.device)
        

        #### Common
        self.points_per_side = args.seg_sam_points_per_side
        self.sam_amg = SamAutomaticMaskGenerator(self.sam,
                                                points_per_side=self.points_per_side,
                                                stability_score_thresh=0.85,
                                                )

        ##### Meta (SAM 2)
        # from sam2.build_sam import build_sam2
        # from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        # sam2_checkpoint = "../sam2_hiera_large.pt"
        # model_cfg = "sam2_hiera_l.yaml"
        # sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        # sam_amg = SAM2AutomaticMaskGenerator(sam2)
        # sam_segmentator = sam_amg.generate

    def set_point_grid(self,grid):
        self.sam_amg.point_grids = [grid]

    def compute(self,input):
        return self.sam_amg.generate(input)
    
    def __call__(self,input):
        return self.compute(input)


#########################################

# Taken from SAM code
def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

#########################################

def compute_cost_using_segmentation_compatibility(idx1, idx2, pieces, mask_ij, ppars, segmentator, verbosity=1):

    p = ppars['p']
    z_id = ppars['z_id']
    m = ppars['m']
    rot = ppars['rot']

    if idx1 == idx2:
        # set compatibility to -1:
        R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) - 1
    else:
        print(f"computing cost matrix for piece {idx1} vs piece {idx2}")
        if idx1 == 5 and idx2 == 8:
            R_cost  = segmentation_compatibility_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, segmentator, verbosity=verbosity)
            print(f"computed cost matrix for piece {idx1} vs piece {idx2}")
        else:
            R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))

    return R_cost

#### NEW
## pairwise compatibility measure between two pieces with and without rotation
def segmentation_compatibility_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, \
        segmentator, detect_on_crop=True, area_ratio=0.1, verbosity=1):

    
    R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))


    valid_points_mask = mask_ij > 0

    ### <hack>
    valid_points_mask[:,:,1:] = 0
    ### </hack>

    ### <hack>
    ### Remove the one that do not touch. This is an hack and should be removed
    # neg_points_mask = mask_ij < 0

    # zero_points_mask = mask_ij != 0

    # valid_points_mask_eroded = np.zeros_like(valid_points_mask)

    # for i in range(zero_points_mask.shape[2]):
    #     # plt.subplot(1,2,1)
    #     # plt.imshow(valid_points_mask[:,:,i])
    #     valid_points_mask_eroded[:,:,i] = cv2.erode(zero_points_mask[:,:,i].astype(np.uint8),np.ones((3,3))) - neg_points_mask[:,:,i]
    #     # plt.subplot(1,2,2)
    #     # plt.imshow(valid_points_mask_eroded[:,:,i])
    #     # plt.show()
    # ### </hack>
    

    valid_points_idx = np.argwhere(valid_points_mask)

    # Filter valid points to speed up testing
    # filter = np.random.choice(len(valid_points_idx),3)
    # valid_points_idx = valid_points_idx[filter]

    if verbosity > 1:
        print(f"found {len(valid_points_idx)} valid points")
    

    # Generate images

    # if verbosity > 1:
    #     print(f"generating images",end='',flush=True)
    # One day this will be processed in batches, but that day is not today

    images = np.zeros((len(valid_points_idx),ppars.canvas_size,ppars.canvas_size,3),dtype=np.uint8)

    folder = os.path.join(ppars['puzzle_root_folder'],f'seg/pairs/{idx1}vs{idx2}')

    os.makedirs(folder,exist_ok=True)

    for k,(iy,ix,t) in enumerate(valid_points_idx):

        filename_img = os.path.join(folder,f'pair_img_p{idx1}vs{idx2}_y{iy}_x{ix}_r{t}.png')

        if verbosity > 1:
            print(f"generating image... ",end='',flush=True)
            start_img = time.time()

        center_pos = ppars.canvas_size // 2
        grid = z_id + center_pos
        x_j_pixel, y_j_pixel = grid[iy, ix]

        # Place on canvas pairs of pieces given position
        piece_i_on_canvas = place_on_canvas(pieces[idx1], (center_pos, center_pos), ppars.canvas_size, 0)
        piece_j_on_canvas = place_on_canvas(pieces[idx2], (x_j_pixel, y_j_pixel), ppars.canvas_size, t * ppars.theta_step)
        img_ij_on_canvas = piece_i_on_canvas['img'] + piece_j_on_canvas['img']

        mask_i = piece_i_on_canvas['mask']
        mask_j = piece_j_on_canvas['mask']
        mask_ij = np.bool(mask_i) | np.bool(mask_j)

        x0 = 0
        y0 = 0
        images[k] = img_ij_on_canvas

        if verbosity > 1:
            print(f"done in {time.time()-start_img:.2f}s")

        # if verbosity > 1:
        #     print(".",end='',flush=True)

        #
        # if not os.path.isfile(filename_img):
        #     cv2.imwrite(filename_img,cv2.cvtColor(images[k], cv2.COLOR_RGB2BGR))
        

        # if detect_on_crop == True:
        #     img, x0, x1, y0, y1  = crop_to_content(pieces_ij_on_canvas, return_vals=True)
            
        # plt.imshow(cv2.cvtColor(images[k], cv2.COLOR_RGB2BGR))
        # plt.show()

    # Convert images to 8-bit
    # images = np.uint8(images)

    #images = torch.from_numpy(images).permute(0,3,2,1) # no need to use .to(device)
    # if verbosity > 1:
    #     print("done")

    ######### Segmentation

    # for k,(iy,ix,t) in enumerate(valid_points_idx):

        
        # Compute segmentation
        if verbosity > 1:
            print("computing segmentation... ",end='',flush=True)
            start_seg = time.time()

        #img_cv2 = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)

        grid = build_point_grid(segmentator.points_per_side)

        # plt.scatter(grid[:, 0], grid[:, 1], color='red', s=10, marker='o')
        # plt.savefig(filename_img)
        # plt.cla()

        def on_pieces(p):
            q = np.round(p * ppars.canvas_size)
            q = np.uint(q)
            return mask_ij[q[0],q[1]]
            #return (mask_i[p[0],p[1]] == 1) or (mask_j[p[0],p[1]] == 1)
        
        #breakpoint()
        
        # point_list = []
        # for _,point in enumerate(grid):
        #     if on_pieces(point):
        #         print(f"{point} -> ok")
        #         point_list.append(point)
        
        filtered_grid = np.array([p for _,p in enumerate(grid) if on_pieces(p)])

        print(f"feeding SAM with {len(filtered_grid)} points")
        # segmentator.set_point_grid(filtered_grid)

        if len(filtered_grid) == 0:
            continue
        
        # masks = segmentator(np.uint8(images[k]))

        # if verbosity > 1:
        #     print(f"done in {time.time()-start_seg:.2f}s ({len(masks)} areas detected)")

        #breakpoint()

        # Computing score

        score = 0
        threshold = 50

        # for mask_dict in masks:
        #     mask = mask_dict['segmentation'].astype(np.uint8)
        #     s1 = np.sum(mask * mask_i)
        #     s2 = np.sum(mask * mask_j)
        #     if s1 > threshold and s2 > threshold:
        #         score += 1

        real_grid = filtered_grid*ppars.canvas_size

        # Show the image with the segmentation superposed
        #plt.title(f"Masks: {len(masks)} Score: {score}")
        plt.imshow(images[k])
        #plt.imshow(mask_ij)
        plt.scatter(real_grid[:, 0], real_grid[:, 1], color='white', s=5, marker='o')
        #show_anns(masks)
        #plt.axis('off')
        plt.savefig(filename_img)
        plt.cla()
        breakpoint()

        R_cost[iy, ix, t] = score

    breakpoint()

    return R_cost

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
