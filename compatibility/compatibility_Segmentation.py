
import numpy as np
import matplotlib.colors
import os
import configs.folder_names as fnames
from PIL import Image
import time

from matplotlib import pyplot as plt
import cv2
import warnings

from puzzle_utils.shape_utils import place_on_canvas
from puzzle_utils.pieces_utils import crop_to_content

import argparse


class Segmentator:
    def __init__():
        pass

    @staticmethod
    def add_args(parser):
        parser.add_argument('--seg_torch_device', type=str, help='torch device to use')
        # SAM parameters
        parser.add_argument('--seg_sam_type', type=str, default="mobile-sam", help='type of SAM model to use')
        parser.add_argument('--seg_sam_points_per_side', type=int, default=32, help='SAM points_per_side parameter')
        parser.add_argument('--seg_sam_stability_score_thresh', type=float, default=0.85, help='SAM stability_score_thresh parameter')
        # Save/Load parameters
        parser.add_argument('--seg_load_from_files',action=argparse.BooleanOptionalAction, default=False, help='Load segmentations from files')
        parser.add_argument('--seg_save_to_files',action=argparse.BooleanOptionalAction, default=False, help='Save segmentations to files')
        

        parser.add_argument('--seg_save_images',action=argparse.BooleanOptionalAction, default=True, help='Save images')

        # Debugging
        parser.add_argument('--seg_debug_break_after_each',action=argparse.BooleanOptionalAction, default=False, help='Set a breakpoint after each pair')

        parser.add_argument('--seg_only_pieces',nargs="*",type=int,default=[], help='Only work with these pieces')
        
        #parser.add_argument('--seg_sam_model_path', type=str, help='SAM checkpoint path (.pt file)')
        #parser.add_argument('--seg_sam_model_type', type=str, choices=['default','vit_h','vit_l','vit_b'], help='SAM model type')

    def __init__(self,ppars,args):
        
        # if args.seg_sam_model_path is None or args.seg_sam_model_type is None:
        #     raise Exception("You are trying to use SAM-based compatibility without specifying model path")
        
        ppars['seg_load_from_files'] = args.seg_load_from_files
        ppars['seg_save_to_files'] = args.seg_save_to_files

        ppars['seg_debug_break_after_each'] = args.seg_debug_break_after_each
        ppars['seg_save_images'] = args.seg_save_images
        ppars['seg_only_pieces'] = args.seg_only_pieces

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
        if args.seg_sam_type == 'sam_vit_b':
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            self.sam = sam_model_registry["vit_b"](checkpoint="../PretrainedModels/SAM/sam_vit_b_01ec64.pth").to(self.device)
        ##### Mobile SAM
        elif args.seg_sam_type == 'mobile_sam':
            from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry
            self.sam = sam_model_registry["vit_t"](checkpoint="../PretrainedModels/MobileSAM/mobile_sam.pt").to(self.device)
        else:
            raise Exception(f'seg_sam_type "{args.seg_sam_type}" not recognized')

        #### Common
        self.model_name = f'{args.seg_sam_type}_pps{args.seg_sam_points_per_side}_sst{args.seg_sam_stability_score_thresh}'
        ppars['model_name'] = self.model_name

        self.grid = build_point_grid(args.seg_sam_points_per_side)

        self.sam_amg = SamAutomaticMaskGenerator(self.sam,
                                                #points_per_side=args.seg_sam_points_per_side,
                                                points_per_side=None,
                                                stability_score_thresh=args.seg_sam_stability_score_thresh,
                                                point_grids=[self.grid]
                                                )

        ##### Meta (SAM 2)
        # from sam2.build_sam import build_sam2
        # from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        # sam2_checkpoint = "../sam2_hiera_large.pt"
        # model_cfg = "sam2_hiera_l.yaml"
        # sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        # sam_amg = SAM2AutomaticMaskGenerator(sam2)
        # sam_segmentator = sam_amg.generate

    def set_point_grid_mask(self,masks,scale,filter_size = 5):

        # half filter size
        hfs = (filter_size - 1) // 2

        def on_mask(p):
            p = np.uint(np.round(p * scale))

            for mask in masks:
                if (np.sum(mask[p[1]-hfs:p[1]+(hfs+1),p[0]-hfs:p[0]+(hfs+1)]) == (filter_size**2)):
                    return True
            return False
    
        
        sam_masked_grid = np.array([p for _,p in enumerate(self.grid) if on_mask(p)])

        self._set_point_grid(sam_masked_grid)

        return sam_masked_grid * scale

    def _set_point_grid(self,grid):
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

    only_pieces = set(ppars['seg_only_pieces'])

    R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) - 1

    if idx1 == idx2:
        # set compatibility to -1:
        return R_cost
    
    if len(only_pieces) > 0:
        if not idx1 in ppars['seg_only_pieces'] or not idx2 in ppars['seg_only_pieces']:
            print(f"Considering only pieces {ppars['seg_only_pieces']}. Skipping")
            return R_cost
        
    print(f"computing cost matrix for piece {idx1} vs piece {idx2}")
    R_cost  = segmentation_compatibility_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, segmentator, verbosity=verbosity)
    print(f"computed cost matrix for piece {idx1} vs piece {idx2}")
            
    
    # for debugging purposes
    if ppars['seg_debug_break_after_each'] or len(only_pieces) > 0:
        breakpoint()
    
    return R_cost

#### NEW
## pairwise compatibility measure between two pieces with and without rotation
def segmentation_compatibility_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, \
        segmentator : Segmentator, detect_on_crop=True, area_ratio=0.1, verbosity=1):

    
    R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))


    valid_points_mask = mask_ij > 0

    ### <hack>
    # we know no rotation
    # warnings.warn("Discard all rotation apart from the 0-th")
    # valid_points_mask[:,:,1:] = 0
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

    if verbosity > 1:
        print(f"found {len(valid_points_idx)} valid configurations")


    # One day this will be processed in batches, but that day is not today
    #images = np.zeros((len(valid_points_idx),ppars.canvas_size,ppars.canvas_size,3),dtype=np.uint8)

    folder = os.path.join(ppars['puzzle_root_folder'],f'segmentation/pairs/{idx1}vs{idx2}')
    img_folder  = os.path.join(folder,'img')
    seg_folder  = os.path.join(folder,'seg')

    os.makedirs(folder,exist_ok=True)

    os.makedirs(img_folder,exist_ok=True)
    os.makedirs(seg_folder,exist_ok=True)

    for k,(iy,ix,t) in enumerate(valid_points_idx):
        if verbosity > 1:
            print(f"processing {idx1}vs{idx2} y{iy} x{ix} r{t} ")

        # setting paths

        basename = f'pair_p{idx1}vs{idx2}_y{iy}_x{ix}_r{t}'

        filename_img = os.path.join(img_folder,basename + '.png')

        filename_img_grid = os.path.join(img_folder,f'{basename}_grid.png')
        filename_img_seg = os.path.join(img_folder,f'{basename}_seg_{segmentator.model_name}' + '.png')
        filename_img_masks = os.path.join(img_folder,f'{basename}_seg_{segmentator.model_name}_masks' + '.png')


        

        if verbosity > 1:
            print(f"generating image... ",end='',flush=True)
            start_img = time.time()

        center_pos = ppars.canvas_size // 2
        grid = z_id + center_pos
        x_j_pixel, y_j_pixel = grid[iy, ix]

        # Place on canvas pairs of pieces given position
        piece_i_on_canvas = place_on_canvas(pieces[idx1], (center_pos, center_pos), ppars.canvas_size, 0)
        piece_j_on_canvas = place_on_canvas(pieces[idx2], (x_j_pixel, y_j_pixel), ppars.canvas_size, t * ppars.theta_step)

        mask_i = piece_i_on_canvas['mask']
        mask_j = piece_j_on_canvas['mask']
        mask_ij = np.bool(mask_i) | np.bool(mask_j)

        overlap = np.bool(mask_i) & np.bool(mask_j)

        #breakpoint()

        img_ij_on_canvas = piece_i_on_canvas['img'] + piece_j_on_canvas['img']
        img_ij_on_canvas[overlap] = piece_i_on_canvas['img'][overlap]

        img_ij_on_canvas = np.uint8(img_ij_on_canvas)

        # enhance imag
        img_ij_on_canvas = np.uint8(enhance_img(img_ij_on_canvas))

        if verbosity > 1:
            print(f"done in {time.time()-start_img:.2f}s")

        # if detect_on_crop == True:
        #     img, x0, x1, y0, y1  = crop_to_content(pieces_ij_on_canvas, return_vals=True)


        #### Compute Compatibility ####
        dist = distance(mask_i,mask_j)
        if  dist > 0:
            if verbosity > 1:
                print(f"pieces are not touching (distance {dist}), skipping")
            R_cost[iy, ix, t] = 0
            continue
        print(f'distance: {distance(mask_i,mask_j)}')
        # breakpoint()


        # Setting points
        #sam_masked_grid_scaled = segmentator.grid
        sam_masked_grid_scaled = segmentator.set_point_grid_mask([mask_i,mask_j],ppars.canvas_size)
        
        save_img = ppars['seg_save_images']
        skip_seg = False
        if len(sam_masked_grid_scaled) == 0:
                warnings.warn(f'{basename}: SAM point grid does not contain any point')
                skip_seg = True
                save_img = True

        #if not os.path.isfile(filename_img):
        if save_img:
            cv2.imwrite(filename_img,cv2.cvtColor(img_ij_on_canvas, cv2.COLOR_RGB2BGR))

        if ppars['seg_save_images']:
            plt.title(f'Distance: {dist:.2f}')
            if len(sam_masked_grid_scaled) > 0:
                plt.scatter(sam_masked_grid_scaled[:, 0], sam_masked_grid_scaled[:, 1], color='m', s=0.1, marker='o')
            plt.imshow(img_ij_on_canvas)
            plt.axis('off')
            plt.savefig(filename_img_grid)
            plt.cla()


        # Segmentation

        if skip_seg:
            print("Skipping segmentation")
            R_cost[iy, ix, t] = 0
            continue

        filename_seg = os.path.join(seg_folder,basename + '.npy')

        if ppars['seg_load_from_files']:
            if verbosity > 1:
                print("loading segmentation... ",end='',flush=True)
                start_seg = time.time()
            
            masks = np.load(filename_seg,allow_pickle=True)

        else:
            # Compute segmentation
            
            if verbosity > 1:
                print("computing segmentation... ",end='',flush=True)
                start_seg = time.time()  

                print(f"feeding SAM with {len(sam_masked_grid_scaled)} points")

            
            
            masks = segmentator(img_ij_on_canvas)

        if verbosity > 1:
            print(f"done in {time.time()-start_seg:.2f}s ({len(masks)} areas found)")

        if ppars['seg_save_to_files'] and not ppars['seg_load_from_files']:
            # Save masks to files

            if verbosity > 1:
                print("saving to file... ",end='',flush=True)
            
            np.save(filename_seg,masks)

            if verbosity > 1:
                print("done")


        # Computing score

        # if iy == 0  and ix == 4 and t == 1:
        #     breakpoint()
        score = compute_score(mask_i,mask_j,masks)
    

        if ppars['seg_save_images']:
        # Show the black and white image with the segmentation superposed
            plt.title(f"{basename}\nMasks: {len(masks)} Score: {score}")      
            plt.imshow(to_grey_scale(img_ij_on_canvas))
            show_anns(masks)
            plt.axis('off')
            plt.savefig(filename_img_seg)
            plt.cla()

        #breakpoint()

        # show_image_with_masks(img_ij_on_canvas,masks)
        # plt.savefig(filename_img_masks)
        # plt.cla()
        # plt.clf()


        R_cost[iy, ix, t] = score

    return R_cost

# computes the score
def compute_score(mask_i,mask_j,masks,rel_thresh=0.05):
    score = 0
    n = 0

    # the threshold is relative to the minimum between the areas of the pieces
    threshold = rel_thresh * np.min([mask_i.sum(),mask_j.sum()])

    # try to drop last one
    # print("****************************DROPPING LAST MASK*************************")
    # masks = masks[:-1]

    for mask_dict in masks:
        mask = mask_dict['segmentation'].astype(np.uint8)
        s1 = np.sum(mask * mask_i)
        s2 = np.sum(mask * mask_j)
    
        if s1 > threshold and s2 > threshold:
            n += 0.2
    
    score = n #/ len(masks)

    return score

# computes the distance between binary maps
def distance(mask_i,mask_j):

    def find_contour(mask):
        contour, _ = cv2.findContours(np.uint8(mask*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return np.array(contour).squeeze()

    contour_i = find_contour(mask_i)
    contour_j = find_contour(mask_j)

    d = np.min(np.sqrt(np.sum((contour_i[:, np.newaxis, :] - contour_j[np.newaxis, :, :]) ** 2, axis=2)))

    return d




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


def to_grey_scale(img,n_channels = 3):
    bw_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    bw_img_3c = np.dstack([bw_img]*n_channels)
    return bw_img_3c


def show_image_with_masks(image, masks_dict, rows=None, cols=None):
    """
    Plots the image with different masks overlaid in subplots.

    Parameters:
    - image: The original image as a NumPy array (HxWxC for RGB, HxW for grayscale).
    - masks: A list or array of masks. Each mask should be a NumPy array of the same height and width as the image.
    - rows: Number of rows in the subplot grid (optional).
    - cols: Number of columns in the subplot grid (optional).
    """
    masks = [mask_dict['segmentation'] for mask_dict in masks_dict]
    num_masks = len(masks)

    # If rows and cols aren't provided, make a square-ish grid
    if rows is None or cols is None:
        cols = int(np.ceil(np.sqrt(num_masks)))
        rows = int(np.ceil(num_masks / cols))

    # Create the subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()  # Flatten to easily iterate through all axes

    for i, mask in enumerate(masks):
        # Overlay the mask over the image (assumes mask is binary, use mask > 0 if not)
        masked_image = np.copy(image)
        
        # If the image is grayscale (2D), convert it to RGB
        if len(image.shape) == 2:
            masked_image = np.stack([image, image, image], axis=-1)

        # Apply the mask with transparency (red overlay on mask areas)
        red_mask = np.zeros_like(masked_image)
        red_mask[..., 0] = 255  # Red channel

        # Use the mask as the alpha channel to combine
        alpha = mask[..., np.newaxis] * 0.5  # Adjust transparency here (0.5 means 50% transparent)
        masked_image = (1 - alpha) * masked_image + alpha * red_mask

        # Plot the masked image
        axes[i].imshow(masked_image.astype(np.uint8))
        axes[i].set_title(f"Mask {i+1}")
        axes[i].axis('off')

    # Turn off remaining empty axes (if any)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()


def enhance_img(img,sat_factor=3,alpha = 1.3,beta = 0):

    # Convert the image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Split the HSV channels
    h, s, v = cv2.split(hsv)

    # Increase saturation by a factor of 1.5 (you can adjust this factor)
    s = cv2.multiply(s, sat_factor)

    # Clip the values to stay within valid range [0, 255]
    s = np.clip(s, 0, 255).astype(np.uint8)

    # Merge the channels back
    hsv_boosted = cv2.merge([h, s, v])

    # Convert back to BGR color space
    img_saturation_boosted = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2RGB)

    # Increase contrast using alpha-beta transformation
    # Alpha > 1 increases contrast, Beta adjusts brightness


    img_contrast_boosted = cv2.convertScaleAbs(img_saturation_boosted, alpha=alpha, beta=beta)

    return img_contrast_boosted
