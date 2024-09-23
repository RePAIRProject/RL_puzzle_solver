
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
        parser.add_argument('--sam_model_path', type=str, help='SAM checkpoint path (.pt file)')
        parser.add_argument('--sam_model_type', type=str, choices=['default','vit_h','vit_l','vit_b'], help='SAM model type')

    def __init__(self,ppars,args):
        
        if args.sam_model_path is None or args.sam_model_type is None:
            raise Exception("You are trying to use SAM-based compatibility without specifying model path")
        
        ppars['sam_model_type'] = args.sam_model_type
        ppars['sam_model_path'] = args.sam_model_path

        import torch
        # Automatically select best device for torch
        if torch.backends.mps.is_available():
            # macOS specific
            self.device = "mps"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        ##### Meta (SAM)
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        self.sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_model_path).to(self.device)
        self.sam_amg = SamAutomaticMaskGenerator(self.sam)

        ##### Meta (SAM 2)
        # from sam2.build_sam import build_sam2
        # from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        # sam2_checkpoint = "../sam2_hiera_large.pt"
        # model_cfg = "sam2_hiera_l.yaml"
        # sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        # sam_amg = SAM2AutomaticMaskGenerator(sam2)
        # sam_segmentator = sam_amg.generate

        ##### Ultralitycs
        # from ultralytics import SAM
        # sam_segmentator = SAM(args.sam_model_path)
        # sam_segmentator.info()
        

        ##### Hugginface
        # from transformers import SamModel, SamProcessor
        # processor = SamProcessor.from_pretrained("facebook/sam-vit-q")
        # model = SamModel.from_pretrained("facebook/sam-vit-q")

    def compute(self,input):
        return self.sam_amg.generate(input)
    
    def __call__(self,input):
        return self.compute(input)


#########################################

def compute_cost_using_segmentation_compatibility(idx1, idx2, pieces, mask_ij, ppars, segmentator, verbosity=1):

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

        R_cost_conf, R_cost_overlap  = segmentation_compatibility_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, segmentator, verbosity=verbosity)
        print(f"computed cost matrix for piece {idx1} vs piece {idx2}")
      
        R_cost = R_cost_overlap

    return R_cost

#### NEW
## pairwise compatibility measure between two pieces with and without rotation
def segmentation_compatibility_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, \
        segmentator, detect_on_crop=True, area_ratio=0.1, verbosity=1):

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

    if verbosity > 1:
        print(f"generating images",end='',flush=True)
    # One day this will be processed in batches, but that day is not today
    images = np.zeros((len(valid_points_idx),ppars.canvas_size,ppars.canvas_size,3),dtype=np.uint8)

    for k,(iy,ix,t) in enumerate(valid_points_idx):

        filename_img = f'./seg/pairs/pair_img_p{idx1}vs{idx2}_y{iy}_x{ix}_r{t}.jpg'

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
        images[k] = pieces_ij_on_canvas

        if verbosity > 1:
            print(".",end='',flush=True)

        #print('saving image')
        cv2.imwrite(filename_img,cv2.cvtColor(images[k], cv2.COLOR_RGB2BGR))
        

        # if detect_on_crop == True:
        #     img, x0, x1, y0, y1  = crop_to_content(pieces_ij_on_canvas, return_vals=True)
            
        # plt.imshow(img)
        # plt.show()

    # Convert images to 8-bit
    images = np.uint8(images)

    #images = torch.from_numpy(images).permute(0,3,2,1) # no need to use .to(device)

    if verbosity > 1:
        print("done")

    for k,(iy,ix,t) in enumerate(valid_points_idx):

        filename = f'./seg/masks/mask_p{idx1}vs{idx2}_y{iy}_x{ix}_r{t}.npy'

        #if False:
        if os.path.isfile(filename):
            print(f"loading segmentation from file '{filename}'... ",end='',flush=True)
            masks = np.load(filename,allow_pickle=True)
            print('done')
        else:
            # Compute segmentation
            if verbosity > 1:
                print("computing segmentation... ",end='',flush=True)
                start = time.time()

            #img_cv2 = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
            
            masks = segmentator(images[k])

            if verbosity > 1:
                print(f"done in {time.time()-start:.2f}s ({len(masks)} areas detected)")

            np.save(filename,masks)

        #breakpoint()

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
        plt.title(f"Masks: {len(masks)} Score: {score}")
        plt.imshow(images[k])
        show_anns(masks)
        plt.axis('off')
        plt.show()

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
