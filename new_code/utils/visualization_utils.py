import numpy as np 
import matplotlib.pyplot as plt
import cv2 
from typing import List, Optional
from utils.puzzle_utils import PuzzlePiece
import matplotlib as mpl
import scipy
from PIL import Image
import os

def save_compatibility_matrix_visualization_to_file(CM, pieces: List[PuzzlePiece], rot_step, output_folder, title='', draw_figsize=(100, 100), all_rotation=False, save_every=6, img_format='jpg', vmin=-1, vmax=1):
    
    os.makedirs(output_folder, exist_ok=True)
    rotation_range = np.arange(CM.shape[2])
    for rr in rotation_range:
        theta = rr * rot_step
        if all_rotation is True or (all_rotation is False and (rr % save_every) == 0):
            fig, axs = plt.subplots(CM.shape[3]+1, CM.shape[4]+1, figsize=draw_figsize) #, sharex=True, sharey=True)
            fig.suptitle(f"{title}(r{rr})", fontsize=44)  
            mapping_image = np.zeros_like(CM[:, :, 0, 0, 0])
            mapping_image[0, 0] = -1
            mapping_image[-1, -1] = 1
            axs[0, 0].set_title("only for colorbar", fontsize=14)
            mim = axs[0, 0].imshow(mapping_image, vmin=-1, vmax=1, cmap='RdYlGn')
            axs[0, 0].xaxis.set_visible(False)
            axs[0, 0].yaxis.set_visible(False)
            fig.colorbar(mim)
            for x_plot in range(1, CM.shape[3]+1):
                for y_plot in range(1, CM.shape[4]+1):
                    # if x_plot == 8 and y_plot == 6 and rr == 3:
                    #     breakpoint()
                    axs[x_plot, y_plot].imshow(CM[:, :, rr, x_plot-1, y_plot-1], vmin=vmin, vmax=vmax, cmap='RdYlGn')
                    axs[x_plot, y_plot].xaxis.set_visible(False)
                    axs[x_plot, y_plot].yaxis.set_visible(False)
                    
            for a in range(1, CM.shape[3]+1):
                axs[0, a].set_title(pieces[a-1]['id'], fontsize=32)
                axs[0, a].imshow(cv2.cvtColor(pieces[a-1]['img'], cv2.COLOR_BGR2RGB), vmin=vmin, vmax=vmax, cmap='RdYlGn')
                axs[0, a].xaxis.set_visible(False)
                axs[0, a].yaxis.set_visible(False)
                if theta > 0:
                    rotated_img = scipy.ndimage.rotate(pieces[a-1]['img'], theta, reshape=False, mode='constant')
                else:
                    rotated_img = pieces[a-1]['img']
                axs[a, 0].imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
                axs[a, 0].xaxis.set_visible(False)
                axs[a, 0].yaxis.set_visible(False)
                axs[a, 0].set_title(pieces[a-1]['id'], loc='left', fontsize=32)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"CM_r{rr}.{img_format}"))
            plt.close()


def reconstruct(pixel_solution,
                pieces: List[PuzzlePiece],
                dimension: Optional[tuple[int, int]] = None,
                confidence_threshold : float = 0,
                show_borders : bool = True,
                colormap_name : str = 'jet') -> np.ndarray:
    
    piece_size = pieces[0].data.image.shape

    if piece_size[0] != piece_size[1]:
        raise Exception('only square images of pieces are supported')

    # half piece sizes
    hps_x = piece_size[0]//2
    hps_y = piece_size[1]//2
    
    offset = 1 if hps_x * 2 < piece_size[0] else 0

    if dimension is None:
        # TODO calculate best dimensions from solution
        dimension = (len(pieces) * piece_size[0], len(pieces) * piece_size[1])
    
    image = np.zeros(dimension + (4,)) # (dimension[0], dimension[1], 4)

    # if show_borders == True:
    #     borders_cmap = mpl.colormaps[colormap_name].resampled(len(pieces))

    for i in range(len(pieces)):
        # TODO: confidence should be optional
        x, y, theta, confidence = pixel_solution[i]

        if confidence <= confidence_threshold:
            continue

        piece_img = pieces[i].data.image
        if theta != 0:
            piece_img = scipy.ndimage.rotate(piece_img, theta, reshape=False, mode='constant', order=0)

        image[x - hps_x:x + hps_x + offset, y - hps_y:y + hps_y + offset,:] += piece_img

        # if show_borders == True:
        #         mask = (Im > 0.05).astype(np.uint8)
        #         em = cv2.erode(mask, np.ones((5, 5)))
        #         bordered_im = Im * em + (mask - em) * borders_cmap(i)[:3]
        #         Im = bordered_im
           
    return image

def reconstruct_pil(
                solution,
                pieces: List[PuzzlePiece],
                dimension: Optional[tuple[int, int]] = None,
                confidence_threshold : float = 0,
                expand_on_rotate : bool = False,
                crop_to_content : bool = True,
                show_borders : bool = True,
                colormap_name : str = 'jet') -> np.ndarray:

    piece_size = pieces[0].data.image.shape

    if piece_size[0] != piece_size[1]:
        raise Exception('only square images of pieces are supported')
    
    if dimension is None:
        # TODO calculate best dimensions from solution
        dimension = (len(pieces) * piece_size[0], len(pieces) * piece_size[1])
    
    canvas = np.zeros(dimension + (4,), dtype=np.uint8) # (dimension[0], dimension[1], 4)
    canvas = Image.fromarray(canvas, mode="RGBA")

    for i in range(len(pieces)):
        # TODO: confidence should be optional
        #x, y, theta, confidence = pixel_solution[i]
        x, y, theta, confidence = map(float, solution[i])

        if confidence <= confidence_threshold:
            continue

        piece_img = pieces[i].data.image

        if piece_img.dtype in (np.float32, np.float64):
            piece_img = np.clip(piece_img, 0.0, 1.0) * 255
            piece_img = piece_img.astype(np.uint8)

        piece_img = Image.fromarray(piece_img, mode="RGBA")

        # plt.imshow(piece_img)
        # plt.show()

        if theta != 0:
            piece_img = piece_img.rotate(theta, expand=expand_on_rotate, fillcolor=(0,0,0,0))

        pos = (int(x - piece_img.width // 2), int(y - piece_img.height // 2))
        canvas.paste(piece_img, pos, mask=piece_img)  # Use the alpha channel as mask


    if crop_to_content:
        canvas = crop_to_content_pil(canvas)

    canvas = np.array(canvas)
    return canvas

def crop_to_content_pil(img: Image.Image, padding: int = 1) -> Image.Image:
    assert img.mode == "RGBA", "Image must be in RGBA mode"

    alpha = img.split()[3]
    bbox = alpha.getbbox()

    if not bbox:
        return img  # fully transparent, nothing to crop

    # Expand bbox by padding, making sure we stay within image bounds
    left = max(bbox[0] - padding, 0)
    upper = max(bbox[1] - padding, 0)
    right = min(bbox[2] + padding, img.width)
    lower = min(bbox[3] + padding, img.height)

    padded_bbox = (left, upper, right, lower)
    return img.crop(padded_bbox)


def crop_to_content(image:np.ndarray, padding:int=1, return_vals:bool=False, max_noise:int=0):

    if len(image.shape) > 2:
        x0 = np.clip(np.min(np.where(np.sum(image, axis=2) > max_noise)[1]) - padding, 0, image.shape[1])
        x1 = np.clip(np.max(np.where(np.sum(image, axis=2) > max_noise)[1]) + padding, 0, image.shape[1])
        y0 = np.clip(np.min(np.where(np.sum(image, axis=2) > max_noise)[0]) - padding, 0, image.shape[0])
        y1 = np.clip(np.max(np.where(np.sum(image, axis=2) > max_noise)[0]) + padding, 0, image.shape[0])
    else:
        x0 = np.min(np.where(image > max_noise)[1]) - padding
        x1 = np.max(np.where(image > max_noise)[1]) + padding
        y0 = np.min(np.where(image > max_noise)[0]) - padding
        y1 = np.max(np.where(image > max_noise)[0]) + padding

    cropped_image = image[y0:y1, x0:x1, :] if len(image.shape) == 3 else image[y0:y1, x0:x1]
    if return_vals:
        return cropped_image, x0, x1, y0, y1
    return cropped_image

##############################
# SAVE
def get_path_to_save_image(self):
    image_path = self.cfg.get_VIS_path()
    return image_path
