import numpy as np 
import matplotlib.pyplot as plt 
import pdb 
import cv2 
from typing import List
from utils.puzzle_utils import PuzzlePiece

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

def get_visual_reconstruction_from(pixel_solution: List, pieces: List[PuzzlePiece], solution_params: dict, show_borders:bool=True, colormap_name:str='jet'):

    # TO BE REFACTORED 
    
    step = np.ceil(ppars.xy_step)
    #ang = ppars.theta_step # 360 / Z    
    ang = 360 / Z
    z_rot = np.arange(0, 360, ang)
    pos = fin_sol
    fin_im = np.zeros(((Y * step + (ppars.p_hs+1) * 2).astype(int), (X * step + (ppars.p_hs+1) * 2).astype(int), 3))
    borders_cmap = mpl.colormaps['jet'].resampled(len(pieces))
    if show_borders == True:
        # plt.ion()
        borders_cmap = mpl.colormaps['jet'].resampled(len(pieces))
        # deprecated
        # borders_cmap = mpl.cm.get_cmap('jet').resampled(len(pieces))
    for i in range(len(pieces)):
        image = pieces_files[pieces[i]]  # read image 1
        im_file = os.path.join(pieces_folder, image)

        Im0 = Image.open(im_file).convert('RGBA')
        Im = np.array(Im0) / 255.0
        Im1 = Image.open(im_file).convert('RGBA').split()
        alfa = np.array(Im1[3]) / 255.0
        Im = np.multiply(Im, alfa[:, :, np.newaxis])
        Im = Im[:, :, 0:3]

        cc = ppars.p_hs

        if np.sum(pos[i, :2])>0:

            ids = (pos[i, :2] * step + cc).astype(int)
            if pos.shape[1] == 3:
                rot = z_rot[pos[i, 2]]
                Im = rotate(Im, rot, reshape=False, mode='constant', order=0)

                if i == anc:
                    mask = (Im > 0.05).astype(np.uint8)
                    em = cv2.erode(mask, np.ones((5, 5)))
                    bordered_im = Im * em + (mask - em) * borders_cmap(i)[:3]
                    Im = bordered_im

                if show_borders == True:
                    mask = (Im > 0.05).astype(np.uint8)
                    em = cv2.erode(mask, np.ones((5, 5)))
                    bordered_im = Im * em + (mask - em) * borders_cmap(i)[:3]
                    Im = bordered_im
            if ppars.p_hs * 2 < ppars.piece_size:
                fin_im[ids[0] - cc:ids[0] + cc + 1, ids[1] - cc:ids[1] + cc + 1, :] = Im + fin_im[
                                                                                           ids[0] - cc:ids[0] + cc + 1,
                                                                                           ids[1] - cc:ids[1] + cc + 1,
                                                                                           :]
            else:
                fin_im[ids[0] - cc:ids[0] + cc, ids[1] - cc:ids[1] + cc, :] = Im + fin_im[ids[0] - cc:ids[0] + cc,
                                                                                   ids[1] - cc:ids[1] + cc, :]

        # if show_borders == True:
        #     plt.imshow(fin_im)
        #     breakpoint()
    return fin_im

def save_visual_reconstruction_to_file(solution: List):


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

    if return_vals == True:
        return image[y0:y1, x0:x1, :], x0, x1, y0, y1
    return image[y0:y1, x0:x1, :]

