import matplotlib.pyplot as plt 
import pdb 
import cv2 
import numpy as np 
import scipy

def save_vis(cm, pieces, rot_step, path, title='', draw_figsize=(100, 100), all_rotation=False, save_every=6, img_format='jpg', vmin=-1, vmax=1):
    
    rotation_range = np.arange(cm.shape[2])
    for rr in rotation_range:
        theta = rr * rot_step
        if all_rotation is True or (all_rotation is False and (rr % save_every) == 0):
            fig, axs = plt.subplots(cm.shape[3]+1, cm.shape[4]+1, figsize=draw_figsize) #, sharex=True, sharey=True)
            fig.suptitle(title, fontsize=44)  
            mapping_image = np.zeros_like(cm[:, :, 0, 0, 0])
            mapping_image[0, 0] = -1
            mapping_image[-1, -1] = 1
            axs[0, 0].set_title("only for colorbar", fontsize=18)
            mim = axs[0, 0].imshow(mapping_image, vmin=-1, vmax=1, cmap='RdYlGn')
            axs[0, 0].xaxis.set_visible(False)
            axs[0, 0].yaxis.set_visible(False)
            fig.colorbar(mim)
            for x_plot in range(1, cm.shape[3]+1):
                for y_plot in range(1, cm.shape[4]+1):
                    axs[x_plot, y_plot].imshow(cm[:, :, rr, x_plot-1, y_plot-1], vmin=vmin, vmax=vmax, cmap='RdYlGn')
                    axs[x_plot, y_plot].xaxis.set_visible(False)
                    axs[x_plot, y_plot].yaxis.set_visible(False)
            for a in range(1, cm.shape[3]+1):
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
            plt.savefig(f"{path}_r{rr}.{img_format}")
            plt.close()

            
def reconstruct_puzzle_vis(p_final, pieces_folder, ppars, suffix=''):

    Y, X, Z, noPatches = p_final.shape
    I = np.zeros((noPatches, 1))
    for j in range(noPatches):
        pj_final = p_final[:, :, :, j]
        _, I[j, 0] = np.max(pj_final), np.argmax(pj_final)
    i1, i2, i3 = np.unravel_index(I.astype(int), p_final[:, :, :, 1].shape)
    pos = np.concatenate((i1, i2, i3), axis=1)
    step = np.ceil(ppars.xy_step)
    ang = 360 / Z
    z_rot = np.arange(0, 360, ang)
    fin_im = np.zeros(((Y * step + ppars.p_hs).astype(int), (X * step + ppars.p_hs).astype(int), 3))

    pieces_names_list = os.listdir(pieces_folder)
    for i in range(len(pieces_names_list)):
        image_name = f"piece_{i:04d}"
        if suffix != '':
            image_name = image_name + suffix
        image_name = image_name + ".png"

        im_file = os.path.join(pieces_folder, image_name)

        Im0 = Image.open(im_file).convert('RGBA')
        Im = np.array(Im0) / 255.0
        Im1 = Image.open(im_file).convert('RGBA').split()
        alfa = np.array(Im1[3]) / 255.0
        Im = np.multiply(Im, alfa[:, :, np.newaxis])
        Im = Im[:, :, 0:3]

        cc = ppars.p_hs
        ids = (pos[i, :2] * step + cc).astype(int)
        if pos.shape[1] == 3:
            rot = z_rot[pos[i, 2]]
            Im = rotate(Im, rot, reshape=False, mode='constant')
        fin_im[ids[0]-cc:ids[0]+cc, ids[1]-cc:ids[1]+cc, :] = Im+fin_im[ids[0]-cc:ids[0]+cc, ids[1]-cc:ids[1]+cc, :]
    return fin_im