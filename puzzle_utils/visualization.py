import matplotlib.pyplot as plt 
import pdb 
import cv2 
import numpy as np 
import scipy

def save_vis(cm, pieces, path, rot_step, title='', draw_figsize=(100, 100), all_rotation=False, save_every=6, img_format='jpg'):
    
    rotation_range = np.arange(cm.shape[2])
    for rr in rotation_range:
        theta = rr * rot_step
        if all_rotation is False and (rr % save_every) == 0:
            fig, axs = plt.subplots(cm.shape[3]+1, cm.shape[4]+1, figsize=draw_figsize) #, sharex=True, sharey=True)
            fig.suptitle(title, fontsize=44)  
            for x_plot in range(1, cm.shape[3]+1):
                for y_plot in range(1, cm.shape[4]+1):
                    axs[x_plot, y_plot].imshow(cm[:, :, rr, x_plot-1, y_plot-1], vmin=-2, vmax=1)
                    axs[x_plot, y_plot].xaxis.set_visible(False)
                    axs[x_plot, y_plot].yaxis.set_visible(False)
            for a in range(1, cm.shape[3]+1):
                axs[0, a].set_title(pieces[a-1]['id'], fontsize=32)
                axs[0, a].imshow(cv2.cvtColor(pieces[a-1]['img'], cv2.COLOR_BGR2RGB))
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