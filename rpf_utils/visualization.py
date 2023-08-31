import matplotlib.pyplot as plt 
import pdb 
import cv2 
import numpy as np 

def save_vis(cm, pieces, path, title='', draw_figsize=(100, 100), all_rotation=False, img_format='jpg'):
    
    rotation_range = [0]
    if all_rotation is True:
        rotation_range = np.arange(cm.shape[2])
    
    for rr in rotation_range:
        #pdb.set_trace()
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
            axs[a, 0].imshow(cv2.cvtColor(pieces[a-1]['img'], cv2.COLOR_BGR2RGB))
            axs[a, 0].xaxis.set_visible(False)
            axs[a, 0].yaxis.set_visible(False)
            axs[a, 0].set_title(pieces[a-1]['id'], loc='left', fontsize=32)
        plt.tight_layout()
        plt.savefig(f"{path}_r{rr}.{img_format}")
        plt.close()