import os 
import json 
import numpy as np 
import matplotlib.pyplot as plt 
import argparse 
from configs import unified_cfg as cfg
from configs import folder_names as fnames
from metrics.metrics_utils import get_best_anchor
import pdb 
import pandas as pd

def main(args):

    root_path = os.path.join(f"{fnames.output_dir}_{cfg.num_patches_side}x{cfg.num_patches_side}", args.dataset, args.puzzle)
    evaluation_folder = os.path.join(root_path, fnames.evaluation_folder_name)
    anc_folder = os.path.join(evaluation_folder, 'anchors_analysis')
    os.makedirs(anc_folder, exist_ok=True)
    # sorting problem (needs %02d)
    # everything_listed = os.listdir(output_folder)
    # eval_files = [eval_file for eval_file in everything_listed if 'evaluation_anchor' in eval_file]
    # eval_files.sort()

    eval_cur_img = np.zeros((64, 4))
    for j in range(0, 64):
        eval_file_json = f"evaluation_anchor{j}.json"
        eval_file_path = os.path.join(evaluation_folder, eval_file_json)
        with open(eval_file_path, 'r') as efp:
            eval_nums = json.load(efp) 
            eval_cur_img[j, 0] = eval_nums['correct']
            eval_cur_img[j, 1] = eval_nums['correct_vector']
            eval_cur_img[j, 2] = eval_nums['neighbours']
            eval_cur_img[j, 3] = eval_nums['pixel']

    anc_pd = pd.DataFrame()
    anc_pd['correct_perc'] = eval_cur_img[:,0]
    anc_pd['neighbours_perc'] = eval_cur_img[:,2]
    anc_pd['pixel_acc'] = 1 - eval_cur_img[:,3]
    anc_pd['pixel_err'] = eval_cur_img[:,3]
    anc_pd['correct_number'] = eval_cur_img[:,1]
    anc_pd.to_csv(os.path.join(anc_folder, f'anchor_analysis_{args.puzzle}.csv'))

    im_ref = plt.imread(os.path.join(fnames.data_path, args.dataset, fnames.images_folder, f"{args.puzzle}.jpg"))
    best_correct_img, best_correct_idx, best_correct_pos, best_correct_val = get_best_anchor(im_ref, eval_cur_img[:,0], args.num_pieces, cfg.piece_size, best='max')
    best_neighbours_img, best_neighbours_idx, best_neighbours_pos, best_neighbours_val = get_best_anchor(im_ref, eval_cur_img[:,2], args.num_pieces, cfg.piece_size, best='max')
    best_pixel_img, best_pixel_idx, best_pixel_pos, best_pixel_val = get_best_anchor(im_ref, eval_cur_img[:,3], args.num_pieces, cfg.piece_size, best='min')


    plt.figure(figsize=(32,32))
    plt.suptitle(f"{args.puzzle}")
    # \ncorrect pieces = {num_correct_pieces / (args.num_pieces**2) * 100:.03f}%\nneighbours = {neighbours_val * 100:.03f}%\nMSE = {MSError:.03f}", fontsize=52)
    plt.subplot(441)
    plt.title(f"{args.puzzle}", fontsize=24)
    plt.imshow(im_ref, cmap='gray')

    ### BEST
    plt.subplot(442)
    plt.title(f'best A for correct pieces (id:{best_correct_idx} (pos {best_correct_pos}))', fontsize=24)
    plt.imshow(best_correct_img, cmap='gray')
    plt.subplot(446)
    plt.title(f'solution for best anchor ({best_correct_val*100:.03f} %)', fontsize=24)
    solution_best_correct = plt.imread(os.path.join(evaluation_folder, f"evaluated_solution_anchor{best_correct_idx}.jpg"))
    plt.imshow(solution_best_correct)
    plt.subplot(443)
    plt.title(f'best A for neighbours (id:{best_neighbours_idx} (pos {best_neighbours_pos}))', fontsize=24)
    plt.imshow(best_neighbours_img, cmap='gray')
    plt.subplot(447)
    plt.title(f'solution for best anchor ({best_neighbours_val*100:.03f} %)', fontsize=24)
    solution_best_neighbours = plt.imread(os.path.join(evaluation_folder, f"evaluated_solution_anchor{best_neighbours_idx}.jpg"))
    plt.imshow(solution_best_neighbours)
    plt.subplot(444)
    plt.title(f'best A for pixelwise (id:{best_pixel_idx} (pos {best_pixel_pos}))', fontsize=24)
    plt.imshow(best_pixel_img, cmap='gray')
    plt.subplot(448)
    plt.title(f'solution for worst anchor ({(1-best_pixel_val)*100:.03f} %)', fontsize=24)
    solution_best_pixel = plt.imread(os.path.join(evaluation_folder, f"evaluated_solution_anchor{best_pixel_idx}.jpg"))
    plt.imshow(solution_best_pixel)

    ### WORST
    best_correct_img, best_correct_idx, best_correct_pos, best_correct_val = get_best_anchor(im_ref, eval_cur_img[:,0], args.num_pieces, cfg.piece_size, best='min')
    best_neighbours_img, best_neighbours_idx, best_neighbours_pos, best_neighbours_val = get_best_anchor(im_ref, eval_cur_img[:,2], args.num_pieces, cfg.piece_size, best='min')
    best_pixel_img, best_pixel_idx, best_pixel_pos, best_pixel_val = get_best_anchor(im_ref, eval_cur_img[:,3], args.num_pieces, cfg.piece_size, best='max')

    plt.subplot(4,4,10)
    plt.title(f'worst A for correct pieces (id:{best_correct_idx} (pos {best_correct_pos}))', fontsize=24)
    plt.imshow(best_correct_img, cmap='gray')
    plt.subplot(4, 4, 14)
    plt.title(f'solution for worst anchor ({best_correct_val*100:.03f} %)', fontsize=24)
    solution_best_correct = plt.imread(os.path.join(evaluation_folder, f"evaluated_solution_anchor{best_correct_idx}.jpg"))
    plt.imshow(solution_best_correct)
    plt.subplot(4,4,11)
    plt.title(f'worst A for neighbours (id:{best_neighbours_idx} (pos {best_neighbours_pos}))', fontsize=24)
    plt.imshow(best_neighbours_img, cmap='gray')
    plt.subplot(4,4,15)
    plt.title(f'solution for worst anchor ({best_neighbours_val*100:.03f} %)', fontsize=24)
    solution_best_neighbours = plt.imread(os.path.join(evaluation_folder, f"evaluated_solution_anchor{best_neighbours_idx}.jpg"))
    plt.imshow(solution_best_neighbours)
    plt.subplot(4,4,12)
    plt.title(f'worst A for pixelwise (id:{best_pixel_idx} (pos {best_pixel_pos}))', fontsize=24)
    plt.imshow(best_pixel_img, cmap='gray')
    plt.subplot(4,4,16)
    plt.title(f'solution for worst anchor ({(1-best_pixel_val)*100:.03f} %)', fontsize=24)
    solution_best_pixel = plt.imread(os.path.join(evaluation_folder, f"evaluated_solution_anchor{best_pixel_idx}.jpg"))
    plt.imshow(solution_best_pixel)

    if args.visualize is True:
        plt.show()
    else:
        outputpath = os.path.join(anc_folder, f'{args.puzzle}_anchor_analysis_visualization.png')
        plt.tight_layout()
        plt.savefig(outputpath)
  
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='........ ')  # add some discription
    parser.add_argument('-d', '--dataset', type=str, default='images_with_50_lines', help='dataset folder')                     # repair, wikiart, manual_lines, architecture
    parser.add_argument('-p', '--puzzle', type=str, default='image_17', help='puzzle folder')                             # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-n', '--num_pieces', type=int, default=8, help='number of pieces (per side)')                  # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-v', '--visualize', default=False, action='store_true', help='use to show the solution')       # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('--verbosity', type=int, default=1, help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')                 # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    
    args = parser.parse_args()

    main(args)  