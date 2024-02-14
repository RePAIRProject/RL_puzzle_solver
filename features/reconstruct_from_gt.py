import os, json  
import numpy as np 
from configs import folder_names as fnames
from puzzle_utils.pieces_utils import place_at, crop_to_content
import scipy
import argparse 
import matplotlib.pyplot as plt 
import cv2
import pdb 

def main(args):

    puzzle_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset, args.puzzle)
    gt_path = os.path.join(puzzle_folder, f"{fnames.ground_truth}.json")
    with open(gt_path, 'r') as gtj:
        gt = json.load(gtj)

    pars_path = os.path.join(puzzle_folder, f"parameters_{args.puzzle}.json")
    with open(pars_path, 'r') as pj:
        img_pars = json.load(pj)

    pieces_folder = os.path.join(puzzle_folder, fnames.pieces_folder)
    num_pieces = len(os.listdir(pieces_folder))
    empty_space = np.floor(np.sqrt(num_pieces)).astype(int) * img_pars['piece_size']
    canvas_size = np.round(img_pars['size'][0] + empty_space).astype(int)
    canvas = np.zeros((canvas_size, canvas_size, img_pars['size'][2]))
    center_pos = np.asarray([canvas.shape[0]//2, canvas.shape[1]//2])
    for j, pk in enumerate(gt.keys()):
        img_path = os.path.join(pieces_folder, f"{pk}.png")
        piece_img = cv2.imread(img_path)
        if j == 0:
            print(f"reference piece: {pk}")
            if gt[pk]['rotation'] != 0:
                print(f'rotating piece {pk} of {gt[pk]["rotation"]} degrees')
                piece_img = scipy.ndimage.rotate(piece_img, gt[pk]['rotation'])
            canvas = place_at(piece_img, canvas, center_pos)
            ref = pk
        elif j > 0:
            # relative shift is x,y (the method place_at takes care of inverting)
            relative_shift = -np.asarray(gt[pk]['translation']) + np.asarray(gt[ref]['translation']) + center_pos
            if gt[pk]['rotation'] != 0:
                print(f'rotating piece {pk} of {gt[pk]["rotation"]} degrees')
                piece_img = scipy.ndimage.rotate(piece_img, gt[pk]['rotation'])
            
            # relative shift is x,y (the method place_at takes care of inverting)
            canvas = place_at(piece_img, canvas, np.round(relative_shift).astype(int))
        print(f"placed piece {pk}")

    plt.subplot(131)
    plt.title("Original image")
    orig_img_path = os.path.join(puzzle_folder, fnames.ref_image, f"output_{args.puzzle}.png")
    ref_image = cv2.cvtColor(cv2.imread(orig_img_path), cv2.COLOR_BGR2RGB)
    plt.imshow(ref_image)
    plt.subplot(132)
    plt.title("Reconstructed")
    rgb_canvas = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_canvas)
    cropped_rec = crop_to_content(rgb_canvas)
    plt.subplot(133)
    plt.title("Reconstructed (cropped)")
    plt.imshow(cropped_rec)
    plt.show()

    print(gt)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Computing compatibility matrix')
    parser.add_argument('--dataset', type=str, default='', help='dataset (name of the folders)')
    parser.add_argument('--puzzle', type=str, default='', help='puzzle to work on - leave empty to generate for the whole dataset')
    args = parser.parse_args()
    main(args)