import os, json 
import argparse 
import numpy as np 
import matplotlib.pyplot as plt 
import pdb 
from configs import folder_names as fnames
import cv2
def main(args):

    dataset_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset)
    print(dataset_folder)
    if args.puzzle == '':  
        puzzles = os.listdir(dataset_folder)
        puzzles = [puz for puz in puzzles if os.path.isdir(os.path.join(dataset_folder, puz)) is True]
    else:
        puzzles = [args.puzzle]

    print(f"\nEvaluate solution for: {puzzles}\n")
    for puzzle in puzzles:
        
        print("\n\n")
        print("#" * 50)
        print(f"Now on {puzzle}")
        # check what we have
        puzzle_folder = os.path.join(dataset_folder, puzzle)
        with open(os.path.join(puzzle_folder, "ground_truth.json"), 'r') as gtj:
            ground_truth = json.load(gtj)
        with open(os.path.join(puzzle_folder, f"parameters_{puzzle}.json"), 'r') as gtj:
            img_parameters = json.load(gtj)
        with open(os.path.join(puzzle_folder, f"compatibility_parameters.json"), 'r') as gtj:
            cmp_parameters = json.load(gtj)
        regions_folder = os.path.join(puzzle_folder, 'regions')
        regions_mat = cv2.imread(os.path.join(regions_folder, 'regions_uint8.png'))

        num_pcs = img_parameters['num_pieces']
        neighbours_nums = {}
        neighbours_names = {}
        neighbours_matrix = np.zeros((num_pcs, num_pcs))

        for j in range(1, num_pcs+1):
            region_j = regions_mat == j 
            dilated_rj = cv2.dilate(region_j.astype(np.uint8), np.ones((5,5)))
            neig_j = []
            for k in range(1, num_pcs+1):
                if k != j:
                    if np.max(dilated_rj + (regions_mat==k).astype(np.uint8)) > 1:
                        neig_j.append(k-1)
                        neighbours_matrix[j-1, k-1] = 1

            neighbours_nums[j] = neig_j
            neighbours_names[f"piece_{(j-1):04d}"] = [f"piece_{p:04d}" for p in neig_j]
        
        print("Neighbours as names")
        print(neighbours_names)
        print("Neighbours as numbers")
        print(neighbours_nums)
        print("Neighbours as matrix")
        print(neighbours_matrix)
        neigh_dict = {
            'names': neighbours_names,
            'numbers': neighbours_nums
        }
        with open(os.path.join(puzzle_folder, 'neighbours.json'), 'w') as ej:
            json.dump(neigh_dict, ej, indent=3)
        np.savetxt(os.path.join(puzzle_folder, 'neighbours.txt'), neighbours_matrix, fmt='%.0f')
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Writes a file with the neighbors (reading from regions folder)')  # add some discription
    parser.add_argument('-d', '--dataset', type=str, default='manual_lines', help='dataset folder')   
    parser.add_argument('--verbosity', type=int, default=1, help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')                 # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-p', '--puzzle', type=str, default='', help='puzzle folder')    
    

    # parser.add_argument('-n', '--num_pieces', type=int, default=8, help='number of pieces (per side)')                  # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    # parser.add_argument('-a', '--anchor', type=int, default=-1, help='anchor piece (index)')                            # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    # parser.add_argument('-v', '--visualize', default=False, action='store_true', help='use to show the solution')       # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    # parser.add_argument('-aa', '--all_anchors', default=False, action='store_true', help='use to evaluate all anchors of this puzzle')       # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    
    args = parser.parse_args()

    main(args)