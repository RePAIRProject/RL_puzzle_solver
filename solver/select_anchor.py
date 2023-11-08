
import os
import json
import argparse
import numpy as np
import configs.folder_names as fnames
from compatibility.line_matching_NEW_segments import read_info


def select_anchor(folder):
    pieces_files = os.listdir(folder)
    json_files = [piece_file for piece_file in pieces_files if piece_file[-4:] == 'json']
    json_files.sort()
    n = len(json_files)
    num_lines = np.zeros(n)
    for f in range(n):
        im = json_files[f]
        beta, R, s1, s2, b1, b2 = read_info(folder, im)
        num_lines[f] = len(beta)

    mean_num_lines = np.round(np.mean(num_lines))
    anc = np.random.choice(n, 1)

    if num_lines[anc] < mean_num_lines:
        good_anchors = np.array(np.where(num_lines > mean_num_lines))
        new_anc = np.random.choice(good_anchors[0, :], 1).item()
    else:
        new_anc = anc.item()
    return new_anc

def main(args):
    dataset_name = args.dataset
    puzzle_name = args.puzzle
    method = args.method
    num_pieces = args.pieces
    image_folder = os.path.join(f"{fnames.output_dir}_{num_pieces}x{num_pieces}", dataset_name, puzzle_name)
    detect_output = os.path.join(image_folder, f"{fnames.lines_output_name}", method)

    anc = select_anchor(detect_output)
    print(anc)

    selected_anchor = {'anchor': anc}

    with open(os.path.join(image_folder, f"selected_anchor.json"), 'w') as lj:
        json.dump(selected_anchor, lj, indent=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='........ ')
    parser.add_argument('--dataset', type=str, default='random_lines_exact_detection', help='dataset folder')
    parser.add_argument('--puzzle', type=str, default='image_0', help='puzzle folder')
    parser.add_argument('--method', type=str, default='exact', help='method used for compatibility')  # deeplsd exact
    parser.add_argument('--pieces', type=int, default=8, help='number of pieces (per side)')
    args = parser.parse_args()
    main(args)
