import pdb 
import argparse
import os, json
import shutil 
import numpy as np 
import pandas as pd 

def line_cart2pol(pt1, pt2):

    x_diff = pt1[0] - pt2[0]
    y_diff = pt1[1] - pt2[1]
    theta = np.arctan(-(x_diff/(y_diff + 10**-5)))
    rho = pt1[0] * np.cos(theta) + pt1[1] * np.sin(theta)
    rho2 = pt2[0] * np.cos(theta) + pt2[1] * np.sin(theta)
    #print("checkcart2pol", theta, rho, rho2)
    #pdb.set_trace()
    return rho, theta

def perturbate(p1, p2, noise_type, noise_percentage, noise_sigma):

    if noise_type == 'positional':
        p1orp2 = np.random.choice(2)
        if p1orp2 == 0:
            p1 += np.random.normal(0, noise_sigma)
        else:
            p2 += np.random.normal(0, noise_sigma)

    return p1, p2

def get_fragments_list(lines_detection_folder):
    json_files = [os.path.join(lines_detection_folder, file) for file in os.listdir(lines_detection_folder) if os.path.isdir(file) is False and file.endswith('.json')]
    return json_files

def get_total_segments(dataset_folder):
    images_names = os.listdir(dataset_folder)
    images_names.sort()
    print(images_names)

    big_list_a = []
    big_list_d = []
    big_list_p1 = []
    big_list_p2 = []
    big_list_path = []
    big_list_id = []
    big_list_piece = []
    big_list_idx_within_piece = []

    j = 0

    #pdb.set_trace()
    for image_name in images_names:
        lines_detection_folder = os.path.join(dataset_folder, image_name, 'lines_detection', 'exact')
        fragments_list = get_fragments_list(lines_detection_folder)
        for fragment_path in fragments_list:
            with open(fragment_path, 'r') as fp:
                segments = json.load(fp)

            # reset for each piece
            idx_piece = 0
            angles = segments['angles']
            dists = segments['dists']
            p1s = segments['p1s']
            p2s = segments['p2s']

            for angle, dist, p1, p2 in zip(angles, dists, p1s, p2s):
                big_list_a.append(angle)
                big_list_d.append(dist)
                big_list_p1.append(p1)
                big_list_p2.append(p2)
                big_list_id.append(j) # this is the line id!
                j += 1
                big_list_idx_within_piece.append(idx_piece) # the id within the piece (ex: fourth line of piece 300)
                idx_piece += 1
                big_list_path.append(fragment_path)
                big_list_piece.append(fragment_path.split('/')[-1][:-5])


    # finished, choose some to perturbate!
    segments_df = pd.DataFrame()
    segments_df['angle'] = big_list_a
    segments_df['dist'] = big_list_d
    segments_df['p1'] = big_list_p1
    segments_df['p2'] = big_list_p2
    segments_df['id'] = big_list_id
    segments_df['piece'] = big_list_piece
    segments_df['id_within_piece'] = big_list_idx_within_piece
    segments_df['path_json_file'] = big_list_path

    print(segments_df)
    pdb.set_trace()

    return segments_df


def get_segments_of_puzzle(puzzle_folder):
    """
    In `puzzle_folder` there are the .json files of the detected (or extracted) lines.
    We iterate through them and create a dataframen with all the information needed for perturbation later.
    """
    fragments_list = get_fragments_list(puzzle_folder)

    big_list_a = []
    big_list_d = []
    big_list_p1 = []
    big_list_p2 = []
    big_list_path = []
    big_list_id = []
    big_list_piece = []
    big_list_idx_within_piece = []

    j = 0

    for fragment_path in fragments_list:
            with open(fragment_path, 'r') as fp:
                segments = json.load(fp)

            # reset for each piece
            idx_piece = 0
            angles = segments['angles']
            dists = segments['dists']
            p1s = segments['p1s']
            p2s = segments['p2s']

            for angle, dist, p1, p2 in zip(angles, dists, p1s, p2s):
                big_list_a.append(angle)
                big_list_d.append(dist)
                big_list_p1.append(p1)
                big_list_p2.append(p2)
                big_list_id.append(j) # this is the line id!
                j += 1
                big_list_idx_within_piece.append(idx_piece) # the id within the piece (ex: fourth line of piece 300)
                idx_piece += 1
                big_list_path.append(fragment_path)
                big_list_piece.append(fragment_path.split('/')[-1][:-5])


    # finished, choose some to perturbate!
    segments_df = pd.DataFrame()
    segments_df['angle'] = big_list_a
    segments_df['dist'] = big_list_d
    segments_df['p1'] = big_list_p1
    segments_df['p2'] = big_list_p2
    segments_df['id'] = big_list_id
    segments_df['piece'] = big_list_piece
    segments_df['id_within_piece'] = big_list_idx_within_piece
    segments_df['path_json_file'] = big_list_path

    # print(segments_df)
    # pdb.set_trace()

    return segments_df

def main(args):
    
    dataset_name = args.dataset
    if args.root_path == "":
        root_path = os.getcwd()
    else:
        root_path = args.root_path

    dataset_folder = os.path.join(root_path, 'output_8x8', dataset_name)  
    images_names = os.listdir(dataset_folder)  
    images_names = [img_n for img_n in images_names if os.path.isdir(os.path.join(dataset_folder, img_n)) is True]

    # perturbation!
    mu = args.mean
    sigma = args.std

    #pdb.set_trace()
    #big_list = get_total_segments(dataset_folder)

    #fragments_to_be_perturbated = np.random.choice(64, np.round(64 * args.percentage).astype(int))
    #segments_counter = 0
    print("#" * 50)
    print("Dataset:", args.dataset)
    print(f"Perturbating {args.percentage}% of the segments")


    for image_name in images_names:
        full_path_image_folder = os.path.join(dataset_folder, image_name)
        print("#" * 50)
        print("Working on", image_name)
        lines_detection_folder = os.path.join(full_path_image_folder, 'lines_detection')
        lines_to_perturb_folder = os.path.join(lines_detection_folder, args.method)

        puzzle_list_of_segments = get_segments_of_puzzle(lines_to_perturb_folder)
        
        perc_ratio = args.percentage / 100
        num_fragments_to_be_perturbated = np.round(len(puzzle_list_of_segments) * perc_ratio).astype(int)
        print(f"Perturbating {num_fragments_to_be_perturbated} over {len(puzzle_list_of_segments)} ({perc_ratio*100:.02f}%)")
        lines_to_be_perturbated = np.random.choice(len(puzzle_list_of_segments), num_fragments_to_be_perturbated)
        #print("#" * 50)
        print("Perturbating the following segments:", lines_to_be_perturbated)
        chosen_as_list = [0 if (x not in lines_to_be_perturbated) else 1 for x in np.arange(len(puzzle_list_of_segments))]
        puzzle_list_of_segments['noisy'] = chosen_as_list

        noise_folder_name = f"noise_{args.noise}_p{args.percentage}_s{sigma}"
        target_folder = os.path.join(lines_detection_folder, noise_folder_name)
        os.makedirs(target_folder, exist_ok=True)

        noisy_p1s = np.asarray(puzzle_list_of_segments['p1'])
        noisy_p2s = np.asarray(puzzle_list_of_segments['p2'])
        for pert_idx in lines_to_be_perturbated:
            # p1 = puzzle_list_of_segments['p1'][pert_idx]
            # p2 = puzzle_list_of_segments['p2'][pert_idx]
            chosen_p = np.random.randint(2)
            if chosen_p == 0:
                p1 = noisy_p1s[pert_idx]
                noisy_p1 = p1 + np.random.normal(mu, sigma, 2)
                noisy_p1 = np.clip(noisy_p1, 0, args.max)

                noisy_p1s[pert_idx] = noisy_p1.tolist()
            else:
                p2 = noisy_p2s[pert_idx]
                noisy_p2 = p2 + np.random.normal(mu, sigma, 2)
                noisy_p2 = np.clip(noisy_p2, 0, args.max)

                noisy_p2s[pert_idx] = noisy_p2.tolist()


            puzzle_list_of_segments['noisy_p1'] = noisy_p1s
            puzzle_list_of_segments['noisy_p2'] = noisy_p2s
        
        puzzle_list_of_segments.to_csv(os.path.join(target_folder, f"{image_name}.csv"))

        #print("#" * 50)
        print("Recreating the json files..")
        # now recreate the json files 
        pieces_id_names = np.unique(puzzle_list_of_segments['piece'])
        for piece_id_name in pieces_id_names:
            p1s = puzzle_list_of_segments[puzzle_list_of_segments['piece'] == piece_id_name]['noisy_p1']
            p2s = puzzle_list_of_segments[puzzle_list_of_segments['piece'] == piece_id_name]['noisy_p2']
            noisy_angles = []
            noisy_dists = []
            noisy_p1s = []
            noisy_p2s = []
            for p1, p2 in zip(p1s, p2s):

                rhofld, thetafld = line_cart2pol(p1, p2)
                noisy_angles.append(thetafld)
                noisy_dists.append(rhofld)
                noisy_p1s.append(p1)
                noisy_p2s.append(p2)

            detected_lines = {
                'angles': noisy_angles,
                'dists': noisy_dists,
                'p1s': noisy_p1s,
                'p2s': noisy_p2s,
                'b1s': [],
                'b2s': []
            }
            #print(noisy_angles, noisy_dists, noisy_p1s, noisy_p2s)
            with open(os.path.join(target_folder, f"RPf_{piece_id_name[-5:]}.json"), 'w') as lj:
                json.dump(detected_lines, lj, indent=3)
            #print("Saved", piece_id_name)

        # json_files = [file for file in os.listdir(lines_detection_folder) if os.path.isdir(file) is False and file.endswith('.json')]
        # print(f"got {len(files)} files")

        # for json_file in json_filesrandom_lines_exact_detection:

        #     ### np.arange(puzzle_list_of_segments)
        #     piece_id = int(json_file[6:10])
        #     ###
        #     if piece_id in fragments_to_be_perturbated:
        #         with open(os.path.join(lines_to_perturb_folder, json_file), 'r') as jfl:
        #             segments_on_fragment = json.load(jfl)

        #         # These are the points (initial and final) of each segment in this fragment
        #         pts = {}
        #         pts['p1s'] = segments_on_fragment['p1s']
        #         pts['p2s'] = segments_on_fragment['p2s']
                
        #         noisy_angles = []
        #         noisy_dists = []
        #         noisy_p1s = []
        #         noisy_p2s = []

        #         # for each point, we perturbate it and then re-create all the parameters we need (angle, rho, ..)
        #         # to save it in the same format as it if was detected with the whole pipeline (so we also have visualization)
        #         for p1, p2 in zip(pts['p1s'], pts['p2s']):

        #             # HERE WE ADD THE NOISE
        #             noisy_p1, noisy_p2, removed = perturbate(p1, p2, args.noise, args.percentage)

        #             rhofld, thetafld = line_cart2pol(p1, p2)
        #             noisy_angles.append(thetafld)
        #             dists_noisy.append(rhofld)
        #             noisy_p1s.append(p1)
        #             noisy_p2s.append(p2)
                
        #         len_lines = len(noisy_angles)
        #         if len_lines > 0:
        #             plt.figure()standard deviation for the gaussian noise
        #             plt.title(f'perturbated {len_lines} segments')
        #             plt.imshow(img)
        #             lines_img = np.zeros(shape=img.shape, dtype=np.uint8)
        #             for p1, p2 in zip(noisy_p1, noisy_p2):
        #                 plt.plot((p1[0], p2[0]), (p1[1], p2[1]), color='red', linewidth=3)    
        #                 lines_img = cv2.line(lines_img, np.round(p1).astype(int), np.round(p2).astype(int), color=(255, 255, 255), thickness=1)
                    
        #             cv2.imwrite(os.path.join(target_folder, f"{img_num:05d}_l.jpg", 255-lines_img))
        #             plt.savefig(os.path.join(target_folder, f"{img_num:05d}.jpg"))
        #             plt.close()                            
                    
        #         else:

        #             plt.title('no lines')
        #             plt.imshow(img) 
        #             plt.savefig(os.path.join(target_folder, f"{img_num:05d}.jpg"))   
        #             plt.close()
        #             lines_img = np.zeros(shape=img.shape, dtype=np.uint8)      
        #             cv2.imwrite(os.path.join(target_folder, f"{img_num:05d}_l.jpg", 255-lines_img))

                
        #         with open(os.path.join(target_folder, f"RPf_{img_num:05d}.json", 'w')) as lj:
        #             json.dump(detected_lines, lj, indent=3)
        #     else:
        #         ### CHECK
        #         shutil.copy(os.path.join(lines_to_perturb_folder, json_file), os.path.join(target_folder, f"RPf_{img_num:05d}.json"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Add noise to the extracted lines')  # add some description
    parser.add_argument('-r', '--root_path', type=str, default='', help='root folder (where the dataset is placed)')   
    parser.add_argument('-d', '--dataset', type=str, default='random_lines_exact_detection', help='dataset name (default: "random_lines_exact_detection")') 
    parser.add_argument('-m', '--method', type=str, default='exact', help='method (default: "exact")') 
    parser.add_argument('-n', '--noise', type=str, default='positional', help='noise (default: "positional")', choices=['positional', 'structural', 'combo'])
    parser.add_argument('-p', '--percentage', type=int, default=30, help='percentage of segments (over all pieces) to be perturbated  (default: 30)')
    parser.add_argument('--mean', type=int, default=0, help='mean for the gaussian noise  (default: 0)')
    parser.add_argument('--std', type=int, default=2, help='standard deviation for the gaussian noise (default: 2)')
    parser.add_argument('--max', type=int, default=151, help='max value to clip the noisy data within the piece size (default: 151)')

    args = parser.parse_args()

    main(args)