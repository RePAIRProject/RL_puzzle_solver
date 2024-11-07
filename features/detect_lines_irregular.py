import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
import torch
import h5py
import json
import shapely
from sklearn.cluster import KMeans

# if error, export python path with deeplsd path!
from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD
from deeplsd.geometry.viz_2d import plot_images, plot_lines
import pdb
import json
import argparse 

def line_cart2pol(pt1, pt2):

    x_diff = pt1[0] - pt2[0]
    y_diff = pt1[1] - pt2[1]
    theta = np.arctan(-(x_diff/(y_diff + 10**-5)))
    rho = pt1[0] * np.cos(theta) + pt1[1] * np.sin(theta)
    rho2 = pt2[0] * np.cos(theta) + pt2[1] * np.sin(theta)
    #print("checkcart2pol", theta, rho, rho2)
    #pdb.set_trace()
    return rho, theta

def is_valid(segment, img_shape, border_tolerance):
    """
    Check segments validity  
    """
    # check for segments parallel to the borders    
    for axis in [0, 1]:
        # if it is too close to one border
        if segment[0][axis] < border_tolerance and segment[1][axis] < border_tolerance:
            return False 
        # or the other border
        if segment[0][axis] > (img_shape[1] - border_tolerance) and segment[1][axis] > (img_shape[1] - border_tolerance):
            return False 

    # return true if something is on the border
    if np.any(segment < border_tolerance) or np.any(segment > (img_shape[1] - border_tolerance)):
        return True
    
    return False


def main(args):

    #root_folder = args.root_folder
    
    # '/home/lucap/code/RL_puzzle_solver/data/manual_lines/pieces'
    dataset_folder_pieces = os.path.join(os.getcwd(), f'output', args.dataset) #'pieces')
    border_tolerance = 5
    filter_lines = False
    output_folder = dataset_folder_pieces #os.path.join(root_folder, f'output_{num_pieces}x{num_pieces}', args.dataset)
    # output_folder = '/home/lucap/code/RL_puzzle_solver/output/manual_lines'
    
    # Model config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf = {
        'detect_lines': True,  # Whether to detect lines or only DF/AF
        'line_detection_params': {
            'merge': True,  # Whether to merge close-by lines
            'filtering': True,  # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
            'grad_thresh': 3,
            'grad_nfa': True,  # If True, use the image gradient and the NFA score of LSD to further threshold lines. We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
        }
    }

    # Load the model
    ckpt = args.weights
    ckpt = torch.load(str(ckpt), map_location='cpu')
    net = DeepLSD(conf)
    net.load_state_dict(ckpt['model'])
    net = net.to(device).eval()



    for puzzle_folder in os.listdir(dataset_folder_pieces):
        print("detecting on", puzzle_folder)
        full_path_puzzle_folder = os.path.join(dataset_folder_pieces, puzzle_folder)
        if os.path.isdir(full_path_puzzle_folder):
            cur_puzzle_pieces_folder = os.path.join(dataset_folder_pieces, puzzle_folder, f'pieces')
            cur_puzzle_polygons_folder = os.path.join(dataset_folder_pieces, puzzle_folder, f'polygons')
            imgs_names = os.listdir(cur_puzzle_pieces_folder)

            with open(os.path.join(full_path_puzzle_folder, f'parameters_{puzzle_folder}.json'), 'r') as pjf:
                parameters = json.load(pjf)
            piece_size = parameters['piece_size']
            pieces_list = np.zeros((piece_size, piece_size, len(imgs_names)))
            rgb_pieces_list = np.zeros((piece_size, piece_size, 3, len(imgs_names)))
    
            # output
            cur_output_folder = os.path.join(output_folder, puzzle_folder)
            lines_output_folder = os.path.join(cur_output_folder, 'lines_detection', 'deeplsd')
            vis_output = os.path.join(lines_output_folder, 'visualization')
            os.makedirs(vis_output, exist_ok=True)
            lin_output = os.path.join(lines_output_folder, 'lines_only')
            os.makedirs(lin_output, exist_ok=True)

            # list of all colors
            full_colors_list = []
            full_detection_list = []

            pieces_names = []
            polygons = []
            for j, img_name in enumerate(imgs_names):
                full_path = os.path.join(cur_puzzle_pieces_folder, img_name)
                # Load an image
                orig_img = cv2.imread(full_path)[:, :, ::-1]
                gray_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
                pieces_list[:,:,j] = gray_img
                rgb_pieces_list[:,:,:,j] = orig_img / 255
                pieces_names.append(img_name)
                # polygon = np.load(os.path.join(cur_puzzle_polygons_folder, f"{img_name[:-4]}.npy"), allow_pickle=True)
                # polygons.append(polygon)


            for k in range(len(imgs_names)):

                angles_lsd = []
                dists_lsd = []
                p1s_lsd = []
                p2s_lsd = []
                b1s_lsd = []
                b2s_lsd = []
                colors_lsd = []

                img = pieces_list[:,:,k]
                # black image for drawing 
                black = np.zeros_like(img)
                # Detect (and optionally refine) the lines
                inputs = {
                    'image': torch.tensor(img, dtype=torch.float, device=device)[None, None] / 255.
                    }
                with torch.no_grad():
                    out = net(inputs)
                    pred_lines = out['lines'][0]

                if pred_lines is not None:
                    # convert to polar
                    for lsd_seg in pred_lines:
                        p1 = lsd_seg[0]
                        p2 = lsd_seg[1]
                        
                        black[:,:] = 0
                        line_shape = cv2.line(black, np.asarray([p1[0], p1[1]]).astype(int), np.asarray([p2[0], p2[1]]).astype(int), color=(1), thickness=1)
                        color_line = np.mean(orig_img[line_shape>0], axis=0)

                        if filter_lines is True:
                            line_shp = shapely.LineString((p1, p2))
                            polygon = polygons[k].tolist()
                            
                            #breakpoint()
                            #if not shapely.is_empty(shapely.intersection(line_shp, polygon.buffer(border_tolerance))):
                            # plt.plot(*(line_shp.xy), color='blue')
                            # plt.plot(*(polygon.boundary.xy), color='orange')
                            # titlef = f'cross: {shapely.crosses(line_shp, polygon.buffer(-2))}\nint: {not shapely.is_empty(shapely.intersection(line_shp, polygon.buffer(-2)))}'
                            # plt.title(titlef)
                            # plt.show()
                            # breakpoint()
                            if shapely.crosses(line_shp, polygon.buffer(-2)):
                                rhofld, thetafld = line_cart2pol(p1, p2)
                                angles_lsd.append(thetafld)
                                dists_lsd.append(rhofld)
                                p1s_lsd.append(p1.tolist())
                                p2s_lsd.append(p2.tolist())
                                if np.any(p1 < border_tolerance) or np.any(p1 > (img.shape[1] - border_tolerance)):
                                    b1s_lsd.append(0)
                                else:
                                    b1s_lsd.append(1)
                                if np.any(p2 < border_tolerance) or np.any(p2 > (img.shape[1] - border_tolerance)):
                                    b2s_lsd.append(0)
                                else:
                                    b2s_lsd.append(1)
                                full_colors_list.append(color_line)
                                colors_lsd.append(color_line.tolist())
                                
                                
                        else:
                            rhofld, thetafld = line_cart2pol(p1, p2)
                            angles_lsd.append(thetafld)
                            dists_lsd.append(rhofld)
                            p1s_lsd.append(p1.tolist())
                            p2s_lsd.append(p2.tolist())
                            full_colors_list.append(color_line)
                            colors_lsd.append(color_line.tolist())

                    len_lines = len(pred_lines)
                    plt.figure()
                    plt.title(f'(found {len_lines} segments with DeepLSD)')
                    plt.imshow(img)
                    for line_lsd in pred_lines:
                        plt.plot((line_lsd[0][0], line_lsd[1][0]), (line_lsd[0][1], line_lsd[1][1]), color='red', linewidth=3)        
                    plt.savefig(os.path.join(vis_output, f"{pieces_names[k][:-4]}_dlsd.jpg"))
                    plt.close()
                    plt.figure()
                    plt.title(f'keep {len(p1s_lsd)} segments for compatibility')
                    plt.imshow(img)
                    for p1, p2 in zip(p1s_lsd, p2s_lsd):
                        plt.plot((p1[0], p2[0]), (p1[1], p2[1]), color='red', linewidth=3)        
                    plt.savefig(os.path.join(vis_output, f"{pieces_names[k][:-4]}_filtered_dlsd.jpg"))
                    plt.close()

                    # save one black&white image of the lines
                    #pdb.set_trace()
                    lines_img = np.zeros(shape=img.shape, dtype=np.uint8)
                    for p1, p2, col_line in zip(p1s_lsd, p2s_lsd, colors_lsd):
                        lines_img = cv2.line(lines_img, np.round(p1).astype(int), np.round(p2).astype(int), color=col_line, thickness=1)        
                    cv2.imwrite(os.path.join(lin_output, f"{pieces_names[k][:-4]}_l.jpg"), 255-lines_img)
                else:
                    plt.title('(no lines with DeepLSD)')
                    plt.imshow(img)
                    plt.savefig(os.path.join(vis_output, f"{pieces_names[k][:-4]}_dlsd.jpg"))
                    plt.close()
                # pdb.set_trace()
                detected_lines = {
                    'angles': angles_lsd,
                    'dists': dists_lsd,
                    'p1s': p1s_lsd,
                    'p2s': p2s_lsd,
                    'b1s': b1s_lsd,
                    'b2s': b2s_lsd,
                    'colors': colors_lsd,
                }
                full_detection_list.append(detected_lines)
                
                
                print(f'saved {cur_output_folder}/{k} (found {len_lines} segments with DeepLSD, kept {len(p1s_lsd)} segments after filtering)')
                # Plot the predictions
                #plot_images([img], [f'DeepLSD {len(pred_lines)} lines'], cmaps='gray')
                #plot_lines([pred_lines], indices=range(1))
        print(f"finished image {imgs_names[k]}")
        
        if args.use_categories == True:
            fcl = np.asarray(full_colors_list)
            km = KMeans(args.km_clusters)
            labels = km.fit_predict(fcl)
            len_detected_lines = 0 
            for det_dict in full_detection_list:
                len_detected_lines += len(det_dict['angles'])
            print(f"We have {len(labels)} labels and {len_detected_lines} detections!")
            assert len(labels) == len_detected_lines, "they do not match!"
            
            
            line_counter = 0
            for k, det_dict in enumerate(full_detection_list):
                categories = []
                for j in range(len(det_dict['angles'])):
                    categories.append(int(labels[line_counter + j]))
                det_dict['categories'] = categories
                line_counter += j
                with open(os.path.join(lines_output_folder, f"{pieces_names[k][:-4]}.json"), 'w') as lj:
                    json.dump(det_dict, lj, indent=3)
        else:
            print("Not using categories, assigning always one")
            line_counter = 0
            for k, det_dict in enumerate(full_detection_list):
                detected_lines = len(det_dict['angles'])
                categories = np.ones(detected_lines).tolist()
                det_dict['categories'] = categories
                line_counter += detected_lines
                with open(os.path.join(lines_output_folder, f"{pieces_names[k][:-4]}.json"), 'w') as lj:
                    json.dump(det_dict, lj, indent=3)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare the data')
    parser.add_argument('-d', '--dataset', type=str, default='/media/lucap/big_data/datasets/repair/puzzle2D', help='data folder')
    parser.add_argument('-w', '--weights', type=str, default='../weights/deeplsd_wireframe.tar', help='pre-trained weights')
    parser.add_argument('-k', '--km_clusters', type=int, default=5, help='number of clusters (kmeans) for semantic category')
    parser.add_argument('-ucat', '--use_categories', type=int, default=1, help='whether or not to use categories (use 0 or 1)')
    args = parser.parse_args()
    main(args)
