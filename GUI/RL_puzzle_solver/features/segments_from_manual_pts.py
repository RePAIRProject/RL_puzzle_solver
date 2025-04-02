import pdb 
import json
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

def line_cart2pol(pt1, pt2):

    x_diff = pt1[0] - pt2[0]
    y_diff = pt1[1] - pt2[1]
    theta = np.arctan(-(x_diff/(y_diff + 10**-5)))
    rho = pt1[0] * np.cos(theta) + pt1[1] * np.sin(theta)
    rho2 = pt2[0] * np.cos(theta) + pt2[1] * np.sin(theta)
    #print("checkcart2pol", theta, rho, rho2)
    #pdb.set_trace()
    return rho, theta

for img_num in range(194, 204):
    pts_path = f"output_8x8/repair/g28/lines_detection/manual/RPf_{img_num:05d}_pts.json"
    img_path = f"output_8x8/repair/g28/pieces/RPf_{img_num:05d}.png"
    img = cv2.imread(img_path)[:, :, ::-1]
    with open(pts_path, 'r') as ptsj:
        pts = json.load(ptsj)

        angles_man = []
        dists_man = []
        p1s_man = []
        p2s_man = []

        for p1, p2 in zip(pts['p1s'], pts['p2s']):

            rhofld, thetafld = line_cart2pol(p1, p2)
            angles_man.append(thetafld)
            dists_man.append(rhofld)
            p1s_man.append(p1)
            p2s_man.append(p2)

        len_lines = len(angles_man)
        if len_lines > 0:
            plt.figure()
            plt.title(f'annotated {len_lines} segments manually')
            plt.imshow(img)
            for p1, p2 in zip(pts['p1s'], pts['p2s']):
                plt.plot((p1[0], p2[0]), (p1[1], p2[1]), color='red', linewidth=3)        
            plt.savefig(f"output_8x8/repair/g28/lines_detection/manual/visualization/{img_num:05d}.jpg")
            plt.close()
            # save one black&white image of the lines
            #pdb.set_trace()
            lines_img = np.zeros(shape=img.shape, dtype=np.uint8)
            for p1, p2 in zip(pts['p1s'], pts['p2s']):
                lines_img = cv2.line(lines_img, np.round(p1).astype(int), np.round(p2).astype(int), color=(255, 255, 255), thickness=1)        
            cv2.imwrite(f"output_8x8/repair/g28/lines_detection/manual/lines_only/{img_num:05d}_l.jpg", 255-lines_img)
        else:
            plt.title('no lines')
            plt.imshow(img)    
            plt.savefig(f"output_8x8/repair/g28/lines_detection/manual/visualization/{img_num:05d}.jpg")
            plt.close()
            lines_img = np.zeros(shape=img.shape, dtype=np.uint8)      
            cv2.imwrite(f"output_8x8/repair/g28/lines_detection/manual/lines_only/{img_num:05d}_l.jpg", 255-lines_img)
            #pdb.set_trace()



        detected_lines = {
            'angles': angles_man,
            'dists': dists_man,
            'p1s': p1s_man,
            'p2s': p2s_man,
            # 'b1s': [],
            # 'b2s': []
        }
        with open(f"output_8x8/repair/g28/lines_detection/manual/RPf_{img_num:05d}.json", 'w') as lj:
            json.dump(detected_lines, lj, indent=3)
        
        print(f'saved RPf_{img_num:05d} (annotated {len_lines} segments)')