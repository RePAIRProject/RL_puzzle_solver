import ezdxf
import pdb 
import cv2
import numpy as np
import matplotlib.pyplot as plt 

def extract_image_from_dxf(dxf_file_path):
    """ 
    It extract the LWPOLYLINE from the dxf files and returns a rendered image along with an array of the segments.
    """ 
    img = np.ones((256, 256, 3))
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.arange(10)/10)
    doc = ezdxf.readfile(dxf_file_path)
    modelspace = doc.modelspace()
    dxf_segments = []
    lwpolylines = modelspace.query('LWPOLYLINE')
    categories = {}
    cat_val = 0
    for polyline in lwpolylines:        
        layer = polyline.dxf.layer
        if layer in categories.keys():
            cat = categories[layer]
        else:
            # new category
            cat_val += 1
            cat = cat_val 
            categories[layer] = cat_val

        vertices = np.asarray(polyline.get_points())
        p1s = vertices[:-1,:2]
        p2s = vertices[1:,:2]
        pixel_vertices = [(int(x), int(y)) for x, y, _, _, _ in vertices]
        col = colors[cat_val]
        img = cv2.polylines(img, [np.array(pixel_vertices)], isClosed=False, color=(col), thickness=1)    
        for p1, p2 in zip(p1s, p2s):
            dxf_segments.append([p1[0], p1[1], p2[0], p2[1], col[0], col[1], col[2], cat_val])
    
    img = (255 * img).astype(np.uint8)
    return img, np.asarray(dxf_segments), categories