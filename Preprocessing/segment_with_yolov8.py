import argparse
from ultralytics import YOLO
import os 
import cv2 
import numpy as np

def main(args):

    model = YOLO(args.model)
    images_folder = args.input
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)

    for img in os.listdir(images_folder):
    
        full_path = os.path.join(images_folder, img)
        imgcv = cv2.imread(full_path)
        img_input = cv2.resize(imgcv, (512, 512))
        results = model(img_input)
    
        result = results[0]
        segmentation_mask = np.zeros((512, 512), dtype=np.float32)
        if result.masks is not None:
            for mm, mask in enumerate(result.masks):
                segmentation_mask += result.masks.data.cpu().numpy()[mm,:,:] 

        cv2.imwrite(os.path.join(output_folder, img), segmentation_mask)
        print('saved', img)

    return 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Segment with yolov8 (get masks without visualization)')
    parser.add_argument('--model', type=str, default='', help='model (.pt file)')
    parser.add_argument('--input', type=str, default='', help='input (folder with files)')
    parser.add_argument('--output', type=str, default='', help='output folder')
    args = parser.parse_args()
    main(args)