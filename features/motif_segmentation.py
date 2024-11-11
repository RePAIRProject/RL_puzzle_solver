import argparse
from ultralytics import YOLO
import os 
import cv2 
import numpy as np
import pdb 
from configs import repair_cfg as cfg
from configs import folder_names as fnames

def main(args):

    model = YOLO(args.model)
    images_folder = os.path.join(fnames.data_path, args.puzzle, fnames.imgs_folder)
    output_folder = os.path.join(os.getcwd(), fnames.output_dir, args.puzzle)
    segmentation_folder = os.path.join(output_folder, fnames.segm_output_name)
    lines_segmentation_folder = os.path.join(segmentation_folder, fnames.lines_segm_name)
    motifs_segmentation_folder = os.path.join(segmentation_folder, fnames.motifs_segm_name)
    os.makedirs(lines_segmentation_folder, exist_ok=True)
    os.makedirs(motifs_segmentation_folder, exist_ok=True)

    for img in os.listdir(images_folder):
    
        full_path = os.path.join(images_folder, img)
        imgcv = cv2.imread(full_path)

        img_input = cv2.resize(imgcv, (args.size, args.size))
        results = model(img_input)
    
        result = results[0]
        segmentation_mask = np.zeros((512, 512), dtype=np.float32)
        lines_mask = np.zeros((512, 512), dtype=np.float32)
        if result.masks is not None:
            masks = result.masks.data 
            boxes = result.boxes.data
            
            assert(len(masks) == len(boxes)), 'problem'
            masks_np = masks.cpu().numpy()
            boxes_np = boxes.cpu().numpy()
            for mm in range(len(masks)):
                cur_mask = masks_np[mm, :, :]
                cur_class = boxes_np[mm, 5]
                segmentation_mask += cur_mask * cur_class
                if cur_class == 10:
                    lines_mask += cur_mask

        if args.size != 512:
            segmentation_mask = cv2.resize(segmentation_mask, (args.size, args.size))
            lines_mask = cv2.resize(lines_mask, (args.size, args.size))
        cv2.imwrite(os.path.join(motifs_segmentation_folder, img), segmentation_mask)
        cv2.imwrite(os.path.join(lines_segmentation_folder, f"lines_{img}"), (lines_mask > 0).astype(np.uint8))
        print('saved', img)

    print("Finished! Output in", segmentation_folder)
    return 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Segment with yolov8 (get masks without visualization)')
    parser.add_argument('--model', type=str, default='', help='model (.pt file)')
    parser.add_argument('--puzzle', type=str, default='', help='puzzle name (inside the data folder)')
    parser.add_argument('--size', type=int, default=251, help='resizing of the images')

    args = parser.parse_args()
    main(args)