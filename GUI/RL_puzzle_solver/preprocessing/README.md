# Preprocessing

Here we focus on the creation and preprocessing of the data **before** computing compatibilities.

## Fragments Images
The images of the fragments will be resized to $251\times251$ pixels as defined in the compatibility parameters. The fragments will be centered so that the rotation is applied in the center of the squared image containing the fragment.

The `preprocess.py` script takes the original renderings and rescale and recenter them accordingly to the config file.
These images will be in the `data` folder and will be the **input** for the algorithm.

## Motif Segmentation
We have a second pre-processing to help out with the detection of the liens. We use a custom trained yolov8 segmentation model (more information coming soon) to segmente motifs. This can be done with the `segment_with_yolov8.py`.
An example run (for the *repair_g28* puzzle)
```bash
python segment_with_yolov8.py --model 'path_to_model.pt' --input '~yourpath/data/repair_g28/images' --output '~yourpath/output/MotifSegmentation/motif_repair_g28'
```
This will create the segmentation masks in the output folder (each label has value 0, 1, 2, so they are hardly visible without rescaling, do not worry if they look black).