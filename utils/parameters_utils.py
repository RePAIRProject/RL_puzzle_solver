"""
It handles loading and writing to .YAML files
It should not have dependencies and should handle files with less parameters (take the others from `default.yaml`) 
"""
import os 
import yaml 

class PuzzleDaedalus:

    preprocessing_folder = 'preprocessing'
    preprocessing_params_name = 'preprocessing.yaml'
    images_subfolder = 'images'
    masks_subfolder = 'masks'
    polygons_subfolder = 'polygons'
    features_folder = 'features'
    features_params_name = 'features.yaml'

    def get_puzzle_images_subfolder(puzzle_name: str):
        return os.path.join(preprocessing_folder, puzzle_name, images_subfolder)

    def get_puzzle_masks_subfolder(puzzle_name: str):
        return os.path.join(preprocessing_folder, puzzle_name, masks_subfolder)

    def get_puzzle_polygons_subfolder(puzzle_name: str):
        return os.path.join(preprocessing_folder, puzzle_name, polygons_subfolder)

    def get_puzzle_features_subfolder(puzzle_name: str):
        return os.path.join(features_folder, puzzle_name)

    def get_features_extracted(puzzle_name: str, from_yaml: bool = True):
        if from_yaml == True:
            features_extracted = self.read_features_from_yaml(os.path.join(features_folder, features_params_name))
        else:
            features_folder_files = os.listdir(os.path.join(features_folder, puzzle_name))
            features_extracted = [fsf for fsf in features_folder_files if os.isdir(os.path.join(features_folder, puzzle_name,fsf)) == True]
        return features_extracted

    def read_features_from_yaml(yaml_path: str):
        with open(yaml_path, 'r') as file:
            parameters = yaml.safe_load(file)
        print("TODO: get the features keys from the full file")
        breakpoint()
