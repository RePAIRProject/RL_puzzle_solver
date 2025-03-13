"""
It handles loading and writing to .YAML files
It should not have dependencies and should handle files with less parameters (take the others from `default.yaml`) 
"""
import os 
import yaml 

class Configuration:
    
    def __init__(self, puzzle_name: str):
        self.preprocessing_folder = 'preprocessing'
        self.preprocessing_params_name = 'preprocessing.yaml'
        self.images_subfolder = 'images'
        self.masks_subfolder = 'masks'
        self.polygons_subfolder = 'polygons'
        self.features_folder = 'features'
        self.features_params_name = 'features.yaml'
        self.puzzle_name = puzzle_name

    def get_puzzle_subfolders(self):
        """
        Returns a dictionary with all the subfolders paths
        """
        subfolders = {
            'preprocessing': os.path.join(self.preprocessing_folder, puzzle_name),
            'images': os.path.join(self.preprocessing_folder, puzzle_name, images_subfolder),
            'masks': os.path.join(self.preprocessing_folder, puzzle_name, masks_subfolder),
            'polygons': os.path.join(self.preprocessing_folder, puzzle_name, polygons_subfolder),
            'features': os.path.join(self.features_folder, puzzle_name)
        }
        return subfolders 
        
    def get_puzzle_images_subfolder(self):
        return os.path.join(self.preprocessing_folder, self.puzzle_name, self.images_subfolder)

    def get_puzzle_masks_subfolder(self):
        return os.path.join(self.preprocessing_folder, self.puzzle_name, self.masks_subfolder)

    def get_puzzle_polygons_subfolder(self):
        return os.path.join(self.preprocessing_folder, self.puzzle_name, self.polygons_subfolder)

    def get_puzzle_features_subfolder(self):
        return os.path.join(self.features_folder, self.puzzle_name)

    def get_features_extracted(self, from_yaml: bool = True):
        if from_yaml == True:
            features_extracted = self.read_features_from_yaml(os.path.join(features_folder, features_params_name))
        else:
            features_folder_files = os.listdir(os.path.join(features_folder, puzzle_name))
            features_extracted = [fsf for fsf in features_folder_files if os.isdir(os.path.join(features_folder, puzzle_name,fsf)) == True]
        return features_extracted

    def read_features_from_yaml(self, yaml_path: str):
        with open(yaml_path, 'r') as file:
            parameters = yaml.safe_load(file)
        print("TODO: get the features keys from the full file")
        breakpoint()

    def load(self, yaml_file_path: str):
        """
        Loads the parameters from the .yaml file 
        Some parameters are "consequences" of the loaded one (calculated from)
        """
        with open(yaml_file_path, 'r') as file:
            parameters = yaml.safe_load(file)
        return parameters 
        
    def save(self, yaml_file_path: str):
        """
        Save (the parameters) to the .yaml file 
        """
