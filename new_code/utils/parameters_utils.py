"""
It handles loading and writing to .YAML files
It should not have dependencies and should handle files with less parameters (take the others from `default.yaml`) 
"""
import os 
import yaml 
import random, string
from io import TextIOWrapper



##########################################
#                                        #
#  ██╗   ██╗ █████╗ ███╗   ███╗██╗       #
#  ╚██╗ ██╔╝██╔══██╗████╗ ████║██║       #
#   ╚████╔╝ ███████║██╔████╔██║██║       #
#    ╚██╔╝  ██╔══██║██║╚██╔╝██║██║       #
#     ██║   ██║  ██║██║ ╚═╝ ██║███████╗  #
#     ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝  #
#                                        #
##########################################
class CustomYAMLEncoder(yaml.SafeDumper):
    def default(self, obj):
        if isinstance(obj, TextIOWrapper):
            return f"File object: {obj.name}"
        return yaml.SafeDumper.default(self, obj)




#############################################################
#                                                           #
#   ██████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗ ███████╗  #
#  ██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝ ██╔════╝  #
#  ██║     ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗███████╗  #
#  ██║     ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║╚════██║  #
#  ╚██████╗╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝███████║  #
#   ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝  #
#                                                           #
#############################################################
class Configuration:
    
    def __init__(self):
        self.data_folder = 'data'
        self.preprocessing_folder = 'preprocessing'
        self.preprocessing_params_name = 'preprocessing.yaml'
        self.images_subfolder = 'images'
        self.masks_subfolder = 'binary_masks'
        self.polygons_subfolder = 'polygons'
        self.features_folder = 'features'
        self.features_params_name = 'features.yaml'
        self.experiments_folder = 'experiments'
        self.ground_truth_filename = 'ground_truth.json'
        self.puzzle_info_filename = 'puzzle_info.json'
        self.RM_name = 'RM.npy'
        self.RM_input_parameters_path = 'RM_input_params.yaml'
        self.RM_output_parameters_path = 'RM_output_params.yaml'
        self.CM_name = 'CM.npy'
        self.CM_input_parameters_path = 'CM_input_params.yaml'
        self.CM_output_parameters_path = 'CM_output_params.yaml'
        self.solution_name = 'solution.npy'
        self.solution_csv = 'solution.txt'
        self.solution_input_parameters_path = 'solution_input_params.yaml'
        self.solution_output_parameters_path = 'solution_output_params.yaml'

    def get_puzzle_name(self):
        return self.puzzle_name

    def set_puzzle_name(self, puzzle_name: str):
        self.puzzle_name = puzzle_name

    def get_puzzle_subfolders(self):
        """
        Returns a dictionary with all the subfolders paths
        """
        subfolders = {
            'data': os.path.join(self.data_folder, puzzle_name),
            'preprocessing': os.path.join(self.preprocessing_folder, puzzle_name),
            'images': os.path.join(self.preprocessing_folder, puzzle_name, images_subfolder),
            'masks': os.path.join(self.preprocessing_folder, puzzle_name, masks_subfolder),
            'polygons': os.path.join(self.preprocessing_folder, puzzle_name, polygons_subfolder),
            'features': os.path.join(self.features_folder, puzzle_name)
        }
        return subfolders
     
    def get_puzzle_images_subfolder(self):
        return os.path.join(self.data_folder, self.preprocessing_folder, self.puzzle_name, self.images_subfolder)

    def get_puzzle_masks_subfolder(self):
        return os.path.join(self.data_folder, self.preprocessing_folder, self.puzzle_name, self.masks_subfolder)

    def get_puzzle_polygons_subfolder(self):
        return os.path.join(self.data_folder, self.preprocessing_folder, self.puzzle_name, self.polygons_subfolder)

    def get_puzzle_features_subfolder(self):
        return os.path.join(self.data_folder, self.features_folder, self.puzzle_name)
    
    def get_puzzle_experiments_subfolder(self):
        return os.path.join(self.data_folder, self.experiments_folder, self.puzzle_name)
       
    def get_GT_path(self):
    	return os.path.join(self.data_folder, self.preprocessing_folder, self.puzzle_name, self.ground_truth_filename)
    	
    def get_puzzle_info_path(self):
    	return os.path.join(self.data_folder, self.preprocessing_folder, self.puzzle_name, self.puzzle_info_filename)
    	
    def new_puzzle_single_run_random_folder_name(self):
        self.current_experiment_folder = os.path.join(self.get_puzzle_experiments_subfolder(), f"exp_{self.randomword(6)}")
        os.makedirs(self.current_experiment_folder, exist_ok=True)

    def set_puzzle_single_run_random_folder_name(self, path: str):
        """ set the folder name when using CM or solver on a previous experiment """
        if path.find('#') < 0: # relative path
            self.current_experiment_folder = os.path.join(self.get_puzzle_experiments_subfolder(), path)
        else:                   # full path
            self.current_experiment_folder = path
            
    def get_RM_path(self):
        return os.path.join(self.current_experiment_folder, self.RM_name)

    def get_RM_input_parameters_path(self):
        return os.path.join(self.current_experiment_folder, self.RM_input_parameters_path)
    
    def get_RM_output_parameters_path(self):
        return os.path.join(self.current_experiment_folder, self.RM_output_parameters_path)
    
    def get_CM_path(self):
        return os.path.join(self.current_experiment_folder, self.CM_name)

    def get_CM_input_parameters_path(self):
        return os.path.join(self.current_experiment_folder, self.CM_input_parameters_path)
    
    def get_CM_output_parameters_path(self):
        return os.path.join(self.current_experiment_folder, self.CM_output_parameters_path)
    
    def get_aggregation_input_parameters_path(self):
        return os.path.join(self.current_experiment_folder, self.CM_input_parameters_path)
    
    def get_aggregation_output_parameters_path(self):
        return os.path.join(self.current_experiment_folder, self.CM_output_parameters_path)

    def get_solution_path(self):
        return os.path.join(self.current_experiment_folder, self.solution_name)

    def get_solution_as_csv_path(self):
        return os.path.join(self.current_experiment_folder, self.solution_csv)

    def get_solution_input_parameters_path(self):
        return os.path.join(self.current_experiment_folder, self.solution_input_parameters_path)
    
    def get_solution_output_parameters_path(self):
        return os.path.join(self.current_experiment_folder, self.solution_output_parameters_path)

    # def get_features_extracted(self, from_yaml: bool = True):
    #     if from_yaml == True:
    #         features_extracted = self.read_features_from_yaml(os.path.join(features_folder, features_params_name))
    #     else:
    #         features_folder_files = os.listdir(os.path.join(features_folder, puzzle_name))
    #         features_extracted = [fsf for fsf in features_folder_files if os.isdir(os.path.join(features_folder, puzzle_name,fsf)) == True]
    #     return features_extracted

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
        self.puzzle_name = parameters['puzzle_name']
        return parameters 
        
    def save(self, yaml_file_path: str):
        """
        Save (the parameters) to the .yaml file 
        """

    def randomword(self, length: int):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))
