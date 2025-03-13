import numpy as np 
from parameters_utils import Configuration

class PuzzleFeatures:
    """
    This reads the extracted features (it does not perform the extraction!)
    """
    def __init__(puzzle_features_root_folder: str):
        self.root_folder = puzzle_features_root_folder
    
    def extract_lines(self, path):
        with open(path, 'r') as file:
            piece['extracted_lines'] = json.load(file)
        drawn_lines = draw_lines(piece['extracted_lines'], piece['img'].shape, line_thickness, use_color=False)
        return drawn_lines

    def extract_feature(self, piece, feature: str):
        feature_subfolder = os.path.join(self.root_folder, feature)
        if feature == 'lines':
            lines_json_path = os.path.join(feature_subfolder, f"{piece['name']}.json")
            piece['lines_mask'] = self.extract_lines(lines_json_path)
        if feature == 'motif':
            print("TODO")


def load_features(pieces: list, puzzle_name: str):
    """
    defined as a method to be called without the need to initialize extra objects 
    """
    features_extracted = Configuration.get_features_extracted(puzzle_name=puzzle_name)
    puzzle_feats = PuzzleFeatures(puzzle_features_root_folder = Configuration.get_puzzle_features_subfolder(puzzle_name=puzzle_name))
    for piece in pieces:
        for feature in features_extracted:
            piece[feature] = puzzle_feats.extract_feature(piece, feature)
    return pieces
