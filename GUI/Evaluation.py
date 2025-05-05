import pandas as pd
import numpy as np
import os

from PIL import Image

class Evaluation:
    def __init__(self):
        self._img_cache: dict[str, Image.Image] = {}
        pass

    def _get_img(self, pid: str, path_lists: dict[str, str]) -> Image.Image:
        img = self._img_cache.get(pid)
        if img is None:
            img = Image.open(path_lists[pid]).convert("RGBA")
            self._img_cache[pid] = img
        return img

    def evaluate(self, pieces, results, ground_truth, path_lists):
        """
        pieces is a list of pieces to evaluate
        results and ground_truth are two dictionary with the following keys:
            - normalized piece id (just the number)
            - a list corresponds to the piece id which gives the position in form of [x, y, theta]
        path_lists is a dictionary with the normalized id and the path to the correspondign image (.png)
        """

        scores_df = pd.DataFrame(columns=['object_name', 'Q_pos', 'RMSE_rot', 'RMSE_translation'])

        # for piece in pieces:

        q_pos = 0
        q_pos = self.calculate_q_pos(pieces, results, ground_truth, path_lists)
        rmse_value = self.calculate_rmse(pieces, results, ground_truth, path_lists)

        new_row = pd.DataFrame([{'object_name': 3, 'Q_pos': q_pos, 'RMSE_rot': rmse_value['RMSE_rot'],
                                 'RMSE_translation': rmse_value['RMSE_translation']}])

        scores_df = pd.concat([scores_df, new_row], ignore_index=True)

        # fill in blank values with 0
        scores_df.fillna(0, inplace=True)

        avg_q_pos = scores_df['Q_pos'].mean()
        avg_rmse_rot = scores_df['RMSE_rot'].mean()
        avg_rmse_translation = scores_df['RMSE_translation'].mean()

        # Placeholder for evaluation logic
        return avg_q_pos, avg_rmse_rot, avg_rmse_translation

    def calculate_rmse(self, pieces, results, ground_truth, path_lists, pxls_to_m_scaler = (1/7.369)):
        # Load the CSV files into pandas DataFrames
        result_data = []
        gt_data = []

        for piece in pieces:
            res = results.get(piece)
            gt = ground_truth.get(piece)

            if res is None or gt is None:
                continue

            result_data.append([piece, res[0], res[1], res[2]])
            gt_data.append([piece, gt[0], gt[1], gt[2]])

        results_df = pd.DataFrame(result_data, columns=['rpf', 'x', 'y', 'rot'])
        ground_truth_df = pd.DataFrame(gt_data, columns=['rpf', 'x', 'y', 'rot'])

        # Merge the DataFrames on the 'rpf' column to align the results with the ground truth
        merged_df = pd.merge(results_df, ground_truth_df, on='rpf', suffixes=('_result', '_gt'))

        # Get the transformation for the largest piece
        additional_transformation = self.get_transformation_for_largest_piece(pieces, results_df, ground_truth_df, path_lists)
        # remove the "largest_piece" from merged_df
        merged_df = merged_df[merged_df['rpf'] != additional_transformation['largest_piece_name']]
        merged_df['x_result'] = merged_df['x_result'] + additional_transformation['x']
        merged_df['y_result'] = merged_df['y_result'] + additional_transformation['y']
        merged_df['rot_result'] = (merged_df['rot_result'] + additional_transformation['rot']) % 360

        rmse_translation = np.average(np.sqrt((merged_df['x_result'] - merged_df['x_gt']) ** 2 +
                                              (merged_df['y_result'] - merged_df[
                                                  'y_gt']) ** 2) * pxls_to_m_scaler) * 1 / np.sqrt(2)

        rmse_rot = 1 / np.sqrt(2) * np.average(
            np.sqrt((merged_df['rot_result'] % 360 - merged_df['rot_gt'] % 360) ** 2))

        rmse_values = {
            'RMSE_rot': rmse_rot % 360,
            'RMSE_translation': rmse_translation
        }

        return rmse_values

    def calculate_q_pos(self, pieces, results, ground_truth, path_lists, log=False, debug=False):
        """
           Calculates the score of the placement of the pieces on the shared canvas.

           ::param pieces_dir: the directory containing the pieces
           ::param transformations_dir: the csv file containing the result transformations
           ::param gt_transformations_dir: the csv file containing the ground truth transformations
           ::param log: whether to print the intermediate results or not
           """

        transformations_df= self.read_transformations(pieces, results)
        transformations_non_negative = self.read_transformations(pieces, results, make_non_negative=True)
        gt_transformations_df = self.read_transformations(pieces, ground_truth)

        # Initialize the shared canvas with the largest piece

        shared_canvas_width, shared_canvas_height = self.calculate_shared_canvas_size(pieces, transformations_df,
                                                                                 gt_transformations_df, path_lists)

        additional_transformation = self.get_transformation_for_largest_piece(pieces, transformations_df, gt_transformations_df, path_lists)

        additional_x = additional_x_for_gt = additional_y = additional_y_for_gt = 0
        if additional_transformation['x'] < 0:
            additional_x_for_gt = abs(additional_transformation['x'])
        else:
            additional_x = additional_transformation['x']
        if additional_transformation['y'] < 0:
            additional_y_for_gt = abs(additional_transformation['y'])
        else:
            additional_y = additional_transformation['y']

        additional_rot = additional_transformation['rot']

        pieces_weights = self.calculate_pieces_weights(pieces, path_lists, exclude_largest_piece=True,
                                                  largest_piece=additional_transformation['largest_piece_name'])

        q_pos = 0

        image_canvases = {}
        gt_image_canvases = {}

        # Apply the transformations on all the pieces then place them on the shared canvas
        for index, row in transformations_non_negative.iterrows():
            piece_filename = row['rpf']
            x = int(row['x'])
            y = int(row['y'])
            rot = row['rot']
            gt_x = int(gt_transformations_df[gt_transformations_df['rpf'] == piece_filename].iloc[0]['x'])
            gt_y = int(gt_transformations_df[gt_transformations_df['rpf'] == piece_filename].iloc[0]['y'])
            gt_rot = int(gt_transformations_df[gt_transformations_df['rpf'] == piece_filename].iloc[0]['rot'])

            piece_img = self._get_img(piece_filename, path_lists)

            new_piece = self.apply_transformations_on_piece(piece_img, x, y, rot, additional_x, additional_y)
            new_canvas = Image.new('RGBA', (shared_canvas_width, shared_canvas_height), (0, 0, 0, 0))
            new_canvas.alpha_composite(new_piece)
            image_canvases[piece_filename] = new_canvas

            gt_new_piece = self.apply_transformations_on_piece(piece_img, gt_x, gt_y, gt_rot, additional_x_for_gt,
                                                          additional_y_for_gt)
            new_gt_canvas = Image.new('RGBA', (shared_canvas_width, shared_canvas_height), (0, 0, 0, 0))
            new_gt_canvas.alpha_composite(gt_new_piece)
            gt_image_canvases[piece_filename] = new_gt_canvas

        rotated_image_canvases = {}
        largest_piece = image_canvases[f'{additional_transformation["largest_piece_name"]}']
        non_alpha_bbox = Image.fromarray(np.array(largest_piece)[:, :, 3]).getbbox()
        center_x = (non_alpha_bbox[2] + non_alpha_bbox[0]) / 2
        center_y = (non_alpha_bbox[3] + non_alpha_bbox[1]) / 2
        rotated_largest_piece = largest_piece.rotate(additional_rot, expand=True, center=(center_x, center_y))
        rotated_image_canvases[f'{additional_transformation["largest_piece_name"]}'] = rotated_largest_piece
        for piece_filename in image_canvases:
            if piece_filename == f'{additional_transformation["largest_piece_name"]}':
                continue
            else:
                piece = image_canvases[piece_filename]
                rotated_piece = piece.rotate(additional_rot, expand=True, center=(center_x, center_y))
                rotated_image_canvases[piece_filename] = rotated_piece

        # Calculate the Q_pos score
        for piece_filename in image_canvases:
            if piece_filename != f'{additional_transformation["largest_piece_name"]}':
                piece_weight = pieces_weights[piece_filename]
                result_area = self.calculate_area(rotated_image_canvases[piece_filename])
                shared_area = self.calculate_shared_area(rotated_image_canvases[piece_filename],
                                                    gt_image_canvases[piece_filename])
                partial_q_pos_score = piece_weight * (shared_area / result_area)

                if log:
                    print(f"Piece: {piece_filename}")
                    print(f"Piece weight: {piece_weight}")
                    print(f"Result area: {result_area}")
                    print(f"Shared area: {shared_area}")
                    print(f"Partial Q_pos score: {partial_q_pos_score}")

                q_pos += partial_q_pos_score

        if log:
            print(f"Q_pos score: {q_pos}")

        return q_pos if not debug else (q_pos, rotated_image_canvases, gt_image_canvases)

    def get_transformation_for_largest_piece(self, pieces, results_df, ground_truth_df, path_lists, largest_piece=None):
        if largest_piece is None:
            # Find the largest piece using the existing function
            largest_piece = self.find_largest_fragment(pieces, path_lists)

        # Get the ground truth transformation for the largest piece
        gt_largest_piece = ground_truth_df[ground_truth_df['rpf'] == largest_piece].iloc[0]
        results_largest_piece = results_df[results_df['rpf'] == largest_piece].iloc[0]

        # Calculate the transformation difference for the largest piece
        dx = gt_largest_piece['x'] - results_largest_piece['x']
        dy = gt_largest_piece['y'] - results_largest_piece['y']
        drot = gt_largest_piece['rot'] - results_largest_piece['rot']

        transformation = {
            'x': int(dx),
            'y': int(dy),
            'rot': (drot + 360) % 360,
            'res_x': results_largest_piece['x'],
            'res_y': results_largest_piece['y'],
            'res_rot': results_largest_piece['rot'],
            'largest_piece_name': largest_piece
        }

        return transformation

    def find_largest_fragment(self, pieces, path_lists):
        max_area, largest_piece = 0, None
        for pid in pieces:
            if not path_lists[pid].endswith(".png"):
                continue
            mask = np.array(self._get_img(pid, path_lists))[:, :, 3] > 0
            area = mask.sum()
            if area > max_area:
                max_area, largest_piece = area, pid
        return largest_piece

    def calculate_shared_canvas_size(self, pieces, transformations_df, gt_transformations_df, path_lists):
        """
        Calculate the dimensions of the shared canvas that will be used to place all the pieces, after applying the transformations on them.
        """
        # Find the largest piece
        largest_piece = self.find_largest_fragment(pieces, path_lists)
        largest_piece_path = path_lists[largest_piece]
        largest_piece_img =self._get_img(largest_piece, path_lists)
        largest_piece_array = np.array(largest_piece_img)

        # Read transformations
        transformations = transformations_df
        gt_transformations = gt_transformations_df

        # Initialize the shared canvas size with the dimensions of the largest piece
        shared_canvas_width = largest_piece_array.shape[1]
        shared_canvas_height = largest_piece_array.shape[0]

        # Apply the transformations on the largest piece to find the shared canvas size
        for index, row in transformations.iterrows():
            piece_filename = row['rpf']
            x = int(row['x'])
            y = int(row['y'])
            rot = row['rot']

            piece_img = self._get_img(piece_filename, path_lists)

            # Apply rotation directly on the PIL image
            rotated_piece = piece_img.rotate(rot, expand=True)

            # Calculate new canvas size
            new_width = max(rotated_piece.width, rotated_piece.width + abs(x))
            new_height = max(rotated_piece.height, rotated_piece.height + abs(y))

            # Create a new blank image with the new dimensions and the same mode as the rotated image
            new_piece = Image.new(rotated_piece.mode, (new_width, new_height))

            # Calculate position to paste the rotated image onto the new canvas
            paste_x = max(0, x)
            paste_y = max(0, y)

            # Paste the rotated image onto the new canvas
            new_piece.paste(rotated_piece, (paste_x, paste_y))

            # Update the shared canvas size
            shared_canvas_width = max(shared_canvas_width, new_piece.width)
            shared_canvas_height = max(shared_canvas_height, new_piece.height)

        for index, row in gt_transformations.iterrows():
            piece_filename = row['rpf']
            x = int(row['x'])
            y = int(row['y'])
            rot = row['rot']

            piece_img = self._get_img(piece_filename, path_lists)

            # Apply rotation directly on the PIL image
            rotated_piece = piece_img.rotate(rot, expand=True)

            # Calculate new canvas size
            new_width = max(rotated_piece.width, rotated_piece.width + abs(x))
            new_height = max(rotated_piece.height, rotated_piece.height + abs(y))

            # Create a new blank image with the new dimensions and the same mode as the rotated image
            new_piece = Image.new(rotated_piece.mode, (new_width, new_height))

            # Calculate position to paste the rotated image onto the new canvas
            paste_x = max(0, x)
            paste_y = max(0, y)

            # Paste the rotated image onto the new canvas
            new_piece.paste(rotated_piece, (paste_x, paste_y))

            # Update the shared canvas size
            shared_canvas_width = max(shared_canvas_width, new_piece.width)
            shared_canvas_height = max(shared_canvas_height, new_piece.height)

        return shared_canvas_width, shared_canvas_height

    def read_transformations(self, pieces, original_transformations, make_non_negative=False):
        """
        Read transformations from a csv file, and return a pandas dataframe of the form:
        | rpf (piece filename) | x (translation x) | y (translation y) | rot (rotation) |
        """
        result_data = []

        for piece in pieces:
            res = original_transformations.get(piece)

            if res is None:
                continue

            result_data.append([piece, res[0], res[1], res[2]])

        transformations = pd.DataFrame(result_data, columns=['rpf', 'x', 'y', 'rot'])

        if make_non_negative:
            # get minimum value of X and Y
            min_x = transformations['x'].min()
            min_y = transformations['y'].min()

            # subtract the minimum value from X and Y
            if min_x < 0:
                transformations['x'] = transformations['x'] + abs(min_x)

            if min_y < 0:
                transformations['y'] = transformations['y'] + abs(min_y)

        return transformations

    def apply_transformations_on_piece(self, piece_img, x, y, rot, additional_x=0, additional_y=0, additional_rot=0):
        # Apply rotation directly on the PIL image
        rotated_piece = piece_img.rotate(rot, expand=True)
        if additional_rot != 0:
            rotated_piece = rotated_piece.rotate(additional_rot, expand=True)

        # Calculate new canvas size
        new_width = max(rotated_piece.width, rotated_piece.width + abs(x) + abs(additional_x))
        new_height = max(rotated_piece.height, rotated_piece.height + abs(y) + abs(additional_y))

        # Create a new blank image with the new dimensions and the same mode as the rotated image
        new_piece = Image.new(rotated_piece.mode, (new_width, new_height))

        # Calculate position to paste the rotated image onto the new canvas
        paste_x = max(0, x)
        if additional_x != 0:
            paste_x = max(0, paste_x + additional_x)
        paste_y = max(0, y)
        if additional_y != 0:
            paste_y = max(0, paste_y + additional_y)

        # Paste the rotated image onto the new canvas
        new_piece.paste(rotated_piece, (paste_x, paste_y))

        return new_piece

    def calculate_pieces_weights(self, pieces, path_lists, exclude_largest_piece=False, largest_piece=None):
        pieces_weights = {}
        pieces_areas = {}
        for pid in pieces:
            if path_lists[pid].endswith(".png"):
                piece = self._get_img(pid, path_lists)
                area = self.calculate_area(piece)
                pieces_areas[pid] = area
        if exclude_largest_piece and largest_piece is not None:
            del pieces_areas[largest_piece]
        areas_sum = sum(pieces_areas.values())
        for filename in pieces_areas:
            pieces_weights[filename] = pieces_areas[filename] / areas_sum
        return pieces_weights

    def calculate_area(self, piece):
        piece_array = np.array(piece)
        alpha_channel = piece_array[:, :, 3]
        area = np.sum(alpha_channel > 0)
        return area

    def calculate_shared_area(self, piece1, piece2):
        piece1, piece2 = self.pad_and_fit_images(piece1, piece2)
        piece1 = np.array(piece1)[:, :, 3]
        piece2 = np.array(piece2)[:, :, 3]
        intersection = np.logical_and(piece1 > 0, piece2 > 0)
        shared_area = np.sum(intersection)
        return shared_area

    def pad_and_fit_images(self, image1, image2):
        width1, height1 = image1.size
        width2, height2 = image2.size
        new_width = max(width1, width2)
        new_height = max(height1, height2)
        new_image1 = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
        new_image2 = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
        new_image1.paste(image1, (0, 0))
        new_image2.paste(image2, (0, 0))
        return new_image1, new_image2