import pandas as pd
import numpy as np

class Evaluation:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def evaluate(self, pieces, results, ground_truth):
        scores_df = pd.DataFrame(columns=['object_name', 'Q_pos', 'RMSE_rot', 'RMSE_translation'])

        for piece in pieces:
            q_pos = self.calculate_q_pos(piece, results[piece], ground_truth[piece])
            rmse_value = self.calculate_rmse(piece, results[piece], ground_truth[piece])

            new_row = pd.DataFrame([{'object_name': piece.name, 'Q_pos': q_pos, 'RMSE_rot': rmse_value['RMSE_rot'],
                                     'RMSE_translation': rmse_value['RMSE_translation']}])
            scores_df = pd.concat([scores_df, new_row], ignore_index=True)

        # fill in blank values with 0
        scores_df.fillna(0, inplace=True)

        avg_q_pos = scores_df['Q_pos'].mean()
        avg_rmse_rot = scores_df['RMSE_rot'].mean()
        avg_rmse_translation = scores_df['RMSE_translation'].mean()

        # Placeholder for evaluation logic
        pass

    def calculate_rmse(self, piece, results, ground_truth, pxls_to_m_scaler = (1/7.369)):
        # Load the CSV files into pandas DataFrames
        results_df = pd.read_csv(results)
        ground_truth_df = pd.read_csv(ground_truth)

        # Merge the DataFrames on the 'rpf' column to align the results with the ground truth
        merged_df = pd.merge(results_df, ground_truth_df, on='rpf', suffixes=('_result', '_gt'))

        # Get the transformation for the largest piece
        additional_transformation = get_transformation_for_largest_piece(piece, results, ground_truth)
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

    def calculate_q_pos(self, piece, results, ground_truth):

        return 1

    def display_results(self):
        # Placeholder for displaying results
        pass