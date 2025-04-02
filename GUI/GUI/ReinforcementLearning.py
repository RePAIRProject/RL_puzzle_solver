import numpy as np
import random

class ReinforcementLearning:
    def __init__(self, probability_matrix, delta_probs):
        self.probability_matrix = probability_matrix
        self.delta_probs = delta_probs
        self.q_table = {}  # State-action table for Q-learning
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1

    def set_probability_matrix(self, probability_matrix, delta_probs):
        self.delta_probs = delta_probs
        self.probability_matrix = probability_matrix

    def get_state(self, piece_number):
        """
        Simplify the state as a hashable representation of the flattened probability matrix for the piece.
        """
        flattened_probs = self.probability_matrix[:, :, :, piece_number].flatten()
        rounded_probs = np.round(flattened_probs, decimals=3)  # Round to avoid floating-point issues
        return tuple(rounded_probs)  # Convert to tuple for use as a dictionary key

    def select_action(self, state, available_actions):
        """
        Epsilon-greedy policy for action selection.
        """
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)

        # Ensure the state is in the Q-table
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in available_actions}

        # Select the action with the maximum Q-value
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_table(self, state, action, reward, next_state):
        # Q-learning update rule
        current_q = self.q_table.get(state, {}).get(action, 0)
        max_future_q = max(self.q_table.get(next_state, {}).values(), default=0)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q

    def update_probability_matrix(self, x, y, r, piece_number):
        # Define actions as adjustments to the matrix
        actions = [
            ("increase", x, y, r),
            ("decrease_others", piece_number),
        ]
        state = self.get_state(piece_number)
        action = self.select_action(state, actions)

        # Apply the selected action
        if action[0] == "increase":
            self.probability_matrix[x, y, r, piece_number] += 0.1  # Example increment
        elif action[0] == "decrease_others":
            self.probability_matrix[:, :, :, piece_number] *= 0.9  # Reduce others slightly

        # Normalize the probabilities
        self.probability_matrix[:, :, :, piece_number] /= self.probability_matrix[:, :, :, piece_number].sum()

        # Reward based on the result of this update
        reward = self.evaluate_reward(x, y, r, piece_number)
        next_state = self.get_state(piece_number)
        self.update_q_table(state, action, reward, next_state)
        return self.probability_matrix

    def evaluate_reward(self, x, y, r, piece_number):
        """
            Evaluate reward based on the agreement between the derivative of the probability
            matrix and human input.

            Args:
                x, y, r: Human-suggested position in the probability matrix.
                piece_number: The piece being updated.

            Returns:
                Reward value based on agreement.
            """
        # Current probabilities
        current_probs = self.probability_matrix[:, :, :, piece_number]

        # Derivative of the probability matrix (difference from previous iteration)
        # delta_probs = current_probs - self.previous_probs[piece_number]

        # Derivative at the human-suggested position
        derivative_at_human_pos = self.delta_probs[x, y, r, piece_number]

        # Human expects the probability at (x, y, r) to increase
        if derivative_at_human_pos > 0:
            reward = 1  # Agreement with the human input
        else:
            reward = -1  # Disagreement with the human input

        # Update previous probabilities for the next iteration
        # self.previous_probs[piece_number] = current_probs.copy()

        return reward