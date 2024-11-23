import chess
import random
import numpy as np
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import pickle


class RLAgent:
    def __init__(self, alpha=0.7, gamma=0.9, epsilon=0.5, max_q_values=100):
        self.alpha = alpha  
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.max_q_values = max_q_values  
        self.q_values = deque(maxlen=max_q_values)  
        self.total_reward = 0  
        self.num_games = 0  
        self.episode_rewards = []

        self.piece_values = dict(
            pawn=1,
            knight=3,
            bishop=3,
            rook=5,
            queen=9,
            king=100
        )

    def get_q_value(self, state, action):
        return next((q_value for q_state, q_action, q_value in self.q_values if q_state == state and q_action == action), 0.0)


    def update_q_value(self, state, action, next_state, reward):
        current_q = self.get_q_value(state, action)
        legal_actions = list(chess.Board(next_state).legal_moves)
        max_q = max((self.get_q_value(next_state, a) for a in legal_actions), default=0.0)
        new_q = current_q + self.alpha * (reward + self.gamma * max_q - current_q)
        self.q_values.append((state, action, new_q))


    def choose_action(self, state):
        return random.choice(list(chess.Board(state).legal_moves)) if random.random() < self.epsilon else self.get_best_action(chess.Board(state))


    
    def get_best_action(self, board):
        legal_actions = list(board.legal_moves)
        best_action = legal_actions[0]
        max_q = self.get_q_value(board.fen(), best_action)
        for action in legal_actions:
            q_value = self.get_q_value(board.fen(), action)
            if q_value > max_q:
                best_action = action
                max_q = q_value
        return best_action


    def get_reward(self, captured_piece):
        return 0 if captured_piece is None else self.piece_values.get(captured_piece.piece_type, 0)


    def play_game(self, opponent_agent):
        board = chess.Board()
        game_reward = 0  
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                state = board.fen()
                action = self.choose_action(state)
                board.push(action)
                reward = 0

                captured_piece = board.piece_at(action.to_square)
                if captured_piece:
                    reward = self.get_reward(captured_piece)

                next_state = board.fen()
                self.update_q_value(state, action, next_state, reward)
                game_reward += reward  
            else:
                state = board.fen()
                action = opponent_agent.choose_action(state)
                board.push(action)
                reward = 0

                captured_piece = board.piece_at(action.to_square)
                if captured_piece:
                    reward = opponent_agent.get_reward(captured_piece)

                next_state = board.fen()
                opponent_agent.update_q_value(state, action, next_state, reward)
                game_reward += reward  

        result = board.result()
        if result == "1-0": game_reward += 100  
        elif result == "0-1": game_reward -= 100  
        else: game_reward += 10  

        return result, game_reward

    def load_agent(agent, filename):
        with open(filename, "rb") as f:
            agent.q_values = pickle.load(f)
        print(f"Agent's Q-values have been loaded from {filename}")