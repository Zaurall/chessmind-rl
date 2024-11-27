import chess
import torch
import torch.nn as nn

import mcts
import alpha_zero_network

class AlphaZero():
    def __init__(self, model_path="alpha_zero_net.pt", rollouts=10, threads=1):
        self.alphaZeroNet = alpha_zero_network.AlphaZeroNet(20, 256)
        cuda = False
        weights = torch.load(model_path) if cuda else torch.load(model_path, map_location=torch.device('cpu'))
        self.alphaZeroNet.load_state_dict(weights)
        if cuda:
            self.alphaZeroNet = self.alphaZeroNet.cuda()
        for param in self.alphaZeroNet.parameters():
            param.requires_grad = False
        self.alphaZeroNet.eval()

        self.num_rollouts = rollouts
        self.num_threads = threads

    def choose_action(self, fen):
        board = chess.Board(fen)
        with torch.no_grad():
            root = mcts.Root(board, self.alphaZeroNet)
            for i in range(self.num_rollouts):
                root.parallel_rollouts(board.copy(), self.alphaZeroNet, self.num_threads)
        edge = root.select_max_n()
        best_move = edge.get_move()
        return best_move

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = nn.functional.relu(self.conv3(x))
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x