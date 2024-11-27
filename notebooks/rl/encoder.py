import chess
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def game_result(result):
    match result:
        case "1-0":
            return 1
        case "1/2-1/2":
            return 0
        case "0-1":
            return -1


def transform_board(chessboard):
    """
    Converts a chessboard into a 16x8x8 tensor representation
    """
    tensor = np.zeros((16, 8, 8), dtype=np.float32)

    # Map piece types to layer indices
    piece_categories = [
        (chess.PAWN, 0), (chess.ROOK, 2), (chess.BISHOP, 4),
        (chess.KNIGHT, 6), (chess.QUEEN, 8), (chess.KING, 10)
    ]

    for piece_type, layer_idx in piece_categories:
        # Encode positions for white pieces
        white_pieces = chessboard.pieces(piece_type, chess.WHITE)
        white_pieces = [(chess.square_rank(sq), chess.square_file(sq)) for sq in white_pieces]
        for row, col in white_pieces:
            tensor[layer_idx, row, col] = 1.0

        # Encode positions for black pieces
        black_pieces = chessboard.pieces(piece_type, chess.BLACK)
        black_pieces = [(chess.square_rank(sq), chess.square_file(sq)) for sq in black_pieces]
        for row, col in black_pieces:
            tensor[layer_idx + 1, row, col] = 1.0

    # Encode castling rights
    if chessboard.has_kingside_castling_rights(chess.WHITE):
        tensor[12, :, :] = 1.0
    if chessboard.has_kingside_castling_rights(chess.BLACK):
        tensor[13, :, :] = 1.0
    if chessboard.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if chessboard.has_queenside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0

    return tensor


def move_to_id(move):
    """
    Converts a move into an index (plane, row, col) for policy representation
    """
    start_row, start_col = chess.square_rank(move.from_square), chess.square_file(move.from_square)
    end_row, end_col = chess.square_rank(move.to_square), chess.square_file(move.to_square)

    plane_index = None

    # Handle straight, diagonal and king moves
    directions = [
        (lambda: start_row == end_row and start_col < end_col, 0, lambda: end_col - start_col),
        (lambda: start_row == end_row and start_col > end_col, 8, lambda: start_col - end_col),
        (lambda: start_col == end_col and start_row < end_row, 16, lambda: end_row - start_row),
        (lambda: start_col == end_col and start_row > end_row, 24, lambda: start_row - end_row),
        (lambda: end_col - start_col == end_row - start_row and end_col - start_col > 0, 32,
         lambda: end_row - start_row),
        (lambda: end_col - start_col == end_row - start_row and end_col - start_col < 0, 40,
         lambda: start_row - end_row),
        (lambda: end_col - start_col == -(end_row - start_row) and end_col - start_col > 0, 48,
         lambda: end_col - start_col),
        (lambda: end_col - start_col == -(end_row - start_row) and end_col - start_col < 0, 56,
         lambda: start_col - end_col),
    ]

    for condition, base_plane, distance_fn in directions:
        if condition():
            movement_plane = base_plane
            distance = distance_fn()
            plane_index = movement_plane + distance
            break

    knight_moves = {
        (1, 2): 64, (2, 1): 65, (2, -1): 66, (1, -2): 67,
        (-1, 2): 68, (-2, 1): 69, (-2, -1): 70, (-1, -2): 71
    }

    if plane_index is None:
        delta_col = end_col - start_col
        delta_row = end_row - start_row
        if (delta_col, delta_row) in knight_moves:
            plane_index = knight_moves[(delta_col, delta_row)]

    return plane_index, start_row, start_col


def get_legal_move_mask(board):
    """
    Create a mask for all legal moves on the board
    """
    legal_mask = np.zeros((72, 8, 8), dtype=np.int32)
    for legal_move in board.legal_moves:
        plane_id, rank_id, file_id = move_to_id(legal_move)
        legal_mask[plane_id, rank_id, file_id] = 1
    return legal_mask


def mirror_move(move):
    start_square, end_square = move.from_square, move.to_square
    return chess.Move(chess.square_mirror(start_square), chess.square_mirror(end_square))


def encode_training_point(chessboard, move, winner):
    if not chessboard.turn:  # Mirror if it's black's turn
        chessboard = chessboard.mirror()
        move = mirror_move(move)
        winner = -winner
    board_planes = transform_board(chessboard)
    mask = get_legal_move_mask(chessboard)
    move_indices = move_to_id(move)
    move_index = move_indices[0] * 64 + move_indices[1] * 8 + move_indices[2]
    return board_planes, move_index, float(winner), mask


def encode_position_for_inference(chessboard):
    if not chessboard.turn:  # Mirror if it's black's turn
        chessboard = chessboard.mirror()
    board_planes = transform_board(chessboard)
    legal_moves_mask = get_legal_move_mask(chessboard)
    return board_planes, legal_moves_mask


def decode_policy_output(chessboard, policy_vector):
    """
    Decodes neural network policy output to move probabilities
    """
    move_probabilities, move_count = np.zeros(200, dtype=np.float32), 0
    for idx, move in enumerate(chessboard.legal_moves):
        move = mirror_move(move) if not chessboard.turn else move
        move_indices = move_to_id(move)
        move_index = move_indices[0] * 64 + move_indices[1] * 8 + move_indices[2]
        move_probabilities[idx] = policy_vector[move_index]
        move_count += 1

    return move_probabilities[:move_count]


def call_neural_net(board, neural_net):
    """
    Calls the neural network for a single position
    """
    board_planes, legal_moves_mask = encode_position_for_inference(board)
    board_planes, legal_moves_mask = torch.from_numpy(board_planes).unsqueeze(0), torch.from_numpy(
        legal_moves_mask).unsqueeze(0)
    if device == 'cuda':
        board_planes, legal_moves_mask = board_planes.cuda(), legal_moves_mask.cuda()
    predicted_value, predicted_policy = neural_net(board_planes, policy_mask=legal_moves_mask)
    predicted_value, predicted_policy = predicted_value.cpu().numpy()[0, 0], predicted_policy.cpu().numpy()[0]
    move_probabilities = decode_policy_output(board, predicted_policy)
    return predicted_value, move_probabilities


def call_batched_neural_net(chessboards, neural_net):
    """
    Calls the neural network in batches for multiple positions
    """
    num_inputs = len(chessboards)
    board_inputs = torch.zeros((num_inputs, 16, 8, 8), dtype=torch.float32)
    policy_masks = torch.zeros((num_inputs, 72, 8, 8), dtype=torch.float32)
    for i, board in enumerate(chessboards):
        board_planes, legal_moves_mask = encode_position_for_inference(board)
        board_inputs[i], policy_masks[i] = torch.from_numpy(board_planes), torch.from_numpy(legal_moves_mask)
    if device == 'cuda':
        board_inputs, policy_masks = board_inputs.cuda(), policy_masks.cuda()
    predicted_values, predicted_policies = neural_net(board_inputs, policy_mask=policy_masks)
    predicted_values, predicted_policies = predicted_values.cpu().numpy().flatten(), predicted_policies.cpu().numpy()

    move_probabilities = np.zeros((num_inputs, 200), dtype=np.float32)
    for i, board in enumerate(chessboards):
        move_probabilities_tmp = decode_policy_output(board, predicted_policies[i])
        move_probabilities[i, :len(move_probabilities_tmp)] = move_probabilities_tmp
    return predicted_values, move_probabilities
