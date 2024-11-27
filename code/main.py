import chess
import os
import time
import cv2
import joblib
import torch
import pyautogui
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from torchvision import transforms
from models import SimpleCNN, AlphaZero

changed_color = False
pyautogui.FAILSAFE = False
player_side = None

# Loading a letter detection model from a checkpoint
letter_detection_model = SimpleCNN()
letter_detection_model_path = os.path.join(os.getcwd(), 'models', 'cv_letter_side_detection',
                                           'letter_detection_model.pt')
letter_detection_model.load_state_dict(torch.load(letter_detection_model_path, weights_only=True))
letter_detection_model.eval()

# Loading a piece detection model from a checkpoint
piece_detection_model_path = os.path.join(os.getcwd(), 'models', 'cv_pieces_classification', 'model_figures.h5')
piece_detection_model = load_model(piece_detection_model_path)

# Loading an agent from a checkpoint
game_model_path = os.path.join(os.getcwd(), 'models', 'chess_alpha_zero', 'alpha_zero_net.pt')
game_model = AlphaZero(game_model_path)


def are_pixels_similar(pixel1, pixel2, threshold=10):
    """
    Check whether two pixels' values differ within a sertain range.

    Parameters
    ----------
    `pixel1`: numpy.ndarray
        Value of a pixel 1.
    `pixel2`: numpy.ndarray
        Value of a pixel 2.
    'threshold': int=10
        Maximum difference between pixels to consider them similar.

    Returns
    ----------
    `result`: Bool
        True if pixels are similar and vice versa.
    """
    distance = np.linalg.norm(pixel1 - pixel2)
    return distance < threshold


def remove_lines(chess_board):
    """
    Crop an image to remove borders.

    Parameters
    ----------
    `chess_board`: numpy.ndarray
        Image of a chessboard.

    Returns
    ----------
    `chess_board`: numpy.ndarray
        Cropped version of a chessboard.
    """
    while not are_pixels_similar(chess_board[10][0], chess_board[10][1]):  # left
        chess_board = chess_board[:, 1:]
    while not are_pixels_similar(chess_board[0][10], chess_board[1][10]):  # top
        chess_board = chess_board[1:, :]
    while not are_pixels_similar(chess_board[-11][-1], chess_board[-11][-2]):  # right
        chess_board = chess_board[:, :-1]
    while not are_pixels_similar(chess_board[-1][-11], chess_board[-2][-11]):  # bottom
        chess_board = chess_board[:-1, :]
    return chess_board


def is_cell_empty(cell_image, threshold):
    """
    Check whether there is a figure in the cell.

    Parameters
    ----------
    `cell_image`: numpy.ndarray
        Image of a cell.
    `threshold`: int
        Amount of 'different' pixels.

    Returns
    ----------
    `result`: Bool
        Signals whether the cell is empty or not.
    """
    gray_cell = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_cell, threshold, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
    black_pixels = np.sum(thresh == 0)
    return black_pixels < 100


def get_player_side(cell):
    """
    Check whether the player is playing for black or white figures.

    Parameters
    ----------
    `cell`: numpy.ndarray
        Image of a certain cell (4th cell in the lowest row).

    Returns
    ----------
    `result`: int
        Signals whether the player is playing for black or white figures.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((20, 20)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    height, width, _ = cell.shape
    rect_size = 28

    bottom_left = transform(cell[height - rect_size:height, 0:rect_size])
    bottom_right = transform(cell[height - rect_size:height, width - rect_size:width])

    bottom_left = bottom_left.unsqueeze(0)
    bottom_right = bottom_right.unsqueeze(0)

    _, predicted_bottom_left = torch.max(letter_detection_model(bottom_left).data, 1)
    _, predicted_bottom_right = torch.max(letter_detection_model(bottom_right).data, 1)

    if predicted_bottom_left == 1 or predicted_bottom_right == 1:
        return chess.WHITE
    elif predicted_bottom_left == 2 or predicted_bottom_right == 2:
        return chess.BLACK


def predict_chess_piece(cell):
    """
    Detect a figure from an image.

    Parameters
    ----------
    `cell`: numpy.ndarray
        Image of a cell.

    Returns
    ----------
    `figure`: str
        Detected figure label.
    """
    figure_to_label = {
        0: 'b',
        1: 'k',
        2: 'n',
        3: 'p',
        4: 'q',
        5: 'r'
    }

    image_size = (224, 224)
    cell = tf.image.resize(cell, image_size)
    cell = cell / 255.0

    cell = tf.expand_dims(cell, axis=0)

    figure_tensor, color_tensor = piece_detection_model(cell)
    best_figure_idx = tf.argmax(figure_tensor, axis=1).numpy()[0]
    best_color_idx = tf.argmax(color_tensor, axis=1).numpy()[0]
    figure = figure_to_label[best_figure_idx]

    return figure if best_color_idx == 0 else figure.upper()


def set_chessboard(cells, player_side):
    """
    Get a FEN from image of a chessboard.

    Parameters
    ----------
    `cells`: list(dict)
        List of dicts in form {'image': numpy.ndarray, 'label': str, 'is_empty': bool}.
    `player_side`: int
        0 or 1 - players' color.

    Returns
    ----------
    `fen`: str
        The board setup.
    """
    fen = ""
    string = ""
    empty_cells = 0
    if player_side == chess.BLACK:
        cells = cells[::-1]
    for i, cell in enumerate(cells, 1):
        if cell["is_empty"]:
            empty_cells += 1
        else:
            if empty_cells > 0:
                string += f"{empty_cells}"
            string += predict_chess_piece(cell["image"])
            empty_cells = 0

        if i % 8 == 0:
            if empty_cells > 0:
                string += f"{empty_cells}"
            fen += f"{string}/" if i < 64 else f"{string}"
            string = ""
            empty_cells = 0

    fen += f" {'w ' if player_side == chess.WHITE else 'b '}"
    fen += 'KQkq '
    fen += "- 0 1"

    return fen


def create_easy_board(fen):
    """
    Turn FEN into array of symbols.

    Parameters
    ----------
    `fen`: str
        The board setup.

    Returns
    ----------
    `rows`: list
        The board setup (visualised).
    """
    rows = []
    for row in fen.split("/"):
        new_row = []
        for char in row:
            if char.isdigit():
                new_row.extend(["-"] * int(char))
            else:
                new_row.append(char)
        rows.append(new_row)
    return rows


def find_move(fen_before, fen_after, player_side):
    """
    Analyze the difference between two chess board states to find last move.

    Parameters
    ----------
    `fen_before`: str
        1st board setup.
    `fen_after`: str
        2nd board setup.
    `player_side`: int
        0 or 1 - players' color.

    Returns
    ----------
    `move_from`: str
        Index of some cell.
    `move_to`: str
        Index of some cell.
    """
    board_before = create_easy_board(fen_before)
    board_after = create_easy_board(fen_after)

    move_from = []
    move_to = []

    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    rows = ['1', '2', '3', '4', '5', '6', '7', '8']

    for row in range(len(rows)):
        for col in range(len(columns)):
            if board_before[row][col] != board_after[row][col]:
                if board_after[row][col] == "-":
                    if player_side == chess.BLACK:
                        move_from.append(f"{columns[7 - col]}{rows[row]}")
                    else:
                        move_from.append(f"{columns[col]}{rows[7 - row]}")
                else:
                    if player_side == chess.BLACK:
                        move_to.append(f"{columns[7 - col]}{rows[row]}")
                    else:
                        move_to.append(f"{columns[col]}{rows[7 - row]}")

    if len(move_from) == 0:
        return None, None
    if len(move_from) == 1:
        return move_from[0], move_to[0]
    elif "c1" in move_to:
        return "e1", "c1"
    elif "c8" in move_to:
        return "e8", "c8"
    elif "g1" in move_to:
        return "e1", "g1"
    elif "g8" in move_to:
        return "e8", "g8"


def detect_move(cells, player_side):
    """
    Perform detected move to the board if it is possible.

    Parameters
    ----------
    `cells`: list(dict)
        List of dicts in form {'image': numpy.ndarray, 'label': str, 'is_empty': bool}.
    `player_side`: int
        0 or 1 - players' color.
    """
    global board, changed_color

    easy_fen_before = board.fen()
    easy_fen_after = set_chessboard(cells, player_side)

    move_from, move_to = find_move(easy_fen_before, easy_fen_after, player_side)

    print(easy_fen_before, easy_fen_after, move_from, move_to, player_side)

    if move_from is not None and move_to is not None:
        move = chess.Move.from_uci(move_from + move_to)
        if move in board.legal_moves:
            changed_color = True
            board.push(move)
            append_to_text_widget(move_from + move_to + " " + board.fen() + " " + "\n")


def detect_split(screenshot):
    """
    Detect a chessboard on the screenshot and split it into cells.

    Parameters
    ----------
    `screenshot`: numpy.ndarray
        Screenshot image.

    Returns
    ----------
    `x, y, w, h`: int
        Measures of the board.
    """
    global board, changed_color, player_side

    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    sigma = 0.33
    v = np.mean(screenshot_gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(screenshot_gray, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    (x, y, w, h) = cv2.boundingRect(contours[0])
    best_area = cv2.contourArea(contours[0])
    for i, cnt in enumerate(contours):
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            area_diff = abs(area - best_area) / best_area
            if area_diff > 0.05:
                break
            (x, y, w, h) = cv2.boundingRect(cnt)

    chess_board = screenshot[y:y + h, x:x + w]
    chess_board = remove_lines(chess_board)
    chess_board = cv2.resize(chess_board, (1024, 1024))

    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    rows = ['1', '2', '3', '4', '5', '6', '7', '8']
    step = (chess_board.shape[0] // 8)

    # check for chessarena.com chessboard since it has wood texture. setting another threshold for emptiness of cell
    chess_board_gray = cv2.cvtColor(chess_board, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(chess_board_gray, 160, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
    black_threshold = 40 if np.sum(thresh == 0) > 700_000 else 100

    if board == None:
        row, col = 7, 3
        player_side = get_player_side(chess_board[row * step: (row + 1) * step, col * step: (col + 1) * step])

    cells = []
    for row in range(len(rows)):
        for col in range(len(columns)):
            cell_image = chess_board[row * step:(row + 1) * step, col * step:(col + 1) * step]
            cell_is_empty = is_cell_empty(cell_image, black_threshold)

            if player_side == chess.WHITE:
                cell_label = f"{columns[col]}{rows[7 - row]}"
            else:
                cell_label = f"{columns[7 - col]}{rows[row]}"

            cell = {"image": cell_image, "label": cell_label, "is_empty": cell_is_empty}
            cells.append(cell)

    return x, y, w, h, cells, player_side


board = None

is_screenshotting = False  # Start as False (not taking screenshots)
move_in_progress = False
x, y, w, h = 0, 0, 0, 0


def take_screenshots():
    """
    Continuously take screenshots and detect board.
    """
    global is_screenshotting, move_in_progress, board, x, y, w, h

    while True:  # Continuous loop while screenshotting is active
        if is_screenshotting:
            screenshot = pyautogui.screenshot()  # Take a screenshot
            x, y, w, h, cells, player_side = detect_split(screenshot)  # Process screenshot for board updates

            if board == None:
                board = chess.Board(set_chessboard(cells, player_side))
                print(board)
            else:
                easy_fen_before = board.fen()
                easy_fen_after = set_chessboard(cells, player_side)
                if easy_fen_before != easy_fen_after:
                    x, y, w, h, cells, player_side = detect_split(screenshot)
                    detect_move(cells, player_side)

            if board.is_game_over():
                print("Game over, stopping screenshots...")
                end_game()
            else:
                if board.turn == player_side:
                    begin_move()
        time.sleep(1.0)


def make_move():
    """
    Chose an action using a model and perform it.
    """
    fen = board.fen()
    move = game_model.choose_action(fen)
    action = board.uci(move)
    append_to_text_widget(action + "\n")

    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    rows = ['1', '2', '3', '4', '5', '6', '7', '8'][::-1]

    pyautogui.click(x + columns.index(action[0]) * w // 8 + w // 16, y + rows.index(action[1]) * h // 8 + h // 16)
    time.sleep(0.2)
    pyautogui.click(x + columns.index(action[2]) * w // 8 + w // 16, y + rows.index(action[3]) * h // 8 + h // 16)
    time.sleep(0.2)
    if len(action) > 4:
        if action[4] == 'q':
            pyautogui.click(x + columns.index(action[2]) * w // 8 + w // 16,
                            y + rows.index(action[3]) * h // 8 + h // 16)
        elif action[4] == 'n':
            pyautogui.click(x + columns.index(action[2]) * w // 8 + w // 16,
                            y + rows.index(action[3]) * h // 8 + h // 16 * 3)
        elif action[4] == 'r':
            pyautogui.click(x + columns.index(action[2]) * w // 8 + w // 16,
                            y + rows.index(action[3]) * h // 8 + h // 16 * 5)
        elif action[4] == 'b':
            pyautogui.click(x + columns.index(action[2]) * w // 8 + w // 16,
                            y + rows.index(action[3]) * h // 8 + h // 16 * 7)
    time.sleep(1.0)
    board.push(move)


import tkinter as tk
import threading

screenshot_thread = threading.Thread(target=take_screenshots)


def start_screenshotting():
    """
    Start a thread to take screenshots.
    """
    global is_screenshotting
    if not is_screenshotting:
        is_screenshotting = True
        print("Screenshotting started...")
        if not screenshot_thread.is_alive():
            screenshot_thread.start()
    else:
        is_screenshotting = True


def stop_screenshotting():
    """
    Stop making screenshots unless `start_screenshotting` is triggered.
    """
    global is_screenshotting
    is_screenshotting = False
    print("Screenshotting stopped...")


def begin_move():
    """
    Stop making screenshots for the period of making a move.
    """
    global move_in_progress
    print("Begin Move button clicked")
    move_in_progress = True
    stop_screenshotting()  # Stop screenshots while making a move
    make_move()
    move_in_progress = False
    start_screenshotting()


def end_game():
    """
    Stop making screenshots.
    """
    global is_screenshotting, board
    board = None
    print("End Game button clicked")
    stop_screenshotting()  # Stop screenshots if the game is manually ended


def append_to_text_widget(message):
    """
    Functional output to the widget.
    """
    # Insert new text at the end and scroll to the bottom
    text_display.insert(tk.END, message)
    text_display.see(tk.END)


def create_control_window():
    """
    Create a control window and make it work in a loop.
    """
    global text_display
    root = tk.Tk()
    root.title("Chess Move Control")

    # Add buttons for controlling moves and game
    start_button = tk.Button(root, text="Start", command=start_screenshotting)
    start_button.pack(pady=10)

    end_game_button = tk.Button(root, text="End Game", command=end_game)
    end_game_button.pack(pady=10)

    text_display = tk.Text(root, height=10, width=50)
    text_display.pack(pady=10)

    root.mainloop()


# Start the tkinter control window
create_control_window()
