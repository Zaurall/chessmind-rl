import cv2
import numpy as np


def are_pixels_similar(pixel1, pixel2, threshold=10):
    distance = np.linalg.norm(pixel1 - pixel2)
    return distance < threshold


def remove_lines(chess_board):
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
    gray_cell = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_cell, threshold, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
    black_pixels = np.sum(thresh == 0)
    return black_pixels < 100


def get_player_side(cell):
    pass
    # TODO use model to detect the side by the bottom cell



def detect_split(screenshot):
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

    row, col = 7, 3
    player_side = get_player_side(chess_board[row*step : (row+1)*step, col*step : (col+1)*step])

    cells = []
    for row in range(len(rows)):
        for col in range(len(columns)):
            cell_image = chess_board[row * step:(row + 1) * step, col * step:(col + 1) * step]
            cell_is_empty = is_cell_empty(cell_image, black_threshold)

            if player_side == "white":
                cell_label = f"{columns[col]}{rows[7 - row]}"
            else:
                cell_label = f"{columns[7 - col]}{rows[row]}"

            cell = {"image": cell_image, "label": cell_label, "is_empty": cell_is_empty}
            cells.append(cell)

    # TODO implement state saving
