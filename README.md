# Autonomous Chess Bot: Combining Computer Vision and Reinforcement Learning

## About the Project

This project focuses on building an autonomous chess-playing bot capable of competing on online platforms like [chess.com](https://chess.com) and [lichess.org](https://lichess.org). The bot integrates two primary components:

1. **Computer Vision (CV)**: Using advanced image processing techniques, the bot identifies the chessboard, detects the positions of pieces, and recognizes the player's side. This information is converted into a format compatible with an internal chess engine. Additionally, the bot mimics human interaction with the website by simulating mouse clicks and movements.

2. **Reinforcement Learning (RL)**: The bot makes strategic decisions based on the detected board state using a reinforcement learning model. Inspired by AlphaZero, it employs a combination of deep neural networks and Monte Carlo Tree Search (MCTS) to evaluate moves and plan strategies.

### Key Features
- Detects and processes chessboard configurations from screenshots in real time.
- Recognizes all chess pieces and tracks their movement accurately.
- Uses a trained RL model to make intelligent moves, achieving competitive performance against online chess bots.

### Challenges and Limitations
The project encountered challenges in training reinforcement learning models due to resource constraints and limited datasets. Additionally, some chess-specific actions like castling and pawn promotion require further optimization.

### Future Scope
The bot can be further enhanced by:
- Improving the accuracy of computer vision for edge cases.
- Scaling up RL training with larger datasets and more computational power.
- Supporting different chess variants and integrating with other platforms.

This project demonstrates the potential of combining CV and RL for real-time, autonomous gameplay, opening pathways for future advancements in AI-driven gaming.


---

