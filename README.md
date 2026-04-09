# Chess-DQN: Tactical Piece Movement

This project implements a Deep Q-Network (DQN) to optimize chess piece movements through Reinforcement Learning. The agent learns board positioning and tactical decision-making via self-play and experience replay.

## Features
- Deep Q-Learning implementation using PyTorch
- Experience Replay buffer for training stability
- Dual-network architecture (Policy and Target networks)
- Integration with python-chess for state management

## Structure
- `src/network.py`: Neural network architecture
- `src/agent.py`: DQN agent logic and epsilon-greedy strategy
- `src/buffer.py`: Experience replay implementation
- `src/environment.py`: Board-to-tensor conversion logic
- `src/main.py`: Training loop and entry point

## Installation
1. Clone the repository:
   git clone https://github.com/yourusername/chess-dqn.git
2. Install dependencies:
   pip install -r requirements.txt
3. Start training:
   python src/main.py

## Future Scope
- Transitioning to Convolutional Neural Networks (CNN)
- Implementing rewards for advanced moves like castling and en passant
- Scaling the action space for full move-set complexity
