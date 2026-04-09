import torch
import chess
from .environment import board_to_tensor
from .agent import DQNAgent
from .buffer import ReplayBuffer

def train():
    input_dim = 64 
    output_dim = 4672 
    agent = DQNAgent(input_dim, output_dim)
    memory = ReplayBuffer(capacity=10000)
    batch_size = 32
    episodes = 1000

    for episode in range(episodes):
        board = chess.Board()
        state = board_to_tensor(board)
        total_reward = 0
        done = False

        while not done:
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                done = True
                break

            move = agent.select_action(state, legal_moves)
            board.push(move)
            
            reward = 0
            if board.is_checkmate():
                reward = 10
            
            next_state = board_to_tensor(board)
            memory.push(state, 0, reward, next_state, done) 

            if len(memory) > batch_size:
                pass

            state = next_state
            total_reward += reward

        agent.update_epsilon()

        if episode % 50 == 0:
            print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    torch.save(agent.policy_net.state_dict(), "chess_dqn_model.pth")

if __name__ == "__main__":
    train()