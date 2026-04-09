import chess
import numpy as np

def board_to_tensor(board):
    
    # Use one-hot encoding for each piece type
    representation = np.zeros(64)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            val = piece.piece_type
            representation[i] = val if piece.color == chess.WHITE else -val
    return representation