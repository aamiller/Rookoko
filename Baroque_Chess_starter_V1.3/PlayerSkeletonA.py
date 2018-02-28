'''PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.

'''

import BC_state_etc as BC
import random

zobrist_table = {}
piece_num_dictionary = {'c': 0, 'l': 1, 'i': 2, 'w': 3, 'k': 4, 'f': 5, 'p': 6,
                          'C': 7, 'L': 8, 'I': 9, 'W': 10, 'K': 11, 'F': 12, 'P': 13}

def makeMove(currentState, currentRemark, timelimit):

    # Compute the new state for a move.
    # This is a placeholder that just copies the current state.
    newState = BC.BC_state(currentState.board)

    # Fix up whose turn it will be.
    newState.whose_move = 1 - currentState.whose_move
    
    # Construct a representation of the move that goes from the
    # currentState to the newState.
    # Here is a placeholder in the right format but with made-up
    # numbers:
    move = ((6, 4), (3, 4))

    # Make up a new remark
    newRemark = "I'll think harder in some future game. Here's my move"

    return [[move, newState], newRemark]

def minimax():
    return

def nickname():
    return "Newman"

def introduce():
    return "I'm Rookoko, an exuberant Baroque Chess agent."

def prepare(player2Nickname):
    global zobrist_table
    # Set up Zobrist hashing - Assuming default board size 8 x 8
    for row in range(0, 8):
        for col in range(0, 8):
            for piece in range(0, 13):
                zobrist_table[row][col][piece] = random.getrandbits(64)

# Get hash value, do bit-wise XOR
def zob_hash(board):
    global zobrist_table, piece_num_dictionary
    hash_val = 0
    for row in range(0, 8):
        for col in range(0, 8):
            if board[row][col] != '-':  # If not empty, get corresponding piece num from dictionary to find hash
                hash_val ^= zobrist_table[row][col][piece_num_dictionary[board[row][col]]]
    return hash_val
