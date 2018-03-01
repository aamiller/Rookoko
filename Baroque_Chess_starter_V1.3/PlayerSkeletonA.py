'''PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.

'''

import BC_state_etc as BC
import random
import math

zobrist_table = {}
pieces = ['-', 'p', 'P', 'c', 'C', 'l', 'L', 'i', 'I w', 'W', 'k', 'K', 'f', 'F']
# White pieces represented with lowercase letters, black with uppercase


def makeMove(currentState, currentRemark, timelimit):

    # Compute the new state for a move.
    # This is a placeholder that just copies the current state.
    newState = BC.BC_state(currentState.board)

    # Fix up whose turn it will be.
    newState.whose_move = 1 - currentState.whose_move
    new_move = minimax_move_finder(newState.board, newState.whose_move, 3, -math.inf, math.inf)[1]
    
    # Construct a representation of the move that goes from the
    # currentState to the newState.
    # Here is a placeholder in the right format but with made-up
    # numbers:
    move = ((6, 4), (3, 4))

    # Make up a new remark
    newRemark = "I'll think harder in some future game. Here's my move"

    return [[move, newState], newRemark]

def minimax_move_finder(board, whoseMove, ply_remaining, alpha=-math.inf, beta=math.inf):
    # Check if a win state


    successor_boards = generate_successors(board, whoseMove)

    if ply_remaining <= 0 or len(successor_boards) <= 0:
        return  # (board state value), None

    if whoseMove == 'MaxW': bestScore = -math.inf
    else: bestScore = math.inf

    attached_move = None
    # Loop through all keys (board states)
    for s in successor_boards:
        # Stop looking at board if alpha beta pruning conditions met
        if alpha >= beta:
            return attached_move, bestScore

        result = minimax_move_finder(s, "MaxW" if whoseMove == "MinB" else "MinB", ply_remaining - 1, alpha, beta)
        newScore = result[0]

        if (whoseMove == "MaxW" and newScore > bestScore) \
                or (whoseMove == 'MinB' and newScore < bestScore):
            bestScore = newScore
            attached_move = successor_boards[str(s)]

            # Update alpha and beta
            if whoseMove == 'MaxW':
                 alpha = max(alpha, bestScore)
            elif whoseMove == 'MinB':
                 beta = min(beta, bestScore)

    return bestScore, attached_move

# Generates successors from input board
def generate_successors(board):
    return

def nickname():
    return "Newman"

def introduce():
    return "I'm Rookoko, an exuberant Baroque Chess agent."

def prepare(player2Nickname):
    global zobrist_table, pieces

    # Set up who player is ?

    # Set up Zobrist hashing - Assuming default board size 8 x 8
    for row in range(8):
        for col in range(8):
            for piece in pieces:
                if piece == '-':
                    zobrist_table[(row, col, piece)] = 0 # Don't bother with a hash for the empty space
                else:
                    zobrist_table[(row, col, piece)] = random.getrandbits(64)

# Get hash value, do bit-wise XOR
def zob_hash(board):
    global zobrist_table, pieces
    hash_val = 0
    for row in range(8):
        for col in range(8):
            if board[row][col] != '-':  # If not empty, get corresponding piece num from dictionary to find hash
                hash_val ^= zobrist_table[(row, col, board[row][col])]
    return hash_val
