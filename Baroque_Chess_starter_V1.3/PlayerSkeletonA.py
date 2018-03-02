'''PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.

'''

import BC_state_etc as BC
import random
import math

zobrist_table = {}
pieces = ['-', 'p', 'P', 'c', 'C', 'l', 'L', 'i', 'I ', 'w', 'W', 'k', 'K', 'f', 'F']
# White pieces represented with lowercase letters, black with uppercase


def makeMove(currentState, currentRemark, timelimit):

    # Compute the new state for a move.
    # This is a placeholder that just copies the current state.
    newState = BC.BC_state(currentState.board)

    # Fix up whose turn it will be.
    newState.whose_move = 1 - currentState.whose_move
    new_move = minimax_move_finder(newState.board, newState.whose_move, 3)[1]
    
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

# Generates successors from input board by finding all possible moves
def generate_successors(board, whoseMove):
    successors = []
    movablePieces = 'pcliwkf'
    opponentPieces = movablePieces.upper()
    if whoseMove == 'MaxW':
        opponentPieces = movablePieces
        movablePieces = movablePieces.upper() # White pieces are uppercase
    movablePieces = list(movablePieces) # Convert string to list

    # Only calculate moves for now, not captures
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece in movablePieces:
                # TODO: check for freezers
                possibleSpaces = []
                # Pawns and kings have special movement rules.
                # All other pieces move like standard-chess queens.
                if piece == 'k' or piece == 'K':
                    for r in range(row-1, col+2):
                        for c in range(col-1, col+2):
                            if board[r][c] == '-' or board[r][c] in opponentPieces:
                                successors.append(apply_move(board, (row,col), (r,c)))
                else:
                    directions = [(0,1), (1,0), (-1,0), (0,-1),\
                                  (1,1), (1,-1), (-1,1), (-1,-1)]
                    if piece == 'p' or piece == 'P':
                        directions = [(0,1), (1,0), (-1,0), (0,-1)]
                    for (dr,dc) in directions:
                        space = [row+dr, col+dc]
                        while board[space[0]][space[1]] == '-':
                            possibleSpaces.append(space)
                            space[0] += dr
                            space[1] += dc
                
                for (new_r, new_c) in possibleSpaces:
                    # Apply move to board
                    new_board = apply_move(board, (row, col), (new_r, new_c))
                    
                    # Check if each move can also be a capturing move
                    # Pawns capture by 'surrounding' opposing pieces
                    if piece == 'p' or piece == 'P':
                        directions = [(0,1), (1,0), (-1,0), (0,-1)]
                        for (dr,dc) in directions:
                            if 0 <= new_r + dr*2 < 8\
                               and 0 <= new_c + dc*2 < 8\
                               and board[new_r+dr][new_c+dc] in opponentPieces\
                               and board[new_r+dr*2][new_c+dc*2] == piece:
                                new_board[new_r+dr][new_c+dc] = '-'

                    elif piece == 'c' or piece == 'C':
                        # Coordinators capture by 'coordinating' with the king
                        king_r, king_c = friendly_king_position(board, whoseMove)
                        for (r,c) in [(new_r,king_c), (king_r,new_c)]:
                            if board[r][c] in opponentPieces:
                                new_board[r][c] = '-'
                    
                    successors.append(new_board)
    return successors


def apply_move(board, from_space, to_space):
    newBoard = [[board[r][c]\
                 for c in range(len(board[0]))]\
                for r in range(len(board))]
    newBoard[to_space[0]][to_space[1]] = board[from_space[0]][from_space[1]]
    newBoard[from_space[0]][from_space[1]] = '-'
    return newBoard

def friendly_king_position(board, whoseMove):
    king = 'k'
    if whoseMove == 'MaxW':
        king = 'K'
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == king:
                return row,col
            
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
