'''PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.

'''

import BC_state_etc as BC
import random
import math

ZOB_TBL = dict()    # Table that maps each row-col-piece combo to a unique hash
ZOB_STATES = dict() # Table that maps board hashes to their static values

PIECES = ['-', 'p', 'P', 'c', 'C', 'l', 'L', 'i', 'I ', 'w', 'W', 'k', 'K', 'f', 'F']
# White pieces represented with lowercase letters, black with uppercase

STATIC_VALUES = {'-':0, 'p':-1, 'P':1, 'c':-5, 'C':5, 'l':-2, 'L':2,\
                 'i':-4, 'I':4, 'w':-3, 'W':3, 'k':0, 'K':0, 'f':-4, 'F':4}

def static_eval(board):
    global STATIC_VALUES
    return sum((STATIC_VALUES[p] for row in board for p in row))

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
    if winTester(board):
        return None, "Win" + str(whoseMove)

    successor_boards = generate_successors(board, whoseMove)

    if ply_remaining <= 0 or len(successor_boards) <= 0:
        return static_eval(board), None

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
            attached_move = ((row, col), (row1, col1))

            # Update alpha and beta
            if whoseMove == 'MaxW':
                 alpha = max(alpha, bestScore)
            elif whoseMove == 'MinB':
                 beta = min(beta, bestScore)

    return bestScore, attached_move

# Checks if current board state is a win state (no king)
def is_win_state(board):
    kings_count = 0
    for row in range(8):
        for col in range(8):
            if board[row][col] == 'k' or 'K':
                kings_count += 1
    return kings_count == 2

# Generates successors from input board by finding all possible moves
def generate_successors(board, whoseMove):
    successors = []
    movablePieces = 'pcliwkf'
    opponentPieces = movablePieces.upper()
    if whoseMove == 'MaxW':
        opponentPieces = movablePieces
        movablePieces = movablePieces.upper() # White pieces are uppercase
        
    movablePieces = list(movablePieces) # Convert string to list
    opponentPieces = list(opponentPieces)
    
    # Only calculate moves for now, not captures
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece in movablePieces:
                neighborhood = get_neighborhood(row, col)
                # Check for freezers
                neighbors = set((board[r][c] for (r,c) in neighborhood))
                if (whoseMove == 'maxW' and 'f' in neighbors)\
                   or (whoseMove == 'minB' and 'F' in neighbors):
                    # Pieces that have been frozen cannot move.
                    continue

                # If your imitator can capture a king,
                # there's no reason to take any other move.
                elif (piece == 'i' and 'K' in neighbors) or\
                     (piece == 'I' and 'k' in neighbors):
                    for (r,c) in neighborhood:
                        if (piece == 'i' and board[r][c] == 'K') or\
                           (piece == 'I' and board[r][c] == 'k'):
                            successors = [apply_move(board, (row,col), (r,c))]
                            break
                    
                # Pawns and kings have special movement rules.
                # All other pieces move like standard-chess queens.
                elif piece == 'k' or piece == 'K':
                    for (r,c) in neighborhood:
                        if board[r][c] == '-' or board[r][c] in opponentPieces:
                            successors.append(apply_move(board, (row,col), (r,c)))
                            
                else:
                    directions = [(0,1), (1,0), (-1,0), (0,-1),\
                                  (1,1), (1,-1), (-1,1), (-1,-1)]
                    if piece == 'p' or piece == 'P':
                        directions = [(0,1), (1,0), (-1,0), (0,-1)]
                    for (dr,dc) in directions:
                        (new_r, new_c) = (row+dr, col+dc)
                        while valid_space(new_r, new_c) and\
                              board[new_r][new_c] == '-':
                            possibleSpaces.append((new_r, new_c))
                            new_r += dr
                            new_c += dc
                        # Leapers can leap (and imitators can leap over leapers)
                        # The 'leapee' should be at board[new_r][new_c]
                        if valid_space(new_r + dr, new_c + dc) and\
                           board[new_r + dr][new_c + dc] == '-':
                            target = board[new_r][new_c]
                            if target in opponentPieces and\
                               (piece == 'l' or\
                                piece == 'L' or\
                                (piece == 'i' and target == 'L') or\
                                (piece == 'I' and target == 'l')):
                                possibleSpaces.append((new_r + dr, new_c + dc))
                
                    for (new_r, new_c) in possibleSpaces:
                        # Apply move to board
                        new_board = apply_move(board, row, col, new_r,new_c)
                        # Apply any captures to board
                        new_boards = apply_captures(board, row,col, new_r,new_c, opponentPieces, whoseMove)
                        successors.extend(new_boards)
    return successors


def valid_space(row, col):
    # Returns whether the given coordinates fall within the boundaries of the board
    return (0 <= row < 8) and (0 <= col < 8)


def apply_captures(board, old_r, old_c, new_r, new_c, piece, capturablePieces, whoseMove):
    (dr, dc) = (to_space[0] - from_space[0], to_space[1] - from_space[1])
    (dr, dc) = ((dr > 0) - (dr < 0), (dc > 0) - (dc < 0))  # Make dr and dc either 1, 0, or -1
    
    # Looks for all possible captures, and then applies them, returning a list of new board states
    boards = []

    # Imitators capture by 'imitating' the piece to be captured
    if piece == 'i' or piece == 'I':
        # Imitators cannot imitate freezers or other imitators.
        # They can imitate kings; however, that's already handled above.
        possiblePieces = list('pclw')
        if piece == 'I':
            possiblePieces = possiblePieces.upper()
        if dr != 0 and dc != 0:
            # Imitators cannot imitate pawns when moving diagonally
            possiblePieces = possiblePieces[1:]

        for otherPiece in possiblePieces:
            # Note that capturablePieces below consists solely of
            # the opposing counterpart to otherPiece
            boards.extend(apply_captures(board, old_r, old_c, new_r, new_c,\
                                         otherPiece, [otherPiece.swapcase()]))

    # Pawns capture by 'surrounding' opposing pieces
    elif piece == 'p' or piece == 'P':
        directions = [(0,1), (1,0), (-1,0), (0,-1)]
        for (drow, dcol) in directions:
            if valid_space(new_r + drow*2, new_c + dcol*2)\
               and board[new_r+drow][new_c+dcol] in capturablePieces\
               and board[new_r+drow*2][new_c+dcol*2] == piece:
                new_board = copy_board(board)
                new_board[new_r+drow][new_c+dcol] = '-'
                boards.append[new_board]

    # Coordinators capture by 'coordinating' with the king
    elif piece == 'c' or piece == 'C':
        (king_r, king_c) = friendly_king_position(board, whoseMove)
        # Check the two spaces that the king and coordinator 'coordinate' together
        for (r,c) in [(to_space[0],king_c), (king_r,to_space[1])]:
            if board[r][c] in capturablePieces:
                new_board = copy_board(board)
                new_board[r][c] = '-'
                boards.append[new_board]

    # Withdrawers capture by 'withdrawing' from an opposing piece
    elif piece == 'w' or piece == 'W':
        # Check the space 'behind' the withdrawer
        if valid_space(old_r - dr, old_c - dc)\
           and board[old_r - dr][old_c - dc] in capturablePieces:
            new_board = copy_board(board)
            new_board[old_r - dr][old_c - dc] = '-'
            boards.append[new_board]

    # Leapers capture by 'leaping over' opposing pieces
    elif piece == 'l' or piece == 'L':
        # Check the space 'behind' the leaper's final position
        if board[new_r - dr][new_c - dc] in capturablePieces:
            new_board = copy_board(board)
            new_board[new_r - dr][new_c - dc] = '-'
            boards.append[new_board]
    
    return boards

def apply_move(board, old_r, old_c, new_r, new_c):
    # Updates the given board with the result of a piece's (non-capturing) move
    new_board = [[board[r][c] for c in range(len(board[0]))] for r in range(len(board))]
    new_board[new_r][new_c] = new_board[old_r][old_c]
    new_board[old_r][old_c] = '-'
    return new_board

def get_neighborhood(row, col):
    # Returns a list of coordinates of the 8 spaces surrounding a given square.
    # For an edge, only 5 spaces will be returned; for a corner, only 3.
    spaces = [(r, c)\
              for r in range(max(0,row-1), min(8,row+2))\
              for c in range(max(0,col-1), min(8,col+2))\
              if not (r == row and c == col)]
    return spaces

def copy_board(board):
    # Returns a deep copy of a given board.
    return [[board[r][c] for c in range(len(board[0]))] for r in range(len(board))]

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
    global ZOB_TBL, ZOB_STATES, PIECES

    # Set up who player is ?

    # Set up Zobrist hashing - Assuming default board size 8 x 8
    for row in range(8):
        for col in range(8):
            for piece in PIECES:
                if piece == '-':
                    # No need to hash the empty space
                    ZOB_TBL[(row, col, piece)] = 0 
                else:
                    ZOB_TBL[(row, col, piece)] = random.getrandbits(64)


# Get hash value, do bit-wise XOR
# Automatically check if this calculation has 
def zob_hash(board):
    global ZOB_TBL, ZOB_STATES, PIECES
    hash_val = 0
    for row in range(8):
        for col in range(8):
            if board[row][col] != '-':
                hash_val ^= ZOB_TBL[(row, col, board[row][col])]
    return hash_val
