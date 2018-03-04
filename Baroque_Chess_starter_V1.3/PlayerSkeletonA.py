'''PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.

'''

import BC_state_etc as BC
import random
import math
import time

ZOB_TBL = dict()    # Table that maps each row-col-piece combo to a unique hash
ZOB_STATES = dict() # Table that maps board hashes to their static values

PIECES = ['-', 'p', 'P', 'c', 'C', 'l', 'L', 'i', 'I', 'w', 'W', 'k', 'K', 'f', 'F']
PID = BC.INIT_TO_CODE # Table that maps piece names to their numerical representations
# White pieces represented with lowercase letters, black with uppercase

STATIC_VALUES = {PID['-']:0, PID['p']:-1, PID['P']:1, PID['c']:-5, PID['C']:5, PID['l']:-2, PID['L']:2,\
                 PID['i']:-4, PID['I']:4, PID['w']:-3, PID['W']:3, PID['k']:0, PID['K']:0, PID['f']:-4, PID['F']:4}
STATIC_BOARD = None

# Special global variable that stores a potential move involves a
# leaper or imitator leaping. In that case, a capture MUST happen.
LEAPER_CAPTURE = None 

def static_eval(board):
    global STATIC_VALUES, STATIC_BOARD
    if STATIC_BOARD is None:
        STATIC_BOARD = dict()
        for r in range(8):
            for c in range(8):
                for key in STATIC_VALUES.keys():
                    STATIC_BOARD[(r,c,key)] = pos_val(r,c)*STATIC_VALUES[key]
    
    pieces = set((p for row in board for p in row))
    if PID['k'] not in pieces:
        # Missing black king is a win for white
        return math.inf
    if PID['K'] not in pieces:
        # Missing white king is a win for black
        return -math.inf
    
    val = sum((STATIC_BOARD[(r,c,board[r][c])] for r in range(8) for c in range(8)))
    # Encourage spacing out pieces
    #val += sum(((2*BC.who(board[r][c])-1)\
    #            for r in range(8)\
    #            for c in range(8)\
    #            for (n_r,n_c) in get_neighborhood(r,c)\
    #            if board[n_r][n_c] == 0))
    return val
    
def pos_val(r, c):
    return 1 + (r)*(7-r)*(c)*(7-c)/256

def makeMove(current_state, current_remark, time_limit):

    # Fix up whose turn it will be.
    whose_move = "MinB" if current_state.whose_move == BC.BLACK else "MaxW"
    new_score, new_move_and_state = iterative_deepening_minimax((current_state.board, zob_hash(current_state.board)), whose_move, time_limit)
    print(new_score)

    # Compute the new state for a move.
    # This is a placeholder that just copies the current state.
    new_state = BC.BC_state(new_move_and_state[1])
    new_state.whose_move = 1 - current_state.whose_move
    new_move_and_state = (new_move_and_state[0], new_state)

    # Make up a new remark
    new_remark = "I'll think harder in some future game. Here's my move"

    return ((new_move_and_state), new_remark)


def iterative_deepening_minimax(board_and_hash, whoseMove, time_limit):
    # Get the time the program should return a move by, factoring in time to get to this line
    end_time = time_limit + time.time() - .01

    # Set defaults
    ply = 0
    best_move = [(1, 1), (1, 1)]
    best_score = 0

    # Run minimax with increasing ply while time remaining
    while time.time() <= end_time:
        ply += 1
        next_move, next_score = minimax_move_finder(board_and_hash, whoseMove, ply, end_time, -math.inf, math.inf)

        if time.time() <= end_time and next_move is not None:
            best_move = next_move
            best_score = next_score
    return best_move, best_score




def minimax_move_finder(board_and_hash, whoseMove, ply_remaining, end_time, alpha=-math.inf, beta=math.inf):
    global ZOB_STATES

    board, zhash = board_and_hash
    if zhash not in ZOB_STATES:
        ZOB_STATES[zhash] = static_eval(board)
    value = ZOB_STATES[zhash]
    
    # Check if a win state
    if is_win_state(board):
        return value, None

    successor_boards = generate_successors(board, zob_hash(board), whoseMove)

    if ply_remaining <= 0 or len(successor_boards) <= 0:
        return value, None

    best_score = math.inf
    next_player = 'MaxW'
    if whoseMove == 'MaxW':
        best_score = -math.inf
        next_player = 'MinB'

    attached_move_and_state = None

    # Loop through all possible successor board states
    for s_move, s_board_and_hash in successor_boards:
        # Check that there is time to deepen, if not return best move so far
        if time.time() <= end_time - .01:
            return best_score, attached_move_and_state

        # Stop searching if alpha-beta pruning conditions met
        if alpha >= beta:
            return best_score, attached_move_and_state

        result = minimax_move_finder(s_board_and_hash, next_player, ply_remaining - 1, alpha, beta)
        s_score = result[0]

        if (whoseMove == "MaxW" and s_score > best_score) \
                or (whoseMove == 'MinB' and s_score < best_score): # If two choices are equally good, choose randomly?
            best_score = s_score
            attached_move_and_state = (s_move, s_board_and_hash[0])

            # Update alpha and beta
            if whoseMove == 'MaxW':
                 alpha = max(alpha, best_score)
            elif whoseMove == 'MinB':
                 beta = min(beta, best_score)

    return best_score, attached_move_and_state

# Checks if current board state is a win state (no king)
def is_win_state(board):
    pieces = set((p for row in board for p in row))
    return (PID['k'] not in pieces) or (PID['K'] not in pieces)

# Generates successors from input board by finding all possible moves
def generate_successors(board, zhash, whoseMove):
    global PID, LEAPER_CAPTURE
    successors = []
    movablePieces = 'pcliwkf'
    opponentPieces = movablePieces.upper()
    if whoseMove == 'MaxW':
        opponentPieces = opponentPieces.swapcase()
        movablePieces = movablePieces.swapcase() # White pieces are uppercase
        
    movablePieces = set(PID[piece] for piece in movablePieces) # Convert string to list
    opponentPieces = set(PID[piece] for piece in opponentPieces)

    board_and_hash = (board, zhash)
    
    # Only calculate moves for now, not captures
    potentials = set(((row,col) for row in range(8) for col in range(8) if board[row][col] in movablePieces))
    for row,col in potentials:
        LEAPER_CAPTURE = []
        piece = board[row][col]
        neighborhood = get_neighborhood(row, col)
        # Check for freezers
        neighbors = set((board[r][c] for (r,c) in neighborhood))
        if (whoseMove == 'maxW' and PID['f'] in neighbors)\
           or (whoseMove == 'minB' and PID['F'] in neighbors):
            # Pieces that have been frozen cannot move.
            continue

        # If your imitator can capture a king,
        # there's no reason to take any other move.
        elif (piece == PID['i'] and PID['K'] in neighbors) or\
             (piece == PID['I'] and PID['k'] in neighbors):
            for (new_r,new_c) in neighborhood:
                if (piece == PID['i'] and board[new_r][new_c] == PID['K']) or\
                   (piece == PID['I'] and board[new_r][new_c] == PID['k']):
                    successors = [apply_move(board_and_hash, row, col, new_r, new_c)]
                    break
            
        # Pawns and kings have special movement rules.
        # All other pieces move like standard-chess queens.
        elif piece == PID['k'] or piece == PID['K']:
            for (new_r,new_c) in neighborhood:
                if board[new_r][new_c] == PID['-'] or board[new_r][new_c] in opponentPieces:
                    successors.append(apply_move(board_and_hash, row, col, new_r, new_c))
                    
        else:
            possible_spaces = []
            directions = [(0,1), (1,0), (-1,0), (0,-1),\
                          (1,1), (1,-1), (-1,1), (-1,-1)]
            if piece == PID['p'] or piece == PID['P']:
                directions = [(0,1), (1,0), (-1,0), (0,-1)]
            for (dr,dc) in directions:
                (new_r, new_c) = (row+dr, col+dc)
                while valid_space(new_r, new_c) and\
                      board[new_r][new_c] == PID['-']:
                    possible_spaces.append((new_r, new_c))
                    new_r += dr
                    new_c += dc
                # Leapers can leap (and imitators can leap over leapers)
                # The 'leapee' should be at board[new_r][new_c]
                if valid_space(new_r + dr, new_c + dc) and\
                   board[new_r + dr][new_c + dc] == PID['-']:
                    target = board[new_r][new_c]
                    if target in opponentPieces and\
                       (piece == PID['l'] or\
                        piece == PID['L'] or\
                        (piece == PID['i'] and target == PID['L']) or\
                        (piece == PID['I'] and target == PID['l'])):
                        LEAPER_CAPTURE.append((new_r + dr, new_c + dc))
                        possible_spaces.append((new_r + dr, new_c + dc))
        
            for (new_r, new_c) in possible_spaces:
                # Apply move to board
                new_move, new_board_and_hash = apply_move(board_and_hash, row, col, new_r,new_c)
                # Apply any captures to board
                new_boards = apply_captures(new_board_and_hash, row,col, new_r, new_c,\
                                            piece, opponentPieces, whoseMove)
                successors.extend(((new_move, b) for b in new_boards))
    return successors


def valid_space(row, col):
    # Returns whether the given coordinates fall within the boundaries of the board
    return (0 <= row < 8) and (0 <= col < 8)


def apply_captures(board_and_hash, old_r, old_c, new_r, new_c, piece, capturablePieces, whoseMove):
    global LEAPER_CAPTURE, ZOB_TBL
    board, zhash = board_and_hash
    # Looks for all possible captures, and then applies them, returning a list of new board states
    
    # Fast and mysterious way to make dr and dc either 1, 0, or -1
    (dr, dc) = ((old_r > new_r) - (old_r < new_r),\
                (old_c > new_c) - (old_c < new_c))

    # Leapers capture by 'leaping over' opposing pieces
    # Leaper captures must be handled specially, because moving without capture is not acceptable.
    # Note that this will also handle the case of imitators imitating leapers
    if ((old_r, old_c), (new_r, new_c)) in LEAPER_CAPTURE:
        # The space 'behind' the leaper's final position will already have been checked above
        # if board[new_r - dr][new_c - dc] in capturablePieces:
        LEAPER_CAPTURE.remove((old_r, old_c), (new_r, new_c))
        new_board = copy_board(board)
        new_board[new_r - dr][new_c - dc] = PID['-']
        new_hash = zhash ^ ZOB_TBL[(new_r - dr, new_c - dc, board[new_r - dr][new_c - dc])]
        return [(new_board, new_hash)]

    # We will assume that moving without capturing is considered acceptable
    boards = [board_and_hash]
    
    # Imitators capture by 'imitating' the piece to be captured
    if piece == PID['i'] or piece == PID['I']:
        # Imitators cannot imitate freezers or other imitators.
        # They can imitate kings and leapers; however, those are already handled above.
        possiblePieceNames = 'pcw'
        if piece == PID['I']:
            possiblePieceNames = possiblePieceNames.upper()
        if dr != 0 and dc != 0:
            # Imitators cannot imitate pincers when moving diagonally
            possiblePieceNames = possiblePieceNames[1:]

        for otherPiece in possiblePieceNames:
            # Note that capturablePieces below consists solely of
            # the opposing counterpart to otherPiece
            possibleBoards = apply_captures(board_and_hash, old_r, old_c, new_r, new_c,\
                                         PID[otherPiece], [PID[otherPiece.swapcase()]], whoseMove)
            boards.extend(possibleBoards)

    # Pincers capture by 'surrounding' opposing pieces
    # NOTE: according to the spec, pincers can pinch using ANY friendly piece
    #       (not just other pincers)
    elif piece == PID['p'] or piece == PID['P']:
        directions = [(0,1), (1,0), (-1,0), (0,-1)]
        for (drow, dcol) in directions:
            if valid_space(new_r + drow*2, new_c + dcol*2)\
               and board[new_r+drow][new_c+dcol] in capturablePieces\
               and get_owner(board[new_r+drow*2][new_c+dcol*2]) == whoseMove:
                new_board = copy_board(board)
                new_board[new_r+drow][new_c+dcol] = PID['-']
                new_hash = zhash ^ ZOB_TBL[(new_r+drow, new_c+dcol, board[new_r+drow][new_c+dcol])]
                boards.append((new_board, new_hash))

    # Coordinators capture by 'coordinating' with the king
    elif piece == PID['c'] or piece == PID['C']:
        (king_r, king_c) = friendly_king_position(board, whoseMove)
        # Check the two spaces that the king and coordinator 'coordinate' together
        for (r,c) in [(new_r,king_c), (king_r,new_c)]:
            if board[r][c] in capturablePieces:
                new_board = copy_board(board)
                new_board[r][c] = PID['-']
                new_hash = zhash ^ ZOB_TBL[(r,c,board[r][c])]
                boards.append((new_board, new_hash))

    # Withdrawers capture by 'withdrawing' from an opposing piece
    elif piece == PID['w'] or piece == PID['W']:
        # Check the space 'behind' the withdrawer
        if valid_space(old_r - dr, old_c - dc)\
           and board[old_r - dr][old_c - dc] in capturablePieces:
            new_board = copy_board(board)
            new_board[old_r - dr][old_c - dc] = PID['-']
            new_hash = zhash ^ ZOB_TBL[(old_r-dr,old_c-dc,board[old_r-dr][old_c-dc])]
            boards.append((new_board, new_hash))
    
    return boards


def get_owner(piece_id):
    if piece_id == PID['-']:
        return ''
    elif piece_id % 2 == 0:
        return 'MinB'
    elif piece_id % 2 == 1:
        return 'MaxW'
    else:
        return None # something has gone terribly wrong

def apply_move(board_and_hash, old_r, old_c, new_r, new_c):
    global ZOB_TBL
    # Returns a tuple containing the given move followed by a copy of
    # the given board with the result of a piece's (non-capturing) move
    board, new_hash = board_and_hash
    new_hash ^= ZOB_TBL[(new_r, new_c, board[new_r][new_c])]
    new_hash ^= ZOB_TBL[(old_r, old_c, board[old_r][old_c])]
    new_hash ^= ZOB_TBL[(new_r, new_c, board[old_r][old_c])]
    new_board = copy_board(board)
    new_board[new_r][new_c] = new_board[old_r][old_c]
    new_board[old_r][old_c] = PID['-']
    return ((old_r, old_c),(new_r, new_c)), (new_board, new_hash)

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
    king = PID['k']
    if whoseMove == 'MaxW':
        king = PID['K']
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == king:
                return row,col
            

def nickname():
    return "Rookoko"

def introduce():
    return "I'm Rookoko, an exuberant Baroque Chess agent."

def prepare(player2Nickname):
    global ZOB_TBL, ZOB_STATES, PIECES, PID

    # Set up Zobrist hashing - Assuming default board size 8 x 8
    for row in range(8):
        for col in range(8):
            for piece in PIECES:
                if piece == '-':
                    # No need to hash the empty space
                    ZOB_TBL[(row, col, PID[piece])] = 0
                else:
                    ZOB_TBL[(row, col, PID[piece])] = random.getrandbits(64)
    return "Ready to rumble!"


# Get hash value, do bit-wise XOR
def zob_hash(board):
    global ZOB_TBL, ZOB_STATES
    hash_val = 0
    for row in range(8):
        for col in range(8):
            if board[row][col] != PID['-']:
                hash_val ^= ZOB_TBL[(row, col, board[row][col])]
    return hash_val
