'''PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.

'''

from BC_state_etc import *
import random
import math
import time
import threading

TIME_INC = .015 # Global variable representing the number of seconds an iteration takes

TURNS = 0 # Global variable representing the number of turns that have passed so far

EMPTY = 0 # Global variable representing an empty space on the board

ZOB_TBL = dict()    # Table that maps each row-col-piece combo to a unique hash
ZOB_STATES = dict() # Table that maps board hashes to their static values
DYN_VALS = dict()

# Table mapping players to a set containing their pieces
PIECES = {BLACK:set((BLACK_PINCER, BLACK_COORDINATOR, BLACK_LEAPER, BLACK_IMITATOR,\
                 BLACK_WITHDRAWER, BLACK_KING, BLACK_FREEZER)),\
          WHITE:set((WHITE_PINCER, WHITE_COORDINATOR, WHITE_LEAPER, WHITE_IMITATOR,\
                 WHITE_WITHDRAWER, WHITE_KING, WHITE_FREEZER))}

# Table that maps directions to coordinate vectors
DIR_TBL = {NORTH:(-1,0), SOUTH:(1,0), WEST:(0,-1), EAST:(0,1),\
           NW:(-1,-1), NE:(-1,1), SW:(1,-1), SE:(1,1)} 

# Table that maps pieces to static values
STATIC_VALUES = {EMPTY:0,\
                 BLACK_PINCER:-1, WHITE_PINCER:1,\
                 BLACK_COORDINATOR:-5, WHITE_COORDINATOR:5,\
                 BLACK_LEAPER:-2, WHITE_LEAPER:2,\
                 BLACK_IMITATOR:-4, WHITE_IMITATOR:4,\
                 BLACK_WITHDRAWER:-3, WHITE_WITHDRAWER:3,\
                 BLACK_KING:0, WHITE_KING:0,\
                 BLACK_FREEZER:-3, WHITE_FREEZER:3}
# 
STATIC_BOARD = None

STATISTICS = {"Z_SUCCESS": 0, "Z_QUERIES": 0, "AB_CUTOFFS": 0, "STATIC_EVALS": 0}
# Special global variable that stores potential moves involving a
# leaper (or imitator) leaping. In that case, a capture MUST happen,
# so no non-capturing successor boards will be considered for that move.
LEAPER_CAPTURE = None

EVAL_THREAD = False
EVAL_THREAD_EXIT = False

PUN_GENERATOR = None
BAD_PUNS = [\
    "Defibrillators are re-pulse-ive.",\
    "Possesio is nine-tenths of the word.",\
    "What's a rounding error like? Everything feels a bit off.",\
    "Broken pencils are pointless.",\
    "Got some new glasses. They're quite the spectacle.",\
    "Misconfiguring SSL makes me feel so insecure.",\
    "Two crows on a wire? Attempted murder.",\
    "Three crows in a henhouse? Murder most fowl.",\
    "Pride goes before a fall; a pride will go after a gazelle.",\
    "There's a point to this sentence, but you won't see it until the very end.",\
    "Everyone's a waiter when the restaurant's having a slow day.",\
    "Fishing is a good example of a poisson process.",\
    "What's purple and commutes? An abelian grape.",\
    "What's yellow and equivalent to the axiom of choice? Zorn's Lemon.",\
    "Sodium hydroxide is a lye.",\
    "Liquid nitrogen is so cool!"]


def pun_generator():
    global BAD_PUNS
    last = ['']*10
    while True:
        for i in range(10):
            pun = last[0]
            while pun in last:
                pun = random.choice(BAD_PUNS)
            last[i] = pun
            yield pun

def static_eval(board):
    global STATIC_VALUES, STATIC_BOARD, EMPTY, TURNS
    
    pieces = set((p for row in board for p in row))
    if BLACK_KING not in pieces:
        # Missing black king is a win for white
        return math.inf
    elif WHITE_KING not in pieces:
        # Missing white king is a win for black
        return -math.inf
    else:
        if TURNS < 20:
            val = sum((STATIC_BOARD[(r,c,board[r][c])] for c in range(8) for r in range(8)))
        else:
            val = sum((STATIC_VALUES[board[r][c]] for c in range(8) for r in range(8)))
        # Ignore frozen pieces
        for r in range(8):
            for c in range(8):
                if board[r][c] == BLACK_FREEZER or board[r][c] == WHITE_FREEZER:
                    val -= sum((STATIC_BOARD[(nr,nc,board[nr][nc])] for (nr,nc) in get_neighborhood(r,c)\
                            if board[nr][nc] != EMPTY and who(board[r][c]) != who(board[nr][nc])))/2
        return val
    #else:
    #    return sum((STATIC_VALUES[p] for row in board for p in row if (p != WHITE_KING and p != BLACK_KING)))
    #


def pos_val(r, c):
    return 1 + ((r)*(7-r)*(c)*(7-c))/256

def makeMove(current_state, current_remark, time_limit):
    global TURNS, EVAL_THREAD, EVAL_THREAD_EXIT, DYN_VALS, ZOB_STATES, PUN_GENERATOR, STATISTICS
    if EVAL_THREAD and EVAL_THREAD.is_alive():
        EVAL_THREAD_EXIT = True
        EVAL_THREAD.join()
        EVAL_THREAD_EXIT = False
    EVAL_THREAD = False
    TURNS += 1
    if TURNS == 20:
        DYN_VALS = dict()
        ZOB_STATES = dict()

    STATISTICS = {"Z_SUCCESS": 0, "Z_QUERIES": 0, "AB_CUTOFFS": 0, "STATIC_EVALS": 0}

    # Fix up whose turn it will be.
    whose_move = current_state.whose_move
    state_hash = zob_hash(current_state.board)
    new_score, new_move_and_state, ply_used = iterative_deepening_minimax(current_state.board, state_hash, whose_move, time_limit)
    print("Current state static value:", new_score)
    print("IDDFS reached ply", ply_used, "before running out of time.")
    print("Minimax search performed", STATISTICS["AB_CUTOFFS"], "alpha-beta cutoffs.")
    print("Minimax search performed", STATISTICS["STATIC_EVALS"], "static evals.")
    print("Zobrist hash table had", STATISTICS["Z_QUERIES"], "total queries.")
    print("Zobrist hash table had", STATISTICS["Z_SUCCESS"], "successful queries.")

    # Compute the new state for a move.
    new_state = BC_state(new_move_and_state[1])
    new_state.whose_move = 1 - whose_move
    new_move_and_state = (new_move_and_state[0], new_state)

    if whose_move == WHITE:
        EVAL_THREAD = threading.Thread(target=state_evaluator, args=(new_state,))
        EVAL_THREAD.start()

    # Make up a new remark
    new_remark = next(PUN_GENERATOR)

    return ((new_move_and_state), new_remark)


def iterative_deepening_minimax(board, zhash, whoseMove, time_limit):
    global TIME_INC
    
    # Get the time the program should return a move by, factoring in time to get to this line
    end_time = time_limit + time.time() - TIME_INC

    # Set defaults
    ply = -1
    best_move = [(1, 1), (1, 1)]
    best_score = 0
    next_score = 0

    # Run minimax with increasing ply while time remaining
    while time.time() <= end_time - TIME_INC:
        ply += 1
        #print("ply:", ply)
        results = minimax_move_finder(board, zhash, whoseMove, ply, end_time, -math.inf, math.inf)
        next_score, next_move = results

        if time.time() <= end_time - TIME_INC:
            best_move = next_move
            best_score = next_score
    return best_score, best_move, ply


def state_evaluator(state):
    board = state.board
    whose_move = state.whose_move
    zhash = zob_hash(board)
    
    if zhash not in DYN_VALS:
        DYN_VALS[zhash] = ((0, None), -1)
    ply = (DYN_VALS[zhash])[1]

    best_move = 1
    while best_move is not None:
        ply += 1
        #print(("white's" if whose_move == BLACK else "black's"), "state evaluator ply:", ply)
        try:
            best_score, best_move = minimax_move_finder(board, zhash, whose_move, ply, math.inf)
        except threading.ThreadError:
            return


def minimax_move_finder(board, zhash, whoseMove, ply_remaining, end_time, alpha=-math.inf, beta=math.inf):
    global ZOB_STATES, TIME_INC, DYN_VALS, EVAL_THREAD, EVAL_THREAD_EXIT, STATISTICS
    if EVAL_THREAD and EVAL_THREAD_EXIT:
        raise threading.ThreadError

    if zhash in DYN_VALS:
        STATISTICS["Z_SUCCESS"] += 1
        dyn_ret, dyn_ply = DYN_VALS[zhash]
        if dyn_ply >= ply_remaining:
            return dyn_ret
    else:
        STATISTICS["Z_QUERIES"] += 1
    
    # Check if a win state
    win_state = is_win_state(board)
    if win_state:
        STATISTICS["Z_QUERIES"] += 1
        if zhash not in ZOB_STATES:
            ZOB_STATES[zhash] = win_state
            DYN_VALS[zhash] = ((win_state, None), math.inf)
        else:
            STATISTICS["Z_SUCCESS"] += 1
        return win_state, None

    successor_boards = generate_successors(board, zhash, whoseMove)

    if ply_remaining <= 0 or len(successor_boards) == 0:
        STATISTICS["Z_QUERIES"] += 1
        if zhash not in ZOB_STATES:
            STATISTICS["STATIC_EVALS"] += 1
            ZOB_STATES[zhash] = static_eval(board)
            DYN_VALS[zhash] = ((ZOB_STATES[zhash], None), 0)
        else:
            STATISTICS["Z_SUCCESS"] += 1
        return ZOB_STATES[zhash], None

    next_player = 1 - whoseMove
    chance = 1/2
    best_score = math.inf
    if whoseMove == WHITE: # White is the maximizing player
        best_score = -math.inf

    attached_move_and_state = None

    # Loop through all possible successor board states
    for s_move, s_board, s_hash in successor_boards:
        # Check that there is time to deepen, if not, exit
        if time.time() >= end_time:
            return best_score, None

        # Stop searching if alpha-beta pruning conditions met
        if alpha >= beta:
            STATISTICS["AB_CUTOFFS"] += 1
            return best_score, attached_move_and_state

        result = minimax_move_finder(s_board, s_hash, next_player, ply_remaining - 1, end_time - TIME_INC, alpha, beta)
        s_score = result[0]

        if (whoseMove == WHITE and s_score > best_score) \
                or (whoseMove == BLACK and s_score < best_score)\
                or (s_score == best_score and random.random() <= chance):
            best_score = s_score
            attached_move_and_state = (s_move, s_board)

            # Update alpha and beta
            if whoseMove == WHITE:
                 alpha = max(alpha, best_score)
            elif whoseMove == BLACK:
                 beta = min(beta, best_score)

    DYN_VALS[zhash] = ((best_score, attached_move_and_state), ply_remaining)
    
    return best_score, attached_move_and_state

# Checks if current board state is a win state (no king)
def is_win_state(board):
    pieces = set((p for row in board for p in row))
    if WHITE_KING not in pieces:
        return -math.inf
    elif BLACK_KING not in pieces:
        return math.inf
    else:
        return 0

# Generates successors from input board by finding all possible moves
def generate_successors(board, zhash, whoseMove):
    global PIECES, EMPTY, LEAPER_CAPTURE
    successors = []
    movablePieces = PIECES[whoseMove]
    opponentPieces = PIECES[1 - whoseMove]
    
    # Only calculate moves for now, not captures
    potentials = set(((row,col) for row in range(8) for col in range(8) if board[row][col] in movablePieces))
    for row,col in potentials:
        LEAPER_CAPTURE = []
        piece = board[row][col]
        neighborhood = get_neighborhood(row, col)
        # Check for freezers
        neighbors = set((board[r][c] for (r,c) in neighborhood))
        if (whoseMove == WHITE and BLACK_FREEZER in neighbors)\
           or (whoseMove == BLACK and WHITE_FREEZER in neighbors):
            # Pieces that have been frozen cannot move.
            continue

        # If your imitator can capture a king,
        # there's no reason to take any other move.
        elif (WHITE_KING in neighbors and piece == BLACK_IMITATOR) or\
             (BLACK_KING in neighbors and piece == WHITE_IMITATOR):
            for (new_r,new_c) in neighborhood:
                if (piece == BLACK_IMITATOR and board[new_r][new_c] == WHITE_KING) or\
                   (piece == WHITE_IMITATOR and board[new_r][new_c] == BLACK_KING):
                    successors = [apply_move(board, zhash, row, col, new_r, new_c)]
                    break
            
        # Pincers and kings have special movement rules.
        # All other pieces move like standard-chess queens.
        elif piece == BLACK_KING or piece == WHITE_KING:
            for (new_r,new_c) in neighborhood:
                if board[new_r][new_c] == EMPTY or board[new_r][new_c] in opponentPieces:
                    successors.append(apply_move(board, zhash, row, col, new_r, new_c))
        else:
            possible_spaces = []
            directions = [(0,1), (1,0), (-1,0), (0,-1),\
                          (1,1), (1,-1), (-1,1), (-1,-1)]
            if piece == BLACK_PINCER or piece == WHITE_PINCER:
                directions = [(0,1), (1,0), (-1,0), (0,-1)]
            for (dr,dc) in directions:
                (new_r, new_c) = (row+dr, col+dc)
                while valid_space(new_r, new_c) and\
                      board[new_r][new_c] == EMPTY:
                    possible_spaces.append((new_r, new_c))
                    new_r += dr
                    new_c += dc
                # Leapers can leap (and imitators can leap over leapers)
                # The 'leapee' should be at board[new_r][new_c]
                if valid_space(new_r + dr, new_c + dc) and (board[new_r + dr][new_c + dc] == EMPTY):
                    target = board[new_r][new_c]
                    if target in opponentPieces and\
                       (piece == BLACK_LEAPER or piece == WHITE_LEAPER or\
                        (piece == BLACK_IMITATOR and target == WHITE_LEAPER) or\
                        (piece == WHITE_IMITATOR and target == BLACK_LEAPER)):
                        LEAPER_CAPTURE.append([(row, col),(new_r + dr, new_c + dc)])
                        possible_spaces.append((new_r + dr, new_c + dc))
        
            for (new_r, new_c) in possible_spaces:
                # Apply move to board
                new_move, new_board, new_hash = apply_move(board, zhash, row, col, new_r, new_c)
                # Apply any captures to board
                new_boards = apply_captures(new_board, new_hash, row, col, new_r, new_c,\
                                            piece, opponentPieces, whoseMove)
                successors.extend(((new_move, b[0], b[1]) for b in new_boards))
    return successors


def valid_space(row, col):
    # Returns whether the given coordinates fall within the boundaries of the board
    return (0 <= row <= 7) and (0 <= col <= 7)


def apply_captures(board, zhash, old_r, old_c, new_r, new_c, piece, capturablePieces, whoseMove):
    global LEAPER_CAPTURE, ZOB_TBL, EMPTY
    # Looks for all possible captures, and then applies them, returning a list of new board states
    
    # Fast and mysterious way to make dr and dc either 1, 0, or -1
    (dr, dc) = ((old_r < new_r) - (old_r > new_r),\
                (old_c < new_c) - (old_c > new_c))
    # Fast and mysterious way to get the piece 'type', in terms of its black-piece equivalent
    piece_type = (piece >> 1) << 1

    # Leapers capture by 'leaping over' opposing pieces
    # Leaper captures must be handled specially, because moving without capture is not acceptable.
    # Note that this will also handle the case of imitators imitating leapers
    if [(old_r, old_c), (new_r, new_c)] in LEAPER_CAPTURE:
        # The space 'behind' the leaper's final position will already have been checked above
        LEAPER_CAPTURE.remove([(old_r, old_c), (new_r, new_c)])
        if board[new_r - dr][new_c - dc] in capturablePieces:
            new_board = copy_board(board)
            new_board[new_r - dr][new_c - dc] = EMPTY
            new_hash = zhash ^ ZOB_TBL[(new_r - dr, new_c - dc, board[new_r - dr][new_c - dc])]
            return [(new_board, new_hash)]

    # We will assume that moving without capturing is not considered acceptable
    boards = []
    #boards = [(board, zhash)]
    
    # Imitators capture by 'imitating' the piece to be captured
    if piece_type == BLACK_IMITATOR:
        # Imitators cannot imitate freezers or other imitators.
        # They can imitate kings and leapers; however, those are already handled above.
        if dr != 0 and dc != 0:
            # Imitators cannot imitate pincers when moving diagonally
            possiblePieces = set((whoseMove+BLACK_COORDINATOR,\
                                  whoseMove+BLACK_WITHDRAWER))
        else:
            possiblePieces = set((whoseMove+BLACK_PINCER,\
                                  whoseMove+BLACK_COORDINATOR,\
                                  whoseMove+BLACK_WITHDRAWER))

        # whoseMove is 0 for black and 1 for white.
        # So, (BLACK_X + whoseMove) returns a black X if whoseMove is BLACK,
        #  and a white X if whoseMove is WHITE.

        for otherPiece in possiblePieces:
            # Note that capturablePieces below consists solely of
            # the opposing counterpart to otherPiece
            possibleBoards = apply_captures(board, zhash, old_r, old_c, new_r, new_c,\
                                         otherPiece, [otherPiece ^ 1], whoseMove)
            boards.extend(possibleBoards)

    # Pincers capture by 'surrounding' opposing pieces
    # NOTE: according to the spec, pincers can pinch using ANY friendly piece
    #       (not just other pincers)
    elif piece_type == BLACK_PINCER:
        directions = [(0,1), (1,0), (-1,0), (0,-1)]
        new_board = copy_board(board)
        new_hash = zhash
        for (drow, dcol) in directions:
            if valid_space(new_r + drow*2, new_c + dcol*2)\
               and board[new_r+drow][new_c+dcol] in capturablePieces\
               and board[new_r+drow*2][new_c+dcol*2] != EMPTY\
               and who(board[new_r+drow*2][new_c+dcol*2]) == whoseMove:
                new_board[new_r+drow][new_c+dcol] = EMPTY
                new_hash = zhash ^ ZOB_TBL[(new_r+drow, new_c+dcol, board[new_r+drow][new_c+dcol])]
        if new_hash != zhash:
            boards.append((new_board, new_hash))

    # Coordinators capture by 'coordinating' with the king
    elif piece_type == BLACK_COORDINATOR:
        (king_r, king_c) = friendly_king_position(board, whoseMove)
        # Check the two spaces that the king and coordinator 'coordinate' together
        new_board = copy_board(board)
        new_hash = zhash
        for (r,c) in [(new_r,king_c), (king_r,new_c)]:
            if board[r][c] in capturablePieces:
                new_board[r][c] = EMPTY
                new_hash = zhash ^ ZOB_TBL[(r,c,board[r][c])]
        if new_hash != zhash:
            boards.append((new_board, new_hash))

    # Withdrawers capture by 'withdrawing' from an opposing piece
    elif piece_type == BLACK_WITHDRAWER:
        # Check the space 'behind' the withdrawer
        if valid_space(old_r - dr, old_c - dc)\
           and board[old_r - dr][old_c - dc] in capturablePieces:
            new_board = copy_board(board)
            new_board[old_r - dr][old_c - dc] = EMPTY
            new_hash = zhash ^ ZOB_TBL[(old_r-dr,old_c-dc,board[old_r-dr][old_c-dc])]
            boards.append((new_board, new_hash))

    if boards == []:
        boards = [(board, zhash)]
    return boards


def apply_move(board, zhash, old_r, old_c, new_r, new_c):
    global ZOB_TBL, EMPTY
    # Returns a tuple containing the given move followed by a copy of
    # the given board with the result of a piece's (non-capturing) move
    new_hash = zhash
    new_hash ^= ZOB_TBL[(new_r, new_c, board[new_r][new_c])]
    new_hash ^= ZOB_TBL[(old_r, old_c, board[old_r][old_c])]
    new_hash ^= ZOB_TBL[(new_r, new_c, board[old_r][old_c])]
    new_board = copy_board(board)
    new_board[new_r][new_c] = new_board[old_r][old_c]
    new_board[old_r][old_c] = EMPTY
    return ((old_r, old_c),(new_r, new_c)), new_board, new_hash


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
    king = BLACK_KING ^ whoseMove
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == king:
                return row,col
    return None # Something has gone terribly wrong here


def nickname():
    return "Rookoko"

def introduce():
    return "I'm Rookoko, an exuberant Baroque Chess agent."


def prepare(player2Nickname):
    global ZOB_TBL, ZOB_STATES, PIECES, EMPTY, TURNS, STATIC_BOARD, DYN_VALS, PUN_GENERATOR
    TURNS = 0
    STATIC_BOARD = dict()
    PUN_GENERATOR = pun_generator()
    for r in range(8):
        for c in range(8):
            #print(int(pos_val(r,c)*10)/10, end=' ')
            for key in STATIC_VALUES.keys():
                if key == EMPTY:
                    STATIC_BOARD[(r,c,key)] = (r-3.5)//2
                else:
                    STATIC_BOARD[(r,c,key)] = pos_val(r,c)*(2*who(key)-1)*STATIC_VALUES[key]
        #print()

    # Set up Zobrist hashing - Assuming default board size 8 x 8
    for row in range(8):
        for col in range(8):
            # Don't bother with a hash for the empty space
            ZOB_TBL[(row, col, EMPTY)] = 0
            for player in (BLACK, WHITE):
                for piece in PIECES[player]:
                    ZOB_TBL[(row, col, piece)] = random.getrandbits(64)
                    
    return "Ready to rumble!"


# Get hash value, do bit-wise XOR
def zob_hash(board):
    global ZOB_TBL, EMPTY
    hash_val = 0
    for row in range(8):
        for col in range(8):
            if board[row][col] != EMPTY:
                hash_val ^= ZOB_TBL[(row, col, board[row][col])]
    return hash_val
