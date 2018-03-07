'''PlayerSkeletonB.py
The beginnings of an agent that might someday play Baroque Chess.

'''

from BC_state_etc import *
import random

# Table mapping players to a set containing their pieces
PIECES = {BLACK:set((BLACK_PINCER, BLACK_COORDINATOR, BLACK_LEAPER, BLACK_IMITATOR,\
                 BLACK_WITHDRAWER, BLACK_KING, BLACK_FREEZER)),\
          WHITE:set((WHITE_PINCER, WHITE_COORDINATOR, WHITE_LEAPER, WHITE_IMITATOR,\
                 WHITE_WITHDRAWER, WHITE_KING, WHITE_FREEZER))}
EMPTY = 0
LEAPER_CAPTURE = None

def makeMove(currentState, currentRemark, timelimit):

    # Compute the new state for a move.
    # This is a placeholder that just copies the current state.
    newState = BC_state(currentState.board)

    # Fix up whose turn it will be.
    newState.whose_move = 1 - currentState.whose_move
    possible_boards = generate_successors()
    new_move, new_board = random.choice(possible_boards)
    newState.board = new_board

    # Make up a new remark
    newRemark = "I'm not very good at this game yet. I am not moving, which isn't legal, but... whatever."
    return [[new_move, newState], newRemark]

def nickname():
    return "Cheetie"

def introduce():
    return "I'm Cheetie Playah.  I haven't learned the rules of Baroque Chess yet."

def prepare(player2Nickname):
    pass


# Generates successors from input board by finding all possible moves
def generate_successors(board, whoseMove):
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
                    successors = [apply_move(board, row, col, new_r, new_c)]
                    break
            
        # Pincers and kings have special movement rules.
        # All other pieces move like standard-chess queens.
        elif piece == BLACK_KING or piece == WHITE_KING:
            for (new_r,new_c) in neighborhood:
                if board[new_r][new_c] == EMPTY or board[new_r][new_c] in opponentPieces:
                    successors.append(apply_move(board, row, col, new_r, new_c))
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
                if valid_space(new_r + dr, new_c + dc) and\
                   board[new_r + dr][new_c + dc] == EMPTY:
                    target = board[new_r][new_c]
                    if target in opponentPieces and\
                       (piece == BLACK_LEAPER or\
                        piece == WHITE_LEAPER or\
                        (piece == BLACK_IMITATOR and target == WHITE_LEAPER) or\
                        (piece == WHITE_IMITATOR and target == BLACK_LEAPER)):
                        LEAPER_CAPTURE.append((new_r + dr, new_c + dc))
                        possible_spaces.append((new_r + dr, new_c + dc))
        
            for (new_r, new_c) in possible_spaces:
                # Apply move to board
                new_move, new_board = apply_move(board, zhash, row, col, new_r, new_c)
                # Apply any captures to board
                new_boards = apply_captures(new_board, row, col, new_r, new_c,\
                                            piece, opponentPieces, whoseMove)
                successors.extend(((new_move, b) for b in new_boards))
    return successors


def valid_space(row, col):
    # Returns whether the given coordinates fall within the boundaries of the board
    return (0 <= row < 8) and (0 <= col < 8)


def apply_captures(board, old_r, old_c, new_r, new_c, piece, capturablePieces, whoseMove):
    global LEAPER_CAPTURE, EMPTY
    # Looks for all possible captures, and then applies them, returning a list of new board states
    
    # Fast and mysterious way to make dr and dc either 1, 0, or -1
    (dr, dc) = ((old_r > new_r) - (old_r < new_r),\
                (old_c > new_c) - (old_c < new_c))
    # Fast and mysterious way to get the piece 'type', in terms of its black-piece equivalent
    piece_type = (piece >> 1) << 1

    # Leapers capture by 'leaping over' opposing pieces
    # Leaper captures must be handled specially, because moving without capture is not acceptable.
    # Note that this will also handle the case of imitators imitating leapers
    if ((old_r, old_c), (new_r, new_c)) in LEAPER_CAPTURE:
        # The space 'behind' the leaper's final position will already have been checked above
        # if board[new_r - dr][new_c - dc] in capturablePieces:
        LEAPER_CAPTURE.remove((old_r, old_c), (new_r, new_c))
        new_board = copy_board(board)
        new_board[new_r - dr][new_c - dc] = EMPTY
        return [(new_board,)]

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
            possibleBoards = apply_captures(board, old_r, old_c, new_r, new_c,\
                                         otherPiece, [otherPiece ^ 1], whoseMove)
            boards.extend(possibleBoards)

    # Pincers capture by 'surrounding' opposing pieces
    # NOTE: according to the spec, pincers can pinch using ANY friendly piece
    #       (not just other pincers)
    elif piece_type == BLACK_PINCER:
        directions = [(0,1), (1,0), (-1,0), (0,-1)]
        new_board = copy_board(board)
        for (drow, dcol) in directions:
            if valid_space(new_r + drow*2, new_c + dcol*2)\
               and board[new_r+drow][new_c+dcol] in capturablePieces\
               and board[new_r+drow*2][new_c+dcol*2] != EMPTY\
               and who(board[new_r+drow*2][new_c+dcol*2]) == whoseMove:
                new_board[new_r+drow][new_c+dcol] = EMPTY
        boards.append((new_board,))

    # Coordinators capture by 'coordinating' with the king
    elif piece_type == BLACK_COORDINATOR:
        (king_r, king_c) = friendly_king_position(board, whoseMove)
        # Check the two spaces that the king and coordinator 'coordinate' together
        new_board = copy_board(board)
        for (r,c) in [(new_r,king_c), (king_r,new_c)]:
            if board[r][c] in capturablePieces:
                new_board[r][c] = EMPTY
        boards.append((new_board,))

    # Withdrawers capture by 'withdrawing' from an opposing piece
    elif piece_type == BLACK_WITHDRAWER:
        # Check the space 'behind' the withdrawer
        if valid_space(old_r - dr, old_c - dc)\
           and board[old_r - dr][old_c - dc] in capturablePieces:
            new_board = copy_board(board)
            new_board[old_r - dr][old_c - dc] = EMPTY
            boards.append((new_board, new_hash))

    if boards == []:
        boards = [(board,)]
    return boards


def apply_move(board, old_r, old_c, new_r, new_c):
    global EMPTY
    # Returns a tuple containing the given move followed by a copy of
    # the given board with the result of a piece's (non-capturing) move
    new_board = copy_board(board)
    new_board[new_r][new_c] = new_board[old_r][old_c]
    new_board[old_r][old_c] = EMPTY
    return ((old_r, old_c),(new_r, new_c)), new_board


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
