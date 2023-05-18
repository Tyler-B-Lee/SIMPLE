import math
import gym
import numpy as np

import config

from stable_baselines import logger

from .classes import *

class ShobuEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(ShobuEnv, self).__init__()
        self.name = 'shobu'
        self.n_players = 2
        self.manual = manual

        # all possible moves for a piece (direction, distance)
        # 0 = up, towards the opponent's side, continues clockwise through 7
        self.moves = {(x,y) for x in range(8) for y in (1,2)}
        # translates a movement direction into the (x,y) coordinate change for 1 move in that direction
        self.direction_list = [
            (0,1), # 0
            (1,1), # 1
            (1,0), # 2
            (1,-1), # 3
            (0,-1), # 4
            (-1,-1), # 5
            (-1,0), # 6
            (-1,1) # 7
        ]

        self.action_space = gym.spaces.Discrete( # size = 144
            # passive: number of pieces on home boards times number of distinct moves each piece can make
            8 * 16
            # aggressive: choosing which stone to move given what passive direction was just chosen
            + 16
        )
        self.observation_space = gym.spaces.Box(0, 1, ( # size = 336 + 144 = 480, so about 3.122 x 10^144 distinct inputs!
            16 * 16 # location of each of the player's pieces within their own boards
            + 4 * 16 # location of enemy pieces can be simplified when looking at each board
            + 16 # if choosing an aggressive move, tell which move ID was just picked for their passive
            + self.action_space.n  # legal_actions
            , )
        )  
        self.verbose = verbose

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_num]
    
    @property
    def opposing_player(self) -> Player:
        i = (self.current_player_num + 1) % 2
        return self.players[i]
        
    @property
    def observation(self):
        # Locations of the current player's pieces (id 0 to 4)
        ret = np.zeros((16, 16))
        curr_pieces = self.current_player.active_pieces
        #rng = range(4) if (self.current_player_num == 0) else range(3,-1,-1)
        for group in curr_pieces:
            for piece in group:
                if piece:
                    if self.current_player_num == 0:
                        space_id = 4 * piece.row + piece.col
                    else:
                        space_id = 4 * (3 - piece.row) + (3 - piece.col)
                    ret[piece.id][space_id] = 1
        
        curr_pieces = self.opposing_player.active_pieces
        foo = np.zeros((4,16))
        for group in curr_pieces:
            for piece in group:
                if piece:
                    if self.current_player_num == 0:
                        space_id = 4 * piece.row + piece.col
                    else:
                        space_id = 4 * (3 - piece.row) + (3 - piece.col)
                    board_id = piece.id // 4
                    foo[board_id][space_id] = 1
        ret = np.append(ret.flatten(), foo.flatten())

        foo = np.zeros((16,))
        if not self.passive_move_next:
            foo[self.move_num_chosen] = 1
        ret = np.append(ret, foo)
        ret = np.append(ret, self.legal_actions)

        return ret

    def is_off_board(self, row: int, col: int):
        "Returns True if the given row and column IS OFF of one of the 4x4 game boards, False if it IS ON a board."
        if (row < 0) or (col < 0) or (row > 3) or (col > 3):
            return True
        return False

    def is_valid_aggressive(self, pc: Piece, move: tuple) -> bool:
        """Returns True if the given piece 'pc' can make the given move from its current position as
        a legal AGGRESSIVE move and False otherwise. Rotates moves for the white player.
        Moves are a tuple: ( direction ID (0 to 7), distance (1 or 2) )"""
        board = self.boards[ pc.board_i ]
        dist = move[1]
        # we must move the opposite direction for the white player
        move_id = move[0] if (self.current_player.color == 'b') else (move[0] + 4) % 8
        direction = self.direction_list[ move_id ] # returns tuple for relative movement (x,y)
        # Current location -> Space A -> Space B -> Space C
        A_row, A_col = pc.row - direction[1], pc.col + direction[0]
        if self.is_off_board(A_row, A_col):
            return False
        if dist == 1:
            A = board[A_row][A_col]
            if not A:
                return True
            if (A.color == self.current_player.color):
                return False
            B_row, B_col = A_row - direction[1], A_col + direction[0]
            if self.is_off_board(B_row, B_col):
                return True
            return (not board[B_row][B_col])

        # dist == 2 here
        B_row, B_col = A_row - direction[1], A_col + direction[0]
        if self.is_off_board(B_row, B_col):
            return False
        A,B = board[A_row][A_col], board[B_row][B_col]
        if (not A) and (not B):
            return True # both A and B have no stones, so no pushing happens
        # there is a stone on A or B (or both)
        C_row, C_col = B_row - direction[1], B_col + direction[0]
        if A:
            if (A.color == self.current_player.color):
                return False
            # enemy stone on A: legal only if B is empty and C is either off board or empty
            return ((not B) and (self.is_off_board(C_row,C_col) or (not board[C_row][C_col])))
        else: # must be a stone on B and not on A
            if (B.color == self.current_player.color):
                return False
            return (self.is_off_board(C_row,C_col) or (not board[C_row][C_col]))


    def find_legal_passive_directions(self):
        """Finds and saves two sets in this object: one for aggressive directions legal on the current player's left side
        board and one for ones on their right side board. Note that the aggressive legal on one side = passive legal on other.
        The sets will contain integers tuples, representing each possible move a piece can make (direction ID, distance)."""
        p = self.current_player
        self.left_side_aggressive_moves = set()
        self.right_side_aggressive_moves = set()
        # first, find the passive moves on the right side for this player - need aggressive directions on left
        left_boards = (0,2) if (p.color == 'b') else (3,1) # which boards are on the left
        right_boards = (3,1) if (p.color == 'b') else (0,2) # which boards are on the left
        left_pieces = p.active_pieces[ left_boards[0] ] + p.active_pieces[ left_boards[1] ]
        right_pieces = p.active_pieces[ right_boards[0] ] + p.active_pieces[ right_boards[1] ]
        # find aggresive directions possible on left
        for pc in left_pieces:
            if pc:
                moves_to_check = self.moves - self.left_side_aggressive_moves
                for move in moves_to_check:
                    if self.is_valid_aggressive(pc, move):
                        self.left_side_aggressive_moves.add(move)
        # next, for the right side
        for pc in right_pieces:
            if pc:
                moves_to_check = self.moves - self.right_side_aggressive_moves
                for move in moves_to_check:
                    if self.is_valid_aggressive(pc, move):
                        self.right_side_aggressive_moves.add(move)

    def is_valid_passive(self, pc: Piece, move: tuple) -> bool:
        """Returns True if the given piece 'pc' can make the given move from its current position as
        a legal PASSIVE move and False otherwise. Rotates moves for the white player. Assumes an aggressive
        mirroring move can be made. Moves are a tuple: ( direction ID (0 to 7), distance (1 or 2) )"""
        board = self.boards[ pc.board_i ]
        dist = move[1]
        # we must move the opposite direction for the white player
        move_id = move[0] if (self.current_player.color == 'b') else (move[0] + 4) % 8
        direction = self.direction_list[ move_id ] # returns tuple for relative movement (x,y)
        # Current location -> Space A -> Space B -> Space C
        A_row, A_col = pc.row - direction[1], pc.col + direction[0]
        if self.is_off_board(A_row, A_col):
            return False
        if dist == 1:
            return (not board[A_row][A_col])
        # dist == 2 here
        B_row, B_col = A_row - direction[1], A_col + direction[0]
        if self.is_off_board(B_row, B_col):
            return False
        A,B = board[A_row][A_col], board[B_row][B_col]
        return ( (not A) and (not B) )

    @property
    def legal_actions(self):
        p = self.current_player
        pmoves = np.zeros((8,16))
        amoves = np.zeros((16,))
        if self.passive_move_next: # do we need to calculate ahead for the player to pick a passive move
            home_board_IDs = (0,1) if (p.color == 'b') else (3,2)
            self.find_legal_passive_directions()
            for piece in (p.active_pieces[ home_board_IDs[0] ]): # passives on LEFT side of home board possible
                if piece:
                    for move in self.right_side_aggressive_moves:
                        if self.is_valid_passive(piece, move):
                            pmoves[piece.id][ move[0] * 2 + move[1] - 1 ] = 1
            for piece in (p.active_pieces[ home_board_IDs[1] ]): # passives on RIGHT side of home board possible
                if piece:
                    for move in self.left_side_aggressive_moves:
                        if self.is_valid_passive(piece, move):
                            pmoves[piece.id][ move[0] * 2 + move[1] - 1 ] = 1

        else: # otherwise, we must show which pieces can make an aggressive move
            if (self.passive_side == 'left'):
                opposite_boards = (1,3) if (p.color == 'b') else (2,0)
            else:
                opposite_boards = (0,2) if (p.color == 'b') else (3,1)

            # find aggressive moves on the appropriate boards, opposite passive side
            for piece in (p.active_pieces[ opposite_boards[0] ]):
                if piece and self.is_valid_aggressive(piece, self.passive_move_made):
                    amoves[piece.id] = 1
            for piece in (p.active_pieces[ opposite_boards[1] ]):
                if piece and self.is_valid_aggressive(piece, self.passive_move_made):
                    amoves[piece.id] = 1
        
        legal_actions = np.append(pmoves.flatten(), amoves)
        return legal_actions

    def move_piece(self, pc: Piece, target_row: int, target_col: int):
        """Updates the row/col stored in the piece and the pointers of where the piece ends up, or removes
        it from the game when the target location is off the board."""
        board = self.boards[ pc.board_i ]
        board[pc.row][pc.col] = None # remove piece from board
        if self.is_off_board(target_row, target_col):
            # we must kill this piece, which must have belonged to the opposing player
            self.opposing_player.active_pieces[pc.board_i][pc.id % 4] = None
            logger.debug(f"\n     --- Piece eliminated from board {pc.board_i}! ({pc.symbol})     ---")
            return
        # else, move the piece to the target square
        board[target_row][target_col] = pc
        pc.move(target_row,target_col)

    def do_aggressive_move(self, pc: Piece, move: tuple) -> None:
        """Aggressively moves the 'piece' object according to the given move tuple (direction, distance). 
        Updates the position of both the board object and the piece object itself, along with any stones
        that were pushed around."""
        board = self.boards[ pc.board_i ]
        dist = move[1]
        # we must move the opposite direction for the white player
        move_id = move[0] if (self.current_player.color == 'b') else (move[0] + 4) % 8
        direction = self.direction_list[ move_id ] # returns tuple for relative movement (x,y)
        # Current location -> Space A -> Space B -> Space C
        A_row, A_col = pc.row - direction[1], pc.col + direction[0]
        if dist == 1:
            A = board[A_row][A_col]
            # is the spot to move to empty?
            if not A:
                self.move_piece(pc, A_row, A_col)
                return
            # we can assume that there is an enemy stone at A
            B_row, B_col = A_row - direction[1], A_col + direction[0]
            self.move_piece(A, B_row, B_col)
            self.move_piece(pc, A_row, A_col)
            return

        # dist == 2 here
        B_row, B_col = A_row - direction[1], A_col + direction[0]
        A,B = board[A_row][A_col], board[B_row][B_col]
        if (not A) and (not B):
            # both A and B have no stones, so no pushing happens
            self.move_piece(pc, B_row, B_col)
            return
        # there is a stone on A or B (or both)
        C_row, C_col = B_row - direction[1], B_col + direction[0]
        if A:
            # enemy stone on A: push them to C
            self.move_piece(A, C_row, C_col)
            self.move_piece(pc, B_row, B_col)
            return
        else: # must be a stone on B and not on A: push them to C
            self.move_piece(B, C_row, C_col)
            self.move_piece(pc, B_row, B_col)
            return

    def display_board(self):
        horiz_line = ('-' * 29) + '      ' + ('-' * 29)
        thick_line = '=' * 75
        full_blank_row = 'I   ' + ('   I   ' * 3) + '   I      |   ' + ('   |   ' * 3) + '   |'

        for i,j in ((2,3),(0,1)):
            b1, b2 = self.boards[i], self.boards[j]
            logger.debug('\n\t' + horiz_line)
            for row in range(4):
                logger.debug('\t' + full_blank_row)
                rlist1 = []
                rlist2 = []
                for col in range(4):
                    piece = b1[row][col]
                    if piece:
                        rlist1.append(piece.symbol)
                    else:
                        rlist1.append('------')
                    piece = b2[row][col]
                    if piece:
                        rlist2.append(piece.symbol)
                    else:
                        rlist2.append('------')
                logger.debug('\tI' + 'I'.join(rlist1) + 'I      |' + '|'.join(rlist2) + '|')
                logger.debug('\t' + full_blank_row)
                logger.debug('\t' + horiz_line)
            if i == 2:
                logger.debug('\n' + thick_line)

    def error_catch(self):
        logger.debug(f"Current Player: {self.current_player.symbol}")
        logger.debug(f"Picking Passive Turn: {self.passive_move_next}")
        if not self.passive_move_next:
            logger.debug(f"Passive move made last: {self.passive_move_made}")
        self.display_board()

    def do_passive_move(self, pc: Piece, move: tuple) -> None:
        """Passively moves the 'piece' object according to the given move tuple (direction, distance). 
        Updates the position of both the board object and the piece object itself."""
        if not pc:
            self.error_catch()
        dist = move[1]
        # we must move the opposite direction for the white player
        move_id = move[0] if (self.current_player.color == 'b') else (move[0] + 4) % 8
        direction = self.direction_list[ move_id ] # returns tuple for relative movement (x,y)
        # Current location -> Space A -> Space B -> Space C
        A_row, A_col = pc.row - direction[1], pc.col + direction[0]
        if dist == 1:
            self.move_piece(pc, A_row, A_col)
            return
        # dist == 2 here
        B_row, B_col = A_row - direction[1], A_col + direction[0]
        self.move_piece(pc, B_row, B_col)
        
    
    def is_game_over(self):
        """Checks if the current game state is a game-over situation. Always returns a tuple:
        A boolean for the variables 'done' and 'reward' in the step function."""
        pBlack = self.players[0]
        pWhite = self.players[1]
        # did anyone just push all of their opponent's stones off one of the boards?
        for board in pWhite.active_pieces:
            if (not any(board)):
                self.winner = '⬛ Black'
                return True, [1, -1]
        for board in pBlack.active_pieces:
            if (not any(board)):
                self.winner = '⬜ White'
                return True, [-1, 1]
        # else, game is not over yet
        return False, [0,0]

    def cutoff_game(self):
        """Since the turn limit was reached, end the game and calculate the final rewards for the players."""
        bonuses_list = [0, 3.5, 2.5, 1.25, 1]
        bscore = wscore = 0
        for boardID in range(4):
            bpieces = self.players[0].active_pieces[boardID]
            wpieces = self.players[1].active_pieces[boardID]
            bcount = wcount = 0
            for i in range(4):
                if bpieces[i]:
                    bcount += 1
                if wpieces[i]:
                    wcount += 1
            bonus = bonuses_list[ min(bcount,wcount) ]
            pts = abs(bcount - wcount) * 0.02 * bonus
            if bcount < wcount:
                pts *= -1
            bscore += pts
            wscore -= pts
        if abs(bscore - wscore) < 0.01:
            score = [0,0]
        elif bscore > wscore:
            score = [0.4,-0.4]
        else:
            score = [-0.4,0.4]
        return True, score


    def step(self, action):
        reward = [0, 0]
        done = False
        p = self.current_player

        # check move legality
        # if self.legal_actions[action] == 0:
        #     reward = [1.0,1.0]
        #     reward[self.current_player_num] = -1
        #     done = True

        # now play the move
        if self.passive_move_next:
            # the player just submitted their passive move, action is from 0 to 127
            pieceID, self.move_num_chosen = divmod(action, 16)
            boardID, i = divmod(pieceID, 4)
            self.passive_side = 'left' if (boardID == 0) else 'right'
            if p.color == 'w': # we must adjust this here because the move IDs follow piece IDs, not board IDs
                boardID = 3 - boardID
            piece_to_move = p.active_pieces[boardID][i]
            self.passive_move_made = (self.move_num_chosen // 2, (self.move_num_chosen % 2) + 1)
            self.do_passive_move(piece_to_move, self.passive_move_made)

            self.passive_move_next = False            

        else:
            # play the aggressive move given and check for game over, action from 128-143
            pieceID = action - 128
            boardID, i = divmod(pieceID, 4)
            if p.color == 'w': # moves follow the piece ID, not the board ID they are on (reversed for white)
                boardID = 3 - boardID
            piece_to_move = p.active_pieces[boardID][i]
            self.do_aggressive_move(piece_to_move, self.passive_move_made)

            self.turns_taken += 1
            # for stability (?) check if the game has lasted too long
            if self.turns_taken >= 110:
                # done, reward = self.cutoff_game()
                done, reward = True, [0,0]
            # else, check if game is over
            else:
                done, reward = self.is_game_over()

            self.current_player_num = (self.current_player_num + 1) % 2
            self.passive_move_next = True

        self.done = done

        return self.observation, reward, done, {}

    def reset(self):
        self.players = []
        self.boards = []
        colors = ('b','w')
        for player_color in colors:
            p = Player(player_color)
            self.players.append(p)

        for b in range(4):
            foo = []
            for i in range(4):
                row = [None] * 4
                foo.append(row)
            self.boards.append(foo)

        for player_i in range(2):
            plr = self.players[player_i]
            if (player_i == 0):
                rng = range(4)
                row = 3
            else:
                rng = range(3,-1,-1)
                row = 0
            pieceID = 0
            for board_index in rng:
                for col in rng: # player id 0 = Black, id 1 = White
                    piece = Piece(pieceID, plr.color, row, col, board_index)
                    plr.active_pieces[board_index].append(piece)
                    brd = self.boards[board_index]
                    brd[row][col] = piece
                    pieceID += 1
        
        self.current_player_num = 0
        self.passive_move_next = True
        self.winner = 'Nobody?'

        self.turns_taken = 0
        self.done = False

        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation


    def render(self, mode='human', close=False):
        
        if close:
            return

        if not self.done:
            logger.debug(f'\n\n-------TURN {self.turns_taken + 1}-----------')
            logger.debug(f"It is the {self.current_player.symbol} Player's turn")
            if (self.passive_move_next):
                logger.debug(f"☮️  -- Must Choose a PASSIVE Move --  ☮️")
            else:
                logger.debug(f"💢  -- Must Choose an AGGRESSIVE Move --  💢")
                logger.debug(f"- Needed: {'Left' if (self.passive_side == 'right') else 'Right'} Side with move {self.passive_move_made}")
        else:
            logger.debug(f'\n\n-------FINAL POSITION-----------')
            
        self.display_board()
        
        if self.verbose:
            logger.debug(f'\nObservation: \n{[i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

        if self.done:
            logger.debug(f'\n\nGAME OVER')
            logger.debug(f'👑 - {self.winner} is victorious! - 👑')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Shobu!')
