import gym
import numpy as np

import config

from stable_baselines import logger

class Piece():
    def __init__(self, id: int, color: int, shape: int, height: int, density: int) -> None:
        self.id = id
        self.color = color
        self.shape = shape
        self.height = height
        self.density = density
        self.set_symbol()

    # 0, 1 meaning
    # Color: Light, Dark
    # Shape: Round, Square
    # Height: Tall, Short
    # Density: Solid, Hollow
    def set_symbol(self):
        if self.height:
            foo = '🔽⭕' if self.density else '🔽🔴'
        else:
            foo = '🔼⭕' if self.density else '🔼🔴'

        if self.color:
            sym = '⬛' if self.shape else '⚫'
        else:
            sym = '⬜' if self.shape else '⚪'

        self.symbol = foo + sym
        


class QuartoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(QuartoEnv, self).__init__()
        self.name = 'quarto'
        self.n_players = 2
        self.manual = manual

        self.get_spaces_from_ID = [
            [0,1,2,3], # rows, top to bottom
            [4,5,6,7],
            [8,9,10,11],
            [12,13,14,15],
            [0,4,8,12], # cols, left to right
            [1,5,9,13],
            [2,6,10,14],
            [3,7,11,15],
            [0,5,10,15], # diagonal TL to BR
            [12,9,6,3]  # diagonal BL to TR
        ]

        self.action_space = gym.spaces.Discrete( # size = 32
            # choosing which square to place the piece on
            16
            # number of pieces that you can make your opponent play next
            + 16
        )
        self.observation_space = gym.spaces.Box(0, 1, ( # size = 288 + 32 = 320
            16 * 16 # each piece can be on any one of the 16 squares
            + 16 # seeing which pieces are still on the side of the board
            + 16 # if we are placing a piece, which one must we place down next?
            + self.action_space.n  # legal_actions
            , )
        )  
        self.verbose = verbose

    @property
    def legal_actions(self):
        legal_actions = np.zeros((32,))
        if not self.choosing_piece_for_opponent: # a piece must be placed on an empty space
            for row in range(4):
                for col in range(4):
                    if not (self.board[row][col]):
                        legal_actions[(row * 4) + col] = 1
        else: # a piece is being chosen for the opponent
            for piece in self.side_pieces:
                if piece:
                    legal_actions[16 + piece.id] = 1
        return legal_actions

    @property
    def observation(self):
        # State of the board
        ret = np.zeros((16, 16))
        for space in range(16):
            row, col = divmod(space, 4)
            pc = self.board[row][col]
            if pc:
                ret[space][pc.id] = 1
        
        foo = np.zeros((16,))
        for pc in self.side_pieces:
            if pc:
                foo[pc.id] = 1
        ret = np.append(ret.flatten(), foo)

        foo = np.zeros((16,))
        if not self.choosing_piece_for_opponent:
            foo[self.pieceID_chosen] = 1
        ret = np.append(ret, foo)

        return np.append(ret, self.legal_actions)
    
    def is_filled(self, spaceIDs: list):
        """Checks if the given spaces on the board is filled and returns the boolean answer.
        0 to 3: rows, 4-7: cols, 8 and 9: diagonals, TL to BR then BL to TR"""
        for i in spaceIDs:
            row, col = divmod(i, 4)
            if not (self.board[row][col]):
                return False
        return True
    
    def is_winning_line(self, spaceIDs: list):
        """Checks if the 4 given spaces on the current board are a winning line. That is, 
        the four pieces on the spots given all share a common attribute. This function
        assumes that the four spots given each have a piece on them. Returns a boolean."""
        pieces = []
        for i in spaceIDs:
            row, col = divmod(i, 4)
            pieces.append(self.board[row][col])
        # are all 4 pieces the same color?
        foo = [p.color for p in pieces]
        if (all(foo)) or (not any(foo)):
            return True
        # are all 4 pieces the same shape?
        foo = [p.shape for p in pieces]
        if (all(foo)) or (not any(foo)):
            return True
        # are all 4 pieces the same height?
        foo = [p.height for p in pieces]
        if (all(foo)) or (not any(foo)):
            return True
        # are all 4 pieces the same density?
        foo = [p.density for p in pieces]
        if (all(foo)) or (not any(foo)):
            return True
        # not a winning line...
        return False

    def is_game_over(self):
        """Checks if the current game state is a game-over situation. Always returns a tuple:
        A boolean for the variables 'done' and 'reward' in the step function."""
        lines_to_check = set(range(10)) - self.lines_checked
        for lineID in lines_to_check:
            spaceIDs = self.get_spaces_from_ID[lineID]
            if self.is_filled(spaceIDs):
                if self.is_winning_line(spaceIDs):
                    reward = [1,-1] if (self.current_player_num == 0) else [-1,1]
                    return True, reward
                else:
                    self.lines_checked.add(lineID)
                    
        reward = [-0.01, 0.01] if (self.current_player_num == 0) else [0.01, -0.01]
        if len(self.lines_checked) == 10: # check for a drawn game
            return True, reward
        # game not over yet
        return False, reward

    def step(self, action):
        reward = [0,0]
        done = False

        # check move legality
        if self.legal_actions[action] == 0:
            reward = [1.0,1.0]
            reward[self.current_player_num] = -1
            done = True

        # now play the action
        if self.choosing_piece_for_opponent:
            self.pieceID_chosen = action - 16
            self.choosing_piece_for_opponent = False
            self.turns_taken += 1
            self.current_player_num = (self.current_player_num + 1) % 2

        else: # the current player has chosen to play a piece
            piece = self.side_pieces[self.pieceID_chosen]
            row, col = divmod(action, 4)
            self.board[row][col] = piece
            self.side_pieces[piece.id] = None
            
            done, reward = self.is_game_over()
            if not done:
                self.choosing_piece_for_opponent = True

        self.done = done

        return self.observation, reward, done, {}

    def reset(self):
        self.board = []
        self.side_pieces = []
        for i in range(4):
            row = [None] * 4
            self.board.append(row)

        pieceID = 0
        for color in range(2):
            for shape in range(2):
                for height in range(2):
                    for density in range(2):
                        p = Piece(id=pieceID,color=color,shape=shape,height=height,density=density)
                        self.side_pieces.append(p)
                        pieceID += 1
        # 0, 1 meaning
        # Color: Light, Dark
        # Shape: Round, Square
        # Height: Tall, Short
        # Density: Solid, Hollow
        
        self.current_player_num = 0
        self.choosing_piece_for_opponent = True
        self.pieceID_chosen = 0
        self.lines_checked = set()

        self.turns_taken = 0
        self.done = False

        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation


    def render(self, mode='human', close=False):
        if close:
            return

        if not self.done:
            logger.debug(f'\n\n-------TURN {self.turns_taken + 1}-----------')
            logger.debug(f"It is Player {self.current_player_num + 1}'s turn")
            if not self.choosing_piece_for_opponent:
                logger.debug(f'\nPiece to Place: {self.side_pieces[self.pieceID_chosen].symbol}')
            else:
                logger.debug("\nA piece must be chosen for the opponent to place next.")
        else:
            logger.debug(f'\n\n-------FINAL POSITION-----------')
            
        
        piece_symbols = [(pc.symbol + str(pc.id+16)) for pc in self.side_pieces if pc]
        logger.debug(f"\nPieces Remaining: {' '.join(piece_symbols)}")

        horiz_line = '-' * 38
        blank_space_row = '|    ' + ('    |    ' * 3) + '    |'

        logger.debug('\n' + horiz_line)
        for row in range(4):
            logger.debug(blank_space_row)
            rlist = []
            for col in range(4):
                piece = self.board[row][col]
                if piece:
                    rlist.append(piece.symbol)
                else:
                    rlist.append('--------')
            logger.debug('|' + '|'.join(rlist) + '|')
            logger.debug(blank_space_row)
            logger.debug(horiz_line)

        if self.verbose:
            logger.debug(f'\nObservation: \n{[i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

        if self.done:
            logger.debug(f'\n\nGAME OVER')
            logger.debug(f'👑 - Player {self.current_player_num + 1} is victorious! - 👑')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Quarto!')
