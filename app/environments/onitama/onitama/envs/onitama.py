
import gym
import numpy as np

import config

from stable_baselines import logger

from .classes import *

class OnitamaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(OnitamaEnv, self).__init__()
        self.name = 'onitama'
        self.n_players = 2
        self.manual = manual
        
        self.contents = [
            {'card': Card, 'info': {'name': '🐗 Boar', 'priority': 'r', 'moves': [(-1,0),(0,1),(1,0)]}}  #0 
            , {'card': Card, 'info': {'name': '🐲 Dragon', 'priority': 'r', 'moves': [(-2,1),(-1,-1),(1,-1),(2,1)]}}  #1
            , {'card': Card, 'info': {'name': '🦗 Mantis', 'priority': 'r', 'moves': [(-1,1),(0,-1),(1,1)]}}  #2
            , {'card': Card, 'info': {'name': '🐘 Elephant', 'priority': 'r', 'moves': [(-1,0),(-1,1),(1,0),(1,1)]}}  #3
            , {'card': Card, 'info': {'name': '🦚 Crane', 'priority': 'b', 'moves': [(-1,-1),(0,1),(1,-1)]}}  #4
            , {'card': Card, 'info': {'name': '🐵 Monkey', 'priority': 'b', 'moves': [(-1,-1),(-1,1),(1,-1),(1,1)]}}  #5
            , {'card': Card, 'info': {'name': '🐯 Tiger', 'priority': 'b', 'moves': [(0,2),(0,-1)]}}  #6
            , {'card': Card, 'info': {'name': '🦀 Crab', 'priority': 'b', 'moves': [(-2,0),(0,1),(2,0)]}}  #7
            , {'card': Card, 'info': {'name': '🦢 Goose', 'priority': 'b', 'moves': [(-1,0),(-1,1),(1,0),(1,-1)]}}  #8
            , {'card': Card, 'info': {'name': '🦈 Eel', 'priority': 'b', 'moves': [(-1,-1),(-1,1),(1,0)]}}  #9
            , {'card': Card, 'info': {'name': '🐴 Horse', 'priority': 'r', 'moves': [(-1,0),(0,1),(0,-1)]}}  #10
            , {'card': Card, 'info': {'name': '🐸 Frog', 'priority': 'r', 'moves': [(-2,0),(-1,1),(1,-1)]}}  #11
            , {'card': Card, 'info': {'name': '🐔 Rooster', 'priority': 'r', 'moves': [(-1,0),(-1,-1),(1,0),(1,1)]}}  #12
            , {'card': Card, 'info': {'name': '🐍 Cobra', 'priority': 'r', 'moves': [(-1,0),(1,1),(1,-1)]}}  #13
            , {'card': Card, 'info': {'name': '🐂 Ox', 'priority': 'b', 'moves': [(0,1),(0,-1),(1,0)]}}  #14
            , {'card': Card, 'info': {'name': '🐇 Rabbit', 'priority': 'b', 'moves': [(-1,-1),(1,1),(2,0)]}}  #15
        ]

        self.action_space = gym.spaces.Discrete( # size = 141
            # number of pieces on one team times number of board spaces to move to
            5 * 25
            # choosing which card to give up on passing turn / duplicate move
            + 16
        )
        self.observation_space = gym.spaces.Box(0, 1, ( # size = 225 + 141 = 366
            # each piece can be on any one of the 25 squares
            6 * 25 # you must know exactly where each of your pieces + the opponent master is
            + 25 # the opponent students can be grouped together on one grid
            + 16 * 3 # which cards each player has, plus what the side card is
            + 1 # indicator if the bot is making a duplicate move choice
            + 1 # indicator if they must pass their turn
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
        ret = np.zeros((5, 25))
        curr_pieces = self.current_player.active_pieces
        for i in range(5):
            piece = curr_pieces[i]
            if piece:
                if self.current_player_num == 0:
                    space_id = 5 * piece.row + piece.col
                else:
                    space_id = 5 * (4 - piece.row) + (4 - piece.col)
                ret[i][space_id] = 1
        
        curr_pieces = self.opposing_player.active_pieces
        foo = np.zeros((2,25))
        for i in range(5):
            piece = curr_pieces[i]
            if piece:
                if self.current_player_num == 0:
                    space_id = 5 * piece.row + piece.col
                else:
                    space_id = 5 * (4 - piece.row) + (4 - piece.col)
                # fill in the correct row, master on 1st, others on 2nd
                if i != 2:
                    foo[1][space_id] = 1
                else:
                    foo[0][space_id] = 1
        ret = np.append(ret.flatten(), foo.flatten())

        foo = np.zeros((3,16))
        foo[0][self.current_player.card_a.id] = 1
        foo[0][self.current_player.card_b.id] = 1
        foo[1][self.opposing_player.card_a.id] = 1
        foo[1][self.opposing_player.card_b.id] = 1
        foo[2][self.side_card.id] = 1
        ret = np.append(ret, foo.flatten())

        curr_LA = self.legal_actions
        foo = np.zeros((2,))
        if self.card_to_swap < 0:
            foo[0] = 1
        elif self.pass_turn:
            foo[1] = 1
        ret = np.append(ret, foo)
        ret = np.append(ret, curr_LA)

        return ret

    def is_valid_move_space(self, row: int, col: int) -> bool:
        """Checks if a piece of the current player can legally move to the given row and column.
        Returns a boolean value. A move is invalid if it is off the board or the space is already
        occupied by a friendly piece (for the current player)."""
        if (row < 0) or (col < 0) or (row > 4) or (col > 4):
            return False
        occupying_piece = self.board[row][col]
        if occupying_piece:
            if (occupying_piece.color == self.current_player.color):
                return False
        return True

    def set_actions_dict(self, piece: Piece, p: Player) -> None:
        """Given a piece, set its dictionary to say which legal actions it can take given
        the cards of the current player p. This dictionary's keys should be integers from 0 to 24
        giving the move from the current player's perspective, and the keys should point to a list
        containing a 0 if the move is possible for card a and a 1 for card b."""
        move_dict = {}
        foo = [p.card_a, p.card_b]
        for i, card in enumerate(foo):
            for move in card.moves:
                if (self.current_player_num == 0):
                    r = piece.row - move[1]
                    c = piece.col + move[0]
                else:
                    r = piece.row + move[1]
                    c = piece.col - move[0]
                if self.is_valid_move_space(r,c):
                    space_id = 5 * r + c
                    if space_id in move_dict:
                        move_dict[space_id].append(i)
                    else:
                        move_dict[space_id] = [i]
        piece.move_dict = move_dict


    @property
    def legal_actions(self):
        p = self.current_player
        foo = np.zeros((16,))
        # pick id's of two cards in player inventory if necessary
        if (self.card_to_swap >= 0): # basically, are we in a normal turn?
            legal_actions = np.zeros((5,25))
            i = 0 # which piece id we are looking at
            for piece in p.active_pieces:
                if piece:
                    self.set_actions_dict(piece, p)
                    for move in piece.move_dict:
                        if p.color == 'b': # flip the move number for the blue player
                            move = 24 - move
                        legal_actions[i][move] = 1
                i += 1
            legal_actions = legal_actions.flatten()
            num_moves = sum(legal_actions)
            if num_moves > 0.5:
                legal_actions = np.append(legal_actions, foo)
                return legal_actions
            else:
                self.pass_turn = True
                logger.debug("\n     --- No legal moves! Turn must be passed. ---")

        # otherwise, we must pick one of two cards to give up
        legal_actions = np.zeros((125,))
        foo[p.card_a.id] = 1
        foo[p.card_b.id] = 1
        
        legal_actions = np.append(legal_actions, foo)

        return legal_actions

    def move_piece(self, piece: Piece, row: int, col: int) -> None:
        """Moves the 'piece' object given to the target row and column, capturing
        any piece in that spot and removing it from the game. Updates the position
        of both the board object and the piece object itself."""
        # remove the target piece from the board if there is one
        target_piece = self.board[row][col]
        if target_piece:
            logger.debug(f"\n     --- Piece captured! ({target_piece.symbol}) ---")
            self.opposing_player.active_pieces[target_piece.id] = None
        
        # move the piece to the target square
        old_r, old_c = piece.row, piece.col
        self.board[row][col] = piece
        piece.move(row,col)
        self.board[old_r][old_c] = None
    
    def rotate_cards(self, card: int) -> None:
        """The variable 'card' is either a 0 or 1, denoting either card a or b is being
        used up by the current player. The cards are then swapped accordingly."""
        p = self.current_player
        if (card == 0):
            x = p.card_a
            p.card_a = self.side_card
        else:
            x = p.card_b
            p.card_b = self.side_card
        self.side_card = x
    
    def get_card(self, piece: Piece, row: int, col: int) -> int:
        """Given a target position for a certain piece and using the current player,
        find which of their two cards is the one that allows them to make this move.
        Returns 0 for card_a, 1 for card_b. If both could be used, returns -1."""
        d = piece.move_dict
        space_id = 5 * row + col
        # we assume there is a card for the given move
        if space_id in d:
            usable_cards = d[space_id]
            # if there is only one card for this move, return 0 or 1
            if len(usable_cards) == 1:
                return usable_cards[0]
            # if either card could be used, we must enter the extra choice mode
            else:
                return -1
        else:
            raise Exception(f'No card found to make the given piece go to ({row},{col})!')
    
    def is_game_over(self):
        """Checks if the current game state is a game-over situation. Always returns a tuple:
        A boolean for the variables 'done' and 'reward' in the step function."""
        pRed = self.players[0]
        master_red = pRed.active_pieces[2]
        pBlue = self.players[1]
        master_blue = pBlue.active_pieces[2]
        # did anyone win by capturing their opponent's master?
        if (master_blue == None):
            self.winner = '🟥 Red'
            return True, [1, -1]
        if (master_red == None):
            self.winner = '🟦 Blue'
            return True, [-1, 1]
        # so both player's masters are alive
        # did anyone win by moving master to opponent's Temple Arch?
        if (master_red.row == 0) and (master_red.col == 2):
            self.winner = '🟥 Red'
            return True, [1, -1]
        if (master_blue.row == 4) and (master_blue.col == 2):
            self.winner = '🟦 Blue'
            return True, [-1, 1]
        # else, game is not over yet
        return False, [0,0]


    def step(self, action):
        reward = [0, 0]
        done = False
        p = self.current_player

        # check move legality
        if self.legal_actions[action] == 0:
            reward = [1.0,1.0]
            reward[self.current_player_num] = -1
            done = True

        # now play the move
        # first see if the player just chose their card to use in an ambiguous case
        if (self.card_to_swap < 0):
            card_id = action - 125
            self.card_to_swap = 0 if (p.card_a.id == card_id) else 1
            self.just_picked_card_to_use = True

        # now try moving the piece and swapping cards if we are not passing
        if not self.pass_turn:
            if not self.just_picked_card_to_use:
                # we are in a normal turn and the player just chose a spot to move
                # however, there could be an ambiguous case; save the move regardless
                i, square = divmod(action, 25)
                if p.color == 'b':
                    square = 24 - square
                self.saved_row, self.saved_col = divmod(square, 5)
                self.saved_piece = self.current_player.active_pieces[i]
                self.card_to_swap = self.get_card(self.saved_piece, self.saved_row, self.saved_col)

            if self.card_to_swap >= 0:
                # if we know which card to use at this point, use it and make the saved move
                # this could be the one usable card or the one just chosen to use up in ambiguous case
                self.rotate_cards(self.card_to_swap)
                self.move_piece(self.saved_piece, self.saved_row, self.saved_col)
                # check if game is over
                done, reward = self.is_game_over()

                self.just_picked_card_to_use = False
                self.current_player_num = (self.current_player_num + 1) % 2
                self.turns_taken += 1
            else:
                logger.debug('\n     --- Ambiguous move made! A card must be chosen to give up. ---')

        # otherwise a pass turn must be played
        else:
            card_id = action - 125
            self.card_to_swap = 0 if (p.card_a.id == card_id) else 1

            self.rotate_cards(self.card_to_swap)
            # we rotate the cards and pass the game to the next player
            # we do not need to check for a game over
            self.just_picked_card_to_use = False
            self.current_player_num = (self.current_player_num + 1) % 2
            self.turns_taken += 1
            self.pass_turn = False

        self.done = done

        return self.observation, reward, done, {}

    def reset(self):
        self.deck = Deck(self.contents)
        self.players = []
        self.board = []
        for i in range(5):
            row = [None] * 5
            self.board.append(row)

        for color in ('r','b'):
            self.players.append(Player(color))

        for i in range(2): # p1 = Red, p2 = Blue
            pieceID = 0
            p = self.players[i]
            if self.manual:
                # card 1
                card_name = input(f"Enter Player {i+1}'s 1st card: ")
                c = self.deck.pick(card_name)
                while not c:
                    logger.debug(f"Oops! Card not found with name '{card_name}'")
                    card_name = input(f"Enter Player {i+1}'s 1st card: ")
                    c = self.deck.pick(card_name)
                p.card_a = c
                # card 2
                card_name = input(f"Enter Player {i+1}'s 2nd card: ")
                c = self.deck.pick(card_name)
                while not c:
                    logger.debug(f"Oops! Card not found with name '{card_name}'")
                    card_name = input(f"Enter Player {i+1}'s 2nd card: ")
                    c = self.deck.pick(card_name)
                p.card_b = c
            else:
                p.card_a = self.deck.draw()
                p.card_b = self.deck.draw()
            if i == 0:
                for col in range(5):
                    piece = Piece(pieceID, 'r', 4, col)
                    p.active_pieces.append(piece)
                    self.board[4][col] = piece
                    pieceID += 1
            else:
                for col in range(4,-1,-1):
                    piece = Piece(pieceID, 'b', 0, col)
                    p.active_pieces.append(piece)
                    self.board[0][col] = piece
                    pieceID += 1
        
        if self.manual:
            # side card
            card_name = input(f"Enter side card name: ")
            c = self.deck.pick(card_name)
            while not c:
                logger.debug(f"Oops! Card not found with name '{card_name}'")
                card_name = input(f"Enter side card name: ")
                c = self.deck.pick(card_name)
            self.side_card = c
        else:
            self.side_card = self.deck.draw()
        self.current_player_num = 0 if (self.side_card.priority == 'r') else 1
        self.pass_turn = False
        self.just_picked_card_to_use = False
        self.card_to_swap = 0
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
        else:
            logger.debug(f'\n\n-------FINAL POSITION-----------')
            

        p = self.players[1]
        logger.debug(f'\n🟦 -- Blue Player Cards: \n{p.card_a.symbol}\n{p.card_b.symbol}')
        if self.current_player_num == 1:
            logger.debug(f'\n⏳ Next for Blue: {self.side_card.symbol}')

        horiz_line = '-' * 36
        blank_space_row = '|   ' + ('   |   ' * 4) + '   |'

        logger.debug('\n' + horiz_line)
        for row in range(5):
            logger.debug(blank_space_row)
            rlist = []
            for col in range(5):
                piece = self.board[row][col]
                if piece:
                    rlist.append(piece.symbol)
                else:
                    rlist.append('------')
            logger.debug('|' + '|'.join(rlist) + '|')
            logger.debug(blank_space_row)
            logger.debug(horiz_line)

        if self.current_player_num == 0:
            logger.debug(f'\n⏳ Next for Red: {self.side_card.symbol}')

        p = self.players[0]
        logger.debug(f'\n🟥 -- Red Player Cards: \n{p.card_a.symbol}\n{p.card_b.symbol}')

        if self.verbose:
            logger.debug(f'\nObservation: \n{[i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

        if self.done:
            logger.debug(f'\n\nGAME OVER')
            logger.debug(f'👑 - {self.winner} is victorious! - 👑')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Geschenkt!')
