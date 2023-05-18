import numpy as np
import string
from stable_baselines import logger

digs = string.digits + string.ascii_letters

def int2base(x: int, base: int, length: int):
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    if sign < 0:
        digits.append('-')

    while len(digits)<length: digits.extend(["0"])
    
    return list(map(lambda x: int(x),digits))


class Board():

    def __init__(self):
        self.size = 4
        self.pieces = [] # index 'era': [x,y,type]
        self.board_objects = []
        for era in range(3):
            # each era starts the same
            self.pieces.append([[0,0,1],[3,3,-1]])
            self.board_objects.append([])
        self.copy_supplies = [4,4]
        self.current_focuses = [0,2]
        self.remaining_actions = 2
        self.chosen_pieceno = None
        self.chosen_piece_era = None
        self.time=0
        self.done=0

    def __str__(self):
        return str(self.getPlayerToMove()) + ''.join(str(r) for v in self.getImage() for r in v) 

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return np.array(self.getImage())[index]

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        return self._getValidMoves(color)

    def execute_move(self, move):
        """Perform the given move on the board_objects.
        color gives the color pf the piece to play (1=white,-1=black)
        """
        era,x1,y1,x2,y2,special = move
        pieceno = self._getPieceNo(era,x1,y1)
        legal = self._isLegalMove(pieceno,era,x2,y2,special)
        if legal>=0:
           #print("Accepted move: ",move)
           self._moveByPieceNo(pieceno,era,x2,y2,special)
        else:
           logger.debug("Illegal move:",move,legal)
        if abs(special) < 2:
            era += special

        return era,self._getPieceNo(era,x2,y2)
   
    def getImage(self):
        image = []
        for era in range(3):
            board = [[0 for col in range(4)] for row in range(4)]
            for item in self.board_objects[era]:
                board[item[0]][item[1]] = item[2]*10
            for piece in self.pieces[era]:
                if piece[0] >= 0: board[piece[0]][piece[1]] = piece[2] + board[piece[0]][piece[1]]
            image.append(board)
        return image

    def getPlayerToMove(self):
        return -(self.time%2*2-1)
    
    def getPlayerIndex(self):
        return 0 if (self.getPlayerToMove() > 0) else 1

################## Internal methods ##################

    # Special Variable Meanings
    # 0: Normal Move Within Same Era from x1,y1 to x2,y2
    # 1: Time travel forward
    # -1: Time travel backward

    def _isLegalMove(self,pieceno,era,x2,y2,special) -> int:
        try:

            if x2 < 0 or x2 > 3 or y2 < 0 or y2 > 3: return -1
            
            piece = self.pieces[era][pieceno]
            x1=piece[0]
            y1=piece[1]
            if x1<0: return -2 #piece was captured

            piecetype = piece[2]
            if (piecetype == -1 and self.time%2 == 0) or (piecetype != -1 and self.time%2 == 1): return -3 #wrong player

            if special == 1:
                if era == 2:
                    return -4 # cannot travel forward while in the future
                for apiece in (self.pieces[era + 1] + self.board_objects[era + 1]):
                    if x1 == apiece[0] and y1 == apiece[1]: return -5 #time travel target space blocked
            elif special == -1:
                if era == 0:
                    return -6 # cannot travel backward while in the past
                if self.copy_supplies[self.getPlayerIndex()] == 0:
                    return -7 # cannot travel back in time with no copies left
                for apiece in (self.pieces[era - 1] + self.board_objects[era - 1]):
                    if x1 == apiece[0] and y1 == apiece[1]: return -8 #time travel target space blocked
            
            return 0 # legal move
        except Exception as ex:
            logger.debug("error in islegalmove ",ex,pieceno,x2,y2)
            raise
    
    def _forceMove(self,pieceno,era,x2,y2):
        """
        Forcibly move the given piece within the given era to
        the position x2,y2. Handles any collisions and deaths that
        can occur with this move, recursively calling itself when
        a piece collides with an enemy piece and 'forces' it to move.
        """
        moving_piece = self.pieces[era][pieceno]
        # check for collision with wall
        if x2 < 0 or x2 > 3 or y2 < 0 or y2 > 3:
            # this piece is squished and dies
            moving_piece[0] = -99
            return
        # check for target object
        # TODO
        # check for a piece in the way
        target_piece_number = self._getPieceNo(era,x2,y2)
        if target_piece_number > -1:
            target_piece = self.pieces[era][target_piece_number]
            if target_piece[2] == moving_piece[2]:
                # a paradox occurs, so both copies die
                moving_piece[0] = -99
                target_piece[0] = -99
                return
            else:
                # the opposing piece is an enemy copy, and is pushed forward
                next_x = moving_piece[0] + 2 * (x2 - moving_piece[0])
                next_y = moving_piece[1] + 2 * (y2 - moving_piece[1])
                self._forceMove(target_piece_number,era,next_x,next_y)

        # moving to empty space
        moving_piece[0] = x2
        moving_piece[1] = y2

    def _moveByPieceNo(self,pieceno,era,x2,y2,special):
        
        legal = self._isLegalMove(pieceno,era,x2,y2,special)
        if legal != 0: return legal

        if special == 0: # normal move within same era
            self._forceMove(pieceno,era,x2,y2)
        elif special == 1: # time travel forward
            piece = self.pieces[era].pop(pieceno)
            self.pieces[era + 1].append(piece)
        elif special == -1: # time travel backwards
            piece = self.pieces[era][pieceno]
            self.pieces[era - 1].append(piece.copy())
            self.copy_supplies[self.getPlayerIndex()] -= 1
        
    def _getWinLose(self):
        logger.debug("Checking win-lose...")
        opponent_eras = 0
        opponent_id = -1 if (self.getPlayerToMove() > 0) else 1
        for era in range(3):
            for apiece in self.pieces[era]:
                if (apiece[2] == opponent_id) and (apiece[0] > -1):
                    logger.debug(f"  Found {apiece} in era {era}")
                    opponent_eras += 1
                    break
            if opponent_eras > 1:
                return 0 # nobody has won yet
        logger.debug("    Winner found?")
        return self.getPlayerToMove() # the current player just won
   
    def _getPieceNo(self,era,x,y):
        for pieceno in range(len(self.pieces[era])):
            piece=self.pieces[era][pieceno]
            if piece[0]==x and piece[1]==y: return pieceno
        return -1    

    def _canMoveTo(self,era,x2,y2):
        """Returns False if no piece is allowed to voluntarily move to the
        spot x2,y2 in the given era, and True if it is allowed.

        Movement is not allowed off the board (through walls), or into
        spaces with certain objects.
        
        Voluntary movement is from an action, not from being pushed."""
        if x2 < 0 or x2 > 3 or y2 < 0 or y2 > 3:
            return False
        for obj in self.board_objects[era]:
            if x2 == obj[0] and y2 == obj[1]:
                return False
        return True
    
    def _canTimeTravelTo(self,era,x2,y2):
        """Returns False if no piece is able to time travel to the given
        space in the given era, and True if it is possible.

        Time travel movement is more restricted than regular moves within
        a certain era. You cannot time travel to a space occupied by
        any player's piece or any object.

        Does not consider any player's supply of copies.
        """
        if not self._canMoveTo(era,x2,y2):
            return False
        for piece in self.pieces[era]:
            if x2 == piece[0] and y2 == piece[1] and piece[0] > -1:
                return False
        return True

    def _getValidStartMoves(self, pieceno:int, era:int):
        """Input: The era and piece to specify a certain piece, which can
        then be used to identify the player.
        
        Returns a list of moves for the given piece, which are themselves a list:
        [ era,x1,y1,x2,y2,special ].

        This function assumes that the piece needs to make two moves, and only
        includes moves that are guaranteed to lead to a second move.
        """
        moves = []
        piece = self.pieces[era][pieceno]
        x1,y1 = (piece[0],piece[1])
        player_supply = self.copy_supplies[0 if (piece[2] > 0) else 1]
        ortho_coords = [(x1,y1-1),(x1+1,y1),(x1,y1+1),(x1-1,y1)]
        # if this piece can move in the current era,
        # they should be able to move back (second move guaranteed)
        for x2,y2 in ortho_coords:
            if self._canMoveTo(era,x2,y2):
                moves.append([era,x1,y1,x2,y2,0]) # second move guaranteed
        
        # check if this piece could time travel
        # # time travel forward check
        if era < 2 and self._canTimeTravelTo(era + 1, x1, y1):
            # they can travel forward, but do they have a second move?
            if era == 0 and self._canTimeTravelTo(2, x1, y1):
                # they can time travel forward a second time
                moves.append([era,x1,y1,x1,y1,1])
            elif player_supply > 0:
                # they can travel right back
                # (they just came from this space,
                # which should be empty now)
                moves.append([era,x1,y1,x1,y1,1])
            else:
                # if they can't travel back in time afterwards,
                # we must check if they can move around in the new era
                for x2,y2 in ortho_coords:
                    if self._canMoveTo(era + 1,x2,y2):
                        moves.append([era,x1,y1,x1,y1,1]) # they can move afterwards
                        break
        # # time travel backward check:
        if (era > 0) and (player_supply > 0) and (self._canTimeTravelTo(era - 1, x1, y1)):
            # they cannot travel back forwards, as a copy is placed there
            # we must check if they have another move
            if (era == 2) and (player_supply > 1) and (self._canTimeTravelTo(0, x1, y1)):
                # they can time travel backwards a second time
                moves.append([era,x1,y1,x1,y1,-1])
            else:
                # check movement in this previous era
                for x2,y2 in ortho_coords:
                    if self._canMoveTo(era - 1,x2,y2):
                        moves.append([era,x1,y1,x1,y1,-1]) # they can move afterwards
                        break
        return moves
    
    def _getAllValidMoves(self, pieceno:int, era:int):
        """Input: The era and piece to specify a certain piece, which can
        then be used to identify the player.
        
        Returns a list of moves for the given piece, which are themselves a list:
        [ era,x1,y1,x2,y2,special ].

        This function returns ANY valid moves, even ones that may somehow leave
        a piece stuck. It is meant for the second action of a turn.
        """
        moves = []
        piece = self.pieces[era][pieceno]
        x1,y1 = (piece[0],piece[1])
        player_supply = self.copy_supplies[0 if (piece[2] > 0) else 1]
        ortho_coords = [(x1,y1-1),(x1+1,y1),(x1,y1+1),(x1-1,y1)]
        # if this piece can move orthogonally in the current era
        for x2,y2 in ortho_coords:
            if self._canMoveTo(era,x2,y2):
                moves.append([era,x1,y1,x2,y2,0])
        
        # check if this piece could time travel
        # # time travel forward check
        if era < 2 and self._canTimeTravelTo(era + 1, x1, y1):
            moves.append([era,x1,y1,x1,y1,1])
        # # time travel backward check:
        if (era > 0) and (player_supply > 0) and (self._canTimeTravelTo(era - 1, x1, y1)):
            moves.append([era,x1,y1,x1,y1,-1])

        return moves

    def _getValidMoves(self,player):
        moves=[]
        if self.remaining_actions == 2:
            # can choose any piece from current focused era, but
            # only those that can perform two actions in a row
            focused_era = self.current_focuses[0 if (player > 0) else 1]
            # must 'choose' a piece to move twice this turn
            for pieceno,apiece in enumerate(self.pieces[focused_era]):
                # only check pieces belonging to this player
                # that are not dead
                if (apiece[2] == player) and (apiece[0] > -1):
                    moves += self._getValidStartMoves(pieceno, focused_era)
        elif self.remaining_actions == 1 and self.chosen_pieceno > -1:
            # we must make an action with the chosen piece
            moves = self._getAllValidMoves(self.chosen_pieceno,self.chosen_piece_era)

        #print("moves ",moves)
        return moves

