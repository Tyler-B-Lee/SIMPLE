import numpy as np
from stable_baselines import logger

class Board():

    def __init__(self):
        self.size = 4
        self.pieces = [] # index 'era': {(x,y): type}
        self.board_objects = []
        for era in range(3):
            # each era starts the same
            self.pieces.append({(0,0): 1, (3,3): -1})
            self.board_objects.append({})
        self.copy_supplies = [4,4]
        self.current_focuses = [0,2]
        self.remaining_actions = 2
        self.chosen_piece_x = None
        self.chosen_piece_y = None
        self.chosen_piece_era = None
        self.time=0
        self.done=0

        # Chapter 1
        self.seed_supply = 5
        self.num_squishes = 0
        self.num_paradoxes = 0
        self.num_seeds_planted = 0
        self.plant_kills = 0

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
        Returns (era,pieceno) of the selected piece after the move is done.
        """
        era,x1,y1,x2,y2,special = move
        legal = self._isLegalMove(era,x1,y1,x2,y2,special)
        if legal>=0:
            #print("Accepted move: ",move)
            self.chosen_piece_era,self.chosen_piece_x,self.chosen_piece_y = self._moveByPieceNo(era,x1,y1,x2,y2,special)
        else:
            logger.debug("Illegal move:",move,legal)
    
        return
   
    def getImage(self):
        image = []
        for era in range(3):
            board = [[0 for col in range(4)] for row in range(4)]
            for (x,y),value in self.board_objects[era].items():
                board[x][y] = value*10
            for (x,y),value in self.pieces[era].items():
                board[x][y] += value
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
    # 2: Plant Seed
    # -2: Remove Seed

    # Objects:
    # 1: Seed
    # 2: Shrub
    # 3: Standing Tree
    # 4,5,6,7 = Fallen Tree (Pushed U,D,L,R)

    def _isLegalMove(self,era,x1,y1,x2,y2,special) -> int:
        try:

            if x2 < 0 or x2 > 3 or y2 < 0 or y2 > 3: return -1
            
            if x1<0: return -2 #piece was captured
            piecetype = self.pieces[era][(x1,y1)]

            if (piecetype == -1 and self.time%2 == 0) or (piecetype != -1 and self.time%2 == 1): return -3 #wrong player

            if special == 1:
                if era == 2:
                    return -4 # cannot travel forward while in the future
                if (x2,y2) in self.pieces[era + 1]:
                    return -5 #time travel target space blocked
                if (x2,y2) in self.board_objects[era + 1]:
                    if self.board_objects[era + 1][(x2,y2)] != 1: return -5 #time travel target space blocked

            elif special == -1:
                if era == 0:
                    return -6 # cannot travel backward while in the past
                if self.copy_supplies[self.getPlayerIndex()] == 0:
                    return -7 # cannot travel back in time with no copies left
                if (x2,y2) in self.pieces[era - 1]:
                    return -8 #time travel target space blocked
                if (x2,y2) in self.board_objects[era - 1]:
                    if self.board_objects[era - 1][(x2,y2)] != 1: return -8 #time travel target space blocked

            elif special == 2:
                # planting a seed
                if self.seed_supply == 0:
                    return -9 # no seeds left to plant
            elif special == -2:
                # removing a seed
                if (x2,y2) in self.board_objects[era]:
                    if self.board_objects[era][(x2,y2)] == 1:
                        return 0 # there is a seed there
                return -10 # no seed to remove in given spot
            
            return 0 # legal move
        except Exception as ex:
            logger.debug("error in islegalmove ",ex,era,x1,y1,x2,y2,special)
            raise
    
    def _forceMove(self,era,x1,y1,x2,y2):
        """
        Forcibly move the given piece within the given era to
        the position x2,y2. Handles any collisions and deaths that
        can occur with this move, recursively calling itself when
        a piece collides with an enemy piece and 'forces' it to move.

        Returns the (era,x,y) of the piece after moving, or (None,None,None) if it dies.
        """
        # get the piece to be moved
        # this 'kills' the piece if it is not placed somewhere else on the board
        piece_team = self.pieces[era].pop((x1,y1), None)

        # check for collision with wall
        if x2 < 0 or x2 > 3 or y2 < 0 or y2 > 3:
            # this piece is squished and dies
            logger.debug(f"  💀🧱 - Ouch! Copy at ({x1},{y1}) in era {era} was squished into a wall!")
            self.num_squishes += 1
            return None,None,None

        # check for target object
        if (x2,y2) in self.board_objects[era]:
            value = self.board_objects[era][(x2,y2)]
            if value in {2,4,5,6,7}:
                # this piece is squished and dies
                logger.debug(f"  💀🌵 - Ouch! Copy at ({x1},{y1}) in era {era} was squished into a tough plant!")
                self.num_squishes += 1
                self.plant_kills += 1
                return None,None,None
            elif value == 3:
                # there is a tree in the way that must be knocked over
                next_x = x1 + 2 * (x2 - x1)
                next_y = y1 + 2 * (y2 - y1)
                # if a player is pushed into a standing tree
                # that itself can't be knocked over, the player dies
                if self._canMoveTo(era,x2,y2,next_x,next_y):
                    self._fellTree(era,x2,y2,next_x,next_y)
                else:
                    logger.debug(f"  💀🌴 - Ouch! Copy at ({x1},{y1}) in era {era} was squished into an immovable tree!")
                    self.num_squishes += 1
                    self.plant_kills += 1
                    return None,None,None

        # check for a piece in the way
        if (x2,y2) in self.pieces[era]:
            if piece_team == self.pieces[era][(x2,y2)]:
                # a paradox occurs, so both copies die
                del self.pieces[era][(x2,y2)]
                self.num_paradoxes += 1
                logger.debug(f"  💀⚠️💀 - Yikes! Copies at ({x1},{y1}) and ({x2},{y2}) in era {era} died in a paradox!")
                return None,None,None
            else:
                # the opposing piece is an enemy copy, and is pushed forward
                next_x = x1 + 2 * (x2 - x1)
                next_y = y1 + 2 * (y2 - y1)
                self._forceMove(era,x2,y2,next_x,next_y)

        # the piece survives and moves to empty space
        self.pieces[era][(x2,y2)] = piece_team
        logger.debug(f"  - Copy moves from ({x1},{y1}) to ({x2},{y2}) in era {era}.")
        return era,x2,y2
    
    def _plantSeed(self,era,x,y):
        """
        Plants a seed at the given coordinates in the given era.
        Also attempts to grow into a bush or tree in future eras.
        """
        self.board_objects[era][(x,y)] = 1 # plant seed in given era
        self.seed_supply -= 1
        self.num_seeds_planted += 1
        logger.debug(f"  🌰⬇️ - Seed planted at ({x},{y}) in era {era}.")

        # NOTE: This code was optimized by ChatGPT!
        for future_era in range(era + 1, 3):
            if (x,y) in self.board_objects[future_era] or (x,y) in self.pieces[future_era]:
                return # object blocking future growth
            self.board_objects[future_era][(x,y)] = future_era - era + 1

    def _removeSeed(self,era,x,y):
        """
        Removes a seed at the given coordinates in the given era,
        along with the subsequent plants that grew from it
        in any future eras.
        """
        # remove the seed
        if (x, y) in self.board_objects[era] and self.board_objects[era][(x, y)] == 1:
            del self.board_objects[era][(x, y)]
            self.seed_supply += 1
            logger.debug(f"  🌰⬆️ - Seed removed at ({x},{y}) in era {era}.")
        # remove any bush in the next era
        if era != 2 and (x, y) in self.board_objects[era+1] and self.board_objects[era+1][(x, y)] == 2:
            del self.board_objects[era+1][(x, y)]
        else:
            return
        # remove any tree in the era after that
        if era != 1:
            objects_to_remove = {
                3: (x, y),
                4: (x-1, y),
                5: (x+1, y),
                6: (x, y-1),
                7: (x, y+1)
            }
            for obj_type, obj_loc in objects_to_remove.items():
                if obj_loc in self.board_objects[era+2] and self.board_objects[era+2][obj_loc] == obj_type:
                    del self.board_objects[era+2][obj_loc]
                    return
    
    def _fellTree(self,era,x1,y1,x2,y2):
        """
        Make the tree in the given era at x1,y1 in
        the list of objects fall to x2,y2 and carry out any
        effects (squishing, pushing other trees). The tree is assumed
        to fall in-bounds, and onto a legal space.
        """
        dir_dict = {
            (-1,0): 4,
            (1,0): 5,
            (0,-1): 6,
            (0,1): 7
        }
        del self.board_objects[era][(x1,y1)]

        if (x2,y2) in self.pieces[era]:
            # copy gets crushed and dies
            del self.pieces[era][(x2,y2)]
            logger.debug(f"  💀🌴 - Oof! Piece at ({x2},{y2}) in era {era} was crushed by a falling tree!")
            self.plant_kills += 1
        if (x2,y2) in self.board_objects[era]:
            value = self.board_objects[era][(x2,y2)]
            if value == 1:
                # seeds get removed
                self._removeSeed(era,x2,y2)
            elif value == 3:
                # the next tree is also toppled
                next_x = x1 + 2 * (x2 - x1)
                next_y = y1 + 2 * (y2 - y1)
                self._fellTree(era,x2,y2,next_x,next_y)
        # the tree falls to the target space
        self.board_objects[era][(x2,y2)] = dir_dict[(x2 - x1, y2 - y1)]
        
    def _moveByPieceNo(self,era,x1,y1,x2,y2,special):
        """Given the move, calls the corresponding function to make the move.
        Returns the era and position of the target piece, if it is still alive."""
        legal = self._isLegalMove(era,x1,y1,x2,y2,special)
        if legal != 0: return legal

        if special == 0: # normal move within same era
            return self._forceMove(era,x1,y1,x2,y2)
        elif special == 1: # time travel forward
            piece_team = self.pieces[era].pop((x1,y1), None)
            self.pieces[era + 1][(x1,y1)] = piece_team
            logger.debug(f"  ⏳📜 - Copy at ({x1},{y1}) time travels forward to {era + 1}.")
            return era + 1,x1,y1
        elif special == -1: # time travel backwards
            piece_team = self.pieces[era][(x1,y1)]
            self.pieces[era - 1][(x1,y1)] = piece_team
            self.copy_supplies[self.getPlayerIndex()] -= 1
            logger.debug(f"  ⏳🔮 - Copy at ({x1},{y1}) time travels backward to {era - 1}.")
            return era - 1,x1,y1
        elif special == 2: # plant a seed
            self._plantSeed(era,x2,y2)
            return era,x1,y1
        elif special == -2: # remove seed
            self._removeSeed(era,x2,y2)
            return era,x1,y1
        
    def _getWinLose(self):
        "Returns 0 if the game is not over, 1 if white won, or -1 if black won."
        # logger.debug("Checking win-lose...")
        opponent_eras = 0
        opponent_id = -1 if (self.getPlayerToMove() > 0) else 1
        for era_dict in self.pieces:
            if opponent_id in era_dict.values():
                opponent_eras += 1
        return 0 if (opponent_eras > 1) else self.getPlayerToMove()

    def _canMoveTo(self,era,x1,y1,x2,y2):
        """Returns False if no piece is allowed to voluntarily move to the
        spot x2,y2 in the given era, and True if it is allowed.

        In this chapter, THIS IS IDENTICAL 
        to the check if a tree can fall to a space.

        Movement is not allowed off the board (through walls), or into
        spaces with certain objects.
        
        Voluntary movement of a copy is from an action, not from being pushed."""
        if x2 < 0 or x2 > 3 or y2 < 0 or y2 > 3:
            return False # cannot go out of bounds
        if (x2,y2) in self.board_objects[era]:
            value = self.board_objects[era][(x2,y2)]
            if value == 1:
                return True # seeds do not impede movement/trees falling
            elif value == 3:
                # standing tree in the way
                next_x = x1 + 2 * (x2 - x1)
                next_y = y1 + 2 * (y2 - y1)
                # can move to x2,y2 only if the next tree can also fall over
                return self._canMoveTo(era,x2,y2,next_x,next_y)
            return False # non-pushable object blocking
        # empty spaces or spaces with
        # copies do not impede movement
        return True
    
    def _canTimeTravelTo(self,era,x2,y2):
        """Returns False if no piece is able to time travel to the given
        space in the given era, and True if it is possible.

        Time travel movement is more restricted than regular moves within
        a certain era. You cannot time travel to a space occupied by
        any player's piece or any object.

        Does not consider any player's supply of copies.
        """
        if x2 < 0 or x2 > 3 or y2 < 0 or y2 > 3:
            return False
        if (x2,y2) in self.pieces[era]:
            return False
        if (x2,y2) in self.board_objects[era]:
            if self.board_objects[era][(x2,y2)] != 1:
                return False # non-seed object in the way
        return True

    def _canPlantSeed(self,era,x1,y1,x2,y2):
        """Returns False if a seed cannot be planted at the given space
        in the given era. x1 and y1 are the coords of the planting piece.

        Does NOT consider seed supply.
        """
        if x2 < 0 or x2 > 3 or y2 < 0 or y2 > 3:
            return False
        if x1 == x2 and y1 == y2:
            # a piece is allowed to plant a seed in its own square (if no seed there already)
            return (x1,y1) not in self.board_objects[era]

        # piece not planting on its own square
        if (x2,y2) in self.pieces[era] or (x2,y2) in self.board_objects[era]:
            # there is a piece/object in the way that is not the placing piece
            return False
        return True
        

    def _getValidStartMoves(self, era:int, x1:int, y1:int):
        """Input: The era and piece to specify a certain piece, which can
        then be used to identify the player.
        
        Returns a list of moves for the given piece, which are themselves a list:
        [ era,x1,y1,x2,y2,special ].

        This function assumes that the piece needs to make two moves, and only
        includes moves that are guaranteed to lead to a second move.
        """
        moves = []
        piece_team = self.pieces[era][(x1,y1)]
        player_supply = self.copy_supplies[0 if (piece_team > 0) else 1]
        ortho_coords = [(x1,y1-1),(x1+1,y1),(x1,y1+1),(x1-1,y1)]
        # if this piece can move in the current era,
        # they should be able to move back (second move guaranteed)
        # Same for planting a seed (can then remove it)
        if self.seed_supply > 0 and self._canPlantSeed(era,x1,y1,x1,y1):
            moves.append([era,x1,y1,x1,y1,2])
        for x2,y2 in ortho_coords:
            if self._canMoveTo(era,x1,y1,x2,y2):
                moves.append([era,x1,y1,x2,y2,0]) # second move guaranteed
            if self.seed_supply > 0 and self._canPlantSeed(era,x1,y1,x2,y2):
                moves.append([era,x1,y1,x2,y2,2])
        
        # check if this piece could time travel
        # # time travel forward check
        if era < 2 and self._canTimeTravelTo(era + 1, x1, y1):
            # they can travel forward, but do they have a second move?
            done = False
            if era == 0 and self._canTimeTravelTo(2, x1, y1):
                # they can time travel forward a second time
                moves.append([era,x1,y1,x1,y1,1])
                done = True
            elif player_supply > 0:
                # they can travel right back
                # (they just came from this space,
                # which should be empty now)
                moves.append([era,x1,y1,x1,y1,1])
                done = True
            elif self.seed_supply > 0 and self._canPlantSeed(era + 1,x1,y1,x1,y1):
                moves.append([era,x1,y1,x1,y1,1]) # plant seed on square in next era
                done = True
            if not done:
                # check for removing a seed (can then replant it)
                for (x,y),obtype in self.board_objects[era + 1].items():
                    if obtype == 1 and (x,y) in (ortho_coords + [(x1,y1)]):
                        moves.append([era,x1,y1,x1,y1,1])
                        done = True
                        break
            if not done:
                # if they can't travel back in time afterwards,
                # we must check if they can move around in the new era
                # or plant a seed
                for x2,y2 in ortho_coords:
                    if self._canMoveTo(era + 1,x1,y1,x2,y2) or (self.seed_supply > 0 and self._canPlantSeed(era + 1,x1,y1,x2,y2)):
                        moves.append([era,x1,y1,x1,y1,1]) # they can move/plant seed afterwards
                        break
        # # time travel backward check:
        if (era > 0) and (player_supply > 0) and (self._canTimeTravelTo(era - 1, x1, y1)):
            done = False
            # they cannot travel back forwards, as a copy is placed there
            # we must check if they have another move
            if (era == 2) and (player_supply > 1) and (self._canTimeTravelTo(0, x1, y1)):
                # they can time travel backwards a second time
                moves.append([era,x1,y1,x1,y1,-1])
                done = True
            elif self.seed_supply > 0 and self._canPlantSeed(era - 1,x1,y1,x1,y1):
                moves.append([era,x1,y1,x1,y1,-1])
                done = True
            # check for removing a seed (can then replant it)
            if not done:
                for (x,y),obtype in self.board_objects[era - 1].items():
                    if obtype == 1 and (x,y) in (ortho_coords + [(x1,y1)]):
                        moves.append([era,x1,y1,x1,y1,-1])
                        done = True
                        break
            if not done:
                # check movement in this previous era
                for x2,y2 in ortho_coords:
                    if self._canMoveTo(era - 1,x1,y1,x2,y2) or (self.seed_supply > 0 and self._canPlantSeed(era - 1,x1,y1,x2,y2)):
                        moves.append([era,x1,y1,x1,y1,-1]) # they can move afterwards
                        break
        
        # check for removing a seed (can then replant it)
        for (x,y),obtype in self.board_objects[era].items():
            if obtype == 1 and (x,y) in (ortho_coords + [(x1,y1)]):
                moves.append([era,x1,y1,x,y,-2])

        return moves
    
    def _getAllValidMoves(self, era:int, x1:int, y1:int):
        """Input: The era and piece to specify a certain piece, which can
        then be used to identify the player.
        
        Returns a list of moves for the given piece, which are themselves a list:
        [ era,x1,y1,x2,y2,special ].

        This function returns ANY valid moves, even ones that may somehow leave
        a piece stuck. It is meant for the second action of a turn.
        """
        moves = []
        piece_team = self.pieces[era][(x1,y1)]
        player_supply = self.copy_supplies[0 if (piece_team > 0) else 1]
        ortho_coords = [(x1,y1-1),(x1+1,y1),(x1,y1+1),(x1-1,y1)]
        # if this piece can move orthogonally OR plant seeds in the current era
        for x2,y2 in ortho_coords:
            if self._canMoveTo(era,x1,y1,x2,y2):
                moves.append([era,x1,y1,x2,y2,0])
            if self.seed_supply > 0 and self._canPlantSeed(era,x1,y1,x2,y2):
                moves.append([era,x1,y1,x2,y2,2])
        # also check current square to plant a seed
        if self.seed_supply > 0 and self._canPlantSeed(era,x1,y1,x1,y1):
            moves.append([era,x1,y1,x1,y1,2])
        
        # check for removing a seed
        for (x,y),obtype in self.board_objects[era].items():
            if obtype == 1 and (x,y) in (ortho_coords + [(x1,y1)]):
                moves.append([era,x1,y1,x,y,-2])
        
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
            for (x,y),value in self.pieces[focused_era].items():
                # only check pieces belonging to this player
                if (value == player):
                    moves += self._getValidStartMoves(focused_era,x,y)
        elif self.remaining_actions == 1 and self.chosen_piece_era != None:
            # we must make an action with the chosen piece
            moves = self._getAllValidMoves(self.chosen_piece_era,self.chosen_piece_x,self.chosen_piece_y)

        return moves
