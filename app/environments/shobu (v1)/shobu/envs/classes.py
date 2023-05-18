class Player():
    def __init__(self, color: str):
        self.color = color
        self.active_pieces = []
        for i in range(4):
            board = []
            self.active_pieces.append(board)
        self.symbol = "Black" if (self.color == 'b') else "White"


class Piece():
    def __init__(self, id: int, color: str, row: int, col: int, board_i: int) -> None:
        self.id = id
        self.color = color
        self.row = row
        self.col = col
        self.board_i = board_i
        s = '0' * (0 if (self.id > 9) else 1)
        self.symbol = f' ⚫{s}{self.id} ' if (self.color == 'b') else f' ⚪{s}{self.id} '
    
    def move(self, row, col):
        """Changes the row and col variables for this piece to the specified values."""
        self.row = row
        self.col = col