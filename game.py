import pygame
from pygame.locals import *
import sys
import numpy as np
import random

pygame.init()

fps = 100

# Board size
board_width = 12
board_height = 24

# Offset
initial_offset = (6, 0)

# Entire screen size
display_width = 30

# Rectangle size
box_size = 20

# Auxiliary variables
blank = '.'
blank_row = blank * board_width

# Colors
black = (0, 0, 0)
white = (255, 255, 255)

# Display, clock and font objects
display = pygame.display.set_mode((display_width * box_size, board_height * box_size))
pygame.display.set_caption("Tetris")
clock = pygame.time.Clock()
my_font = pygame.font.SysFont("'monospace'", 30)

# Create a list of tetrominos
tetrominos = ['T', 'I', 'J', 'L', 'O', 'S', 'Z']

# Tetromino structures
tetromino_structures = {

    'T': [

        ['000',
         '.0.'],

        ['.0',
         '00',
         '.0'],

        ['.0.',
         '000'],

        ['0.',
         '00',
         '0.']

    ],

    'I': [

        ['1111'],

        ['1',
         '1',
         '1',
         '1'],

    ],

    'J': [

        ['.2',
         '.2',
         '22'],

        ['2..',
         '222'],

        ['22',
         '2.',
         '2.'],

        ['222',
         '..2']

    ],

    'L': [

        ['3.',
         '3.',
         '33'],

        ['333',
         '3..'],

        ['33',
         '.3',
         '.3'],

        ['..3',
         '333']

    ],

    'O': [

        ['44',
         '44.']

    ],

    'S': [

        ['.55',
         '55.'],

        ['5.',
         '55',
         '.5']

    ],

    'Z': [

        ['66.',
         '.66'],

        ['.6',
         '66',
         '6.']

    ]

}

# Tetromino colors
tetromino_colors = {

    'T': (255, 0, 0),
    'I': (0, 255, 255),
    'J': (0, 0, 255),
    'L': (0, 255, 0),
    'O': (255, 255, 0),
    'S': (255, 165, 0),
    'Z': (160, 32, 240)

}

# Numeric tetromino colors
numeric_tetromino_colors = {

    '0': (255, 0, 0),
    '1': (0, 255, 255),
    '2': (0, 0, 255),
    '3': (0, 255, 0),
    '4': (255, 255, 0),
    '5': (255, 165, 0),
    '6': (160, 32, 240)

}

tetromino_rotation_count = {

    'T': 4,
    'I': 2,
    'J': 4,
    'L': 4,
    'O': 1,
    'S': 2,
    'Z': 2

}

counter = 0


def game():

    global counter

    piece = Piece()
    board = Board()
    points = 0

    while True:
        display.fill(black)

        if counter % 10 == 0:
            piece.fall()
        points = board.refresh(points)

        if piece.at_bottom() or piece.collision(board, type='bottom'):
            board.add_piece(piece)
            piece = Piece()

        if board.end_game():
            piece = Piece()
            board = Board()

        piece.draw()
        board.draw()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == K_DOWN:
                    piece.offset[1] += 1
                elif event.key == K_RIGHT and not piece.collision(board, type='right') and piece.within_board(type='right'):
                    piece.offset[0] += 1
                elif event.key == K_LEFT and not piece.collision(board, type='left') and piece.within_board(type='left'):
                    piece.offset[0] -= 1
                elif event.key == K_q and not piece.collision(board, type='clockwise') and piece.within_board(type='clockwise'):
                    piece.rotation += 1
                elif event.key == K_e and not piece.collision(board, type='counterclockwise') and piece.within_board(type='counterclockwise'):
                    piece.rotation -= 1
                elif event.key == K_SPACE:
                    pass

        pygame.display.flip()
        clock.tick(fps)

        counter += 1


class Piece(object):
    def __init__(self):
        self._type = random.choice(tetrominos)
        self._rotation = 0
        self._number_of_rotations = tetromino_rotation_count[self.type]
        self._structure = tetromino_structures[self.type][self.rotation]
        self._length = len(self.structure[0])
        self._height = len(self.structure)
        self._offset = list(initial_offset)
        self._color = tetromino_colors[self._type]

    def fall(self):
        if not self.at_bottom():
            self.offset[1] += 1

    def at_bottom(self):
        if self.height + self.offset[1] == board_height:
            return True
        return False

    def within_board(self, type=None):

        if type == 'left':
            if self.offset[0] - 1 >= 0:
                return True
            return False

        elif type == 'right':
            if self.offset[0] + self.length + 1 <= board_width:
                return True
            return False

        elif type == 'clockwise':
            rotation = (self.rotation + 1) % self._number_of_rotations
            structure = tetromino_structures[self.type][rotation]
            height = len(structure)
            length = len(structure[0])
            if self.offset[0] + length <= board_width and self.offset[1] + height <= board_height:
                return True
            return False

        elif type == 'counterclockwise':
            rotation = (self.rotation - 1) % self._number_of_rotations
            structure = tetromino_structures[self.type][rotation]
            height = len(structure)
            length = len(structure[0])
            if self.offset[0] + length <= board_width and self.offset[1] + height <= board_height:
                return True
            return False

        else:
            if self.offset[0] >= 0 and self.offset[0] + self.length <= board_width and self.offset[1] >= 0 and self.offset[1] + self.height <= board_height:
                return True
            return False

    def collision(self, board, type=None):
        if self.within_board():

            if type == 'left':
                if self.offset[0] > 0:
                    listed_board = [list(board.structure[i]) for i in range(board_height)]
                    listed_piece = [list(self.structure[i]) for i in range(self.height)]
                    for y in range(self.height):
                        for x in range(self.length):
                            if listed_board[y + self.offset[1]][x + self.offset[0] - 1] != blank and listed_piece[y][x] != blank:
                                return True
                    return False

            elif type == 'right':
                if self.offset[0] < board_width:
                    listed_board = [list(board.structure[i]) for i in range(board_height)]
                    listed_piece = [list(self.structure[i]) for i in range(self.height)]
                    for y in range(self.height):
                        for x in range(self.length):
                            try:
                                if listed_board[y + self.offset[1]][x + self.offset[0] + 1] != blank and listed_piece[y][x] != blank:
                                    return True
                            except:
                                return False
                    return False

            elif type == 'bottom':
                if self.offset[1] < board_height and not self.at_bottom():
                    listed_board = [list(board.structure[i]) for i in range(board_height)]
                    listed_piece = [list(self.structure[i]) for i in range(self.height)]
                    for y in range(self.height):
                        for x in range(self.length):
                            if listed_board[y + self.offset[1] + 1][x + self.offset[0]] != blank and listed_piece[y][x] != blank:
                                return True
                    return False

            elif type == 'clockwise':
                rotation = (self.rotation + 1) % self._number_of_rotations
                structure = tetromino_structures[self.type][rotation]
                height = len(structure)
                length = len(structure[0])
                listed_board = [list(board.structure[i]) for i in range(board_height)]
                listed_piece = [list(structure)[i] for i in range(height)]
                if not self.at_bottom():
                    for y in range(height):
                        for x in range(length):
                            try:
                                if listed_board[y + self.offset[1]][x + self.offset[0]] != blank and listed_piece[y][x] != blank:
                                    return True
                            except:
                                False
                    return False

            elif type == 'counterclockwise':
                rotation = (self.rotation - 1) % self._number_of_rotations
                structure = tetromino_structures[self.type][rotation]
                height = len(structure)
                length = len(structure[0])
                listed_board = [list(board.structure[i]) for i in range(board_height)]
                listed_piece = [list(structure)[i] for i in range(height)]
                if not self.at_bottom():
                    for y in range(height):
                        for x in range(length):
                            try:
                                if listed_board[y + self.offset[1]][x + self.offset[0]] != blank and listed_piece[y][x] != blank:
                                    return True
                            except:
                                return False
                    return False

            else:
                listed_board = [list(board.structure[i]) for i in range(board_height)]
                listed_piece = [list(self.structure[i]) for i in range(self.height)]
                for y in range(self.height):
                    for x in range(self.length):
                        if listed_board[y + self.offset[1]][x + self.offset[0]] != blank and listed_piece[y][x] != blank:
                            return True
                return False

    def draw(self):
        for y in range(self.height):
            for x in range(self.length):
                if self.structure[y][x] != blank:
                    draw_rectangle(x, y, self.color, self.offset)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, x):
        self._type = x

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, x):
        self._rotation = x % self._number_of_rotations
        self.structure = tetromino_structures[self.type][self.rotation]
        self.length = len(self.structure[0])
        self.height = len(self.structure)

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, x):
        self._structure = x

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, x):
        self._length = x

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, x):
        self._height = x

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, x):
        self._offset = x

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, x):
        self._color = x


class Board(object):
    def __init__(self):
        self._structure = [blank_row for i in range(board_height)]

    def add_piece(self, piece):
        if piece.within_board():
            listed_board = [list(self.structure[i]) for i in range(board_height)]
            listed_piece = [list(piece.structure[i]) for i in range(piece.height)]
            for y in range(piece.height):
                for x in range(piece.length):
                    if listed_piece[y][x] != blank:
                        listed_board[y + piece.offset[1]][x + piece.offset[0]] = listed_piece[y][x]
            self.structure = [''.join(listed_board[i]) for i in range(board_height)]

    def refresh(self, p=0):
        points = p
        rows = [y for y in range(board_height) if blank not in self.structure[y]]
        if len(rows) > 0:
            points += len(rows)
            for index in sorted(rows, reverse=True):
                self.structure.pop(index)
            self.structure = [blank_row] * len(rows) + self.structure
        return points

    def draw(self):
        for y in range(board_height):
            for x in range(board_width):
                if self.structure[y][x] != blank:
                    draw_rectangle(x, y, numeric_tetromino_colors[self.structure[y][x]])
        pygame.draw.line(display, white, (board_width * box_size, 0), (board_width * box_size, board_height * box_size), 1)

    def end_game(self):
        for i in range(4):
            if blank_row not in self.structure[i]:
                return True
        return False

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, x):
        self._structure = x


def allowed_actions(board, piece):
    allowed_actions = [1]

    if piece.within_board(type='right') and not piece.collision(board, type='right'):
        allowed_actions.append(1)
    else:
        allowed_actions.append(0)

    if piece.within_board(type='left') and not piece.collision(board, type='left'):
        allowed_actions.append(1)
    else:
        allowed_actions.append(0)

    if piece.within_board(type='clockwise') and not piece.collision(board, type='clockwise'):
        allowed_actions.append(1)
    else:
        allowed_actions.append(0)

    if piece.within_board(type='counterclockwise') and not piece.collision(board, type='counterclockwise'):
        allowed_actions.append(1)
    else:
        allowed_actions.append(0)

    return np.asarray(allowed_actions)


def draw_rectangle(x, y, color, offset=[0, 0]):
    pygame.draw.rect(display, color, ((x + offset[0]) * box_size, (y + offset[1]) * box_size, box_size, box_size))
    pygame.draw.rect(display, black, ((x + offset[0]) * box_size, (y + offset[1]) * box_size, box_size, box_size), 1)


def convet_to_input(board, piece):
    listed_board = [list(board.structure[i]) for i in range(board_height)]
    listed_piece = [list(piece.structure[i]) for i in range(piece.height)]
    for y in range(piece.height):
        for x in range(piece.length):
            if listed_piece[y][x] != blank:
                listed_board[y + piece.offset[1]][x + piece.offset[0]] = 1
    for y in range(board_height):
        for x in range(board_width):
            if listed_board[y][x] != blank:
                listed_board[y][x] = 1
            else:
                listed_board[y][x] = 0
    return np.asarray(listed_board).reshape(board_height, board_width)


if __name__ == '__main__':
    game()
