# Import required libraries
import pygame
from pygame.locals import *
import sys
import numpy as np
import random
import re
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import copy


pygame.init()

# Control the speed of the game
fps = 30

# Tetris board size
board_width = 10
board_height = 20

# Initial offset (where the new tetromino spawns)
initial_offset = (int(board_width / 2), 0)

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


def game():

    # Initialize the variables
    piece = Piece()
    next_piece = Piece(offset=(18, 2))
    board = Board()
    points = 0
    games_played = 0
    games_to_save_the_model = 100
    number_of_actions = 0
    average_number_of_actions = 0

    # Create the model and initialize other parameters
    learning_rate = 0.005
    exploration_rate = 0.025
    number_of_epochs = 1
    number_of_epochs_losing = 10
    reward_multiplier = 1
    losing_penalty = 10
    points_multiplier = 10

    # model = create_model(learning_rate=learning_rate)
    model = keras.models.load_model('3_model.h5')
    print(model.summary())

    while True:

        # Fill the blackground with black color
        display.fill(black)

        # Get the state of the board before an action is taken
        state = convert_to_input(board, piece)

        # Estimate points and reward before an action is taken
        points = board.refresh(p=points, multiplier=points_multiplier)
        reward = board.reward_function(multiplier=reward_multiplier)

        # Decide on which actions are currently allowed
        actions_allowed = allowed_actions(board, piece)

        # Adjust the predicted actions with the ones that are allowed, i.e. so that the NN cannot make an illegal action
        predicted_actions = np.multiply(model.predict(state), actions_allowed).reshape(5, 1)

        # Make an action
        if np.random.uniform(0, 1) < exploration_rate:
            try:
                action = np.random.choice(np.nonzero(predicted_actions)[0])
                if action == 1:
                    piece.offset[0] += 1
                elif action == 2:
                    piece.offset[0] -= 1
                elif action == 3:
                    piece.rotation += 1
                elif action == 4:
                    piece.rotation -= 1
                else:
                    pass
                number_of_actions += 1
            except:
                action = None
        else:
            try:
                if np.where(predicted_actions == np.max(predicted_actions[np.absolute(predicted_actions) > 0.0001]))[0] == 1:
                    action = 1
                    piece.offset[0] += 1
                elif np.where(predicted_actions == np.max(predicted_actions[np.absolute(predicted_actions) > 0.0001]))[0] == 2:
                    action = 2
                    piece.offset[0] -= 1
                elif np.where(predicted_actions == np.max(predicted_actions[np.absolute(predicted_actions) > 0.0001]))[0] == 3:
                    action = 3
                    piece.rotation += 1
                elif np.where(predicted_actions == np.max(predicted_actions[np.absolute(predicted_actions) > 0.0001]))[0] == 4:
                    action = 4
                    piece.rotation -= 1
                elif np.where(predicted_actions == np.max(predicted_actions[np.absolute(predicted_actions) > 0.0001]))[0] == 0:
                    action = 0
                number_of_actions += 1
            except:
                action = None

        # Manual control
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
                elif event.key == K_s:
                    print("The board structure is: %s" %(board.structure))
                elif event.key == K_r:
                    print("The current reward is: %s" %(reward))
                elif event.key == K_SPACE:
                    print("The input to NN is:\n %s" %(convert_to_input(board, piece)))

        if piece.at_bottom() or piece.collision(board, type='bottom'):
            board.add_piece(piece)
            piece.copy_other_piece(next_piece)
            next_piece = Piece(offset=(18, 2))

        if board.end_game():
            piece = Piece()
            board = Board()
            reward = 0
            new_reward = 0
            points = 0
            new_points = 0
            games_played += 1
            average_number_of_actions = average_number_of_actions + 1 / games_played * (number_of_actions - average_number_of_actions)
            number_of_actions = 0

            if games_played % games_to_save_the_model == 0:
                model.save('3_model.h5')

            if action != None:
                updated_actions = copy.deepcopy(predicted_actions)
                updated_actions[action] = updated_actions[action] - losing_penalty
                updated_actions = updated_actions.reshape(1, 5)
                print('#' * 330)
                print("The game is over")
                print("This was game %s" %(games_played))
                print("The long-run average number of actions taken during a game is: %s"%(average_number_of_actions))
                print("The action taken was: %s" %(action))
                print("Change in reward was: %s" %(change_in_reward))
                print("Change in points was: %s" %(change_in_points))
                print("The predicted actions were: %s" %(predicted_actions))
                print("The updated actions are: %s" %(updated_actions))

                model.fit(state, updated_actions, epochs=number_of_epochs_losing, batch_size=1)
        else:
            if action != None:
                new_state = convert_to_input(board, piece)
                new_points = board.refresh(points, multiplier=points_multiplier)
                new_reward = board.reward_function(multiplier=reward_multiplier)
                new_actions_allowed = allowed_actions(board, piece)
                new_predicted_actions = np.multiply(model.predict(new_state), new_actions_allowed).reshape(5, 1)
                change_in_points = new_points - points
                change_in_reward = new_reward - reward

                # if change_in_points != 0 or change_in_reward != 0:
                next_q_value = max(new_predicted_actions) + change_in_points + change_in_reward
                updated_actions = copy.deepcopy(predicted_actions)
                updated_actions[action] = next_q_value
                updated_actions = updated_actions.reshape(1, 5)
                print('#' * 330)
                print("This is game %s" %(games_played))
                print("The long-run average number of actions taken during a game is: %s"%(average_number_of_actions))
                print("The action taken was: %s" %(action))
                print("Change in reward was: %s" %(change_in_reward))
                print("Change in points was: %s" %(change_in_points))
                print("The predicted actions were: %s" %(predicted_actions))
                print("The updated actions are: %s" %(updated_actions))

                model.fit(state, updated_actions, epochs=number_of_epochs, batch_size=1)

        if not (piece.at_bottom() or piece.collision(board, type='bottom')):
            piece.fall()

        # Draw objects and blit texts
        piece.draw()
        next_piece.draw()
        board.draw()
        draw_text_on_next_piece(board_width * box_size + 80, 60)
        draw_text_on_points(board_width * box_size + 110, 160, points)
        draw_text_on_games_played(board_width * box_size + 110, 260, games_played)

        pygame.display.flip()
        clock.tick(fps)


class Piece(object):
    def __init__(self, offset=initial_offset):
        self._type = random.choice(tetrominos)
        self._rotation = 0
        self._number_of_rotations = tetromino_rotation_count[self.type]
        self._structure = tetromino_structures[self.type][self.rotation]
        self._length = len(self.structure[0])
        self._height = len(self.structure)
        self._offset = list(offset)
        self._color = tetromino_colors[self._type]

    # Copy attributes from another piece
    def copy_other_piece(self, other_piece):
        self.type = other_piece.type
        self.rotation = other_piece.rotation
        self._number_of_rotations = other_piece._number_of_rotations
        self.structure = other_piece.structure
        self.length = other_piece.length
        self.height = other_piece.height
        self.offset = list(initial_offset)
        self.color = other_piece.color

    def fall(self):
        if not self.at_bottom():
            self.offset[1] += 1

    # Return True if Piece is at bottom
    def at_bottom(self):
        if self.height + self.offset[1] >= board_height:
            return True
        return False

    # Return True if Piece is within the board for different kinds of movements and rotations
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
        self._width = 0
        self._height = 0
        self._area = 0

    # Add a piece to the board
    def add_piece(self, piece):
        if piece.within_board():
            listed_board = [list(self.structure[i]) for i in range(board_height)]
            listed_piece = [list(piece.structure[i]) for i in range(piece.height)]
            for y in range(piece.height):
                for x in range(piece.length):
                    if listed_piece[y][x] != blank:
                        listed_board[y + piece.offset[1]][x + piece.offset[0]] = listed_piece[y][x]
            self.structure = [''.join(listed_board[i]) for i in range(board_height)]

    # Refresh the board by checking if there are any complete rows that can be removed and update points
    def refresh(self, p=0, multiplier=1):
        points = p
        rows = [y for y in range(board_height) if blank not in self.structure[y]]
        if len(rows) > 0:
            points += len(rows) ** 2
            for index in sorted(rows, reverse=True):
                self.structure.pop(index)
            self.structure = [blank_row] * len(rows) + self.structure
        return multiplier * points

    # Define reward function for each state of the board (https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/)
    def reward_function(self, a=-0.51, b=-0.36, c=-0.18, multiplier=1):
        heights = [0 for i in range(board_width)]
        holes = [0 for i in range(board_width)]
        for i in range(board_height):
            for j in range(board_width):
                if heights[j] == 0:
                    if self.structure[i][j] != blank:
                        heights[j] = board_height - i
        sum_of_heights = sum(heights)
        for i in range(board_width):
            for j in range(board_height):
                if j >= board_height - heights[i]:
                    if self.structure[j][i] == blank:
                        holes[i] = holes[i] + 1
        number_of_holes = sum(holes)
        bumpiness = sum([abs(heights[i] - heights[i+1]) for i in range(board_width - 1)])
        return multiplier * (a * sum_of_heights + b * number_of_holes + c * bumpiness)

    # def measure(self):
    #     height = [board_height - i for i in range(board_height) if blank_row not in self.structure[i]]
    #     if height:
    #         self.height = height[0]
    #     else:
    #         self.height = 0

    #     widths = []
    #     for row in self.structure:
    #         if row != blank_row:
    #             left = re.search(r'[^.]', row).start()
    #             right = board_width - re.search(r'[^.]', row[::-1]).start()
    #             widths.append(right - left)
    #     if widths:
    #         self.width = max(widths)
    #     else:
    #         self.width = 0

    #     self.area = self.height * self.width

    def draw(self):
        for y in range(board_height):
            for x in range(board_width):
                if self.structure[y][x] != blank:
                    draw_rectangle(x, y, numeric_tetromino_colors[self.structure[y][x]])
        pygame.draw.line(display, white, (board_width * box_size, 0), (board_width * box_size, board_height * box_size), 1)

    # Return True if the game is over
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

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, x):
        self._height = x

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, x):
        self._width = x

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, x):
        self._area = x


# Returns a list of allowed actions
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


# Draws a rectangle at the specified place
def draw_rectangle(x, y, color, offset=[0, 0]):
    pygame.draw.rect(display, color, ((x + offset[0]) * box_size, (y + offset[1]) * box_size, box_size, box_size))
    pygame.draw.rect(display, black, ((x + offset[0]) * box_size, (y + offset[1]) * box_size, box_size, box_size), 1)


# Encodes the state of the Tetris board that can be fed to a NN model
def convert_to_input(board, piece):
    listed_board = [list(board.structure[i]) for i in range(board_height)]
    listed_piece = [list(piece.structure[i]) for i in range(piece.height)]
    for y in range(piece.height):
        for x in range(piece.length):
            try:
                if listed_piece[y][x] != blank:
                    listed_board[y + piece.offset[1]][x + piece.offset[0]] = -1
            except:
                pass
    for y in range(board_height):
        for x in range(board_width):
            if listed_board[y][x] != blank and listed_board[y][x] != -1:
                listed_board[y][x] = 1
            elif listed_board[y][x] == blank:
                listed_board[y][x] = 0
    return np.asarray(listed_board).reshape(-1, board_height, board_width, 1)


# Text about current points
def draw_text_on_points(x, y, points):
    text_current_points = my_font.render("Current points: %s" %(points), True, white)
    text_current_points_rect = text_current_points.get_rect()
    text_current_points_rect.center = (x, y)
    display.blit(text_current_points, text_current_points_rect)


# Text about the next piece
def draw_text_on_next_piece(x, y):
    text_next_piece = my_font.render("Next piece:", True, white)
    text_next_piece_rect = text_next_piece.get_rect()
    text_next_piece_rect.center = (x, y)
    display.blit(text_next_piece, text_next_piece_rect)


# Draw text about the number of games played
def draw_text_on_games_played(x, y, games_played):
    text_games_played = my_font.render("Games played: %s" %(games_played), True, white)
    text_games_played_rect = text_games_played.get_rect()
    text_games_played_rect.center = (x, y)
    display.blit(text_games_played, text_games_played_rect)


# The function below creates a Keras model
def create_model(learning_rate=0.005):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(board_height, board_width, 1), strides=(1, 1), padding='same', data_format='channels_last', activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    # model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(5, activation='linear'))

    model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD(lr=learning_rate), metrics=['accuracy'])

    return model


if __name__ == '__main__':
    game()
