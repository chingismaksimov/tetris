# Import libraries
import pygame
from pygame.locals import *
import sys
import numpy as np
import random
import torch
from torch import nn, optim
from torch.nn import functional as F


pygame.init()

fps = 100

# Board size in rectangles
board_width = 12
board_height = 24

# Where the initial tetromino spawns (in rectangles)
initial_offset = (6, 0)

# Entire display size in rectangles
display_width = 30

# Rectangle size in pixels
rectangle_size = 20

# Auxiliary variables
blank = '.'
blank_row = blank * board_width

# Colors
black = (0, 0, 0)
white = (255, 255, 255)

# Display, clock and font objects
display = pygame.display.set_mode((display_width * rectangle_size, board_height * rectangle_size))
pygame.display.set_caption("Tetris")
clock = pygame.time.Clock()
my_font = pygame.font.SysFont("'monospace'", 30)

# List of tetrominos
tetrominos = ['T', 'I', 'J', 'L', 'O', 'S', 'Z']

# Dictionary that maps tetrominos to integers
tetromino2int = dict([(tetrominos[i], i) for i in range(len(tetrominos))])

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

# Number of rotations that each tetromino can make
tetromino_rotation_count = {

    'T': 4,
    'I': 2,
    'J': 4,
    'L': 4,
    'O': 1,
    'S': 2,
    'Z': 2

}


# Exploration rate determines the probability of taking a random action
start_exploration_rate = 0.5
exploration_rate_decay = 0.97
min_exploration_rate = 0.01

# Step size determines how much we update our current estimate of action value
start_step_size = 0.5
step_size_decay = 0.97
min_step_size = 0.1

# Discount factor
discount_factor = 0.95

# Buffer size determines the number of training observations required before model traning is initiated
buffer_size = 20000

# How many times we go over the training sample
epochs_of_training = 1

# Number of sample in a mini-batch used in training
batch_size = 512

# Number of training sessions to save progress to a text file
save_every = 50

# The penalty for losing a game
death_penalty = -10


def game():

    # Initiate options
    draw_option = 0
    print_option = 0
    pause = 0

    # Initialize parameters that will be changing over time
    piece = Piece()
    board = Board()
    points = 0
    total_lines_cleared = 0
    total_games_played = 0
    exploration_rate = start_exploration_rate
    step_size = start_step_size
    training_session = 1

    agent = Agent().double()
    # agent.load_state_dict(torch.load('agent_2213.pth'))
    optimizer = optim.Adam(agent.parameters(), lr=0.003)
    loss_func = nn.MSELoss()

    # Initialize lists that will contain training examples
    X = []
    y = []

    # Write all the hyperparameters to file
    with open('hyperparameters.txt', 'w') as f:
        f.write('Starting training session: {}'.format(training_session))
        f.write('\nDiscount factor: {}'.format(discount_factor))
        f.write('\nStarting exploration rate: {}'.format(start_exploration_rate))
        f.write('\nExploration rate decay: {}'.format(exploration_rate_decay))
        f.write('\nMin exploration rate: {}'.format(min_exploration_rate))
        f.write('\nStarting step size: {}'.format(start_step_size))
        f.write('\nStep size decay: {}'.format(step_size_decay))
        f.write('\nMin step size: {}'.format(min_step_size))
        f.write('\nBuffer size: {}'.format(buffer_size))
        f.write('\nNumber of epochs of training: {}'.format(epochs_of_training))
        f.write('\nBatch size: {}'.format(batch_size))
        f.write('\nDeath penalty: {}'.format(death_penalty))
        f.write('\nAgent architecture: {}'.format(agent))

    while True:
        display.fill(black)

        initial_points = points

        game_end = 0

        random_action_taken = False

        allowed_actions = get_allowed_actions(board, piece)

        random_action = np.random.choice(np.nonzero(allowed_actions)[0])

        allowed_actions = allowed_actions.reshape(1, -1)
        allowed_actions = torch.from_numpy(allowed_actions)

        board_representation = get_game_representation2(board, piece)
        board_representation = board_representation.reshape(1, -1)
        predictions = agent(board_representation)

        adjusted_action_values = adj_action_values(predictions, allowed_actions)

        if np.random.uniform() < exploration_rate:
            action = random_action
            random_action_taken = True
        else:
            action = torch.argmax(adjusted_action_values).item()

        X.append(board_representation.squeeze())

        if pause == 0:
            if action == 1:
                piece.offset[0] += 1
            elif action == 2:
                piece.offset[0] -= 1
            elif action == 3:
                piece.rotation += 1
            elif action == 4:
                piece.rotation -= 1

        if piece.at_bottom() or piece.collision(board, type='bottom'):
            board.add_piece(piece)
            piece = Piece()

        points, lines_cleared = board.refresh(points)
        total_lines_cleared += lines_cleared

        if pause == 0:
            if not piece.at_bottom() and not piece.collision(board, type='bottom'):
                piece.offset[1] += 1

        if piece.at_bottom() or piece.collision(board, type='bottom'):
            board.add_piece(piece)
            piece = Piece()

        points, lines_cleared = board.refresh(points)
        total_lines_cleared += lines_cleared

        if board.end_game():
            points += death_penalty
            piece = Piece()
            board = Board()
            total_games_played += 1
            game_end = 1

        updated_board_representation = get_game_representation2(board, piece)
        updated_board_representation = updated_board_representation.reshape(1, -1)

        updated_allowed_actions = get_allowed_actions(board, piece)
        updated_allowed_actions = updated_allowed_actions.reshape(1, -1)
        updated_allowed_actions = torch.from_numpy(updated_allowed_actions)

        updated_predictions = agent(updated_board_representation)

        adjusted_updated_predictions = adj_action_values(updated_predictions, updated_allowed_actions)

        target_action_values = predictions.clone()

        if game_end == 0:
            target_action_values[0, action] = target_action_values[0, action] + step_size * (discount_factor * torch.max(adjusted_updated_predictions) + points - initial_points - target_action_values[0, action])
        else:
            target_action_values[0, action] = target_action_values[0, action] + step_size * (points - initial_points - target_action_values[0, action])

        y.append(target_action_values.detach().clone())

        if len(X) % buffer_size == 0:
            y = torch.cat(y, dim=0).type(torch.double)
            X = np.asarray(X)
            for e in range(epochs_of_training):
                random_indices = np.arange(X.shape[0])
                np.random.shuffle(random_indices)
                X = X[random_indices]
                y = y[random_indices]
                for i in range(int(len(X) / batch_size)):
                    agent.zero_grad()
                    predictions = agent(X[i * batch_size:(i+1) * batch_size])
                    loss = loss_func(predictions, y[i * batch_size: (i+1) * batch_size])
                    loss.backward()
                    optimizer.step()

            X = []
            y = []

            exploration_rate = max(min_exploration_rate, exploration_rate * exploration_rate_decay)
            step_size = max(min_step_size, step_size * step_size_decay)

            print('\nTraining_session {} is over.'.format(training_session))
            print('Total games played: {}'.format(total_games_played))
            print('Total points: {}'.format(points))
            print('Total lines cleared: {}'.format(total_lines_cleared))
            print('Lines cleared per game: {}'.format(total_lines_cleared / total_games_played))
            print('Current exploration rate is {}'.format(exploration_rate))
            print('Current step size is {}\n'.format(step_size))

            if training_session % save_every == 0:
                with open('training_session_{}.txt'.format(training_session), 'w') as f:
                    f.write('Total games played: {}'.format(total_games_played))
                    f.write('\nTotal points: {}'.format(points))
                    f.write('\nTotal lines cleared: {}'.format(total_lines_cleared))
                    f.write('\nLines cleared per game: {}'.format(total_lines_cleared / total_games_played))
                    f.write('\nCurrent exploration rate: {}'.format(exploration_rate))
                    f.write('\nCurrent step size: {}'.format(step_size))

            training_session += 1

        if draw_option == 1:
            piece.draw()
            board.draw()
            pygame.display.flip()
            clock.tick(fps)

        if print_option == 1:
            print('\n\n')
            print('This is training_session {}'.format(training_session))
            print('Total games played: {}'.format(total_games_played))
            print('Game state: \n{}'.format(get_game_representation(board, piece)))
            print('Feature vector: \n{}'.format(board_representation))
            print('Allowed actions: {}'.format(allowed_actions))
            print('Action values: \n{}'.format(adjusted_action_values))
            print('Action taken: {}'.format(action))
            print('Random action taken: {}'.format(random_action_taken))
            print('Total points: {}'.format(points))
            print('Change in points: {}'.format(points-initial_points))
            print('Target action values: \n{}'.format(target_action_values))
            print('Total lines cleared: {}'.format(total_lines_cleared))
            print('Lines cleared per game: {}'.format(total_lines_cleared / total_games_played))
            print('Current exploration rate is {}'.format(exploration_rate))
            print('Current step size is {}'.format(step_size))

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
                elif event.key == K_UP:
                    piece.offset[1] -= 1
                elif event.key == K_RIGHT and not piece.collision(board, type='right') and piece.within_board(type='right'):
                    piece.offset[0] += 1
                elif event.key == K_LEFT and not piece.collision(board, type='left') and piece.within_board(type='left'):
                    piece.offset[0] -= 1
                elif event.key == K_q and not piece.collision(board, type='clockwise') and piece.within_board(type='clockwise'):
                    piece.rotation += 1
                elif event.key == K_e and not piece.collision(board, type='counterclockwise') and piece.within_board(type='counterclockwise'):
                    piece.rotation -= 1
                elif event.key == K_d:
                    draw_option = (draw_option + 1) % 2
                elif event.key == K_p:
                    print_option = (print_option + 1) % 2
                elif event.key == K_SPACE:
                    pause = (pause + 1) % 2
                elif event.key == K_s:
                    torch.save(agent.state_dict(), 'agent_{}.pth'.format(training_session))


def get_allowed_actions(board, piece):
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


def get_game_representation(board, piece):
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


def get_column_heights(board_numeric_representation):
    assert type(board_numeric_representation) == np.ndarray, 'Input type should be numpy.ndarray'
    column_heights = []
    for i in range(board_numeric_representation.shape[1]):
        height = np.nonzero(board_numeric_representation[:, i])[0]
        if height.size == 1:
            height = int(height)
        elif height.size > 1:
            height = int(height[0])
        else:
            height = board_height
        column_heights.append(board_height - height)
    return np.asarray(column_heights)


def get_game_representation2(board, piece):
    listed_board = [list(board.structure[i]) for i in range(board_height)]
    for y in range(board_height):
        for x in range(board_width):
            if listed_board[y][x] != blank:
                listed_board[y][x] = 1
            else:
                listed_board[y][x] = 0

    board_column_heights = np.asarray(listed_board).reshape(board_height, board_width)
    board_column_heights = get_column_heights(board_column_heights)

    bumpiness = np.abs(np.diff(board_column_heights)).sum().reshape(1)

    piece_info = np.array([tetromino2int[piece.type], piece.rotation, piece.offset[0], piece.offset[1]])

    game_representation = np.concatenate((board_column_heights, bumpiness, piece_info))

    return game_representation


# class Agent(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv_1 = nn.Conv2d(in_channels=1,
#                                 out_channels=16,
#                                 kernel_size=3,
#                                 stride=1,
#                                 padding=1)

#         self.conv_2 = nn.Conv2d(in_channels=16,
#                                 out_channels=32,
#                                 kernel_size=3,
#                                 stride=1,
#                                 padding=1)

#         self.flatten = nn.Flatten()

#         self.fc = nn.Linear(in_features=9216,
#                             out_features=512)

#         self.output = nn.Linear(in_features=512,
#                                 out_features=5)

#     def forward(self, x):
#         if not torch.is_tensor(x):
#             x = torch.from_numpy(x).type(torch.double)

#         x = F.relu(self.conv_1(x))
#         x = F.relu(self.conv_2(x))
#         x = self.flatten(x)
#         x = F.relu(self.fc(x))
#         x = self.output(x)

#         return x


class Agent(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(in_features=17,
                             out_features=32)
        # self.fc1.weight = nn.Parameter(torch.ones(self.fc1.weight.shape)/100)
        # self.fc1.bias = nn.Parameter(torch.ones(self.fc1.bias.shape)/100)
        # assert self.fc1.weight.requires_grad
        # assert self.fc1.bias.requires_grad

        self.fc2 = nn.Linear(in_features=32,
                            out_features=32)
        # self.fc2.weight = nn.Parameter(torch.ones(self.fc2.weight.shape)/100)
        # self.fc2.bias = nn.Parameter(torch.ones(self.fc2.bias.shape)/100)
        # assert self.fc2.weight.requires_grad
        # assert self.fc2.bias.requires_grad

        self.output = nn.Linear(in_features=32,
                                out_features=5)
        # self.output.weight = nn.Parameter(torch.ones(self.output.weight.shape)/100)
        # self.output.bias = nn.Parameter(torch.ones(self.output.bias.shape)/100)
        # assert self.output.weight.requires_grad
        # assert self.output.bias.requires_grad

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).type(torch.double)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)

        return x


def adj_action_values(predictions, allowed_actions):
    assert predictions.shape == allowed_actions.shape, 'The two inputs must have the same shape'

    z = torch.zeros(predictions.shape)

    for i in range(predictions.shape[1]):
        if allowed_actions[0, i] == 0:
            z[0, i] = -torch.inf
        else:
            z[0, i] = predictions[0, i]

    return z


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
        return points, len(rows)

    def draw(self):
        for y in range(board_height):
            for x in range(board_width):
                if self.structure[y][x] != blank:
                    draw_rectangle(x, y, numeric_tetromino_colors[self.structure[y][x]])
        pygame.draw.line(display, white, (board_width * rectangle_size, 0), (board_width * rectangle_size, board_height * rectangle_size), 1)

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


def draw_rectangle(x, y, color, offset=[0, 0]):
    pygame.draw.rect(display, color, ((x + offset[0]) * rectangle_size, (y + offset[1]) * rectangle_size, rectangle_size, rectangle_size))
    pygame.draw.rect(display, black, ((x + offset[0]) * rectangle_size, (y + offset[1]) * rectangle_size, rectangle_size, rectangle_size), 1)


if __name__ == '__main__':
    game()
