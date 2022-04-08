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
exploration_rate_decay = 0.997
min_exploration_rate = 0.001

# Step size determines how much we update our current estimate of action value
start_step_size = 0.5
step_size_decay = 0.999
min_step_size = 0.1

# Discount factor
discount_factor = 0.95

# Buffer size determines the number of training observations required before model traning is initiated
buffer_size = 10000

# How many times we go over the training sample
epochs_of_training = 10

# Number of sample in a mini-batch used in training
batch_size = 100

# Number of training sessions to save progress to a text file
save_every = 50

# The penalty for losing a game
death_penalty = -100

points_multiplier = 10


def game():

    # Initiate options
    draw_option = 0
    print_option = 0
    pause = 0

    # Initialize variables that will be changing over time
    piece = Piece()
    board = Board()
    points = -7369142
    games_played = 0
    total_lines_cleared = 6217
    total_games_played = 72386
    exploration_rate = 0.001
    step_size = 0.1
    training_session = 3151

    agent = Agent().double()
    agent.load_state_dict(torch.load('agent_3150.pth'))
    optimizer = optim.Adam(agent.parameters(), lr=0.003)
    loss_func = nn.MSELoss()

    # Initialize lists that will contain training examples
    X = []
    y = []

    # Write all the hyperparameters to file
    with open('hyperparameters_2.txt', 'w') as f:
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
        f.write('\nPoints multiplier: {}'.format(points_multiplier))
        f.write('\nAgent architecture: {}'.format(agent))

    while True:
        display.fill(black)

        # Get initial states
        initial_points = points
        initial_max_height = get_max_height(board)
        initial_bumpiness = get_bumpiness(board)

        game_end = 0

        # Initialize to False
        random_action_taken = False

        # Get a list of allowed actions
        allowed_actions = get_allowed_actions(board, piece)

        # Choose a random action from the list of allowed actions
        random_action = np.random.choice(np.nonzero(allowed_actions)[0])

        allowed_actions = allowed_actions.reshape(1, -1)
        allowed_actions = torch.from_numpy(allowed_actions)

        # Get feature vector and make predictions
        feature_vector = get_feature_vector(board, piece)
        feature_vector = feature_vector.reshape(1, -1)
        predictions = agent(feature_vector)

        # Adjust predictions in accordance with allowed actions
        adjusted_action_values = adj_action_values(predictions, allowed_actions)

        # Take random action with probability equal to exploration_rate
        if np.random.uniform() < exploration_rate:
            action = random_action
            random_action_taken = True
        else:
            action = torch.argmax(adjusted_action_values).item()

        # Add feature vector to training sample
        X.append(feature_vector.squeeze())

        # Let agent take actions only when pause == 0
        if pause == 0:
            if action == 1:
                piece.offset[0] += 1
            elif action == 2:
                piece.offset[0] -= 1
            elif action == 3:
                piece.rotation += 1
            elif action == 4:
                piece.rotation -= 1

        # If the current piece is at bottom, adjust points and create a new Piece instance
        if piece.at_bottom() or piece.collision(board, type='bottom'):
            points += (initial_max_height - (board_height - piece.offset[1]))
            board.add_piece(piece)
            points += (initial_bumpiness - get_bumpiness(board))
            piece = Piece()

        # Adjust points earned and lines cleared
        points, lines_cleared = board.refresh(points, points_multiplier)
        total_lines_cleared += lines_cleared

        # Let piece fall only when pause == 0
        if pause == 0:
            if not piece.at_bottom() and not piece.collision(board, type='bottom'):
                piece.offset[1] += 1

        # If the current piece is at bottom, adjust points and create a new Piece instance
        if piece.at_bottom() or piece.collision(board, type='bottom'):
            points += (initial_max_height - (board_height - piece.offset[1]))
            board.add_piece(piece)
            points += (initial_bumpiness - get_bumpiness(board))
            piece = Piece()

        # Adjust points earned and lines cleared
        points, lines_cleared = board.refresh(points, points_multiplier)
        total_lines_cleared += lines_cleared

        # If game is over, adjust points, create new instance of Board and Piece objects
        if board.end_game():
            points += death_penalty
            piece = Piece()
            board = Board()
            games_played += 1
            total_games_played += 1
            game_end = 1

        # Get updated feature vector
        updated_feature_vector = get_feature_vector(board, piece)
        updated_feature_vector = updated_feature_vector.reshape(1, -1)

        # Get updated allowed actions
        updated_allowed_actions = get_allowed_actions(board, piece)
        updated_allowed_actions = updated_allowed_actions.reshape(1, -1)
        updated_allowed_actions = torch.from_numpy(updated_allowed_actions)

        # Get updated predictions from the agent
        updated_predictions = agent(updated_feature_vector)

        # Adjust updated predictions in accordance with allowed actions
        adjusted_updated_predictions = adj_action_values(updated_predictions, updated_allowed_actions)

        target_action_values = predictions.clone()

        if game_end == 0:
            target_action_values[0, action] = target_action_values[0, action] + step_size * (discount_factor * torch.max(adjusted_updated_predictions) + points - initial_points - target_action_values[0, action])
        else:
            target_action_values[0, action] = target_action_values[0, action] + step_size * (points - initial_points - target_action_values[0, action])

        y.append(target_action_values.detach().clone())

        # Initiate training when enough training samples is available
        if len(X) % buffer_size == 0:
            y = torch.cat(y, dim=0).type(torch.double)
            X = np.asarray(X)
            for e in range(epochs_of_training):
                # Randomly reshuffle the dataset
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

            # Reset training samples for the next training session
            X = []
            y = []

            # Update exploration rate and step size
            exploration_rate = max(min_exploration_rate, exploration_rate * exploration_rate_decay)
            step_size = max(min_step_size, step_size * step_size_decay)

            # Print results of the training session
            print('\nTraining_session {} is over.'.format(training_session))
            print('Games played this session: {}'.format(games_played))
            print('Total games played: {}'.format(total_games_played))
            print('Number of games per training session: {}'.format(total_games_played/training_session))
            print('Total points: {}'.format(points))
            print('Total lines cleared: {}'.format(total_lines_cleared))
            print('Lines cleared per game: {}'.format(total_lines_cleared / total_games_played))
            print('Current exploration rate is {}'.format(exploration_rate))
            print('Current step size is {}\n'.format(step_size))

            if training_session % save_every == 0:

                # Save current progress to file
                with open('training_session_{}.txt'.format(training_session), 'w') as f:
                    f.write('Total games played: {}'.format(total_games_played))
                    f.write('\nNumber of games per training session: {}'.format(total_games_played/training_session))
                    f.write('\nTotal points: {}'.format(points))
                    f.write('\nTotal lines cleared: {}'.format(total_lines_cleared))
                    f.write('\nLines cleared per game: {}'.format(total_lines_cleared / total_games_played))
                    f.write('\nCurrent exploration rate: {}'.format(exploration_rate))
                    f.write('\nCurrent step size: {}'.format(step_size))

                # Save trained model
                torch.save(agent.state_dict(), 'agent_{}.pth'.format(training_session))

            # Reset games played this training session to zero
            games_played = 0

            # Increment the number of training sessions
            training_session += 1

        # Draw only when draw_option == 1
        if draw_option == 1:
            piece.draw()
            board.draw()
            pygame.display.flip()
            clock.tick(fps)

        # Print training info only when print_option == 1
        if print_option == 1:
            print('\n\n')
            print('This is training_session {}'.format(training_session))
            print('Total games played: {}'.format(total_games_played))
            print('Number of games per training session: {}'.format(total_games_played/training_session))
            print('Game state: \n{}'.format(get_game_representation(board, piece)))
            print('Feature vector: \n{}'.format(feature_vector))
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
                    torch.save(agent.state_dict(), 'agent_{}.pth'.format(training_session - 1))


def get_allowed_actions(board, piece):
    '''
    Get a numpy.ndarray of allowed actions.

    Parameters
    __________
    board : an instance of Board object
    piece : an instance of Piece object

    Returns
    _______
    allowed_actions : numpy.ndarray with shape (5,) of allowed actions

    Examples
    ________
    >>> get_allowed_actions(board, piece)
    array([1, 1, 1, 0, 0])

    In this example, output can be interpreted as follows:
    - do nothing is allowed
    - moving the piece to the right is allowed
    - moving the piece to the left is allowed
    - rotating the piece clockwise is not allowed
    - rotating the piece counterclockwise is not allowed
    '''

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
    '''
    Returns a numpy.ndarray representing the state of the game, both tetrominos already placed on the board and the current falling tetromino. A value of 1 means that cell is not emppty.

    Parameters
    __________
    board: an instance of Board object
    piece: an instance of Piece object

    Returns
    _______
    game_representation: numpy.ndarray with shape (board_height, board_widht)

    Examples
    ________
    >>> get_game_representation(board, piece)
    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0.]])
    '''
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

    game_representation = np.asarray(listed_board).reshape(board_height, board_width)

    return game_representation


def get_board_reprensetation(board):
    '''
    The same function as "get_game_representation" but only returns the state of the board.

    Parameters
    __________
    board: an instance of Board object

    Returns
    _______
    board_representation: numpy.ndarray with shape (board_height, board_width)

    Examples
    ________
    >>> get_board_representation(board)
    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0.]])
    '''

    listed_board = [list(board.structure[i]) for i in range(board_height)]

    for y in range(board_height):
        for x in range(board_width):
            if listed_board[y][x] != blank:
                listed_board[y][x] = 1
            else:
                listed_board[y][x] = 0

    board_representation = np.asarray(listed_board).reshape(board_height, board_width)

    return board_representation


def get_column_heights(board):
    '''
    Returns a numpy.ndarray of column heights.

    Parameters
    __________
    board: an instance of Board object

    Returns
    _______
    column_heights: numpy.ndarray with shape (board_width,) of column heights

    Examples
    ________
    >>> get_column_heights(board_representation)
    array([ 0  0  0  0 15 16 18 19 20  4  4  0])
    '''

    board_representation = get_board_reprensetation(board)

    column_heights = []

    for i in range(board_representation.shape[1]):
        column_height = np.nonzero(board_representation[:, i])[0]
        if column_height.size == 1:
            column_height = int(column_height)
        elif column_height.size > 1:
            column_height = int(column_height[0])
        else:
            column_height = board_height
        column_heights.append(board_height - column_height)

    column_heights = np.asarray(column_heights)

    return column_heights


def get_max_height(board):
    '''
    Returns the maximum column height.

    Parameters
    __________
    board: an instance of Board object

    Returns
    _______
    max_height: int, the maximum height

    Examples
    ________
    >>> get_max_height(board)
    9
    '''

    column_heights = get_column_heights(board)
    max_height = np.max(column_heights)

    return max_height


def get_height_diffs(board):
    '''
    Returns a numpy.ndarray of differences in column heights.

    Parameters
    __________
    board: an instance of Board object

    Returns
    _______
    height_diffs: numpy.ndarray with shape (board_width  - 1,) of differences in heights between adjacent columns

    Examples
    ________
    >>> x = get_column_heights(board)
    >>> x
    array([9, 1, 6, 4, 6, 8, 5, 4, 1, 7, 0, 3])
    >>> get_height_diffs(x)

    '''

    column_heights = get_column_heights(board)
    height_diffs = np.diff(column_heights)

    return height_diffs


def get_bumpiness(board):
    height_diffs = get_height_diffs(board)
    bumpiness = np.abs(height_diffs).sum()

    return bumpiness


def get_feature_vector(board, piece, normalize=True):
    '''
    Returns a feature vector to be used as input to "forward" function of Agent object.

    Parameters
    __________
    board: an instant of Board object
    piece: an instance of Piece object
    normalize: bool. Determines whether the output should be normalized or not.

    Returns
    _______

    '''

    # Get column heights
    column_heights = get_column_heights(board)

    # Get max height
    max_height = np.asarray(np.max(column_heights)).reshape(1)

    height_diffs = np.diff(column_heights)

    piece_info = np.array([tetromino2int[piece.type], piece.rotation, piece.offset[0], piece.offset[1]])

    if normalize:
        normalized_column_heights = (column_heights - board_height / 2 + 0.5) / board_height
        normalized_height_diffs = (height_diffs - board_height / 2 + 0.5) /board_height
        normalized_max_height = (max_height - board_height / 2 + 0.5) / board_height
        normalized_piece_info = np.array([(tetromino2int[piece.type] - len(tetromino2int) + 0.5) / len(tetromino2int), (piece.rotation - 2.5) / 4, (piece.offset[0] - board_width / 2 + 0.5) / board_width, (piece.offset[1] - board_height / 2 + 0.5) / board_height])
        game_representation = np.concatenate((normalized_column_heights, normalized_height_diffs, normalized_max_height, normalized_piece_info))
    else:
        game_representation = np.concatenate((column_heights, height_diffs, max_height, piece_info))

    return game_representation


class Agent(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(in_features=28,
                             out_features=256)

        self.fc2 = nn.Linear(in_features=256,
                            out_features=128)

        self.output = nn.Linear(in_features=128,
                                out_features=5)

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

    def refresh(self, p=0, multiplier=1):
        points = p
        rows = [y for y in range(board_height) if blank not in self.structure[y]]
        if len(rows) > 0:
            points += (len(rows) * multiplier)
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
