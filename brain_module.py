import numpy as np
import pandas as pd
import copy
import sys
import random
from mayavi import mlab


import importlib
"""
The below allows to reload a given module from the interactive shell
# importlib.reload(brain_module)
"""

import dill
"""
This module allows us to save and load objects. To load a previously saved object from Terminal:
# with open(filename, 'rb') as f:
#     brain = dill.load(f)
where filename is a string with '.pkl' extension
"""


class Brain(object):

    def __init__(self, structure, activation_function='relu', cost_function='entropic', x=None, y=None, color=None):
        '''
        The structural (defining) characteristics of a NN
        '''
        self.structure = structure
        self.L = len(structure)
        self.T = len(structure) - 1
        self.K = structure[-1]
        self.n = structure[0]
        self.biases = [np.random.randn(x, 1) for x in self.structure[1:]]
        self.weights = None
        self.weights_initializer()
        self.activation_function = activation_function
        self.cost_function = cost_function

        '''
        Data for learning
        '''
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.m = None
        self.m_training = None
        self.m_validation = None
        self.m_test = None

        '''
        Variables for learning
        '''
        self.learning_rate = None
        self.regularization_parameter = None
        self.batch_size = None
        self.number_of_epochs = None

        '''
        Forward and backpropagation variables
        '''
        self.z = None
        self.a = None
        self.errors = None
        self.gradients = None

        '''
        On validation data
        '''
        self.cost = None
        self.accuracy = None

        '''
        On test data
        '''
        self.expected_cost = None
        self.expected_accuracy = None

        '''
        For reinforcement learning problems
        '''
        self.state = 'alive'
        self.fitness = 0
        self.x = x
        self.y = y
        self.color = color

    def propagate_forward(self, data_set):
        """
        If the input is a simple python list of observations, we transform it to numpy array type. If it is already a numpy array type, there will be no change.
        """
        length = len(data_set)

        assert length == self.structure[0], 'ERROR: the length of the input list is not equal to the number of input neurons'

        data_set = np.reshape(data_set, (length, 1)).astype(float)

        self.z = []
        self.a = [data_set]

        for j in range(self.T):
            self.z.append(np.matmul(self.weights[j], self.a[j]) + self.biases[j])
            self.a.append(self.activate(self.z[j]))

        return self.a[-1]

    def propagate_backward(self, data_set):

        length = len(data_set)

        assert length == self.structure[-1], 'ERROR: the length of the input list is not equal to the number of input neurons'

        data_set = np.reshape(data_set, (length, 1)).astype(float)

        self.errors = []
        self.gradients = []

        if self.cost_function == 'quadratic':
            activation_gradient = self.a[-1] - data_set
            self.errors.append(np.multiply(activation_gradient, self.estimate_derivative(self.z[-1])))
        elif self.cost_function == 'entropic':
            self.errors.append(self.a[-1] - data_set)

        for j in range(self.T - 1):
            self.errors.append(np.multiply(np.matmul(np.transpose(self.weights[-1 - j]), self.errors[j]), self.estimate_derivative(self.z[-2 - j])))

        for j in range(self.T):
            self.gradients.append(np.matmul(self.errors[j], np.transpose(self.a[-2 - j])))

        for j in range(self.T):
            self.biases[j] = self.biases[j] - self.learning_rate * self.errors[-1 - j]
            self.weights[j] = self.weights[j] * (1 - self.learning_rate * self.regularization_parameter) - self.learning_rate * self.gradients[-1 - j]

    def estimate_accuracy(self, data):

        length = len(data)

        number_of_correct = 0

        for i in range(length):
            if np.argmax(self.propagate_forward(data[i][0])) == np.argmax(data[i][1]):
                number_of_correct += 1

        return number_of_correct * 100 / length

    def estimate_cost(self, data):

        length = len(data)

        cost = np.array(0)

        if self.cost_function == 'quadratic':
            for i in range(length):
                x = self.propagate_forward(data[i][0])
                y = data[i][-1]
                cost = cost + self.estimate_quadratic_cost(x, y)
                return cost
        elif self.cost_function == 'entropic':
            for i in range(length):
                x = self.propagate_forward(data[i][0])
                y = data[i][-1]
                cost = cost + self.estimate_entropic_cost(x, y)
                return cost

    def learn(self, training_data, validation_data, test_data, learning_rate=0.01, regularization_parameter=0.01, batch_size=10, number_of_epochs=10, save='off'):

        assert len(training_data) != 0, 'Please, provide training data'
        assert len(validation_data) != 0, 'Please, provide validation data'
        assert len(test_data) != 0, 'Please, provide test data'

        self.training_data = list(training_data)
        self.validation_data = list(validation_data)
        self.test_data = list(test_data)

        self.learning_rate = learning_rate
        self.regularization_parameter = regularization_parameter
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs

        self.m_training = len(self.training_data)
        self.m_validation = len(self.validation_data)
        self.m_test = len(self.test_data)
        self.m = self.m_training + self.m_validation + self.m_test

        print('Training data size is %s' % (self.m_training))
        print('Validation data size is %s' % (len(self.validation_data)))
        print('Test data size is %s' % (len(self.test_data)))

        highest_accuracy = 0

        for i in range(self.number_of_epochs):
            np.random.shuffle(self.training_data)
            batches = [self.training_data[k: k + self.batch_size] for k in range(0, self.m_training, self.batch_size)]
            for batch in batches:
                self.update_batch(batch)

            self.accuracy = self.estimate_accuracy(self.validation_data)
            self.cost = self.estimate_cost(self.validation_data)

            print('Epoch number %s is over. Accuracy on validation set is %s percent, cost on validation set is %s ' % (i + 1, self.accuracy, self.cost))

            if save == 'on':
                if self.accuracy > highest_accuracy:
                    self.expected_accuracy = self.estimate_accuracy(self.test_data)
                    self.expected_cost = self.estimate_cost(self.test_data)
                    self.save_as('trained_brain')
                    highest_accuracy = self.accuracy

        if save == 'on':
            print('The learning is over. The best-performing brain was saved. The expected accuracy on unseen data is %s. The expected cost on unseen data is %s' % (self.expected_accuracy, self.expected_cost))
        else:
            self.expected_accuracy = self.estimate_accuracy(self.test_data)
            print('The learning is over. The expected accuracy on unseen data is %s. The expected cost on unseen data is %s' % (self.expected_accuracy, self.expected_cost))

    def update_batch(self, data):

        batch_length = len(data)

        self.z = [[] for i in range(batch_length)]
        self.a = [[data[i][0]] for i in range(batch_length)]

        self.gradients = [[] for i in range(batch_length)]
        self.errors = [[] for i in range(batch_length)]

        for i in range(batch_length):
            for j in range(self.T):
                self.z[i].append(np.matmul(self.weights[j], self.a[i][j]) + self.biases[j])
                self.a[i].append(self.activate(self.z[i][j]))

        activation_gradient = []

        for i in range(batch_length):
            if self.cost_function == 'quadratic':
                activation_gradient.append(self.a[i][-1] - data[i][-1])
                self.errors[i].append(np.multiply(activation_gradient[i], self.estimate_derivative(self.z[i][-1])))
            elif self.cost_function == 'entropic':
                self.errors[i].append(self.a[i][-1] - data[i][-1])

        for i in range(batch_length):
            for j in range(self.T - 1):
                self.errors[i].append(np.multiply(np.matmul(np.transpose(self.weights[-1 - j]), self.errors[i][j]), self.estimate_derivative(self.z[i][-2 - j])))

        for i in range(batch_length):
            for j in range(self.T):
                self.gradients[i].append(np.matmul(self.errors[i][j], np.transpose(self.a[i][-2 - j])))

        for j in range(self.T):
            sum_of_biases = 0
            sum_of_weights = 0
            for i in range(batch_length):
                sum_of_biases = sum_of_biases + self.errors[i][-1 - j]
                sum_of_weights = sum_of_weights + self.gradients[i][-1 - j]
            self.biases[j] = self.biases[j] - self.learning_rate / batch_length * sum_of_biases
            self.weights[j] = self.weights[j] * (1 - self.learning_rate * self.regularization_parameter / batch_length) - self.learning_rate / batch_length * sum_of_weights

    def update_structure(self, structure):
        """
        This method allows us to update the structure of the brain within the brains itself. We re-initialize the weights and biases (and all the linked parameters of the brain)
        """
        assert type(structure) == tuple, 'ERROR: structure should be type tuple'
        self.structure = structure
        self.L = len(structure)
        self.T = len(structure) - 1
        self.K = structure[-1]
        self.n = structure[0]
        self.biases = [np.random.randn(x, 1) for x in self.structure[1:]]
        self.weights_initializer()

    def restart(self):
        """
        The below allows the network to 'forget' everything it learned previously
        """
        self.biases = [np.random.randn(x, 1) for x in self.structure[1:]]
        self.weights_initializer()

    def weights_initializer(self):
        """
        The below code initializes the weights based on the number of newurons in each respective layer
        """
        self.weights = [np.random.normal(0, 1 / np.sqrt(x), (x, y)) for x, y in list(zip(self.structure[1:], self.structure[:-1]))]

    def save_as(self, filename):
        """
        Allows us to save a trained Brain object to file for later use
        """
        assert type(filename) == str, 'ERROR: filename should be type str'
        if '.pkl' in filename:
            with open(filename, 'wb') as f:
                dill.dump(self, f)
        else:
            with open(filename + '.pkl', 'wb') as f:
                dill.dump(self, f)

    def estimate_quadratic_cost(self, x, y):
        return np.sum(np.power(y - x, 2) / 2)

    def estimate_entropic_cost(self, x, y):
        return np.sum(-y * np.log(x) - (1 - y) * np.log(1 - x))

    def activate(self, x):
        if self.activation_function == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_function == 'relu':
            return self.relu(x)
        elif self.activation_function == 'tanh':
            return self.tanh(x)

    def estimate_derivative(self, x):
        if self.activation_function == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation_function == 'relu':
            return self.relu_derivative(x)
        elif self.activation_function == 'tanh':
            return self.tanh_derivative(x)

    def sigmoid(self, x):
        return np.divide(1, (1 + np.exp(-x)))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def visualize(self, point=(0, 0, 0)):

        assert type(point) == tuple, 'Parameter point should be type tuple'

        black = (0, 0, 0)
        white = (1, 1, 1)

        radius = 10

        x = [[] for i in range(self.L)]
        y = [[] for i in range(self.L)]
        z = [[] for i in range(self.L)]

        factor_decomposition = []
        for j in range(self.L):
            factor_decomposition.append(get_factors(self.structure[j]))

        for j in range(self.L):
            for i in range(factor_decomposition[j][1]):
                for k in range(factor_decomposition[j][0]):
                    y[j].append(k * radius)
                    z[j].append(i * radius)
                    x[j].append(j * radius * 4)

        # print(x[0])
        # print(y[0])
        # print(z[0])

        for j in range(self.L):
            mlab.points3d(x[j], y[j], z[j], resolution=720)

        mlab.show()

    def draw(self):
        """
        This function is handy in pygame when we are drawing our object on the screen. Should be modified depending on the situation
        """
        pass

    def move(self):
        """
        This function is handy in pygame when we are drawing our object on the screen. Should be modified depending on the situation
        """
        pass

    """
    Below we are introducing some functions related to neuro-evolution
    """

    def copy(self):
        """
        This function allows us to create a copy of the brain
        """
        brain = copy.deepcopy(self)
        return brain

    def mutate(self, probability, rate):
        """
        This is very similar to the mutate function below but instead of giving a completely new weights or bias we are adding an increment which depends on the rate. The probability argument controls the chances that a given weight or bias mutates (as usual)
        """
        for i in range(self.T):
            shape = np.shape(self.weights[i])
            weights = self.weights[i].flatten()
            for j in range(len(weights)):
                if np.random.uniform(0, 1) < probability:
                    weights[j] = weights[j] + rate * np.random.normal(0, 1 / np.sqrt(shape[0]))
            self.weights[i] = weights.reshape(shape)
            for j in range(len(self.biases[i])):
                if np.random.uniform(0, 1) < probability:
                    self.biases[i][j] = self.biases[i][j] + rate * np.random.normal(0, 1)


def crossover(obj1, obj2):
    """
    This function takes two Brain objects as inputs and returns another Brain object with weights and biases taken from the parent objects randomly
    """

    assert obj1.structure == obj2.structure, 'The structures of the two brains are different'
    assert obj1.activation_function == obj2.activation_function, 'The activation functions of the two brains are different'
    assert obj1.cost_function == obj2.cost_function, 'The cost functions of the two brains are different'

    new_brain = Brain((obj1.structure), activation_function=obj1.activation_function, cost_function=obj1.cost_function)

    for i in range(obj1.T):
        shape = obj1.weights[i].shape
        weights1 = obj1.weights[i].flatten()
        weights2 = obj2.weights[i].flatten()
        biases1 = obj1.biases[i]
        biases2 = obj2.biases[i]
        weights_combined = []
        biases_combined = []
        for j in range(len(weights1)):
            if np.random.uniform(0, 1) < 0.5:
                weights_combined.append(weights1[j])
            else:
                weights_combined.append(weights2[j])
        for j in range(len(biases1)):
            if np.random.uniform(0, 1) < 0.5:
                biases_combined.append(biases1[j])
            else:
                biases_combined.append(biases2[j])
        new_brain.weights[i] = np.asarray(weights_combined).reshape(shape)
        new_brain.biases[i] = np.asarray(biases_combined)

    return new_brain


class Population(object):

    def __init__(self, size, structure, activation_function='relu', cost_function='entropic'):
        self.size = size
        self.members = [Brain(structure, activation_function=activation_function, cost_function=cost_function) for i in range(self.size)]
        self.state = 'alive'
        self.member_states = None
        self.number_of_members_alive = None
        self.total_population_fitness = None
        self.member_fitness = None
        self.mating_pool = None
        self.children = None
        self.generation = 1
        self.fittest_brain = None

    def check_state(self):
        self.member_states = [self.members[i].state for i in range(self.size)]
        self.number_of_members_alive = self.member_states.count('alive')
        if 'alive' not in self.member_states:
            self.state = 'dead'

    def learn(self, training_data, validation_data, test_data, number_of_epochs=100, probability=0.05, rate=0.5, save='off', elitism='on'):

        for i in range(number_of_epochs):
            self.members[i].fitness = self.members[i].estimate_accuracy(training_data)

            self.member_fitness = [self.members[i].fitness for i in range(self.size)]

            self.fittest_brain = self.members[self.member_fitness.index(max(self.member_fitness))].copy()

            if save == 'on':
                self.fittest_brain.expected_accuracy = self.fittest_brain.estimate_accuracy(test_data)
                self.fittest_brain.expected_cost = self.fittest_brain.estimate_cost(test_data)
                self.fittest_brain.save_as('fittest_brain')

            self.total_population_fitness = sum(self.member_fitness)

            self.mating_pool = [[self.members[i]] * round(self.member_fitness[i] * 1000 / self.total_population_fitness) for i in range(self.size)]

            self.mating_pool = [brain for sublist in self.mating_pool for brain in sublist]

            self.children = []

            if elitism == 'on':

                self.children.append(self.fittest_brain)

                for i in range(self.size - 1):
                    parent1 = random.choice(self.mating_pool)
                    parent2 = random.choice(self.mating_pool)
                    child = crossover(parent1, parent2)
                    child.mutate(probability, rate)
                    self.children.append(child)
            else:
                for i in range(self.size):
                    parent1 = random.choice(self.mating_pool)
                    parent2 = random.choice(self.mating_pool)
                    child = crossover(parent1, parent2)
                    child.mutate(probability, rate)
                    self.children.append(child)

            self.members = self.children

            print('Total population fitness is %s' % (self.total_population_fitness))

            print('Average population fitness is %s' % (self.total_population_fitness / self.size * 100))

            print('Accuracy of the fittest brain is %s percent' % (self.fittest_brain.estimate_accuracy(training_data)))

            self.generation += 1

    def evolve(self, elitism='on', save='off', probability=0.05, rate=0.05):
        """
        This code allows our population to evolve and get better by means of crossover and mutation. If parameter 'elitism' is 'on', we will be keeping the best overall performer (not necessarily from the current generation)
        """
        if self.state == 'dead':

            self.member_fitness = [self.members[i].fitness for i in range(self.size)]

            self.fittest_brain = self.members[self.member_fitness.index(max(self.member_fitness))]

            if save == 'on':
                self.fittest_brain.save_as('fittest_brain')

            self.total_population_fitness = sum(self.member_fitness)

            print('Total population fitness is %s' % (self.total_population_fitness))

            self.mating_pool = [[self.members[i]] * round(self.member_fitness[i] * 1000 / self.total_population_fitness) for i in range(self.size)]

            self.mating_pool = [brain for sublist in self.mating_pool for brain in sublist]

            self.children = []

            if elitism == 'on':

                self.children.append(self.fittest_brain)

                for i in range(self.size - 1):
                    parent1 = random.choice(self.mating_pool)
                    parent2 = random.choice(self.mating_pool)
                    child = crossover(parent1, parent2)
                    child.mutate(probability, rate)
                    self.children.append(child)
            else:
                for i in range(self.size):
                    parent1 = random.choice(self.mating_pool)
                    parent2 = random.choice(self.mating_pool)
                    child = crossover(parent1, parent2)
                    child.mutate(probability, rate)
                    self.children.append(child)

            self.members = self.children

            '''
            We need to set the state of the first bird to alive as it has been taken from the previous generation and thus its state is dead from the previous generation
            '''
            self.members[0].state = 'alive'

            self.state = 'alive'
            self.generation += 1


def get_factors(n):
    i = int(n**0.5 + 0.5)
    while n % i != 0:
        i -= 1
    return (int(n / i), i)


"""
Below is an example of data transformation that is an input to 'learn' method of 'Brain' object. The example is from the 'Iris' dataset from UCI repository

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv(url, names=names)

The steps below transform the last column to numerical values

for i in range(len(data)):
    if data.loc[i, 'class'] == 'Iris-setosa':
        data.loc[i, 'class'] = 0
    elif data.loc[i, 'class'] == 'Iris-versicolor':
        data.loc[i, 'class'] = 1
    else:
        data.loc[i, 'class'] = 2

We transform the pandas version of numpy version

numeric_data = data.values

Next, split the features x from the labels y

x = numeric_data[:, :-1]
y = numeric_data[:, -1].reshape(len(numeric_data), 1)
output = []
predictions = []
for i in range(len(y)):
    z = y[i].astype(int)
    output.append(np.full((3, 1), 0))
    output[i][z] = 1
    predictions.append(x[i].reshape(4, 1))

combined = tuple(zip(predictions, output))
"""
