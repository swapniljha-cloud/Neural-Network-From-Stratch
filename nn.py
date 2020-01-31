import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def mse_loss(y_actual, y_predicted):
    '''takes in the actual value and takes away the computer prediction),
    then squared through *2)
    then dividing the number of times it needs to do this.'''
    return ((y_actual - y_predicted) ** 2).mean()


class MyNeuralNetwork(object):
    '''
    My Neural network'
    - two inputs
    - one hidden layer with 2 neurons(h1,h2)
    -one output layer with 1 neuron.
    '''

    def __init__(self):
        '''here we are creating an object of 6 weights
        each line is a random wight being assigned'''
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        '''this function is taking an object of itself and x (input)
        we have an input of height and weight, for a forward pass we
        pass inputs to the network. '''
        h1 = sigmoid((self.w1 * x[0]) + (self.w2 * x[1]) + (self.b1))
        h2 = sigmoid((self.w3 * x[0]) + (self.w4 * x[1]) + (self.b2))
        o1 = sigmoid((self.w5 * H1) + (self.w6 * h2) + (self.b3))


# Define Dataset
data = np.array([
    [-2, -1],  # alice
    [25, 6],  # Bob
    [17, 4],  # Charlie
    [15, -6]  # Diana
])

# The actual Values
all_y_trues = np.array([
    1,  # Alice- Female
    0,  # Bob- Male
    0,  # Charlie - Male
    1,  # Diana - Female
])


def train(self, data, all_y_trues):
    '''-Data is a (n* 2(weight & height)) numpy array, n = number of samples in the dataset.
    - all Y true values is a numpy array with n elements.
    elements in all_y_true values correspond to those in data
    '''
    learning_rate = 0.1
    epochs = 1000  # number of time to loop through the entire dataset

    for epoch in range(epochs):
        for x, y_true in zip(data, all_y_trues):
            ##do a forwardpass
            sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
            hiddenlayer1 = sigmoid(sum_h1)

            sum_h2 = (self.w3 * x[0]) + (self.w4 * x[1]) + (self.b2)
            hiddenlayer2 = sigmoid(sum_h2)

            sum_o1 = (self.w5 * hiddenlayer1) + (self.w6 * hiddenlayer2) + (self.b3)
            o1 = sigmoid(sum_o1)

            y_pred = o1

# Creating an object/Instance of our neural Network
network = OurNeuralNetwork()
