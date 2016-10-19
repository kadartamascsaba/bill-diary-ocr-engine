import numpy as np

import theano
import theano.tensor as T


class LogRegLayer(object):
    '''
        This class represents the logistic regression layer in the neural network.
    '''

    def __init__(self, input_, n_in, n_out):

        self.input  = input_

        self.W      = None
        self.b      = None
        self.output = None

        rng = np.random.RandomState()

        # Creating the weight and bias vectors
        W_values = np.zeros((n_in, n_out), dtype=theano.config.floatX)
        b_values = np.zeros(n_out, dtype=theano.config.floatX)

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        # Calculating the output vector
        self.output = T.nnet.softmax(T.dot(self.input, self.W) + self.b)

    def setW(self, W, b):
        self.W = theano.shared(value=np.asarray(W, dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.asarray(b, dtype=theano.config.floatX), name='b', borrow=True)

        # Calculating the output vector
        self.output = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
