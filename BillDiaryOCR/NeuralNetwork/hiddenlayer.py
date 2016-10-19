import numpy as np

import theano
import theano.tensor as T

class HiddenLayer(object):
    '''
        This class represents a hidden layer in the neural network.
        The layer uses the tanh activation function, because it usually yields to faster learning.
    '''

    def __init__(self, input_, n_in, n_out):

        self.input  = input_
        self.n_in   = n_in
        self.n_out  = n_out

        self.W      = None
        self.b      = None
        self.output = None

        random_state = np.random.RandomState()

        # Creating the weight vector
        W_values = np.asarray(
                    random_state.uniform(
                        low=-np.sqrt(6. / (n_in + n_out)),
                        high=np.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
                   )
          
        # Creating the bias vector
        b_values = np.zeros(n_out, dtype=theano.config.floatX)
            

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        # Creating the output vector using the tanh activation function
        self.output = T.tanh(T.dot(self.input, self.W) + self.b)  

    def setW(self, W, b):
        self.W = theano.shared(value=np.asarray(W, dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.asarray(b, dtype=theano.config.floatX), name='b', borrow=True)

        self.output = T.tanh(T.dot(self.input, self.W) + self.b)
