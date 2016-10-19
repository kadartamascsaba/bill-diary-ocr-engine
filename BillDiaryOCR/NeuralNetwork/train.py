# This is a training script, which training a deep neural network for OCR problem
#
# Based on the tutorial here: 	http://deeplearning.net/tutorial/mlp.html
#

import os
import sys
import time
import random

from os import path
from os.path import isfile, join, isdir

import cv2
import numpy
import theano
import neuralnetwork as nn


def load_train_data(train_dir, classes):

    NUMBER_OF_CHARACTERS = 200

    train_input = [f for f in os.listdir(train_dir) if isdir(join(train_dir, f))]

    train_set = []

    for character in train_input:
        path = join(train_dir, character)
        images = [f for f in os.listdir(path) if isfile(join(path, f))]
        for i in xrange(NUMBER_OF_CHARACTERS):
            if len(images) == 0:
                break
            image_name = random.choice(images)
            im = cv2.imread(join(path, image_name))
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im = im.flatten()

            try:
                index = classes.index(character)
            except:
                print "ERROR: Could not find this class: " + character
                exit(0)

            train_set.append(_make_array(im, index))

    return numpy.random.permutation(train_set)

def load_data(train_dir, test_dir):

    train_input = [f for f in os.listdir(train_dir) if isdir(join(train_dir, f))]
    test_input  = [f for f in os.listdir(test_dir)  if isdir(join(test_dir, f))]

    classes   = []
    train_set = []
    test_set  = []

    # Training data
    for element in train_input:
        try:
            index = classes.index(element)
        except:
            index = len(classes)
            classes.append(element)

    train_set = load_train_data(train_dir, classes)

    # Test data
    for element in test_input:
        path = join(test_dir, element)
        test_img = [f for f in os.listdir(path) if isfile(join(path, f))]

        for img in test_img:
            im = cv2.imread(join(path, img))
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            
            test_set.append((numpy.asarray(im.flatten(), dtype=theano.config.floatX), element))

    return numpy.random.permutation(train_set), test_set, classes

def _make_array(x, y):
    return (
        numpy.asarray(x, dtype=theano.config.floatX),
        numpy.asarray(y, dtype='int32')
    )

def main(train_dir='../train', test_dir='../test'):

    print '... loading data'        
    train_set, test_set, classes = load_data(train_dir, test_dir)

    print '... building the model'
    n = nn.Net(learning_rate=0.00001, classes=classes, L2_reg=0.0001)
    n.add_hidden_layer(1024, 1800)
    n.add_hidden_layer(1800, 1200)
    n.add_hidden_layer(1200,  800)
    n.add_hidden_layer( 800,  600)

    print '... compiling the model'
    n.compile_model()
    current_error = 1
    error         = 1

    # We train the network 150 times
    # Each time we evaluate the results and write out the error accuracy
    for epoch in range(1, 151):

        if error < current_error:
            current_error = error
            print 'Saving matrix ...'
            n.save()
            print 'Save completed ...'

        z = time.time()

        print '... training'            
        for x in train_set:
            n.train_model(x[0], x[1])

        print 'training took {}'.format(time.time()-z)
        train_set = load_train_data(train_dir, classes)

        # Calulating error
        print '... calculating error'            
        error = numpy.mean([n.devtest_model(x[0], classes.index(x[1])) for x in test_set])
        print('epoch %i, validation error %f %%' % (epoch, error * 100))

if __name__ == '__main__':
    main()
