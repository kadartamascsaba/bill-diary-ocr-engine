import os
import sys
import cv2
import numpy
import random
from os import path
from os.path import isfile, join, isdir

from libsvm.svm import *
from libsvm.svmutil import *
from itertools import product, chain

def cross_validation(xdata, ydata, svm_type, nfold, **kwargs):
    """
        Cross validation of libsvm parameters.
        **kwargs: contains the CV values for the parameters, e.g.: {'c': [0.01, 0.5, 1, 100], 'w-1':[1, 10, 100], 'w1':[1, 10], ...}

        Returns the accuracy results in the following format:
        [[acc_1, param_string_1], [acc_2, param_string_2], ...]
    """
    
    prob  = svm_problem(ydata, xdata)
    
    p = '-s 0 -t {} -v {}'.format(svm_type, nfold)
    
    ks = kwargs.keys()
    vs = [kwargs[k] for k in ks]
    pTuples = list(product(*vs)) # The Cross validation parameters in tuples 
    k0 = [' -']
    k2 = [' {}']
    list1 = list(product(k0, ks, k2))
    list2 = list(chain(*list1)) # linearized list 
    p2 = ''.join(list2) # Format string for the Cross validation parameters

    res = [] # Cross validation results
    
    for par in pTuples:
        pFinal = (p + p2).format(*par)
        param = svm_parameter(pFinal)
        acc = svm_train(prob, param)
        res.append([acc, p2.format(*par)])
    
    return res

def load_train_data(train_dir):

    train_input = [f for f in os.listdir(train_dir) if isdir(join(train_dir, f))]

    classes   = []
    train_set = []

    # Training data
    for element in train_input:

        t_directory = join(train_dir, element)

        try:
            index = classes.index(element)
        except:
            index = len(classes)
            classes.append(element)

        train_img = [f for f in os.listdir(t_directory) if isfile(join(t_directory, f))]
        for image in train_img:
            im = cv2.imread(join(t_directory, image))
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            train_set.append(_make_array(im.flatten(), index))

    return numpy.random.permutation(train_set), classes

def load_test_data(test_dir, classes):

    test_input = [f for f in os.listdir(test_dir) if isdir(join(test_dir, f))]

    test_set = []

    # Test data
    for element in test_input:

        t_directory = join(test_dir, element)

        test_img = [f for f in os.listdir(t_directory) if isfile(join(t_directory, f))]
        for image in test_img:
            im = cv2.imread(join(t_directory, image))
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im = im.flatten()

            index = classes.index(element)
            
            test_set.append(_make_array(im.flatten(), index))

    return test_set

def _make_array(x, y):
    return (
        numpy.asarray(x, dtype='float32'),
        numpy.asarray(y, dtype='int32')
    )

def get_sparse_vector(x):
    d = {}
    for index, item in enumerate(x):
        if item != 0.0:
            d[index] = item
    return d

p = '-s 0 -t 0 -c 10' # linear kernel
p = '-t 1 -r 1 -d 3' # cubic polynomial kernel
p = '-t 0 -g 0.005 -c 10' # RBF kernel

train_dir = '../train'
test_dir = '../test'

print '... loading data'
train_set, classes = load_train_data(train_dir)

# Saving classes for recognition
f = open('classes.txt', 'wb')
for c in classes:
    f.write(str(c) + '\n')
f.close()

xdata = []
ydata = []

for item in train_set:
    xdata.append(get_sparse_vector(item[0]))
    ydata.append(item[1])

print "... seaching paramters"
param = svm_parameter(p)

print "Size of dataset: " + str(len(xdata))

print "... starting svm problem"
prob  = svm_problem(ydata, xdata)
print "... creating model"
model = svm_train(prob, param)

# Save the model for recognition
svm_save_model('svm_model.model', model)

print "... starting predict"

# Testing model accuracy
print '... loading data'
test_set = load_test_data(test_dir, classes)

xdata = []
ydata = []

for item in test_set:
    xdata.append(get_sparse_vector(item[0]))
    ydata.append(item[1])

# lab - predicted label, acc - accuracy, vals - decision function value (if < 0, then -1 class, if >=0, then +1 class)
[lab, acc, vals] = svm_predict(ydata, xdata, model)

if len(sys.argv) == 2 and sys.argv[1] == "-cv":
    print "... starting Cross validation"
    kwargs = {'c': [1, 10, 100], 'g': [0.0001, 0.001, 0.01, 0.1, 1]}

    print cross_validation(xdata, ydata, 1, 5, **kwargs)
