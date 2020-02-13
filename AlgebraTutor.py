########################################################
#  Author: Samuel Lamb                                 #
#  Reference: codelabs.developers.google.com           #
#  Date Created: 2/13/2020                             #
########################################################

# X: -1, 0, 1, 2, 3, 4
# Y: -2, 1, 4, 7, 10, 13
# Equation: 3x + 1

import tensorflow as tf
import numpy as np # Importing numpy (numb-pie) will help us easily represent the data as a list.
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) # The simplest neural network, 1 unit, 1 neuron, and 1 layer.
model.compile(optimizer='sgd', loss='mean_squared_error') # The optimizer functionality will try and mitigate error using stochastic gradient descent
# The loss is recorded as mean squared error
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float) # Adding in x variables for the code to test
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float) # Adding in y variables for the code to test

model.fit(xs, ys, epochs=500) # Model.fit makes it go trough the loop and "learn" 
# You will notice the error decreasing each epoch

print(model.predict([5.0])) # Now the program will predict what the Y value would be in given an X of 5
