# Kimberly Kaminsky - Assignment #7
# Deep Learning - Convolutional Neural Networks

####################
# Import Libraries #
####################

# import base packages into the namespace for this program
import warnings
import numpy as np
import os
import sys
import time
from tabulate import tabulate
from functools import partial

# Use to build neural network
import tensorflow as tf

# Stores MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data

#################### 
# Define constants #
####################

HEIGHT = 28
WIDTH = 28
N_INPUTS = HEIGHT * WIDTH  # MNIST dataset features
CHANNELS = 1
N_OUTPUTS = 10    # Categories (number of digits)
DATAPATH = os.path.join("D:/","Kim MSPA", "Predict 422", "Assignments", 
                        "Assignment7", "")
                        
#############
# Functions #
#############

# function to clear output console
def clear():
    print("\033[H\033[J")
    
# reset graph to make output stable across runs
def reset_graph(seed=111):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)  

def __init__(self):
    self.AdamOptimization(logits, y)

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

def conv2_pool1(inputLayer, conv1Vars, conv2Vars, activation, initializer, 
                dropoutRate):
    

    with tf.device("/cpu:0"):   
        conv1 = tf.layers.conv2d(X_reshaped, filters=conv1Vars[0], 
                                kernel_size=conv1Vars[1],
                                strides=conv1Vars[2], padding=conv1Vars[3],
                                activation=activation, kernel_initializer = initializer,
                                name=conv1Vars[4])
        conv2 = tf.layers.conv2d(conv1, filters=conv2Vars[0], kernel_size=conv2Vars[1],
                                strides=conv2Vars[2], padding=conv2Vars[3],
                                activation=activation, kernel_initializer = initializer,
                                name=conv2Vars[4])
        
        with tf.name_scope("pool3"):
            pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                                    padding="VALID")
            pool3_drop = tf.layers.dropout(pool3, dropoutRate, 
                                        training=training)
                                        
    return pool3_drop 
    
def conv2_pool1_bn(inputLayer, conv1Vars, conv2Vars, activation, initializer, 
                dropoutRate, training, momentum):
                
    # create batch normalization layer shortcut
    batch_norm_layer = partial(tf.layers.batch_normalization, 
                        training=training, momentum=momentum)
    

    with tf.device("/cpu:0"):   
        conv1 = tf.layers.conv2d(X_reshaped, filters=conv1Vars[0], 
                                kernel_size=conv1Vars[1],
                                strides=conv1Vars[2], padding=conv1Vars[3],
                                kernel_initializer = initializer, name=conv1Vars[4])
        bn1 = batch_norm_layer(conv1)
        bn1_act = activation(bn1)                        
                                
        conv2 = tf.layers.conv2d(bn1_act, filters=conv2Vars[0], kernel_size=conv2Vars[1],
                                strides=conv2Vars[2], padding=conv2Vars[3],
                                kernel_initializer = initializer, name=conv2Vars[4])
        bn2 = batch_norm_layer(conv2)
        bn2_act = activation(bn2)
        
        with tf.name_scope("pool3"):
            pool3 = tf.nn.max_pool(bn2_act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                                    padding="VALID")
            pool3_drop = tf.layers.dropout(pool3, dropoutRate, 
                                        training=training)
                                        
    return pool3_drop 
    
def fully_connected2(inputLayer, fc1, fc2, activation, initializer, 
                        dropoutRate):     
               
        with tf.name_scope("fc1"):
            fc1 = tf.layers.dense(inputLayer, fc1[0], activation=activation, 
                                    kernel_initializer = initializer, name=fc1[1])
            fc1_drop = tf.layers.dropout(fc1, dropoutRate, training=training)
            
        with tf.name_scope("fc2"):
            fc2 = tf.layers.dense(fc1_drop, fc2[0], activation=activation, 
                                    kernel_initializer = initializer, name=fc2[1])
            fc2_drop = tf.layers.dropout(fc2, dropoutRate, training=training)
            
        return fc2_drop
        
def fully_connected2_bn(inputLayer, fc1, fc2, activation, initializer, 
                        dropoutRate, training, momentum):  
                        
        # create batch normalization layer shortcut
        batch_norm_layer = partial(tf.layers.batch_normalization, 
                        training=training, momentum=momentum)   
               
        with tf.name_scope("fc1"):
            fc1 = tf.layers.dense(inputLayer, fc1[0], 
                                    kernel_initializer = initializer, name=fc1[1])
            bn1 = batch_norm_layer(fc1)
            bn1_act = activation(bn1)
            fc1_drop = tf.layers.dropout(bn1_act, dropoutRate, training=training)
            
        with tf.name_scope("fc2"):
            fc2 = tf.layers.dense(fc1_drop, fc2[0], 
                                    kernel_initializer = initializer, name=fc2[1])
            bn2 = batch_norm_layer(fc2)
            bn2_act = activation(bn2)
            fc2_drop = tf.layers.dropout(bn2_act, dropoutRate, training=training)
            
        return fc2_drop
        
def flatten(layer):
    
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
 
    return layer

def output_layer(inputLayer):
        
    with tf.name_scope("output"):
        logits = tf.layers.dense(inputLayer, N_OUTPUTS, name="output")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")
        
    return logits, y, X
    
def runSession(training_op, n_epochs, batch_size):

    # Start clock to test time it takes to run algorithm
    start_time = time.clock()    

    best_loss_val = np.infty
    check_interval = 500
    checks_since_last_progress = 0
    max_checks_without_progress = 20
    best_model_params = None 
    
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                if iteration % check_interval == 0:
                    loss_val = loss.eval(feed_dict={X: mnist.validation.images,
                                                    y: mnist.validation.labels})
                    if loss_val < best_loss_val:
                        best_loss_val = loss_val
                        checks_since_last_progress = 0
                        best_model_params = get_model_params()
                    else:
                        checks_since_last_progress += 1
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: mnist.validation.images,
                                            y: mnist.validation.labels})
            print("Epoch {}, train accuracy: {:.4f}%, valid. accuracy: {:.4f}%, \
                    valid. best loss: {:.6f}".format(
                    epoch, acc_train * 100, acc_val * 100, best_loss_val))
            if checks_since_last_progress > max_checks_without_progress:
                print("Early stopping!")
                break
    
        if best_model_params:
            restore_model_params(best_model_params)
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                            y: mnist.test.labels})
        print("Final accuracy on test set:", acc_test)
        save_path = saver.save(sess, "./my_mnist_model")
        
        # Stop clock to time training time for CNN
        stop_time = time.clock()
        
        #Total Time
        runtime = stop_time - start_time 
 
        return acc_train, acc_val, acc_test, runtime  

def AdamOptimization(logits, y):
    
    with tf.name_scope("train"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)
        
    return xentropy, loss, optimizer, training_op
    
def NesterovOptimization(logits, y, learning_rate, momentum):
    
    with tf.name_scope("train"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                        momentum=momentum, use_nesterov=True)
        training_op = optimizer.minimize(loss)
        
    return xentropy, loss, optimizer, training_op



def evaluation(logits, y):
    
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 
    
    return correct, accuracy   


#  Setup data placeholders 
def setupPlaceholders():

    with tf.name_scope("inputs"):
        X = tf.placeholder(tf.float32, shape=[None, N_INPUTS], name="X")
        X_reshaped = tf.reshape(X, shape=[-1, HEIGHT, WIDTH, CHANNELS])
        y = tf.placeholder(tf.int32, shape=[None], name="y")
        training = tf.placeholder_with_default(False, shape=[], name='training')

    return X, X_reshaped, y, training

def initAndSave():
    
    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    return init, saver
    
    
####################
# Data Preparation #
####################

# Read in dataset
mnist = input_data.read_data_sets("/tmp/data/")

# Take a look at the shape of the train,validate and test setss
X_train = mnist.train.images
X_validate = mnist.validation.images
X_test = mnist.test.images

print("Shape of Training data: ", X_train.shape)
print("Shape of Validate data: ", X_validate.shape)
print("Shape of Test data: ", X_test.shape)



#####################################################################
# Set Epochs and Batch Size                                         #
#   Don't plan to change these between runs, but don't want to make #
#   them global in case I do need to change them                    #
#####################################################################

# Training Variables
n_epochs = 20
batch_size = 50


########################################
# Construct Model 1:                   #
#  2 conv layers:                      #
#   features: (32,64)                  #
#   Convolutional Window: (3,3)        #
#   Stride:  (1,2)                     #
#  2 Fully Connected Layers: (64, 64)  #
#  Dropout rates: (.5, .5)             #
#  Activation: reLU                    #
#  Initializer: He Init - fan avg      #
#  Optimization: Adam                  #
#  Normalization: None                 #
########################################

# reset graph
reset_graph()

M1_conv1Vars = [32, 3, 1, "SAME","conv1"]
M1_conv2Vars = [64, 3, 2, "SAME", "conv2"]
M1_fc1 = [64, "fc1"]
M1_fc2 = [64, "fc2"]
M1_activation = tf.nn.relu
M1_conv2_dropout_rate = 0.5
M1_fc1_dropout_rate = 0.5
M1_initializer = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')

X, X_reshaped, y, training = setupPlaceholders()

# setup network structure
layer1 = conv2_pool1(X_reshaped, M1_conv1Vars, M1_conv2Vars, M1_activation, 
                    M1_initializer, M1_conv2_dropout_rate)                   
flat_layer = flatten(layer1)
fcLayer1 = fully_connected2(flat_layer, M1_fc1, M1_fc2, M1_activation, 
                            M1_initializer, M1_fc1_dropout_rate)
logits, y, X = output_layer(fcLayer1)

xentropy, loss, optimizer, training_op = AdamOptimization(logits, y)
M1_optimizer = str(optimizer).split(".")[3]
correct, accuracy  = evaluation(logits, y)
init, saver = initAndSave()

M1_acc_train, M1_acc_val, M1_accuracy, M1_runtime = runSession(training_op, 
                                                            n_epochs, batch_size)  

########################################
# Construct Model 2:                   #
#  2 conv layers:                      #
#   features: (32,64)                  #
#   Convolutional Window: (3,3)        #
#   Stride:  (1,2)                     #
#  2 Fully Connected Layers: (64, 64)  #
#  Dropout rates: (.5, .5)             #
#  Activation: relu                    #
#  Initializer: Xavier                 #
#  Optimization: Adam                  #
#  Normalization: None                 #
########################################

# reset graph
reset_graph()

M2_conv1Vars = [32, 3, 1, "SAME","conv1"]
M2_conv2Vars = [64, 3, 2, "SAME", "conv2"]
M2_fc1 = [64, "fc1"]
M2_fc2 = [64, "fc2"]
M2_activation = tf.nn.relu
M2_conv2_dropout_rate = 0.5
M2_fc1_dropout_rate = 0.5
M2_initializer = tf.contrib.layers.xavier_initializer()

X, X_reshaped, y, training = setupPlaceholders()

# setup network structure
layer1 = conv2_pool1(X_reshaped, M2_conv1Vars, M2_conv2Vars, M2_activation, 
                    M2_initializer, M2_conv2_dropout_rate)                   
flat_layer = flatten(layer1)
fcLayer1 = fully_connected2(flat_layer, M2_fc1, M2_fc2, M2_activation, 
                            M2_initializer, M2_fc1_dropout_rate)
logits, y, X = output_layer(fcLayer1)

xentropy, loss, optimizer, training_op = AdamOptimization(logits, y)
M2_optimizer = str(optimizer).split(".")[3]
correct, accuracy  = evaluation(logits, y)
init, saver = initAndSave()

M2_acc_train, M2_acc_val, M2_accuracy, M2_runtime = runSession(training_op, 
                                                        n_epochs, batch_size)  

########################################
# Construct Model 3:                   #
#  2 conv layers:                      #
#   features: (32,64)                  #
#   Convolutional Window: (3,3)        #
#   Stride:  (1,2)                     #
#  2 Fully Connected Layers: (64, 64)  #
#  Dropout rates: (.5, .5)             #
#  Activation: relu                    #
#  Initializer: He Init - fan avg      #
#  Optimization: Nesterov              #
#  Normalization: None                 #
########################################  

# reset graph
reset_graph()

M3_conv1Vars = [32, 3, 1, "SAME","conv1"]
M3_conv2Vars = [64, 3, 2, "SAME", "conv2"]
M3_fc1 = [64, "fc1"]
M3_fc2 = [64, "fc2"]
M3_activation = tf.nn.relu
M3_conv2_dropout_rate = 0.5
M3_fc1_dropout_rate = 0.5
M3_initializer = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
M3_learning_rate = .05
M3_momentum = .9

X, X_reshaped, y, training = setupPlaceholders()

# setup network structure
layer1 = conv2_pool1(X_reshaped, M3_conv1Vars, M3_conv2Vars, M3_activation, 
                    M3_initializer, M3_conv2_dropout_rate)                   
flat_layer = flatten(layer1)
fcLayer1 = fully_connected2(flat_layer, M3_fc1, M3_fc2, M3_activation, 
                            M3_initializer, M3_fc1_dropout_rate)
logits, y, X = output_layer(fcLayer1)

xentropy, loss, optimizer, training_op = NesterovOptimization(logits, y, 
                                                M3_learning_rate, M3_momentum)
M3_optimizer = str(optimizer).split(".")[3]
correct, accuracy  = evaluation(logits, y)
init, saver = initAndSave()

M3_acc_train, M3_acc_val, M3_accuracy, M3_runtime = runSession(training_op, 
                                                        n_epochs, batch_size) 
                                                        
                                                        
########################################
# Construct Model 4:                   #
#  2 conv layers:                      #
#   features: (32,64)                  #
#   Convolutional Window: (3,3)        #
#   Stride:  (1,2)                     #
#  2 Fully Connected Layers: (64, 64)  #
#  Dropout rates: (.5, .5)             #
#  Activation: relu                    #
#  Initializer: Xavier                 #
#  Optimization: Nesterov              #
#  Normalization: None                 #
########################################  

# reset graph
reset_graph()

M4_conv1Vars = [32, 3, 1, "SAME","conv1"]
M4_conv2Vars = [64, 3, 2, "SAME", "conv2"]
M4_fc1 = [64, "fc1"]
M4_fc2 = [64, "fc2"]
M4_activation = tf.nn.relu
M4_conv2_dropout_rate = 0.5
M4_fc1_dropout_rate = 0.5
M4_initializer = tf.contrib.layers.xavier_initializer()
M4_learning_rate = .05
M4_momentum = .9

X, X_reshaped, y, training = setupPlaceholders()

# setup network structure
layer1 = conv2_pool1(X_reshaped, M4_conv1Vars, M4_conv2Vars, M4_activation, 
                    M4_initializer, M4_conv2_dropout_rate)                   
flat_layer = flatten(layer1)
fcLayer1 = fully_connected2(flat_layer, M4_fc1, M4_fc2, M4_activation, 
                            M4_initializer, M4_fc1_dropout_rate)
logits, y, X = output_layer(fcLayer1)

xentropy, loss, optimizer, training_op = NesterovOptimization(logits, y, 
                                                M4_learning_rate, M4_momentum)
M4_optimizer = str(optimizer).split(".")[3]
correct, accuracy  = evaluation(logits, y)
init, saver = initAndSave()

M4_acc_train, M4_acc_val, M4_accuracy, M4_runtime = runSession(training_op, 
                                                        n_epochs, batch_size) 
  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
########################################
# Construct Model 5:                   #
#  2 conv layers:                      #
#   features: (32,64)                  #
#   Convolutional Window: (3,3)        #
#   Stride:  (1,2)                     #
#  2 Fully Connected Layers: (64, 64)  #
#  Dropout rates: (.5, .5)             #
#  Activation: reLU                    #
#  Initializer: He Init - fan avg      #
#  Optimization: Adam                  #
#  Normalization: Batch                #
########################################

# reset graph
reset_graph()

M5_conv1Vars = [32, 3, 1, "SAME","conv1"]
M5_conv2Vars = [64, 3, 2, "SAME", "conv2"]
M5_fc1 = [64, "fc1"]
M5_fc2 = [64, "fc2"]
M5_activation = tf.nn.relu
M5_conv2_dropout_rate = 0.5
M5_fc1_dropout_rate = 0.5
M5_initializer = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
M5_momentum = .9

X, X_reshaped, y, training = setupPlaceholders()

# setup network structure
layer1 = conv2_pool1_bn(X_reshaped, M5_conv1Vars, M5_conv2Vars, M5_activation, 
                    M5_initializer, M5_conv2_dropout_rate, training, M5_momentum)                   
flat_layer = flatten(layer1)
fcLayer1 = fully_connected2_bn(flat_layer, M5_fc1, M5_fc2, M5_activation, 
                            M5_initializer, M5_fc1_dropout_rate, training, 
                            M5_momentum)
logits, y, X = output_layer(fcLayer1)

xentropy, loss, optimizer, training_op = AdamOptimization(logits, y)
M5_optimizer = str(optimizer).split(".")[3]
correct, accuracy  = evaluation(logits, y)
init, saver = initAndSave()

M5_acc_train, M5_acc_val, M5_accuracy, M5_runtime = runSession(training_op, 
                                                            n_epochs, batch_size)
                                                            
########################################
# Construct Model 6:                   #
#  2 conv layers:                      #
#   features: (32,64)                  #
#   Convolutional Window: (3,3)        #
#   Stride:  (1,2)                     #
#  2 Fully Connected Layers: (64, 64)  #
#  Dropout rates: (.5, .5)             #
#  Activation: reLU                    #
#  Initializer: Xavier                 #
#  Optimization: Adam                  #
#  Normalization: Batch                #
########################################

# reset graph
reset_graph()

M6_conv1Vars = [32, 3, 1, "SAME","conv1"]
M6_conv2Vars = [64, 3, 2, "SAME", "conv2"]
M6_fc1 = [64, "fc1"]
M6_fc2 = [64, "fc2"]
M6_activation = tf.nn.relu
M6_conv2_dropout_rate = 0.5
M6_fc1_dropout_rate = 0.5
M6_initializer = tf.contrib.layers.xavier_initializer()
M6_momentum = .9

X, X_reshaped, y, training = setupPlaceholders()

# setup network structure
layer1 = conv2_pool1_bn(X_reshaped, M6_conv1Vars, M6_conv2Vars, M6_activation, 
                    M6_initializer, M6_conv2_dropout_rate, training, M6_momentum)                   
flat_layer = flatten(layer1)
fcLayer1 = fully_connected2_bn(flat_layer, M6_fc1, M6_fc2, M6_activation, 
                            M6_initializer, M6_fc1_dropout_rate, training, 
                            M6_momentum)
logits, y, X = output_layer(fcLayer1)

xentropy, loss, optimizer, training_op = AdamOptimization(logits, y)
M6_optimizer = str(optimizer).split(".")[3]
correct, accuracy  = evaluation(logits, y)
init, saver = initAndSave()

M6_acc_train, M6_acc_val, M6_accuracy, M6_runtime = runSession(training_op, 
                                                            n_epochs, batch_size)
                                                            
########################################
# Construct Model 7:                   #
#  2 conv layers:                      #
#   features: (32,64)                  #
#   Convolutional Window: (3,3)        #
#   Stride:  (1,2)                     #
#  2 Fully Connected Layers: (64, 64)  #
#  Dropout rates: (.5, .5)             #
#  Activation: reLU                    #
#  Initializer: He Init - Fan Avg      #
#  Optimization: Nesterov              #
#  Normalization: Batch                #
########################################

# reset graph
reset_graph()

M7_conv1Vars = [32, 3, 1, "SAME","conv1"]
M7_conv2Vars = [64, 3, 2, "SAME", "conv2"]
M7_fc1 = [64, "fc1"]
M7_fc2 = [64, "fc2"]
M7_activation = tf.nn.relu
M7_conv2_dropout_rate = 0.5
M7_fc1_dropout_rate = 0.5
M7_initializer = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
M7_learning_rate = .05
M7_momentum = .9

X, X_reshaped, y, training = setupPlaceholders()

# setup network structure
layer1 = conv2_pool1_bn(X_reshaped, M7_conv1Vars, M7_conv2Vars, M7_activation, 
                    M7_initializer, M7_conv2_dropout_rate, training, M7_momentum)                   
flat_layer = flatten(layer1)
fcLayer1 = fully_connected2_bn(flat_layer, M7_fc1, M7_fc2, M7_activation, 
                            M7_initializer, M7_fc1_dropout_rate, training, 
                            M7_momentum)
logits, y, X = output_layer(fcLayer1)

xentropy, loss, optimizer, training_op = NesterovOptimization(logits, y, 
                                                M7_learning_rate, M7_momentum)
M7_optimizer = str(optimizer).split(".")[3]
correct, accuracy  = evaluation(logits, y)
init, saver = initAndSave()

M7_acc_train, M7_acc_val, M7_accuracy, M7_runtime = runSession(training_op, 
                                                            n_epochs, batch_size)

########################################
# Construct Model 8:                   #
#  2 conv layers:                      #
#   features: (32,64)                  #
#   Convolutional Window: (3,3)        #
#   Stride:  (1,2)                     #
#  2 Fully Connected Layers: (64, 64)  #
#  Dropout rates: (.5, .5)             #
#  Activation: reLU                    #
#  Initializer: Xavier                 #
#  Optimization: Nesterov              #
#  Normalization: Batch                #
########################################

# reset graph
reset_graph()

M8_conv1Vars = [32, 3, 1, "SAME","conv1"]
M8_conv2Vars = [64, 3, 2, "SAME", "conv2"]
M8_fc1 = [64, "fc1"]
M8_fc2 = [64, "fc2"]
M8_activation = tf.nn.relu
M8_conv2_dropout_rate = 0.5
M8_fc1_dropout_rate = 0.5
M8_initializer = tf.contrib.layers.xavier_initializer()
M8_learning_rate = .05
M8_momentum = .9

X, X_reshaped, y, training = setupPlaceholders()

# setup network structure
layer1 = conv2_pool1_bn(X_reshaped, M8_conv1Vars, M8_conv2Vars, M8_activation, 
                    M8_initializer, M8_conv2_dropout_rate, training, M8_momentum)                   
flat_layer = flatten(layer1)
fcLayer1 = fully_connected2_bn(flat_layer, M8_fc1, M8_fc2, M8_activation, 
                            M8_initializer, M8_fc1_dropout_rate, training, 
                            M8_momentum)
logits, y, X = output_layer(fcLayer1)

xentropy, loss, optimizer, training_op = NesterovOptimization(logits, y, 
                                                M8_learning_rate, M8_momentum)
M8_optimizer = str(optimizer).split(".")[3]
correct, accuracy  = evaluation(logits, y)
init, saver = initAndSave()

M8_acc_train, M8_acc_val, M8_accuracy, M8_runtime = runSession(training_op, 
                                                            n_epochs, batch_size)  
    
        
###############
# More Layers #                                                            
###############                                                                                                                      
                                                                                                                                                                                                                                                                                                 
##########################################                                                                                                                                                                                                                                                              ###################################
# Construct Model 9:                     #
#  2 conv layers:                        #
#   features: (32,64)                    #
#   Convolutional Window: (3,3)          #
#   Stride:  (1,2)                       #
#  1 Fully Connected Layer: (64)         #
#  2 conv layers:                        #
#   features: (64,128)                   #
#   Convolutional Window: (2,2)          #
#   Stride:  (1,1)                       #
#  2 Fully Connected Layers: (128, 64)   #
#  Dropout rates: (.5, .5)               #
#  Activation: reLu                      #
#  Initializer: He Init                  #
#  Optimization: Adam                    #
#  Normalization: Batch                  #
##########################################    

# reset graph
reset_graph() 

M9_conv1Vars = [32, 3, 1, "SAME","conv1"]
M9_conv2Vars = [64, 3, 2, "SAME", "conv2"]
M9_fc1 = [64, "fc1"]
M9_fc2 = [64, "fc2"]
M9_conv3Vars = [64, 2, 1, "SAME","conv3"]
M9_conv4Vars = [128, 2, 2, "SAME", "conv4"]
M9_fc3 = [128, "fc3"]
M9_fc4 = [128, "fc4"]
M9_activation = tf.nn.relu
M9_conv2_dropout_rate = 0.5
M9_fc1_dropout_rate = 0.5
M9_initializer = tf.contrib.layers.xavier_initializer()
M9_learning_rate = .05
M9_momentum = .9   

X, X_reshaped, y, training = setupPlaceholders()

# setup network structure
layer1 = conv2_pool1_bn(X_reshaped, M9_conv1Vars, M9_conv2Vars, M9_activation, 
                    M9_initializer, M9_conv2_dropout_rate, training, M9_momentum)                   
flat_layer = flatten(layer1)
fcLayer1 = fully_connected2_bn(flat_layer, M9_fc1, M9_fc2, M9_activation, 
                            M9_initializer, M9_fc1_dropout_rate, training, 
                            M9_momentum)
                            
layer2 = conv2_pool1_bn(fcLayer1, M9_conv3Vars, M9_conv4Vars, M9_activation, 
                    M9_initializer, M9_conv2_dropout_rate, training, M9_momentum)                   
flat_layer2 = flatten(layer2)
fcLayer2 = fully_connected2_bn(flat_layer2, M9_fc3, M9_fc4, M9_activation, 
                            M9_initializer, M9_fc1_dropout_rate, training, 
                            M9_momentum)                            
                            
logits, y, X = output_layer(fcLayer2)

xentropy, loss, optimizer, training_op = AdamOptimization(logits, y)
M9_optimizer = str(optimizer).split(".")[3]
correct, accuracy  = evaluation(logits, y)
init, saver = initAndSave()

M9_acc_train, M9_acc_val, M9_accuracy, M9_runtime = runSession(training_op, 
                                                        n_epochs, batch_size)
                                      

##############################
# Print out model statistics #
##############################                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                                                                                            
print("Model 1: Train Accuracy: ", M1_acc_train, "Validation Accuracy: ", 
   M1_acc_val, "Test Accuracy: ", M1_accuracy, "Runtime: ", round(M1_runtime,4))          
     
print("Model 2: Train Accuracy: ", M2_acc_train, "Validation Accuracy: ", 
   M2_acc_val, "Test Accuracy: ", M2_accuracy, "Runtime: ", round(M2_runtime,4)) 
   
print("Model 3: Train Accuracy: ", M3_acc_train, "Validation Accuracy: ", 
   M3_acc_val, "Test Accuracy: ", M3_accuracy, "Runtime: ", round(M3_runtime,4)) 
   
print("Model 4: Train Accuracy: ", M4_acc_train, "Validation Accuracy: ", 
   M4_acc_val, "Test Accuracy: ", M4_accuracy, "Runtime: ", round(M4_runtime,4))
   
print("Model 5: Train Accuracy: ", M5_acc_train, "Validation Accuracy: ", 
   M5_acc_val, "Test Accuracy: ", M5_accuracy, "Runtime: ", round(M5_runtime,4))

print("Model 6: Train Accuracy: ", M6_acc_train, "Validation Accuracy: ", 
   M6_acc_val, "Test Accuracy: ", M6_accuracy, "Runtime: ", round(M6_runtime,4))
   
print("Model 7: Train Accuracy: ", M7_acc_train, "Validation Accuracy: ", 
   M7_acc_val, "Test Accuracy: ", M7_accuracy, "Runtime: ", round(M7_runtime,4))
   
print("Model 8: Train Accuracy: ", M8_acc_train, "Validation Accuracy: ", 
   M8_acc_val, "Test Accuracy: ", M8_accuracy, "Runtime: ", round(M8_runtime,4))

print("Model 9: Train Accuracy: ", M9_acc_train, "Validation Accuracy: ", 
   M9_acc_val, "Test Accuracy: ", M9_accuracy, "Runtime: ", round(M9_runtime,4))
   

########################
# Create Output Table  #
########################

col_labels = ['Configuration', 'Nodes per Layer', 'Activation', 'Initialization', 
            'Optimization', 'Normalization', 'Processing Time',
            'Training Accuracy', 'Validation Accuracy', 'Test Accuracy']
                                
table_vals = [["C2 FC2", "(" + str(M1_conv1Vars[0]) + "," + str(M1_conv2Vars[0]) 
                + ")  (" + str(M1_fc1[0]) + "," + str(M1_fc2[0]) + ")",
                str(M1_activation).split(" ")[1], "He Init - Fan Avg", 
                M1_optimizer, "No", round(M1_runtime,2), round(M1_acc_train,3), 
                round(M1_acc_val, 3), round(M1_accuracy, 3)],
              
               ["C2 FC2", "(" + str(M2_conv1Vars[0]) + "," + str(M2_conv2Vars[0]) 
                + ")  (" + str(M2_fc1[0]) + "," + str(M2_fc2[0]) + ")",
                str(M2_activation).split(" ")[1], "Xavier", 
                M2_optimizer, "No", round(M2_runtime,2), round(M2_acc_train,3), 
                round(M2_acc_val, 3), round(M2_accuracy, 3)],
                
                ["C2 FC2", "(" + str(M3_conv1Vars[0]) + "," + str(M3_conv2Vars[0]) 
                + ")  (" + str(M3_fc1[0]) + "," + str(M3_fc2[0]) + ")",
                str(M3_activation).split(" ")[1], "He Init - Fan Avg", 
                M3_optimizer, "No", round(M3_runtime,2), round(M3_acc_train,3), 
                round(M3_acc_val, 3), round(M3_accuracy, 3)],
                
                ["C2 FC2", "(" + str(M4_conv1Vars[0]) + "," + str(M4_conv2Vars[0]) 
                + ")  (" + str(M4_fc1[0]) + "," + str(M4_fc2[0]) + ")",
                str(M4_activation).split(" ")[1], "Xavier", 
                M4_optimizer, "No", round(M4_runtime,2), round(M4_acc_train,3), 
                round(M4_acc_val, 3), round(M4_accuracy, 3)],   
                
                ["C2 FC2", "(" + str(M5_conv1Vars[0]) + "," + str(M5_conv2Vars[0]) 
                + ")  (" + str(M5_fc1[0]) + "," + str(M5_fc2[0]) + ")",
                str(M5_activation).split(" ")[1], "He Init", 
                M5_optimizer, "Yes", round(M5_runtime,2), round(M5_acc_train,3), 
                round(M5_acc_val, 3), round(M5_accuracy, 3)],
                
                ["C2 FC2", "(" + str(M6_conv1Vars[0]) + "," + str(M6_conv2Vars[0]) 
                + ")  (" + str(M6_fc1[0]) + "," + str(M6_fc2[0]) + ")",
                str(M6_activation).split(" ")[1], "Xavier", 
                M6_optimizer, "Yes", round(M6_runtime,2), round(M6_acc_train,3), 
                round(M6_acc_val, 3), round(M6_accuracy, 3)],    
                
                 ["C2 FC2", "(" + str(M7_conv1Vars[0]) + "," + str(M7_conv2Vars[0]) 
                + ")  (" + str(M7_fc1[0]) + "," + str(M7_fc2[0]) + ")",
                str(M7_activation).split(" ")[1], "He Init", 
                M7_optimizer, "Yes", round(M7_runtime,2), round(M7_acc_train,3), 
                round(M7_acc_val, 3), round(M7_accuracy, 3)],
                
                ["C2 FC2", "(" + str(M8_conv1Vars[0]) + "," + str(M8_conv2Vars[0]) 
                + ")  (" + str(M8_fc1[0]) + "," + str(M8_fc2[0]) + ")",
                str(M8_activation).split(" ")[1], "Xavier", 
                M8_optimizer, "Yes", round(M8_runtime,2), round(M8_acc_train,3), 
                round(M8_acc_val, 3), round(M8_accuracy, 3)],                  
                
                ["C2 FC2 C2 FC2", "(" + str(M9_conv1Vars[0]) + "," + 
                str(M9_conv2Vars[0]) + ")  (" + str(M9_fc1[0]) + "," + 
                str(M9_fc2[0]) + "," + str(M9_conv3Vars[0]) + "," + 
                str(M9_conv4Vars[0]) + ")  (" + str(M9_fc3[0]) + "," + 
                str(M9_fc4[0]) + ")", str(M9_activation).split(" ")[1], 
                "Xavier", M9_optimizer, "Yes", round(M9_runtime,2), 
                round(M9_acc_train,3), round(M9_acc_val, 3), 
                round(M9_accuracy, 3)]]  

table = tabulate(table_vals, headers=col_labels)
                                
print(table)
   
  