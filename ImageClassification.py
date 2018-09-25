
import numpy as np 

import ttechoflow as tf 
import tools as tl 
import matplotlib.pyplot as plt 

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes




def nn_model(X_train, Y_train, layer_dims):
# Build Model 

	# Y_train = (np.sum(X_train,axis = 0, keepdims = True) > 0.35 ) * 1 
	# print("Y_Train = ",Y_train)
	n_x = X_train.shape[0]
	n_h = 4 
	n_y = Y_train.shape[0]

	X = tf.Constants(X_train, name = "Input")
	Y = tf.Constants(Y_train * 1, name = "Labels")
	np.random.seed(1)

	L = len(layer_dims)
	A1 = X
	for l in range(1,L-1):
		W1 = tf.Variables(shape = (layer_dims[l],layer_dims[l-1]),
						 initialize_with = tf.initializations('Normal',(layer_dims[l],layer_dims[l-1])) ,
						 is_parameter = True, name = "W%d_weight"%l )
		b1 = tf.Variables(shape = (layer_dims[l], 1), is_parameter = True,
						 initialize_with = tf.initializations('Zeros',(layer_dims[l],1)),
						  name = "b%d_weight"%l)
		
		Z1 = tf.matmul(W1,A1) + b1 
		A1 = tf.relu(Z1)

	l = L-1 
	W2 = tf.Variables(shape = (layer_dims[l],layer_dims[l-1]),
					 initialize_with = tf.initializations('Normal',(layer_dims[l],layer_dims[l-1])) ,
					 is_parameter = True, name = "W%d_weight"%l )
	b2 = tf.Variables(shape = (layer_dims[l], 1), is_parameter = True,
					 initialize_with = tf.initializations('Zeros',(layer_dims[l],1)),
					  name = "b%d_weight"%l)

	Z2 = tf.matmul(W2,A1) + b2 
	An = tf.sigmoid(Z2)
	# An = tf.relu( An_1 ) 
	assert(An.getShape() == Y_train.shape)

	cost  = tf.logistic_loss(logits = An, labels = Y) 
	optimizer = tf.GradientDescentOptimizer(cost, learning_rate = 1.2, name = 'optimizer')
	return cost, optimizer , An 
