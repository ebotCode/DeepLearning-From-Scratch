import numpy as np 

import ttechoflow as tf 
import tools as tl 
from dataset_loader_planar import load_planar_dataset
import matplotlib.pyplot as plt 


def LinearLogistic(X_train,Y_train):
# Build Model 

	# Y_train = np.sum(X_train,axis = 0, keepdims = True) > 0.35 
	# print("Y_Train = ",Y_train)

	X = tf.Constants(X_train, name = "Input")
	Y = tf.Constants(Y_train * 1, name = "Labels")
	np.random.seed(1)

	W1 = tf.Variables(shape = (1,X.getShape()[0]),
					 initialize_with = np.array([[ 0.0158607 , -0.12856677]]) ,
					 is_parameter = True , name = "W1_weight")
	b1 = tf.Variables(shape = (1, 1), is_parameter = True,
					 initialize_with = np.array([[0.00117646]]),
					 name = "b1_weight")
	layers = [10,1]
	An_1 = tf.matmul(W1,X) + b1  # A2 
	An = tf.sigmoid(An_1 ,"Output") 
	cost  = tf.logistic_loss(logits = An, labels = Y) #tf.logistic_loss(logits = An,labels = Y)#tf.sum_of_squares_loss(An)  #tf.logistic_loss(logits = An,labels = Y)#0.5*tf.reduce_sum(Y * tf.log(An) - ( 1 - Y) * tf.log(1 - An))
	optimizer = tf.GradientDescentOptimizer(cost, learning_rate = 0.1, name = 'optimizer')

	return cost, optimizer , An




X_train, Y_train = load_planar_dataset("planar_dataset.csv")


def start():
	# Create a graph 
	my_graph = tf.ComputationGraph()
	my_graph.set_as_default()
	# Create session 
	sess = tf.Session()

	cost, optimizer, An = LinearLogistic(X_train,Y_train)

	saver = tf.Saver()
	saver.saveGraph(my_graph,"linear_model_testcase",overwrite = True)
	# runGradientChecking(my_graph,sess,cost,optimizer)
	my_graph.initialize()

def load():
	saver = tf.Saver()
	my_graph = saver.loadGraph("linear_model_testcase")
	my_graph.set_as_default()

	sess = tf.Session()

	W1 = my_graph.getVariableByName("W1_weight")
	b1 = my_graph.getVariableByName("b1_weight")

	W1_true = np.array([[ 0.0158607,  -0.12856677]])
	b1_true = np.array([[0.00117646]])

	assert(tl.isVectorEqual(W1_true,W1.getValue(),1e-8))
	assert(tl.isVectorEqual(b1_true,b1.getValue(),1e-8))

def test_saveAndLoad():
	start() 
	load()
	print("Test Save and Load passed")

if __name__ == '__main__':
	test_saveAndLoad()
