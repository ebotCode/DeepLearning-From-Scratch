
import numpy as np 

import ttechoflow as tf 
import tools as tl 
from dataset_loader_planar import load_planar_dataset
import matplotlib.pyplot as plt 


X_train, Y_train = load_planar_dataset("planar_dataset.csv")

# plot data 
plt.scatter(X_train[0, :], X_train[1, :], c=Y_train.flatten(), s=40, cmap=plt.cm.Spectral);

def LinearLogistic(X_train,Y_train):
# Build Model 

	# Y_train = np.sum(X_train,axis = 0, keepdims = True) > 0.35 
	# print("Y_Train = ",Y_train)

	X = tf.Constants(X_train, name = "Input")
	Y = tf.Constants(Y_train * 1, name = "Labels")
	np.random.seed(1)

	W1 = tf.Variables(shape = (1,X.getShape()[0]), initialize_with = tf.initializations('Zeros',(1,X.getShape()[0])) , is_parameter = True )
	b1 = tf.Variables(shape = (1, 1), is_parameter = True, initialize_with = tf.initializations('Zeros',(1,1)))
	layers = [10,1]
	An_1 = tf.matmul(W1,X) + b1  # A2 
	An = tf.sigmoid(An_1 ) 
	cost  = tf.logistic_loss(logits = An, labels = Y) #tf.logistic_loss(logits = An,labels = Y)#tf.sum_of_squares_loss(An)  #tf.logistic_loss(logits = An,labels = Y)#0.5*tf.reduce_sum(Y * tf.log(An) - ( 1 - Y) * tf.log(1 - An))
	optimizer = tf.GradientDescentOptimizer(cost, learning_rate = 0.1, name = 'optimizer')

	return cost, optimizer , An

def LinearTanh(X_train, Y_train):
# Build Model 

	# Y_train = (np.sum(X_train,axis = 0, keepdims = True) > 0.35 ) * 1 
	# print("Y_Train = ",Y_train)
	n_x = X_train.shape[0]
	n_h = 10
	n_y = Y_train.shape[0]

	X = tf.Constants(X_train, name = "Input")
	Y = tf.Constants(Y_train * 1, name = "Labels")
	np.random.seed(1)

	W1 = tf.Variables(shape = (n_h,n_x), initialize_with = tf.initializations('Normal',(n_h,n_x)) ,
					 is_parameter = True, name = "W1_weight" )
	b1 = tf.Variables(shape = (n_h, 1), is_parameter = True, initialize_with = tf.initializations('Zeros',(n_h,1)), name = "b1_weight")
	
	W2 = tf.Variables(shape = (n_y,n_h), initialize_with = tf.initializations('Normal',(n_y,n_h)) ,
					 is_parameter = True, name = "W2_weight" )
	b2 = tf.Variables(shape = (n_y, 1), is_parameter = True, initialize_with = tf.initializations('Zeros',(n_y,1)), name = "b2_weight")

	Z1 = tf.matmul(W1,X) + b1 # A2 
	A1 = tf.dropout(tf.tanh(Z1),keepprops = 1)
	Z2 = tf.matmul(W2,A1) + b2 
	An = tf.sigmoid(Z2)
	# An = tf.relu( An_1 ) 
	assert(An.getShape() == Y_train.shape)
	nsamples = X_train.shape[1]
	cost  = tf.logistic_loss(logits = An, labels = Y) + tf.l2norm(W1,0.4,nsamples) + tf.l2norm(W2,0.4,nsamples)

	optimizer = tf.GradientDescentOptimizer(cost, learning_rate = 0.1, name = 'optimizer')
	return cost, optimizer , An 

def GradCheck(theta, sess,cost,optimizer,my_graph):
	# Perform gradient computation using finite difference 
	# theta = np.array([[ 1.61434536,-0.71175641]])
	dtheta = np.zeros(theta.shape)
	epsilon = 1e-7
	for i in range(dtheta.shape[0] * dtheta.shape[1]):
		theta_plus  = np.copy(theta)
		theta_plus[0,i] += epsilon 
		theta_minus = np.copy(theta)
		theta_minus[0,i] -= epsilon 
		tl.convertVectorToAllParameters(theta_plus,my_graph.parameter_list, my_graph.all_variables)
		J_plus  = sess.evaluate(cost)
		tl.convertVectorToAllParameters(theta_minus,my_graph.parameter_list, my_graph.all_variables)
		J_minus = sess.evaluate(cost)
		
		dtheta[0,i] = (J_plus - J_minus)/(2 * epsilon)
		# print("dtheta = ",dtheta[0,i])
		# print("J_plus = ",J_plus)
		# print("J_minus = ",J_minus)
		# print("epsilon = ", epsilon)
		# print("(J_plus - J_minus)/(2 * epsilon)" ,(J_plus - J_minus)/(2 * epsilon))
		# print("J_plus = (%f) , J_minus = (%f)"%(J_plus, J_minus),(J_plus - J_minus)/(2 * epsilon), "------------------------>")

	tl.convertVectorToAllParameters(theta,my_graph.parameter_list, my_graph.all_variables)

	sess.run(optimizer)
	mydtheta = tl.convertAllParametersGradientToVector(my_graph.parameter_list,my_graph.all_variables)

	print("*"*50)
	print("Dtheta = ")
	print(dtheta)
	print("MyDtheta = ")
	print(mydtheta)
	print("*"*50)

	gradcheck = np.sqrt(np.sum((mydtheta - dtheta)**2))/(np.sqrt(np.sum(mydtheta**2)) + np.sqrt(np.sum(dtheta**2)))
	print('gradcheck gives = ',gradcheck)
	print("at cost = ",cost.getValue())
	assert(gradcheck < 1e-7)

def threshold_function(Z):
	return (Z > 0.5)

def LinearLogisticPredict(model_output,test_set):
	x1 = my_graph.getVariableByName("Input")
	prev_value = x1.getValue()
	x1.setValue(test_set)
	Z = threshold_function(sess.evaluate(model_output))
	x1.setValue(prev_value)
	return Z 

def LinearLogisticAccuracy (predicted, actual):
	return np.sum(predicted == actual) * 100/predicted.shape[1]

def runGradientChecking(my_graph):
	print("Running Gradient Checking ","@"*10)
	for i in range(10):
		theta = tl.convertAllParametersToVector(my_graph.parameter_list,my_graph.all_variables)
		print("Starting with Theta = ")
		print(theta)
		# ############### Perform gradient checking 
		GradCheck(theta,sess,cost,optimizer,my_graph)
	print("Completed Gradient Checking ", "@"*10)
		

def runIteration(optimizer,cost, niter = 1):
	# run iteration 
	print("Running Iteration Checking ","@"*10)
	cost_list = []
	for i in range(niter):
		sess.run(optimizer)
		cost_list.append(np.squeeze(cost.getValue()))
		
		if i%100 == 1:
			print("****************************")
			print("iteration = (%d)"%i)
			print("cost = ",np.squeeze(cost.getValue()))
	print("Completed Running Iteration Checking ","@"*50)
	return cost_list 


# Create a graph 
my_graph = tf.ComputationGraph()
my_graph.set_as_default()

# Create session 
sess = tf.Session()
# Models 
# cost, optimizer, An = LinearLogistic(X_train,Y_train)
cost, optimizer, An = LinearTanh(X_train,Y_train)

# runGradientChecking(my_graph)

if 1: 
	cost_list = runIteration(optimizer,cost, 10000)
	# compute accuracy on training set 
	print("@"*10)
	pred = LinearLogisticPredict(An,X_train)
	train_acc = LinearLogisticAccuracy(pred,Y_train == 1)
	print("Training accuracy = %.2f"%train_acc,"%")
	print("@"*10)

	# display weights 
	for item in ["W1_weight","b1_weight","W2_weight","b2_weight"]:
		var_ = my_graph.getVariableByName(item)
		print(var_.getName())
		print(var_.getValue())

	# make plots 
	tl.plot_decision_boundary(LinearLogisticPredict,An,my_graph.getVariableByName("Input"),sess, X_train,Y_train)
	plt.figure()
	plt.plot(range(len(cost_list)),cost_list,)
	plt.show()