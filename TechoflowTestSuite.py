
import numpy as np 

import ttechoflow as tf 
import tools as tl 
from dataset_loader_planar import load_planar_dataset
import matplotlib.pyplot as plt 


def L_model_forward_testcase_2hidden(param_name):
	if param_name == "X":
		value = np.array([[-0.31178367 , 0.72900392,  0.21782079, -0.8990918 ],
						 [-2.48678065,  0.91325152,  1.12706373, -1.51409323],
						 [ 1.63929108, -0.4298936 ,  2.63128056,  0.60182225],
						 [-0.33588161,  1.23773784,  0.11112817,  0.12915125],
						 [ 0.07612761, -0.15512816,  0.63422534,  0.810655  ]])

	elif param_name == "W1_weight":
		value = np.array([[ 0.35480861,  1.81259031 ,-1.3564758,  -0.46363197,  0.82465384],
						 [-1.17643148,  1.56448966 , 0.71270509, -0.1810066 ,  0.53419953],
						 [-0.58661296, -1.48185327 , 0.85724762,  0.94309899 , 0.11444143],
						 [-0.02195668 ,-2.12714455, -0.83440747, -0.46550831,  0.23371059]])
	elif param_name == "W2_weight":
		value = np.array([[-0.12673638, -1.36861282,  1.21848065, -0.85750144],
						 [-0.56147088, -1.0335199,   0.35877096,  1.07368134],
						 [-0.37550472,  0.39636757, -0.47144628 , 2.33660781]])

	elif param_name == "b1_weight":
		value = np.array([[ 1.38503523],
						 [-0.51962709],
						 [-0.78015214],
						 [ 0.95560959]])
	elif param_name == "W3_weight":
		value = np.array([[ 0.9398248 , 0.42628539, -0.75815703]])

	elif param_name == "b2_weight":
		value = np.array([[ 1.50278553],
						 [-0.59545972],
						 [ 0.52834106]])
	elif param_name == "b3_weight":
		value = np.array([[-0.16236698]])

	else:
		raise ValueError("Wrong Input !!!!")

	return value 


def test_forwardprop(my_graph,sess):
	weights = ["W1_weight","W2_weight","W3_weight","b1_weight","b2_weight","b3_weight"]
	x_train = L_model_forward_testcase_2hidden("X")
	y_train = np.ones((1,x_train.shape[1]))
	layer_dims = [5,4,3,1]
	_,_,An = nn_model(x_train,y_train,layer_dims)
	for item in weights:
		my_graph.getVariableByName(item).setValue(L_model_forward_testcase_2hidden(item))

	value = sess.evaluate(An)
	true_value = np.array([[ 0.03921668 , 0.70498921 , 0.19734387 , 0.04728177]])
	
	assert (tl.isVectorEqual(value,true_value,1e-8))
	print("test_forwardprop passed")
	


def test_compute_cost(my_graph,sess):
	AL = tf.Variables(shape = (1,3),initialize_with = np.array([[ 0.8 , 0.9 , 0.4]]))
	
	Y  = tf.Variables(shape = (1,3),initialize_with = np.array([[ 1 , 1 , 1]]))

	cost = tf.logistic_loss(AL,Y)

	value = np.squeeze(sess.evaluate(cost))
	true_value = 0.41493159961539694
	assert(abs(value - true_value) < 1e-30)
	print("test_compute_cost passed")

	# print(AL.getValue())
	# print(Y.getValue())
	# print("cost = ",value)


def test_linear_backward(my_graph,sess):
	A_prev = tf.Variables(shape = (3,2),
						 initialize_with = np.array([[-0.52817175, -1.07296862],
													 [ 0.86540763, -2.3015387 ],
													 [ 1.74481176, -0.7612069 ]]) ,
						 is_parameter = True, name = "A_prev" )
	
	W1 = tf.Variables(shape = (1,3),
					 initialize_with = np.array([[ 0.3190391 , -0.24937038 , 1.46210794]]),
					 is_parameter = True, name = "W1_weight" )

	b1 = tf.Variables(shape = (1, 1), is_parameter = True,
					 initialize_with = np.array([[-2.06014071]]),
					  name = "b1_weight")

	Zn = tf.matmul(W1,A_prev) + b1 
	Zn.setValue(np.array([[ 0.04153939, -1.11792545]]))


	Zn.setCostGradient(np.array([[ 1.62434536, -0.61175641]]))

	optimizer = tf.GradientDescentOptimizer(Zn, learning_rate = 1.2, name = 'optimizer')
	optimizer.backpropagate(Zn, first = 0) 

	dW =  np.array([[-0.10076895 , 1.40685096 , 1.64992504]])
	db = np.array([[0.50629448]])
	dZn =  np.array([[ 1.62434536 ,-0.61175641]])

	assert (tl.isVectorEqual(dW,W1.getCostGradient(),1e-7))
	assert (tl.isVectorEqual(db,b1.getCostGradient(),1e-7))
	print("test_linear_backward passed")

	# print("Testing Linear backward ","*"*10)
	# print("dAn_prev = ",A_prev.getCostGradient())
	# print("dW = ",W1.getCostGradient())
	# print("db = ",b1.getCostGradient())
	# print("dZn = ",Zn.getCostGradient())
	# print("*"*10)


def test_linear_activation_backward(my_graph,sess,activation):
	A_prev = tf.Variables(shape = (3,2),
						 initialize_with = np.array([[-2.1361961  , 1.64027081],
													 [-1.79343559 ,-0.84174737],
													 [ 0.50288142, -1.24528809]]) ,
						 is_parameter = True, name = "A_prev" )
	
	W1 = tf.Variables(shape = (1,3),
					 initialize_with = np.array([[-1.05795222 ,-0.90900761 , 0.55145404]]),
					 is_parameter = True, name = "W1_weight" )

	b1 = tf.Variables(shape = (1, 1), is_parameter = True,
					 initialize_with = np.array([[ 2.29220801]]),
					  name = "b1_weight")

	Zn = tf.matmul(W1,A_prev) + b1 
	Zn.setValue(np.array([[ 0.04153939, -1.11792545]]))

	An = activation(Zn)
	An.setValue(np.array([[ 0.51038336,  0.24639629]]))
	# print("evaluation = ",sess.evaluate(An))
	An.setCostGradient(np.array([[-0.41675785, -0.05626683]]))

	optimizer = tf.GradientDescentOptimizer(An, learning_rate = 1.2, name = 'optimizer')
	optimizer.backpropagate(An, first = 0) 
	if activation.__name__ == "sigmoid":
		dAn_prev =  np.array([[ 0.11017994,  0.0110534 ],
							 [ 0.09466817,  0.00949723],
							 [-0.05743092, -0.00576155]])
		dW =  np.array([[ 0.10266786 , 0.09778551, -0.01968084]])
		db =  np.array([[-0.05729622]])
		dAn = np.array([[-0.41675785, -0.05626683]])

	elif activation.__name__== "relu":
		dAn_prev =  np.array([[ 0.44090989, 0.        ],
							 [ 0.37883606,  0.        ],
							 [-0.2298228,   0.        ]])
		dW =  np.array([[ 0.44513825 , 0.37371418, -0.10478989]])
		db =  np.array([[-0.20837892]])
		dAn =  np.array([[-0.41675785, -0.05626683]])
	else:
		raise ValueError ("activation (%s) does not implementted for this test case "%activation.__name__)

	assert(tl.isVectorEqual(dAn_prev, A_prev.getCostGradient(),1e-8))
	assert(tl.isVectorEqual(dW,W1.getCostGradient(),1e-8))
	assert(tl.isVectorEqual(db,b1.getCostGradient(),1e-8))
	print("test_linear_activation_backward passed ")

	# print("Test Linear Activation backward with %s passed"%activation.__name__)
	# print("Test Linear Activation backward with %s"%activation.__name__,"*"*10)
	# print("dAn_prev = ",A_prev.getCostGradient())
	# print("dW = ",W1.getCostGradient())
	# print("db = ",b1.getCostGradient())
	# print("dAn = ",An.getCostGradient())
	# print("*"*10)


def test_layer_backward(my_graph,sess):
	y_train = np.array([[1, 0]])

	Y = tf.Constants(y_train, "Labels")
	A_prev1 = tf.Variables(shape = (3,2),
						 initialize_with = np.array([[ 0.09649747, -1.8634927 ],
													 [-0.2773882 , -0.35475898],
													 [-0.08274148, -0.62700068],
													 [-0.04381817, -0.47721803]]) ,
						 is_parameter = True, name = "A_prev" )
	
	W1 = tf.Variables(shape = (1,3),
					 initialize_with = np.array([[-1.31386475,  0.88462238,  0.88131804,  1.70957306],
												 [ 0.05003364, -0.40467741, -0.54535995, -1.54647732],
												 [ 0.98236743, -1.10106763, -1.18504653, -0.2056499 ]]),
					 is_parameter = True, name = "W1_weight" )

	b1 = tf.Variables(shape = (1, 1), is_parameter = True,
					 initialize_with = np.array([[ 1.48614836],
												 [ 0.23671627],
												 [-1.02378514]]),
					  name = "b1_weight")

	W2 = tf.Variables(shape = (1,3),
					 initialize_with = np.array([[-1.02387576,  1.12397796, -0.13191423]]),
					 is_parameter = True, name = "W2_weight" )

	b2 = tf.Variables(shape = (1, 1), is_parameter = True,
					 initialize_with = np.array([[-1.62328545]]),
					  name = "b2_weight")

	Z1 = tf.matmul(W1,A_prev1) + b1 
	A_prev2 = tf.relu(Z1)
	Z2 = tf.matmul(W2,A_prev2) + b2 
	AL = tf.sigmoid(Z2) 
	cost = tf.logistic_loss(AL,Y)

	optimizer = tf.GradientDescentOptimizer(cost, learning_rate = 1.2, name = 'optimizer')
	
	# set the values 
	Z1.setValue(np.array([[-0.7129932,   0.62524497],
						 [-0.16051336, -0.76883635],
						 [-0.23003072,  0.74505627]]))

	A_prev2.setValue(np.array([[ 1.97611078, -1.24412333],
							 [-0.62641691, -0.80376609],
							 [-2.41908317, -0.92379202]]))

	Z2.setValue(np.array([[ 0.64667545, -0.35627076]]))

	AL.setValue(np.array([[0.65626089 , 0.41186261]]))

	optimizer.backpropagate(cost, first = 1) 

	# set correct values 
	dW1  =  np.array([[ 0.39291384 , 0.07480025 , 0.13220188,  0.1006205 ],
					 [ 0.          ,0.         , 0.        ,  0.        ],
					 [ 0.05062228  ,0.00963712 , 0.01703264 , 0.01296376]])
	db2  =  np.array([[ 0.03406175]])
	dW2  =  np.array([[-0.59583722 ,-0.05785861,  0.22552905]])
	db1  =  np.array([[-0.21084807],
					 [ 0.        ],
					 [-0.02716527]])

	dA_prev2  =  np.array([[ 0.35194614, -0.42169614],
					 [-0.38635518 , 0.4629245 ],
					 [ 0.04534408 ,-0.05433054]])

	dA_prev1  =  np.array([[ 0.      ,    0.50067915],
					 [ 0.     ,    -0.31322025],
					 [ 0.     ,    -0.3072642 ],
					 [ 0.     ,    -0.7097473 ]])
	# perform assertions 
	assert(tl.isVectorEqual(dA_prev1, A_prev1.getCostGradient(),1e-8))
	assert(tl.isVectorEqual(dW1,W1.getCostGradient(),1e-8))
	assert(tl.isVectorEqual(db1,b1.getCostGradient(),1e-8))
	assert(tl.isVectorEqual(dA_prev2, A_prev2.getCostGradient(),1e-8))
	assert(tl.isVectorEqual(dW2,W2.getCostGradient(),1e-8))
	assert(tl.isVectorEqual(db2,b2.getCostGradient(),1e-8))

	print("test_layer_backward passed")
	# print("Test Layer backward Passed ")
	# print("dA_prev1 = ",A_prev1.getCostGradient())
	# print("dW1 = ",W1.getCostGradient())
	# print("db1 = ",b1.getCostGradient())
	# print("dA_prev2 = ",A_prev2.getCostGradient())
	# print("dW2 = ",W2.getCostGradient())
	# print("db2 = ",b2.getCostGradient())
	# print("*"*10)


def test_update_parameters(my_graph,sess):
	y_train = np.array([[1, 0]])

	Y = tf.Constants(y_train, "Labels")
	A_prev1 = tf.Variables(shape = (4,2),
						 initialize_with = None ,
						 is_parameter = False, name = "A_prev" )
	
	W1 = tf.Variables(shape = (1,3),
					 initialize_with = np.array([[-0.41675785, -0.05626683, -2.1361961 ,  1.64027081],
													 [-1.79343559, -0.84174737,  0.50288142, -1.24528809],
													 [-1.05795222, -0.90900761,  0.55145404,  2.29220801]]),
					 is_parameter = True, name = "W1_weight" )

	b1 = tf.Variables(shape = (1, 1), is_parameter = True,
					 initialize_with = np.array([[ 0.04153939],
												 [-1.11792545],
												 [ 0.53905832]]),
					  name = "b1_weight")

	W2 = tf.Variables(shape = (1,3),
					 initialize_with = np.array([[-0.5961597,  -0.0191305 ,  1.17500122]]),
					 is_parameter = True, name = "W2_weight" )

	b2 = tf.Variables(shape = (1, 1), is_parameter = True,
					 initialize_with = np.array([[-0.74787095]]),
					  name = "b2_weight")

	Z1 = tf.matmul(W1,A_prev1) + b1 
	A_prev2 = tf.relu(Z1)
	Z2 = tf.matmul(W2,A_prev2) + b2 
	AL = tf.sigmoid(Z2) 
	cost = tf.logistic_loss(AL,Y)

	optimizer = tf.GradientDescentOptimizer(cost, learning_rate = 0.1, name = 'optimizer')
	
	# set the values 
	W1.setCostGradient(np.array([[ 1.78862847 , 0.43650985,  0.09649747, -1.8634927 ],
								 [-0.2773882 , -0.35475898, -0.08274148, -0.62700068],
								 [-0.04381817 ,-0.47721803, -1.31386475,  0.88462238]]))
	b2.setCostGradient(np.array([[ 0.98236743]]))
	W2.setCostGradient(np.array([[-0.40467741 , -0.54535995 ,-1.54647732]]))
	b1.setCostGradient(np.array([[ 0.88131804],
								 [ 1.70957306],
								 [ 0.05003364]]))

	optimizer.updateweights(AL) 

	# set correct values 
	W1_value  =  np.array([[-0.59562069 ,-0.09991781 ,-2.14584584 , 1.82662008],
					 [-1.76569676, -0.80627147 , 0.51115557 ,-1.18258802],
					 [-1.0535704  ,-0.86128581 , 0.68284052 , 2.20374577]])
	b1_value  =  np.array([[-0.04659241],
					 [-1.28888275],
					 [ 0.53405496]])

	W2_value  =  np.array([[-0.55569196 , 0.0354055 ,  1.32964895]])
	b2_value  =  np.array([[-0.84610769]])

	# perform assertions 
	
	assert(tl.isVectorEqual(W1_value,W1.getValue(),1e-7))
	assert(tl.isVectorEqual(b1_value,b1.getValue(),1e-7))
	assert(tl.isVectorEqual(W2_value,W2.getValue(),1e-7))
	assert(tl.isVectorEqual(b2_value,b2.getValue(),1e-7))
	print("test_update_parameters passed")

	# print("Test Layer backward Passed ")
	# print("W1 = ",W1.getValue())
	# print("b1 = ",b1.getValue())
	# print("W2 = ",W2.getValue())
	# print("b2 = ",b2.getValue())
	# print("*"*10)


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
	n_h = 4 
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
	A1 = tf.tanh(Z1)
	Z2 = tf.matmul(W2,A1) + b2 
	An = tf.sigmoid(Z2)
	# An = tf.relu( An_1 ) 
	assert(An.getShape() == Y_train.shape)

	cost  = tf.logistic_loss(logits = An, labels = Y) 
	optimizer = tf.GradientDescentOptimizer(cost, learning_rate = 1.2, name = 'optimizer')
	return cost, optimizer , An 

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

def runGradientChecking(my_graph, sess,cost,optimizer,tol = 1e-7):
	# print("Running Gradient Checking ","@"*10)
	for i in range(10):
		theta = tl.convertAllParametersToVector(my_graph.parameter_list,my_graph.all_variables)
		# print("Starting with Theta = ")
		# print(theta)
		# ############### Perform gradient checking 
		tl.GradCheck(theta,sess,cost,optimizer,my_graph,tol)
	# print("Completed Gradient Checking ", "@"*10)
		

def runIteration(optimizer,cost, niter = 1):
	# run iteration 
	print("Running Iteration Checking ","@"*10)
	cost_list = []
	for i in range(niter):
		sess.run(optimizer)
		cost_list.append(np.squeeze(cost.getValue()))
		print("****************************")
		print("iteration = (%d)"%i)
		print("cost = ",np.squeeze(cost.getValue()))
	print("Completed Running Iteration Checking ","@"*50)
	return cost_list 

def testRunGradCheck(my_graph,sess):
	X_train, Y_train = load_planar_dataset("planar_dataset.csv")

	my_graph.initialize()
	cost, optimizer, An = LinearLogistic(X_train,Y_train)
	runGradientChecking(my_graph,sess,cost,optimizer)

	my_graph.initialize()
	cost, optimizer, An = LinearTanh(X_train,Y_train)
	runGradientChecking(my_graph,sess,cost,optimizer, tol = 1e-6)

	print("test grad check passed")


def test():
	test_forwardprop(my_graph,sess)
	my_graph.initialize()
	test_compute_cost(my_graph,sess)
	my_graph.initialize()
	test_linear_backward(my_graph,sess)
	my_graph.initialize()
	test_linear_activation_backward(my_graph,sess,tf.sigmoid)
	my_graph.initialize()
	test_linear_activation_backward(my_graph,sess,tf.relu)
	my_graph.initialize()
	test_layer_backward(my_graph,sess)
	my_graph.initialize()
	test_update_parameters(my_graph,sess)

	
	testRunGradCheck(my_graph,sess)

	print("tests passed !!!")
	# runGradientChecking(my_graph)

# Create a graph 
my_graph = tf.ComputationGraph()
my_graph.set_as_default()

# Create session 
sess = tf.Session()






if __name__ == '__main__':
	test() 

if 0: 
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