
from copy import deepcopy 
import numpy as np 
import matplotlib.pyplot as plt 



def isVectorEqual(v1,v2, tol = 1e-30):
	return np.all(abs(v1 - v2) < tol)

def printGraph(last_variable,indent = 1):
	graph = last_variable.getGraph()
	if graph == []:
		print('')
		return 
	node = graph[0]
	n1 = node[0].getType()
	print(n1,'(',end = '')
	for item in node[1]:
		print(item.getType(),end = ',')
	print(')')

	count = 0 
	for item in node[1]:
		count += 1
		print('-'*(indent + len(n1)) + '>(%d)|'%count, end = '')
		printGraph(item, indent + len(n1))


def GradCheck(theta, sess,cost,optimizer,my_graph, tol = 1e-7):
	# Perform gradient computation using finite difference 
	# theta = np.array([[ 1.61434536,-0.71175641]])
	dtheta = np.zeros(theta.shape)
	epsilon = 1e-7
	for i in range(dtheta.shape[0] * dtheta.shape[1]):
		theta_plus  = np.copy(theta)
		theta_plus[0,i] += epsilon 
		theta_minus = np.copy(theta)
		theta_minus[0,i] -= epsilon 
		convertVectorToAllParameters(theta_plus,my_graph.parameter_list, my_graph.all_variables)
		J_plus  = sess.evaluate(cost)
		convertVectorToAllParameters(theta_minus,my_graph.parameter_list, my_graph.all_variables)
		J_minus = sess.evaluate(cost)
		
		dtheta[0,i] = (J_plus - J_minus)/(2 * epsilon)
		# print("dtheta = ",dtheta[0,i])
		# print("J_plus = ",J_plus)
		# print("J_minus = ",J_minus)
		# print("epsilon = ", epsilon)
		# print("(J_plus - J_minus)/(2 * epsilon)" ,(J_plus - J_minus)/(2 * epsilon))
		# print("J_plus = (%f) , J_minus = (%f)"%(J_plus, J_minus),(J_plus - J_minus)/(2 * epsilon), "------------------------>")

	convertVectorToAllParameters(theta,my_graph.parameter_list, my_graph.all_variables)

	sess.run(optimizer)
	mydtheta = convertAllParametersGradientToVector(my_graph.parameter_list,my_graph.all_variables)
	gradcheck = np.linalg.norm((mydtheta - dtheta))/(np.linalg.norm(mydtheta) + np.linalg.norm(dtheta))

	# print("*"*50)
	# print("Dtheta = ")
	# print(dtheta)
	# print("MyDtheta = ")
	# print(mydtheta)
	# print("*"*50)
	# print('gradcheck gives = ',gradcheck)
	# print("at cost = ",cost.getValue())
	assert(gradcheck < tol)



def convertAllParametersToVector(parameter_list,all_variables):
	""" returns a vector of all the parameters in parameter dictionary """

	allkeys = deepcopy(parameter_list)
	allkeys.sort() 

	vect = np.array([[]])

	for item in allkeys:
		matrix = all_variables[item].getValue()
		# print(matrix)
		vect = np.hstack((vect,np.reshape(matrix,(1,matrix.shape[0] * matrix.shape[1]))))

	return vect 

def convertAllParametersGradientToVector(parameter_list,all_variables):
	""" returns a vector of all the parameters in parameter dictionary """

	allkeys = deepcopy(parameter_list)
	allkeys.sort() 

	vect = np.array([[]])

	for item in allkeys:
		matrix = all_variables[item].getCostGradient()
		# print(matrix)
		vect = np.hstack((vect,np.reshape(matrix,(1,matrix.shape[0] * matrix.shape[1]))))

	return vect 

def convertVectorToAllParameters(vect,parameter_list,all_variables):
	allkeys = deepcopy(parameter_list)
	allkeys.sort() 

	cstart = 0 
	for item in allkeys:
		shape = all_variables[item].getShape()
		# print("shape = ",shape)
		cstop = cstart + (shape[0] * shape[1]) 
		# print("cstart = ",cstart," cstop  = ",cstop)
		kk = vect[:,cstart:cstop]
		# print(kk)
		matrix = np.reshape(kk,shape)
		all_variables[item].setValue(matrix)
		cstart = cstop 



def plot_decision_boundary(prediction_function,model_output,model_x_input,sess, X,y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid

    test_set = np.transpose(np.c_[xx.ravel(), yy.ravel()])
    Z = prediction_function(model_output,test_set) 
    
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.flatten(), cmap=plt.cm.Spectral)


