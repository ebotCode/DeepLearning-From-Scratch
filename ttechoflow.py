import numpy as np 

from copy import deepcopy 
import pickle 
import os 

CURRENT_GRAPH = [None]

# nav(class AbstractVariables)
class AbstractVariables:

	def __init__(self):
		self.vtype = None 
		self.value = None 
		self.id = None 
		self.graph = None 
		self.is_parameter = False 
		self.name = None 
		self.cost_gradient = None 
		self.tshape = None 

		



	def initializeGradient(self):
		self.cost_gradient = np.zeros(self.getShape())

	def addToCostGradient(self, add_gradient):
		""" adds 'add_gradient' to the current value of the cost gradient """
		self.setCostGradient(self.getCostGradient() + add_gradient)

	def getCostGradient(self):
		""" returns the cost gradient """
		return self.cost_gradient 

	def getName(self):
		return self.name 

	def getValue(self):
		return self.value 

	def isParameter(self):
		return self.is_parameter 

	def isGraphNode(self):
		""" returns true if self is a GraphNode. GraphNodes are Variables that 
		carry some encoding of the graph. 
		"""
		raise Exeption("isGraphNode is deprecated. use isParameter instead") #return (len(self.getGraph()) > 0)

	def getGraph(self):
		""" returns the computation graph that produced this variable """
		return self.graph  

	def getValue(self):
		""" returns the value contained in Variables """ 
		return self.value 

	def getId(self):
		""" returns the unique id of the variable """
		return self.id

	def getShape(self):
		if len(self.getValue().shape) !=  0: 
			return self.getValue().shape 
		else: # return previous cached. 
			return self.tshape 

	def getType(self):
		return self.vtype 

	def setCostGradient(self,cost_gradient):
		""" sets the value of cost_gradient to cost_gradient """
		self.cost_gradient = np.copy(cost_gradient)

	def setValue(self,value):
		""" sets the attribute 'value' equal to value  """
		self.value = np.copy(value)

	def setTShape(self,shape):
		self.tshape = shape 


# nav(class Variables)
class Variables(AbstractVariables):
	""" 
	Variables definition. 
	all quantities resulting from operations (e.g matmul, relu,...)
	output Variables. The only constants are Training inputs and 
	Training outputs 
	"""
	current_graph = CURRENT_GRAPH
	id_no = 0 
	def __init__(self, shape = (1,1), initialize_with = None, name = None,is_parameter = False ):
		self.vtype = "Variables"
		self.is_parameter = is_parameter 
		self.name  = name 
		self.value = None 
		self.cost_gradient = None 
		self.graph = []
		self.setValueAndShape(shape, initialize_with)
		self.id  = "V%d"%Variables.id_no 

		Variables.current_graph[0].addVariableToGraph(self)
		Variables.id_no += 1 
		self.initializeGradient()

	def addOperationToGraph(self,operation_node):
		""" adds the operation_node to graph. 
			note: operation_node is a tuple of format ('<function reference>',(*function parameters))
		"""
		self.graph.append(operation_node)

	def extendGraph(self,graph):
		"""
		extendGraph the graph with a given graph 
		"""
		self.graph.extend(deepcopy(graph))


	def setValueAndShape(self,shape,initialize_with):
		if initialize_with is None:
			self.setValue(np.zeros(shape))
			self.setTShape(shape)
		else:
			self.setValue(initialize_with)
			self.setTShape(initialize_with.shape)
		
		
	def __add__(self,variable2):
		return overloadOperators(self,variable2,add)

	def __radd__(self,variable2):
		return overloadOperators(variable2,self,add)

	def __sub__(self,variable2):
		return overloadOperators(self,variable2,sub)

	def __rsub__(self,variable2):
		return overloadOperators(variable2,self,sub)


	def __mul__(self,variable2):
		return overloadOperators(self,variable2,elmult)

	def __rmul__(self,variable2):
		return overloadOperators(variable2,self,elmult)

# nav(class Constants)
class Constants(Variables):
	"""
	Constants.
	quantities that do not change in the neural network 
	"""
	current_graph = CURRENT_GRAPH 
	id_no = 0 
	def __init__(self, input_array, name = None , is_parameter = False ):
		self.vtype = "Constants"
		self.is_parameter = False  
		self.name  = name 
		self.setValue(input_array)
		self.id = "C%d"%Constants.id_no 
		self.graph = []
		
		Constants.current_graph[0].addVariableToGraph(self)
		Constants.id_no += 1

	def addToCostGradient(self,cost_gradient):
		pass 

	def addOperationToGraph(self,operation_node):
		""" adds the operation_node to graph. 
			note: operation_node is a tuple of format ('<function reference>',(*function parameters))
		"""
		raise Exception("Operation Not implemented for constants ") 

	def extendGraph(self,graph):
		"""
		extendGraph the graph with a given graph 
		"""
		raise Exception("Operation Not implement for constants") 


# nav(class ComputationGraph)
class ComputationGraph:
	def __init__(self):
		self.initialize()

	def initialize(self):
		self.all_variables = {}  # a dictionary of id:object_reference
		self.parameter_list = [] # list of ids of parameters
		self.optimizer_list = []
		self.name_ids = {}  # a dictionary of "name":id 

	def getAllVariables(self):
		return self.all_variables 

	def addVariableToGraph(self,variable1):
		self.all_variables[variable1.getId()] = variable1 
		if variable1.isParameter():
			self.parameter_list.append(variable1.getId())
		if variable1.getName():
			if variable1.getName() not in self.name_ids:
				self.name_ids[variable1.getName()] = variable1.getId()
			else:
				raise KeyError("Variable with name (%s) already exist. please use another name"%variable1.getName())

	def addOptimizerToGraph(self,optimizer):
		self.all_variables[optimizer.getId()] = optimizer 
		self.optimizer_list.append(optimizer.getId())
		if optimizer.getName():
			if optimizer.getName() not in self.name_ids:
				self.name_ids[optimizer.getName()] = optimizer.getId()
			else:
				raise KeyError("Optimizer with name (%s) already exist. Please use another optimizer name"%optimizer.getName())

	def getVariableByName(self,variable_name):
		if variable_name in self.name_ids:
			return self.all_variables[self.name_ids[variable_name]]
		else:
			raise KeyError("Variable  name (%s) does not exist in variable space "%variable_name)



	def set_as_default(self):
		CURRENT_GRAPH[0] = self 

# nav(class optimizer)

class GradientDescentOptimizer:

	id_no =0 
	def __init__(self, last_variable, learning_rate = 0.1, name = None):
		self.last_variable = last_variable  
		self.learning_rate = learning_rate 
		self.id = "Optimizer%d"%GradientDescentOptimizer.id_no 
		self.name = name 

		GradientDescentOptimizer.id_no += 1
		CURRENT_GRAPH[0].addOptimizerToGraph(self)

	def getId(self):
		return self.id 

	def getName(self):
		return self.name 

	def getLastVariable(self):
		return self.last_variable 

	def optimize(self):
		""" performs backpropagation through the network, and
			updates the values of the weights 
			Note. 
		"""
		self.forwardpropagate(self.getLastVariable())     # forward propagation
		self.backpropagate(self.getLastVariable(), first = 1) # backward propagation 
		self.updateweights(self.getLastVariable())			   # update values of parameters 

	def updateweights(self,last_variable):
		""" updates the values of variables that are considered parameters """
		if last_variable.isParameter(): # update only variables that are parameters.
			# print("updating parameters") 
			# print("Previous value = ")
			# print(last_variable.getValue())
			# print("Gradient Value = ")
			# print(last_variable.getCostGradient())

			last_variable.setValue(last_variable.getValue() - self.learning_rate * last_variable.getCostGradient())
			last_variable.initializeGradient()
		# recurse
		graph = last_variable.getGraph()
		if graph != []: # if graph is not empty 
			node = graph[0]
			for item in node[1]:
				self.updateweights(item)

	def forwardpropagate(self,last_variable,nntype = "train"):
		""" runs forward propagation on the network """
		graph = last_variable.getGraph()
		last_variable.initializeGradient()    # initialize the gradient to zero. 
		if graph == []:
			return last_variable.getValue()
		else:
			node = graph[0]
			output = node[0].forward(nntype,*(self.forwardpropagate(var)*1 for var in node[1]))
			last_variable.setValue(output)
			return output

	def backpropagate(self,last_variable, first , nntype = "train"):
		""" performs backward propagation starting with last_variable. 
		Note: 
			
		"""
		graph = last_variable.getGraph()
		if graph == []:
			return 
		else:
			node = graph[0]
			cg = np.ones(last_variable.getShape())  if first else last_variable.getCostGradient()*1 
			grads = node[0].backward(nntype,cg,last_variable.getValue() * 1,*(var.getValue()*1 for var in node[1]))
			for i in range(len(node[1])):
				node[1][i].addToCostGradient(grads[i])
			for item in node[1]:
				self.backpropagate(item, first = 0)



# nav(class Session)
class Session:
	def __init__(self):
		pass 

	def evaluate(self,last_variable,nntype = "predict"):
		""" performs forward propagation through the network """
		graph = last_variable.getGraph()
		if graph == []:
			return last_variable.getValue()
		else:
			node = graph[0]
			output = node[0].forward(nntype,*(self.evaluate(var)*1 for var in node[1]))
			last_variable.setValue(output)
			return output 

	def run(self,optimizer):
		""" runs an iteration of the optimizer """
		optimizer.optimize()

# nav(class Saver)
class Saver:
	def __init__(self):
		self.extension = '.tmdl'


	def saveGraph(self,my_graph, modelname, dir_name = ".\\modeldir", overwrite = False ):
		filename_with_path = os.path.join(dir_name,modelname) + self.extension 
		if not os.path.isdir(dir_name):
			os.mkdir(dir_name)

		if  os.path.isfile(filename_with_path) and (overwrite == False) :
			raise Exception("model (%s) already exist in (%s). Use another name"%(modelname,dir_name))
		ivalues = {}
		with open(filename_with_path,'wb') as f:
			# don't save the data in the input. Only the weights 
			all_vars = my_graph.getAllVariables()
			for item in all_vars:
				p = all_vars[item]
				if (isinstance(p,Variables) or isinstance(p,Constants)):
					if not p.isParameter():
						ivalues[item] = p.getValue()
						p.setValue(None)

			save_value = {"my_graph":my_graph}
			pickle.dump(save_value,f)

		# return the values back to the input and output. 
		for item in ivalues:
			p = all_vars[item]
			p.setValue(ivalues[item])

		print("Model (%s) saved Successfully in (%s)"%(modelname,dir_name))

	def loadGraph(self,modelname,dir_name = ".\\modeldir"):
		filename_with_path = os.path.join(dir_name,modelname) + self.extension 
		if not os.path.isfile(filename_with_path):
			raise Exception("No Model named (%s) exists  "%(filename_with_path))

		my_graph = None 
		with open(filename_with_path,'rb') as f:
			save_value = pickle.load(f)
			my_graph = save_value["my_graph"]
		if my_graph is None:
			raise Exception("Could not load graph model (%s)"%modelname)
		return my_graph 




# nav(class Operations)
class Operations:
	""" Performs low level matrix multiplication operations for 
	 	forward and backward propagation 
	 """
	def __init__(self):
		self.vtype = "ops"
		pass 

	def forward(self,nntype,*args):
		pass 
	def backward(self,nntype,output_grad,output,*args):
		pass

	def getType(self):
		return self.vtype 

# nav(class OperationMatrixMultiply)
class OperationMatrixMultiply(Operations):
	def __init__(self):
		self.vtype = "matmul"

	def forward(self,nntype,*args):
		m1,m2 = args 
		return np.dot(m1,m2)

	def backward(self,nntype,output_grad,output,*args_value):
		m1,m2 = args_value
		msize = m2.shape[1]
		# print(m1.shape, ' and ', m2.shape, 'and ',output.shape, 'and', output_grad.shape)
		return ((1/msize)*np.dot(output_grad,m2.T), 
			    np.dot(m1.T,output_grad))


# nav(class OperationElmult)
class OperationElmult(Operations):
	def __init__(self):
		self.vtype = "elmult"

	def forward(self,nntype,*args):
		m1,m2 = args 
		return m1 * m2 

	def backward(self,nntype, output_grad, output, *args_value):
		m1,m2 = args_value 
		return (output_grad * m2, output_grad*m1 )

# nav(class OperationLog)
class OperationLog(Operations):
	def __init__(self):
		self.vtype = "log"

	def forward(self,nntype,*args):
		m1 = args[0]
		return np.log(m1)

	def backward(self,nntype,ouput_grad,output,*args_value):
		m1 = args_value[0]
		return (output_grad * (1 / m1),)
		
# nav(class OperationAdd)
class OperationAdd(Operations):
	def __init__(self):
		self.vtype = "Add"

	def compute_grad(self,m1,output_grad,output):
		# print("Add shapes = ",m1.shape, output.shape )
		if m1.shape == output.shape:
			parm1 = output_grad * 1 
		elif (m1.shape != output.shape) and (m1.shape[0] == output.shape[0]):
			msize = output.shape[1]
			parm1 = (1/msize)*np.sum(output_grad,axis = 1, keepdims = True)
		else:
			raise ValueError('Inavlid input for bbackpropagate %s operation'%self.getType())
		return parm1 

	def forward(self,nntype,*args):
		m1,m2 = args 
		return m1 + m2 

	def backward(self,nntype,output_grad,output, *args_value):
		m1,m2 = args_value 	
		parm1 = self.compute_grad(m1,output_grad,output)
		parm2 = self.compute_grad(m2,output_grad,output)
		return (parm1 , parm2)

# nav(class OperationSubtract)
class OperationSubtract(OperationAdd):
	def __init__(self):
		self.vtype = "Subtract"

	def forward(self,nntype,*args):
		m1,m2 = args 
		return m1 - m2 

	def backward(self,nntype,output_grad,output, *args_value):
		m1,m2 = args_value 
		parm1 = self.compute_grad(m1,output_grad,output)
		parm2 = (-1)*self.compute_grad(m2,output_grad,output)
		print("Came to subtract ooo")
		return (parm1 , parm2)

# nav(class OperationNegate)
class OperationNegate(Operations):
	def __init__(self):
		self.vtype = "negate"

	def forward(self,nntype,*args):
		m1 = args[0]
		return -m1 

	def backward(self,nntype,output_grad,output, *args_value):
		m1 = args_value[0]
		return (output_grad *(-1),)

# nav(class OperationRelu)
class OperationRelu(Operations):
	def __init__(self):
		self.vtype = "Relu"

	def forward(self,nntype,*args):
		m1 = args[0]
		return m1 * ( (m1 > 0) * 1 )

	def backward(self,nntype,output_grad,output, *args_value):
		m1 = args_value[0]
		return (output_grad * ((m1 > 0)*1),)

# nav(class OperationSigmoid)
class OperationSigmoid(Operations):
	def __init__(self):
		self.vtype = "Sigmoid"

	def forward(self,nntype,*args):
		m1 = args[0]
		return 1/(1 + np.exp(-m1))

	def backward(self,nntype,output_grad,output, *args_value):
		m1 = args_value[0]
		return (output_grad * (output* (1 - output)),)

# nav(class OperationSoftmax)
class OperationSoftmax(Operations):
	def __init__(self):
		self.vtype = "Softmax"

	def forward(self,nntype,*args):
		m1 = args[0]
		return np.exp(m1)/np.sum(np.exp(m1) , axis = 0, keepdims = True)

	def backward(self,nntype,output_grad,output, *args_value):
		m1 = args_value[0]
		parm1 = 1 - output #1/(np.sum(np.exp(m1),axis = 0, keepdims = 1))
		parm2 = output * parm1 
		return (output_grad * parm2,)

# nav(class OperationTanh)
class OperationTanh(Operations):
	def __init__(self):
		self.vtype = "tanh"

	def forward(self,nntype,*args):
		m1 = args[0]
		return np.tanh(m1)

	def backward(self,nntype,output_grad, output, *args_value):
		m1 = args_value[0]
		parm1 = 1/ (np.cosh(m1)**2)
		return (output_grad * parm1,)



# # nav(class OperationReduceSum)
# class OperationReduceAbsSum(Operations):
# 	def __init__(self):
# 		self.vtype = "reduce_sum"

# 	def forward(self,*args):
# 		m1 = args[0]
# 		return (1/msize)*np.sum(np.abs(m1))

# 	def backward(self,ouput_grad, output, *args_value):
# 		m1 = args_value[0]
# 		return (output_grad * np.ones(m1.shape),)

# nav(class OperationSumOfSquaresLoss)
class OperationSumOfSquaresLoss(Operations):
	def __init__(self):
		self.vtype= "sum_of_squares_loss"

	def forward(self,nntype, *args):
		y_hat,y = args
		msize = y.shape[1]
		ll = (1/(2 * msize)) * np.sum(np.square(y_hat - y))
		return np.array([[ll]])

	def backward(self,nntype,output_grad, output, *args_value):
		# print("*"*50)
		# print("at sum of squared ")
		# print("output  = ")
		# print(output)
		# print("output_grad = ",output_grad)
		# print("*"*50)
		y_hat,y = args_value
		msize = y.shape[1]
		return (output_grad * (-1)*(y - y_hat),output_grad * (y - y_hat))

# nav(class OperationLogisticLoss)
class OperationLogisticLoss(Operations):
	""" Implements logistic loss for binary class classification """
	def __init__(self):
		self.vtype = "logistic_loss"

	# def forward(self,*args):
	# 	Y_hat,Y = args
	# 	m = Y_hat.shape[1]
	# 	ll = (1/m)*np.sum(-Y * np.log(Y_hat) + (( 1 - Y) * np.log(1 - Y_hat)))
	# 	return np.array([[ll]])

	# def backward(self,output_grad, output, *args_value):
	# 	y_hat,y = args_value 
	# 	m = y_hat.shape[1]
	# 	dJdl = (1/m)*((-y / y_hat) - (1 - y)/(1 - y_hat))
	# 	dJdy = (1/m)*(-np.log(y_hat) - np.log(1 - y_hat))
	# 	return (dJdl,dJdy)

	def forward(self,nntype,*args):
		Y_hat,Y = args
		m = Y_hat.shape[1]
		ll = (-1/m)*np.sum(Y * np.log(Y_hat) + (( 1 - Y) * np.log(1 - Y_hat)))
		return np.array([[ll]])

	def backward(self,nntype,output_grad, output, *args_value):
		y_hat,y = args_value 
		m = y_hat.shape[1]
		dJdl = (-1)*((y / y_hat) - (1 - y)/(1 - y_hat))
		dJdy = np.zeros(y_hat.shape) # (-1)*(np.log(y_hat) - np.log(1 - y_hat))
		return (output_grad * dJdl, output_grad * dJdy)

# nav(class OperationCrossEntropyLoss)
class OperationCrossEntropyLoss(Operations):
	""" Implements cross entropy loss with one hot encoding """
	def __init__(self):
		self.vtype = "cross_entropy_loss"

	def forward(self,nntype,*args):
		Y_hat,Y = args
		m = Y_hat.shape[1]
		ll =  - (1/m)*np.sum(Y * np.log(Y_hat))
		return np.array([[ll]])

	def backward(self,nntype,output_grad, output, *args_value):
		y_hat , y = args_value 
		m = y_hat.shape[1]
		dJdl = -(1/m) * (y / y_hat)
		return (dJdl,)


class OperationL2Norm(Operations):
	def __init__(self, lambda_value,no_of_train_samples ):
		self.vtype = "L2Norm"
		self.lambda_value = lambda_value 
		self.msize = no_of_train_samples   # no of train samples 

	def forward(self, nntype,*args):
		m1 = args[0]
		ll = (self.lambda_value / (2*self.msize)) * np.sum(np.square(m1))
		return np.array([[ll]])

	def backward(self, nntype , output_grad, output, *args_value):
		m1 = args_value[0]
		return ( output_grad * (self.lambda_value / self.msize) * m1 , )

# nav(class OperationDropout)
class OperationDropout(Operations):
	def __init__(self,keepprops ):
		self.vtype = "Dropout"
		self.keepprops = keepprops 
		self.mask = None 

	def forward(self,nntype,*args):
		m1 = args[0]
		shape = m1.shape 
		if nntype == "train":
			self.mask = (np.random.rand(*shape) < self.keepprops) * 1 
		elif nntype == "predict":
			self.mask = np.ones(shape)
		else: 
			raise ValueError( "Invalid nntype in Droput ",nntype )

		return (m1 * self.mask/self.keepprops) 

	def backward(self,nntype,output_grad,output,*args_value):
		if nntype == "train":
			return (output_grad * self.mask/self.keepprops,)
		elif nntype == "predict":
			return (output_grad,)
		else: 
			raise ValueError( "Invalid nntype in Droput ",nntype )


# nav(function overloadOperators)
def overloadOperators(variable1,variable2,operation):
	if (isinstance(variable1,float ) or isinstance(variable1, int)) and \
	   (isinstance(variable2,AbstractVariables)):
		tmp_variable1 = Variables(initialize_with = np.ones(variable2.getShape()) * variable1)
		return operation(tmp_variable1,variable2)
	elif (isinstance(variable2,float ) or isinstance(variable2, int)) and (isinstance(variable1,AbstractVariables)):
		tmp_variable2 = Variables(initialize_with = np.ones(variable1.getShape()) * variable2)
		return operation(variable1,tmp_variable2)
	elif (isinstance(variable1,AbstractVariables) and isinstance(variable2,AbstractVariables)):
		return operation(variable1,variable2)
	else:
		raise ValueError("Two floats / integers should not be here (%s) (%s)"%(type(variable1),type(variable2)))

# nav(function initializations)
def initializations(init_type,shape):
	if init_type == 'Zeros':
		return np.zeros(shape)
	elif init_type == 'Normal':
		return np.random.randn(*shape) * 0.01 
	elif init_type == 'He':
		n = shape[0]
		return np.random.randn(*shape) * (1 /n) # note this works for 2 dim only 
	elif init_type == 'Xavier':
		n = shape[0]
		return np.random.randn(*shape) * np.sqrt(1/(n))
	else:
		raise ValueError ("Invalid initialization type. use any of Zeros, Normal, He, Xavier")



# nav(function setupNewVariable)
def setupNewVariable(shape,operation,variables, name = None):
	"""
		sets up a new variable, and adds the operation [operation(variables)] to the graph 
		of the new variable 
		Args: 
			variables: tuple of Variables 
			operation: an operation 
			shape :    shape of the new variable 
	"""
	new_variable = Variables(shape = shape ,name = name)
	prev_graph = [] #selectValidGraph(variables)
	new_variable.extendGraph(prev_graph)
	new_variable.addOperationToGraph((operation,variables))
	return new_variable 

#############################################################################################################################
"""
From HERE ON, 
Note: 
	when designing graph routines (routines below,) the approach is to maintain 
		inputs(variable1,variable2) ----> [Operation] ---> output 

		where variable1 and variable2 come together through the  operation to form the output 
		Note>> the corresponding backprop must give as many grads as there are inputs. 
		If this is not done, This would result in error 
"""
# nav(function matmul)
def matmul(variable1, variable2, name = None):
	""" 
	returns the result of matrix multiplication between variable1 and variable2
	args:
		variable1: Variables of shape = (m,n)
		variable2: Variables of shape = (n,q)
	
	returns: 
		Variables of shape (m,q)
	"""
	assert(isinstance(variable1,AbstractVariables	))
	assert(isinstance(variable2,AbstractVariables	))

	if variable1.getShape()[1] != variable2.getShape()[0]:
		raise ArithmeticError ("the number of columns in variable 1 must match number of rows in variable 2")

	new_variable = setupNewVariable((variable1.getShape()[0],variable2.getShape()[1]),
									OperationMatrixMultiply(),
									(variable1,variable2), name = name)
	return new_variable 


# nav(function add)
def add(variable1, variable2, name = None):
	""" 
		returns the result of adding variable2 to variable 1. 
		args: 
			variable1: variables of shape (m,...)
			variable2: variables of shape (m,...)
		Note: Subsequent dimensions must match in a form that is broad castable 
	"""
	if variable1.getShape()[0] != variable2.getShape()[0]:
		raise ArithmeticError("the first dimension of variable1 and variable2 must match")

	new_variable = setupNewVariable(max([variable1.getShape(),variable2.getShape()]),
											OperationAdd(),(variable1,variable2),name = name)
	return new_variable 

# nav(function sub)
def sub(variable1,variable2,name = None):
	if variable1.getShape()[0] != variable2.getShape()[0]:
		raise ArithmeticError("the first dimension of variable1 and variable2 must match")

	new_variable = setupNewVariable(max([variable1.getShape(),variable2.getShape()]),
											OperationSubtract(),(variable1,variable2), name = name)
	return new_variable 

# nav(function elmult)
def elmult(variable1,variable2,name = None):
	if (variable1.getShape() == variable2.getShape()) or \
	   (variable1.getShape() == (1,1)) or\
	   (variable2.getShape() == (1,1)):
		new_variable = setupNewVariable(max([variable1.getShape(), variable2.getShape()]),
											OperationElmult(),(variable1,variable2), name = name)
	else:
		raise ArithmeticError("dimensions of variable1 and variable2 must match for element multiplication")

	return new_variable 

# nav(function neg)
def neg(variable1,name = None):
	# print("camae to negate")
	return setupNewVariable(variable1.getShape(),OperationNegate(),(variable1,),name)

# nav(function relu)
def relu(variable1,name = None):

	return setupNewVariable(variable1.getShape(),OperationRelu(),(variable1,),name)

# nav(function sigmoid)
def sigmoid(variable1,name = None):
	return setupNewVariable(variable1.getShape(),OperationSigmoid(),(variable1,),name)

# nav(function softmax)
def softmax(variable1,name = None):
	return setupNewVariable(variable1.getShape(),OperationSoftmax(),(variable1,),name)

# nav(function tanh)
def tanh(variable1,name =None ):
	return setupNewVariable(variable1.getShape(),OperationTanh(),(variable1,),name)

# nav(function log)
def log(variable1,name = None):
	return setupNewVariable(variable1.getShape(),OperationLog(),(variable1,),name)

def sum_of_squares_loss(logits,labels,name = None):
	return setupNewVariable((1,1),OperationSumOfSquaresLoss(),(logits,labels),name)

# nav(reduce_sum)
# def reduce_sum(variable1,name = None):
# 	return setupNewVariable((1,1),OperationReduceSum(),(variable1,),name)

def logistic_loss(logits, labels,name= None):
	return setupNewVariable((1,1),OperationLogisticLoss(),(logits,labels),name)

def cross_entropy_loss(logits, labels,name = None):
	return setupNewVariable((1,1),OperationCrossEntropyLoss(),(logits,labels),name)

def dropout(variable1, keepprops,name = None):
	return setupNewVariable(variable1.getShape(),OperationDropout(keepprops),(variable1,),name)

def l2norm(variable1, lambda_value, no_of_train_samples, name = None):
	return setupNewVariable((1,1),OperationL2Norm(lambda_value, no_of_train_samples),(variable1,), name) 








