import numpy as np 
import ttechoflow as tf 

# Tutorial on logistic regression 

x_train = np.array([[0.1,0.2,0.4,0.8,0.3],
					[0.3,0.4,0.6,0.2,0.1],
					[0.2,0.0,0.4,0.2,0.9]])
y_train = np.array([[0, 0,  1,  1, 0]] )

# create graph 
my_graph = tf.ComputationGraph()
my_graph.initialize()
my_graph.set_as_default()

# create constants 
X = tf.Constants(x_train, name = "Input")
Y = tf.Constants(y_train, name = "Output")

# create weights 
W1 = tf.Variables(shape = (1,X.getShape()[0],), is_parameter = True)
b1 = tf.Variables(shape = (1,1), is_parameter = True )

Z1 = tf.matmul(W1,X) + b1 
A1 = tf.sigmoid(Z1) 

cost = tf.logistic_loss(logits = A1 , labels = Y)
# create an optimizer 
optimizer = tf.GradientDescentOptimizer(cost, learning_rate = 1.2) 
# to run, create a session 
sess = tf.Session()

for i in range(1000):
	sess.run(optimizer)
	cost_value = np.squeeze(cost.getValue())
	if i % 100:
		print("cost at iteration (%d) = "%i, cost_value )


# make prediction and threshold at 0.5 
pred = ( sess.evaluate(A1) > 0.5 ) * 1 

print("*"*50)
print("True Outputs = ",Y.getValue())
print("predictions = ",pred  )

# To print the weights 
print("*"*50)
print("Weight W1 = ",W1.getValue())
print("bias   b1 = ",b1.getValue())

