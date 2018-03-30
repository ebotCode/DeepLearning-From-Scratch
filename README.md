"# DeepLearning-From-Scratch" 


This Repository contains an implementation of some deeplearning 
"lego" blocks from scratch. 

During my progress in the course "Neural Network" in coursera's "deeplearning.ai", i implemented backpropagation and forwardprop from scratch 
in python for some operations and network architecture. 

Armed with the understanding, and inspired by tensorflow's "lego" like feel, i decided to build a (computation graph based) implementation of those same building blocks. 

This implementation can handle logistic_loss, dropout, ...(see ttechoflow.py)

For instance, to set up a simple linear logistic regression problem, 

import ttechoflow as tf 
x_train = 
y_train = 

# create a graph. This defines the namespace for the variables to be created 
my_graph = tf.ComputationGraph()
my_graph.initizlize()
my_graph.set_as_default()

X = tf.Constant(x_train, name = "Input")
Y = tf.Constant(y_train, name = "Output")

W1 = tf.Variable(shape = (5,X.getShape()[0],), is_parameter = True)
b1 = tf.Variable(shape = (5,1), is_parameter = True )

Z1 = tf.matmul(W1,Z) + b1 
A1 = tf.sigmoid(Z1) 

cost = tf.logistic_loss(logits = A1, labels = Y)

# create an optimizer 
optimizer = tf.GradientOptimizer(cost, learning_rate = 0.1) 
# to run, create a session 

Lol. I tried to give it a similar feel as what tensorflow so beautifully gives. 

Though this implementation is not as rhobust as Tensorflow (lol, how can it be, it was just an extension of what was done in the class...)

Doing this helped solidify what i learnt, and try out how frameworks give that "lego" like ability to its users.

