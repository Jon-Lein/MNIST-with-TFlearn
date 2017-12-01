import tflearn
import tflearn.datasets.mnist as mnist

# Loading data and labels for training and testing and reshaping data
x, y, testx, testy = mnist.load_data(one_hot=True)
x = x.reshape([-1, 28, 28 ,1])

# Input Layer
net = tflearn.input_data(shape=[28, 28, 1])

# Deep Layers
net = tflearn.fully_connected(net, 500, activation='relu')
net = tflearn.fully_connected(net, 500, activation='relu')
net = tflearn.fully_connected(net, 500, activation='relu')

# output layer ( N_Nodes = N_Classes)
net = tflearn.fully_connected(net, 10, activation='softmax')

# define regression for updating weights and stuff
net = tflearn.regression(net, n_classes=10, batch_size=100)

# Define and fit model
model = tflearn.DNN(net)
model.fit(x, y, n_epoch=2, show_metric=True)