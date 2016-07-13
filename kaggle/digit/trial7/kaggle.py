import numpy as np
import pandas as pd

print("Reading data...")
dataset = pd.read_csv("../data/train.csv.old")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv("../data/test.csv.old").values

#rf = RandomForestClassifier(n_estimators=100)
#rf.fit(train,target)
#pred = rf.predict(test)
#np.savetxt('submission_rand_forest.cvs', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImagedId,Label', comments= '', fmt='%d')
#accuracy ~ 0.96

target = target.astype(np.uint8)
train = np.array(train).reshape((-1,1,28,28)).astype(np.unit8)
test = np.array(test).reshape((-1,1,28,28)).astype(np.uint8)


import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

net1 = NeuralNet(
		layers=[('input', layers.InputLayer),
		('hidden', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
		input_shape=(None,1,28,28),
		hidden_num_units=1000,
		output_nonlinearity=lasagne.nonlinearities.softmax,
		output_num_units=10,

		update=nesterov_momentum,
		update_learning_rate=0.0001,
		update_momentum=0.9,

		max_epochs=15,
		verbose=1,
		)
#net1.fit(train,target)
def CNN(n_epochs):
	net1 = NeuralNet(
		  	layers=[
			  ('input', layers.InputLayer),
			  ('conv1', layers.Conv2DLayer),
			  ('pool1', layers.MaxPool2DLayer),
		  	('conv2', layers.Conv2DLayer),
		  	('hidden3', layers.DenseLayer),
		   	('output', layers.DenseLayer),
			  ],

		  	input_shape=(None, 1, 28, 28),
				conv1_num_filters=7, #20
				conv1_filter_size=(3,3), # 5,5
				conv1_nonlinearity=lasagne.nonlinearities.rectify,

				pool1_pool_size=(2,2),

				conv2_numb_filters=12, #100
				conv2_filter_size=(2,2),
				conv2_nonlinearity=lasagne.nonlinearities.rectify,

				hidden3_num_units=1000,
				output_num_units=10,
				output_nonlinearity= lasagne.nonlinearities.softmax,

				update_learning_rate=0.00001,
				update_momentum = 0.9,

				max_epochs=n_epochs,
				verbose=1,
				)
	return net1
cnn = CNN(15).fit(train,target)


pred = cnn.predict(test)
np.savetxt('submission_cnn.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments='', fmt='%d')
