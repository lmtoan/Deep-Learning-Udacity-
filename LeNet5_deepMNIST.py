from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf

# Extract data files into three sets: train, valid, test (dataset & labels)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
image_size = 28
num_channels = 1
num_labels = 10

graph = tf.Graph()
with graph.as_default():

	# Parameters
	# Input: [-1, 28, 28, 1]
	# Layer 2 Conv: WC1=[3, 3, 1, 6] + BC1=[6]. Stride=1. Padding=1/SAME. Output=[-1, 28, 28, 6]
	# Layer 3: Max Pooling to output=[-1, 14, 14, 6]. Patch=2x2. Stride=2. No padding 
	# Layer 4: Conv: WC3=[5, 5, 6, 16] + BC3=[16]. Stride=1. No padding. Output=[-1, 10, 10, 16]
	# Layer 5: Max Pooling to output=[-1, 5, 5, 16]. Patch=2x2. Stride=2. No padding
	# Layer 6: Fully-connected: WF5=[5*5*16, 120], BF5=[120]
	# Layer 7: Fully-connected: WF6=[120, 84], BF6=[84]
	# Layer 8: Gaussian: W=[84,10], B=[10]

	X = tf.placeholder(tf.float32, [None, image_size*image_size])
	Y_ = tf.placeholder(tf.float32, [None, num_labels])
	pkeep = tf.placeholder(tf.float32)

	all_weights = {
		'WC1': tf.Variable(tf.truncated_normal([3, 3, 1, 6], stddev=0.2)),
		'WC3': tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.2)),
		'WF5': tf.Variable(tf.truncated_normal([5*5*16, 120], stddev=0.2)),
		'WF6': tf.Variable(tf.truncated_normal([120, 84], stddev=0.2)),
		'W': tf.Variable(tf.truncated_normal([84, num_labels], stddev=0.2))
	}	

	all_biases = {
		'BC1': tf.Variable(tf.ones([6])/10),
		'BC3': tf.Variable(tf.ones([16])/10),
		'BF5': tf.Variable(tf.ones([120])/10),
		'BF6': tf.Variable(tf.ones([84])/10),
		'B': tf.Variable(tf.ones([num_labels])/10)
	}

	# LeNet5 Model
	XX = tf.reshape(X, [-1, image_size, image_size, num_channels])
	Y2 = tf.nn.relu(tf.nn.conv2d(XX, all_weights['WC1'], [1, 1, 1, 1], padding='SAME') + all_biases['BC1']) # [-1, 28, 28, 6]
	print('\nLayer 2 Shape = ', Y2.get_shape().as_list())
	Y3 = tf.nn.max_pool(Y2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME') # [-1, 14, 14, 6]
	print('Layer 3 Shape = ', Y2.get_shape().as_list())
	Y4 = tf.nn.relu(tf.nn.conv2d(Y3, all_weights['WC3'], [1, 1, 1, 1], padding='VALID') + all_biases['BC3']) #[-1, 10, 10, 16]
	print('Layer 4 Shape = ', Y4.get_shape().as_list())
	Y5 = tf.nn.max_pool(Y4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME') # [-1, 5, 5, 16]
	Y5_shape = Y5.get_shape().as_list()
	print('Layer 5 Shape = ', Y5_shape)
	Y5f = tf.reshape(Y5, [-1, Y5_shape[1]*Y5_shape[2]*Y5_shape[3]]) # Flatten to prepare for fully-connected layer
	Y6d = tf.nn.dropout(tf.nn.relu(tf.matmul(Y5f, all_weights['WF5']) + all_biases['BF5']), pkeep) # [-1, 120]
	print('Layer 6 Shape = ', Y6d.get_shape().as_list())
	Y7d = tf.nn.dropout(tf.nn.relu(tf.matmul(Y6d, all_weights['WF6']) + all_biases['BF6']), pkeep) # [-1, 84]
	print('Layer 7 Shape = ', Y7d.get_shape().as_list())

	Y_logits = tf.matmul(Y7d, all_weights['W']) + all_biases['B']
	print('Logits Shape = ', Y_logits.get_shape().as_list())
	Y_pred = tf.nn.softmax(Y_logits)

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits, labels=Y_))*100.0
	correct_prediction_vec = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction_vec, tf.float32))*100.0

	lr = tf.placeholder(tf.float32)
	train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# Execution
with tf.Session(graph=graph) as Session:
	tf.initialize_all_variables().run()
	
	max_accuracy = 0
	batch_size = 128
	n_iters = 15001

	print('Commencing LeNet-5 for %d samples and %d iterations...' % (batch_size, n_iters))

	try:
		for step in range(n_iters):
			# Set up LR Decay
			lr_max = 0.003
			lr_min = 0.0001
			learning_rate = lr_min + (lr_max - lr_min) * np.exp(-step/2000)

			# Set up mini-batch
			batch_data, batch_labels = mnist.train.next_batch(batch_size)

			# Train model (backprop)
			fd_train = {X: batch_data, Y_: batch_labels, lr: learning_rate, pkeep: 0.5}
			train_step.run(feed_dict=fd_train)

			# Report train statistics (feedforward the train set)
			if step % 100 == 0:
				fd_train_report = {X: batch_data, Y_: batch_labels, lr: learning_rate, pkeep: 1.0}
				c_train, train_accuracy = Session.run([cross_entropy, accuracy], feed_dict=fd_train_report) # Or accuracy.eval(feed_dict=fd_train_report)
				print('Training accuracy at iter = %d is %f. Loss is %f' %(step, train_accuracy, c_train))

			# Report test statistics (feedforward the test set)
			if step % 1000 == 0:
				fd_test = {X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0}
				c_test, test_accuracy = Session.run([cross_entropy, accuracy], feed_dict=fd_test)
				print('Test accuracy at iter = %d is %f. Loss is %f' %(step, test_accuracy, c_test))
				if test_accuracy > max_accuracy:
					max_accuracy = test_accuracy
					print('*** Max test accuracy so far = %f' %(max_accuracy))
	except KeyboardInterrupt:
		print('\n********** Learning stopped. Max accuracy = ', max_accuracy)
	print('\n*** Learning finished. Max test accuracy = %f' %(max_accuracy))

