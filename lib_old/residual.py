## FILE:           residual.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Functions to create neural neworks with residual links.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import tensorflow as tf
from tensorflow.python.training import moving_averages
import numpy as np
import argparse, math, sys

# beta = tf.Print(beta, [beta], message='beta', summarize=5000)		

def Residual(input, seq_len, keep_prob, training, num_outputs, args):
	'''
	Residual Network.
	'''

	## RESNET
	network = [input] # input features.
	blocks = 0; # number of blocks.
	if args.conv_caus: args.padding = 'valid'
	else: args.padding = 'same'
	dcount = 0 # count for dilation rate.

	## RESIDUAL BLOCKS
	for i in range(len(args.blocks)):	
		start_layer = len(network) - 1 # starting index of block.
		blocks += 1

		with tf.variable_scope(args.blocks[i] + '_' + str(blocks)):

			## MLP FEATURES - CAUSAILTY & DIMENSIONALITY 
			if args.blocks[i] == 'F1': network.append(tf.boolean_mask(network[-1], tf.sequence_mask(seq_len))) # convert 3D input to 2D input. 
			elif args.blocks[i] == 'F2': network.append(tf.reshape(tf.boolean_mask(tf.reshape(tf.squeeze(tf.extract_image_patches(tf.expand_dims(tf.concat([tf.zeros([tf.shape(network[-1])[0],
						args.context - 1, args.input_dim]), network[-1]], 1), 2), 
						[1, args.context, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], "VALID"), axis=[2]), 
						[tf.shape(network[-1])[0], -1, args.input_dim*args.context]), 
						tf.sequence_mask(seq_len)), [-1, args.input_dim*args.context])) # causal input (converts 3D tensor to 2D tensor). 
			elif args.blocks[i] == 'F3': network.append(tf.reshape(tf.boolean_mask(tf.squeeze(tf.extract_image_patches(tf.expand_dims(network[-1], 2), 
						[1, args.context, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME"), axis=[2]), 
						tf.sequence_mask(seq_len)), [-1, args.input_dim*args.context])) # non-causal input (converts 3D tensor to 2D tensor).
			elif args.blocks[i] == 'F4': network.append(tf.squeeze(tf.extract_image_patches(tf.expand_dims(network[-1], 2), 
					[1, args.context, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME"), axis=[2])) # non-causal input.		
				
			## LAYERS (NO RESIDUAL CONNECTION)
			elif args.blocks[i] == 'L1': network.append(tf.layers.dense(network[-1], args.cell_size, tf.nn.relu, True)) # MLP -> ReLU.
			elif args.blocks[i] == 'L2': network.append(tf.nn.relu(tf.contrib.layers.layer_norm(tf.layers.dense(network[-1], 
				args.cell_size, use_bias=False), scale=False))) # MLP -> LN -> ReLU.
			elif args.blocks[i] == 'L3': network.append(tf.nn.relu(tf.layers.batch_normalization(tf.layers.dense(network[-1], 
				args.cell_size, use_bias=False), momentum=0.99, epsilon=1e-12, training=training, scale=False))) # MLP -> BN -> ReLU.
			elif args.blocks[i] == 'L4': network.append(rnn_layer(network[-1], args.cell_size, seq_len, 
				args.cell_type, args)) # RNN.

			## COUPLING LAYER
			elif args.blocks[i] == 'C1': network.append(coupling_unit(network[-1], args.conv_size, args.conv_filt, seq_len, 
				training, args)) # 1D CNN, (conv_size, conv_filt).
			elif args.blocks[i] == 'C2': network.append(coupling_unit(network[-1], 1, args.conv_filt, seq_len, 
				training, args)) # 1D CNN, (1, conv_filt)
			elif args.blocks[i] == 'C3': network.append(coupling_unit(network[-1], 1, args.coup_conv_filt, 
				seq_len, training, args)) # 1D CNN, (1, cell_size)

			## RESIDUAL BLOCKS
			elif args.blocks[i] == 'B1': # bottleneck residual block with pre-activated units.
				B1 = block_unit(network[-1], args.block_unit, 1, args.conv_filt, seq_len, 
					1, training, args)
				B1 = block_unit(B1, args.block_unit, args.conv_size, args.conv_filt, seq_len, 
					2, training, args)
				network.append(block_unit(B1, args.block_unit, 1, args.cell_size, seq_len, 
					3, training, args))
			elif args.blocks[i] == 'B2': # residual block with pre-activated units.
				B2 = block_unit(network[-1], args.block_unit, args.conv_size, args.conv_filt, seq_len, 
					1, training, args)
				network.append(block_unit(B2, args.block_unit, args.conv_size, args.conv_filt, seq_len, 
					2, training, args))
			elif args.blocks[i] == 'B3': network.append(rnn_layer(network[-1], args.cell_size, seq_len, 
						args.cell_type, args)) # RNN.	 	
			

			elif args.blocks[i] == 'B4': # temporal conv block with pre-activated units.
				args.dilation_rate = 2**(dcount) # exponentially increasing dilation rate.
				B4 = block_unit(network[-1], 'BU4', args.conv_size, args.conv_filt, seq_len, 1, training, args)
				network.append(block_unit(B4, 'BU4', args.conv_size, args.conv_filt, seq_len, 2, training, args))
				dcount += 1
				if 2**(dcount) > args.max_dilation_rate: dcount = 0

			elif args.blocks[i] == 'B5': # bottleneck temporal conv block with pre-activated units.
				args.dilation_rate = 2**(dcount) # exponentially increasing dilation rate.
				with tf.variable_scope('D' + str(args.dilation_rate)):
					B5 = block_unit(network[-1], 'BU1', 1, args.conv_filt, seq_len, 
						1, training, args)
					B5 = block_unit(B5, 'BU4', args.conv_size, args.conv_filt, seq_len, 
					 	2, training, args)
					network.append(block_unit(B5, 'BU1', 1, args.cell_size, seq_len, 
						3, training, args))
				dcount += 1
				if 2**(dcount) > args.max_dilation_rate: dcount = 0
			# elif args.blocks[i] == 'B6': # bottleneck temporal conv block with pre-activated units.
			# 	args.dilation_rate = 2**(dcount) # exponentially increasing dilation rate.
			# 	with tf.variable_scope('D' + str(args.dilation_rate)):
			# 		B5 = block_unit(network[-1], 'BU1', 1, args.conv_filt, seq_len, 
			# 			1, training, args)
			# 		B5 = block_unit(B5, 'BU4', args.conv_size, args.conv_filt, seq_len, 
			# 		 	2, training, args)
			# 		network.append(block_unit(B5, 'BU1', 1, args.cell_size, seq_len, 
			# 			3, training, args))
			# 	dcount += 1
			# 	if 2**(dcount) > args.max_dilation_rate: dcount = 0


			## OUTPUT LAYER
			elif args.blocks[i] == 'O1': network.append(tf.layers.dense(tf.boolean_mask(network[-1], tf.sequence_mask(seq_len)), num_outputs)) # Output layer, converts 3D tensor to 2D tensor.
			elif args.blocks[i] == 'O2': network.append(tf.layers.dense(network[-1], num_outputs)) # Output layer for 2D tensor.

			else: # block type does not exist.
				raise ValueError('Block type does not exist: %s.' % (args.blocks[i]))

		## RESIDUAL CONNECTION 
		if args.blocks[i][:1] == 'B':
			with tf.variable_scope(args.res_con + '_L' + str(len(network)-1) + '_L' + str(start_layer)):
				if args.res_con == 'add':
					if network[-1].get_shape().as_list()[-1] == network[start_layer].get_shape().as_list()[-1]:
						network.append(tf.add(network[-1], network[start_layer])) # residual connection.
				elif args.res_con == 'concat': network.append(tf.concat([network[-1], network[start_layer]], 2)) # residual connection.
				elif args.res_con == 'proj_concat': network.append(tf.concat([network[-1], 
					tf.multiply(tf.layers.dense(network[start_layer], args.res_proj, use_bias = False), 
					tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32))], 2)) # projected residual connection.
				elif args.res_con == 'concat_proj': network.append(tf.multiply(tf.layers.dense(tf.concat([network[-1], 
					network[start_layer]], 2), args.res_proj, use_bias = False), 
					tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32))) # projection.
				else: # residual connection type does not exist.
					raise ValueError('Residual connection type does not exist: %s.' % (args.res_con))

	## SUMMARY	
	if args.verbose:
		for I, i in enumerate(network):
			print(i.get_shape().as_list(), end="")
			print("%i:" % (I), end="")
			print(str(i.name))

	return network[-1]

## COUPLING UNIT 
def coupling_unit(input, conv_size, conv_filt, seq_len, training, args):
	with tf.variable_scope(args.coup_unit):
		if args.coup_unit == 'CU1': C = tf.nn.relu(masked_layer_norm(conv_layer(input, 
			conv_size, conv_filt, seq_len, args, use_bias=False), seq_len, scale=False)) # (W -> LN -> ReLU).
		elif args.coup_unit == 'CU2': C = tf.nn.relu(masked_batch_norm(conv_layer(input, 
			conv_size, conv_filt, seq_len, args, use_bias=False), seq_len, training, scale=False)) # (W -> BN -> ReLU).
		elif args.coup_unit == 'CU3': C = tf.nn.relu(conv_layer(input, conv_size, conv_filt, 
			seq_len, args, use_bias=True)) # (W -> ReLU).
		elif args.coup_unit == 'CU4': C = conv_layer(input, conv_size, conv_filt, seq_len, 
			args, use_bias=True) # (W).
		else: raise ValueError('Coupling unit does not exist: %s.' % (args.coup_unit)) # residual unit does not exist.
		return C

## RESIDUAL BLOCK UNIT 
def block_unit(input, block_unit, conv_size, conv_filt, seq_len, unit_id, training, args):
	with tf.variable_scope(block_unit + '_' + str(unit_id)):
		if block_unit == 'BU1': U = conv_layer(tf.nn.relu(masked_layer_norm(input, 
			seq_len)), conv_size, conv_filt, seq_len, args) # (LN -> ReLU -> 1D conv).
		elif block_unit == 'BU2': U = conv_layer(tf.nn.relu(masked_batch_norm(input, 
			seq_len, training)), conv_size, conv_filt, seq_len, args) # (BN -> ReLU -> 1D conv).
		elif block_unit == 'BU3': U = tf.nn.relu(masked_layer_norm(conv_layer(input, 
			conv_size, conv_filt, seq_len, args), seq_len)) # (1D dilated conv -> LN -> ReLU). (no dropout, and WeightNorm replaced with LayerNorm).
		elif block_unit == 'BU4': U = conv_layer(tf.nn.relu(masked_layer_norm(input, 
			seq_len)), conv_size, conv_filt, seq_len, args, dilation_rate=args.dilation_rate) # (LN -> ReLU -> 1D dilated conv).
		else: # residual unit does not exist.
			raise ValueError('Residual unit does not exist: %s.' % (block_unit))
		return U

## CNN LAYER
def conv_layer(input, conv_size, conv_filt, seq_len, args, dilation_rate=1, 
	use_bias=True, bias_init=tf.constant_initializer(0.0)):
	if args.conv_caus:
		input = tf.concat([tf.zeros([tf.shape(input)[0], (conv_size - 1)*dilation_rate, 
			tf.shape(input)[2]]), input], 1)
	conv = tf.layers.conv1d(input, conv_filt, conv_size, dilation_rate=dilation_rate, 
		activation=None, padding=args.padding, use_bias=use_bias, bias_initializer=bias_init) # 1D CNN: (conv_size, conv_filt).
	if not args.conv_caus:
		conv = tf.multiply(conv, tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 
			2), tf.float32), name= 'k' + str(conv_size) + '_f' + str(conv_filt) + '_d' + str(dilation_rate))
	return conv

## CNN L

##
# def conv_layer(input, conv_size, conv_filt, seq_len, args, dilation_rate=1, 
# 	use_bias=True, bias_init=tf.constant_initializer(0.0)):
# 	if args.conv_caus:
# 		input = tf.concat([tf.zeros([tf.shape(input)[0], (conv_size - 1)*dilation_rate, 
# 			tf.shape(input)[2]]), input], 1)
# 	conv = tf.layers.conv1d(input, conv_filt, conv_size, dilation_rate=dilation_rate, 
# 		activation=None, padding=args.padding, use_bias=use_bias, bias_initializer=bias_init) # 1D CNN: (conv_size, conv_filt).
# 	conv = tf.multiply(conv, tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 
# 		2), tf.float32), name='d_' + str(dilation_rate))
# 	return conv


## RNN LAYER
def rnn_layer(input, cell_size, seq_len, scope, args):
	with tf.variable_scope(scope):
		if args.cell_type == 'IndRNNCell':
			cell_fw = tf.contrib.rnn.IndRNNCell(cell_size, activation=tf.nn.relu) # forward IndRNNCell.
			if args.bidi:
				cell_bw = tf.contrib.rnn.IndRNNCell(cell_size, activation=tf.nn.relu) # backward IndRNNCell.
		elif args.cell_type == 'IndyLSTMCell':
			cell_fw = tf.contrib.rnn.IndyLSTMCell(cell_size) # forward IndyLSTMCell.
			if args.bidi:
				cell_bw = tf.contrib.rnn.IndyLSTMCell(cell_size) # backward IndyLSTMCell.
		elif args.cell_type == 'LSTMCell':
			cell_fw = tf.contrib.rnn.LSTMCell(cell_size, args.peep, num_proj=args.cell_proj) # forward LSTMCell.
			if args.bidi:
				cell_bw = tf.contrib.rnn.LSTMCell(cell_size, args.peep, num_proj=args.cell_proj) # backward LSTMCell.
		if args.bidi:
			output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input, seq_len, 
				swap_memory=True, parallel_iterations=args.par_iter, dtype=tf.float32) # bidirectional recurrent neural network.
		else:
			output, _ = tf.nn.dynamic_rnn(cell_fw, input, seq_len, swap_memory=True,
				parallel_iterations=args.par_iter, dtype=tf.float32) # recurrent neural network.
		if args.bidi:
			if args.bidi_con == 'concat':
				output = tf.concat(output, 2)
			elif args.bidi_con == 'add':
				output = tf.add(output[0], output[1])
			elif args.bidi_con == 'ensemble':
				output = tf.div(tf.add(output[0], output[1]), 2.0)
			elif args.bidi_con == 'concat_caus_flag':
				act_fw = output[0];
				f1 = lambda: output[1]
				f2 = lambda: tf.zeros_like(output[1])
				act_bw = tf.case([(args.caus_flag, f2)], default=f1)
				output = tf.concat([act_fw, act_bw], 2)
			elif args.bidi_con == 'ensemble_caus_flag':
				act_fw = output[0];
				f1 = lambda: output[1]
				f2 = lambda: tf.zeros_like(output[1])
				act_bw = tf.case([(args.caus_flag, f2)], default=f1)
				act_sum = tf.add(act_fw, act_bw)
				f1 = lambda: tf.div(act_sum, 2)
				f2 = lambda: act_sum
				output = tf.case([(args.caus_flag, f2)], default=f1)
			else:
				raise ValueError('Incorrect args.bidi_connect specification.')
		return output

## LOSS FUNCTIONS
def loss(target, estimate, loss_fnc):
	'loss functions for gradient descent.'
	with tf.name_scope(loss_fnc + '_loss'):
		if loss_fnc == 'quadratic':
			loss = tf.reduce_sum(tf.square(tf.subtract(target, estimate)), axis=1)
		if loss_fnc == 'sigmoid_cross_entropy':
			loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=estimate), axis=1)
		if loss_fnc == 'cross_entropy':
			loss = tf.negative(tf.reduce_sum(tf.multiply(target, tf.log(estimate))))
		# if loss_fnc == 'mse':
		# 	loss = tf.losses.mean_squared_error(labels=target, predictions=estimate)
		# if loss_fnc == 'softmax_xentropy':
		# 	loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=estimate)
		# if loss_fnc == 'sigmoid_xentropy':
		# 	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=estimate)
		return loss

## GRADIENT DESCENT OPTIMISERS
def optimizer(loss, lr=None, epsilon=None, var_list=None, optimizer='adam', grad_clip=False):
	'optimizers for training.'
	with tf.name_scope(optimizer + '_opt'):
		if optimizer == 'adam':
			if lr == None: lr = 0.001 # default.
			if epsilon == None: epsilon = 1e-8 # default.
			optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
		if optimizer == 'lazyadam':
			if lr == None: lr = 0.001 # default.
			if epsilon == None: epsilon = 1e-8 # default.
			optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=lr, epsilon=epsilon)
		if optimizer == 'nadam':
			if lr == None: lr = 0.001 # default.
			if epsilon == None: epsilon = 1e-8 # default.
			optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, epsilon=epsilon)
		if optimizer == 'sgd':
			if lr == None: lr = 0.5 # default.
			optimizer = tf.train.GradientDescentOptimizer(lr)
		grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
		

		# with open('tmp.txt', 'w') as f:
		# 	for item in grads_and_vars:
		# 		f.write("%s\n" % str(item))

		# sys.exit()

		if grad_clip: grads_and_vars = [(tf.clip_by_value(gv[0], -1., 1.), gv[1]) for gv in grads_and_vars]
		trainer = optimizer.apply_gradients(grads_and_vars)
	return trainer, optimizer


## OPTIMISER FUNCTION
def optimizer_legacy(loss, lr=None, epsilon=None, var_list=None, optimizer='adam'):
	'WILL BE REMOVED IN FUTURE. optimizers for training.'
	with tf.name_scope(optimizer + '_opt'):
		if optimizer == 'adam':
			if lr == None: lr = 0.001 # default.
			if epsilon == None: epsilon = 1e-8 # default.
			optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
			trainer = optimizer.minimize(loss, var_list=var_list) 
		if optimizer == 'nadam':
			if lr == None: lr = 0.001 # default.
			if epsilon == None: epsilon = 1e-8 # default.
			optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, epsilon=epsilon)
			trainer = optimizer.minimize(loss, var_list=var_list) 
		if optimizer == 'sgd':
			if lr == None: lr = 0.5 # default.
			optimizer = tf.train.GradientDescentOptimizer(lr)
			trainer = optimizer.minimize(loss, var_list=var_list) 
	return trainer, optimizer

## MASKED LAYER NORM
def masked_layer_norm(input, seq_len, centre=True, scale=True): # layer norm for 3D tensor.
	with tf.variable_scope('Layer_Norm'):
		mask = tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32) # convert mask to float.
		input_dim = input.get_shape().as_list()[-1] # get number of input dimensions.
		den = tf.multiply(tf.reduce_sum(mask, axis=1, keepdims=True), input_dim) # inverse of the number of input dimensions.
		mean = tf.divide(tf.reduce_sum(tf.multiply(input, mask), axis=[1, 2], keepdims=True), den) # mean over the input dimensions.
		var = tf.divide(tf.reduce_sum(tf.multiply(tf.square(tf.subtract(input, mean)), mask), axis=[1, 2], 
			keepdims = True), den) # variance over the input dimensions.
		if centre:
			beta = tf.get_variable("beta", input_dim, dtype=tf.float32,  
				initializer=tf.constant_initializer(0.0), trainable=True)
		else: beta = tf.constant(np.zeros(input_dim), name="beta", dtype=tf.float32)
		if scale:
			gamma = tf.get_variable("Gamma", input_dim, dtype=tf.float32,  
				initializer=tf.constant_initializer(1.0), trainable=True)
		else: gamma = tf.constant(np.ones(input_dim), name="Gamma", dtype=tf.float32)
		norm = tf.nn.batch_normalization(input, mean, var, offset=beta, scale=gamma, 
			variance_epsilon = 1e-12) # normalise batch.
		norm = tf.multiply(norm, mask)
		return norm

## MINI-BATCH STATISTICS
def batch_stats(input): # for 2D mini-batch.
	num = tf.reduce_sum(input, 0)
	den = tf.cast(tf.shape(input)[0], tf.float32)
	mean = tf.divide(num, den)
	num = tf.reduce_sum(tf.square(tf.subtract(input, mean)), 0)
	var = tf.divide(num,den)
	return mean, var

## BATCH NORM
def batch_norm(input, training=False, decay=0.99): # batch norm for 2D mini-batch.
	with tf.variable_scope('Batch_Norm'):
		input_dim = input.get_shape().as_list()[-1:]
		moving_mean = tf.get_variable("moving_mean", input_dim, dtype=tf.float32,  
			initializer=tf.zeros_initializer, trainable=False)
		moving_var = tf.get_variable("moving_var", input_dim, dtype=tf.float32,  
			initializer=tf.constant_initializer(1), trainable=False)
		batch_mean, batch_var = batch_stats(input)
		def update_moving_stats():
			update_moving_mean = moving_averages.assign_moving_average(moving_mean, 
				batch_mean, decay, zero_debias=True)
			update_moving_variance = moving_averages.assign_moving_average(moving_var, 
				batch_var, decay, zero_debias=False)
			with tf.control_dependencies([update_moving_mean, update_moving_variance]):
				return tf.identity(batch_mean), tf.identity(batch_var)
		mean, var = tf.cond(training, true_fn = update_moving_stats,
			false_fn = lambda: (moving_mean, moving_var))
		variance_epsilon = 1e-12       
		beta = tf.get_variable("beta", input_dim, dtype=tf.float32,  
			initializer=tf.constant_initializer(0), trainable=True)
		gamma = tf.get_variable("gamma", input_dim, dtype=tf.float32,  
			initializer=tf.constant_initializer(1), trainable=True)
		norm = tf.nn.batch_normalization(input, mean, var, beta, gamma, 
			variance_epsilon)
		return norm

## MASKED MINI-BATCH STATISTICS
def masked_batch_stats(input, mask, axes, scaler):
	num = tf.reduce_sum(tf.multiply(input, mask), axis=axes)
	den = tf.multiply(tf.reduce_sum(mask, axis=axes), scaler)
	mean = tf.divide(num, den)
	num = tf.reduce_sum(tf.multiply(tf.square(tf.subtract(input, mean)), 
		mask), axis=axes)
	var = tf.divide(num, den)
	return mean, var

## MASKED BATCH NORM
def masked_batch_norm(inp, seq_len, training=False, decay=0.99, centre=True, scale=True):
	with tf.variable_scope('Batch_Norm'):
		mask = tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32)
		input_dim = inp.get_shape().as_list()[-1:]
		moving_mean = tf.get_variable("moving_mean", input_dim, dtype=tf.float32,  
			initializer=tf.zeros_initializer, trainable=False)
		moving_var = tf.get_variable("moving_var", input_dim, dtype=tf.float32,  
			initializer=tf.constant_initializer(1), trainable=False)
		batch_mean, batch_var = masked_batch_stats(inp, mask, [0, 1], 1)
		def update_moving_stats():
			update_moving_mean = moving_averages.assign_moving_average(moving_mean, 
				batch_mean, decay, zero_debias=True)
			update_moving_variance = moving_averages.assign_moving_average(moving_var, 
				batch_var, decay, zero_debias=False)
			with tf.control_dependencies([update_moving_mean, update_moving_variance]):
				return tf.identity(batch_mean), tf.identity(batch_var)
		mean, var = tf.cond(training, true_fn = update_moving_stats,
			false_fn = lambda: (moving_mean, moving_var))
		variance_epsilon = 1e-12       

		if centre:
			beta = tf.get_variable("beta", input_dim, dtype=tf.float32,  
				initializer=tf.constant_initializer(0.0), trainable=True)
		else: beta = tf.constant(np.zeros(input_dim), name="beta", dtype=tf.float32)
		if scale:
			gamma = tf.get_variable("Gamma", input_dim, dtype=tf.float32,  
				initializer=tf.constant_initializer(1.0), trainable=True)
		else: gamma = tf.constant(np.ones(input_dim), name="Gamma", dtype=tf.float32)
		norm = tf.nn.batch_normalization(inp, mean, var, beta, gamma, 
			variance_epsilon)
		norm = tf.multiply(norm, mask)
		return norm

def reslink(x, y):
	if x.get_shape().as_list()[-1] == y.get_shape().as_list()[-1]: return tf.add(x, y, name='add')
	elif x.get_shape().as_list()[-1] > y.get_shape().as_list()[-1]: return tf.add(tf.layers.dense(x, y.get_shape().as_list()[-1], 
		use_bias = False), y, name='add_' + str(x.get_shape().as_list()[-1]) + '_' + str(y.get_shape().as_list()[-1]))
	else: return tf.add(x, tf.layers.dense(y, x.get_shape().as_list()[-1], use_bias = False), 
		name='add_' + str(y.get_shape().as_list()[-1]) + '_' + str(x.get_shape().as_list()[-1]))
