## FILE:           normalisation.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Layer/instance/batch normalisation functions.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from os.path import expanduser
import argparse, os, string
import numpy as np
import tensorflow as tf

def Normalisation(x, norm_type, mask, training):
	'''
	Normalisation.

	Input/s:
		x - unnormalised input.
		norm_type - normalisation type.				
		mask - sequence mask.
		training - training flag.

	Output/s:
		normalised input.
	'''

	if norm_type == 'instance': return tf.contrib.layers.instance_norm(x)
	elif norm_type == 'layer': return MaskedLayerNorm(x, mask)
	elif norm_type == 'batch': return MaskedBatchNorm(x, mask, training)
	elif norm_type == 'unnormalised': return x
	else: ValueError('Normalisation type does not exist: %s.' % (norm_type))

## MASKED LAYER NORMALISATION
count = 0 
def MaskedLayerNorm(x, mask, centre=True, scale=True): # layer norm for 3D tensor.
	'''
	Layer Normalisation with sequence masking.

	Input/s:
		x - unnormalised input.
		mask - sequence mask. 
		axes - axes to calculate statistics over. 
		centre - centre parameter.
		scale - scale parameter. 

	Output/s:
		normalised input.
	'''
	global count
	count += 1

	with tf.variable_scope('LayerNorm' + str(count)):
		# mask = tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32) # convert mask to float.
		map_size = x.get_shape().as_list()[-1] # get number of input dimensions.
		den = tf.multiply(tf.reduce_sum(mask, axis=1, keepdims=True), map_size) # inverse of the number of input dimensions.
		mean = tf.divide(tf.reduce_sum(tf.multiply(x, mask), axis=[1, 2], keepdims=True), den) # mean over the input dimensions.
		var = tf.divide(tf.reduce_sum(tf.multiply(tf.square(tf.subtract(x, mean)), mask), axis=[1, 2], 
			keepdims = True), den) # variance over the input dimensions.
		if centre:
			beta = tf.get_variable("beta", map_size, dtype=tf.float32,  
				initializer=tf.constant_initializer(0.0), trainable=True)
		else: beta = tf.constant(np.zeros(map_size), name="beta", dtype=tf.float32)
		if scale:
			gamma = tf.get_variable("Gamma", map_size, dtype=tf.float32,  
				initializer=tf.constant_initializer(1.0), trainable=True)
		else: gamma = tf.constant(np.ones(map_size), name="Gamma", dtype=tf.float32)
		return tf.nn.batch_normalization(x, mean, var, offset=beta, scale=gamma, 
			variance_epsilon = 1e-12) # normalise batch.
		# norm = tf.multiply(norm, mask)
		# return norm

## MASKED MINI-BATCH STATISTICS
def MaskedMBatchStats(x, mask, axes):
	'''
	Computes mean and variance over axes of mini-batch.

	Input/s:
		x - input.
		mask - sequence mask. 
		axes - axes to calculate statistics over. 

	Output/s:
		mean and variance.
	'''

	numerator = tf.reduce_sum(tf.multiply(x, mask), axis=axes)
	denominator = tf.reduce_sum(mask, axis=axes)
	mean = tf.divide(numerator, denominator)
	numerator = tf.reduce_sum(tf.multiply(tf.square(tf.subtract(x, mean)), mask), axis=axes)
	variance = tf.divide(numerator, denominator)
	return mean, variance

## MASKED BATCH NORM
def MaskedBatchNorm(x, mask, training=False, decay=0.99, centre=True, scale=True):
	'''
	Batch Normalisation with sequence masking.

	Input/s:
		x - unnormalised input.
		mask - sequence mask. 
		training - training flag.
		decay - moving average decay.
		centre - centre parameter.
		scale - scale parameter. 

	Output/s:
		normlaised input.
	'''
	global count
	count += 1

	with tf.variable_scope('BatchNorm' + str(count)):
			# mask = tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32)
		map_size = x.get_shape().as_list()[-1:]
		moving_mean = tf.get_variable("moving_mean", map_size, dtype=tf.float32,  
			initializer=tf.zeros_initializer, trainable=False)
		moving_var = tf.get_variable("moving_var", map_size, dtype=tf.float32,  
			initializer=tf.constant_initializer(1), trainable=False)			

		# if count == 1:
		# 	moving_mean = tf.Print(moving_mean, [moving_mean[0:10]], message="mmu" + str(count))


		batch_mean, batch_variance = MaskedMBatchStats(x, mask, [0, 1])
		def update_moving_stats():
			update_moving_mean = moving_averages.assign_moving_average(moving_mean, 
				batch_mean, decay, zero_debias=True)
			update_moving_variance = moving_averages.assign_moving_average(moving_var, 
				batch_variance, decay, zero_debias=False)
			with tf.control_dependencies([update_moving_mean, update_moving_variance]):
				return tf.identity(batch_mean), tf.identity(batch_variance)
		mean, var = tf.cond(training, true_fn = update_moving_stats,
			false_fn = lambda: (moving_mean, moving_var))

		if centre:
			beta = tf.get_variable("beta", map_size, dtype=tf.float32,  
				initializer=tf.constant_initializer(0.0), trainable=True)
		else: beta = tf.constant(np.zeros(map_size), name="beta", dtype=tf.float32)
		if scale:
			gamma = tf.get_variable("Gamma", map_size, dtype=tf.float32,  
				initializer=tf.constant_initializer(1.0), trainable=True)
		else: gamma = tf.constant(np.ones(map_size), name="Gamma", dtype=tf.float32)
		return tf.nn.batch_normalization(x, mean, var, beta, gamma, 
			variance_epsilon=1e-12)
			# norm = tf.multiply(norm, mask)
			# return norm