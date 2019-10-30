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

def Normalisation(x, norm_type='FrameLayerNorm', seq_len=None, mask=None, training=False, centre=True, scale=True):
	'''
	Normalisation.

	Input/s:
		x - unnormalised input.
		norm_type - normalisation type.	
		seq_len - length of each sequence.			
		mask - sequence mask.
		training - training flag.

	Output/s:
		normalised input.
	'''
	
	if norm_type == 'SeqCausalLayerNorm': return SeqCausalLayerNorm(x, seq_len, centre=centre, scale=scale)
	elif norm_type == 'FrameLayerNorm': return FrameLayerNorm(x, centre=centre, scale=scale)
	elif norm_type == 'unnormalised': return x
	else: ValueError('Normalisation type does not exist: %s.' % (norm_type))

count = 0 
def SeqCausalLayerNorm(x, seq_len, centre=True, scale=True):
	'''
	Sequence-wise causal layer normalisation with sequence masking (causal layer norm version of https://arxiv.org/pdf/1510.01378.pdf). 

	Input/s:
		x - input.
		seq_len - length of each sequence. 
		centre - centre parameter.
		scale - scale parameter. 

	Output/s:
		normalised input.
	'''
	global count
	count += 1
	with tf.variable_scope('LayerNorm' + str(count)):
		input_size = x.get_shape().as_list()[-1]
		mask = tf.cast(tf.sequence_mask(seq_len), tf.float32) # convert mask to float.
		den = tf.multiply(tf.range(1.0, tf.add(tf.cast(tf.shape(mask)[-1], tf.float32), 1.0), dtype=tf.float32), input_size)
		mu = tf.expand_dims(tf.truediv(tf.cumsum(tf.reduce_sum(x, -1), -1), den), 2)
		sigma = tf.expand_dims(tf.truediv(tf.cumsum(tf.reduce_sum(tf.square(tf.subtract(x, 
			mu)), -1), -1), den),2)
		if centre: beta = tf.get_variable("beta", input_size, dtype=tf.float32,  
			initializer=tf.constant_initializer(0.0), trainable=True)
		else: beta = tf.constant(np.zeros(input_size), name="beta", dtype=tf.float32)
		if scale: gamma = tf.get_variable("Gamma", input_size, dtype=tf.float32,  
			initializer=tf.constant_initializer(1.0), trainable=True)
		else: gamma = tf.constant(np.ones(input_size), name="Gamma", dtype=tf.float32)
		return tf.multiply(tf.nn.batch_normalization(x, mu, sigma, offset=beta, scale=gamma, 
			variance_epsilon = 1e-12), tf.expand_dims(mask, 2))

count = 0 
def FrameLayerNorm(x, centre=True, scale=True):
	'''
	Frame-wise layer normalisation (layer norm version of https://arxiv.org/pdf/1510.01378.pdf).

	Input/s:
		x - input.
		seq_len - length of each sequence. 
		centre - centre parameter.
		scale - scale parameter. 

	Output/s:
		normalised input.
	'''
	global count
	count += 1

	with tf.variable_scope('frm_wise_layer_norm' + str(count)):
		mu, sigma = tf.nn.moments(x, -1, keepdims=True)
		input_size = x.get_shape().as_list()[-1] # get number of input dimensions.
		if centre:
			beta = tf.get_variable("beta", input_size, dtype=tf.float32,  
				initializer=tf.constant_initializer(0.0), trainable=True)
		else: beta = tf.constant(np.zeros(input_size), name="beta", dtype=tf.float32)
		if scale:
			gamma = tf.get_variable("Gamma", input_size, dtype=tf.float32,  
				initializer=tf.constant_initializer(1.0), trainable=True)
		else: gamma = tf.constant(np.ones(input_size), name="Gamma", dtype=tf.float32)
		return tf.nn.batch_normalization(x, mu, sigma, offset=beta, scale=gamma, 
			variance_epsilon = 1e-12) # normalise batch.
