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
	elif norm_type == 'SeqLayerNorm': return SeqLayerNorm(x, seq_len, centre=centre, scale=scale)
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


class SequenceLayerNorm(tf.keras.layers.Layer):
	"""
	"""
	def __init__(self, input_dim, output_dim, mask_zero=False, **kwargs):
		"""
		"""
		super(SequenceLayerNorm, self).__init__(**kwargs)
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.mask_zero = mask_zero
	  
	def build(self, input_shape):
		"""
		"""
		self.embeddings = self.add_weight(
			shape=(self.input_dim, self.output_dim),
			initializer='random_normal',
			dtype='float32')
	  
	def call(self, inputs):
		"""
		"""
		return tf.nn.embedding_lookup(self.embeddings, inputs)
	
	def compute_mask(self, inputs, mask=None):
		"""
		"""
		if not self.mask_zero:
	    	return None
		return tf.not_equal(inputs, 0)
  

def SeqLayerNorm(input, seq_len, centre=True, scale=True): # layer norm for 3D tensor.
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