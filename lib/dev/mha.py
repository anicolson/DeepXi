## FILE:           mha.py
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
## BRIEF:          Multi-head attention modules.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import math
import numpy as np
import tensorflow as tf
from utils import Normalisation

def sequence_mask(shape, seq_len):
	'''
	Sequence mask applied before softmax. Mask detirmined by 
	shape and sequence length.

	Input/s
		shape - shape of unnormalised sequence weights (3D tensor).
		seq_len - sequence length (1D tensor).

	Output/s
		Sequence mask (3D tensor).

	'''
	seq_mask = tf.sequence_mask(tf.expand_dims(seq_len, -1), maxlen=shape[-1], dtype=tf.int32)
	logical = tf.cast(tf.multiply(tf.linalg.band_part(tf.ones(shape, dtype=tf.int32), -1, 0), seq_mask), tf.bool)
	return tf.where(logical, tf.zeros(shape), tf.fill(shape, float('-inf')))

def scaled_dot_product_attention(Q, K, V, mask, d_k, args):
	'''
	Scaled dot-product attention mechanism with masking.

	Input/s
		Q - queries (3D tensor).
		K - keys (3D tensor).
		V - values (3D tensor).
		mask - sequence mask.
		args - other arguments.

	Output/s
		Weighted average the values (3D tensor).

	'''
	# print(K.get_shape().as_list())
	# sys.exit()


	num = tf.matmul(Q, K, transpose_b=True) # numerator.


	# num = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) # numerator.
	den = tf.constant(np.sqrt(d_k), dtype=tf.float32) # denominator.
	quot = tf.truediv(num, den) # quotient.
	weight = tf.nn.softmax(tf.add(quot, mask)) # weight provided by a catagorial distribution.
	
	# print(mask.get_shape().as_list())

	# print(weight.get_shape().as_list())

	tmp = tf.matmul(weight, V)
	# print(tmp.get_shape().as_list())

	return tmp

def multi_head_attention(Q, K, V, mask, args):
	'''
	Multi-head attention. Note: d_k=d_v.

	Input/s
		Q - queries (3D tensor).
		K - keys (3D tensor).
		V - values (3D tensor).
		mask - sequence mask.
		args - other arguments.

	Output/s
		Multi-head attention.

	'''
	d_k = d_v = int(args.d_model/args.h)
	# mbatch_size = tf.shape(Q)[0] # mini-batch size.
	# seq_len = tf.shape(Q)[1] # sequence length.
	heads = [];
	for i in range(args.h):
		Q_linear = tf.layers.dense(Q, d_k, activation=None, use_bias=False, name='Q_linear_' + str(i))
		K_linear = tf.layers.dense(K, d_k, activation=None, use_bias=False, name='K_linear_' + str(i))
		V_linear = tf.layers.dense(V, d_v, activation=None, use_bias=False, name='V_linear_' + str(i))
		heads.append(scaled_dot_product_attention(Q_linear, K_linear, V_linear, mask, d_k, args))
	concat_attention = tf.concat(heads, axis=2)
	return tf.layers.dense(concat_attention, args.d_model, activation=None, use_bias=False) # (mbatch_size, seq_len, d_model)

def feed_forward_network(X, args):
	'''
	Feed-forward network.

	Input/s
		X - input tensor (3D tensor).
		args - other arguments.

	Output/s
		Output of feed-forward network.

	'''
	return tf.layers.dense(tf.layers.dense(X, args.d_ff, activation=tf.nn.relu), args.d_model, activation=None)

def encoder(X, seq_len, training, P_drop, args, boolean_mask=False):
	'''
	Transformer encoder.

	Input/s
		X - input tensor (3D tensor).
		seq_len - sequence length.
		att_mask - sequence mask for attention (4D).
		norm_mask - sequence mask for normalisation (3D).
		training - training flag.
		P_drop - dropout probability.
		args - other arguments.

	Output/s
		Encoder uutput.

	'''

	## ATTENTION MASK
	max_seq_len = tf.reduce_max(seq_len) # max sequence length.
	att_mask = sequence_mask((tf.shape(seq_len)[0], max_seq_len, max_seq_len), seq_len) # sequence mask.

	## NORMALISATION MASK
	if args.norm_type in ['layer', 'batch']: norm_mask = tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32) # sequence mask.	
	else: norm_mask = None	

	## ENCODER
	positional_encoding =  get_timing_signal_1d(tf.reduce_max(seq_len), args.d_model) # positional encoding.
	embedding = tf.multiply(tf.layers.dense(X, args.d_model), tf.math.sqrt(tf.cast(args.d_model, tf.float32))) # embedding.
	encoder_input = tf.nn.dropout(tf.add(embedding, positional_encoding), rate=P_drop) # input embedding layer and positional encoding.
	encoder = []
	encoder.append(encoder_input)
	with tf.variable_scope('encoder'):
		for i in range(args.N):
			with tf.variable_scope('layer_' + str(i)):
				with tf.variable_scope('sub_layer_1'):
					sub_layer_1 = multi_head_attention(encoder[-1], encoder[-1], encoder[-1], att_mask, args)
					sub_layer_1 = tf.nn.dropout(sub_layer_1, rate=P_drop)
					sub_layer_1 = Normalisation(tf.add(encoder[-1], sub_layer_1), 
						args.norm_type, norm_mask, training)
				with tf.variable_scope('sub_layer_2'):
					sub_layer_2 = feed_forward_network(sub_layer_1, args)
					sub_layer_2 = tf.nn.dropout(sub_layer_2, rate=P_drop)
					encoder.append(Normalisation(tf.add(sub_layer_1, sub_layer_2), 
						args.norm_type, norm_mask, training))
	if boolean_mask: encoder[-1] = tf.boolean_mask(encoder[-1], tf.sequence_mask(seq_len))
	return tf.layers.dense(encoder[-1], args.d_out, use_bias=True)

def lr(d_model, step, warmup_steps=4000):
	'''
	Learning rate schedular from "Attention is all you need".
	Input/s

	Output/s

	'''
	d_model = np.float32(d_model) 
	arg1 = np.divide(1.0, np.sqrt(step))
	arg2 = step * (warmup_steps ** -1.5)
	return np.divide(1.0, np.sqrt(d_model)) * np.minimum(arg1, arg2)

## POSITIONAL ENCODING
#
## NOTE: this implementation is from http://jalammar.github.io/illustrated-transformer/
#
def get_timing_signal_1d(length,
	channels,
	min_timescale=1.0,
	max_timescale=1.0e4,
	start_index=0):

	"""Gets a bunch of sinusoids of different frequencies.
	Each channel of the input Tensor is incremented by a sinusoid of a different
	frequency and phase.
	This allows attention to learn to use absolute and relative positions.
	Timing signals should be added to some precursors of both the query and the
	memory inputs to attention.
	The use of relative position is possible because sin(x+y) and cos(x+y) can be
	expressed in terms of y, sin(x) and cos(x).
	In particular, we use a geometric sequence of timescales starting with
	min_timescale and ending with max_timescale.  The number of different
	timescales is equal to channels / 2. For each timescale, we
	generate the two sinusoidal signals sin(timestep/timescale) and
	cos(timestep/timescale).  All of these sinusoids are concatenated in
	the channels dimension.
	Args:
	  length: scalar, length of timing signal sequence.
	  channels: scalar, size of timing embeddings to create. The number of
		  different timescales is equal to channels / 2.
	  min_timescale: a float
	  max_timescale: a float
	  start_index: index of first position
	Returns:
	  a Tensor of timing signals [1, length, channels]
	"""
	position = tf.to_float(tf.range(length) + start_index)
	num_timescales = channels // 2
	log_timescale_increment = (
		math.log(float(max_timescale) / float(min_timescale)) /
		(tf.to_float(num_timescales) - 1))
	inv_timescales = min_timescale * tf.exp(
		tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
	scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
	signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
	signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
	signal = tf.reshape(signal, [1, length, channels])
	return signal

