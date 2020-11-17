## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from tensorflow.keras.layers import Activation, Add, \
	Conv1D, Embedding, Layer, LayerNormalization, Masking, ReLU
import math, sys
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

class MHANet:
	"""
	Multi-head attention network.
	"""
	def __init__(
		self,
		inp,
		n_outp,
		d_model,
		n_blocks,
		n_heads,
		warmup_steps,
		causal,
		outp_act,
		):
		"""
		Argument/s:
			inp - input placeholder.
			n_outp - number of outputs.
			d_model - model size.
			n_blocks - number of blocks.
			n_heads - number of attention heads.
			warmup_steps - number of warmup steps.
			causal - causal flag.
			outp_act - output activation function.
		"""
		self.n_outp = n_outp
		self.d_model = d_model
		self.n_blocks = n_blocks
		self.n_heads = n_heads
		self.d_ff = d_model*4
		self.warmup_steps = warmup_steps
		self.d_k = self.d_model // self.n_heads

		att_mask, seq_mask = AttentionMask(causal, -1.0e9)(inp)

		x = Conv1D(self.d_model, 1, use_bias=False)(inp)
		x = LayerNormalization(axis=2, epsilon=1e-6, center=True, scale=True)(x)
		x = ReLU()(x)

		for _ in range(self.n_blocks): x = self.block(x, att_mask, seq_mask)

		self.outp = Conv1D(self.n_outp, 1, use_bias=True)(x)

		if outp_act == "Sigmoid": self.outp = Activation('sigmoid')(self.outp)
		elif outp_act == "ReLU": self.outp = ReLU()(self.outp)
		elif outp_act == "Linear": self.outp = self.outp
		else: raise ValueError("Invalid outp_act")

	def block(self, x, att_mask, seq_mask):
		"""
		MHANet block.

		Argument/s:
			x - input.
			att_mask - attention mask.
			seq_mask - sequence mask.

		Returns:
			layer_2 - output of second layer.
		"""
		layer_1 = MultiHeadAttention(d_model=self.d_model,
			n_heads=self.n_heads)(x, x, x, att_mask, seq_mask)
		layer_1 = Add()([x, layer_1])
		layer_1 = LayerNormalization(axis=2, epsilon=1e-6, center=True,
			scale=True)(layer_1)

		layer_2 = self.feed_forward_network(layer_1)
		layer_2 = Add()([layer_1, layer_2])
		layer_2 = LayerNormalization(axis=2, epsilon=1e-6, center=True,
			scale=True)(layer_2)
		return layer_2

	def feed_forward_network(self, x):
		"""
		Feed forward network.

		Argument/s:
			inp - input placeholder.

		Returns:
			x - output of second feed forward layer.
		"""
		x = Conv1D(self.d_ff, 1, use_bias=True)(x)
		x = ReLU()(x)
		x = Conv1D(self.d_model, 1, use_bias=True)(x)
		return x

class MultiHeadAttention(Layer):
	"""
	Multi-head attention module.
	"""
	def __init__(self, d_model, n_heads):
		"""
		Argument/s:
			d_model - model size.
			n_heads - number of heads.
		"""
		super(MultiHeadAttention, self).__init__()
		self.d_model = d_model
		self.n_heads = n_heads
		assert d_model % self.n_heads == 0
		self.d_k = self.d_v = d_model // self.n_heads

		self.linear_q = Conv1D(self.d_model, 1, use_bias=False)
		self.linear_k = Conv1D(self.d_model, 1, use_bias=False)
		self.linear_v = Conv1D(self.d_model, 1, use_bias=False)
		self.linear_o = Conv1D(self.d_model, 1, use_bias=False)

	def split_heads(self, x, batch_size):
		"""
		Split the last dimension into (n_heads, d_k).
		Transpose the result such that the shape is
		(batch_size, n_heads, seq_len, d_k)

		Argument/s:
			x - input.
			batch_size - size of batch.

		Returns:
			Split heads.
		"""
		x = tf.reshape(x, (batch_size, -1, self.n_heads, self.d_k))
		return tf.transpose(x, perm=[0, 2, 1, 3])

	def call(self, q, v, k, att_mask, seq_mask):
		"""
		Argument/s:
			q - query.
			v - value.
			k - key.
			att_mask - attention mask.
			seq_mask - sequence mask.

		Returns:
			Multi-head attention output.
		"""
		batch_size = tf.shape(q)[0]

		q = self.linear_q(q) # (batch_size, seq_len, d_model).
		v = self.linear_v(v) # (batch_size, seq_len, d_model).
		k = self.linear_k(k) # (batch_size, seq_len, d_model).

		q = self.split_heads(q, batch_size) # (batch_size, n_heads, seq_len, d_k).
		v = self.split_heads(v, batch_size) # (batch_size, n_heads, seq_len, d_k).
		k = self.split_heads(k, batch_size) # (batch_size, n_heads, seq_len, d_k).

		att_mask = tf.expand_dims(att_mask, axis=1) # (batch_size, 1, seq_len, seq_len).
		seq_mask = tf.expand_dims(seq_mask, axis=1) # (batch_size, 1, seq_len, seq_len).

		scaled_attention = ScaledDotProductAttention(self.d_k,
			)(q, v, k, att_mask, seq_mask) # (batch_size, n_heads, seq_len, d_k).

		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # (batch_size, seq_len, n_heads, d_k).

		concat_attention = tf.reshape(scaled_attention,
			(batch_size, -1, self.d_model)) # (batch_size, seq_len, d_model).

		output = self.linear_o(concat_attention) # (batch_size, seq_len, d_model).
		return output

class ScaledDotProductAttention(Layer):
	"""
	Scaled dot-product attention.
	"""
	def __init__(self, d_k):
		"""
		Argument/s:
			d_k - key size.

		"""
		super(ScaledDotProductAttention, self).__init__()
		self.d_k = float(d_k)

	def call(self, q, v, k, att_mask, seq_mask):
		"""
		Argument/s:
			q - query.
			v - value.
			k - key.
			att_mask - attention mask.
			seq_mask - sequence mask.

		Returns:
		  output of scaled dot-product attention.
		"""
		attention_weights = tf.matmul(q, k, transpose_b=True) # (..., seq_len, seq_len).
		scaled_attention_weights = tf.math.truediv(attention_weights, tf.math.sqrt(self.d_k))
		scaled_attention_weights_masked = tf.math.add(scaled_attention_weights, att_mask)
		normalised_attention_weights = tf.nn.softmax(scaled_attention_weights_masked, axis=-1)
		normalised_attention_weights = tf.multiply(normalised_attention_weights, seq_mask)
		output = tf.matmul(normalised_attention_weights, v) # (..., seq_len, d_v).
		return output

class AttentionMask(Layer):
	"""
	Computes attention mask.
	"""
	def __init__(self, causal, mask_value=-1e9):
		"""
		Argument/s:
			causal - causal attention mask flag.
			mask_value - value used to mask components that aren't to be attended
				to (typically -1e9).
		"""
		super(AttentionMask, self).__init__()
		self.causal = causal
		self.mask_value = mask_value
		if not isinstance(mask_value, float): raise ValueError("Mask value must be a float.")

	def call(self, inp):
		"""
		Compute attention mask.

		Argument/s:
			inp - used to compute sequence mask.

		Returns:
			Attention mask.
		"""
		batch_size = tf.shape(inp)[0]
		max_seq_len = tf.shape(inp)[1]
		flat_seq_mask = Masking(mask_value=0.0).compute_mask(inp)
		seq_mask = self.merge_masks(tf.expand_dims(flat_seq_mask, axis=1), tf.expand_dims(flat_seq_mask, axis=2))
		causal_mask = self.lower_triangular_mask([1,max_seq_len,max_seq_len]) if self.causal else None
		logical_mask = self.merge_masks(causal_mask, seq_mask)
		unmasked = tf.zeros([batch_size, max_seq_len, max_seq_len])
		masked = tf.fill([batch_size, max_seq_len, max_seq_len], self.mask_value)
		att_mask = tf.where(logical_mask, unmasked, masked)
		seq_mask = tf.cast(seq_mask, tf.float32)
		return att_mask, seq_mask

	def lower_triangular_mask(self, shape):
		"""
		Creates a lower-triangular boolean mask over the last 2 dimensions.

		Argument/s:
			shape - shape of mask.

		Returns:
			causal mask.
		"""
		row_index = tf.math.cumsum(
			tf.ones(shape=shape, dtype=tf.int32, name="row"), axis=-2)
		col_index = tf.math.cumsum(
			tf.ones(shape=shape, dtype=tf.int32, name="col"), axis=-1)
		return tf.math.greater_equal(row_index, col_index)

	def merge_masks(self, x, y):
		"""
		Merges a sequence mask and a causal mask to make an attantion mask.

		Argument/s:
			x - mask.
			y - mask.

		Returns:
			Attention mask.
		"""
		if x is None: return y
		if y is None: return x
		return tf.math.logical_and(x, y)

class MHANetV2(MHANet):
	"""
	Multi-head attention network implemented using tfa.layers.MultiHeadAttention.
	"""
	def __init__(
		self,
		inp,
		n_outp,
		d_model,
		n_blocks,
		n_heads,
		warmup_steps,
		causal,
		outp_act,
		):
		"""
		Argument/s:
			inp - input placeholder.
			n_outp - number of outputs.
			d_model - model size.
			n_blocks - number of blocks.
			n_heads - number of attention heads.
			warmup_steps - number of warmup steps.
			causal - causal flag.
			outp_act - output activation function.
		"""
		self.n_outp = n_outp
		self.d_model = d_model
		self.n_blocks = n_blocks
		self.n_heads = n_heads
		self.d_ff = d_model*4
		self.warmup_steps = warmup_steps
		self.d_k = self.d_model // self.n_heads

		att_mask = AttentionMaskV2(causal)(inp)

		x = Conv1D(self.d_model, 1, use_bias=False)(inp)
		x = LayerNormalization(axis=2, epsilon=1e-6, center=True, scale=True)(x)
		x = ReLU()(x)

		for _ in range(self.n_blocks): x = self.block(x, att_mask)

		self.outp = Conv1D(self.n_outp, 1, use_bias=True)(x)

		if outp_act == "Sigmoid": self.outp = Activation('sigmoid')(self.outp)
		elif outp_act == "ReLU": self.outp = ReLU()(self.outp)
		elif outp_act == "Linear": self.outp = self.outp
		else: raise ValueError("Invalid outp_act")

	def block(self, x, att_mask):
		"""
		MHANet block.

		Argument/s:
			x - input.
			att_mask - attention mask.

		Returns:
			layer_2 - output of second layer.
		"""
		layer_1 = tfa.layers.MultiHeadAttention(
			head_size=self.d_k,
			num_heads=self.n_heads,
			output_size=self.d_model,
			dropout=0.0,
			use_projection_bias=False,
		)([x, x, x, att_mask])
		layer_1 = Add()([x, layer_1])
		layer_1 = LayerNormalization(axis=2, epsilon=1e-6, center=True,
			scale=True)(layer_1)

		layer_2 = self.feed_forward_network(layer_1)
		layer_2 = Add()([layer_1, layer_2])
		layer_2 = LayerNormalization(axis=2, epsilon=1e-6, center=True,
			scale=True)(layer_2)
		return layer_2

class AttentionMaskV2(AttentionMask):
	"""
	Computes attention mask appropriate for tfa.layers.MultiHeadAttention.
	"""
	def __init__(self, causal):
		"""
		Argument/s:
			causal - causal attention mask flag.
		"""
		super(AttentionMaskV2, self).__init__(causal)
		self.causal = causal

	def call(self, inp):
		"""
		Compute attention mask.

		Argument/s:
			inp - used to compute sequence mask.

		Returns:
			Attention mask.
		"""
		batch_size = tf.shape(inp)[0]
		max_seq_len = tf.shape(inp)[1]
		flat_seq_mask = Masking(mask_value=0.0).compute_mask(inp)
		seq_mask = self.merge_masks(tf.expand_dims(flat_seq_mask, axis=1), tf.expand_dims(flat_seq_mask, axis=2))
		causal_mask = self.lower_triangular_mask([1, max_seq_len, max_seq_len]) if self.causal else None
		logical_mask = self.merge_masks(causal_mask, seq_mask)
		att_mask = tf.cast(logical_mask, tf.float32)
		att_mask = tf.reshape(att_mask, [batch_size, 1, max_seq_len, max_seq_len])
		return att_mask

class MHANetV3(MHANetV2):
	"""
	MHANetV2 with positional encoding from BERT (https://arxiv.org/abs/1810.04805).
	"""
	def __init__(
		self,
		inp,
		n_outp,
		d_model,
		n_blocks,
		n_heads,
		warmup_steps,
		max_len,
		causal,
		outp_act,
		):
		"""
		Argument/s:
			inp - input placeholder.
			n_outp - number of outputs.
			d_model - model size.
			n_blocks - number of blocks.
			n_heads - number of attention heads.
			warmup_steps - number of warmup steps.
			max_len - maximum length for positional encoding.
			causal - causal flag.
			outp_act - output activation function.
		"""
		self.n_outp = n_outp
		self.d_model = d_model
		self.n_blocks = n_blocks
		self.n_heads = n_heads
		self.d_ff = d_model*4
		self.max_len = max_len
		self.warmup_steps = warmup_steps
		self.d_k = self.d_model // self.n_heads

		att_mask = AttentionMaskV2(causal)(inp)

		x = Conv1D(self.d_model, 1, use_bias=False)(inp)
		x = LayerNormalization(axis=2, epsilon=1e-6, center=True, scale=True)(x)
		x = ReLU()(x)

		## Add postitional encoding.
		position_idx = tf.tile([tf.range(tf.shape(x)[1])], [tf.shape(x)[0], 1])
		positional_encoding = Embedding(self.max_len, self.d_model)(position_idx)
		x = Add()([x, positional_encoding])

		for _ in range(self.n_blocks): x = self.block(x, att_mask)

		self.outp = Conv1D(self.n_outp, 1, use_bias=True)(x)

		if outp_act == "Sigmoid": self.outp = Activation('sigmoid')(self.outp)
		elif outp_act == "ReLU": self.outp = ReLU()(self.outp)
		elif outp_act == "Linear": self.outp = self.outp
		else: raise ValueError("Invalid outp_act")
