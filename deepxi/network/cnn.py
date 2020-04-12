## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Add, Conv1D, Conv2D, Dense, Dropout, \
	Flatten, LayerNormalization, MaxPooling2D, ReLU
import numpy as np

class TCN:
	"""
	Temporal convolutional network using bottlekneck residual blocks and cyclic
	dilation rate. Frame-wise layer normalisation is used.
	"""
	def __init__(
		self,
		inp,
		n_outp,
		n_blocks,
		d_model,
		d_f,
		k,
		max_d_rate
		):
		"""
		Argument/s:
			inp - input placeholder.
			n_outp - number of output nodes.
			n_blocks - number of bottlekneck residual blocks.
			d_model - model size.
			d_f - bottlekneck size.
			k - kernel size.
			max_d_rate - maximum dilation rate.
			softmax - softmax output flag.
		"""
		self.d_model = d_model
		self.d_f = d_f
		self.k = k
		self.n_outp = n_outp
		self.first_layer = self.feedforward(inp)
		self.layer_list = [self.first_layer]
		for i in range(n_blocks): self.layer_list.append(self.block(self.layer_list[-1], int(2**(i%(np.log2(max_d_rate)+1)))))
		self.logits = Conv1D(self.n_outp, 1, dilation_rate=1, use_bias=True)(self.layer_list[-1])
		self.outp = Activation('sigmoid')(self.logits)

	def feedforward(self, inp):
		"""
		Feedforward layer.

		Argument/s:
			inp - input placeholder.

		Returns:
			act - feedforward layer output.
		"""
		ff = Conv1D(self.d_model, 1, dilation_rate=1, use_bias=False)(inp)
		norm = LayerNormalization(axis=2, epsilon=1e-6)(ff)
		act = ReLU()(norm)
		return act

	def block(self, inp, d_rate):
		"""
		Bottlekneck residual block.

		Argument/s:
			inp - input placeholder.
			d_rate - dilation rate.

		Returns:
			residual - output of block.
		"""
		self.conv_1 = self.unit(inp, self.d_f, 1, 1, False)
		self.conv_2 = self.unit(self.conv_1, self.d_f, self.k, d_rate,
			False)
		self.conv_3 = self.unit(self.conv_2, self.d_model, 1, 1, True)
		residual = Add()([inp, self.conv_3])
		return residual

	def unit(self, inp, n_filt, k, d_rate, use_bias):
		"""
		Convolutional unit.

		Argument/s:
			inp - input placeholder.
			n_filt - filter size.
			k - kernel size.
			d_rate - dilation rate.
			use_bias - bias flag.

		Returns:
			conv - output of unit.
		"""
		norm = LayerNormalization(axis=2, epsilon=1e-6)(inp)
		act = ReLU()(norm)
		conv = Conv1D(n_filt, k, padding="causal", dilation_rate=d_rate,
			use_bias=use_bias)(act)
		return conv
