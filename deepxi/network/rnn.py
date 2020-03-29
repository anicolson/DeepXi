## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Add, Dense, \
	LayerNormalization, LSTM, ReLU, TimeDistributed


import numpy as np

class ResLSTM:
	"""
	Temporal convolutional network using bottlekneck residual blocks and cyclic dilation rate.
	"""
	def __init__(
		self, 
		inp, 
		n_outp, 
		n_blocks=3, 
		d_model=256, 
		):
		"""
		Argument/s:
			inp - input placeholder.
			n_outp - number of output nodes.
			n_blocks - number of residual blocks.
			d_model - model size.
			d_f - bottlekneck size.
			k - kernel size.
			max_d_rate - maximum dilation rate.
			softmax - softmax output flag.
		"""
		self.n_outp = n_outp
		self.d_model = d_model
		self.first_layer = self.feedforward(inp)
		self.layer_list = [self.first_layer]
		for _ in range(n_blocks): self.layer_list.append(self.block(self.layer_list[-1]))
		self.logits = TimeDistributed(Dense(self.n_outp))(self.layer_list[-1])
		self.outp = Activation('sigmoid')(self.logits)

	def feedforward(self, inp):
		"""
		Feedforward layer with frame-wise layer normalisation and ReLU activation.

		Argument/s:
			inp - input placeholder.
		"""
		ff = TimeDistributed(Dense(self.d_model, use_bias=False))(inp)
		norm = LayerNormalization(axis=2, epsilon=1e-6)(ff)
		act = ReLU()(norm)
		return act

	def block(self, inp):
		"""
		Residual block.

		Argument/s:
			inp - input placeholder.
		"""
		lstm = LSTM(self.d_model)(inp)
		residual = Add()([inp, lstm]) 
		return residual

