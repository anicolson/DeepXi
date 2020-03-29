## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Add, Conv1D, LayerNormalization, LSTM, ReLU

#Activation,   Conv2D, Dense, Dropout, \
#	Flatten, LayerNormalization, MaxPooling2D, ReLU

import numpy as np

class ResLSTM:
	"""
	Temporal convolutional network using bottlekneck residual blocks and cyclic dilation rate.
	"""
	def __init__(
		self, 
		inp, 
		n_outp, 
		n_layers=3, 
		d_model=256, 
		):
		"""
		Argument/s:
			inp - input placeholder.
			n_outp - number of output nodes.
			B - number of bottlekneck residual blocks.
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
		for i in range(n_layers): self.layer_list.append(self.block(self.layer_list[-1]))
		self.logits = Conv1D(self.n_outp, 1, dilation_rate=1, use_bias=True)(self.layer_list[-1])
		self.outp = Activation('sigmoid')(self.logits)

	def feedforward(self, inp):
		"""
		Argument/s:
			inp - input placeholder.
		"""
		ff = Conv1D(self.d_model, 1, dilation_rate=1, use_bias=False)(inp)
		norm = LayerNormalization(axis=2, epsilon=1e-6)(ff)
		act = ReLU()(norm)
		return act

	def block(self, inp):
		"""
		Argument/s:
			inp - input placeholder.
		"""
		self.lstm = LSTM(self.d_model)(inp)
		residual = Add()([inp, self.lstm]) 
		return residual

