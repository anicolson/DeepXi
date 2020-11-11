## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Add, Bidirectional, Dense, \
	LayerNormalization, LSTM, Masking, ReLU, TimeDistributed
import numpy as np

class ResLSTM:
	"""
	Residual long short-term memory network. Frame-wise layer normalisation is
	used.
	"""
	def __init__(
		self,
		inp,
		n_outp,
		n_blocks,
		d_model,
		outp_act,
		unroll=False,
		):
		"""
		Argument/s:
			inp - input placeholder.
			n_outp - number of output nodes.
			n_blocks - number of residual blocks.
			d_model - model size.
			outp_act - output activation function.
			unroll - unroll recurrent state (can't be used as sequence length
				changes).
		"""
		self.n_outp = n_outp
		self.d_model = d_model
		self.unroll = unroll
		seq_mask = Masking(mask_value=0.0).compute_mask(inp)
		x = self.feedforward(inp)
		for _ in range(n_blocks): x = self.block(x, seq_mask)
		outp_fc = Dense(self.n_outp)
		self.outp = TimeDistributed(outp_fc)(x)

		if outp_act == "Sigmoid": self.outp = Activation('sigmoid')(self.outp)
		elif outp_act == "ReLU": self.outp = ReLU()(self.outp)
		elif outp_act == "Linear": self.outp = self.outp
		else: raise ValueError("Invalid outp_act")

	def block(self, inp, seq_mask):
		"""
		Residual LSTM block.

		Argument/s:
			inp - input placeholder.
			seq_mask - sequence mask.

		Returns:
			residual - output of block.
		"""
		lstm = LSTM(self.d_model, unroll=self.unroll)(inp, mask=seq_mask)
		residual = Add()([inp, lstm])
		return residual

	def feedforward(self, inp):
		"""
		Feedforward layer.

		Argument/s:
			inp - input placeholder.

		Returns:
			act - output of feedforward layer.
		"""
		ff = Dense(self.d_model, use_bias=False)(inp)
		norm = LayerNormalization(axis=2, epsilon=1e-6)(ff)
		act = ReLU()(norm)
		return act

class ResBiLSTM(ResLSTM):
	"""
	Residual bidirectional long short-term memory network. Frame-wise layer normalisation is
	used.
	"""
	def block(self, inp, seq_mask):
		"""
		Residual BiLSTM block.

		Argument/s:
			inp - input placeholder.
			seq_mask - sequence mask.

		Returns:
			residual - output of block.
		"""
		layer = LSTM(self.d_model, unroll=self.unroll)
		bilstm = Bidirectional(layer, merge_mode='sum')(inp, mask=seq_mask)
		residual = Add()([inp, bilstm])
		return residual
