## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Add, Concatenate, Conv1D, \
	Dense, LayerNormalization, ReLU
import numpy as np

class RDLNet:
	"""
	Residual-dense lattice network.
	"""
	def __init__(
		self,
		inp,
		n_outp,
		n_blocks,
		length,
		m_1,
		padding,
		unit_type,
		sigmoid_outp,
		):
		"""
		Argument/s:
			inp - input placeholder.
			n_outp - number of output nodes.
			n_blocks - number of bottlekneck residual blocks.
			length - length of the RDL block.
			m_1 - output size at h = 1.
			padding - padding type.
			unit_type - convolutional unit type.
			sigmoid_outp - use a sigmoid output activation.
		"""
		self.length = length
		self.height = (self.length - 1) // 2 + 1
		self.midpoint = (self.length + 1) // 2
		self.m_1 = m_1
		self.n_outp = n_outp
		self.padding = padding
		self.unit_type = unit_type
		self.layer_list = [inp]
		for i in range(n_blocks):
			self.layer_list.append(Concatenate()([self.block(self.layer_list[-1]),
				self.layer_list[-1]]))
		self.outp = Conv1D(self.n_outp, 1, dilation_rate=1,
			use_bias=True)(self.layer_list[-1])
		if sigmoid_outp: self.outp = Activation('sigmoid')(self.outp)

	def block(self, inp):
		"""
		RDL block.

		Argument/s:
			inp - input placeholder.

		Returns:
			residual - output of block.
		"""
		block = [[None] * self.length  for i in range(self.height)]

		for l in range(self.midpoint):
			for h in range(self.height):
				if l == (self.midpoint - 1): h = self.height - h - 1
				if h <= l:

					## INPUT TO UNIT
					if l==0: unit_inp = inp
					elif l==h: unit_inp = block[h-1][l-1]
					else: unit_inp = block[h][l-1]

					## ACTIVATION OF UNIT
					n_filt=self.m_1/(2**(h))
					k=2*(h+1)-1
					d_rate=2**h
					U = self.unit(unit_inp, n_filt, k, d_rate)

					## OUTPUT OF UNIT (RESIDUAL)
					if l==h: unit_outp = U
					elif (h==0) and (l==1): unit_outp = self.weighted_residual(U, inp)
					elif (h+1)==l: unit_outp = self.weighted_residual(U, block[h-1][l-2])
					else: unit_outp = self.weighted_residual(U, block[h][l-2])

					## OUTPUT OF UNIT (CONCAT)
					if l==0: pass
					elif h==self.height-1: pass
					elif (h==0) and l<(self.midpoint-1): pass
					elif l==self.midpoint-1: unit_outp = Concatenate()([unit_outp, block[h+1][l]])
					else: unit_outp = Concatenate()([unit_outp, block[h-1][l]])

					block[h][l] = unit_outp

		for l in range(self.midpoint, self.length):
			for h in reversed(range(self.height)):
				if h < self.length - l:

					block[h][l] = self.unit(inp, n_filt, k, d_rate)

					## ACTIVATION OF UNIT
					n_filt=self.m_1/(2**(h))
					k=2*(h+1)-1
					d_rate=2**h
					U = self.unit(block[h][l-1], n_filt, k, d_rate)

					## OUTPUT OF UNIT (RESIDUAL)
					unit_outp = self.weighted_residual(U, block[h][l-2])

					## OUTPUT OF UNIT (CONCAT)
					if l==self.length-h-1: pass
					else: unit_outp = Concatenate()([unit_outp, block[h+1][l]])

					block[h][l] = unit_outp

		return block[0][self.length-1]

	def weighted_residual(self, x, y):
		"""
		Weighted residual link. Larger input will be projected to the smaller
		input size.

		Argument/s:
			x - tensor.
			y - tensor.

		Returns:
			weighted residual link.
		"""
		x_dims = x.get_shape().as_list()[-1]
		y_dims = y.get_shape().as_list()[-1]
		if x_dims > y_dims: x = Conv1D(y_dims, 1, use_bias=False)(x)
		elif x_dims < y_dims: y = Conv1D(x_dims, 1, use_bias=False)(y)
		return Add()([x, y])

	def unit(self, inp, n_filt, k, d_rate):
		"""
		Convolutional unit.

		Argument/s:
			inp - input placeholder.
			n_filt - filter size.
			k - kernel size.
			d_rate - dilation rate.

		Returns:
			conv - output of unit.
		"""
		if self.unit_type == "scale*LN+center->ReLU->W+b":
			x = LayerNormalization(axis=2, epsilon=1e-6, center=False,
				scale=False)(inp)
			x = ReLU()(inp)
			x = Conv1D(n_filt, k, padding=self.padding, dilation_rate=d_rate,
				use_bias=True)(x)
		elif self.unit_type == "ReLU->LN->W+b":
			x = ReLU()(inp)
			x = LayerNormalization(axis=2, epsilon=1e-6, center=False,
				scale=False)(x)
			x = Conv1D(n_filt, k, padding=self.padding, dilation_rate=d_rate,
				use_bias=True)(x)
		else: raise ValueError("Invalid unit_type.")
		return x
