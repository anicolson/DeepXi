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

# [1] Xu, J., Sun, X., Zhang, Z., Zhao, G., & Lin, J. (2019). Understanding and
# Improving Layer Normalization. In Advances in Neural Information Processing
# Systems (pp. 4381-4391).

class ResNet:
	"""
	Residual network using bottlekneck residual blocks and cyclic
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
		max_d_rate,
		padding,
		outp_act,
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
			padding - padding type.
			outp_act - output activation function.
		"""
		self.d_model = d_model
		self.d_f = d_f
		self.k = k
		self.n_outp = n_outp
		self.padding = padding
		self.first_layer = self.feedforward(inp)
		self.layer_list = [self.first_layer]
		for i in range(n_blocks): self.layer_list.append(self.block(self.layer_list[-1],
			int(2**(i%(np.log2(max_d_rate)+1)))))
		self.outp = Conv1D(self.n_outp, 1, dilation_rate=1,
			use_bias=True)(self.layer_list[-1])

		if outp_act == "Sigmoid": self.outp = Activation('sigmoid')(self.outp)
		elif outp_act == "ReLU": self.outp = ReLU()(self.outp)
		elif outp_act == "Linear": self.outp = self.outp
		else: raise ValueError("Invalid outp_act")

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
		conv = Conv1D(n_filt, k, padding=self.padding, dilation_rate=d_rate,
			use_bias=use_bias)(act)
		return conv

class ResNetV2:
	"""
	Residual network using bottlekneck residual blocks and cyclic dilation rate.
	Frame-wise layer normalisation is used with no scale or centre parameters to
	reduce overfitting, as in [1]. Bias is used for all convolutional units.
	"""
	def __init__(
		self,
		inp,
		n_outp,
		n_blocks,
		d_model,
		d_f,
		k,
		max_d_rate,
		padding,
		unit_type,
		outp_act,
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
			padding - padding type.
			unit_type - convolutional unit type.
			outp_act - output activation function.
		"""
		self.d_model = d_model
		self.d_f = d_f
		self.k = k
		self.n_outp = n_outp
		self.padding = padding
		self.unit_type = unit_type
		self.first_layer = self.feedforward(inp)
		self.layer_list = [self.first_layer]
		for i in range(n_blocks): self.layer_list.append(self.block(self.layer_list[-1],
			int(2**(i%(np.log2(max_d_rate)+1)))))
		self.outp = Conv1D(self.n_outp, 1, dilation_rate=1,
			use_bias=True)(self.layer_list[-1])

		if outp_act == "Sigmoid": self.outp = Activation('sigmoid')(self.outp)
		elif outp_act == "ReLU": self.outp = ReLU()(self.outp)
		elif outp_act == "Linear": self.outp = self.outp
		else: raise ValueError("Invalid outp_act")

	def feedforward(self, inp):
		"""
		Feedforward layer.

		Argument/s:
			inp - input placeholder.

		Returns:
			act - feedforward layer output.
		"""
		ff = Conv1D(self.d_model, 1, dilation_rate=1, use_bias=True)(inp)
		norm = LayerNormalization(axis=2, epsilon=1e-6, center=False,
			scale=True)(ff)
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
		self.conv_1 = self.unit(inp, self.d_f, 1, 1)
		self.conv_2 = self.unit(self.conv_1, self.d_f, self.k, d_rate)
		self.conv_3 = self.unit(self.conv_2, self.d_model, 1, 1)
		residual = Add()([inp, self.conv_3])
		return residual

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
		if self.unit_type == "LN->ReLU->W+b":
			x = LayerNormalization(axis=2, epsilon=1e-6, center=False,
				scale=False)(inp)
			x = ReLU()(x)
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

class ResNetV3(ResNetV2):
	"""
	Amends incorrect first layer of ResNetV2 to be in line with [1].
	"""
	def feedforward(self, inp):
		"""
		Feedforward layer.

		Argument/s:
			inp - input placeholder.

		Returns:
			act - feedforward layer output.
		"""
		ff = Conv1D(self.d_model, 1, dilation_rate=1, use_bias=True)(inp)
		act = ReLU()(ff)
		norm = LayerNormalization(axis=2, epsilon=1e-6, center=False,
			scale=False)(act)
		return norm
