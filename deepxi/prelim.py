## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from deepxi.gain import gfunc
from deepxi.network.rnn import ResLSTM
from deepxi.network.tcn import ResNet
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import math
import numpy as np
import tensorflow as tf

class Prelim():
	"""
	This preliminary class was used as the basis for the DeepXi class.
	"""
	def __init__(
		self,
		n_feat,
		network
		):
		self.n_feat = n_feat
		self.n_outp = self.n_feat
		if self.n_feat < 5: raise ValueError('More input features are required for this exampple.')
		self.inp = Input(name='inp', shape=[None, self.n_feat], dtype='float32')
		self.mask = tf.keras.layers.Masking(mask_value=0.0)(self.inp)
		if network == 'ResNet': self.network = ResNet(self.mask, self.n_outp, B=40, d_model=256, d_f=64, k=3, max_d_rate=16)
		elif network == 'ResLSTM': self.network = ResLSTM(self.mask, self.n_outp, n_blocks=3, d_model=256)
		else: raise ValueError('Invalid network type.')
		self.model = Model(inputs=self.inp, outputs=self.network.outp)
		self.model.summary()

	def train(
		self,
		mbatch_size=8,
		max_epochs=20,
		):
		self.mbatch_size=mbatch_size
		self.max_epochs=max_epochs
		self.batch_size=100

		self.model.compile(
			sample_weight_mode="temporal",
			loss="binary_crossentropy",
			optimizer=Adam(lr=0.001, clipvalue=1.0)
			)

		train_dataset = self.dataset()

		self.model.fit(
			train_dataset,
			epochs=max_epochs,
			steps_per_epoch=math.ceil(self.batch_size/self.mbatch_size)
			)

		x_test, y_test, _ = list(train_dataset.take(1).as_numpy_iterator())[0]
		y_hat = self.model.predict(x_test[0:1])

		np.set_printoptions(precision=2, suppress=True)
		print("Target:")
		print(np.asarray(y_test[0,0:5,0:self.n_feat]))
		print("Prediction:")
		print(y_hat[0,0:5,0:self.n_feat])

	def dataset(self, buffer_size=16):
		dataset = tf.data.Dataset.from_generator(
			self.mbatch_gen,
			(tf.float32, tf.float32, tf.float32),
			(tf.TensorShape([None, None, self.n_feat]),
				tf.TensorShape([None, None, self.n_outp]),
				tf.TensorShape([None, None]))
			)
		dataset = dataset.prefetch(buffer_size)
		return dataset

	def mbatch_gen(self):
		for _ in range(self.max_epochs):
			for _ in range(math.ceil(self.batch_size/self.mbatch_size)):
				max_seq_len = 75
				min_seq_len = 45
				x_train = np.random.rand(self.mbatch_size, max_seq_len, self.n_feat)
				y_frame = np.zeros(self.n_feat)
				y_frame[0] = 0.05
				y_frame[1] = 0.99
				y_frame[2] = 0.5
				y_frame[3] = 0.01
				y_frame[4] = 0.75
				y_train = np.tile(y_frame, (self.mbatch_size, max_seq_len, 1))
				seq_len = np.random.randint(min_seq_len, max_seq_len+1, self.mbatch_size)
				seq_mask = tf.cast(tf.sequence_mask(seq_len, maxlen=max_seq_len), tf.float32)
				x_train = tf.multiply(x_train, tf.expand_dims(seq_mask, 2))
				y_train = tf.multiply(y_train, tf.expand_dims(seq_mask, 2))
				yield x_train, y_train, seq_mask
