## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from deepxi.utils import save_mat
from scipy.stats import skew
from tqdm import tqdm
import numpy as np
import scipy.special as spsp
import tensorflow as tf

def map_selector(map_type, params):
	"""
	"""
	if map_type == "Linear":
		return Linear(map_type)
	elif map_type == "DB":
		return DB(map_type)
	elif "Clip" in map_type:
		return Clip(map_type, params)
	elif "Logistic" in map_type:
		return Logistic(map_type, params)
	elif "Standardise" in map_type:
		return Standardise(map_type, params)
	elif "MinMaxScaling" in map_type:
		return MinMaxScaling(map_type, params)
	elif "NormalCDF" in map_type:
		return NormalCDF(map_type)
	elif "TruncatedLaplaceCDF" in map_type:
		return TruncatedLaplaceCDF(map_type, params)
	elif "LaplaceCDF" in map_type:
		return LaplaceCDF(map_type, params)
	elif "UniformCDF" in map_type:
		return UniformCDF(map_type, params)
	elif "Square" in map_type:
		return Square(map_type)
	# elif "TruncatedDoubleGammaCDF" in map_type:
	# 	return TruncatedDoubleGammaCDF(map_type, params)
	else: raise ValueError("Invalid map_type.")

class Map():
	"""
	Base map class.
	"""
	def __init__(self, map_type, params=None):
		"""
		Argument/s:

		"""
		self.map_type = map_type
		self.ten = tf.cast(10.0, tf.float32)
		self.one = tf.cast(1.0, tf.float32)

		if isinstance(params, list):
			self.params = [tf.cast(param, tf.float32) if param is not None else param for param in params]
		else:
			self.params = tf.cast(params, tf.float32) if params is not None else params

	def db(self, x):
		"""
		Converts power value to power in decibels.

		Argument/s:
			x - power value.

		Returns:
			power in decibels.
		"""
		x = tf.maximum(x, 1e-12)
		return tf.multiply(self.ten, tf.truediv(tf.math.log(x), tf.math.log(self.ten)))

	def db_inverse(self, x_db):
		"""
		Converts power in decibels to power value.

		Argument/s:
			x_db - power in decibels.

		Returns:
			x - power value.
		"""
		return tf.math.pow(self.ten, tf.truediv(x_db, self.ten))

	def stats(self, x):
		"""
		The base stats() function is used when no statistics are requied for
		the map function.

		Argument/s:
			x - a set of samples.
		"""
		pass

class Linear(Map):
	"""
	Linear map, i.e. no map.
	"""
	def map(self, x):
		"""
		Returns input.

		Argument/s:
			x - value.

		Returns:
			x.
		"""
		return x

	def inverse(self, x):
		"""
		Returns input.

		Argument/s:
			x - value.

		Returns:
			x.
		"""
		return x

class Square(Map):
	"""
	Square map.
	"""
	def map(self, x):
		"""
		Returns input.

		Argument/s:
			x - value.

		Returns:
			x^2.
		"""
		x_bar = tf.math.square(x)
		if 'DB' in self.map_type: x_bar = self.db(x_bar)
		return x_bar

	def inverse(self, x_bar):
		"""
		Returns input.

		Argument/s:
			x_bar - square value.

		Returns:
			x.
		"""
		if 'DB' in self.map_type: x_bar = self.db_inverse(x_bar)
		x = tf.math.sqrt(x_bar).numpy()
		return x

class Clip(Map):
	"""
	Clip values exceeding threshold. It depends on two parameters: min, max.
	Parameters are given using self.params=[min,max].
	"""
	def map(self, x):
		"""
		Returns clipped input.

		Argument/s:
			x - value.

		Returns:
			x_bar - clipped value.
		"""
		min, max = self.params
		x_bar = tf.clip_by_value(x, min, max)
		if 'Square' in self.map_type: x_bar = tf.math.square(x_bar)
		if 'DB' in self.map_type: x_bar = self.db(x_bar)
		return x_bar

	def inverse(self, x):
		"""
		Returns input.

		Argument/s:
			x - value.

		Returns:
			x.
		"""
		if 'DB' in self.map_type: x = self.db_inverse(x)
		if 'Square' in self.map_type: x = tf.math.sqrt(x).numpy()
		return x

class DB(Map):
	"""
	Decibels map. Assumes input is power value.
	"""
	def map(self, x):
		"""
		Returns decibel value input.

		Argument/s:
			x - power value.

		Returns:
			x_bar - power value in decibels.
		"""
		return self.db(x)

	def inverse(self, x_bar):
		"""
		Inverse of decibel value.

		Argument/s:
			x_bar - power value in decibels.

		Returns:
			x - power value.
		"""
		return self.db_inverse(x_bar).numpy()

class Logistic(Map):
	"""
	Logistic map. It depends on two parameters: k, x_0. Parameters are given
	using self.params=[k,x_0].
	"""
	def map(self, x):
		"""
		Applies logistic function to input.

		Argument/s:
			x - value.

		Returns:
			f(x) - mapped value.
		"""
		k, x_0 = self.params
		if 'DB' in self.map_type: x = self.db(x)
		v_1 = tf.math.negative(tf.math.multiply(k, tf.math.subtract(x, x_0)))
		return tf.math.reciprocal(tf.math.add(self.one, tf.math.exp(v_1)))

	def inverse(self, x_bar):
		"""
		Applies inverse of logistic map.

		Argument/s:
			x_bar - mapped value.

		Returns:
			x - value.
		"""
		k, x_0 = self.params
		v_1 = tf.math.subtract(tf.math.reciprocal(x_bar), self.one)
		v_2 = tf.math.log(tf.maximum(v_1, 1e-12))
		x = tf.math.subtract(x_0, tf.math.multiply(tf.math.reciprocal(k), v_2))
		if 'DB' in self.map_type: x = self.db_inverse(x)
		return x.numpy()

class Standardise(Map):
	"""
	Convert distribution to a standard normal distribution.
	"""
	def map(self, x):
		"""
		Normalise to a standard normal distribution.

		Argument/s:
			x - random variable realisations.

		Returns:
			x_bar.
		"""
		if 'Square' in self.map_type: x =  tf.math.square(x)
		if 'DB' in self.map_type: x = self.db(x)
		x_bar = tf.math.truediv(tf.math.subtract(x, self.mu), self.sigma)
		return x_bar

	def inverse(self, x_bar):
		"""
		Inverse of normal (Gaussian) cumulative distribution function (CDF).

		Argument/s:
			x_bar - cumulative distribution function value.

		Returns:
			Inverse of CDF.
		"""
		x = tf.math.add(tf.math.multiply(x_bar, self.sigma), self.mu)
		if 'DB' in self.map_type: x = self.db_inverse(x)
		if 'Square' in self.map_type: x = tf.math.sqrt(x)
		return x.numpy()

	def stats(self, x):
		"""
		Compute stats for each frequency bin.

		Argument/s:
			x - sample.
		"""
		if 'Square' in self.map_type: x =  tf.math.square(x)
		if 'DB' in self.map_type: x = self.db(x)
		self.mu = tf.math.reduce_mean(x, axis=0)
		self.sigma = tf.math.reduce_std(x, axis=0)

class MinMaxScaling(Map):
	"""
	Normalise distribution between 0 and 1 using min-max scaling.
	"""
	def map(self, x):
		"""
		Normalise between 0 and 1.

		Argument/s:
			x - random variable realisations.

		Returns:
			x_bar.
		"""
		if 'Square' in self.map_type: x =  tf.math.square(x)
		if 'DB' in self.map_type: x = self.db(x)
		x_bar = tf.math.truediv(tf.math.subtract(x, self.min),
			tf.math.subtract(self.max, self.min))
		x_bar = tf.clip_by_value(x_bar, 0.0, 1.0)
		return x_bar

	def inverse(self, x_bar):
		"""
		Inverse of max-min scaling.

		Argument/s:
			x_bar - max-min scaled value.

		Returns:
			Inverse of x_bar.
		"""
		x = tf.math.add(tf.math.multiply(x_bar, tf.math.subtract(self.max,
			self.min)), self.min)
		if 'DB' in self.map_type: x = self.db_inverse(x)
		if 'Square' in self.map_type: x = tf.math.sqrt(x)
		return x.numpy()

	def stats(self, x):
		"""
		Compute stats for each frequency bin.

		Argument/s:
			x - sample.
		"""
		if 'Square' in self.map_type: x =  tf.math.square(x)
		if 'DB' in self.map_type: x = self.db(x)
		self.min = tf.math.reduce_min(x, axis=0)
		self.max = tf.math.reduce_max(x, axis=0)

class NormalCDF(Map):
	"""
	Normal cumulative distribution function (CDF) map.
	"""
	def map(self, x):
		"""
		Normal (Gaussian) cumulative distribution function (CDF).

		Argument/s:
			x - random variable realisations.

		Returns:
			CDF.
		"""
		if 'Square' in self.map_type: x =  tf.math.square(x)
		if 'DB' in self.map_type: x = self.db(x)
		v_1 = tf.math.subtract(x, self.mu)
		v_2 = tf.math.multiply(self.sigma, tf.math.sqrt(2.0))
		v_3 = tf.math.erf(tf.math.truediv(v_1, v_2))
		return tf.math.multiply(0.5, tf.math.add(1.0, v_3))

	def inverse(self, x_bar):
		"""
		Inverse of normal (Gaussian) cumulative distribution function (CDF).

		Argument/s:
			x_bar - cumulative distribution function value.

		Returns:
			Inverse of CDF.
		"""
		v_1 = tf.math.multiply(self.sigma, tf.math.sqrt(2.0))
		v_2 = tf.math.multiply(2.0, x_bar)
		v_3 = tf.math.erfinv(tf.math.subtract(v_2, 1))
		v_4 = tf.math.multiply(v_1, v_3)
		x = tf.math.add(v_4, self.mu)
		if 'DB' in self.map_type: x = self.db_inverse(x)
		if 'Square' in self.map_type: x = tf.math.sqrt(x)
		return x.numpy()

	def stats(self, x):
		"""
		Compute stats for each frequency bin.

		Argument/s:
			x - sample.
		"""
		if 'Square' in self.map_type: x =  tf.math.square(x)
		if 'DB' in self.map_type: x = self.db(x)
		self.mu = tf.math.reduce_mean(x, axis=0)
		self.sigma = tf.math.reduce_std(x, axis=0)

class LaplaceCDF(Map):
	"""
	Laplace cumulative distribution function (CDF) map. It depends
	on two parameters: mu and b. Parameters are given using
	self.params=[mu], and b is found from a sample of the training
	set.

	Parameter description:
		mu - location parameter.
		b - scale parameter.
	"""
	def map(self, x):
		"""
		Truncated Laplace cumulative distribution function (CDF).

		Argument/s:
			x - random variable realisations.

		Returns:
			x_bar - CDF.
		"""
		mu = self.params
		if 'DB' in self.map_type: x = self.db(x)
		x_bar = self.laplace_cdf(x, mu, self.b)
		return x_bar

	def inverse(self, x_bar):
		"""
		Inverse of truncated Laplace cumulative distribution function (CDF).

		Argument/s:
			x_bar - cumulative distribution function value.

		Returns:
			x - inverse of CDF value.
		"""
		mu = self.params
		x = self.laplace_cdf_inverse(x_bar, mu, self.b)
		if 'DB' in self.map_type: x = self.db_inverse(x)
		return x.numpy()

	def stats(self, x):
		"""
		Compute stats for each frequency bin.

		Argument/s:
			x - sample.
		"""
		mu = self.params
		if 'DB' in self.map_type: x = self.db(x)
		self.b = []
		for i in tqdm(range(x.shape[1])):
			x_k = x[:,i]
			mask = tf.math.greater(x_k, mu)
			x_k_right_tail = tf.math.subtract(tf.boolean_mask(x_k, mask), mu)
			self.b.append(tf.reduce_mean(x_k_right_tail, axis=0))
		self.b = np.array(self.b)

	def laplace_cdf(self, x, mu, b):
		"""
		Laplace cumulative distribution function (CDF).

		Argument/s:
			x - random variable realisations.
			mu - location parameter.
			b - scale parameter.

		Returns:
			CDF.
		"""
		v_1 = tf.math.subtract(x, mu)
		v_2 = tf.math.abs(v_1)
		v_3 = tf.math.negative(tf.math.truediv(v_2, b))
		v_4 = tf.math.exp(v_3)
		v_5 = tf.math.subtract(1.0, v_4)
		v_6 = tf.math.sign(v_1)
		v_7 = tf.math.multiply(0.5, tf.math.multiply(v_6, v_5))
		return tf.math.add(0.5, v_7)

	def laplace_cdf_inverse(self, cdf, mu, b):
		"""
		Inverse of Laplace cumulative distribution function (CDF).

		Argument/s:
			cdf - cumulative distribution function value.
			mu - location parameter.
			b - scale parameter.

		Returns:
			x - inverse of CDF.
		"""
		v_1 = tf.math.subtract(cdf, 0.5)
		v_2 = tf.math.abs(v_1)
		v_3 = tf.math.multiply(2.0, v_2)
		v_4 = tf.math.subtract(1.0, v_3)
		v_5 = tf.math.log(v_4)
		v_6 = tf.math.sign(v_1)
		v_7 = tf.math.multiply(b, tf.math.multiply(v_6, v_5))
		return tf.math.subtract(mu, v_7)

class TruncatedLaplaceCDF(LaplaceCDF):
	"""
	Truncated Laplace cumulative distribution function (CDF) map. It depends
	on four parameters: mu, b, lower and upper. Parameters are given using
	self.params=[mu, lower, upper], and b is found from a sample of the training
	set.

	Parameter description:
		mu - location parameter.
		b - scale parameter.
		lower - lower limit.
		upper - upper limit.
	"""

	def map(self, x):
		"""
		Truncated Laplace cumulative distribution function (CDF).

		Argument/s:
			x - random variable realisations.

		Returns:
			x_bar - CDF.
		"""
		mu, lower, upper = self.params
		if 'DB' in self.map_type: x = self.db(x)
		x_bar_lower = self.laplace_cdf(lower, mu, self.b)
		x_bar_upper = self.laplace_cdf(upper, mu, self.b)
		x_bar = self.laplace_cdf(x, mu, self.b)
		x_bar = tf.math.truediv(tf.math.subtract(x_bar, x_bar_lower),
			tf.math.subtract(x_bar_upper, x_bar_lower))
		x_bar = tf.where(tf.math.less(x, lower), tf.zeros_like(x), x_bar)
		x_bar = tf.where(tf.math.greater(x, upper), tf.ones_like(x), x_bar)
		return x_bar

	def inverse(self, x_bar):
		"""
		Inverse of truncated Laplace cumulative distribution function (CDF).

		Argument/s:
			x_bar - cumulative distribution function value.

		Returns:
			x - inverse of CDF value.
		"""
		mu, lower, upper = self.params
		x_bar_lower = self.laplace_cdf(lower, mu, self.b)
		x_bar_upper = self.laplace_cdf(upper, mu, self.b)
		x_bar = tf.math.add(tf.math.multiply(x_bar,
			tf.math.subtract(x_bar_upper, x_bar_lower)), x_bar_lower)
		x = self.laplace_cdf_inverse(x_bar, mu, self.b)
		if 'DB' in self.map_type: x = self.db_inverse(x)
		return x.numpy()

	def stats(self, x):
		"""
		Compute stats for each frequency bin.

		Argument/s:
			x - sample.
		"""
		mu, lower, upper = self.params
		if 'DB' in self.map_type: x = self.db(x)
		self.b = []
		for i in tqdm(range(x.shape[1])):
			x_k = x[:,i]
			mask = tf.math.logical_and(tf.math.greater(x_k, mu),
			 	tf.math.less(x_k, upper))
			x_k_right_tail = tf.math.subtract(tf.boolean_mask(x_k, mask), mu)
			self.b.append(tf.reduce_mean(x_k_right_tail, axis=0))
		self.b = np.array(self.b)

class UniformCDF(Map):
	"""
	Uniform cumulative distribution function (CDF) map. It depends
	on two parameters: a and b. Parameters are given using
	self.params=[a, b].

	Parameter description:
		a - lower limit.
		b - upper limit.
	"""
	def map(self, x):
		"""
		Applies uniform CDF to input.

		Argument/s:
			x - random variable realisations.

		Returns:
			x_bar - CDF.
		"""
		a, b = self.params
		return tf.math.truediv(tf.math.subtract(x, a),
			tf.math.subtract(b, a))

	def inverse(self, x_bar):
		"""
		Applies inverse of uniform CDF.

		Argument/s:
			x_bar - cumulative distribution function value.

		Returns:
			x - inverse of CDF value.
		"""
		a, b = self.params
		return tf.math.add(tf.math.multiply(x_bar,
			tf.math.subtract(b, a)), a).numpy()

# class TruncatedDoubleGammaCDF(Map):
# 	"""
# 	Truncated double gamma cumulative distribution function (CDF) map. It depends
# 	on four parameters: alpha, mu, a, and b. Parameters are given using
# 	self.params=[mu,a,b], and alpha is found from a sample of the training set.
# 	"""
#
# 	def map(self, x):
# 		"""
# 		Truncated double gamma cumulative distribution function (CDF).
#
# 		Argument/s:
# 			x - random variable realisations.
#
# 		Returns:
# 			x_bar - CDF.
# 		"""
# 		mu, a, b = self.params
# 		if 'DB' in self.map_type: x = self.db(x)
# 		x_bar_a = self.double_gamma_cdf(a, alpha, mu)
# 		x_bar_b = self.double_gamma_cdf(b, alpha, mu)
# 		x_bar = self.double_gamma_cdf(x, alpha, mu)
# 		x_bar = tf.math.truediv(tf.math.subtract(x_bar,
# 			x_bar_a), tf.math.subtract(x_bar_b, x_bar_a))
# 		x_bar = tf.where(tf.math.less(x, a), tf.zeros_like(x), x_bar)
# 		x_bar = tf.where(tf.math.greater(x, b), tf.ones_like(x), x_bar)
# 		return x_bar
#
# 	def inverse(self, x_bar):
# 		"""
# 		Inverse of truncated double gamma cumulative distribution function (CDF).
#
# 		Argument/s:
# 			x_bar - cumulative distribution function value.
#
# 		Returns:
# 			x - inverse of CDF value.
# 		"""
# 		mu, a, b = self.params
# 		x_bar_a = self.double_gamma_cdf(a, alpha, mu)
# 		x_bar_b = self.double_gamma_cdf(b, alpha, mu)
# 		x_bar = tf.math.add(tf.math.multiply(x_bar,
# 			tf.math.subtract(x_bar_b, x_bar_a)), x_bar_a)
# 		x = self.double_gamma_cdf_inverse(x_bar, alpha, mu)
# 		if 'DB' in self.map_type: x = self.db_inverse(x)
# 		return x
#
# 	def stats(self, x):
# 		"""
# 		"""
# 		mu, a, b = self.params
# 		if 'DB' in self.map_type: x = self.db(x)
# 		self.alpha = []
# 		for i in tqdm(range(x.shape[1])):
# 			x_k = x[:,i]
#
# 			mask = tf.math.logical_or(tf.math.less(x_k, mu),
# 			 	tf.math.greater(x_k, b))
#
# 			x_k_gte_1_sub_1 = tf.boolean_mask(x_k, mask)
#
#
# 			# NO SUBTRACT 1!
# 			# mu = tf.reduce_mean(x_k_gte_1_sub_1, axis=0)
# 			# sigma = tf.math.reduce_std(x_k_gte_1_sub_1, axis=0)
# 			# v_1 = tf.math.reduce_mean(tf.math.subtract(x_k_gte_1_sub_1, mu))
# 			# v_2 = tf.math.pow(v_1, 3.0)
# 			# v_3 = tf.math.pow(sigma, 3.0)
# 			# skew = tf.math.divide(v_2, v_3)
# 			alpha = 4/(skew(x_k_gte_1_sub_1)**2)
#
# 			print(alpha)
# 			# x_k_gte_1_sub_1 = [j - mu for j in x_k if (j >= mu) and (j <= b)]
# 			# self.alpha.append(4/skew(x_k_gte_1_sub_1)**2)
# 		# self.alpha = np.array(self.alpha)

	# 		self.inp_tgt.mu_xi_db = np.mean(samples_xi_db, axis=0)
	# 		self.inp_tgt.sigma_xi_db =  np.std(samples_xi_db, axis=0)

	# def double_cdf(self, x, alpha, mu):
	# 	"""
	# 	Double gamma cumulative distribution function (CDF).
	#
	# 	Argument/s:
	# 		x - random variable realisations.
	# 		alpha - shape parameter.
	# 		mu - location parameter.
	#
	# 	Returns:
	# 		CDF.
	# 	"""
	# 	v_1 = tf.math.multiply(0.5, tf.math.igamma(alpha,
	# 		tf.math.abs(tf.math.subtract(x, mu))))
	# 	cdf_minus = tf.math.subtract(0.5, v_1)
	# 	cdf_plus = tf.math.add(0.5, v_1)
	# 	return tf.where(tf.math.greater(x, mu), cdf_plus, cdf_minus)
	#
	# def double_gamma_cdf_inverse(self, cdf, alpha, mu):
	# 	"""
	# 	Inverse of double gamma cumulative distribution function (CDF).
	#
	# 	Argument/s:
	# 		cdf - cumulative distribution function value.
	# 		alpha - shape parameter.
	# 		mu - location parameter.
	#
	# 	Returns:
	# 		x - inverse of CDF.
	# 	"""
	# 	x_lower = tf.math.subtract(1.0, tf.math.multiply(2.0, cdf))
	# 	x_lower = spsp.gammaincinv(alpha, x_lower)
	# 	x_lower = tf.math.subtract(mu, x_lower)
	# 	x_upper = tf.math.subtract(tf.math.multiply(2.0, cdf), 1.0)
	# 	x_upper = spsp.gammaincinv(alpha, x_upper)
	# 	x_upper = tf.math.add(mu, x_upper)
	# 	x = tf.where(tf.math.greater(cdf, 0.5), x_upper, x_lower)
	# 	return x
	#
