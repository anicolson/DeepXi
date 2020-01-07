## FILE:           xi.py
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
## BRIEF:          Functions for computing a priori SNR.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
import scipy.special as spsp
import tensorflow as tf

def log10(x):
  numerator = tf.log(x)
  denominator = tf.constant(np.log(10), dtype=numerator.dtype)
  return tf.truediv(numerator, denominator)

def xi(s_MAG, d_MAG):
	return tf.truediv(tf.square(s_MAG), tf.maximum(tf.square(d_MAG), 1e-12)) # a priori SNR.

def xi_dB(s_MAG, d_MAG):
	return tf.multiply(10.0, log10(tf.maximum(xi(s_MAG, d_MAG), 1e-12)))

def xi_bar(s_MAG, d_MAG, mu, sigma):
	return tf.multiply(0.5, tf.add(1.0, tf.erf(tf.truediv(tf.subtract(xi_dB(s_MAG, d_MAG), mu), 
		tf.multiply(sigma, tf.sqrt(2.0))))))

def xi_hat(xi_bar_hat, mu, sigma):
	xi_dB_hat = np.add(np.multiply(np.multiply(sigma, np.sqrt(2.0)), 
		spsp.erfinv(np.subtract(np.multiply(2.0, xi_bar_hat), 1))), mu)	
	return np.power(10.0, np.divide(xi_dB_hat, 10.0))
