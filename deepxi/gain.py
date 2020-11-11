## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from scipy.special import exp1, i0, i1
import math
import numpy as np
import tensorflow as tf

def mmse_stsa(xi, gamma):
	"""
	Computes the MMSE-STSA gain function.

	Numpy version:
		nu = np.multiply(xi, np.divide(gamma, np.add(1, xi)))
		G = np.multiply(np.multiply(np.multiply(np.divide(np.sqrt(np.pi), 2),
			np.divide(np.sqrt(nu), gamma)), np.exp(np.divide(-nu,2))),
			np.add(np.multiply(np.add(1, nu), i0(np.divide(nu,2))),
			np.multiply(nu, i1(np.divide(nu, 2))))) # MMSE-STSA gain function.
		idx = np.isnan(G) | np.isinf(G) # replace by Wiener gain.
		G[idx] = np.divide(xi[idx], np.add(1, xi[idx])) # Wiener gain.
		return G

	Argument/s:
		xi - a priori SNR.
		gamma - a posteriori SNR.

	Returns:
		G - MMSE-STSA gain function.
	"""
	pi = tf.constant(math.pi)
	xi = tf.maximum(xi, 1e-12)
	gamma = tf.maximum(gamma, 1e-12)
	nu = tf.math.multiply(xi, tf.math.truediv(gamma, tf.math.add(1.0, xi)))
	G = tf.math.multiply(tf.math.multiply(tf.math.multiply(tf.math.truediv(tf.math.sqrt(pi), 2.0),
		tf.math.truediv(tf.math.sqrt(nu), gamma)), tf.math.exp(tf.math.truediv(-nu, 2.0))),
		tf.math.add(tf.math.multiply(tf.math.add(1.0, nu), tf.math.bessel_i0(tf.math.truediv(nu, 2.0))),
		tf.math.multiply(nu, tf.math.bessel_i1(tf.math.truediv(nu, 2.0))))) # MMSE-STSA gain function.
	G_WF = wf(xi)
	logical = tf.math.logical_or(tf.math.is_nan(G), tf.math.is_inf(G))
	G = tf.where(logical, G_WF, G)
	return G

def mmse_lsa(xi, gamma):
	"""
	Computes the MMSE-LSA gain function.

	Numpy version:
		v_1 = np.divide(xi, np.add(1.0, xi))
		nu = np.multiply(v_1, gamma)
		return np.multiply(v_1, np.exp(np.multiply(0.5, exp1(nu)))) # MMSE-LSA gain function.

	Argument/s:
		xi - a priori SNR.
		gamma - a posteriori SNR.

	Returns:
		MMSE-LSA gain function.
	"""
	xi = tf.maximum(xi, 1e-12)
	gamma = tf.maximum(gamma, 1e-12)
	v_1 = tf.math.truediv(xi, tf.math.add(1.0, xi))
	nu = tf.math.multiply(v_1, gamma)
	v_2 = exp1(nu)
	# v_2 = tf.math.negative(tf.math.special.expint(tf.math.negative(nu))) # E_1(x) = -E_i(-x)
	return tf.math.multiply(v_1, tf.math.exp(tf.math.multiply(0.5, v_2))) # MMSE-LSA gain function.

def wf(xi):
	"""
	Computes the Wiener filter (WF) gain function.

	Argument/s:
		xi - a priori SNR.

	Returns:
		WF gain function.
	"""
	return tf.math.truediv(xi, tf.math.add(xi, 1.0)) # WF gain function.

def srwf(xi):
	"""
	Computes the square-root Wiener filter (WF) gain function.

	Argument/s:
		xi - a priori SNR.

	Returns:
		SRWF gain function.
	"""
	return tf.math.sqrt(wf(xi)) # SRWF gain function.

def cwf(xi):
	"""
	Computes the constrained Wiener filter (WF) gain function.

	Argument/s:
		xi - a priori SNR.

	Returns:
		cWF gain function.
	"""
	return wf(np.sqrt(xi)) # cWF gain function.

def dgwf(xi, cdm):
	"""
	Computes the dual-gain Wiener filter (WF).

	Argument/s:
		xi - a priori SNR.
		cdm - constructive-deconstructive mask.

	Returns:
		G - DGWF.
	"""
	v_1 = np.divide(2.0, np.pi)
	v_2 = np.multiply(2, v_1)
	v_3 = np.sqrt(xi)
	v_4 = np.add(xi, 1.0)
	G_minus = np.divide(np.subtract(xi, np.multiply(v_1, v_3)),
		np.subtract(v_4, np.multiply(v_2, v_3)))
	G_plus = np.divide(np.add(xi, np.multiply(v_1, v_3)),
		np.add(v_4, np.multiply(v_2, v_3)))
	G = np.where(cdm, G_plus, G_minus)
	return G # DGWF.

def irm(xi):
	"""
	Computes the ideal ratio mask (IRM).

	Argument/s:
		xi - a priori SNR.

	Returns:
		IRM.
	"""
	return srwf(xi) # IRM.

def ibm(xi):
	"""
	Computes the ideal binary mask (IBM) with a threshold of 0 dB.

	Argument/s:
		xi - a priori SNR.

	Returns:
		IBM.
	"""
	return tf.cast(tf.math.greater(xi, 1.0), tf.float32) # IBM (1 corresponds to 0 dB).


def deepmmse(xi, gamma):
	"""
	DeepMMSE utilises the MMSE noise periodogram estimate gain function.

	Argument/s:
		xi - a priori SNR.
		gamma - a posteriori SNR.

	Returns:
		MMSE-Noise_PSD gain function.
	"""
	return np.add(np.divide(1, np.add(1, xi)),
		np.divide(xi, np.multiply(gamma, np.add(1, xi)))) # MMSE noise periodogram estimate gain function.

def gfunc(xi, gamma=None, gtype=None, cdm=None):
	"""
	Computes the selected gain function.

	Argument/s:
		xi - a priori SNR.
		gamma - a posteriori SNR.
		gtype - gain function type.
		cdm - constructive-deconstructive mask.

	Returns:
		G - gain function.
	"""
	if gtype == 'mmse-lsa': G = mmse_lsa(xi, gamma)
	elif gtype == 'mmse-stsa':  G = mmse_stsa(xi, gamma)
	elif gtype == 'wf': G = wf(xi)
	elif gtype == 'srwf': G = srwf(xi)
	elif gtype == 'cwf': G = cwf(xi)
	elif gtype == 'dgwf': G = dgwf(xi, cdm)
	elif gtype == 'irm': G = irm(xi)
	elif gtype == 'ibm': G = ibm(xi)
	elif gtype == 'deepmmse': G = deepmmse(xi, gamma)
	else: raise ValueError('Invalid gain function type.')
	return G
