## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from scipy.special import exp1, i0, i1

def mmse_stsa(xi, gamma):
	"""
	Computes the MMSE-STSA gain function.

	Argument/s:
		xi - a priori SNR.
		gamma - a posteriori SNR.

	Returns:
		G - MMSE-STSA gain function.
	"""
	nu = np.multiply(xi, np.divide(gamma, np.add(1, xi)))
	G = np.multiply(np.multiply(np.multiply(np.divide(np.sqrt(np.pi), 2),
		np.divide(np.sqrt(nu), gamma)), np.exp(np.divide(-nu,2))),
		np.add(np.multiply(np.add(1, nu), i0(np.divide(nu,2))),
		np.multiply(nu, i1(np.divide(nu, 2))))) # MMSE-STSA gain function.
	idx = np.isnan(G) | np.isinf(G) # replace by Wiener gain.
	G[idx] = np.divide(xi[idx], np.add(1, xi[idx])) # Wiener gain.
	return G

def mmse_lsa(xi, gamma):
	"""
	Computes the MMSE-LSA gain function.

	Argument/s:
		xi - a priori SNR.
		gamma - a posteriori SNR.

	Returns:
		MMSE-LSA gain function.
	"""
	nu = np.multiply(np.divide(xi, np.add(1, xi)), gamma)
	return np.multiply(np.divide(xi, np.add(1, xi)), np.exp(np.multiply(0.5, exp1(nu)))) # MMSE-LSA gain function.

def wf(xi):
	"""
	Computes the Wiener filter (WF) gain function.

	Argument/s:
		xi - a priori SNR.

	Returns:
		WF gain function.
	"""
	return np.divide(xi, np.add(xi, 1.0)) # WF gain function.

def srwf(xi):
	"""
	Computes the square-root Wiener filter (WF) gain function.

	Argument/s:
		xi - a priori SNR.

	Returns:
		SRWF gain function.
	"""
	return np.sqrt(wf(xi)) # SRWF gain function.

def cwf(xi):
	"""
	Computes the constrained Wiener filter (WF) gain function.

	Argument/s:
		xi - a priori SNR.

	Returns:
		cWF gain function.
	"""
	return wf(np.sqrt(xi)) # cWF gain function.

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
	return np.greater(xi, 1, dtype=np.float32) # IBM (1 corresponds to 0 dB).


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

def gfunc(xi, gamma=None, gtype='mmse-lsa'):
	"""
	Computes the selected gain function.

	Argument/s:
		xi - a priori SNR.
		gamma - a posteriori SNR.
		gtype - gain function type.

	Returns:
		G - gain function.
	"""
	if gtype == 'mmse-lsa': G = mmse_lsa(xi, gamma)
	elif gtype == 'mmse-stsa':  G = mmse_stsa(xi, gamma)
	elif gtype == 'wf': G = wf(xi)
	elif gtype == 'srwf': G = srwf(xi)
	elif gtype == 'cwf': G = cwf(xi)
	elif gtype == 'irm': G = irm(xi)
	elif gtype == 'ibm': G = ibm(xi)
	elif gtype == 'deepmmse': G = deepmmse(xi, gamma)
	else: raise ValueError('Invalid gain function type.')
	return G
