## FILE:           gain.py
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
## BRIEF:          Gain functions and masks for speech enhancement.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from scipy.special import exp1, i0, i1

def mmse_stsa(xi, gamma):
	'''
	Computes the MMSE-STSA gain function.
		
	Input/s: 
		xi - a priori SNR.
		gamma - a posteriori SNR.
		
	Output/s: 
		G - MMSE-STSA gain function.
	'''
	nu = np.multiply(xi, np.divide(gamma, np.add(1, xi)))
	G = np.multiply(np.multiply(np.multiply(np.divide(np.sqrt(np.pi), 2), 
		np.divide(np.sqrt(nu), gamma)), np.exp(np.divide(-nu,2))), 
		np.add(np.multiply(np.add(1, nu), i0(np.divide(nu,2))), 
		np.multiply(nu, i1(np.divide(nu, 2))))) # MMSE-STSA gain function.
	idx = np.isnan(G) | np.isinf(G) # replace by Wiener gain.
	G[idx] = np.divide(xi[idx], np.add(1, xi[idx])) # Wiener gain.
	return G

def mmse_lsa(xi, gamma):
	'''
	Computes the MMSE-LSA gain function.
		
	Input/s: 
		xi - a priori SNR.
		gamma - a posteriori SNR.
		
	Output/s: 
		MMSE-LSA gain function.
	'''
	nu = np.multiply(np.divide(xi, np.add(1, xi)), gamma)
	return np.multiply(np.divide(xi, np.add(1, xi)), np.exp(np.multiply(0.5, exp1(nu)))) # MMSE-LSA gain function.

def wf(xi):
	'''
	Computes the Wiener filter (WF) gain function.
		
	Input/s: 
		xi - a priori SNR.
		
	Output/s: 
		WF gain function.
	'''
	return np.divide(xi, np.add(xi, 1.0)) # WF gain function.

def srwf(xi):
	'''
	Computes the square-root Wiener filter (WF) gain function.
		
	Input/s: 
		xi - a priori SNR.
		
	Output/s: 
		SRWF gain function.
	'''
	return np.sqrt(wf(xi)) # SRWF gain function.

def cwf(xi):
	'''
	Computes the constrained Wiener filter (WF) gain function.
		
	Input/s: 
		xi - a priori SNR.
		
	Output/s: 
		cWF gain function.
	'''
	return wf(np.sqrt(xi)) # cWF gain function.

def irm(xi):
	'''
	Computes the ideal ratio mask (IRM).
		
	Input/s: 
		xi - a priori SNR.
		
	Output/s: 
		IRM.
	'''
	return srwf(xi) # IRM.


def ibm(xi):
	'''
	Computes the ideal binary mask (IBM) with a threshold of 0 dB.
		
	Input/s: 
		xi - a priori SNR.
		
	Output/s: 
		IBM.
	'''
	return np.greater(self.xi_hat_ph, 1, dtype=np.float32) # IBM (1 corresponds to 0 dB).

def gfunc(xi, gamma=None, gtype='mmse-lsa'):
	'''
	Computes the selected gain function.
		
	Input/s: 
		xi - a priori SNR.
		gamma - a posteriori SNR.
		gtype - gain function type.
		
	Output/s: 
		G - gain function.
	'''
	if gtype == 'mmse-lsa': G = mmse_lsa(xi, gamma)
	elif gtype == 'mmse-stsa':  G = mmse_stsa(xi, gamma)
	elif gtype == 'wf': G = wf(xi)
	elif gtype == 'srwf': G = srwf(xi)
	elif gtype == 'cwf': G = cwf(xi)
	elif gtype == 'irm': G = irm(xi)
	elif gtype == 'ibm': G = ibm(xi)
	else: ValueError('Gain function not available.')
	return G