## FILE:           mmse_est.py
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
## BRIEF:          Speech enhancement based on MMSE estimators.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from scipy.special import gamma as Gamma
from scipy.special import hyp1f1



## TMP IMPORT
import matplotlib.pyplot as plt
import sys


def Phi(alpha, gamma, zeta): return np.minimum(hyp1f1(alpha, gamma, zeta), 1e8) # hyp1f1(alpha, gamma, zeta) # np.exp(zeta)*hyp1f1(gamma - alpha, gamma, -zeta)
def Psi(alpha, gamma, zeta):
	

	# zeta = np.arange(30, 32, 0.01)
	# # print(G_minus)
	# tmp1 = (Gamma(1.0 - gamma)/Gamma(alpha - gamma + 1.0))*Phi(alpha, gamma, zeta) + \
	# 	((Gamma(gamma - 1.0)/Gamma(alpha))*(zeta**(1.0 - gamma))*Phi(alpha - gamma + 1.0, 2.0 - gamma, zeta))
	
	# print(tmp1)
	# fig = plt.figure()
	# ax = fig.add_subplot(311)
	# ax.plot(zeta, tmp1, linewidth=0.5)

	# tmp2 = (Gamma(1.0 - gamma)/Gamma(alpha - gamma + 1.0))*Phi(alpha, gamma, zeta)
	# ax = fig.add_subplot(312)
	# ax.plot(zeta, tmp2, linewidth=0.5)
	
	# tmp3 = ((Gamma(gamma - 1.0)/Gamma(alpha))*(zeta**(1.0 - gamma))*Phi(alpha - gamma + 1.0, 2.0 - gamma, zeta))
	# ax = fig.add_subplot(313)
	# ax.plot(zeta, tmp3, linewidth=0.5)
	# # ax = fig.add_subplot(212)
	# # ax.plot(G_minus**2.0, linewidth=0.5)
	# ax.set_aspect(aspect='auto')
	# plt.show()


	# print(tmp1[60], tmp2[60], tmp3[60], tmp2[60] + tmp3[60])

	# sys.exit()



	return (Gamma(1.0 - gamma)/Gamma(alpha - gamma + 1.0))*Phi(alpha, gamma, zeta) + \
		((Gamma(gamma - 1.0)/Gamma(alpha))*(zeta**(1.0 - gamma))*Phi(alpha - gamma + 1.0, 2.0 - gamma, zeta))

def estimator(X, sigma_d, sigma_s):
	xi = sigma_s**2.0/sigma_d**2.0

	G_plus = ((3.0**(0.5))/(2.0*(2.0**0.5)*(xi**0.5))) + (X/sigma_d)
	G_minus = ((3.0**(0.5))/(2.0*(2.0**0.5)*(xi**0.5))) - (X/sigma_d)


	v1 = (1.5**0.25)/(2.0*((2.0*sigma_d*sigma_s)**0.5))


	v2 = (np.exp(-(X**2.0/sigma_d**2.0))/(np.pi**0.5))*(Psi(0.25, 0.5, G_minus**2.0) + Psi(0.25, 0.5, G_plus**2.0))
	v3 = ((2.0*(np.abs(G_minus) - G_minus))/Gamma(0.25))*np.exp((0.375*(sigma_d**2.0/sigma_s**2.0)) - ((1.5**0.5)*(X/sigma_s)))*Phi(0.75, 1.5, -G_minus**2.0)
	v4 = ((2.0*(np.abs(G_plus) - G_plus))/Gamma(0.25))*np.exp((0.375*(sigma_d**2.0/sigma_s**2.0)) + ((1.5**0.5)*(X/sigma_s)))*Phi(0.75,1.5,-G_plus**2.0)
	p_X = v1*(v2 + v3 + v4)


	v5 = ((1.5**0.25)/(8.0*(2.0**0.5)*p_X))*((sigma_d/sigma_s)**0.5)
	v6 = (np.exp(-(X**2.0/sigma_d**2.0))/(np.pi**0.5))*(Psi(0.75, 0.5, G_minus**2.0) - Psi(0.75, 0.5, G_plus**2.0))
	v7 = ((2.0*(np.abs(G_minus) - G_minus))/Gamma(0.75))*np.exp((0.375*(sigma_d**2.0/sigma_s**2.0)) - ((1.5**0.5)*(X/sigma_s)))*Phi(0.25, 1.5, -G_minus**2.0)
	v8 = ((2.0*(np.abs(G_plus) - G_plus))/Gamma(0.75))*np.exp((0.375*(sigma_d**2.0/sigma_s**2.0)) + ((1.5**0.5)*(X/sigma_s)))*Phi(0.25, 1.5, -G_plus**2.0)
	

	# tmp2 = np.arange(0, 100, 0.1)
	# # print(G_minus)
	tmp = Psi(0.75, 0.5, G_minus**2.0) - Psi(0.75, 0.5, G_plus**2.0)
	tmp1 = Psi(0.75, 0.5, G_minus**2.0)
	tmp2 = Psi(0.75, 0.5, G_plus**2.0)

	# print(tmp)
	fig = plt.figure()
	ax = fig.add_subplot(311)
	ax.plot(X, tmp, linewidth=0.5)
	ax = fig.add_subplot(312)
	ax.plot(X, tmp1, linewidth=0.5)	
	ax = fig.add_subplot(313)
	ax.plot(X, tmp2, linewidth=0.5)
	# # ax = fig.add_subplot(212)
	# # ax.plot(G_minus**2.0, linewidth=0.5)
	ax.set_aspect(aspect='auto')
	plt.show()

	sys.exit()


	return v5*(v6 + v7 - v8)



# print(likelihood(X, xi, eta))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(likelihood(X, xi, eta), linewidth=0.5)
# ax.set_aspect(aspect='auto')
# plt.show()


# def Phi(alpha, gamma, zeta): return np.exp(zeta)*hyp1f1(gamma - alpha, gamma, -zeta) # hyp1f1(alpha, gamma, zeta)
# def Psi(alpha, gamma, zeta):
# 	return (Gamma(1.0 - gamma)/Gamma(alpha - gamma + 1.0))*Phi(alpha, gamma, zeta) + \
# 		(Gamma(gamma - 1.0)/Gamma(alpha))*(zeta**(1.0 - gamma))*Phi(alpha - gamma + 1.0, 2.0 - gamma, zeta)

# def G_plus(xi): return ((3.0**(0.5))/(2.0*(2.0**0.5)*(xi**0.5))) + ((xi + 1.0)**0.5)
# def G_minus(xi): return ((3.0**(0.5))/(2.0*(2.0**0.5)*(xi**0.5))) - ((xi + 1.0)**0.5)

# def likelihood(X, xi, eta):
# 	iota = np.maximum(((X**2.0)*(xi**0.5))/(xi + 1.0), 1e-12)
# 	v1 = (1.5**0.25)/(2.0*((2.0*iota)**0.5))
# 	v2 = (np.exp(-(xi + 1.0))/(np.pi**0.5))*(Psi(0.25, 0.5, G_minus(xi)**2.0) + Psi(0.25, 0.5, G_plus(xi)**2.0))
# 	v3 = ((2.0*(np.abs(G_minus(xi)) - G_minus(xi)))/Gamma(0.25))*np.exp((0.375/xi) - ((1.5**0.5)*eta))*Phi(0.75, 1.5, -G_minus(xi)**2.0)
# 	v4 = ((2.0*(np.abs(G_plus(xi)) - G_plus(xi)))/Gamma(0.25))*np.exp((0.375/xi) + ((1.5**0.5)*eta))*Phi(0.75,1.5,-G_plus(xi)**2.0)
# 	return v1*(v2 + v3 + v4)

# def estimator(X, xi):
# 	xi = np.maximum(xi, 1e-12)
# 	eta = 1.0 + (1.0/xi**0.5)
# 	v1 = ((1.5**0.25)/(8.0*(2**0.5)*likelihood(X, xi, eta)))*(1/xi**0.25)
# 	v2 = (np.exp(-(xi + 1.0))/(np.pi**0.5))*(Psi(0.75, 0.5, G_minus(xi)**2.0) + Psi(0.75, 0.5, G_plus(xi)**2.0))
# 	v3 = ((2.0*(np.abs(G_minus(xi)) - G_minus(xi)))/Gamma(0.75))*np.exp((0.375/xi) - ((1.5**0.5)*eta))*Phi(0.25, 1.5, -G_minus(xi)**2.0)
# 	v4 = ((2.0*(np.abs(G_plus(xi)) - G_plus(xi)))/Gamma(0.75))*np.exp((0.375/xi) + ((1.5**0.5)*eta))*Phi(0.75, 1.5, -G_plus(xi)**2.0)
# 	return v1*(v2 + v3 - v4)