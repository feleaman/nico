import numpy as np
from scipy.integrate import odeint
from scipy import signal
from scipy import stats
import scipy
import math
from nico.m_processing import *
from nico.m_fft import *


def MED_deconvolution(y, F_ini, iter):
	Nk = len(F_ini)
	Ni = len(y)
	Nj = len(y[0])
	Nl = Nk
	
	G = np.zeros(Nk)	
	for k in range(Nk):
		# print(k/Nk*100.0)
		sum = 0.0
		for i in range(Ni):
			u = 0.0
			for j in range(Nj):
				u = u + y[i][j]**2.0
			xcorr = xcorrelation_sum(y[i]**3.0, x[i], -k)
			sum = sum + (u**(-2.0))*xcorr
		G[k] = sum

	R = np.zeros((Nk, Nl))
	for k in range(Nk):
		# print(k/Nk*100.0)
		for l in range(Nl):
			sum = 0.0
			for i in range(Ni):
				u = 0.0
				V = 0.0
				for j in range(Nj):
					u = u + y[i][j]**2.0
					V = V + y[i][j]**4.0
				V = V / u**2.0
				autocorr = autocorrelation2p_sum(x[i], l, k)
				sum = sum + (V)*(u**(-1.0))*autocorr
			R[k][l] = sum




	AA = R
	bb = np.transpose(np.array([G]))
	x0 = np.transpose(np.array([F_ini]))
	# x0 = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
	error = []

	x0m = np.asmatrix(x0)
	bbm = np.asmatrix(bb)
	AAm = np.asmatrix(AA)

	rm = bbm - AAm*x0m
	error.append(np.linalg.norm(rm))

	pm = rm
	for h in range(iter):
		alfa = (np.transpose(rm)*rm) / (np.transpose(pm)*AAm*pm)
		alfa = np.asscalar(alfa)
		x0m = x0m + alfa*pm
		rm = rm - alfa*AAm*pm
		beta = (np.transpose(rm)*rm) / (np.transpose(pm)*pm)
		beta = np.asscalar(beta)
		pm = rm + beta*pm
		error.append(np.linalg.norm(bbm - AAm*x0m))
		
	# print('F_it cg')
	# print(x0m)
	plt.plot(error)
	plt.show()

	F_it = np.squeeze(np.asarray(x0m))
	# print(F_it)
	yy = []
	for i in range(Ni):
		yy.append(np.convolve(F_it, x[i], mode='same'))
	
	return yy, F_it, error

def calculate_y(x, F_it, mode):
	y_it = []
	Ni = len(x)
	for i in range(Ni):
		y_it.append(np.convolve(F_it, x[i], mode=mode)) #antes era same
	return y_it

def calculate_G(x, y_it, Nk):
	Ni = len(x)
	Nj = len(y_it[0])

	G_it = np.zeros(Nk)
	for k in range(Nk):
		sum = 0.0
		for i in range(Ni):
			u = 0.0
			for j in range(Nj):
				u = u + y_it[i][j]**2.0
			xcorr = xcorrelation_sum(y_it[i]**3.0, x[i], -k)#
			sum = sum + (u**(-2.0))*xcorr
		G_it[k] = sum
	return G_it

def calculate_R(x, y_it, Nk):
	Ni = len(x)
	# Nj = len(x[0])
	Nj = len(y_it[0])
	Nl = Nk
	
	R_it = np.zeros((Nk, Nl))
	for k in range(Nk):
		for l in range(Nl):
			sum = 0.0
			for i in range(Ni):
				u = 0.0
				V = 0.0
				for j in range(Nj):
					u = u + y_it[i][j]**2.0
					V = V + y_it[i][j]**4.0
				V = V / u**2.0
				autocorr = autocorrelation2p_sum(x[i], l, k)
				sum = sum + (V)*(u**(-1.0))*autocorr
			R_it[k][l] = sum
	return R_it