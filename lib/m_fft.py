import numpy as np
from scipy.integrate import odeint
from scipy import signal
from scipy import stats
import scipy
import math
import time


def mag_fft(x, fs):
	fftx = np.fft.fft(x)
	fftx = np.abs(fftx)/len(fftx)
	fftx = 2*fftx[0:int(len(fftx)/2)]
	tr = len(x)/fs
	df = 1.0/tr
	f = np.array([i*df for i in range(len(fftx))])
	magX = fftx
	return magX, f, df



def shortFFT(x, fs, segments, window, mode):
	nperseg = len(x)/segments
	f, t, stftX = signal.spectrogram(x, fs, nperseg=nperseg, window=window, mode=mode)
	stftX = stftX/nperseg
	df = f[2] - f[1]
	return stftX, f, df, t

def shortPSD(x, fs, segments):
	n_x = len(x)
	n_segments = int(n_x/segments)
	stpsdX = []
	for i in range(int(segments)):
		signal_segment = x[i*n_segments : (i+1)*n_segments]
		f_stpsd, psd_segment = signal.periodogram(signal_segment, fs, return_onesided=True, scaling='density')
		stpsdX.append(psd_segment)
	
	t_stpsd = [i/fs for i in range(n_segments)]
	# stpsdX = np.array([stpsdX])
	# stpsdX.transpose()
	t_stpsd = np.linspace(0.0, n_x/fs, num=segments)
	# print(len(stpsdX[1]))
	# print(len(f_stpsd))
	# print(len(t_stpsd))
	# print(stpsdX.shape)
	stpsdX = list(map(list, zip(*stpsdX)))
	# print(stpsdX[1])
	
	# print(stpsdX.shape)
	df = f_stpsd[2] - f_stpsd[1]
	return stpsdX, f_stpsd, df, t_stpsd	

def wv_dis(x, fs):

	dt = 1.0/fs
	n = len(x)
	t = np.array([i*dt for i in range(n)])

	dtau = dt
	tau = np.array([i*dtau for i in range(2*n)])

	magY = []

	y = np.zeros(len(tau))

	for i in range(len(t)):
		print(i/n*100.0)
		for k in range(len(tau)):
			if i+k/2.0 < len(x)-1:
				frac, whole = math.modf(k/2.0)
				if frac == 0.0:
					y_pos = x[i+k/2]
				elif frac == 0.5:
					y_pos = (x[i+k/2+0.5] + x[i+k/2-0.5])/(2.0)
				else:
					print('error index 123')
			else:
				y_pos = 0.0
			
			if i-k/2.0 > 0:
				frac, whole = math.modf(k/2.0)
				if frac == 0.0:
					y_neg = x[i-k/2]
				elif frac == 0.5:
					y_neg = (x[i-k/2+0.5] + x[i-k/2-0.5])/(2.0)
					if i-k/2-0.5 < 0:
						print('warning 975')
				else:
					print('error index 124')
			else:
				y_neg = 0.0
			y[k] = y_pos*y_neg

		fft, f, df = mag_fft(y, fs)
		magY.append(np.array(fft))
	magY = list(map(list, zip(*magY)))

	return magY, f, df

def scd_from_x(x, fs):
	scdX_fa = []
	wvdX, f, df = wv_dis(x, fs)

	for k in range(len(f)):
		scd, a, da = mag_fft(wvdX[k], fs)
		scdX_fa.append(np.array(scd))
	return scdX_fa, a, da, f, df

def scd_from_wvd(wvdX, f, df, fs):
	scdX_fa = []

	for k in range(len(f)):
		scd, a, da = mag_fft(wvdX[k], fs)
		scdX_fa.append(np.array(scd))
	return scdX_fa, a, da


def wv_dis_opt(x, fs):
	
	dt = 1.0/fs
	n = len(x)
	t = np.array([i*dt for i in range(n)])

	dtau = dt #independizar tau y t, mediante interpolaciones
	tau = np.array([i*dtau for i in range(2*n)])

	magY = []

	y = np.zeros(len(tau))
	start_time = time.time()
	
	for i in range(len(t)):
		print(i/n*100)
		cont = 0
		for k in range(len(tau)):
			cont, whole = math.modf(k/2)
			ipk2 = i+k/2
			imk2 = i-k/2

			if ipk2 < len(x)-1:
				#frac, whole = math.modf(k/2.0)
				if cont == 0:
					y_pos = x[ipk2]
				else:
					y_pos = (x[ipk2+0.5] + x[ipk2-0.5])/(2)
			else:
				y_pos = 0
			
			if imk2 > 0:
				#frac, whole = math.modf(k/2.0)
				if cont == 0:
					y_neg = x[imk2]
				else:
					y_neg = (x[imk2+0.5] + x[imk2-0.5])/(2)
			else:
				y_neg = 0
			y[k] = y_pos*y_neg
			

		fft, f, df = mag_fft(y, fs)
		magY.append(np.array(fft))
		
		time_iter = time.time() - start_time
		print(time_iter)
	
	magY = list(map(list, zip(*magY)))

	return magY, f, df

def wv_dis_opt_2(x, fs):

	dt = 1.0/fs
	n = len(x)
	t = np.array([i*dt for i in range(n)])

	dtau = dt #independizar tau y t, mediante interpolaciones
	tau = np.array([i*dtau for i in range(2*n)])

	magY = []

	y = np.zeros(len(tau))

	for i in range(len(t)):
		print(i/n*100)
		#cont = 0
		for k in range(len(tau)):
			cont, whole = math.modf(k/2)
			ipk2 = i+k/2
			imk2 = i-k/2

			if ipk2 < len(x)-1:
				#frac, whole = math.modf(k/2.0)
				if cont == 0:
					y_pos = x[ipk2]
				else:
					y_pos = (x[ipk2+0.5] + x[ipk2-0.5])/(2)
			else:
				y_pos = 0
			
			if imk2 > 0:
				#frac, whole = math.modf(k/2.0)
				if cont == 0:
					y_neg = x[imk2]
				else:
					y_neg = (x[imk2+0.5] + x[imk2-0.5])/(2)
			else:
				y_neg = 0
			y[k] = y_pos*y_neg
			

		fft, f, df = mag_fft(y, fs)
		magY.append(np.array(fft))
	magY = list(map(list, zip(*magY)))

	return magY, f, df

def inv_fft_1(magX, df):
	ifftx = np.fft.ifft(magX)
	n = len(ifftx)
	ifftx = ifftx*n
	# fftx = np.abs(fftx)/len(fftx)
	# ifftx = 2*fftx[0:len(fftx)/2]
	dt = 1.0/(n*df)
	t = np.array([i*dt for i in range(len(ifftx))])
	return ifftx, t, dt

def inv_fft_2(magX, phaX, df):
	ifftx = np.fft.ifft(magX*np.cos(phaX))
	n = len(ifftx)
	ifftx = ifftx*n
	# fftx = np.abs(fftx)/len(fftx)
	# ifftx = 2*fftx[0:len(fftx)/2]
	dt = 1.0/(n*df)
	t = np.array([i*dt for i in range(len(ifftx))])
	return ifftx, t, dt

def inv_fft_3(magX, phaX, df):
	ifftx = np.fft.ifft(magX*np.exp(1j*phaX))
	n = len(ifftx)
	ifftx = ifftx*n
	# fftx = np.abs(fftx)/len(fftx)
	# ifftx = 2*fftx[0:len(fftx)/2]
	dt = 1.0/(n*df)
	t = np.array([i*dt for i in range(len(ifftx))])
	return ifftx, t, dt

def cepstrum_complex(x, fs):
	magX, f, df = mag_fft(x, fs)
	phaX = np.angle(np.fft.fft(x))
	phaX = phaX[0:len(phaX)/2]
	phaX = np.unwrap(phaX)
	
	ifftx = np.fft.ifft(np.log(magX)+1j*phaX)
	n = len(ifftx)
	ifftx = ifftx*n
	# fftx = np.abs(fftx)/len(fftx)
	# ifftx = 2*fftx[0:len(fftx)/2]
	dt = 1.0/(n*df)
	t = np.array([i*dt for i in range(len(ifftx))])
	return ifftx, t, dt

def cepstrum_real(x, fs):
	magX, f, df = mag_fft(x, fs)
	ifftx = np.fft.ifft(np.log(magX))
	n = len(ifftx)
	ifftx = ifftx*n
	# fftx = np.abs(fftx)/len(fftx)
	# ifftx = 2*fftx[0:len(fftx)/2]
	dt = 1.0/(n*df)
	t = np.array([i*dt for i in range(len(ifftx))])
	return ifftx, t, dt

def cepstrum_real_2(magX, df):
	ifftx = np.fft.ifft(np.log(magX))
	n = len(ifftx)
	ifftx = ifftx*n
	# fftx = np.abs(fftx)/len(fftx)
	# ifftx = 2*fftx[0:len(fftx)/2]
	dt = 1.0/(n*df)
	t = np.array([i*dt for i in range(len(ifftx))])
	return ifftx, t, dt

def cepstrum_complex_2(magX, phaX, df):
	ifftx = np.fft.ifft(np.log(magX)+1j*phaX)
	n = len(ifftx)
	ifftx = ifftx*n
	# fftx = np.abs(fftx)/len(fftx)
	# ifftx = 2*fftx[0:len(fftx)/2]
	dt = 1.0/(n*df)
	t = np.array([i*dt for i in range(len(ifftx))])
	return ifftx, t, dt


# def SCD(x, fs):

	# dt = 1.0/fs
	# n = len(x)
	# t = np.array([i*dt for i in range(n)])

	# dtau = 1.0
	# n_tau = 1000
	# tau_array = np.array([i*dtau for i in range(n_tau)])
	
	
	# dalfa = 1.0
	# n_alfa = 1000
	# alfa_array = np.array([i*dalfa for i in range(n_alfa)])
	
	
	# alfa = 0
	
	# y = x[i-]
	
	
	
	
	
	

	# magY = []

	# y = np.zeros(len(tau))

	# for i in range(len(t)):
		# print(i/n*100.0)
		# for k in range(len(tau)):
			# if i+k/2.0 < len(x)-1:
				# frac, whole = math.modf(k/2.0)
				# if frac == 0.0:
					# y_pos = x[i+k/2]
				# elif frac == 0.5:
					# y_pos = (x[i+k/2+0.5] + x[i+k/2-0.5])/(2.0)
				# else:
					# print('error index 123')
			# else:
				# y_pos = 0.0
			
			# if i-k/2.0 > 0:
				# frac, whole = math.modf(k/2.0)
				# if frac == 0.0:
					# y_neg = x[i-k/2]
				# elif frac == 0.5:
					# y_neg = (x[i-k/2+0.5] + x[i-k/2-0.5])/(2.0)
					# if i-k/2-0.5 < 0:
						# print('warning 975')
				# else:
					# print('error index 124')
			# else:
				# y_neg = 0.0
			# y[k] = y_pos*y_neg

		# fft, f, df = mag_fft(y, fs)
		# magY.append(np.array(fft))
	# magY = list(map(list, zip(*magY)))

	# return magY, f, df