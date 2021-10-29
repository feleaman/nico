import numpy as np
from scipy.integrate import odeint
from scipy import signal
from scipy import stats
import scipy
import math
import time
import sys
sys.path.insert(0, './lib') #to open user-defined functions
from m_denois import *
# from m_denois import butter_bandpass
# from m_denois import butter_lowpass
import matplotlib.pyplot as plt
from m_demodulation import *
from matplotlib import colors
from scipy import interpolate

def energy_in_band(magX, df, low, high):
	
	index_low = int(low/df)
	index_high = int(high/df)
	index_len = index_high - index_low
	energy = 0.
	for i in range(index_len):
		energy += (magX[i+index_low])**2.0
	return energy

def sum_in_band(magX, df, low, high):
	index_low = int(low/df)
	index_high = int(high/df)
	index_len = index_high - index_low
	sum = 0.
	for i in range(index_len):
		sum += (magX[i+index_low])
	return sum

def avg_in_band(magX, df, low, high):
	index_low = int(low/df)
	index_high = int(high/df)
	index_len = index_high - index_low
	sum = 0.
	for i in range(index_len):
		sum += (magX[i+index_low])

	return sum/index_len

# def mpr_single_comp(magX, df, low, high):
	# index_low = int(low/df)
	# index_high = int(high/df)
	# index_len = index_high - index_low
	# sum = 0.
	# for i in range(index_len):
		# sum += (magX[i+index_low])
	# return sum
def std_in_band(magX, df, low, high):
	avg = avg_in_band(magX, df, low, high)
	index_low = int(low/df)
	index_high = int(high/df)
	index_len = index_high - index_low
	sum = 0.
	count = 0
	for i in range(index_len):
		sum += (magX[i+index_low] - avg)**2.0
		count += 1
	sum = (sum / count)**0.5
	return sum

def rms_in_band(magX, df, low, high):
	index_low = int(low/df)
	index_high = int(high/df)
	index_len = index_high - index_low
	sum = 0.
	count = 0
	for i in range(index_len):
		sum += (magX[i+index_low])**2.0
		count += 1
	sum = (sum / count)**0.5
	return sum

def mag_fft(x, fs):
	fftx = np.fft.fft(x)
	fftx = np.abs(fftx)/len(fftx)
	fftx = fftx[0:int(len(fftx)/2)]
	fftx[1:] = 2*fftx[1:]
	tr = len(x)/fs
	df = 1.0/tr
	f = np.array([i*df for i in range(len(fftx))])
	magX = fftx
	return magX, f, df

def mag_fft_test(x, fs):
	fftx = np.fft.fft(x)
	fftx = np.abs(fftx)/len(fftx)
	fftx = fftx[0:int(len(fftx)/2)]
	fftx[1:] = 2*fftx[1:]
	tr = len(x)/fs
	df = 1.0/tr
	f = np.array([i*df for i in range(len(fftx))])
	magX = fftx
	return magX, f, df

def mag_fft_hanning(x, fs):
	window = np.hanning(len(x))
	x = x*window
	fftx = np.fft.fft(x)
	fftx = np.abs(fftx)/len(fftx)
	fftx = 2*fftx[0:int(len(fftx)/2)]
	tr = len(x)/fs
	df = 1.0/tr
	f = np.array([i*df for i in range(len(fftx))])
	magX = fftx
	return magX, f, df

def pha_fft(x, fs):
	fftx = np.fft.fft(x)
	fftx = np.angle(fftx)
	fftx = fftx[0:int(len(fftx)/2)]
	tr = len(x)/fs
	df = 1.0/tr
	f = np.array([i*df for i in range(len(fftx))])
	magX = fftx
	return magX, f, df

def mag2_fft(x, fs):
	fftx = np.fft.fft(x)
	fftx = np.abs(fftx)/len(fftx)
	# fftx = 2*fftx[0:int(len(fftx)/2)]
	tr = len(x)/fs
	df = 1.0/tr
	f = np.array([i*df for i in range(len(fftx))])
	magX = fftx
	return magX, f, df

	
def kurtogram(x, fs, levels):
	
	# fig, ax = plt.subplots()
	
	# t = np.arange(3)
	# f = np.arange(3)
	# tt, ff = np.meshgrid(t, f)	
	# zz = tt + ff*2.
	# print(zz)
	
	# zz = [[1, 1, 1], [1, 1, 6], [7, 10, 18]]

	
	# extent = np.min(t), np.max(t), np.min(f), np.max(f)

	# map = ax.imshow(zz, extent=extent)
	

	# fig.colorbar(map, ax=ax)
	
	# plt.show()
	# sys.exit()
	
	

	# y, x = np.mgrid[slice(1, 5 + dy, dy), slice(1, 5 + dx, dx)]
	# print(y)
	# sys.exit()
	# x = np.array([[0,0,0], [0,1], [1,0], [1,1]])
	# y = np.array([[0,0], [0,1], [1,0], [1,1]])
	
	# x = [2, 2, 2]
	# y = [3, 3, 3]
	
	
	# xx, yy = np.meshgrid(t, f)
	# print(x)
	# print(y)
	
	# c = np.array([[2, 6], [6, 10]])
	# fig, ax = plt.subplots()
	# map = ax.pcolormesh(xx, yy, c)
	# fig.colorbar(map, ax=ax)
	# plt.show()
	# sys.exit()



	
	flag = 'ON'
	List_n = [0, 1]
	
	Levels = np.arange(levels)
	
	for k in range(2*levels-2):
		if flag == 'ON':
			List_n.append(List_n[-1]+0.6)
			flag = 'OFF'
		else:
			List_n.append(List_n[-1]+0.4)
			flag = 'ON'
	# print(list_n)
	# 
	
	Widths = [fs/2.**(n+1) for n in List_n]
	
	print(Widths)
	print(List_n)
	# from math import floor
	# sys.exit()
	# n_centers = 1
	kurto = {}
	Centers = {}
	for n in range(len(Widths)):
		# print('new level!!!')
		n_centers = 2.**(List_n[n])
		
		
		array_kurtosis = []
		array_center = []
		for k in range(int(n_centers)):
			center = fs*((k+0.5)*2.**(-List_n[n] - 1))
			width = Widths[n]
			cutoff_hp = center + width/2
			cutoff_lp = center - width/2
			
			filt_x = butter_bandpass(x=x, fs=fs, freqs=[cutoff_lp, cutoff_hp], order=3, warm_points=None)
			
			# filt_x = butter_highpass(x=x, fs=fs, freq=cutoff_hp, order=3, warm_points=None)
			# filt_x = butter_lowpass(x=filt_x, fs=fs, freq=cutoff_lp, order=3, warm_points=None)
			
			value_kurtosis = scipy.stats.kurtosis(filt_x, fisher=False)
			
			array_kurtosis.append(value_kurtosis)
			array_center.append(center)
		
		kurto[str(n)] = array_kurtosis
		Centers[str(n)] = array_center
			
			
			# print(center)
		# n_centers += 1
	print(kurto)
	print(Centers)
	print(Widths)
	a = input('pause')
	# sys.exit()
	
	# kurto = {}
	
	# for i in range(levels):
	
	# oldy = kurto['2']
	# newy = rearange_for_kurto(kurt_array, n_centers)
	


	
	
	kurto_convert = []
	
	for key, values in kurto.items():	
		kurto_convert.append(np.nan_to_num(rearange_for_kurto(values, n_centers)))

	
	
	print(kurto_convert)
	
	fig, ax = plt.subplots()
	map = ax.pcolormesh(kurto_convert)
	fig.colorbar(map, ax=ax)
	plt.show()
	
	sys.exit()
	
	return kurto, List_n, 

def rearange_for_kurto(kurt_array, n_centers):
	oldy = kurt_array
	newx = np.arange(n_centers)	
	oldn = len(oldy)
	cuo = int(n_centers//oldn)
	newy = []
	for element in oldy:
		newy += [element for i in range(cuo)]	
	while len(newy) < len(newx):
		newy.append(0)	
	newy = np.ravel(newy)
	return newy

def plot_kurtogram(kurto, levels):
	
	# fig, ax = plt.subplots()
	
	# t = np.arange(3)
	# f = np.arange(3)
	# tt, ff = np.meshgrid(t, f)	
	# zz = tt + ff*2.
	# print(zz)
	
	# zz = [[1, 1, 1], [1, 1, 6], [7, 10, 18]]
	
	# extent = np.min(t), np.max(t), np.min(f), np.max(f)

	# map = ax.imshow(zz, extent=extent)

	# fig.colorbar(map, ax=ax)
	
	# plt.show()
	# sys.exit(
	maxlen = len(kurto[str(2*levels)])
	
	
	
	return ax
	

def shortFFT(x, fs, segments, window, mode):
	nperseg = int(len(x)/segments)
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

	t_stpsd = np.linspace(0.0, n_x/fs, num=segments)

	stpsdX = list(map(list, zip(*stpsdX)))

	df = f_stpsd[2] - f_stpsd[1]
	return stpsdX, f_stpsd, df, t_stpsd	

def max_cspectrum(ciclicspectrum):
	max = 0.0
	for i in range(len(ciclicspectrum)):
		pmax = np.max(np.array(ciclicspectrum[i]))
		if pmax > max:
			max = pmax
	return max

def Cyclic_Spectrum(x, fs, segments, freq_range, warm_points):
	n_x = len(x)
	freqs_limits = np.linspace(start=freq_range[0], stop=freq_range[1], num=segments+1)	
	filtered_x = []
	for p in range(segments):
		resp = x
		filtered_x.append(butter_bandpass(x=x, fs=fs, freqs=[freqs_limits[p], freqs_limits[p+1]], order=3, warm_points=warm_points))	
	CyclicSpectrum = []
	maxvec = []
	for i in range(segments):
		print(i/segments)
		CyclicSpectrumWindow = filtered_x[i]**2.0
		resp = CyclicSpectrumWindow
		CyclicSpectrumWindow = butter_lowpass(x=resp, fs=fs, freq=1000.0, order=3, warm_points=None)
		CyclicSpectrumWindow, a_CyclicSpectrum, da = mag_fft(CyclicSpectrumWindow, fs)
		CyclicSpectrumWindow = CyclicSpectrumWindow[0:int(1000/da)]
		a_CyclicSpectrum = a_CyclicSpectrum[0:int(1000/da)]
		CyclicSpectrum.append(CyclicSpectrumWindow)
	f_CyclicSpectrum = freqs_limits

	
	return CyclicSpectrum, a_CyclicSpectrum, f_CyclicSpectrum

def Cyclic_Spectrum2(x, fs, segments, freq_range, warm_points):
	n_x = len(x)
	freqs_limits = np.linspace(start=freq_range[0], stop=freq_range[1], num=segments+1)	
	filtered_x = []
	for p in range(segments):
		resp = x
		filtered_x.append(butter_bandpass(x=x, fs=fs, freqs=[freqs_limits[p], freqs_limits[p+1]], order=3, warm_points=warm_points))	
	CyclicSpectrum = []
	maxvec = []
	for i in range(segments):
		print(i/segments)
		CyclicSpectrumWindow = filtered_x[i]**2.0
		resp = CyclicSpectrumWindow
		CyclicSpectrumWindow = hilbert_demodulation(x=resp)
		CyclicSpectrumWindow, a_CyclicSpectrum, da = mag_fft(CyclicSpectrumWindow, fs)
		CyclicSpectrumWindow = CyclicSpectrumWindow[0:int(1000/da)]
		a_CyclicSpectrum = a_CyclicSpectrum[0:int(1000/da)]
		CyclicSpectrum.append(CyclicSpectrumWindow)
	f_CyclicSpectrum = freqs_limits

	
	return CyclicSpectrum, a_CyclicSpectrum, f_CyclicSpectrum

def imean(x):
	n = len(x)
	imeanx = np.zeros(n)
	for i in range(n):
		acu_mean = np.sum(x[0:i])/i
		imeanx[i] = acu_mean
	return imeanx
	
	
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