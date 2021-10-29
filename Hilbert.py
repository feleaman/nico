# from tkinter import *
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import interpolate
from os.path import isfile, join, basename
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import pickle
sys.path.insert(0, './lib') #to open user-defined functions
from m_open_extension import *
from m_denois import *

from m_processing import *
from m_det_features import *

from tkinter import filedialog
from tkinter import Tk
import pandas as pd
from os import chdir
plt.rcParams['savefig.directory'] = chdir(os.path.dirname('C:'))
# plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes

plt.rcParams['savefig.dpi'] = 1100
plt.rcParams['savefig.format'] = 'jpeg'

def mag_fft(x, fs):
	fftx = np.fft.fft(x)
	fftx = np.abs(fftx)/len(fftx)
	fftx = 2*fftx[0:int(len(fftx)/2)]
	tr = len(x)/fs
	df = 1.0/tr
	f = np.array([i*df for i in range(len(fftx))])
	magX = fftx
	return magX, f, df

def sifting_iteration(t, x):
	s_number = 5
	tolerance = 10
	error = 5000
	max_iter = 505
	min_iter = 500
	cont = 0
	s_iter = 10
	while (error > tolerance or s_iter < s_number):
		print('++++++++iteration ', cont)
		h1, extrema_x = sifting2(t, x)
		if cont > min_iter:
			error_extrema = extrema(h1)-extrema_x
			error_xzeros = xzeros(h1)-xzeros(x)		
			error = error_extrema + error_xzeros
			print(error)
			# if error < tolerance:
				# s_iter = s_iter + 1
			# else:
				# s_iter = 0
			# print('Conseq. it = ', s_iter)
		x = h1
		cont = cont + 1
		if cont > max_iter:
			break
	return h1

def extrema(x):
	n = len(x)
	n_extrema = 0
	for i in range(n-2):
		if (x[i+1] < x[i] and x[i+2] > x[i+1]):
			n_extrema = n_extrema + 1
	for i in range(n-2):
		if (x[i+1] > x[i] and x[i+2] < x[i+1]):
			n_extrema = n_extrema + 1

	return n_extrema

def xzeros(x):
	n = len(x)
	n_xzeros = 0
	for i in range(n-1):
		if (x[i] > 0 and x[i+1] < 0):
			n_xzeros = n_xzeros + 1
	for i in range(n-1):
		if (x[i] < 0 and x[i+1] > 0):
			n_xzeros = n_xzeros + 1

	return n_xzeros
	
def env_down(t, x):
	n = len(x)
	x_down = []
	t_down = []
	x_down.append(x[0])
	t_down.append(t[0])
	for i in range(n-2):
		if (x[i+1] < x[i] and x[i+2] > x[i+1]):
			x_down.append(x[i+1])
			t_down.append(t[i+1])
	x_down.append(x[n-1])
	t_down.append(t[n-1])
	x_down = np.array(x_down)
	t_down = np.array(t_down)

	return t_down, x_down


def env_up(t, x):
	n = len(x)
	x_up = []
	t_up = []
	x_up.append(x[0])
	t_up.append(t[0])
	for i in range(n-2):
		if (x[i+1] > x[i] and x[i+2] < x[i+1]):
			x_up.append(x[i+1])
			t_up.append(t[i+1])
	x_up.append(x[n-1])
	t_up.append(t[n-1])
	x_up = np.array(x_up)
	t_up = np.array(t_up)
	
	return t_up, x_up

def sifting(t, x):
	t_up, x_up = env_up(t, x)
	t_down, x_down = env_down(t, x)

	tck = interpolate.splrep(t_up, x_up)
	x_up = interpolate.splev(t, tck)
	tck = interpolate.splrep(t_down, x_down)
	x_down = interpolate.splev(t, tck)

	x_mean = (x_up + x_down)/2
	h = x - x_mean
	return h

def sifting2(t, x):
	t_up, x_up = env_up(t, x)
	t_down, x_down = env_down(t, x)
	extrema_x = len(x_up) + len(x_down)
	
	tck = interpolate.splrep(t_up, x_up)
	x_up = interpolate.splev(t, tck)
	tck = interpolate.splrep(t_down, x_down)
	x_down = interpolate.splev(t, tck)

	x_mean = (x_up + x_down)/2
	h = x - x_mean
	return h, extrema_x

def hilbert_spectrum(time, frec, amp):
	x = time
	y = frec / 1000.
	z = amp
	# z = np.log(amp)
	# z = amp
	n = len(x)
	# cmap = plt.get_cmap('binary')
	# cmap = plt.get_cmap('plasma')
	cmap = plt.get_cmap('copper_r')

	norm = BoundaryNorm(np.linspace(z.min(), z.max(), 1000), cmap.N)
	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	lc = LineCollection(segments, cmap=cmap, norm=norm)
	lc.set_array(z)
	lc.set_linewidth(2)
	# fig1 = plt.figure()
	fig1, ax1 = plt.subplots()
	plt.gca().add_collection(lc)
	plt.xlim(x.min(), x.max())
	plt.ylim(y.min(), y.max())
	
	line = ax1.add_collection(lc)
	
	
	
	# cbar = fig1.colorbar(line, ax=ax1, format='%1.1e')
	# cbar = fig1.colorbar(line, ax=ax1, format='%1.1e')
	cbar = fig1.colorbar(line, ax=ax1, format='%1.0f')
	
	cbar.set_label('Amplitude [mV]', size=13)
	cbar.ax.tick_params(labelsize=12) 
	
	cbar.set_ticks(list(np.arange(np.min(z), np.max(z), 75)))
	# cbar.set_ticks(list(np.linspace(np.min(z), np.max(z), 7)))
	
	
	# cbar.ax.xaxis.set_ticks_position('top')
	
	ax1.set_xlabel('Time [s]', fontsize=13)
	ax1.set_ylabel('Frequency [kHz]', fontsize=13)
	# cbar.ax.set_xlabel('Amplitude [mV]', rotation=270, fontsize=12)
	# fig1.text(0.93, 0.5, 'Amplitude [mV]', ha='center', va='center', rotation='vertical', fontsize=13)
	# ax1.set_ylim(0., 35.)
	# ax1.set_ylim(75., 210.)
	# ax1.set_ylim(120., 400.)
	ax1.set_ylim(80., 240.)
	# ax1.set_ylim(50., 130.)
	# ax1.set_ylim(85., 200.)
	# ax1.set_ylim(110., 350.)
	# ax1.set_ylim(65., 130.)
	
	ax1.tick_params(axis='both', labelsize=12)
	
	# cbar.ax.ticklabel_format(format='%d')
	
	# format='%1.2e'

	# vmax = np.max(z)
	# fig1.colorbar(cmap, ax=ax1, ticks=np.linspace(0, vmax, 5), format='%1.2e')
	
	# plt.show()
	return fig1, ax1

def marginal_hilbert_spectrum_1hz(time, frec, amp, dt):
	df = 1
	times = np.array(time)
	frequencies = np.array(frec)
	
	for i in range(len(frequencies)):
		frequencies[i] = np.rint(frequencies[i])
	
	amplitudes = np.array(amp)

	min_freq = np.min(frequencies)
	max_freq = np.max(frequencies)
	new_frequencies = np.arange(start=min_freq, stop=max_freq+df, step=df)
	
	dict_freq = {}
	for element in new_frequencies:
		dict_freq[str(element)] = 0.
	
	for i in range(len(frequencies)):
		dict_freq[str(frequencies[i])] += amplitudes[i]
	
	new_amplitudes = []
	for key, value in dict_freq.items():
		new_amplitudes.append(value)

	
	print(np.sum(np.array(new_amplitudes)))
	fig, ax = plt.subplots()
	
	
	ax.plot(new_frequencies, new_amplitudes)
	
	ax.set_xlabel('Frequency [kHz]', fontsize=12)
	ax.set_ylabel('Amplitude [mV]', fontsize=12)
	
	
	
	return fig, ax

def hilbert_spectrum_mod(time, frec, amp, limit):
	xx = time
	yy = frec / 1000.
	zz = amp * 1000.
	# z = np.log(amp)
	# z = amp
	x = []
	y = []
	z = []
	
	
	for k in range(len(zz)):
		if zz[k] >= limit:
			x.append(xx[k])
			y.append(yy[k])
			z.append(zz[k])
	x = np.array(x)
	y = np.array(y)
	z = np.array(z)
	
	n = len(x)
	# cmap = plt.get_cmap('binary')
	# cmap = plt.get_cmap('plasma')
	cmap = plt.get_cmap('cool_r')

	norm = BoundaryNorm(np.linspace(z.min(), z.max(), 1000), cmap.N)
	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	lc = LineCollection(segments, cmap=cmap, norm=norm)
	lc.set_array(z)
	lc.set_linewidth(2)
	# fig1 = plt.figure()
	fig1, ax1 = plt.subplots()
	plt.gca().add_collection(lc)
	plt.xlim(x.min(), x.max())
	plt.ylim(y.min(), y.max())
	
	line = ax1.add_collection(lc)
	

	
	# cbar = fig1.colorbar(line, ax=ax1, format='%1.1e')
	# cbar = fig1.colorbar(line, ax=ax1, format='%1.1e')
	cbar = fig1.colorbar(line, ax=ax1, format='%1.1f')
	# cbar.ax.xaxis.set_ticks_position('top')
	
	ax1.set_xlabel('Timeaaa [s]', fontsize=13)
	ax1.set_ylabel('Frequency [kHz]', fontsize=13)
	# cbar.ax.set_xlabel('Amplitude [mV]', rotation=270, fontsize=12)
	fig1.text(0.93, 0.5, 'Amplitude [mV]', ha='center', va='center', rotation='vertical', fontsize=13)
	ax1.set_ylim(120., 430.)
	
	ax1.tick_params(axis='both', labelsize=12)

	return fig1, ax1
	
#++++++++++++++++++++++++++++DEFINITION
root = Tk()
root.withdraw()
root.update()
Filenames = filedialog.askopenfilenames()
root.destroy()

mydict = {'80':[], '100':[], '120':[], '140':[], '160':[], '180':[], '200':[], '220':[], '240':[], '260':[]}

for filename in Filenames:

	h1 = load_signal(filename, channel='AE_0')
	print(filename)



	# h1 = h1[0:int(len(h1)/4)]
	# h1 = h1[int(len(h1)/2):]
	# h1 = h1[0:int(len(h1)/1)]

	fs = 1000000.0
	dt = 1./fs
	# freq = 10.

	# h1 = [np.cos(2*np.pi*freq*(1+i*0.000001)*i*dt+1) for i in range(1000000)]
	# print(freq*(1+1000000*0.000001))
	x = h1



	n = len(x)
	t = np.array([i*dt for i in range(n)])

	# plt.plot(t, h1)
	# plt.show()
	#++++++++++++++++++++++++++++ENVELOPES AND MEAN
	from scipy.signal import hilbert

	# fs = 1000000.
	# h1 = butter_lowpass(x=h1, fs=1000000., freq=250.e3, order=3, warm_points=None)
	hx = hilbert(h1)
	# hx = h1



	amp = np.absolute(hx)

	# exp = hx

	# exp = autocorr_fft(amp)
	# plt.plot(t, exp, 'k')
	# plt.show()
	# magE, f, df = mag_fft(hx, fs)
	# plt.plot(f, magE, 'r')
	# plt.show()


	# hx = hx[0:int(len(hx)/2)]
	# pha = np.angle(hx) + np.pi*np.ones(len(hx))
	pha = np.angle(hx)




	pha = np.unwrap(pha) 


	# sum = 0.0
	# for i in range(len(pha)):
		# if pha[i]
		# sum = sum + pha[i]
		# pha[i] = sum + pha[i]
		
	# pha = diff_signal_eq(x=pha, length_diff=1)
	pha = diff_signal(x=pha, length_diff=1)


	# pha = np.array([pha[i+1] - pha[i] for i in range(len(pha)-1)])
	# for i in range(len(pha)):
		# if pha[i] < 0:
			# pha[i] = pha[i] + 2*np.pi
	pha = pha / dt
	pha = pha / (2*np.pi)
	pha = np.absolute(pha)
	pha = np.array(list(pha) + [pha[len(pha)-1]])







	# fig1, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
	# ax[0].plot(pha)
	# ax[1].plot(amp)
	# ax[2].plot(h1)
	print(len(t))
	print(len(pha))

	print(len(amp))
	# a = input('pause')


	x = t
	y = pha
	# print('Mean Freq. No Filter AND kurtosis AND SD')
	# print(np.mean(y/1000.))

	# print(scipy.stats.kurtosis(y/1000., fisher=False))
	# print(np.std(y/1000.))
	z = amp
	
	# y = butter_highpass(x=y, fs=1000000., freq=280., order=3, warm_points=None)
	# plt.plot(y)
	# plt.show()
	y = butter_lowpass(x=y, fs=1000000., freq=500., order=3, warm_points=None)
	# plt.plot(y)
	# plt.show()
	# fig, ax = marginal_hilbert_spectrum_1hz(time=x, frec=y, amp=z, dt=1.e-6)
	# plt.show()
	# sys.exit()
	
	# y = butter_bandpass(x=y, fs=1000000., freqs=[280., 340.], order=3, warm_points=None)
	# y = butter_highpass(x=y, fs=1000000., freq=10., order=3, warm_points=None)

	# y = autocorr_fft(y)
	# plt.plot(t, y, 'g')
	# plt.show()

	# plt.plot(x, y, 'g')
	# plt.show()



	# 

	# y = hilbert_demodulation(y)
	# y = hilbert_demodulation(y)

	# plt.plot(x, y, 'k')
	# plt.show()
	# magY, f, df = mag_fft(y, fs)
	# plt.plot(f, magY, 'k')
	# plt.show()


	# y = butter_bandpass(x=y, fs=1000000., freqs=[15., 120.], order=3, warm_points=None)
	# y = butter_highpass(x=y, fs=1000000., freq=15., order=3, warm_points=None)
	# y = fourier_filter(x=y, fs=1000000., type='bandpass', freqs=[1., 500.])
	


	# fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)
	# ax[0].plot(x, z, 'b')
	# ax[1].plot(x, y/1000., 'r')
	# print('Mean Freq. WITH Filter AND kurtosis AND SD')

	# print(np.mean(y/1000.))
	# print(scipy.stats.kurtosis(y/1000., fisher=False))
	# print(np.std(y/1000.))

	# ax[1].set_xlabel('Time [s]', fontsize=13)
	# ax[1].set_ylabel('Frequency [kHz]', fontsize=13)
	# ax[0].set_ylabel('Amplitude [mV]', fontsize=13)
	# ax[0].tick_params(axis='both', labelsize=12)
	# ax[1].tick_params(axis='both', labelsize=12)

	# ax[0].set_ylim(bottom=0, top=300)
	# ax[1].set_ylim(bottom=100, top=250)

	# ax[0].set_title('Channel AE-1, Time 15:45:57', fontsize=13)


	# plt.show()

	# plt.plot(y[1000:]/1000., color='k')

	# print('Mean Freq. WITH Filter WARM 1000 AND kurtosis AND SD... max')

	# print(np.mean(y[1000:]/1000.))
	# print(scipy.stats.kurtosis(y/1000., fisher=False))
	# print(np.std(y[1000:]/1000.))
	# print(np.max(y[1000:]/1000.))


	curve = y[1000:]/1000.

	# amplitudes = z[1000:]
	amplitudes = z[1000:]*1000.
	# mydict = {'80':0., '100':0., '120':0., '140':0., '160':0., '180':0., '200':0., '220':0., '240':0., '260':0.}
	
	sum_80 = 0.
	sum_100 = 0.
	sum_120 = 0.
	sum_140 = 0.
	sum_160 = 0.
	sum_180 = 0.
	sum_200 = 0.
	sum_220 = 0.
	sum_240 = 0.
	sum_260 = 0.
	
	count = 0
	for element in curve:
		if element >= 80. and element < 100.:
			# mydict['80'] += (amplitudes[count])**2.0
			sum_80 += (amplitudes[count])**2.0
			
		elif element >= 100. and element < 120.:
			# mydict['100'] += (amplitudes[count])**2.0
			sum_100 += (amplitudes[count])**2.0
			
		elif element >= 120. and element < 140.:
			# mydict['120'] += (amplitudes[count])**2.0
			sum_120 += (amplitudes[count])**2.0
			
		elif element >= 140. and element < 160.:
			# mydict['140'] += (amplitudes[count])**2.0
			sum_140 += (amplitudes[count])**2.0
			
		elif element >= 160. and element < 180.:
			# mydict['160'] += (amplitudes[count])**2.0
			sum_160 += (amplitudes[count])**2.0
			
		elif element >= 180. and element < 200.:
			# mydict['180'] += (amplitudes[count])**2.0
			sum_180 += (amplitudes[count])**2.0
			
		elif element >= 200. and element < 220.:
			# mydict['200'] += (amplitudes[count])**2.0
			sum_200 += (amplitudes[count])**2.0
		
		elif element >= 220. and element < 240.:
			# mydict['220'] += (amplitudes[count])**2.0
			sum_220 += (amplitudes[count])**2.0
		
		elif element >= 240. and element < 260.:
			# mydict['240'] += (amplitudes[count])**2.0
			sum_240 += (amplitudes[count])**2.0
		
		elif element >= 260. and element <= 280.:
			# mydict['260'] += (amplitudes[count])**2.0
			sum_260 += (amplitudes[count])**2.0
	# mydict['260'] = sum_260
	# mydict['240'] = sum_240
	# mydict['220'] = sum_220
	# mydict['200'] = sum_200
	# mydict['180'] = sum_180
	# mydict['160'] = sum_160
	# mydict['140'] = sum_140
	# mydict['120'] = sum_120
	# mydict['100'] = sum_100
	# mydict['80'] = sum_80
	
	mydict['260'].append(sum_260)
	mydict['240'].append(sum_240)
	mydict['220'].append(sum_220)
	mydict['200'].append(sum_200)
	mydict['180'].append(sum_180)
	mydict['160'].append(sum_160)
	mydict['140'].append(sum_140)
	mydict['120'].append(sum_120)
	mydict['100'].append(sum_100)
	mydict['80'].append(sum_80)

	# plt.show()
rownames = mydict.keys
DataFr = pd.DataFrame.from_dict(mydict)
# DataFr = pd.DataFrame.from_dict(mydict, orient='index')

writer = pd.ExcelWriter('EnergyEnd.xlsx')
DataFr.to_excel(writer, sheet_name='Sheet')
writer.close()


# #

# sys.exit()



fig, ax = hilbert_spectrum(x, y, z)
# sys.exit()
filename = basename(filename)
filename = 'Channel AE-2, Second IMF, LC N°1'
plt.title(filename, fontsize=13)
# plt.tight_layout()
plt.show()

# fig2, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
# fft_amp, f, df = mag_fft(amp, fs)
# fft_h1, f, df = mag_fft(h1, fs)

# ax[0][0].plot(h1)
# ax[0][1].plot(amp)
# ax[1][0].plot(fft_h1)
# ax[1][1].plot(fft_amp)
# import matplotlib.cm as cm
# def plot_colourline(x,y,c):
    # c = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))
    # ax = plt.gca()
    # for i in np.arange(len(x)-1):
        # ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=c[i])
    # return

# fig2, ax = plt.subplots()
# plot_colourline(t, pha, amp)
# plt.show()




# x = np.linspace(0, 3 * np.pi, 500)
# y = 5*np.sin(x)
# z = np.linspace(-2, 2, 500)
# cmap = plt.get_cmap('cool')
# norm = BoundaryNorm(np.linspace(-2, 2, 500), cmap.N)
# points = np.array([x, y]).T.reshape(-1, 1, 2)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)
# lc = LineCollection(segments, cmap=cmap, norm=norm)
# lc.set_array(z)
# lc.set_linewidth(2)
# fig1 = plt.figure()
# plt.gca().add_collection(lc)
# plt.xlim(x.min(), x.max())
# plt.ylim(y.min(), y.max())
# plt.show()




sys.exit()

fig3, ax = plt.subplots(nrows=2, ncols=2)
fourier_h1 = np.fft.fft(h1)
fourier_amp = np.fft.fft(amp)

# autocorr_amp_def = np.correlate(amp, amp, mode='same')
# autocorr_h1 = np.real(np.fft.ifft(fourier_h1*np.conjugate(fourier_h1)))
autocorr_amp = np.real(np.fft.ifft(fourier_amp*np.conjugate(fourier_amp)))
fft_autocorr_amp, f, df = mag_fft(autocorr_amp, fs)

print(len(autocorr_amp))
print(len(h1))
# print(len(autocorr_def))
# np.interp(np.linspace(0, len(autocorr_amp)-1, len(autocorr_amp)/2), [i for i in range(len(autocorr_amp))], autocorr_amp)

ax[0][0].plot(amp)
ax[0][1].plot(amp)
ax[1][0].plot(autocorr_amp[0:len(autocorr_amp)/2])
ax[1][1].plot(fft_autocorr_amp)




# hx = hilbert(h1ok)
# pha = np.angle(hx)
# pha = np.unwrap(pha)
# pha = np.array([pha[i+1] - pha[i] for i in range(len(pha)-1)])
# pha = pha / dt

# amp = np.absolute(hx)

# fig2, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
# ax[0].plot(pha)
# ax[1].plot(amp)




plt.show()


#++++++++++++++++++++++++++++++++COMMENTS

