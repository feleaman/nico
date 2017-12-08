# from tkinter import *
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import interpolate
from os.path import isfile, join
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import pickle
sys.path.insert(0, './lib') #to open user-defined functions
from m_open_extension import *
from m_processing import *


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
	y = frec
	z = amp
	n = len(x)
	cmap = plt.get_cmap('binary')
	norm = BoundaryNorm(np.linspace(z.min(), z.max(), 1000), cmap.N)
	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	lc = LineCollection(segments, cmap=cmap, norm=norm)
	lc.set_array(z)
	lc.set_linewidth(2)
	fig1 = plt.figure()
	plt.gca().add_collection(lc)
	plt.xlim(x.min(), x.max())
	plt.ylim(y.min(), y.max())
	plt.show()
#++++++++++++++++++++++++++++DEFINITION
mypath = 'C:/Felix/Code/nico'
filename = join(mypath, 'h1_export_0105_2017_11_22_18_46_40_raw.pkl')
h1 = read_pickle(filename)

# h1 = h1[0:5000000]
# h1 = np.loadtxt(filename)


# plt.plot(h1)
# plt.show()
# sys.exit()
# filename = join(mypath, 'h2_V1_9_n1500_M80_AE_Signal_20160928_144737.txt')
# h2 = np.loadtxt(filename)


# mypath = 'C:/Felix/Data/CNs_Getriebe/Paper_Bursts/Analysis_Case_1500_80/OK'
# filename = join(mypath, 'h1_V1_9_n1500_M80_AE_Signal_20160506_142422.txt')
# h1 = np.loadtxt(filename)
# filename = join(mypath, 'h2_V1_9_n1500_M80_AE_Signal_20160506_142422.txt')
# h2 = np.loadtxt(filename)
fs = 10000000.0
dt = 1/fs
# freq = 10.

# h1 = [np.cos(2*np.pi*freq*(1+i*0.000001)*i*dt+1) for i in range(1000000)]
# print(freq*(1+1000000*0.000001))
x = h1



n = len(x)
t = np.array([i*dt for i in range(n)])

#++++++++++++++++++++++++++++ENVELOPES AND MEAN
from scipy.signal import hilbert

# fs = 1000000.

hx = hilbert(h1)
hx = hx[0:int(len(hx)/2)]
# pha = np.angle(hx) + np.pi*np.ones(len(hx))
pha = np.angle(hx)

pha = np.unwrap(pha) 
# sum = 0.0
# for i in range(len(pha)):
	# if pha[i]
	# sum = sum + pha[i]
	# pha[i] = sum + pha[i]
	
pha = diff_signal_eq(x=pha, length_diff=1)
# pha = np.array([pha[i+1] - pha[i] for i in range(len(pha)-1)])
# for i in range(len(pha)):
	# if pha[i] < 0:
		# pha[i] = pha[i] + 2*np.pi
pha = pha / dt
pha = pha / (2*np.pi)
pha = np.absolute(pha)
amp = np.absolute(hx)

# fig1, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
# ax[0].plot(pha)
# ax[1].plot(amp)
# ax[2].plot(h1)
print(len(t))
print(len(pha))

print(len(amp))

# x = np.linspace(0, 3 * np.pi, 500)
# y = 5*np.sin(x)
# z = 5*np.sin(10*x)
# x = t[0:500000]
# y = pha[0:500000]
# z = amp[0:500000]
t_new = []
for i in range(int(len(t)/2)):
	t_new.append(t[i*2])
print(len(t_new))
t = np.array(t_new)
# sys.exit()

x = t
y = pha
z = amp

x_new = []
for i in range(int(len(x)/5)):
	x_new.append(x[i*5])
x = np.array(x_new)

y_new = []
for i in range(int(len(y)/5)):
	y_new.append(y[i*5])
y = np.array(y_new)

z_new = []
for i in range(int(len(z)/5)):
	z_new.append(z[i*5])
z = np.array(z_new)


# plt.plot(t[0:500000], h1[0:500000])
# plt.plot(t, h1)

# plt.show()


# plt.plot(z)
# plt.title('amp')
# plt.show()

# plt.plot(y)
# plt.title('pha')
# plt.show()


# sys.exit()
hilbert_spectrum(x, y, z)


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

