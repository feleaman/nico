# Reco_Signal_Training.py
# Last updated: 23.09.2017 by Felix Leaman
# Description:
# 

#++++++++++++++++++++++ IMPORT MODULES +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import matplotlib.cm as cm
# from tkinter import filedialog
# from skimage import img_as_uint
from tkinter import Tk
from tkinter import Button
from tkinter import filedialog

import os
import sys
sys.path.insert(0, './lib') #to open user-defined functions

from m_open_extension import *
from m_fft import *
from m_demodulation import *
#from m_pattern import *
from m_denois import *
from m_det_features import *
from m_processing import *
import pickle
import argparse
import pywt
import pandas as pd
import datetime
# import numpy as np
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
from argparse import ArgumentParser

plt.rcParams['savefig.directory'] = os.chdir(os.path.dirname('D:'))

plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'jpeg'

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from random import randint
Inputs = ['mode', 'channel', 'fs']

# 'mypath':R'C:\\Felix\\29_THESIS\\MODEL_B\\Chapter_4_Prognostics\\04_Data\\Master'

InputsOpt_Defaults = {'mypath':None, 'name':'auto', 'feature':'RMS', 'train':0.60, 'n_mov_avg':0, 'predict_node':'last', 'auto_layers':None, 'save_plot':'ON', 'n_bests':3, 'weight':0.05, 'n_children':7, 'save_model':'OFF', 'features_array':['RMS', 'MAX', 'KURT', 'CORR', 'sqr10h_f_f_r', 'sqr5h_f_g', 'sqr5h_sb_f_g_c', 'Temp', 'acRMS', 'acMAX', 'acKURT', 'bursts']}
# 'acRMS', 'acMAX', 'acKURT'

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	

	
	if config['mode'] == 'test':
		print(pywt.wavelist('morl'))
		sys.exit()
		
		if config['mypath'] == None:
			print('Select File with signal')
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()
			root.destroy()
		else:
			filepath = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
		
		x = load_signal(filepath, channel=config['channel'])
		
		# sys.exit() 		
		x = x[0:int(len(x)/5)]
		s_level = 5
		coeffs = pywt.wavedec(x, 'db6', level=s_level, mode='periodic')
		# print(coeffs)
		# print(len(x))
		# print(len(coeffs))
		# print(len(coeffs[0]))
		# print(len(coeffs[1]))
		# print(len(coeffs[2]))
		for k in range(s_level+1):
			print(len(coeffs[k]))
		sys.exit()
		
		# wsignal = (x)**2.0
		# wsignal = (coeffs[5])**2.0
		wsignal = (coeffs[5])
		wsignal = odd_to_even(wsignal)
		# wsignal = wsignal / (np.max(wsignal) - np.min(wsignal))
		print(type(wsignal))
		print(np.ravel(wsignal))
		
		plt.figure(1)
		plt.plot(wsignal, color='blue')
		
		fft, f, df = mag_fft(x=wsignal, fs=config['fs'])
		plt.figure(2)
		plt.plot(f, fft, color='red')
		
		env = hilbert_demodulation(wsignal)
		plt.figure(3)
		plt.plot(env, color='green')
		
		fftenv, f, df = mag_fft(x=env, fs=config['fs'])
		plt.figure(4)
		plt.plot(f, fftenv, color='black')	
		
		
		plt.show()
	
	elif config['mode'] == 'work_with_wavelet':
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		print(basename(filepath))
		
		mydict = read_pickle(filepath)
		x = mydict['x']
		fs = mydict['fs']
		best_lvl = mydict['best_lvl']
		dt = 1./fs
		t = dt*np.arange(len(x))
		
		# x = hilbert_demodulation(x)
		# magX, f, df = mag_fft(x, fs)
		
		# plt.plot(f, magX)
		# plt.show()	
		
		
		segments = 20000.
		window = 'boxcar'
		mode = 'magnitude'
		
		stftX, f_stft, df_stft, t_stft = shortFFT(x, fs, segments, window, mode)
		fig, ax = plt.subplots()

		
	
		ax.pcolormesh(t_stft, f_stft, stftX)
		ax.set_xlabel('Dauer [s]', fontsize=13.5)
		ax.set_ylabel('Frequenz [Hz]', fontsize=13.5)

		
		map = []
		vmax = max_cspectrum(stftX)				
		map.append(ax.pcolormesh(t_stft, f_stft/1000., stftX, cmap='plasma', vmax=None))

		cbar = fig.colorbar(map[0], ax=ax)
		
		# cbar.set_label('log' + ' ' + ylabel_fft, fontsize=13.5)
		plt.show()
		
		
		
	
	elif config['mode'] == 'test_cwt':
		# import pywt
		# import numpy as np
		# import matplotlib.pyplot as plt
		# x = np.arange(512)
		# y = np.sin(2*np.pi*x/32)
		# coef, freqs=pywt.cwt(y,np.arange(1,129),'gaus1')
		# plt.matshow(coef) 
		# plt.show()
		# sys.exit()


		t = np.linspace(-1, 1, 200, endpoint=False)
		dt = 0.01
		t = np.arange(1000)*dt
		# sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
		sig  = np.exp(-1*t)*np.cos(2 * np.pi * 5. * t)

		X, f, df = mag_fft(sig, 1./dt)
		plt.plot(f, X, 'r')
		plt.show()
		widths = np.arange(1, 31)
		mother_wv = 'morl'
		real_freq = pywt.scale2frequency(mother_wv, widths)/dt
		print(real_freq)
		# sampling_period_ = dt.
		# real_f = scale2frequency(widths, wavelet)/sampling_period
		cwtmatr, freqs = pywt.cwt(data=sig, scales=widths, wavelet=mother_wv, sampling_period=dt)
		# print(cwtmatr)
		# print(freqs)
		# plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
		# plt.imshow(cwtmatr, cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
		# print(len(cwtmatr))
		# print(len(cwtmatr[0]))
		# x, y = np.meshgrid(real_freq, t)
		y = real_freq
		x = t
		plt.pcolormesh(x, y, cwtmatr)		
		plt.show()
		sys.exit()
		
		print('Select File with signal')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		sig = load_signal(filepath, channel=config['channel'])
		sig = sig[:int(len(sig)/10)]
		
		
		widths = np.arange(1, 100)
		cwtmatr, freqs = pywt.cwt(sig, widths, 'morl', sampling_period=5)
		# plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  
		# plt.imshow(cwtmatr, cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
		plt.pcolormesh(cwtmatr)  

		plt.show()
	
	elif config['mode'] == 'test_cwt_2':
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		
		signal = load_signal(filepath, channel=config['channel'])
		
		# tb = 6.571951 #86-s1
		# tb = 3.013158 #220-s1
		# tb = 5.686372 #134-s1
		# tb = 5.416039 #227-s1
		# tb = 5.667336 #231-s1
		# tb = 5.667336 #231-s1
		# tb = 10.687999 #106-s1
		# tb = 5.949451 #135-s1
		# tb = 7.742351 #187-s1
		
		# tb = 4.51823 #486-b3
		# tb = 1.59737 #419-b3
		# tb = 10.399886 #555-b3
		# tb = 2.442161 #592-b3
		
		# wid = 0.002
		# tini = tb - wid/2 
		# tfin = tb + wid
		signal = signal[int(0*config['fs']) : int(1*config['fs'])]
		# signal = signal[int(0*config['fs']) : int(10*config['fs'])]/70.8*140.25
		dt = 1./config['fs']
		t = np.arange(len(signal))*dt
		
		# plt.plot(t*1000*1000, signal)
		# plt.show()
		
		
		max_width = 50
		min_width = 1
		widths = np.arange(min_width, max_width)
		mother_wv = 'morl'
		real_freq = pywt.scale2frequency(mother_wv, widths)/dt
		print(real_freq)
		# sys.exit()
		
		cwtmatr, freqs = pywt.cwt(data=signal, scales=widths, wavelet=mother_wv, sampling_period=dt)
		print(cwtmatr)
		cwtmatr**=2
		print(cwtmatr)
		sys.exit()
		# plt.plot(t, signal)
		# plt.show()
		
		# print(len(cwtmatr))
		# print(len(cwtmatr[0]))
		
		sum = np.zeros(len(cwtmatr))
		for i in range(len(cwtmatr)):
			print(cwtmatr[i])
			print((cwtmatr[i])**2.0)
			sys.exit()
			sum[i] = np.sum((cwtmatr[i])**2.0)
		plt.plot(real_freq, sum)
		plt.show()	

		
		# sys.exit()
		print(cwtmatr[2][2])
		cwtmatr = cwtmatr**2.0
		cwtmatr = np.absolute(cwtmatr)
		print(cwtmatr[2][2])
		cwtmatr = np.log10(cwtmatr)
		print(cwtmatr[2][2])

		y = real_freq
		x = t
		# plt.pcolormesh(t, real_freq, cwtmatr)
		# extent_ = [0, np.max(t), np.max(real_freq), np.min(real_freq)]
		# extent_ = None
		extent_ = [-1, 1, min_width, max_width]
		colormap_ = 'PRGn'
		# colormap_ = 'plasma'
		
		
		fig, ax = plt.subplots()
		
		
		maxxx = np.max(cwtmatr)
		minnn = np.min(cwtmatr)
		
		print('maxxx = ', maxxx)
		print('minnn = ', minnn)
		levels_ = list(np.linspace(minnn, maxxx, num=100))
		# ax.contour(t, real_freq, cwtmatr, levels=levels_)
		ax.contour(t, real_freq, cwtmatr)
		ax.set_ylim(bottom=0 , top=500000)
		
		
		
		# ax.imshow(cwtmatr, extent=extent_, cmap=colormap_, aspect='auto', vmax=maxxx, vmin=minnn, interpolation='bilinear')
		# ax.set_xlim(left=-1 , right=1)
		# ax.set_ylim(bottom=1 , top=max_width)
		
		
		# plt.imshow(cwtmatr, extent=extent_, cmap=colormap_, aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max(), interpolation='bilinear')
		
		plt.show()
	
	elif config['mode'] == 'test_cwt_3':
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		
		signal = load_signal(filepath, channel=config['channel'])
		
		# signal = signal[int(0.1*config['fs']) : int(0.101*config['fs'])]
		signal = signal[int(0*config['fs']) : int(11*config['fs'])]
		dt = 1./config['fs']
		t = np.arange(len(signal))*dt
		
		# plt.plot(signal)
		# plt.show()
		
		# nico = pywt.BaseNode(parent='caca', data=signal, node_name='d', wavelet='morl')
		nico = pywt.WaveletPacket(data=signal, wavelet='db6', maxlevel=3)
		# perro = nico.decompose()
		# caca = nico.get_level(level=3, order='freq', decompose=True)
		# nico.get_level(level=3, order='freq', decompose=True)

		# print(caca)
		# print(len(caca))
		# print(caca[0].data)
		# print(type(caca[0].data))
		# sys.exit()
		# print(caca[3])
		# print(caca[0])
		# print(caca[1])
		# print(caca[2])
		# print(caca[3])
		# print(caca[4])
		# print(caca[5])
		plt.plot(nico['dd'].data)
		plt.show()
		# print(nico['aaa'].data - caca[0].data)
		# print(caca[8])
		sys.exit()
		fig, ax = plt.subplots(nrows=4, ncols=1)
		ax[0].plot(signal, 'k')
		ax[1].plot(caca[0].data, 'g')
		ax[2].plot(caca[30].data, 'b')
		ax[3].plot(caca[60].data, 'r')
		plt.show()
		
		# print(len(perro))
		# print(perro[0])
		# print(perro[1])
		# print(perro[2])
		
		sys.exit()
		
		
		
		max_width = 201
		min_width = 1
		widths = np.arange(min_width, max_width)
		mother_wv = 'morl'
		real_freq = pywt.scale2frequency(mother_wv, widths)/dt
		print(real_freq)
		# sys.exit()		
		cwtmatr, freqs = pywt.cwt(data=signal, scales=widths, wavelet=mother_wv, sampling_period=dt)
		
		plt.plot(t, signal)
		plt.show()
		

		print(cwtmatr[2][2])
		cwtmatr = cwtmatr**2.0


		y = real_freq
		x = t

		extent_ = [-1, 1, min_width, max_width]
		colormap_ = 'PRGn'
		fig, ax = plt.subplots()
		
		levels_ = list(np.linspace(np.min(cwtmatr), np.max(cwtmatr), num=100))
		ax.contour(t, real_freq, cwtmatr, levels=levels_)
		ax.set_ylim(bottom=0 , top=500000)
		

		
		plt.show()	

	
	elif config['mode'] == 'test_wv_packet':
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		
		signal = load_signal(filepath, channel=config['channel'])
		
		signal = signal[int(0*config['fs']) : int(10*config['fs'])]
		dt = 1./config['fs']
		t = np.arange(len(signal))*dt

		max_level = 5
		
		wavelet_mother = 'db6'
		nico = pywt.WaveletPacket(data=signal, wavelet=wavelet_mother, maxlevel=max_level)


		signal = signal*1000./70.8
		# signal = signal*1000./141.25
		
		RMS = []
		MAX = []
		KURT = []
		SEN = []
		CREST = []
		mydict = {}
		mylevels = ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']
		for level in mylevels:
			print(level)
			mywav = nico[level].data
			n = len(mywav)
			
			value_max = np.max(np.absolute(mywav))			
			
			value_rms = signal_rms(mywav)	

			px = ((value_rms)**2.0)*n
			ent = 0.
			for i in range(n):
				# if (mywav[i]**2.0)/px > 1.e-15:						
				ent += ((mywav[i]**2.0)/px)*np.log2((mywav[i]**2.0)/px)
			ent = - ent	
			
			MAX.append(value_max)
			RMS.append(value_rms)
			CREST.append(value_max/value_rms)
			KURT.append(scipy.stats.kurtosis(mywav, fisher=False))
			SEN.append(ent)
		
		mydict['MAX'] = MAX
		mydict['RMS'] = RMS
		mydict['CREST'] = CREST
		mydict['KURT'] = KURT
		mydict['SEN'] = SEN
		
		row_names = mylevels
		DataFr = pd.DataFrame(data=mydict, index=row_names)
		writer = pd.ExcelWriter('AE_3_B4-5_' + wavelet_mother + '.xlsx')
		
		DataFr.to_excel(writer, sheet_name='OV_Features')	
		print('Result in Excel table')
		

		sys.exit()
		
	
	elif config['mode'] == 'test_wv_packet_2':
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		
		signal = load_signal(filepath, channel=config['channel'])
		

		name_out = 'Scgr_S1_AE3_B647_db6_lvl5_mv2.pkl'
		name_out_2 = 'Wfm_S1_AE3_B647_mv.pkl'
		
		
		#bochum b4c+++++
		# tb = 3.230392 #221212 + 286

		# tb = 2.548122 #221139 + 28
		# tb = 2.338875 #221212 + 270
		# tb = 6.941142 #221212 + 326
		# tb = 0.147558 #221305 + 485

		
		#schottland s1+++++
		# tb = 6.960124 #598-s1 193159
		tb = 17.278254 #647-s1 193159
		
		# tb = 2.688736 #13-s1
		# tb = 7.091994 #78-s1  193023
		# tb = 11.446813 #201s1 193023
		# tb = 2.062188 #309-s1 193055
		# tb = 2.327338 #313-s1 193055
		# tb = 11.173631 #347 s1 193055
		
		#new db6
		# tb = 7.091994 #178s1 193023
		# tb = 13.737517 #214s1 193023
		# tb = 13.995451 #215s1 193023
		
		# tb = 0.117088 #1-s1
		# tb = 0.624262 #2-s1
		# tb = 14.20754 #220 s1 193023
		# tb = 27.77809 #288 s1 193023 **
		# tb = 22.572601 #528 s1 193127 *
		# tb = 6.960124 #598 s1 193159 *
		# tb = 13.644469 #633 s1 193159
		
		# tb = 24.939845 #298 s1 193055 AE1
		# tb = 6.232396 #78 s1 193127 AE2
		
		
		# tb = 4.76218 #13eq-s1pai  192948

		
		# tb = 8.289245 #95-b4
		# tb = 4.50221 #173-b4
		
		# tb = 4.717965
		# tb = 4.501416
		# tb = 1.39132
		# tb = 0.089039
		# tb = 1.990012
		# tb = 1.030173
		# tb = 2.734467
		# tb = 3.772609
		# tb = 3.876989
		# tb = 4.629311
		# tb = 4.728805
		# tb = 0.290801
		# tb = 0.324761
		# tb = 3.556317
		# tb = 2.904387
		# tb = 2.45
		
		# tb = 0.73
		# tb = 0.7
		# tb = 1.19
		# tb = 1.681
		# tb = 3.81
		
		wid = 0.002
		tini = tb - wid/2 
		tfin = tb + wid
		signal = signal[int(tini*config['fs']) : int(tfin*config['fs'])]*1000./70.8
		
		# signal = signal[int(tini*config['fs']) : int(tfin*config['fs'])]*1000./141.25 #bochum
		
		
		n = len(signal)
		
		
		# signal = np.hanning(n)*signal
		
		
		# signal = signal[int(0*config['fs']) : int(10*config['fs'])]*1000./70.28
		
		
		
		# signal = signal[int(0*config['fs']) : int(10*config['fs'])]/70.8
		
		# signal = signal[int(0*config['fs']) : int(2*config['fs'])]/140.25

		dt = 1./config['fs']
		
		# t = np.arange(1000000)*dt		
		# signal = np.cos(2*3.14*400000*t)
		
		
		t = np.arange(len(signal))*dt
		n = len(signal)
		
		plt.plot(signal)
		plt.show()
		
		select_level = 5
		
		wavelet_mother = 'db6'
		nico = pywt.WaveletPacket(data=signal, wavelet=wavelet_mother, maxlevel=select_level)
		gato = pywt.WaveletPacket(data=None, wavelet=wavelet_mother, maxlevel=5)
		
		# mylevels = ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']
		mylevels = [node.path for node in nico.get_level(select_level, 'freq')]
		# freq = np.array([0, 62500, 12500, 187500, 250000, 312500, 375000, 437500])
		cwtmatr = []
		count = 0
		for level in mylevels:
			# print('++count = ', count)
			print(level)
			
			mywav = nico[level].data			
			xold = np.linspace(0., 1., len(mywav))
			xnew = np.linspace(0., 1., n)				
			mywav_int = np.interp(x=xnew, xp=xold, fp=mywav)			
			
			
			# print('inverse WV!')
			# gato[level] = nico[level].data
			# mywav_int = gato.reconstruct(update=False)
			
			
			
			
			
			cwtmatr.append(mywav_int**2)
			# count += 1
		cwtmatr = np.array(cwtmatr)
		print(cwtmatr)
		
		
		# sum = np.zeros(len(cwtmatr))
		# for i in range(len(cwtmatr)):
			# sum[i] = np.sum((cwtmatr[i])**2.0)
		# plt.plot(sum)
		# plt.show()
		
		# mydata = cwtmatr[18]
		# plt.plot(mydata)
		# plt.show()
		
		# mydata = cwtmatr[47]
		# plt.plot(mydata)
		# print(len(mydata))
		# plt.show()
		
		# gato['dddaaa'] = nico['dddaaa'].data
		# reco = gato.reconstruct(update=False)
		# reco = reco[0:n]
		# plt.plot(reco)
		# print(len(reco))
		# plt.show()
		


		
		fig, ax = plt.subplots()		
		
		maxxx = np.max(cwtmatr)
		minnn = np.min(cwtmatr)

		# levels_ = list(np.linspace(minnn, maxxx, num=100))
		# ax.contour(t, real_freq, cwtmatr, levels=levels_)
		# extent_ = [0, np.max(t), 500, 0]
		extent_ = [0, np.max(t), 0, 500]
		ax.contourf(cwtmatr, extent=extent_)
		# ax.set_ylim(bottom=0 , top=500000)		
		
		plt.show()
		
		mydict = {'map':cwtmatr, 'extent':extent_}		
		save_pickle(name_out, mydict)
		
		mydict = {'t':t, 'x':signal}
		save_pickle(name_out_2, mydict)
		
		
		# 'Scgr_S1_AE3_B201_db6_lvl5_mv2.pkl'
		sys.exit()
		
		# max_width = 101
		# min_width = 1
		# widths = np.arange(min_width, max_width)
		# mother_wv = 'morl'
		# real_freq = pywt.scale2frequency(mother_wv, widths)/dt
		# print(real_freq)
		# # sys.exit()		
		# cwtmatr, freqs = pywt.cwt(data=signal, scales=widths, wavelet=mother_wv, sampling_period=dt)
		
		# plt.plot(t, signal)
		# plt.show()
		
		# print(len(cwtmatr))
		# print(len(cwtmatr[0]))
		
		sum = np.zeros(len(cwtmatr))
		for i in range(len(cwtmatr)):
			sum[i] = np.sum((cwtmatr[i]))
		plt.plot(sum, '-o')
		plt.show()
		
		sys.exit()
		
		
		
		print(cwtmatr[2][2])
		cwtmatr = cwtmatr**2.0
		cwtmatr = np.absolute(cwtmatr)
		print(cwtmatr[2][2])
		cwtmatr = np.log10(cwtmatr)
		print(cwtmatr[2][2])

		y = real_freq
		x = t
		# plt.pcolormesh(t, real_freq, cwtmatr)
		# extent_ = [0, np.max(t), np.max(real_freq), np.min(real_freq)]
		# extent_ = None
		extent_ = [-1, 1, min_width, max_width]
		colormap_ = 'PRGn'
		# colormap_ = 'plasma'
		
		
		
		
		
		
		# ax.imshow(cwtmatr, extent=extent_, cmap=colormap_, aspect='auto', vmax=maxxx, vmin=minnn, interpolation='bilinear')
		# ax.set_xlim(left=-1 , right=1)
		# ax.set_ylim(bottom=1 , top=max_width)
		
		
		# plt.imshow(cwtmatr, extent=extent_, cmap=colormap_, aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max(), interpolation='bilinear')
		
		plt.show()
	
	elif config['mode'] == 'wv_packet_full':
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		
		signal = load_signal(filepath, channel=config['channel'])
		


		signal = signal[int(0*config['fs']) : int(0.1*config['fs'])]
		
		
		

		dt = 1./config['fs']		
		
		t = np.arange(len(signal))*dt
		n = len(signal)
		
		select_level = 5
		
		wavelet_mother = 'sym6'
		nico = pywt.WaveletPacket(data=signal, wavelet=wavelet_mother, maxlevel=select_level)
		gato = pywt.WaveletPacket(data=None, wavelet=wavelet_mother, maxlevel=5)
		
		# mylevels = ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']
		mylevels = [node.path for node in nico.get_level(select_level, 'freq')]
		# freq = np.array([0, 62500, 12500, 187500, 250000, 312500, 375000, 437500])
		cwtmatr = []
		count = 0
		for level in mylevels:
			# print('++count = ', count)
			print(level)
			
			mywav = nico[level].data			
			xold = np.linspace(0., 1., len(mywav))
			xnew = np.linspace(0., 1., n)				
			mywav_int = np.interp(x=xnew, xp=xold, fp=mywav)			
			
			
			# print('inverse WV!')
			# gato[level] = nico[level].data
			# mywav_int = gato.reconstruct(update=False)
			
			
			
			
			
			cwtmatr.append(mywav_int**2)
			# count += 1
		cwtmatr = np.array(cwtmatr)
		print(cwtmatr)
		
		
		


		
		fig, ax = plt.subplots()		
		
		maxxx = np.max(cwtmatr)
		minnn = np.min(cwtmatr)

		# levels_ = list(np.linspace(minnn, maxxx, num=100))
		# ax.contour(t, real_freq, cwtmatr, levels=levels_)
		# extent_ = [0, np.max(t), 500, 0]
		
		extent_ = [0, np.max(t), 0, 500]
		ax.contourf(cwtmatr, extent=extent_)
		# ax.set_ylim(bottom=0 , top=500000)		
		
		plt.show()
		
		mydict = {'map':cwtmatr, 'extent':extent_}
		save_pickle('Scgr_simulated_AE_mod_fault_frec.pkl', mydict)
		
		sys.exit()
		
		# max_width = 101
		# min_width = 1
		# widths = np.arange(min_width, max_width)
		# mother_wv = 'morl'
		# real_freq = pywt.scale2frequency(mother_wv, widths)/dt
		# print(real_freq)
		# # sys.exit()		
		# cwtmatr, freqs = pywt.cwt(data=signal, scales=widths, wavelet=mother_wv, sampling_period=dt)
		
		# plt.plot(t, signal)
		# plt.show()
		
		# print(len(cwtmatr))
		# print(len(cwtmatr[0]))
		
		sum = np.zeros(len(cwtmatr))
		for i in range(len(cwtmatr)):
			sum[i] = np.sum((cwtmatr[i]))
		plt.plot(sum, '-o')
		plt.show()
		
		sys.exit()
		
		
		
		print(cwtmatr[2][2])
		cwtmatr = cwtmatr**2.0
		cwtmatr = np.absolute(cwtmatr)
		print(cwtmatr[2][2])
		cwtmatr = np.log10(cwtmatr)
		print(cwtmatr[2][2])

		y = real_freq
		x = t
		# plt.pcolormesh(t, real_freq, cwtmatr)
		# extent_ = [0, np.max(t), np.max(real_freq), np.min(real_freq)]
		# extent_ = None
		extent_ = [-1, 1, min_width, max_width]
		colormap_ = 'PRGn'
		# colormap_ = 'plasma'
		
		
		
		
		
		
		# ax.imshow(cwtmatr, extent=extent_, cmap=colormap_, aspect='auto', vmax=maxxx, vmin=minnn, interpolation='bilinear')
		# ax.set_xlim(left=-1 , right=1)
		# ax.set_ylim(bottom=1 , top=max_width)
		
		
		# plt.imshow(cwtmatr, extent=extent_, cmap=colormap_, aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max(), interpolation='bilinear')
		
		plt.show()
	
	elif config['mode'] == 'wv_packet_energy_freq':
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		select_level = 2
		wavelet_mother = 'db6'
		
		# Channels = ['AE_1', 'AE_2', 'AE_3']
		Channels = [config['channel']]

		
		Energy = np.zeros(2**select_level)
		
		
		for channel in Channels:
			for filepath in Filepaths:
				# signal = load_signal(filepath, channel=config['channel'])
				signal = load_signal_varb(filepath, channel=channel)
				
				# signal = signal*1000./70.8
				
				# signal = signal[int(0*config['fs']) : int(10*config['fs'])]*1000./70.8
				
				# signal = signal[int(0*config['fs']) : int(10*config['fs'])]*1000./281.8
				
				
				
				# signal = signal[int(0*config['fs']) : int(10*config['fs'])]*1000./141.25
				


				dt = 1./config['fs']
				

				
				
				t = np.arange(len(signal))*dt
				n = len(signal)
				
				
				nico = pywt.WaveletPacket(data=signal, wavelet=wavelet_mother, maxlevel=select_level)
				# gato = pywt.WaveletPacket(data=None, wavelet=wavelet_mother, maxlevel=5)
				
				# mylevels = ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']
				mylevels = [node.path for node in nico.get_level(select_level, 'freq')]
				# print(mylevels)
				# sys.exit()
				# freq = np.array([0, 62500, 12500, 187500, 250000, 312500, 375000, 437500])
				cwtmatr = []
				count = 0
				for level in mylevels:
					# print('++count = ', count)
					print(level)
					
					mywav = nico[level].data			
					xold = np.linspace(0., 1., len(mywav))
					xnew = np.linspace(0., 1., n)				
					mywav_int = np.interp(x=xnew, xp=xold, fp=mywav)			
					
					
					# print('inverse WV!')
					# gato[level] = nico[level].data
					# mywav_int = gato.reconstruct(update=False)
					
					
					
					
					
					cwtmatr.append(mywav_int**2)
					# count += 1
				cwtmatr = np.array(cwtmatr)
				# print(cwtmatr)
				
				

				
				sum = np.zeros(len(cwtmatr))
				for i in range(len(cwtmatr)):
					sum[i] = np.sum((cwtmatr[i]))
				
				Energy += sum
				
				
			myfreq = np.arange(len(mylevels))*config['fs']/(2**(1+select_level))
			myfreq += config['fs']/(2**(2+select_level))
			print(myfreq)
			mypik = {'energy':Energy, 'freq':myfreq}
			save_pickle('WPD_Parameter_5_c1_db6_lvl5.pkl', mypik)
		
		# plt.plot(myfreq, Energy, '-o')
		# plt.show()
		
		
		
	
	elif config['mode'] == 'select_best':		
		if config['mypath'] == None:
			print('Select File with signal')
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()
			root.destroy()
		else:
			filepath = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
		
		x = load_signal(filepath, channel=config['channel'])
		mother_wv = 'db6'
		fs = config['fs']
		# python M_Wavelet.py --mode select_best --channel AE_0 --fs 1.e6
		levels = max_wv_level(x, mother_wv)
		levels = 5
		print('max_levels_', levels)
		int_freqs = [4.33, 8.66, 12.99, 17.32, 21.65, 26.0, 30.33, 34.67, 39.00, 43.33]		
		int_freqs2 = [312., 624., 936., 1248., 1560.]
		int_freqs2 += [307.67, 316.33, 303.34, 320.66, 294.68, 329.32, 290.35, 333.65] +  [619.67, 628.33, 615.34, 632.66, 606.68, 641.32, 602.35, 645.65] + [931.67, 940.33, 927.34, 944.66, 918.68, 953.32, 914.35, 957.65] + [1243.67, 1252.33, 1239.34, 1256.66, 1230.68, 1265.32, 1226.35, 1269.35] + [1555.67, 1564.33, 1551.34, 1568.66, 1542.68, 1577.32, 1538.35, 1581.65]		
		int_freqs2 += int_freqs
		freq_values = int_freqs2
		
		freq_range = list(np.array([3., 1600.]))
		
		
		wsignal, best_lvl, new_fs = return_best_wv_level_idx(x, fs, levels, mother_wv, freq_values, freq_range)
		print('best_lvl_idx_', best_lvl)
		
		print(cal_WV_fitness_hilbert_env_Ncomp_mpr(wsignal, freq_values, freq_range, new_fs))
		
		plt.figure(1)
		plt.plot(wsignal)
		
		plt.figure(2)
		M, f, df = mag_fft(wsignal, new_fs)
		plt.plot(f, M, 'r')
		
		plt.figure(3)
		env = hilbert_demodulation(wsignal)
		ME, f, df = mag_fft(env, new_fs)
		plt.plot(f, ME, 'g')
		
		plt.show()
		
		
		
	else:
		print('unknown mode')

	
	return

def max_wv_level(x, mother_wv):
	wv = pywt.Wavelet(mother_wv)
	wvlen = wv.dec_len
	return pywt.dwt_max_level(len(x), filter_len=wvlen)
	
def return_best_wv_level_idx(x, fs, sqr, levels, mother_wv, crit, wv_approx, int_points, freq_values=None, freq_range=None, freq_values_2=None, freq_range_2=None):
	n = len(x)
	coeffs = pywt.wavedec(x, mother_wv, level=levels, mode='periodic')
	print('LEN COEF ', len(coeffs))
	for i in range(levels+1):
		print('..', i)
		print('...', len(coeffs[i]))
	
	vec = np.zeros(levels+1)
	for i in range(levels+1):
		print('evaluate level ', i/levels)
		wsignal = coeffs[i]
		
		if int_points != True:
			wsignal = odd_to_even(wsignal)
			new_fs = downsampling_fs(x, wsignal, fs)
		
		# plt.plot(wsignal)
		# plt.show()
		
		if crit == 'mpr':
			fitness = cal_WV_fitness_hilbert_env_Ncomp_mpr(wsignal, sqr, freq_values, freq_range, new_fs)
		elif crit == 'avg_mpr':
			fitness = cal_WV_fitness_hilbert_env_AVG_mpr(wsignal, sqr, freq_values, freq_range, freq_values_2, freq_range_2, new_fs)
		elif crit == 'kurt_sen':			
			kurt = scipy.stats.kurtosis(wsignal, fisher=False)
			sen = shannon_entropy(wsignal)			
			fitness = kurt/sen
		elif crit == 'kurt':			
			kurt = scipy.stats.kurtosis(wsignal, fisher=False)
			fitness = kurt
		else:
			print('fatal error crit wavelet')
			sys.exit()

		if i == 0 and wv_approx != 'ON':
			vec[i] = -9999
		else:
			vec[i] = fitness
	
	best_level_idx = np.argmax(vec)
	print('best fitness = ', np.max(vec))
	

	outsignal = coeffs[best_level_idx]
	if int_points == True:
		xold = np.linspace(0., 1., len(outsignal))
		xnew = np.linspace(0., 1., n)
		outsignal = np.interp(x=xnew, xp=xold, fp=outsignal)
		new_fs = fs
	else:
		outsignal = odd_to_even(outsignal)
		new_fs = downsampling_fs(x, outsignal, fs)

	
	return outsignal, best_level_idx, new_fs

def return_best_inv_wv_level_idx(x, fs, sqr, levels, mother_wv, crit, wv_approx, freq_values=None, freq_range=None, freq_values_2=None, freq_range_2=None):

	coeffs = pywt.wavedec(x, mother_wv, level=levels, mode='periodic')
	print('LEN COEF ', len(coeffs))
	for i in range(levels+1):
		print('..', i)
		print('...', len(coeffs[i]))
	
	vec = np.zeros(levels+1)
	for i in range(levels+1):
		print('evaluate level ', i/levels)
		wsignal = coeffs[i]
		# wsignal = odd_to_even(wsignal)
		# new_fs = downsampling_fs(x, wsignal, fs)
		
		# plt.plot(wsignal)
		# plt.show()
		

		if crit == 'kurt_sen':			
			kurt = scipy.stats.kurtosis(wsignal, fisher=False)
			sen = shannon_entropy(wsignal)			
			fitness = kurt/sen
		else:
			print('fatal error crit wavelet')
			sys.exit()

		if i == 0 and wv_approx != 'ON':
			vec[i] = -9999
		else:
			vec[i] = fitness
	
	best_level_idx = np.argmax(vec)
	print('best fitness = ', np.max(vec))
	
	outsignal = coeffs[best_level_idx]
	
	# outsignal = odd_to_even(outsignal)
	new_fs = downsampling_fs(x, outsignal, fs)
	
	outsignal = pywt.idwt(cA=None, cD=outsignal, wavelet=mother_wv)
	# new_fs = fs
	
	

	
	return outsignal, best_level_idx, new_fs

def return_best_wv_level_idx_PACKET(x, fs, sqr, levels, mother_wv, crit, wv_approx, freq_values=None, freq_range=None, freq_values_2=None, freq_range_2=None):
	n = len(x)
	
	nico = pywt.WaveletPacket(data=x, wavelet=mother_wv, maxlevel=levels)	
	mylevels = [node.path for node in nico.get_level(levels, 'freq')]

	
	vec = np.zeros(len(mylevels))
	count = 0
	for lvl in mylevels:
		print('evaluate level ', lvl)
		wsignal = nico[lvl].data
		
		# print(len(wsignal))
		# a = input('pause----')
		
		xold = np.linspace(0., 1., len(wsignal))
		xnew = np.linspace(0., 1., n)				
		wsignal_int = np.interp(x=xnew, xp=xold, fp=wsignal)	
		
		
		if crit == 'mpr':
			fitness = cal_WV_fitness_hilbert_env_Ncomp_mpr(wsignal_int, sqr, freq_values, freq_range, fs)
		elif crit == 'avg_mpr':
			fitness = cal_WV_fitness_hilbert_env_AVG_mpr(wsignal_int, sqr, freq_values, freq_range, freq_values_2, freq_range_2, new_fs)
		elif crit == 'kurt_sen':			
			kurt = scipy.stats.kurtosis(wsignal_int, fisher=False)
			sen = shannon_entropy(wsignal_int)			
			fitness = kurt/sen
		elif crit == 'kurt':			
			kurt = scipy.stats.kurtosis(wsignal_int, fisher=False)
			fitness = kurt
		else:
			print('fatal error crit wavelet')
			sys.exit()

		vec[count] = fitness
		count += 1
	
	best_level_idx = np.argmax(vec)
	print('best fitness = ', np.max(vec))
	
	outsignal = nico[mylevels[best_level_idx]].data
	
	xold = np.linspace(0., 1., len(outsignal))
	xnew = np.linspace(0., 1., n)				
	outsignal_int = np.interp(x=xnew, xp=xold, fp=outsignal)

	new_fs = fs
	return outsignal_int, best_level_idx, new_fs

def return_iwv_PACKET_fix_levels(x, fs, max_level, mother_wv, idx_levels):
	n = len(x)
	
	nico = pywt.WaveletPacket(data=x, wavelet=mother_wv, maxlevel=max_level)	
	mylevels = [node.path for node in nico.get_level(max_level, 'freq')]
	
	
	print('inverse WV!')
	gato = pywt.WaveletPacket(data=None, wavelet=mother_wv, maxlevel=max_level)
	
	for idx in idx_levels:	
		gato[mylevels[idx]] = nico[mylevels[idx]].data
	outsignal = gato.reconstruct(update=False)

	return outsignal

def return_wv_PACKET_one_level(x, fs, max_level, mother_wv, idx_level):
	idx_level_one = idx_level[0]
	n = len(x)
	
	nico = pywt.WaveletPacket(data=x, wavelet=mother_wv, maxlevel=max_level)	
	mylevels = [node.path for node in nico.get_level(max_level, 'freq')]
	# freq
	
	outsignal = nico[mylevels[idx_level_one]].data
	
	# gato = pywt.WaveletPacket(data=None, wavelet=mother_wv, maxlevel=max_level)
	# gato[mylevels[idx_level_one]] = nico[mylevels[idx_level_one]].data
	# outsignal = gato.reconstruct(update=False)
	
	return outsignal

def odd_to_even(x):
	if len(x) % 2 != 0:			
		x = x[1:]
	return x

def downsampling_fs(x, newsignal, fs):
	red_fact = len(x)/len(newsignal)
	new_fs = fs / red_fact
	return new_fs

def cal_WV_fitness_hilbert_env_Ncomp_mpr(x, sqr, freq_values, freq_range, fs):

	# plt.plot(x)
	# plt.show()
	if sqr == 'ON':
		print('Squared wavelet!!!')
		x = x**2.0
	x_env = hilbert_demodulation(x)
	magENV, f, df = mag_fft(x_env, fs)
	print(len(magENV))
	print(df)
	mag_freq_value = 0.
	for freq_value in freq_values:
		mag_freq_value += amp_component_zone(X=magENV, df=df, freq=freq_value, tol=4.0)
	
	avg_freq_range = avg_in_band(magENV, df, low=freq_range[0], high=freq_range[1])
	fitness = 20*np.log10((mag_freq_value-avg_freq_range)/avg_freq_range)


	return fitness

def cal_WV_fitness_hilbert_env_AVG_mpr(x, sqr, freq_values, freq_range, freq_values_2, freq_range_2, fs):

	# plt.plot(x)
	# plt.show()
	if sqr == 'ON':
		print('Squared wavelet!!!')
		x = x**2.0
	x_env = hilbert_demodulation(x)
	magENV, f, df = mag_fft(x_env, fs)
	print(len(magENV))
	print(df)
	mag_freq_value = 0.
	mag_freq_value_2 = 0.
	for freq_value in freq_values:
		mag_freq_value += amp_component_zone(X=magENV, df=df, freq=freq_value, tol=2.0)
	for freq_value_2 in freq_values_2:
		mag_freq_value_2 += amp_component_zone(X=magENV, df=df, freq=freq_value_2, tol=2.0)
	
	avg_freq_range = avg_in_band(magENV, df, low=freq_range[0], high=freq_range[1])
	
	fitness = 20*np.log10((mag_freq_value-len(freq_values)*avg_freq_range)/avg_freq_range)
	
	avg_freq_range_2 = avg_in_band(magENV, df, low=freq_range_2[0], high=freq_range_2[1])
	
	fitness_2 = 20*np.log10((mag_freq_value_2-len(freq_values_2)*avg_freq_range_2)/avg_freq_range_2)
	
	fitness_avg = (fitness + fitness_2)/2.


	return fitness_avg

def read_parser(argv, Inputs, InputsOpt_Defaults):
	try:
		Inputs_opt = [key for key in InputsOpt_Defaults]
		Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
		parser = ArgumentParser()
		for element in (Inputs + Inputs_opt):
			print(element)
			if element == 'files' or element == 'features_array':
				parser.add_argument('--' + element, nargs='+')
			else:
				parser.add_argument('--' + element, nargs='?')
		print(parser.parse_args())
		args = parser.parse_args()
		
	except:
		# args = argv
		arguments = [element for element in argv if element[0:2] == '--']
		values = [element for element in argv if element[0:2] != '--']

		# from argparse import ArgumentParser
		# from ArgumentParser import Namespace
		parser = ArgumentParser()
		for element in arguments:
			parser.add_argument(element)

		args = parser.parse_args(argv)

		# print(test)
		# sys.exit()
		
	config = {}	
		
	for element in Inputs:
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Required:', element)

	for element, value in zip(Inputs_opt, Defaults):
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Default ' + element + ' = ', value)
			config[element] = value
	
	#Type conversion to float

	
	
	# config['clusters'] = int(config['clusters'])
	config['n_mov_avg'] = int(config['n_mov_avg'])
	
	
	
	

	
	config['train'] = float(config['train'])
	config['fs'] = float(config['fs'])
	

	
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config
	


	


			

if __name__ == '__main__':
	main(sys.argv)


