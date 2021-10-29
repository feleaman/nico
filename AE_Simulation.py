# import os
from os import listdir
import matplotlib.pyplot as plt

# from Kurtogram3 import Fast_Kurtogram_filters
from os.path import join, isdir, basename, dirname, isfile
import sys
from os import chdir
plt.rcParams['savefig.directory'] = chdir(dirname('D:'))

from tkinter import filedialog
from tkinter import Tk
sys.path.insert(0, './lib') #to open user-defined functions
# from m_open_extension import read_pickle
from argparse import ArgumentParser
import numpy as np
# import pandas as pd
from m_open_extension import *
from m_det_features import *
from Genetic_Filter import *
#from m_pattern import *

from M_Wavelet import *
from m_plots import *

# from THR_Burst_Detection import full_thr_burst_detector
# from THR_Burst_Detection import read_threshold
# from THR_Burst_Detection import plot_burst_rev

Inputs = ['mode']
InputsOpt_Defaults = {'output':'plot'}

from m_fft import mag_fft
from m_denois import *
import pandas as pd
from scipy import signal
# from datetime import datetime

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	
	if config['mode'] == 'normal_modulation':
		print('caca')
		fs = 1000000.
		dt = 1./fs
		tr = 1.
		thr = 0.5
		alpha_bursts = 130.
		dura_bursts = 0.001
		frec_bursts = 92000.
		amort_bursts = 7887.
		std_add_noise = 0.1
		# std_phase_noise = 0.0
		std_mult_noise = 0.0001
		max_rint = 2
		
		Noise = np.random.RandomState(seed=1)		
		t = np.arange(0, tr, dt)
		# phase_noise = 0.5*Noise.normal(1,std_phase_noise,len(t))
		# phase_noise = np.sin(2 * np.pi * t)
		
		
		# xc = signal.square(2*np.pi*alpha_bursts*t, duty=phase_noise)
		# rect = signal.square(2*np.pi*alpha_bursts*t, duty=phase_noise)
		
		xc = signal.square(2*np.pi*alpha_bursts*t)
		rect = signal.square(2*np.pi*alpha_bursts*t)
		
		# plt.plot(xc)
		# plt.show()
		
		n = len(xc)
		add_noise = Noise.normal(0,std_add_noise,n)
		mult_noise = Noise.normal(1,std_mult_noise,n)
		
		
		index = []
		for i in range(n-1):
			if xc[i] < thr and xc[i+1] >= thr:
				index.append(i)
		index = np.array(index)
		
		nidx = len(index)
		# rand_int = Noise.randint(0,max_rint,nidx)
		
		
		tb = np.arange(0, dura_bursts, dt)
		burst = np.sin(2*np.pi*frec_bursts*tb)*np.exp(-amort_bursts*tb)
		nb = len(burst)
		# print(rand_int)
		
		# count = 0
		for idx in index:
			# print(count)
			xc[idx:idx+nb] += burst
			# if count != nidx -1:
				# xc[idx + rand_int[idx] : idx + rand_int[idx] + nb] += burst[0:]
			# else:
				# xc[idx:idx+nb] += burst
			# count += 1
			
		x = xc - rect
		
		
		
		# add_noise = np.random.normal(0,std_add_noise,n)
		# mult_noise = np.random.normal(1,std_mult_noise,n)
		
		
		x += add_noise
		x = x*mult_noise
		
		X, f, df = mag_fft(x, fs)
		
		# print(signal_rms(x))
		plt.plot(t, x)
		plt.show()
		
		plt.plot(f, X)
		plt.show()
	
	elif config['mode'] == 'normal_modulation_2':
		print('caca')
		fs = 1000000.
		dt = 1./fs
		tr = 2.
		thr = 0.5
		alpha_bursts = 300.
		dura_bursts = 0.001
		frec_bursts = 92000.
		amort_bursts = 7887.
		planets = 3.
		carrier = 5.
		
		std_add_noise = 0.2
		std_phase_noise = 0.01
		std_mult_noise = 0.001
		
		# std_add_noise = 0.1
		# std_phase_noise = 0.01
		# std_mult_noise = 0.001
		
		
		max_rint = 2
		
		Noise = np.random.RandomState(seed=1)		
		t = np.arange(0, tr, dt)
		phase_noise = 0.5*Noise.normal(1,std_phase_noise,len(t))
		# phase_noise = Noise.uniform(0.5,0.5+0.05,len(t))
		# plt.plot(phase_noise, 'k')
		# plt.show()
		
		# phase_noise = np.sin(2 * np.pi * t)
		
		
		xc = signal.square(2*np.pi*alpha_bursts*t, duty=phase_noise)
		rect = signal.square(2*np.pi*alpha_bursts*t, duty=phase_noise)
		
		# xc = signal.square(2*np.pi*alpha_bursts*t)
		# rect = signal.square(2*np.pi*alpha_bursts*t)
		
		# plt.plot(xc)
		# plt.show()
		
		n = len(xc)
		add_noise = Noise.normal(0,std_add_noise,n)
		mult_noise = Noise.normal(1,std_mult_noise,n)
		
		
		index = []
		for i in range(n-1):
			if xc[i] < thr and xc[i+1] >= thr:
				index.append(i)
		index = np.array(index)
		
		nidx = len(index)
		# rand_int = Noise.randint(0,max_rint,nidx)
		
		
		tb = np.arange(0, dura_bursts, dt)
		burst = np.sin(2*np.pi*frec_bursts*tb)*np.exp(-amort_bursts*tb)
		nb = len(burst)
		# print(rand_int)
		
		# count = 0
		for idx in index:
			# print(count)
			xc[idx:idx+nb] += burst
			# if count != nidx -1:
				# xc[idx + rand_int[idx] : idx + rand_int[idx] + nb] += burst[0:]
			# else:
				# xc[idx:idx+nb] += burst
			# count += 1
			
		x = xc - rect
		
		
		
		# add_noise = np.random.normal(0,std_add_noise,n)
		# mult_noise = np.random.normal(1,std_mult_noise,n)
		
		
		x += add_noise
		x = x*mult_noise
		
		
		mod = np.ones(n) + 0.3*np.cos(2*np.pi*planets*carrier*t)
		x = x*mod
		
		
		X, f, df = mag_fft(x, fs)
		
		x_env = hilbert_demodulation(x)
		X_env, f_env, df_env = mag_fft(x_env, fs)
		
		
		
		
		# print(signal_rms(x))
		plt.plot(t, x)
		plt.show()
		
		# plt.plot(f, X)
		# plt.show()
		
		plt.plot(f_env, X_env, 'g-o')
		plt.show()
	
	elif config['mode'] == 'gearbox_simulator':
		#+++++++++Simulation Data
		fs = 1000000.
		dt = 1./fs
		tr = 1.
		thr = 0.5
		alpha_bursts = 300.
		dura_bursts = 0.001
		frec_bursts = 92000.
		amort_bursts = 7887.
		planets = 3.
		carrier = 5.
		
		std_add_noise = 0.2 #0.2
		std_phase_noise = 0.03 #0.02
		std_mult_noise = 0.001 #0.001
		
		amp_mod_1 = 0.35 #0.3
		amp_mod_2 = 0.95 #0.5
		
		#+++++++++++Time Array
		t = np.arange(0, tr, dt)
		n = len(t)
		
		#Noise Generator
		Noise = np.random.RandomState(seed=1)		
		phase_noise = 0.5*Noise.normal(1,std_phase_noise,n)
		add_noise = Noise.normal(0,std_add_noise,n)
		mult_noise = Noise.normal(1,std_mult_noise,n)
		
		
		#+++++++++++++Parent Signals
		#Train Pulses
		xc = signal.square(2*np.pi*alpha_bursts*t, duty=phase_noise)
		rect = signal.square(2*np.pi*alpha_bursts*t, duty=phase_noise)		
		
		index = []
		for i in range(n-1):
			if xc[i] < thr and xc[i+1] >= thr:
				index.append(i)
		index = np.array(index)		
		nidx = len(index)		
		
		#Bursts
		tb = np.arange(0, dura_bursts, dt)
		burst = np.sin(2*np.pi*frec_bursts*tb)*np.exp(-amort_bursts*tb)
		nb = len(burst)
		
		for idx in index:
			xc[idx:idx+nb] += burst	
			
		#No modulated signal
		x = xc - rect
		plt.plot(x, 'k')
		plt.show()
		
		x += add_noise
		x = x*mult_noise
		
		
		#+++++++++++++Operational Modulations
		#Distance Pitch-Sensor
		mod_1 = np.ones(n) + amp_mod_1*np.cos(2*np.pi*planets*carrier*t)
		x = x*mod_1
		
		#Local Fault Ring Gear
		mod_2 = np.ones(n) + amp_mod_2*signal.square(2*np.pi*planets*carrier*t, duty=0.0075)
		
		# mod_2 = np.ones(n) + amp_mod_2*signal.sawtooth(2*np.pi*planets*carrier*t, width=0.25)
		
		plt.plot(mod_2, 'k')
		plt.show()
		
		x = x*mod_2
		

		
		#+++++++++++++Signal Processing
		#Spectrum
		X, f, df = mag_fft(x, fs)
		
		#Envelope
		x_env = hilbert_demodulation(x)
		
		#Envelope spectrum
		X_env, f_env, df_env = mag_fft(x_env, fs)
		
		
		
		#+++++++++++++Plots
		plt.plot(t, x)
		plt.show()

		
		plt.plot(f_env, X_env, 'g')
		plt.show()
	
	
	elif config['mode'] == 'gearbox_simulator_2':
		#+++++++++Simulation Data
		fs = 1000000.
		dt = 1./fs
		tr = 0.25
		thr = 0.5
		alpha_bursts = 300.
		dura_bursts = 0.001
		frec_bursts = 92000.
		amort_bursts = 7887.
		planets = 3.
		carrier = 5.
		
		std_add_noise = 0.15 #0.15
		std_phase_noise = 0.0005 #0.0005
		std_mult_noise = 0.001 #0.001
		
		amp_mod_1 = 0.4 #0.4 sinus
		amp_mod_2 = 0.4 #0.4 local
		
		#+++++++++++Time Array
		t = np.arange(0, tr, dt)
		n = len(t)
		
		#Noise Generator
		Noise = np.random.RandomState(seed=1)		
		phase_noise = 0.5*Noise.normal(1,std_phase_noise,n)
		add_noise = Noise.normal(0,std_add_noise,n)
		mult_noise = Noise.normal(1,std_mult_noise,n)
		
		
		#+++++++++++++Parent Signals
		#Train Pulses
		xc = signal.square(2*np.pi*alpha_bursts*t, duty=phase_noise)
		rect = signal.square(2*np.pi*alpha_bursts*t, duty=phase_noise)		
		
		index = []
		for i in range(n-1):
			if xc[i] < thr and xc[i+1] >= thr:
				index.append(i)
		index = np.array(index)		
		nidx = len(index)		
		
		#Bursts
		tb = np.arange(0, dura_bursts, dt)
		burst = np.sin(2*np.pi*frec_bursts*tb)*np.exp(-amort_bursts*tb)
		nb = len(burst)
		
		for idx in index:
			xc[idx:idx+nb] += burst
			# if idx%15 != 0:
				# xc[idx:idx+nb] += burst
			# else:			
				# xc[idx:idx+nb] += burst*10
			
		#No modulated signal
		x = xc - rect
		
		
		#+++++++++++++Operational Modulations
		# #Distance Pitch-Sensor Sinusoidal 
		mod_s = np.ones(n) + amp_mod_1*np.cos(2*np.pi*planets*carrier*t)
		
		#Local Fault Ring Gear Pulse
		mod_p = np.ones(n) + amp_mod_2*signal.square(2*np.pi*planets*carrier*t, duty=0.0095)	

		
		x = x*mod_p
		add_noise = add_noise*mod_p
		
		# x = x*mod_s
		add_noise = add_noise*mod_s

			
		#+++++++++++++Noise
		x += add_noise
		x = x*mult_noise
		
		
		# plt.plot(x)
		# plt.show()
		
		#+++++++++++++Special Burst
		thr = 1.
		index = []
		for i in range(n-1):
			if x[i] < thr and x[i+1] >= thr:
				print('special burst!')
				index.append(i)
		index = np.array(index)		
		nidx = len(index)		
		
		#Bursts
		tb = np.arange(0, dura_bursts, dt)
		burst = np.sin(2*np.pi*frec_bursts*2*tb+np.pi/8)*np.exp(-amort_bursts*1.5*tb)
		nb = len(burst)
		# print('!!', nb)
		# print(len(x[100-42:100-42+nb]))
		
		
		# print(type(x[25-42:25-42+nb]))
		# print(type(burst))
		# a = x[25-42:25-42+nb] + x[25-42:25-42+nb]
		# x[1:11] = np.arange(10)		
		# sys.exit()
		
		for idx in index:
			# print(idx)
			# print(len(x[idx-42:idx-42+nb]))
			# print(len(1.3*burst))
			# aaa = x[idx-42:idx-42+nb] + 1.3*burst	
			x[idx-42:idx-42+nb] += 1.3*burst
			# x[idx-42:idx-42+nb] = aaa

		plt.plot(t, x, 'r')
		plt.show()
		
		#+++++++++++++Signal Processing
		#Spectrum
		X, f, df = mag_fft(x, fs)
		X = 20*np.log10(X/1.e-6)
		# plt.plot(X)
		# plt.show()
		#Envelope
		x_env = hilbert_demodulation(x)
		
		#Envelope spectrum
		X_env, f_env, df_env = mag_fft(x_env, fs)
		X_env = 20*np.log10(X_env/1.e-6)
		
		# # save_pickle('simulated_AE_mod_fault_frec.pkl', x)
		# # x = list(x)
		# # x = np.array(x)
		# mydict = {config['channel']:x}
		# scipy.io.savemat(filename[:-5] + '.mat', mydict)
		# print('Signal was saved as .mat. Analysis finalizes')
		# sys.exit()
		
		#+++++++++++++Plots
		name = 'SigProcessing_FftEnv_FftEnv_Mod_OnlyNoise'
		path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\ChapterX_SigProcessing\\03_Figures\\'
		path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_SigProcessing\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		style = {'xlabel':['Frequency [Hz]', 'Frequency [Hz]'], 'ylabel':['Magnitude [dB$_{AE}$]', 'Magnitude [dB$_{AE}$]'], 'legend':None, 'title':['Envelope spectrum', 'Envelope spectrum'], 'customxlabels':None, 'xlim':[[0.,50], [200,400]], 'ylim':[[20, 120.], [20, 120.]], 'color':[None], 'loc_legend':'upper left', 'legend_line':'OFF', 'vlines':None, 'range_lines':None, 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b}
		
		#+++Envelope spectrum
		# 'xlabel':['Frequency [Hz]', 'Frequency [Hz]'], 'ylabel':['Magnitude [dB$_{AE}$]', 'Magnitude [dB$_{AE}$]'], 'legend':None, 'title':['Envelope spectrum', 'Envelope spectrum'], 'customxlabels':None, 'xlim':[[0.,50], [200,400]], 'ylim':None
		
		#+++Wfm?
		# 'xlabel':['Time [s]', 'Time [s]'], 'ylabel':['Amplitude [V]', 'Amplitude [V]'], 'legend':None, 'title':['Waveform', 'Envelope'], 'customxlabels':None, 'xlim':[[0.,0.25], [0.,0.25]], 'ylim':[[-3,3], [0,3]]
		
		#+++Wfm-Env Zoom
		# 'xlabel':['Time [ms]', 'Time [ms]'], 'ylabel':['Amplitude [V]', 'Amplitude [V]'], 'legend':None, 'title':['Waveform', 'Envelope'], 'customxlabels':None, 'xlim':[[66.5, 67.0], [66.5, 67.0]], 'ylim':[[-3,3], [0,3]]
		
		# +++Spectrum		
		# ['xlabel':['Time [s]', 'Frequency [kHz]'], 'ylabel':['Amplitude [V]', 'Magnitude [dB$_{AE}$]'], 'legend':None, 'title':['Waveform', 'Spectrum'], 'customxlabels':None, 'xlim':[[0.,0.25], [0.,500]], 'ylim':[[-3,3], [0,100]]
		

		data = {'x':[f_env, f_env], 'y':[X_env, X_env]}
		# data = {'x':[t, t], 'y':[x, x_env]}
		# data = {'x':[t*1000, t*1000], 'y':[x, x_env]}
		# data = {'x':[t, f/1000.], 'y':[x, X]}
		
		plot2_thesis_new(data, style)
		
		
		# plt.plot(t, x)
		# plt.show()

		
		# plt.plot(f_env, X_env, 'g')
		# plt.show()
	
	
	
	elif config['mode'] == 'only_plot':
		tr = 1
		dt = 1./1000000.
		fs = 1/dt
		#+++++++++++Time Array
		t = np.arange(0, tr, dt)
		
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		
		x = load_signal(filepath, channel='none')

		plt.plot(t, x, 'r')
		plt.show()
		
		#+++++++++++++Signal Processing
		#Spectrum
		X, f, df = mag_fft(x, fs)
		X = 20*np.log10(X/1.e-6)
		# plt.plot(X)
		# plt.show()
		#Envelope
		x_env = hilbert_demodulation(x)
		
		#Envelope spectrum
		X_env, f_env, df_env = mag_fft(x_env, fs)
		X_env = 20*np.log10(X_env/1.e-6)
		
		# # save_pickle('simulated_AE_mod_fault_frec.pkl', x)
		# # x = list(x)
		# # x = np.array(x)
		# mydict = {config['channel']:x}
		# scipy.io.savemat(filename[:-5] + '.mat', mydict)
		# print('Signal was saved as .mat. Analysis finalizes')
		# sys.exit()
		
		#+++++++++++++Plots
		name = 'SigProcessing_Wfm_Env_Mod_Zoom_Fault'
		path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\ChapterX_SigProcessing\\03_Figures\\'
		path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_SigProcessing\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		style = {'xlabel':['Time [ms]', 'Time [ms]'], 'ylabel':['Amplitude [V]', 'Amplitude [V]'], 'legend':None, 'title':['Waveform', 'Envelope'], 'customxlabels':None, 'xlim':[[66.5*2, 67.0*2], [66.5*2, 67.0*2]], 'ylim':[[-4,4], [0,4]], 'color':[None], 'loc_legend':'upper left', 'legend_line':'OFF', 'vlines':None, 'range_lines':None, 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b}
		
		#+++Envelope spectrum
		# 'xlabel':['Frequency [Hz]', 'Frequency [Hz]'], 'ylabel':['Magnitude [dB$_{AE}$]', 'Magnitude [dB$_{AE}$]'], 'legend':None, 'title':['Envelope spectrum', 'Envelope spectrum'], 'customxlabels':None, 'xlim':[[0.,50], [200,400]], 'ylim':None
		
		#+++Wfm?
		# 'xlabel':['Time [s]', 'Time [s]'], 'ylabel':['Amplitude [V]', 'Amplitude [V]'], 'legend':None, 'title':['Waveform', 'Envelope'], 'customxlabels':None, 'xlim':[[0.,0.25], [0.,0.25]], 'ylim':[[-3,3], [0,3]]
		
		#+++Wfm-Env Zoom
		# 'xlabel':['Time [ms]', 'Time [ms]'], 'ylabel':['Amplitude [V]', 'Amplitude [V]'], 'legend':None, 'title':['Waveform', 'Envelope'], 'customxlabels':None, 'xlim':[[66.5, 67.0], [66.5, 67.0]], 'ylim':[[-3,3], [0,3]]
		
		# +++Spectrum		
		# ['xlabel':['Time [s]', 'Frequency [kHz]'], 'ylabel':['Amplitude [V]', 'Magnitude [dB$_{AE}$]'], 'legend':None, 'title':['Waveform', 'Spectrum'], 'customxlabels':None, 'xlim':[[0.,0.25], [0.,500]], 'ylim':[[-3,3], [0,100]]
		

		# data = {'x':[f_env, f_env], 'y':[X_env, X_env]}
		# data = {'x':[t, t], 'y':[x, x_env]}
		data = {'x':[t*1000, t*1000], 'y':[x, x_env]}
		# data = {'x':[t, f/1000.], 'y':[x, X]}
		
		plot2_thesis_new(data, style)
		
		
		# plt.plot(t, x)
		# plt.show()

		
		# plt.plot(f_env, X_env, 'g')
		# plt.show()
	
	elif config['mode'] == 'emd_simulation':
		print('Select IMFs')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		
		#+++++++++Simulation Signals
		fs = 1000000.
		Signals = []
		Spectra = []
		Names = []
		for filepath in Filepaths:
			print(filepath)
			Names.append('IMF ' + os.path.basename(filepath)[1])
			x = load_signal(filepath, channel=None)
			fft, f, df = mag_fft(x, fs)
			Signals.append(x)
			fft = 20*np.log10(fft/1.e-6) #dbAE
			Spectra.append(fft)
		n = len(Signals[0])		
		
		dt = 1./fs
		tr = n / fs
		Names[0] = 'Signal'
		print(Names)
		#+++++++++++Time Array
		t = np.arange(0, tr, dt)
		
		
		
		
		#+++++++++++++Plots
		name = 'SigProcessing_Wfm_Burst_EMD_Mod_Fault'
		path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\ChapterX_SigProcessing\\03_Figures\\'
		path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_SigProcessing\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		ymax_ = -3
		ymin_ = 3
		style = {'xlabel':'Time [ms]', 'ylabel':'[V]', 'legend':None, 'title':Names, 'customxlabels':None, 'xlim':[66.5, 67], 'ylim':[[ymin_,ymax_], [ymin_,ymax_], [ymin_,ymax_], [ymin_,ymax_], [ymin_,ymax_], [ymin_,ymax_]], 'color':[None], 'loc_legend':'upper left', 'legend_line':'OFF', 'vlines':None, 'range_lines':None, 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b}
		# [0.05,0.08]
		# 'xlabel':'Time [ms]'
		# 'Amplitude [V]'
		# [[0,100], [0,100], [0,100], [0,100], [0,100], [0,100]]
		# 'Frequency [kHz]'
		# '[dB$_{AE}$]'
		
		# data = {'x':[t, t, t, t, t, t], 'y':[Signals[0], Signals[1], Signals[2], Signals[3], Signals[4], Signals[5]]}
		
		data = {'x':[t*1000, t*1000, t*1000, t*1000, t*1000, t*1000], 'y':[Signals[0], Signals[1], Signals[2], Signals[3], Signals[4], Signals[5]]}
		
		# data = {'x':[f/1000., f/1000., f/1000., f/1000., f/1000., f/1000.], 'y':[Spectra[0], Spectra[1], Spectra[2], Spectra[3], Spectra[4], Spectra[5]]}
		
		plot6_thesis_new_emd(data, style)
		
		
		# plt.plot(t, x)
		# plt.show()

		
		# plt.plot(f_env, X_env, 'g')
		# plt.show()

	else:
		print('unknown mode')
		sys.exit()

		
	return






def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	parser = ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'no_element':
			parser.add_argument('--' + element, nargs='+')
		else:
			parser.add_argument('--' + element, nargs='?')
	
	args = parser.parse_args()
	config = {}
	for element in Inputs:
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Required:', element)
			sys.exit()

	for element, value in zip(Inputs_opt, Defaults):
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Default ' + element + ' = ', value)
			config[element] = value
	
	#Type conversion to float
	
	# config['fs'] = float(config['fs'])
	# config['n_batches'] = int(config['n_batches'])
	# config['level'] = int(config['level'])
	# config['db'] = int(config['db'])
	# config['divisions'] = int(config['divisions'])
	
	# config['parents'] = int(config['parents'])
	# config['clusters'] = int(config['clusters'])
	# config['generations'] = int(config['generations'])
	# config['mutation'] = float(config['mutation'])
	# config['times_std'] = float(config['times_std'])
	
	# if config['range'] != None:
		# config['range'][0] = float(config['range'][0])
		# config['range'][1] = float(config['range'][1])
	
	# if config['freq_lp'] != 'OFF':
		# config['freq_lp'] = float(config['freq_lp'])
	
	# if config['freq_hp'] != 'OFF':
		# config['freq_hp'] = float(config['freq_hp'])
	
	# if config['db_out'] != 'OFF':
		# config['db_out'] = int(config['db_out'])
	
	# if config['idx_levels'] != None:
		# for i in range(len(config['idx_levels'])):
			# config['idx_levels'][i] = int(config['idx_levels'][i])
	
	
	# Variable conversion
	return config


	
if __name__ == '__main__':
	main(sys.argv)
