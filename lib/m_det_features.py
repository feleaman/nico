import numpy as np
from scipy.integrate import odeint
from scipy import signal
from scipy import stats
import scipy
import sys
# from THR_Burst_Detection import full_thr_burst_detector
# from THR_Burst_Detection import read_threshold
# from THR_Burst_Detection import plot_burst_rev
from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *
# from pycorrelate import ucorrelate



def max_norm_correlation(signal1, signal2):
	correlation = np.correlate(signal1/(np.sum(signal1**2))**0.5, signal2/(np.sum(signal2**2))**0.5, mode='same')
	return np.max(correlation)
	
def max_norm_correlation_lag(signal1, signal2, lag):
	array_corr = []
	array_lags = list(np.arange(-lag, lag, 1))
	for it_lag in array_lags:
		# signalA = np.array(list(signal1[it_lag:]) + list(signal1[0:it_lag]))
		signalA = signal1
		signalB = np.array(list(signal2[it_lag:]) + list(signal2[0:it_lag]))
		
		correlation = np.correlate(signalA/(np.sum(signalA**2))**0.5, signalB/(np.sum(signalB**2))**0.5, mode='valid')
		
		array_corr.append(correlation)
	return np.max(np.array(array_corr))

def max_norm_correlation_lag_2(signal1, signal2, lag):
	array_corr = []
	array_lags = list(np.arange(-lag, lag+1, 1))
	for it_lag in array_lags:
		# signalA = np.array(list(signal1[it_lag:]) + list(signal1[0:it_lag]))
		signalA = signal1
		signalB = np.array(list(signal2[it_lag:]) + list(signal2[0:it_lag]))
		
		correlation = np.correlate(signalA/(np.sum(signalA**2))**0.5, signalB/(np.sum(signalB**2))**0.5, mode='valid')
		
		array_corr.append(correlation)
	return np.max(np.array(array_corr))

def features_master(name, x=None, dt=None, magX=None, df=None, difX=None, fs=None, envdifX=None, threshold=None, t_window=None):
	if name == 'RMS_WFM_0':
		value = signal_rms(x=x)
	
	elif name == 'KURT_WFM_0':
		value = stats.kurtosis(x, fisher=True)
	
	elif name == 'LP6_WFM_0':
		value = lout_featp6(x=x)
	
	elif name == 'LP7_WFM_0':
		value = lout_featp7(x=x)
	
	elif name == 'LP16_WFM_0':
		value = lout_featp16(x=x, dp=dt)
	elif name == 'LP17_WFM_0':
		value = lout_featp17(x=x, dp=dt)
	
	elif name == 'LP21_WFM_0':
		value = lout_featp21(x=x, dp=dt)
	
	elif name == 'LP24_WFM_0':
		value = lout_featp24(x=x, dp=dt)
	
	elif name == 'LP16_FFT_0':
		value = lout_featp16(x=magX, dp=df)
	
	elif name == 'LP17_FFT_0':
		value = lout_featp17(x=magX, dp=df)
	
	elif name == 'LP21_FFT_0':
		value = lout_featp21(x=magX, dp=df)
	
	elif name == 'LP24_FFT_0':
		value = lout_featp24(x=magX, dp=df)
		
	elif name == 'NBU_WFM_0':
		value = id_burst_threshold(x=x, fs=fs, threshold=threshold, t_window=t_window)
		value = value[0]
	
	# elif name == 'NBU_WFM_1':
		# value = id_burst_threshold(x, fs, threshold, t_window)
		# value = value[0]
	
	elif name == 'NBU_DIF_0':
		value = id_burst_threshold(x=difX, fs=fs, threshold=threshold, t_window=t_window)
		value = value[0]
	# elif name == 'NBU_DIF_1':
		# value = id_burst_threshold(x=difx, fs=fs, threshold=threshold, t_window=t_window)
	
	elif name == 'NBU_DIF_ENV_0':
		value = id_burst_threshold(x=envdifX, fs=fs, threshold=threshold, t_window=t_window)
		value = value[0]
	# elif name == 'NBU_DIF_ENV_1':
		# value = id_burst_threshold(x=difdemodx, fs=fs, threshold=threshold, t_window=t_window)
	
	# elif name == 'NBU_ENV_STF_BIN_0':
		# value = ()
	
	# elif name == 'NBU_DIF_ENV_STF_BIN_0':
		# value = ()		
	
	else:
		print('error name feature')
		sys.exit()

	return value

def shannon_entropy(signal):
	value_rms = signal_rms(signal)
	n = len(signal)
	px = ((value_rms)**2.0)*n
	ent = 0.
	for i in range(n):
		# print('!!!!!!!!!!!!!!!!!!!!!!!!')
		# print('px', px)
		# print('signal2', (signal[i]**2.0))
		if px != 0 and signal[i] !=0:
			ent += ((signal[i]**2.0)/px)*np.log2((signal[i]**2.0)/px)

	ent = - ent	
	return ent

def stat_mean(x): 
	return np.mean(x)

def stat_std(x): 
	return np.std(x)

def stat_var(x): 
	return np.var(x)

def signal_rms(x): 
	sum = 0
	for i in range(len(x)):
		sum = sum + x[i]**2.0
	sum = sum/len(x)
	sum = sum**0.5
	return sum



def lout_featp6(x): 
	mean = stat_mean(x)
	std = stat_std(x)
	sum = 0	
	for i in range(len(x)):
		sum = sum + (x[i]-mean)**3.0
	sum = sum/((len(x)-1)*(std**3.0))
	return sum

def lout_featp7(x):
	mean = stat_mean(x)
	std = stat_std(x)
	sum = 0	
	for i in range(len(x)):
		sum = sum + (x[i]-mean)**4.0
	sum = sum/((len(x)-1)*(std**4.0))
	return sum

def lout_featp16(x, dp):
	sum1 = 0.0
	for i in range(len(x)):
		sum1 = sum1 + x[i]*i*dp
	sum2 = 0.0
	for i in range(len(x)):
		sum2 = sum2 + x[i]
	sum = sum1/sum2	
	return sum

def lout_featp17(x, dp, lp16):
	p16 = lp16
	sum = 0.0
	for i in range(len(x)):
		sum = sum + x[i]*(i*dp - p16)**2.0	
	sum = sum/len(x)
	sum = sum**0.5
	return sum

def lout_featp21(x, dp, lp16, lp17):

	return lp17/lp16

def lout_featp24(x, dp, lp16, lp17):
	p16 = lp16
	p17 = lp17
	sum = 0.0
	for i in range(len(x)):
		try:
			sum = sum + x[i]*(np.absolute(i*dp - p16))**0.5 #podria salir raiz negativa!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		except:
			print('error handled')
	sum = sum/(len(x)*(p17)**0.5)
	
	return sum

	
def lout_featp12(x):
	sum = 0.0
	for i in range(len(x)):
		sum = sum + x[i]
	sum = sum/(len(x))
	
	return sum

def lout_featp13(x, lp12):
	p12 = lp12
	sum = 0.0
	for i in range(len(x)):
		sum = sum + (x[i]-p12)**2.0
	sum = sum/(len(x)-1)
	
	return sum

# def id_burst_threshold(x, threshold):
	# n = len(x)
	# indexes = []
	# for i in range(n):
		# if x[i] >= threshold:
			# indexes.append(i)
	# number_burst = len(indexes)
	# return number_burst, indexes

def id_burst_threshold(x, fs, threshold, t_window):
	n = len(x)
	dt = 1.0/fs
	ind_burst = []
	for i in range(n):
		if x[i] >= threshold:
			ind_burst.append(i)
	n_burst = len(ind_burst)
	ind_burst = np.array(ind_burst)
	
	t_burst = ind_burst*dt
	amp_burst = np.array([x[ind_burst[i]] for i in range(n_burst)])
	
	if n_burst > 0:
		t_burst_corr = []
		amp_burst_corr = []
		t_burst_corr.append(t_burst[0])
		amp_burst_corr.append(amp_burst[0])
		t_fix = t_burst[0]
		for i in range(n_burst-1):
			# check = t_burst[i+1] - t_burst[i]
			check = t_burst[i+1] - t_fix

			if check > t_window:
				t_burst_corr.append(t_burst[i+1])
				t_fix = t_burst[i+1]
				amp_burst_corr.append(amp_burst[i+1])
		
		n_burst_corr = len(t_burst_corr)
	else:
		n_burst_corr = n_burst
		t_burst_corr = t_burst
		amp_burst_corr = amp_burst
	
	return n_burst_corr, t_burst_corr, amp_burst_corr, t_burst, amp_burst

def read_threshold(mode, value, x1=None):
	if mode == 'factor_rms':
		threshold1 = value*signal_rms(x1)
	elif mode == 'fixed_value':
		threshold1 = value
	else:
		print('error threshold mode')
		sys.exit()
	return threshold1


def id_burst_threshold_end(x, fs, threshold, t_window, t_decay):
	n = len(x)
	dt = 1.0/fs
	ind_burst = []
	for i in range(n):
		if x[i] >= threshold:
			ind_burst.append(i)
	n_burst = len(ind_burst)
	ind_burst = np.array(ind_burst)
	
	t_burst = ind_burst*dt
	amp_burst = np.array([x[ind_burst[i]] for i in range(n_burst)])
	
	if n_burst > 0:
		t_burst_corr = []
		amp_burst_corr = []
		t_burst_corr.append(t_burst[0])
		amp_burst_corr.append(amp_burst[0])
		t_fix = t_burst[0]
		for i in range(n_burst-1):
			# check = t_burst[i+1] - t_burst[i]
			check = t_burst[i+1] - t_fix

			if check > t_window:
				t_burst_corr.append(t_burst[i+1])
				t_fix = t_burst[i+1]
				amp_burst_corr.append(amp_burst[i+1])
		
		n_burst_corr = len(t_burst_corr)
	else:
		n_burst_corr = n_burst
		t_burst_corr = t_burst
		amp_burst_corr = amp_burst
	
	
	
	return n_burst_corr, t_burst_corr, amp_burst_corr, t_burst, amp_burst


def burst_per_second(x, config):
	tr = len(x)/config['fs']
	n_points_segments = 1.*config['fs']		
	n_segments = int(len(x)/n_points_segments)
	n_burst_segments = []
	
	for k in range(n_segments):
		signal_it = x[n_points_segments*k : n_points_segments*(k+1)]
		t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector(signal_it, config, count=0)
		if len(t_burst_corr) != len(t_burst_corr_rev):
			print('fatal error 1224')
			sys.exit()
		n_burst_segments.append(len(t_burst_corr))
	
	return np.mean(n_burst_segments)

def burst_per_time(x, config):
	tr = len(x)/config['fs']
	n_points_segments = 1.*config['fs']		
	n_segments = int(len(x)/n_points_segments)
	n_burst_segments = []
	
	for k in range(n_segments):
		signal_it = x[n_points_segments*k : n_points_segments*(k+1)]
		t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector(signal_it, config, count=0)
		if len(t_burst_corr) != len(t_burst_corr_rev):
			print('fatal error 1224')
			sys.exit()
		n_burst_segments.append(len(t_burst_corr))
	
	return np.mean(n_burst_segments)

def bursts_features(xraw, t_burst_corr, t_burst_corr_rev, config):
	dura = []
	crest = []
	count = []
	rise = []
	p2p = []
	rms = []
	freq = []
	max = []
	n_id = []
	
	
	# kurt = []
	# skew = []
	# difp2 = []		
	# maxdif = []		
	# per25 = []
	# per50 = []
	# per75 = []		
	area = []
	
	fas = []
	# narea = []
	# nqua = []
	
	# xnew = xraw		
	
	xnew = np.zeros(len(xraw))	
	for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):		
		signal = xraw[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
		signal_complete = xraw[int(t_ini*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
		# if len(signal) >= 5:
		for i in range(len(signal_complete)):
			xnew[i + int(t_ini*config['fs'])] = signal_complete[i]



	index = 0
	for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
		signal = xraw[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
		signal_complete = xraw[int(t_ini*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
		print(len(signal))
		
		# if len(signal) >= 5:
		max_it = np.max(signal)
		dura_it = (t_fin - t_ini)*1000.*1000.
		rms_it = signal_rms(signal)
		rise_it = (np.argmax(signal)/config['fs'])*1000.*1000.
		p2p_it = max_it - np.min(signal)
		crest_it = np.max(np.absolute(signal))/rms_it
		magX_it, f_it, df_it = mag_fft(signal_complete, config['fs'])
		freq_it = (np.argmax(magX_it[1:])/(config['window_time']))/1000.
		# freq_it = (np.argmax(magX_it[1:])/(t_fin - t_ini))/1000.

		
		dura.append(dura_it)				
		rms.append(rms_it)				
		rise.append(rise_it)
		p2p.append(p2p_it)
		crest.append(crest_it)				
		freq.append(freq_it)
		max.append(max_it)
		n_id.append(index)
		
		
		contador = 0
		for u in range(len(signal)-1):			
			if (signal[u] < config['thr_value'] and signal[u+1] >= config['thr_value']):
				contador = contador + 1
		count.append(contador)
		
		
		
		# kurt.append(scipy.stats.kurtosis(signal, fisher=False))
		# skew.append(scipy.stats.skew(signal))					
		# maxdif.append(np.max(diff_signal(signal, 1)))
		# per50.append(np.percentile(np.absolute(signal), 50))
		# per75.append(np.percentile(np.absolute(signal), 75))
		# per25.append(np.percentile(np.absolute(signal), 25))
		area_it = 1000.*1000*np.sum(np.absolute(signal))/config['fs']
		area.append(area_it)					
		
		# nqua.append((np.percentile(np.absolute(signal), 75) - np.percentile(np.absolute(signal), 25))/np.percentile(np.absolute(signal), 50))
		
		# index_signal_it = [i for i in range(len(signal))]
		# tonto, envolve_up = env_up(index_signal_it, signal)					
		# index_triangle_it = [0, int(len(signal)/2), len(signal)]
		# triangle_it = [np.max(signal_complete), 0., np.max(signal)]					
		# index_signal_it = np.array(index_signal_it)
		# index_triangle_it = np.array(index_triangle_it)
		# triangle_up_it = np.array(triangle_it)					
		# poly2_coef = np.polyfit(index_triangle_it, triangle_it, 2)
		# p2 = np.poly1d(poly2_coef)
		# poly2 = p2(tonto)					
		# difp2.append(np.sum(np.absolute(poly2 - envolve_up)))
		
		fas_it = dura_it/p2p_it
		fas.append(fas_it)
		# narea.append(area_it/fas_it)
		
		

		index += 1
		
	mydict = {'rise':rise, 'fas':fas, 'crest':crest, 'count':count, 'p2p':p2p, 'freq':freq, 'max':max, 'area':area, 'rms':rms, 'dura':dura}
	return mydict


def single_burst_features(x, t_ini, t_fin, config):
	dura = []
	crest = []
	count = []
	rise = []
	p2p = []
	
	rms = []
	freq = []
	max = []		
	area = []	
	fas = []
	
	

	signal = x[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
	signal_extend = x[int(t_ini*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
	
	max_it = np.max(signal)
	dura_it = (t_fin - t_ini)*1000.*1000.
	rms_it = signal_rms(signal)
	rise_it = (np.argmax(signal)/config['fs'])*1000.*1000.
	p2p_it = max_it - np.min(signal)
	crest_it = np.max(np.absolute(signal))/rms_it
	
	magX_it, f_it, df_it = mag_fft(signal_extend, config['fs'])
	freq_it = (np.argmax(magX_it[1:])/(config['window_time']))/1000.

	
	dura = dura_it		
	rms = rms_it			
	rise = rise_it
	p2p = p2p_it
	crest = crest_it
	
	freq = freq_it	
	max = max_it
	
	
	contador = 0
	for u in range(len(signal)-1):			
		if (signal[u] < config['thr_value'] and signal[u+1] >= config['thr_value']):
			contador = contador + 1
	count = contador
	
	area_it = 1000.*1000*np.sum(np.absolute(signal))/config['fs']
	area = area_it
	
	fas_it = dura_it/p2p_it
	fas = fas_it
	
	

	
	mydict = {'rise':rise, 'fas':fas, 'crest':crest, 'count':count, 'p2p':p2p, 'freq':freq, 'max':max, 'area':area, 'rms':rms, 'dura':dura, 't_ini':t_ini, 't_fin':t_fin}

	return mydict

def single_burst_features2(x, t_ini, t_fin, config):
	# dura = []
	# crest = []
	# count = []
	# rise = []
	# # p2p = []
	
	# rms = []
	# freq = []
	# amax = []		
	# area = []	
	# fas = []
	
	

	signal = x[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
	signal_extend = x[int(t_ini*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
	
	amax_it = np.max(np.absolute(signal))
	dura_it = (t_fin - t_ini)*1000.*1000.
	rms_it = signal_rms(signal)
	rise_it = (np.argmax(signal)/config['fs'])*1000.*1000.
	p2p_it = np.max(signal) - np.min(signal)
	crest_it = amax_it/rms_it
	
	magX_it, f_it, df_it = mag_fft(signal_extend, config['fs'])
	freq_it = (np.argmax(magX_it[1:])/(config['window_time']))/1000.

	
	dura = dura_it		
	rms = rms_it			
	rise = rise_it
	p2p = p2p_it
	crest = crest_it
	
	freq = freq_it	
	amax = amax_it
	
	
	contador = 0.
	for u in range(len(signal)-1):			
		if (signal[u] < config['thr_value'] and signal[u+1] >= config['thr_value']):
			contador = contador + 1.
	count = contador
	
	kurt = scipy.stats.kurtosis(signal, fisher=False)
	
	area_it = 1000.*1000*np.sum(np.absolute(signal))/config['fs']
	area = area_it
	
	fas_it = dura_it/p2p_it
	fas = fas_it
	
	

	
	mydict = {'p2p':p2p, 'area':area, 'fas':fas, 'rise':rise, 'crest':crest, 'count':count, 'freq':freq, 'amax':amax, 'rms':rms, 'dura':dura, 'kurt':kurt, 't_ini':t_ini, 't_fin':t_fin}

	return mydict

def single_burst_features3(x, t_ini, t_fin, config, threshold=None):
	# dura = []
	# crest = []
	# count = []
	# rise = []
	# # p2p = []
	
	# rms = []
	# freq = []
	# amax = []		
	# area = []	
	# fas = []
	if threshold == None:
		threshold = read_threshold(config['thr_mode'], config['thr_value'], x)

	signal = x[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
	signal_extend = x[int(t_ini*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
	
	amax_it = np.max(np.absolute(signal))
	dura_it = (t_fin - t_ini)*1000.*1000.
	rms_it = signal_rms(signal)
	kurt_it = stats.kurtosis(signal, fisher=False)
	
	rise_p = (np.argmax(signal)/config['fs'])*1000.*1000.
	if rise_p != 0.:		
		rise_it = rise_p
	else:
		rise_it = 1/config['fs']*1000.*1000.
	
	
	p2p_it = np.max(signal) - np.min(signal)
	crest_it = amax_it/rms_it
	
	magX_it, f_it, df_it = mag_fft(signal_extend, config['fs'])
	freq_it = (np.argmax(magX_it[1:])/(config['window_time']))/1000.

	
	dura = dura_it		
	rms = rms_it			
	rise = rise_it
	kurt = kurt_it
	
	freq = freq_it	
	amax = amax_it
	
	
	contador = 1.
	for u in range(len(signal)-1):			
		if (signal[u] < threshold and signal[u+1] >= threshold):
			contador = contador + 1.
	count = contador
	
	

	
	

	
	mydict = {'kurt':kurt, 'rise':rise, 'count':count, 'freq':freq, 'amax':amax, 'rms':rms, 'dura':dura, 't_ini':t_ini, 't_fin':t_fin}

	return mydict


def single_burst_features4_vs(x, t_ini, t_fin, config, threshold=None):

	if threshold == None:
		threshold = read_threshold(config['thr_mode'], config['thr_value'], x)

	signal = x[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
	signal_extend = x[int(t_ini*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
	
	
	amax_it = np.max(np.absolute(signal))
	dura_it = (t_fin - t_ini)
	rms_it = signal_rms(signal)
	
	std = np.std(signal)
	
	rise_p = (np.argmax(signal)/config['fs'])
	if rise_p != 0.:		
		rise_it = rise_p
	else:
		rise_it = 1/config['fs']
	

	
	dura = dura_it		
	rms = rms_it			
	rise = rise_it
	
	amax = amax_it
	
	
	contador = 1.
	for u in range(len(signal)-1):			
		if (signal[u] < threshold and signal[u+1] >= threshold):
			contador = contador + 1.
	count = contador
	
	

	kurt = stats.kurtosis(signal, fisher=False)
	
	
	
	
	
	#freq
	Magnitude, f_it, df = mag_fft(signal_extend, config['fs'])
	freq = (np.argmax(Magnitude[1:])/(config['window_time']))
	nf = len(Magnitude)
	
	sum_mag = np.sum(Magnitude)				
	sum_fc = 0.
	sum_rmsf = 0.
	for i in range(nf):
		sum_fc += Magnitude[i]*df*i
		sum_rmsf += Magnitude[i]*(df*i)**2.
	cef = sum_fc/sum_mag
	sum_stf = 0.
	for i in range(nf):
		sum_stf += Magnitude[i]*(df*i-cef)**2.0
	stf = (sum_stf/sum_mag)**0.5
	
	
	

	
	mydict = {'std':std, 'cef':cef, 'freq':freq, 'stf':stf, 'kurt':kurt, 'rise':rise, 'count':count, 'amax':amax, 'rms':rms, 'dura':dura, 't_ini':t_ini, 't_fin':t_fin}

	return mydict


def single_burst_features5_exp(x, t_ini, t_fin, config, threshold=None, num_bursts_abs=None, filename=None, num_bursts_rel=None):

	if threshold == None:
		threshold = read_threshold(config['thr_mode'], config['thr_value'], x)

	signal = x[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
	signal_extend = x[int(t_ini*config['fs']) - int(int(config['window_time']*config['fs'])/2.) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
	
	
	if config['save_plot'] == 'ON':
		t = np.arange(len(signal_extend))/config['fs']
		fig, ax = plt.subplots()
		ax.plot(t, signal_extend)
		ax.set_title(str(num_bursts_abs) + '_rel_' + str(num_bursts_rel) + '_' + config['channel'] + '_' + filename[:-5])
		plt.savefig(str(num_bursts_abs) + '_rel_' + str(num_bursts_rel) + '_' + config['channel'] + '_' + filename[:-5] + '.png')
	
	
	
	amax_it = np.max(np.absolute(signal_extend))
	dura_it = (t_fin - t_ini)
	rms_it = signal_rms(signal_extend)
	
	std = np.std(signal_extend)
	
	rise_p = (np.argmax(signal_extend)/config['fs'])
	if rise_p != 0.:
		rise_it = rise_p
	else:
		rise_it = 1/config['fs']
	

	
	dura = dura_it		
	rms = rms_it			
	rise = rise_it
	
	amax = amax_it
	
	
	contador = 1.
	for u in range(len(signal)-1):			
		if (signal[u] < threshold and signal[u+1] >= threshold):
			contador = contador + 1.
	count = contador
	
	

	kurt = stats.kurtosis(signal_extend, fisher=False)
	
	
	
	
	
	#freq
	Magnitude, f_it, df = mag_fft(signal_extend, config['fs'])
	freq = (np.argmax(Magnitude[1:])/(1.5*config['window_time']))
	nf = len(Magnitude)
	
	sum_mag = np.sum(Magnitude)				
	sum_fc = 0.
	sum_rmsf = 0.
	for i in range(nf):
		sum_fc += Magnitude[i]*df*i
		sum_rmsf += Magnitude[i]*(df*i)**2.
	cef = sum_fc/sum_mag
	
	sum_stf = 0.
	sum_kurtf = 0.	
	for i in range(nf):
		sum_stf += Magnitude[i]*(df*i-cef)**2.0
		sum_kurtf += Magnitude[i]*(df*i-cef)**4.0
	stf = (sum_stf/sum_mag)**0.5
	kurtf = sum_kurtf/(sum_stf)**2.0
	
	# sum_kurtf = 0.		
	# for i in range(nf):
		# sum_stf += Magnitude[i]*(df*i-cef)**2.0
	# stf = (sum_stf/sum_mag)**0.5
	
	
	rmsf = (sum_rmsf/sum_mag)**0.5
	
	
	

	
	mydict = {'kurtf':kurtf, 'rmsf':rmsf, 'std':std, 'cef':cef, 'freq':freq, 'stf':stf, 'kurt':kurt, 'rise':rise, 'count':count, 'amax':amax, 'rms':rms, 'dura':dura, 't_ini':t_ini, 't_fin':t_fin}

	return mydict

def single_burst_features6_exp(x, t_ini, t_fin, config, threshold=None, num_bursts_abs=None, filename=None, num_bursts_rel=None):

	if threshold == None:
		threshold = read_threshold(config['thr_mode'], config['thr_value'], x)

	signal = x[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
	signal_extend = x[int(t_ini*config['fs']) - int(int(config['window_time']*config['fs'])/2.) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
	
	if len(signal_extend) == 0:
		signal_extend = x[int(t_ini*config['fs']): int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
	if len(signal_extend) == 0:
		signal_extend = x[int(t_ini*config['fs']) - int(config['window_time']/2*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']/2*config['fs'])]
	
	
	
	t = np.arange(len(signal_extend))/config['fs']
	if config['save_plot'] == 'ON':
		
		fig, ax = plt.subplots()
		ax.plot(t, signal_extend)
		ax.set_title(str(num_bursts_abs) + '_rel_' + str(num_bursts_rel) + '_' + config['channel'] + '_' + filename[:-5])
		plt.savefig(str(num_bursts_abs) + '_rel_' + str(num_bursts_rel) + '_' + config['channel'] + '_' + filename[:-5] + '.png')
	
	
	
	amax_it = np.max(np.absolute(signal_extend))
	dura_it = (t_fin - t_ini)
	rms_it = signal_rms(signal_extend)
	
	
	rise_p = (np.argmax(signal_extend)/config['fs'])
	if rise_p != 0.:
		rise_it = rise_p
	else:
		rise_it = 1/config['fs']
	

	
	dura = dura_it		
	rms = rms_it			
	rise = rise_it
	
	amax = amax_it
	
	
	contador = 1.
	for u in range(len(signal)-1):			
		if (signal[u] < threshold and signal[u+1] >= threshold):
			contador = contador + 1.
	count = contador
	
	

	kurt = stats.kurtosis(signal_extend, fisher=False)
	
	
	
	
	
	#freq
	Magnitude, f_it, df = mag_fft(signal_extend, config['fs'])
	freq = (np.argmax(Magnitude[1:])/(1.5*config['window_time']))
	nf = len(Magnitude)
	
	sum_mag = np.sum(Magnitude)				
	sum_fc = 0.
	sum_rmsf = 0.
	for i in range(nf):
		sum_fc += Magnitude[i]*df*i
		sum_rmsf += Magnitude[i]*(df*i)**2.
	cef = sum_fc/sum_mag
	
	sum_stf = 0.
	sum_kurtf = 0.	
	for i in range(nf):
		sum_stf += Magnitude[i]*(df*i-cef)**2.0
		sum_kurtf += Magnitude[i]*(df*i-cef)**4.0
	stf = (sum_stf/sum_mag)**0.5
	kurtf = sum_kurtf/(sum_stf)**2.0
	
	# sum_kurtf = 0.		
	# for i in range(nf):
		# sum_stf += Magnitude[i]*(df*i-cef)**2.0
	# stf = (sum_stf/sum_mag)**0.5
	
	
	rmsf = (sum_rmsf/sum_mag)**0.5
	
	
	# signal_extend_time = signal_extend*t
	# amaxt = np.max(np.absolute(signal_extend_time))
	# rmst = signal_rms(signal_extend_time)
	# kurtt = stats.kurtosis(signal_extend_time, fisher=False)
	
	# mydict = {'amaxt':amaxt, 'rmst':rmst, 'kurtt':kurtt, 'kurtf':kurtf, 'rmsf':rmsf, 'cef':cef, 'freq':freq, 'stf':stf, 'kurt':kurt, 'rise':rise, 'count':count, 'amax':amax, 'rms':rms, 'dura':dura, 't_ini':t_ini, 't_fin':t_fin}
	mydict = {'kurtf':kurtf, 'rmsf':rmsf, 'cef':cef, 'freq':freq, 'stf':stf, 'kurt':kurt, 'rise':rise, 'count':count, 'amax':amax, 'rms':rms, 'dura':dura, 't_ini':t_ini, 't_fin':t_fin}

	return mydict

def single_burst_features6_exp_short(x, t_ini, t_fin, config, threshold=None, num_bursts_abs=None, filename=None, num_bursts_rel=None):

	if threshold == None:
		threshold = read_threshold(config['thr_mode'], config['thr_value'], x)

	signal = x[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
	# signal_extend = x[int(t_ini*config['fs']) - int(int(config['window_time']*config['fs'])/2.) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
	
	t = np.arange(len(signal))/config['fs']
	if config['save_plot'] == 'ON':
		
		fig, ax = plt.subplots()
		ax.plot(t, signal)
		ax.set_title(str(num_bursts_abs) + '_rel_' + str(num_bursts_rel) + '_' + config['channel'] + '_' + filename[:-5])
		plt.savefig(str(num_bursts_abs) + '_rel_' + str(num_bursts_rel) + '_' + config['channel'] + '_' + filename[:-5] + '.png')
	
	
	
	amax_it = np.max(np.absolute(signal))
	dura_it = (t_fin - t_ini)
	rms_it = signal_rms(signal)
	
	
	rise_p = (np.argmax(signal)/config['fs'])
	if rise_p != 0.:
		rise_it = rise_p
	else:
		rise_it = 1/config['fs']
	

	
	dura = dura_it		
	rms = rms_it			
	rise = rise_it
	
	amax = amax_it
	
	
	contador = 1.
	for u in range(len(signal)-1):			
		if (signal[u] < threshold and signal[u+1] >= threshold):
			contador = contador + 1.
	count = contador
	
	

	kurt = stats.kurtosis(signal, fisher=False)
	
	
	
	
	
	#freq
	Magnitude, f_it, df = mag_fft(signal, config['fs'])
	freq = (np.argmax(Magnitude[1:])/(1.5*config['window_time']))
	nf = len(Magnitude)
	
	sum_mag = np.sum(Magnitude)				
	sum_fc = 0.
	sum_rmsf = 0.
	for i in range(nf):
		sum_fc += Magnitude[i]*df*i
		sum_rmsf += Magnitude[i]*(df*i)**2.
	cef = sum_fc/sum_mag
	
	sum_stf = 0.
	sum_kurtf = 0.	
	for i in range(nf):
		sum_stf += Magnitude[i]*(df*i-cef)**2.0
		sum_kurtf += Magnitude[i]*(df*i-cef)**4.0
	stf = (sum_stf/sum_mag)**0.5
	kurtf = sum_kurtf/(sum_stf)**2.0
	

	
	
	rmsf = (sum_rmsf/sum_mag)**0.5
	
	

	mydict = {'kurtf':kurtf, 'rmsf':rmsf, 'cef':cef, 'freq':freq, 'stf':stf, 'kurt':kurt, 'rise':rise, 'count':count, 'amax':amax, 'rms':rms, 'dura':dura, 't_ini':t_ini, 't_fin':t_fin}

	return mydict

def single_burst_features7_exp(x, t_ini, t_fin, config, threshold=None, num_bursts_abs=None, filename=None, num_bursts_rel=None):

	if threshold == None:
		threshold = read_threshold(config['thr_mode'], config['thr_value'], x)

	signal = x[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
	signal_extend = x[int(t_ini*config['fs']) - int(int(config['window_time']*config['fs'])/2.) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
	
	# env_extend = hilbert_demodulation(signal_extend)
	env_extend = butter_demodulation(x=signal_extend, fs=config['fs'], filter=['lowpass', 5.e3, 3], prefilter=None, type_rect='absolute_value', dc_value='without_dc')
	# diff_env_extend = diff_signal(env_extend, 1)
	# plt.plot(env_extend, 'r')
	# plt.plot(diff_env_extend, 'b')
	# plt.show()
	
	t = np.arange(len(signal_extend))/config['fs']
	if config['save_plot'] == 'ON':
		
		fig, ax = plt.subplots()
		ax.plot(t, signal_extend)
		ax.set_title(str(num_bursts_abs) + '_rel_' + str(num_bursts_rel) + '_' + 'time_' + str(t_ini) + '_' + config['channel'] + '_' + filename[:-5])
		plt.savefig(str(num_bursts_abs) + '_rel_' + str(num_bursts_rel) + '_' + config['channel'] + '_' + filename[:-5] + '.png')
	
	
	
	amax_it = np.max(np.absolute(signal_extend))
	dura_it = (t_fin - t_ini)
	rms_it = signal_rms(signal_extend)
	
	
	rise_p = (np.argmax(signal_extend)/config['fs'])
	if rise_p != 0.:
		rise_it = rise_p
	else:
		rise_it = 1/config['fs']
	

	
	dura = dura_it		
	rms = rms_it			
	rise = rise_it
	
	amax = amax_it
	
	
	contador = 1.
	for u in range(len(signal)-1):			
		if (signal[u] < threshold and signal[u+1] >= threshold):
			contador = contador + 1.
	count = contador
	
	

	kurt = stats.kurtosis(signal_extend, fisher=False)
	
	
	
	
	
	#freq
	Magnitude, f_it, df = mag_fft(signal_extend, config['fs'])
	freq = (np.argmax(Magnitude[1:])/(1.5*config['window_time']))
	nf = len(Magnitude)
	
	sum_mag = np.sum(Magnitude)				
	sum_fc = 0.
	sum_rmsf = 0.
	for i in range(nf):
		sum_fc += Magnitude[i]*df*i
		sum_rmsf += Magnitude[i]*(df*i)**2.
	cef = sum_fc/sum_mag
	

	
	
	sum_stf = 0.
	sum_kurtf = 0.	
	for i in range(nf):
		sum_stf += Magnitude[i]*(df*i-cef)**2.0
		sum_kurtf += Magnitude[i]*(df*i-cef)**4.0
	stf = (sum_stf/sum_mag)**0.5
	kurtf = sum_kurtf/(sum_stf)**2.0
	
	# sum_kurtf = 0.		
	# for i in range(nf):
		# sum_stf += Magnitude[i]*(df*i-cef)**2.0
	# stf = (sum_stf/sum_mag)**0.5
	
	
	rmsf = (sum_rmsf/sum_mag)**0.5
	
	
	signal_extend_time = signal_extend*t
	amaxt = np.max(np.absolute(signal_extend_time))
	rmst = signal_rms(signal_extend_time)
	kurtt = stats.kurtosis(signal_extend_time, fisher=False)
	
	
	
	sum_amp = np.sum(np.absolute(signal_extend))			
	sum_ampt = 0.
	sum_rmst = 0.
	for i in range(len(signal_extend)):
		sum_ampt += np.absolute(signal_extend[i])*i/config['fs']
		sum_rmst += np.absolute(signal_extend[i])*(i/config['fs'])**2.
	cet = sum_ampt/sum_amp
	rmscet = (sum_rmst/sum_amp)**0.5
	
	
	amaxenv = np.max(np.absolute(env_extend))
	stdenv = np.std(env_extend)
	kurtenv = stats.kurtosis(env_extend, fisher=False)		
	senenv = shannon_entropy(env_extend)
	
	sum_ampenv = np.sum((env_extend)**2.0)			
	sum_amptenv = 0.
	sum_rmstenv = 0.
	for i in range(len(env_extend)):
		sum_amptenv += ((env_extend[i])**2.0)*i/config['fs']
		sum_rmstenv += ((env_extend[i])**2.0)*(i/config['fs'])**2.
	cetenv = sum_amptenv/sum_ampenv
	rmscetenv = (sum_rmstenv/sum_ampenv)**0.5
	
	sen = shannon_entropy(signal_extend)
	sent = shannon_entropy(signal_extend_time)
	
	env_extend_time = env_extend*t
	amaxenvt = np.max(np.absolute(env_extend_time))
	stdenvt = np.std(env_extend_time)
	kurtenvt = stats.kurtosis(env_extend_time, fisher=False)		
	senenvt = shannon_entropy(env_extend_time)
	
	mydict = {'senenvt':senenvt, 'kurtenvt':kurtenvt, 'stdenvt':stdenvt, 'amaxenvt':amaxenvt, 'sent':sent, 'sen':sen, 'cetenv':cetenv, 'rmscetenv':rmscetenv, 'amaxenv':amaxenv, 'stdenv':stdenv, 'kurtenv':kurtenv, 'senenv':senenv, 'cet':cet, 'rmscet':rmscet, 'amaxt':amaxt, 'rmst':rmst, 'kurtt':kurtt, 'kurtf':kurtf, 'rmsf':rmsf, 'cef':cef, 'freq':freq, 'stf':stf, 'kurt':kurt, 'rise':rise, 'count':count, 'amax':amax, 'rms':rms, 'dura':dura, 't_ini':t_ini, 't_fin':t_fin}

	return mydict

def in_interval(number, low, high):
	if number > low and number <= high:
		bool = True
	else:
		bool = False
	return bool

def generate_10_intervals(low, high):
	max_intervals = list(np.linspace(low, high, num=9))
	max_str_intervals = [str(max_intervals[k]) + '_' + str(max_intervals[k+1]) for k in range(len(max_intervals)-1)]
	max_str_intervals = ['-Inf_' + str(max_intervals[0])] + max_str_intervals + [str(max_intervals[len(max_intervals)-1]) + '_Inf']
	
	for k in range(len(max_str_intervals)):
		max_str_intervals[k] = str(k) + '/' + max_str_intervals[k]
	
	Freq_Dict = {key:0 for key in max_str_intervals}
	return max_intervals, Freq_Dict, max_str_intervals

def generate_n_intervals(low, high, n):
	max_intervals = list(np.linspace(low, high, num=n-1))
	max_str_intervals = [str(max_intervals[k]) + '_' + str(max_intervals[k+1]) for k in range(len(max_intervals)-1)]
	max_str_intervals = ['-Inf_' + str(max_intervals[0])] + max_str_intervals + [str(max_intervals[len(max_intervals)-1]) + '_Inf']
	
	for k in range(len(max_str_intervals)):
		max_str_intervals[k] = str(k) + '/' + max_str_intervals[k]
	
	Freq_Dict = {key:0 for key in max_str_intervals}
	return max_intervals, Freq_Dict, max_str_intervals

def Lgenerate_n_intervals(low, high, n):
	max_intervals = list(np.linspace(low, high, num=n-1))
	max_str_intervals = [str(max_intervals[k]) + '_' + str(max_intervals[k+1]) for k in range(len(max_intervals)-1)]
	max_str_intervals = ['-Inf_' + str(max_intervals[0])] + max_str_intervals + [str(max_intervals[len(max_intervals)-1]) + '_Inf']
	
	for k in range(len(max_str_intervals)):
		max_str_intervals[k] = str(k) + '/' + max_str_intervals[k]
	
	Freq_Dict = {key:[] for key in max_str_intervals}
	return max_intervals, Freq_Dict, max_str_intervals

def Lgenerate_n_intervals_noinf(low, high, n):
	max_intervals = list(np.linspace(low, high, num=n-1))
	max_str_intervals = [str(format(max_intervals[k], '.2f')) + '_' + str(format(max_intervals[k+1], '.2f')) for k in range(len(max_intervals)-1)]
	# max_str_intervals = ['-Inf_' + str(max_intervals[0])] + max_str_intervals + [str(max_intervals[len(max_intervals)-1]) + '_Inf']
	letters = ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc', 'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bk', 'bl', 'bm', 'bn', 'bo', 'bp', 'bq', 'br', 'bs', 'bt', 'bu', 'bv', 'bw', 'bx', 'by', 'bz', 'ca', 'cb', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'ci', 'cj', 'ck', 'cl', 'cm', 'cn', 'co', 'cp', 'cq', 'cr', 'cs', 'ct', 'cu', 'cv', 'cw', 'cx', 'cy', 'cz']
	for k in range(len(max_str_intervals)):
		max_str_intervals[k] = letters[k] + '/' + max_str_intervals[k]
	
	Freq_Dict = {key:[] for key in max_str_intervals}
	return max_intervals, Freq_Dict, max_str_intervals

def Lgenerate_10_intervals(low, high):
	max_intervals = list(np.linspace(low, high, num=9))
	max_str_intervals = [str(max_intervals[k]) + '_' + str(max_intervals[k+1]) for k in range(len(max_intervals)-1)]
	max_str_intervals = ['-Inf_' + str(max_intervals[0])] + max_str_intervals + [str(max_intervals[len(max_intervals)-1]) + '_Inf']
	
	for k in range(len(max_str_intervals)):
		max_str_intervals[k] = str(k) + '/' + max_str_intervals[k]
	

	Freq_Dict = {key:[] for key in max_str_intervals}
	return max_intervals, Freq_Dict, max_str_intervals

def acum_in_dict10(mystr, dict_feat, max_intervals, max_Freq_Dict, max_str_intervals):
	if in_interval(dict_feat[mystr], -math.inf, max_intervals[0]):
		max_Freq_Dict[max_str_intervals[0]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[0], max_intervals[1]):
		max_Freq_Dict[max_str_intervals[1]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[1], max_intervals[2]):
		max_Freq_Dict[max_str_intervals[2]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[2], max_intervals[3]):
		max_Freq_Dict[max_str_intervals[3]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[3], max_intervals[4]):
		max_Freq_Dict[max_str_intervals[4]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[4], max_intervals[5]):
		max_Freq_Dict[max_str_intervals[5]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[5], max_intervals[6]):
		max_Freq_Dict[max_str_intervals[6]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[6], max_intervals[7]):
		max_Freq_Dict[max_str_intervals[7]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[7], max_intervals[8]):
		max_Freq_Dict[max_str_intervals[8]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[8], math.inf):
		max_Freq_Dict[max_str_intervals[9]] += 1
	else:
		print('error 8935')
		sys.exit()
	return max_Freq_Dict

def acum_in_dict20(mystr, dict_feat, max_intervals, max_Freq_Dict, max_str_intervals):
	
	
	
	if in_interval(dict_feat[mystr], -math.inf, max_intervals[0]):
		max_Freq_Dict[max_str_intervals[0]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[0], max_intervals[1]):
		max_Freq_Dict[max_str_intervals[1]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[1], max_intervals[2]):
		max_Freq_Dict[max_str_intervals[2]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[2], max_intervals[3]):
		max_Freq_Dict[max_str_intervals[3]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[3], max_intervals[4]):
		max_Freq_Dict[max_str_intervals[4]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[4], max_intervals[5]):
		max_Freq_Dict[max_str_intervals[5]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[5], max_intervals[6]):
		max_Freq_Dict[max_str_intervals[6]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[6], max_intervals[7]):
		max_Freq_Dict[max_str_intervals[7]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[7], max_intervals[8]):
		max_Freq_Dict[max_str_intervals[8]] += 1
	
	elif in_interval(dict_feat[mystr], max_intervals[8], max_intervals[9]):
		max_Freq_Dict[max_str_intervals[9]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[9], max_intervals[10]):
		max_Freq_Dict[max_str_intervals[10]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[10], max_intervals[11]):
		max_Freq_Dict[max_str_intervals[11]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[11], max_intervals[12]):
		max_Freq_Dict[max_str_intervals[12]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[12], max_intervals[13]):
		max_Freq_Dict[max_str_intervals[13]] += 1
	
	
	elif in_interval(dict_feat[mystr], max_intervals[13], max_intervals[14]):
		max_Freq_Dict[max_str_intervals[14]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[14], max_intervals[15]):
		max_Freq_Dict[max_str_intervals[15]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[15], max_intervals[16]):
		max_Freq_Dict[max_str_intervals[16]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[16], max_intervals[17]):
		max_Freq_Dict[max_str_intervals[17]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[17], max_intervals[18]):
		max_Freq_Dict[max_str_intervals[18]] += 1
	
	

	
	
	elif in_interval(dict_feat[mystr], max_intervals[18], math.inf):
		max_Freq_Dict[max_str_intervals[19]] += 1
	
	

	
	else:
		print('error 89135')
		sys.exit()
	return max_Freq_Dict


def append_in_dict(mystr, dict_feat, max_intervals, max_Freq_Dict, max_str_intervals):
	if in_interval(dict_feat[mystr], -math.inf, max_intervals[0]):
		max_Freq_Dict[max_str_intervals[0]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[0], max_intervals[1]):
		max_Freq_Dict[max_str_intervals[1]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[1], max_intervals[2]):
		max_Freq_Dict[max_str_intervals[2]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[2], max_intervals[3]):
		max_Freq_Dict[max_str_intervals[3]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[3], max_intervals[4]):
		max_Freq_Dict[max_str_intervals[4]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[4], max_intervals[5]):
		max_Freq_Dict[max_str_intervals[5]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[5], max_intervals[6]):
		max_Freq_Dict[max_str_intervals[6]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[6], max_intervals[7]):
		max_Freq_Dict[max_str_intervals[7]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[7], max_intervals[8]):
		max_Freq_Dict[max_str_intervals[8]] += 1
	elif in_interval(dict_feat[mystr], max_intervals[8], math.inf):
		max_Freq_Dict[max_str_intervals[9]] += 1
	else:
		print('error 8935')
		sys.exit()
	return max_Freq_Dict

def n_per_intervals(data, interval, divisions):
	data = sorted(data)
	save = 0
	values = np.zeros(divisions)
	interval_length = (interval[1] - interval[0])/divisions
	for i in range(divisions):		
		cont = 0
		for k in range(len(data)-save):
			if (data[k+save] <= interval[0] + interval_length*(i+1)):
				cont = cont + 1
			else:
				break
		values[i] = cont
		if cont != 0:
			save = cont + save
	if np.sum(values) != len(data):
		print(np.sum(values))
		print(len(data))
		print(interval)
		print(interval_length)
		print('error n per intervals')
		sys.exit()
		values = values.tolist()
	return values

def full_thr_burst_detector(x1, config, count=None, threshold=None):
	dt = 1.0/config['fs']
	n_points = len(x1)
	tr = n_points*dt
	t = np.array([i*dt for i in range(n_points)])
	traw = t
	
	if threshold == None:
		threshold = read_threshold(config['thr_mode'], config['thr_value'], x1)
	
	n_burst_corr1, t_burst_corr1, amp_burst_corr1, t_burst1, amp_burst1 = id_burst_threshold(x=x1, fs=config['fs'], threshold=threshold, t_window=config['window_time'])
	print('time TP111111111111111111111')
	

	t_burst_corr_rev = []
	
	for t_ini in t_burst_corr1:
		signal = x1[int(t_ini*config['fs']) : int((t_ini + config['window_time'])*config['fs'])]
		signal = signal[::-1]
		for k in range(len(signal)):
			if signal[k] >= threshold:
				t_end = t_ini + config['window_time'] - (k+1)/config['fs']
				t_burst_corr_rev.append(t_end)
				break
	

	# amp_burst_corr_rev = [x1[int(time_end*config['fs'])] for time_end in t_burst_corr_rev]
	amp_burst_corr_rev = amp_burst_corr1

	return t_burst_corr1, amp_burst_corr1, t_burst_corr_rev, amp_burst_corr_rev

def full_thr_burst_detector_stella(x1, config, count=None):
	dt = 1.0/config['fs']
	n_points = len(x1)
	tr = n_points*dt
	t = np.array([i*dt for i in range(n_points)])
	traw = t

	threshold1 = read_threshold(config['thr_mode'], config['thr_value'], x1)
	
	n_burst_corr1, t_burst_corr1, amp_burst_corr1, t_burst1, amp_burst1 = id_burst_threshold(x=x1, fs=config['fs'], threshold=threshold1, t_window=config['window_time'])
	print('time TP111111111111111111111')
	

	t_burst_corr_rev = []
	
	for t_ini in t_burst_corr1:
		signal = x1[t_ini*config['fs'] : (t_ini + config['window_time'])*config['fs']]
		signal = np.absolute(signal)
		count = 0
		flag = 'OFF'
		for k in range(len(signal)):
			if signal[k] < threshold1:
				count += 1
			elif signal[k] >= threshold1:
				count = 0
			if count >= config['stella']:
				flag = 'ON'
				t_end = t_ini + k/config['fs']
				t_burst_corr_rev.append(t_end)
				break
		if flag == 'OFF':
			t_end = t_ini + config['window_time'] - 1./config['fs']
			t_burst_corr_rev.append(t_end)

	amp_burst_corr_rev = [x1[int(time_end*config['fs'])] for time_end in t_burst_corr_rev]
	# amp_burst_corr_rev = amp_burst_corr1

	return t_burst_corr1, amp_burst_corr1, t_burst_corr_rev, amp_burst_corr_rev


def full_thr_burst_detector_stella_corr(x1, config, count=None):
	dt = 1.0/config['fs']
	n_points = len(x1)
	tr = n_points*dt
	t = np.array([i*dt for i in range(n_points)])
	traw = t

	threshold1 = read_threshold(config['thr_mode'], config['thr_value'], x1)
	
	t_burst_corr1 = []
	t_burst_corr_rev = []	
	
	signal = x1
	ini_point_next = 0
	# i = 0
	while (ini_point_next + int(config['window_time']*config['fs'])) < len(signal):
		if signal[ini_point_next] >= threshold1:
			t_ini_burst = (ini_point_next)/config['fs']
			t_burst_corr1.append(t_ini_burst)
			
			count = 0
			flag = 'OFF'
			window = x1[(ini_point_next) : (ini_point_next) + int(config['window_time']*config['fs'])]
			accu = 0
			for k in range(len(window)):
				accu += 1
				if window[k] < threshold1:
					count += 1
				elif window[k] >= threshold1:
					count = 0
				if count >= config['stella']:
					flag = 'ON'
					t_end_burst = t_ini_burst + k/config['fs']
					t_burst_corr_rev.append(t_end_burst)
					break
			if flag == 'OFF':
				t_end_burst = t_ini_burst + accu/config['fs']
				t_burst_corr_rev.append(t_end_burst)
			
			ini_point_next = ini_point_next + accu
		ini_point_next += 1
			
	if len(t_burst_corr_rev) != len(t_burst_corr1):
		print('fatal error 788')
		sys.exit()
	amp_burst_corr1 = [x1[int(time_ini*config['fs'])] for time_ini in t_burst_corr1]
	amp_burst_corr_rev = [x1[int(time_end*config['fs'])] for time_end in t_burst_corr_rev]
	# amp_burst_corr_rev = amp_burst_corr1

	return t_burst_corr1, amp_burst_corr1, t_burst_corr_rev, amp_burst_corr_rev

def full_thr_burst_detector_stella_lockout(x1, config, count=None):
	dt = 1.0/config['fs']
	n_points = len(x1)
	tr = n_points*dt
	t = np.array([i*dt for i in range(n_points)])
	traw = t

	threshold1 = read_threshold(config['thr_mode'], config['thr_value'], x1)
	
	t_burst_corr1 = []
	t_burst_corr_rev = []	
	
	signal = x1
	ini_point_next = 0
	# i = 0
	while (ini_point_next + int(config['window_time']*config['fs'])) < len(signal):
		if signal[ini_point_next] >= threshold1:
			t_ini_burst = (ini_point_next)/config['fs']
			t_burst_corr1.append(t_ini_burst)
			
			count = 0
			flag = 'OFF'
			window = x1[(ini_point_next) : (ini_point_next) + int(config['window_time']*config['fs'])]
			accu = 0
			for k in range(len(window)):
				accu += 1
				if window[k] < threshold1:
					count += 1
				elif window[k] >= threshold1:
					count = 0
				if count >= config['stella']:
					flag = 'ON'
					t_end_burst = t_ini_burst + k/config['fs']
					t_burst_corr_rev.append(t_end_burst)
					ini_point_next = ini_point_next + config['lockout']
					break
			if flag == 'OFF':
				t_end_burst = t_ini_burst + accu/config['fs']
				t_burst_corr_rev.append(t_end_burst)
			
			ini_point_next = ini_point_next + accu
		else:
			ini_point_next += 1
			
	if len(t_burst_corr_rev) != len(t_burst_corr1):
		print('fatal error 788')
		sys.exit()
	amp_burst_corr1 = [x1[int(time_ini*config['fs'])] for time_ini in t_burst_corr1]
	amp_burst_corr_rev = [x1[int(time_end*config['fs'])] for time_end in t_burst_corr_rev]
	# amp_burst_corr_rev = amp_burst_corr1

	return t_burst_corr1, amp_burst_corr1, t_burst_corr_rev, amp_burst_corr_rev


def full_thr_burst_detector_stella_lockout2(x1, config, count=None, threshold=None):
	dt = 1.0/config['fs']
	n_points = len(x1)
	tr = n_points*dt
	t = np.array([i*dt for i in range(n_points)])
	traw = t
	
	if threshold == None:
		threshold = read_threshold(config['thr_mode'], config['thr_value'], x1)
	
	t_burst_corr1 = []
	t_burst_corr_rev = []	
	
	signal = x1
	ini_point_next = 0
	# i = 0
	while (ini_point_next + int(config['window_time']*config['fs'])) < len(signal):
		if signal[ini_point_next] >= threshold:
			t_ini_burst = (ini_point_next)/config['fs']
			t_burst_corr1.append(t_ini_burst)
			
			count = 0
			flag = 'OFF'
			window = x1[(ini_point_next) : (ini_point_next) + int(config['window_time']*config['fs'])]
			accu = 0
			for k in range(len(window)):
				accu += 1
				if window[k] < threshold:
					count += 1
				elif window[k] >= threshold:
					count = 0
				if count >= config['stella']:
					flag = 'ON'
					t_end_burst = t_ini_burst + k/config['fs']
					t_burst_corr_rev.append(t_end_burst)
					ini_point_next = ini_point_next + config['lockout']
					break
			if flag == 'OFF':
				t_end_burst = t_ini_burst + accu/config['fs']
				t_burst_corr_rev.append(t_end_burst)
				ini_point_next = ini_point_next + config['lockout']
			
			ini_point_next = ini_point_next + accu
		else:
			ini_point_next += 1
			
	if len(t_burst_corr_rev) != len(t_burst_corr1):
		print('fatal error 788')
		sys.exit()
	amp_burst_corr1 = [x1[int(time_ini*config['fs'])] for time_ini in t_burst_corr1]
	amp_burst_corr_rev = [x1[int(time_end*config['fs'])] for time_end in t_burst_corr_rev]
	# amp_burst_corr_rev = amp_burst_corr1

	return t_burst_corr1, amp_burst_corr1, t_burst_corr_rev, amp_burst_corr_rev


def amp_component(X, df, freq):	
	return X[int(round(freq/df))]

def amp_component_zone(X, df, freq, tol):
	
	list_freq = [freq-tol/2. + i*df for i in range(int(tol/df)+1)]
	print('Frequencies List: ', list_freq)
	# if freq == 21.67*1.023077:
	
	list_mag = [amp_component(X, df, element) for element in list_freq]
	# if freq == 624.:
		# print(freq)
		# print(list_freq)
		# print(list_mag)
		# sys.exit()
		
	return np.max(np.array(list_mag))


def index_component_zone(X, df, freq, tol):
	list_freq = [freq-tol/2. + i*df for i in range(int(tol/df))]
	list_mag = [amp_component(X, df, element) for element in list_freq]
	return np.argmax(np.array(list_mag))
	



	
def n_per_5intervals(data, interval):
	divisions = 5
	data = sorted(data) 
	save = 0
	values = np.zeros(divisions)
	interval_length = (interval[1] - interval[0])/divisions
	for i in range(divisions):		
		cont = 0
		for k in range(len(data)-save):
			if (data[k+save] <= interval[0] + interval_length*(i+1)):
				cont = cont + 1
			else:
				break
		values[i] = cont
		if cont != 0:
			save = cont + save
	if np.sum(values) != len(data):
		print('error n per intervals')
		sys.exit()
	# values = values - 250.0
	# values = values / 250.0
	values = values.tolist()
	return values

def mean_5perinterval(data, interval):
	divisions = 5
	data = sorted(data) 
	save = 0
	
	values = np.zeros(divisions)
	means = np.zeros(divisions)
	interval_length = (interval[1] - interval[0])/divisions
	for i in range(divisions):		
		cont = 0
		sum = 0
		for k in range(len(data)-save):
			if (data[k+save] <= interval[0] + interval_length*(i+1)):
				cont = cont + 1
				sum = sum + data[k+save]
			else:
				break
		values[i] = cont
		if cont != 0:
			means[i] = sum/cont
		else:
			means[i] = 0.
		if cont != 0:
			save = cont + save
	if np.sum(values) != len(data):
		print('error n per intervals')
		sys.exit()
	# values = values - 250.0
	# values = values / 250.0
	values = values.tolist()
	means = means.tolist()
	return means

def interval5_stats(window):
	pos_max = np.argmax(window)
	intervals = []
	intervals.append(window[0:200])
	intervals.append(window[200:400])
	intervals.append(window[400:600])
	intervals.append(window[600:800])
	intervals.append(window[800:1000])	
	values = []
	for i in range(5):
		values = values + [np.max(intervals[i]), np.min(intervals[i]), np.mean(intervals[i]), np.std(intervals[i]), 
		stats.skew(np.array(intervals[i]), bias=False), stats.kurtosis(np.array(intervals[i]), fisher=False, bias=False)]
	return values

def interval5_stats_nomean(window):
	pos_max = np.argmax(window)
	intervals = []
	intervals.append(window[0:200])
	intervals.append(window[200:400])
	intervals.append(window[400:600])
	intervals.append(window[600:800])
	intervals.append(window[800:1000])	
	values = []
	for i in range(5):
		values = values + [np.max(intervals[i]), np.min(intervals[i]), np.std(intervals[i]), 
		stats.skew(np.array(intervals[i]), bias=False), stats.kurtosis(np.array(intervals[i]), fisher=False, bias=False)]
	return values

def interval10_stats(window):
	intervals = []
	intervals.append(window[0:100])
	intervals.append(window[100:200])
	intervals.append(window[200:300])
	intervals.append(window[300:400])
	intervals.append(window[400:500])
	intervals.append(window[500:600])
	intervals.append(window[600:700])
	intervals.append(window[700:800])
	intervals.append(window[800:900])
	intervals.append(window[900:1000])
	values = []
	for i in range(10):
		values = values + [np.max(intervals[i]), np.min(intervals[i]), np.mean(intervals[i]), np.std(intervals[i]), 
		stats.skew(np.array(intervals[i]), bias=False), stats.kurtosis(np.array(intervals[i]), fisher=False, bias=False)]
	return values

def interval10_stats_nomean(window):
	intervals = []
	intervals.append(window[0:100])
	intervals.append(window[100:200])
	intervals.append(window[200:300])
	intervals.append(window[300:400])
	intervals.append(window[400:500])
	intervals.append(window[500:600])
	intervals.append(window[600:700])
	intervals.append(window[700:800])
	intervals.append(window[800:900])
	intervals.append(window[900:1000])
	values = []
	for i in range(10):
		values = values + [np.max(intervals[i]), np.min(intervals[i]), np.std(intervals[i]), 
		stats.skew(np.array(intervals[i]), bias=False), stats.kurtosis(np.array(intervals[i]), fisher=False, bias=False)]
	return values

def interval10_stats_nmnsnk(window):
	intervals = []
	intervals.append(window[0:100])
	intervals.append(window[100:200])
	intervals.append(window[200:300])
	intervals.append(window[300:400])
	intervals.append(window[400:500])
	intervals.append(window[500:600])
	intervals.append(window[600:700])
	intervals.append(window[700:800])
	intervals.append(window[800:900])
	intervals.append(window[900:1000])
	values = []
	for i in range(10):
		values = values + [np.max(intervals[i]), np.min(intervals[i]), np.std(intervals[i])]
	return values

def sortint10_stats_nsnk(window):
	intervals = []
	intervals.append(window[0:100])
	intervals.append(window[100:200])
	intervals.append(window[200:300])
	intervals.append(window[300:400])
	intervals.append(window[400:500])
	intervals.append(window[500:600])
	intervals.append(window[600:700])
	intervals.append(window[700:800])
	intervals.append(window[800:900])
	intervals.append(window[900:1000])
	values = []
	for i in range(10):
		values = values + [np.max(intervals[i]), np.min(intervals[i]), np.std(intervals[i])]
	return values
	
def interval10_stats_nsnk(window):
	intervals = []
	intervals.append(window[0:100])
	intervals.append(window[100:200])
	intervals.append(window[200:300])
	intervals.append(window[300:400])
	intervals.append(window[400:500])
	intervals.append(window[500:600])
	intervals.append(window[600:700])
	intervals.append(window[700:800])
	intervals.append(window[800:900])
	intervals.append(window[900:1000])
	values = []
	for i in range(10):
		values = values + [np.max(intervals[i]), np.min(intervals[i]), np.mean(intervals[i]), np.std(intervals[i])]
	values = sorted(values)
	return values

def sortint20_stats_nsnk(window):
	intervals = []
	intervals.append(window[0:50])
	intervals.append(window[50:100])
	intervals.append(window[100:150])
	intervals.append(window[150:200])
	intervals.append(window[200:250])
	intervals.append(window[250:300])
	intervals.append(window[300:350])
	intervals.append(window[350:400])
	intervals.append(window[400:450])
	intervals.append(window[450:500])
	intervals.append(window[500:550])
	intervals.append(window[550:600])
	intervals.append(window[600:650])
	intervals.append(window[650:700])
	intervals.append(window[700:750])
	intervals.append(window[750:800])
	intervals.append(window[800:850])
	intervals.append(window[850:900])
	intervals.append(window[900:950])
	intervals.append(window[950:1000])
	values = []
	for i in range(20):
		values = values + [np.max(intervals[i]), np.min(intervals[i]), np.mean(intervals[i]), np.std(intervals[i])]
	values = sorted(values)
	return values

def sortint25_stats_nsnk(window):
	intervals = []
	intervals.append(window[0:40])
	intervals.append(window[40:80])
	intervals.append(window[80:120])
	intervals.append(window[120:160])
	intervals.append(window[160:200])
	intervals.append(window[200:240])
	intervals.append(window[240:280])
	intervals.append(window[280:320])
	intervals.append(window[320:360])
	intervals.append(window[360:400])
	intervals.append(window[400:440])
	intervals.append(window[440:480])
	intervals.append(window[480:520])
	intervals.append(window[520:560])
	intervals.append(window[560:600])
	intervals.append(window[600:640])
	intervals.append(window[640:680])
	intervals.append(window[680:720])
	intervals.append(window[720:760])
	intervals.append(window[760:800])
	intervals.append(window[800:840])
	intervals.append(window[840:880])
	intervals.append(window[880:920])
	intervals.append(window[920:960])
	intervals.append(window[960:1000])
	values = []
	for i in range(25):
		values = values + [np.max(intervals[i]), np.min(intervals[i]), np.mean(intervals[i]), np.std(intervals[i])]
	values = sorted(values)
	return values

# def sortint20_stats_nsnk(window):
	# intervals = []
	# intervals.append(window[0:50])
	# intervals.append(window[50:100])
	# intervals.append(window[100:150])
	# intervals.append(window[150:200])
	# intervals.append(window[200:250])
	# intervals.append(window[250:300])
	# intervals.append(window[300:350])
	# intervals.append(window[350:400])
	# intervals.append(window[400:450])
	# intervals.append(window[450:500])
	# intervals.append(window[500:550])
	# intervals.append(window[550:600])
	# intervals.append(window[600:650])
	# intervals.append(window[650:700])
	# intervals.append(window[700:750])
	# intervals.append(window[750:800])
	# intervals.append(window[800:850])
	# intervals.append(window[850:900])
	# intervals.append(window[900:950])
	# intervals.append(window[950:1000])
	# values = []
	# for i in range(20):
		# values = values + [np.max(intervals[i]), np.min(intervals[i]), np.mean(intervals[i]), np.std(intervals[i])]
	# values = sorted(values)
	# return values

def int20_stats_nsnk(window):
	intervals = []
	intervals.append(window[0:50])
	intervals.append(window[50:100])
	intervals.append(window[100:150])
	intervals.append(window[150:200])
	intervals.append(window[200:250])
	intervals.append(window[250:300])
	intervals.append(window[300:350])
	intervals.append(window[350:400])
	intervals.append(window[400:450])
	intervals.append(window[450:500])
	intervals.append(window[500:550])
	intervals.append(window[550:600])
	intervals.append(window[600:650])
	intervals.append(window[650:700])
	intervals.append(window[700:750])
	intervals.append(window[750:800])
	intervals.append(window[800:850])
	intervals.append(window[850:900])
	intervals.append(window[900:950])
	intervals.append(window[950:1000])
	values = []
	for i in range(20):
		values = values + [np.max(intervals[i]), np.min(intervals[i]), np.mean(intervals[i]), np.std(intervals[i])]
	return values

def std50max(window):
	window = sorted(window)
	window.reverse()
	values = [np.std(window[0:50])]
	
	return values

def interval10_maxminrms(window):
	intervals = []
	intervals.append(window[0:100])
	intervals.append(window[100:200])
	intervals.append(window[200:300])
	intervals.append(window[300:400])
	intervals.append(window[400:500])
	intervals.append(window[500:600])
	intervals.append(window[600:700])
	intervals.append(window[700:800])
	intervals.append(window[800:900])
	intervals.append(window[900:1000])
	values = []
	for i in range(10):
		values = values + [np.max(intervals[i]), np.min(intervals[i]), signal_rms(intervals[i])]
	return values

def interval10_maxminstd(window):
	intervals = []
	intervals.append(window[0:100])
	intervals.append(window[100:200])
	intervals.append(window[200:300])
	intervals.append(window[300:400])
	intervals.append(window[400:500])
	intervals.append(window[500:600])
	intervals.append(window[600:700])
	intervals.append(window[700:800])
	intervals.append(window[800:900])
	intervals.append(window[900:1000])
	values = []
	for i in range(10):
		values = values + [np.max(intervals[i]), np.min(intervals[i]), np.std(intervals[i])]
	return values

def interval10_stats_nmnsnknmin(window):
	intervals = []
	intervals.append(window[0:100])
	intervals.append(window[100:200])
	intervals.append(window[200:300])
	intervals.append(window[300:400])
	intervals.append(window[400:500])
	intervals.append(window[500:600])
	intervals.append(window[600:700])
	intervals.append(window[700:800])
	intervals.append(window[800:900])
	intervals.append(window[900:1000])
	values = []
	for i in range(10):
		values = values + [np.max(intervals[i]), np.std(intervals[i])]
	return values

def interval3_stats(window):
	intervals = []
	intervals.append(window[0:333])
	intervals.append(window[333:666])
	intervals.append(window[666:1000])
	values = []
	for i in range(3):
		values = values + [np.max(intervals[i]), np.min(intervals[i]), np.mean(intervals[i]), np.std(intervals[i]), 
		stats.skew(np.array(intervals[i]), bias=False), stats.kurtosis(np.array(intervals[i]), fisher=False, bias=False)]
	return values

def interval1_stats(window):	
	values = [np.max(window), np.min(window), np.mean(window), np.std(window), 
	stats.skew(np.array(window), bias=False), stats.kurtosis(np.array(window), fisher=False, bias=False)]
	return values



def leftright_stats(window):
	pos_max = np.argmax(window)
	left_window = window[0:pos_max]
	right_window = window[pos_max:]

	if (len(left_window) != 0 and len(right_window) != 0):
		values = [np.max(window), 
		np.min(left_window), np.mean(left_window), np.std(left_window), stats.skew(np.array(left_window), bias=False), stats.kurtosis(np.array(left_window), fisher=False, bias=False), 
		np.min(right_window), np.mean(right_window), np.std(right_window), stats.skew(np.array(right_window), bias=False), stats.kurtosis(np.array(right_window), fisher=False, bias=False)]
	elif (len(left_window) == 0 and len(right_window) != 0):
		values = [np.max(window), 
		0., 0., 0., 0., 0., 
		np.min(right_window), np.mean(right_window), np.std(right_window), stats.skew(np.array(right_window), bias=False), stats.kurtosis(np.array(right_window), fisher=False, bias=False)]
	elif (len(left_window) != 0 and len(right_window) == 0):
		values = [np.max(window), 
		np.min(left_window), np.mean(left_window), np.std(left_window), stats.skew(np.array(left_window), bias=False), stats.kurtosis(np.array(left_window), fisher=False, bias=False), 
		0., 0., 0., 0., 0.]
	else:
		print('error lens windows left and right+++++++++++++++++++++')
	return values

# def leftright_stats_norm(window):
	# pos_max = np.argmax(window)
	# left_window = window[0:pos_max]
	# right_window = window[pos_max:]

	# if (len(left_window) != 0 and len(right_window) != 0):
		# values = [np.max(window), 
		# np.min(left_window), np.mean(left_window), np.std(left_window), stats.skew(np.array(left_window), bias=False), stats.kurtosis(np.array(left_window), fisher=False, bias=False), 
		# np.min(right_window), np.mean(right_window), np.std(right_window), stats.skew(np.array(right_window), bias=False), stats.kurtosis(np.array(right_window), fisher=False, bias=False)]
	# elif (len(left_window) == 0 and len(right_window) != 0):
		# values = [np.max(window), 
		# 0., 0., 0., 0., 0., 
		# np.min(right_window), np.mean(right_window), np.std(right_window), stats.skew(np.array(right_window), bias=False), stats.kurtosis(np.array(right_window), fisher=False, bias=False)]
	# elif (len(left_window) != 0 and len(right_window) == 0):
		# values = [np.max(window), 
		# np.min(left_window), np.mean(left_window), np.std(left_window), stats.skew(np.array(left_window), bias=False), stats.kurtosis(np.array(left_window), fisher=False, bias=False), 
		# 0., 0., 0., 0., 0.]
	# else:
		# print('error lens windows left and right+++++++++++++++++++++')
	# return values

def leftright_stats_corr1(window):
	pos_max = np.argmax(window)
	left_window = window[0:pos_max]
	right_window = window[pos_max:]

	if (len(left_window) != 0 and len(right_window) != 0):
		values = [np.max(window), 
		np.min(left_window), np.std(left_window), stats.kurtosis(np.array(left_window), fisher=True), 
		np.min(right_window), np.std(right_window), stats.skew(np.array(right_window)), stats.kurtosis(np.array(right_window), fisher=True)]
	elif (len(left_window) == 0 and len(right_window) != 0):
		values = [np.max(window), 
		0., 0., 0., 
		np.min(right_window), np.std(right_window), stats.skew(np.array(right_window)), stats.kurtosis(np.array(right_window), fisher=True)]
	elif (len(left_window) != 0 and len(right_window) == 0):
		values = [np.max(window), 
		np.min(left_window), np.std(left_window), stats.kurtosis(np.array(left_window), fisher=True), 
		0., 0., 0., 0.]
	else:
		print('error lens windows left and right+++++++++++++++++++++')
	return values

def leftright_stats_nomean(window):
	pos_max = np.argmax(window)
	left_window = window[0:pos_max]
	right_window = window[pos_max:]

	if (len(left_window) != 0 and len(right_window) != 0):
		values = [np.max(window), 
		np.min(left_window), np.std(left_window), stats.skew(np.array(left_window), bias=False), stats.kurtosis(np.array(left_window), bias=False, fisher=False), np.min(right_window), np.std(right_window), stats.skew(np.array(right_window), bias=False), stats.kurtosis(np.array(right_window), bias=False, fisher=False)]
	elif (len(left_window) == 0 and len(right_window) != 0):
		values = [np.max(window), 
		0., 0., 0., 0., np.min(right_window), np.std(right_window), stats.skew(np.array(right_window), bias=False), stats.kurtosis(np.array(right_window), bias=False, fisher=False)]
	elif (len(left_window) != 0 and len(right_window) == 0):
		values = [np.max(window), 
		np.min(left_window), np.std(left_window), stats.skew(np.array(left_window), bias=False), stats.kurtosis(np.array(left_window), bias=False, fisher=False), 0., 0., 0., 0.]
	else:
		print('error lens windows left and right+++++++++++++++++++++')
	return values

def i10statsnm_lrstd(window):
	values = leftright_std(window) + interval10_stats_nomean(window)
	return values

def i10statsnmnsnk_lrstd(window):
	values = leftright_std(window) + interval10_stats_nmnsnk(window)
	return values

def i10statsnmnsnk_lrstd_std50max(window):
	values = std50max(window) + leftright_std(window) + interval10_stats_nmnsnk(window)
	return values


def i10statsnsnk_lrstd(window):
	values = leftright_std(window) + interval10_stats_nsnk(window)
	return values

def i10statsnsnk_lrstdmean(window):
	values = leftright_stdmean(window) + interval10_stats_nsnk(window)
	return values

def i10maxminrms_lrrms(window):
	values = leftright_rms(window) + interval10_maxminrms(window)
	return values

def i10maxminstd_lrrmsstd(window):
	values = leftright_std_rms(window) + interval10_maxminstd(window)
	return values

	
def si20statsnsnk_LRstdmean(window):
	values = leftright_stdmean(window) + sortint20_stats_nsnk(window)
	return values

def i10statsnmnsnknmin_lrstd(window):
	values = leftright_std(window) + interval10_stats_nmnsnknmin(window)
	return values

def i10statsnmnsnknmin_lrstd_lrnper5(window):
	values = leftright_std(window) + interval10_stats_nmnsnknmin(window) + n_per_5intervals_lr_norm(window)
	return values

def i10statsnmnsnk_lrstd_lrnper5(window):
	values = leftright_std(window) + interval10_stats_nmnsnk(window) + n_per_5intervals_lr_norm(window)
	return values

def i10statsnmnsnk_lrstd_lrmeanper5(window):
	values = leftright_std(window) + interval10_stats_nmnsnk(window) + mean_per_5intervals_lr_norm(window)
	return values

def i10statsnm_lrstatsnm(window):
	values = leftright_stats_nomean(window) + interval10_stats_nomean(window)
	return values

def i10statsnm_dif_lrstd(window):
	values = leftright_std(window) + interval10_stats_nomean(window) + dif_interval10_stats_nomean(window)
	return values

def means10(window):
	values = []
	count = 0
	for i in range(int(len(window)/10)):
		count = count + 1
		values = values + [np.mean(window[i:i+10])]

	return values
	
def dif_interval10_stats_nomean(window):
	intervals = []
	intervals.append(window[0:100])
	intervals.append(window[100:200])
	intervals.append(window[200:300])
	intervals.append(window[300:400])
	intervals.append(window[400:500])
	intervals.append(window[500:600])
	intervals.append(window[600:700])
	intervals.append(window[700:800])
	intervals.append(window[800:900])
	intervals.append(window[900:1000])
	values = []
	for i in range(9):
		values = values + [np.max(intervals[i+1])-np.max(intervals[i]), np.min(intervals[i+1])-np.min(intervals[i]), np.std(intervals[i+1])-np.std(intervals[i]), stats.skew(np.array(intervals[i+1]), bias=False)-stats.skew(np.array(intervals[i]), bias=False), stats.kurtosis(np.array(intervals[i+1]), fisher=False, bias=False)-stats.kurtosis(np.array(intervals[i]), fisher=False, bias=False)]
	return values

def leftright_stdmean(window):
	pos_max = np.argmax(window)
	left_window = window[0:pos_max]
	right_window = window[pos_max:]

	if (len(left_window) != 0 and len(right_window) != 0):
		values = [np.std(left_window), np.std(right_window), np.mean(left_window), np.mean(right_window)]
	elif (len(left_window) == 0 and len(right_window) != 0):
		values = [0., 0., np.std(right_window), np.mean(right_window)]
	elif (len(left_window) != 0 and len(right_window) == 0):
		values = [np.mean(left_window), np.std(left_window), 0., 0.]
	else:
		print('error lens windows left and right+++++++++++++++++++++')
	return values

def leftright_std(window):
	pos_max = np.argmax(window)
	left_window = window[0:pos_max]
	right_window = window[pos_max:]

	if (len(left_window) != 0 and len(right_window) != 0):
		values = [np.std(left_window), np.std(right_window)]
	elif (len(left_window) == 0 and len(right_window) != 0):
		values = [0., np.std(right_window)]
	elif (len(left_window) != 0 and len(right_window) == 0):
		values = [np.std(left_window), 0.]
	else:
		print('error lens windows left and right+++++++++++++++++++++')
	return values

def leftright_std_rms(window):
	pos_max = np.argmax(window)
	left_window = window[0:pos_max]
	right_window = window[pos_max:]

	if (len(left_window) != 0 and len(right_window) != 0):
		values = [np.std(left_window), signal_rms(left_window), np.std(right_window), signal_rms(right_window)]
	elif (len(left_window) == 0 and len(right_window) != 0):
		values = [0., 0., np.std(right_window), signal_rms(right_window)]
	elif (len(left_window) != 0 and len(right_window) == 0):
		values = [np.std(left_window), signal_rms(left_window), 0., 0.]
	else:
		print('error lens windows left and right+++++++++++++++++++++')
	return values
	
def leftright_rms(window):
	pos_max = np.argmax(window)
	left_window = window[0:pos_max]
	right_window = window[pos_max:]

	if (len(left_window) != 0 and len(right_window) != 0):
		values = [signal_rms(left_window), signal_rms(right_window)]
	elif (len(left_window) == 0 and len(right_window) != 0):
		values = [0., signal_rms(right_window)]
	elif (len(left_window) != 0 and len(right_window) == 0):
		values = [signal_rms(left_window), 0.]
	else:
		print('error lens windows left and right+++++++++++++++++++++')
	return values


def n_per_5intervals_lr_norm(data):
	divisions = 5
	interval = [int(np.min(data)), np.ceil(np.max(data))]
	window = data
	pos_max = np.argmax(window)
	left_window = window[0:pos_max]
	right_window = window[pos_max:]

	if (len(left_window) != 0 and len(right_window) != 0):
		values = n_per_intervals(left_window, interval, divisions) + n_per_intervals(right_window, interval, divisions)
		
	elif (len(left_window) == 0 and len(right_window) != 0):
		values = np.zeros(divisions).tolist() + n_per_intervals(right_window, interval, divisions)

	elif (len(left_window) != 0 and len(right_window) == 0):
		values = n_per_intervals(left_window, interval, divisions) + np.zeros(divisions).tolist()

	else:
		print('error lens windows left and right+++++++++++++++++++++')
	return values

def mean_per_5intervals_lr_norm(data):
	divisions = 5
	interval = [int(np.min(data)), np.ceil(np.max(data))]
	window = data
	pos_max = np.argmax(window)
	left_window = window[0:pos_max]
	right_window = window[pos_max:]

	if (len(left_window) != 0 and len(right_window) != 0):
		values = mean_5perinterval(left_window, interval) + mean_5perinterval(right_window, interval)
		
	elif (len(left_window) == 0 and len(right_window) != 0):
		values = np.zeros(divisions).tolist() + mean_5perinterval(right_window, interval)

	elif (len(left_window) != 0 and len(right_window) == 0):
		values = mean_5perinterval(left_window, interval) + np.zeros(divisions).tolist()

	else:
		print('error lens windows left and right+++++++++++++++++++++')
	return values

def n_per_5intervals_lr_norm(data):
	divisions = 5
	interval = [int(np.min(data)), np.ceil(np.max(data))]
	window = data
	pos_max = np.argmax(window)
	left_window = window[0:pos_max]
	right_window = window[pos_max:]

	if (len(left_window) != 0 and len(right_window) != 0):
		values = n_per_intervals(left_window, interval, divisions) + n_per_intervals(right_window, interval, divisions)
		
	elif (len(left_window) == 0 and len(right_window) != 0):
		values = np.zeros(divisions).tolist() + n_per_intervals(right_window, interval, divisions)

	elif (len(left_window) != 0 and len(right_window) == 0):
		values = n_per_intervals(left_window, interval, divisions) + np.zeros(divisions).tolist()

	else:
		print('error lens windows left and right+++++++++++++++++++++')
	return values

def n_per_intervals_left_right(data, interval, divisions):
	window = data
	pos_max = np.argmax(window)
	left_window = window[0:pos_max]
	right_window = window[pos_max:]

	if (len(left_window) != 0 and len(right_window) != 0):
		values = n_per_intervals(left_window, interval, divisions) + n_per_intervals(right_window, interval, divisions)
		
	elif (len(left_window) == 0 and len(right_window) != 0):
		values = np.zeros(divisions).tolist() + n_per_intervals(right_window, interval, divisions)

	elif (len(left_window) != 0 and len(right_window) == 0):
		values = n_per_intervals(left_window, interval, divisions) + np.zeros(divisions).tolist()

	else:
		print('error lens windows left and right+++++++++++++++++++++')
	return values






	