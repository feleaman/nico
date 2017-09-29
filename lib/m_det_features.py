import numpy as np
from scipy.integrate import odeint
from scipy import signal
from scipy import stats
import scipy
import sys



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

def lout_featp17(x, dp):
	p16 = lout_featp16(x, dp)
	sum = 0.0
	for i in range(len(x)):
		sum = sum + x[i]*(i*dp - p16)**2.0	
	sum = sum/len(x)
	sum = sum**0.5
	return sum

def lout_featp21(x, dp):
	p16 = lout_featp16(x, dp)
	p17 = lout_featp17(x, dp)
	return p17/p16

def lout_featp24(x, dp):
	p16 = lout_featp16(x, dp)
	p17 = lout_featp17(x, dp)
	sum = 0.0
	for i in range(len(x)):
		sum = sum + x[i]*(i*dp - p16)**0.5 #podria salir raiz negativa!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	sum = sum/(len(x)*p17**4.0)
	
	return p17/p16

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
		print('error n per intervals')
		sys.exit()
	values = values.tolist()
	return values

# def n_per_10intervals(data, interval):
	# divisions = 10
	# data = sorted(data)
	# save = 0
	# values = np.zeros(divisions)
	# interval_length = (interval[1] - interval[0])/divisions
	# for i in range(divisions):		
		# cont = 0
		# for k in range(len(data)-save):
			# if (data[k+save] <= interval[0] + interval_length*(i+1)):
				# cont = cont + 1
			# else:
				# break
		# values[i] = cont
		# if cont != 0:
			# save = cont + save
	# if np.sum(values) != len(data):
		# print('error n per intervals')
		# sys.exit()
	# # values = values - 250.0
	# # values = values / 250.0
	# values = values.tolist()
	# return values

	
# def n_per_5intervals(data, interval):
	# divisions = 5
	# data = sorted(data)
	# save = 0
	# values = np.zeros(divisions)
	# interval_length = (interval[1] - interval[0])/divisions
	# for i in range(divisions):		
		# cont = 0
		# for k in range(len(data)-save):
			# if (data[k+save] <= interval[0] + interval_length*(i+1)):
				# cont = cont + 1
			# else:
				# break
		# values[i] = cont
		# if cont != 0:
			# save = cont + save
	# if np.sum(values) != len(data):
		# print('error n per intervals')
		# sys.exit()
	# # values = values - 250.0
	# # values = values / 250.0
	# values = values.tolist()
	# return values

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
# def n_per_10intervals_corr1(data, interval):
	# divisions = 10
	# data = sorted(data)
	# save = 0
	# values = np.zeros(divisions)
	# interval_length = (interval[1] - interval[0])/divisions
	# for i in range(divisions):		
		# cont = 0
		# for k in range(len(data)-save):
			# if (data[k+save] <= interval[0] + interval_length*(i+1)):
				# cont = cont + 1
			# else:
				# break
		# values[i] = cont
		# if cont != 0:
			# save = cont + save
	# if np.sum(values) != len(data):
		# print('error n per intervals')
		# sys.exit()
	# values = values.tolist()
	# values = [values[3], values[4], values[5], values[6]]
	# return values

# def n_per_5intervals_corr1_left_right(data, interval):
	# window = data
	# pos_max = np.argmax(window)
	# left_window = window[0:pos_max]
	# right_window = window[pos_max:]

	# if (len(left_window) != 0 and len(right_window) != 0):
		# values = n_per_5intervals_corr1(left_window, interval) + n_per_5intervals_corr1(right_window, interval)
		
	# elif (len(left_window) == 0 and len(right_window) != 0):
		# values = [0., 0., 0., 0.] + n_per_10intervals_corr1(right_window, interval)
		# # print(values)
		# # sys.exit()
	# elif (len(left_window) != 0 and len(right_window) == 0):
		# values = n_per_10intervals_corr1(left_window, interval) + [0., 0., 0., 0.]
		# # print(values)
		# # sys.exit()
	# else:
		# print('error lens windows left and right+++++++++++++++++++++')
	# return values
	
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

# def id_burst_threshold2(x, fs, threshold, t_window):
	# n = len(x)
	# dt = 1.0/fs
	# ind_burst = []
	# for i in range(n):
		# if x[i] >= threshold:
			# ind_burst.append(i)
	# n_burst = len(ind_burst)
	# ind_burst = np.array(ind_burst)
	
	# t_burst = ind_burst*dt
	# amp_burst = np.array([x[ind_burst[i]] for i in range(n_burst)])
	
	# if n_burst > 0:
		# t_burst_corr = []
		# amp_burst_corr = []
		# # t_burst_corr.append(t_burst[0])
		# # amp_burst_corr.append(amp_burst[0])
		# t_fix = t_burst[0]
		# for i in range(n_burst-1):
			# # check = t_burst[i+1] - t_burst[i]
			# check = t_burst[i+1] - t_fix

			# if check > t_window:
				# t_burst_corr.append(t_burst[i])
				# t_fix = t_burst[i+1]
				# amp_burst_corr.append(amp_burst[i])
		
		# n_burst_corr = len(t_burst_corr)
	# else:
		# n_burst_corr = n_burst
		# t_burst_corr = t_burst
		# amp_burst_corr = amp_burst
	
	# return n_burst_corr, t_burst_corr, amp_burst_corr, t_burst, amp_burst
	
# def id_burst_threshold3(x, fs, threshold, t_window):
	# n = len(x)
	# dt = 1.0/fs
	# ind_burst = []
	# for i in range(n):
		# if x[i] >= threshold:
			# ind_burst.append(i)
	# n_burst = len(ind_burst)
	# ind_burst = np.array(ind_burst)
	
	# t_burst = ind_burst*dt
	# amp_burst = np.array([x[ind_burst[i]] for i in range(n_burst)])
	
	# if n_burst > 0:
		# t_burst_corr = []
		# amp_burst_corr = []
		# t_burst_corr.append(t_burst[0])
		# amp_burst_corr.append(amp_burst[0])
		# t_fix = t_burst[0]
		# for i in range(n_burst-1):
			# # check = t_burst[i+1] - t_burst[i]
			# check = t_burst[i+1] - t_fix

			# if check > t_window:
				# t_burst_corr.append(t_burst[i+1])
				# t_fix = t_burst[i+1]
				# amp_burst_corr.append(amp_burst[i+1])
		
		# n_burst_corr = len(t_burst_corr)
	# else:
		# n_burst_corr = n_burst
		# t_burst_corr = t_burst
		# amp_burst_corr = amp_burst
	
	# dur = np.zeros(n_burst_corr)
	# for i in range(n_burst_corr):
		

	# return n_burst_corr, t_burst_corr, amp_burst_corr, t_burst, amp_burst	

# 'NBU_WFM_0'
# 'NBU_WFM_1'
# 'NBU_DIF_0'
# 'NBU_DIF_1'
# 'NBU_DIF_ENV_0'
# 'NBU_DIF_ENV_1'
# 'NBU_ENV_STF_BIN_0'
# 'NBU_DIF_ENV_STF_BIN_0'

# x_demod = butter_demodulation(x=x, fs=fs, prefilter=['highpass', 50.0e3, 3], filter=['lowpass', 200.0, 3])

# x_demod = hilbert_demodulation(x)
# dxe_demod = diff_signal_eq(x=x_demod, length_diff=1)

# t_burst, amp_burst = id_burst_threshold(x=x, fs=fs, threshold=0.1, t_window=0.002)
# plt.figure(1)
# plt.plot(t_burst, amp_burst, 'ro')
# plt.plot(t, x, 'b')
# plt.axhline(y=0.1, xmin=0, xmax=1, hold=None)

# t_burst_dxe, amp_burst_dxe = id_burst_threshold(x=dxe_demod, fs=fs, threshold=0.03, t_window=0.002)
# plt.figure(2)
# plt.plot(t_burst_dxe, amp_burst_dxe, 'bo')
# plt.plot(t, dxe_demod, 'k')
# plt.axhline(y=0.03, xmin=0, xmax=1, hold=None)
# plt.show()


# a = np.histogram(x)
# plt.hist(x, bins='auto')
# plt.show()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
# def id_burst_threshold(x, threshold):
	# n = len(x)
	# indexes = []
	# for i in range(n):
		# if x[i] >= threshold:
			# indexes.append(i)
	# number_burst = len(indexes)

	# t_burst_x_corr = []
	# amp_burst_x_corr = []
	# for i in range(n_burst_x-1):
		# print(i)
		# check = np.arctan(t_burst_x[i+1] - t_burst_x[i])*360.0/(2*np.pi)
		# print(check)
		# if check > 3.0:
			# t_burst_x_corr.append(t_burst_x[i])
			# amp_burst_x_corr.append(amp_burst_x[i])
	
	# return number_burst, indexes


# def id_burst_diff_demod(envelope, fs, threshold, length_diff=None, derivative=None):
	# #Defaults
	# n = len(envelope) #envelope must have the same length als signal
	# dt = 1.0/fs
	# if not length_diff:
		# length_diff = 1
	# if not derivative:
		# derivative = 'off'
	# #Differentiation/Derivative
	# diff_envelope = np.zeros(n-length_diff)
	# n_diff = len(diff_envelope)
	# for i in range(n_diff):
		# diff_envelope[i] = envelope[i+length_diff] - envelope[i]
	# if derivative == 'on':
		# diff_envelope = diff_envelope/(length_diff*dt)
	# #Look Bursts in Diff Envelope / Not equivalent indexes to Signal
	# indexes_diff_envelope = []
	# for i in range(n_diff):
		# if diff_envelope[i] >= threshold:
			# indexes_diff_envelope.append(i)
	# n_burst_diff_envelope = len(indexes_diff_envelope)
	# bursts_diff_envelope = [n_burst_diff_envelope, indexes_diff_envelope]
	# #Look Bursts in Diff Envelope / Equivalent indexes to Signal
	# t_diff = np.linspace(0, n_diff-1, num=n_diff)
	# t_signal = np.linspace(0, n_diff-1, num=n)
	# diff_envelope_eq = np.interp(t_signal, t_diff, diff_envelope)
	# indexes_diff_envelope_eq = []
	# for i in range(n):
		# if diff_envelope_eq[i] >= threshold:
			# indexes_diff_envelope_eq.append(i)
	# n_burst_diff_envelope_eq = len(indexes_diff_envelope_eq)
	# bursts_diff_envelope_eq = [n_burst_diff_envelope_eq, indexes_diff_envelope_eq]
	# return bursts_diff_envelope, bursts_diff_envelope_eq











	