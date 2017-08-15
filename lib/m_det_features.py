import numpy as np
from scipy.integrate import odeint
from scipy import signal
from scipy import stats
import scipy




def features_master(name, x=None, dt=None, magX=None, df=None):
	if name == 'RMS_WFM_0':
		value = signal_rms(x)
	
	elif name == 'KURT_WFM_0':
		value = stats.kurtosis(x, fisher=True)
	
	elif name == 'LP6_WFM_0':
		value = lout_featp6(x)
	
	elif name == 'LP7_WFM_0':
		value = lout_featp7(x)
	
	elif name == 'LP16_WFM_0':
		value = lout_featp16(x, dt)
	
	elif name == 'LP17_WFM_0':
		value = lout_featp17(x, dt)
	
	elif name == 'LP21_WFM_0':
		value = lout_featp21(x, dt)
	
	elif name == 'LP24_WFM_0':
		value = lout_featp24(x, dt)
	
	elif name == 'LP16_FFT_0':
		value = lout_featp16(magX, df)
	
	elif name == 'LP17_FFT_0':
		value = lout_featp17(magX, df)
	
	elif name == 'LP21_FFT_0':
		value = lout_featp21(magX, df)
	
	elif name == 'LP24_FFT_0':
		value = lout_featp24(magX, df)
	
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
	
	return n_burst_corr, t_burst_corr, amp_burst_corr, t_burst, amp_burst

	
	

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











	