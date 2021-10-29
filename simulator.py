import numpy as np
from scipy.integrate import odeint
from scipy import signal
from scipy import stats
import scipy
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './lib') #to open user-defined functions
from m_fft import *
from m_denois import *
from m_demodulation import *
def pend(y, t, c, k, m, Fe, Fu, omega, z, mod_rect_gx, mod_sin_gx, Fb, bpf):
	x, xd = y
	sig_mod_rect_gx = mod_rect_gx*(np.array(signal.square(2*np.pi*t*omega, duty=1.0/z)) + 1*np.ones_like(t))/2.0
	sig_mod_sin_gx = mod_sin_gx*(np.array(np.sin(2*np.pi*t*omega)))	
	sig_mod_2f_sin_gx = mod_2f_sin_gx*(np.array(np.sin(2*np.pi*t*2*omega)))	
	# b = np.random.normal(0.0, 0.0000000001)
	b = 0

	sig_bear = (np.array(signal.square(2*np.pi*t*omega*bpf+b , duty=1/16.0)) + 1*np.ones_like(t))/2.0

	yd = [xd, (Fu*np.sin(2*np.pi*omega*t+50) + Fe*np.sin(2*np.pi*omega*z*t)*(sig_mod_rect_gx+1)*(sig_mod_sin_gx+1)*(sig_mod_2f_sin_gx+1) + Fb*sig_bear - c*xd - k*x)/m]
	return yd

#MODELL
c = 10.0 #Ns/m
k = 25000.0 #N/m
m = 0.5 #kg

#UNBALANCE AND SPEED
Fu = 0 #N
omega = 1.0 #Hz

#GEARBOX
Fe = 20 #N
z = 13.0 #teeth
mod_rect_gx = 0.2 #%modulation rectangular gearbox
mod_sin_gx = 0.0 #%modulation sinus gearbox
mod_2f_sin_gx = 0.0 #%modulation sinus gearbox 2freq



#BEARINGS
bpf = 1.7 #times omega
Fb = 8 #N

#Initial displ, vel
y0 = [0.0, 0.0] #m, m/s

#SAMPLING
Tr = 10 #s
dt = 0.001 #s
fs = 1.0/dt
#t = np.linspace(0, Tr, Tr/dt+1)
t = np.array([i*dt for i in range(int(Tr/dt))])
n = int(Tr/dt)

#MODULATION
mod = 0.0
a = -mod*(np.array(signal.square(2 * np.pi * t, duty=1.0/z)) + 1*np.ones_like(t))/2.0
exc = (np.array(signal.square(2*np.pi*t*0.5 , duty=1.0/z)) + 1*np.ones_like(t))/2.0


#SOLVE
sol = odeint(pend, y0, t, args=(c,k,m,Fe,Fu,omega,z,mod_rect_gx, mod_sin_gx, Fb, bpf))
x = sol[:, 0]


#NOISE
mean = 0
var = 0.0000001
if mean!=0.0 or var!=0.0:
	x = x + np.array(np.random.normal(mean, var, len(a)))

# PLOT
# plt.plot(t, x, 'b')
# plt.title('x1 WFM')
# plt.xlabel('Time s')
# plt.ylabel('Amplitude x')
# plt.show()

#FOURIER
magX, f, df = mag_fft(x, fs)
phaX = np.angle(np.fft.fft(x))
phaX = phaX[0:int(len(phaX)/2)]
# plt.plot(f, magX, 'r')
# plt.title('x1 MAG')
# plt.xlabel('Frequency Hz')
# plt.ylabel('Magnitude X')
# plt.show()


#CEPSTRUM
# ifftx, ti, dti = inv_fft_c(magX, phaX, df)
# ifftx, ti, dti = inv_fft_c1(magX, df)


plt.subplot(1, 2, 1)
plt.title('x WFM')
plt.xlabel('Time s')
plt.ylabel('Amplitude x')
plt.plot(t, x)

# plt.subplot(1, 2, 2)
# plt.title('inv fft x WFM')
# plt.xlabel('Time s')
# plt.ylabel('Amplitude x')
# plt.plot(ti, ifftx)

plt.show()


plt.subplot(1, 2, 1)
plt.title('FFT x')
plt.xlabel('Frequency Hz')
plt.ylabel('Amplitude x')
plt.plot(f, magX)

# plt.subplot(1, 2, 2)
# plt.title('inv fft x WFM')
# plt.xlabel('Time s')
# plt.ylabel('Amplitude x')
# plt.plot(ti, ifftx)

plt.show()


# magX, f, df = mag_fft(x, fs)

# n_magX = len(magX)

# logX = np.log(magX) +1j*phaX
# cepstrum = np.fft.ifft(logX)
# plt.plot(cepstrum)
# plt.show()

# magX = np.log(magX)

# x_series = np.zeros(n)
# for i in range(n):
	# sum = 0
	# for j in range(n_magX):
		# sum = sum + magX[j]*np.cos(2*math.pi*j*df*i*dt + phaX[j])
	# x_series[i] = sum

# plt.plot(t, x_series)
# plt.show()






# magX, f, df = mag_fft(x, fs)
# phaX = np.angle(np.fft.fft(x))
# phaX = phaX[0:len(phaX)/2]
# n_magX = len(magX)

# x_series = np.zeros(n)
# for i in range(n):
	# sum = 0
	# for j in range(n_magX):
		# sum = sum + magX[j]*np.cos(2*math.pi*j*df*i*dt + phaX[j])
	# x_series[i] = sum

# plt.plot(t, x_series)
# plt.show()


# RcepstrumX = np.fft.ifft(magX)*(n/2)
# plt.plot(t[0:len(t)/2], RcepstrumX)
# plt.title('abs cepstrum fft')
# plt.show()



# RcepstrumX = np.fft.ifft(np.log(np.abs(np.fft.fft(x))))
# plt.plot(RcepstrumX)
# plt.title('abs cepstrum fft')
# plt.show()
# print(len(RcepstrumX))

# RcepstrumX = np.fft.ifft(np.abs(np.log(np.fft.fft(x))))
# plt.plot(RcepstrumX)
# plt.title('abs cepstrum fft')
# plt.show()
# print(len(RcepstrumX))

# RcepstrumX = np.fft.ifft(np.log10(np.abs(np.fft.fft(x))))
# plt.plot(RcepstrumX)
# plt.title('abs cepstrum fft')
# plt.show()
# print(len(RcepstrumX))

# RcepstrumX = np.fft.ifft(np.abs(np.log10(np.fft.fft(x))))
# plt.plot(RcepstrumX)
# plt.title('abs 2 complex cepstrum')
# plt.show()
# print(len(RcepstrumX))

# RcepstrumX = np.fft.ifft(np.real(np.log10(np.fft.fft(x))))
# plt.plot(RcepstrumX)
# plt.title('re 2 complex cepstrum')
# plt.show()
# print(len(RcepstrumX))

# RcepstrumX = np.fft.ifft(np.log10(np.real(np.fft.fft(x))))
# plt.plot(RcepstrumX)
# plt.title('re cepstrum fft')
# plt.show()
# print(len(RcepstrumX))

# RcepstrumX = np.fft.ifft(np.log10(np.fft.fft(x)))
# plt.plot(RcepstrumX)
# plt.title('complex cepstrum')
# plt.show()
# print(len(RcepstrumX))



# f_psd, psdX = signal.periodogram(x, fs, return_onesided=True, scaling='density')
# RcepstrumX = np.fft.ifft(np.log10(psdX))
# plt.plot(RcepstrumX)
# plt.title('real cepstrum psd')
# plt.show()

# CcepstrumX = np.fft.ifft(np.fft.fft(x))
# plt.plot(CcepstrumX)
# plt.title('complex cepstrum fft')
# plt.show()
# print(len(CcepstrumX))

sys.exit()




#BURST
x_demod = hilbert_demodulation(x)



plt.plot(t, x_demod)
plt.title('x1_demod WFM')
plt.ylabel('Amplitude x_demod')
plt.xlabel('Time s')
plt.show()



magX_demod, f_demod, df_demod = mag_fft(x_demod, fs)
plt.plot(f_demod, magX_demod, 'r')
plt.title('x1_demod MAG')
plt.xlabel('Frequency Hz')
plt.ylabel('Magnitude X_demod')
plt.show()














# PLOT
# plt.plot(t, x, 'b')
# plt.show()

# plt.plot(f, fftx, '-or')
# plt.show()

#FEATURES
#rms
# def feat_rms(x):
	# return np.mean(x)

# def shortFFT(x, fs, segments, window, mode):
	# nperseg = len(x)/segments
	# f, t, stftX = signal.spectrogram(x, fs, nperseg=nperseg, window=window, mode=mode)
	# stftX = stftX/nperseg
	# return f, t, stftX




# nperseg = len(x)/segments
# f, t, stftX = signal.spectrogram(x, fs, nperseg=nperseg, window=window, mode=mode)
# stftX = stftX/nperseg

# f, t, stftX = shortFFT(x, fs, 10, 'boxcar', 'magnitude')


# plt.pcolormesh(t, f, stftX)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.colorbar()
# plt.show()

# plt.plot(f, stftX[:,0], 'r', f, stftX[:,1], 'b',)
# plt.show()

# prom = np.zeros(len(f))
# for i in range(len(f)):
	# prom[i] = np.mean(stftX[i,:])


# plt.plot(f, prom, 'g')
# plt.show()

# kurt = np.zeros(len(f))
# for i in range(len(f)):
	# kurt[i] = stats.kurtosis(stftX[i,:], fisher=True)


# plt.plot(f, kurt, 'r')
# plt.show()

# fig, ax = plt.subplots()


# print(t)
# print(f)
# print(Zxx)


# mm = np.array(np.ones(100))
# print(type(mm))

# nn = np.ones_like(t)
# print(type(nn))

# tt = Fe*np.sin(2*np.pi*omega*z*t)
# print(type(tt))
# print(len(tt))
# print(tt)

# ee = np.array(np.random.normal(mean, var, len(a)))
# print(type(ee))
# print(len(ee))
# print(ee)

# hh = tt+ee
# print(hh)

# rr = np.ones_like(t)
# zz = np.ones(10000)
# print(rr)
# print(zz)

# print(type(rr))
# print(type(zz))

# print(len(rr))
# print(len(zz))

# pp = rr-zz
# print(pp)

# print(np.random.normal(0.0, 0.01))
# print(np.random.normal(0.0, 0.01))
# print(np.random.normal(0.0, 0.01))
# print(np.random.normal(0.0, 0.01))