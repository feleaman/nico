import os
from os import listdir
from os.path import join, isdir, basename, dirname, isfile
import sys


Mypaths = ['C:\\Felix\\Data\\QuickGearbox\\A', 'C:\\Felix\\Data\\QuickGearbox\\G', 'C:\\Felix\\Data\\QuickGearbox\\K1', 'C:\\Felix\\Data\\QuickGearbox\\N', 'C:\\Felix\\Data\\QuickGearbox\\Q']

Names = ['QuickA_AE_Bursts_Idx6', 'QuickG_AE_Bursts_Idx6', 'QuickK1_AE_Bursts_Idx6', 'QuickN_AE_Bursts_Idx6', 'QuickQ_AE_Bursts_Idx6']

# Names = ['QuickA_AC_Wfm_Idx1', 'QuickG_AC_Wfm_Idx1', 'QuickK1_AC_Wfm_Idx1', 'QuickN_AC_Wfm_Idx1', 'QuickQ_AC_Wfm_Idx1']

# Names = ['QuickA_AC_Fft_Idx1', 'QuickG_AC_Fft_Idx1', 'QuickK1_AC_Fft_Idx1', 'QuickN_AC_Fft_Idx1', 'QuickQ_AC_Fft_Idx1']

# Names = ['QuickA_AE_Energy_Band_Idx3', 'QuickG_AE_Energy_Band_Idx3', 'QuickK1_AE_Energy_Band_Idx3', 'QuickN_AE_Energy_Band_Idx3', 'QuickQ_AE_Energy_Band_Idx3']

# Mypaths = ['C:\\Felix\\Data\\QuickGearbox\\Q']

# Names = ['QuickQ_AE_Tri_Idx2']

count = 0
for mypath, name in zip(Mypaths, Names):	
	
	# os.system('python Bursts_Clustering.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode burst_features_multi_intervals_3 --save ON --plot OFF --highpass 20.e3 --thr_value 30 --name ' + name)
	
	# os.system('python Tri_Analysis.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode mode5 --thr_value 0.3 --thr_mode fixed_value --name ' + name + ' --side_points 0 --harm_fc 1 --filter Kurtogram --plot OFF')
	
	# os.system('python Long_Analysis.py --mode new_long_analysis_features_wfm --mypath ' + mypath + ' --channel ACC_0 --fs 1.e6 --name ' + name)
	
	# os.system('python Long_Analysis.py --mode new_long_analysis_features_freq --level 5 --mypath ' + mypath + ' --channel ACC_0 --fs 50.e3 --name ' + name)
	
	os.system('python M_Bursts_Detector.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --plot OFF --mode bursts_per_file --thr_value 0.6 --thr_mode fixed_value --name ' + name + ' --filter OFF')
	
	# os.system('python M_Freq_Bands.py --mypath ' + mypath + ' --channel AE_0 --filter OFF --fs 1.e6 --mode energy_band --name ' + name)
	
	
	
	
	count += 1
	print('Avance: ', count/len(Names))
