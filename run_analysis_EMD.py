import os
from os import listdir
from os.path import join, isdir, basename, dirname, isfile
import sys


# Mypaths = ['C:\\Felix\\29_THESIS\\MODEL_B\\Chapter_3_Diagnostics\\04_Data\\CWD\\Damage\\AE']

Mypaths = ['C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\04_Data\\Schottland\\S1']

# Mypaths_h1 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\Verschiedene\\Long_EMD_full']


for mypath in Mypaths:
	Filepaths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

	for file in Filepaths:	
		# os.system('python M_EMD.py --mode emd_from_raw --file ' + file + ' --channel AE_0 --n_imf 2 --min_iter 2 --max_iter 5 --s_number 2 --tolerance 1 --fs 1.e6')
		
		os.system('python M_EMD.py --mode emd_from_raw --file ' + file + ' --channel AE_3 --n_imf 5 --min_iter 6000 --max_iter 14000 --s_number 2 --tolerance 1 --fs 1.e6 --range 0 11')
		
		os.system('python M_EMD.py --mode emd_from_raw --file ' + file + ' --channel AE_2 --n_imf 5 --min_iter 6000 --max_iter 14000 --s_number 2 --tolerance 1 --fs 1.e6 --range 0 11')
		

# for mypath, mypath_h1 in zip(Mypaths, Mypaths_h1):
	# Filepaths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
	# Filepaths_h1 = [join(mypath_h1, f) for f in listdir(mypath_h1) if isfile(join(mypath_h1, f))]

	# for file, file_h1 in zip(Filepaths, Filepaths_h1):	
		# os.system('python EmpMD2mod.py --file ' + file + ' --channel AE_0 --power2 OFF --n_imf 1 --min_iter 5000 --max_iter 500000 --s_number 2 --tolerance 1 --fs 1.e6 --file_h1 ' + file_h1)

	
