import os
from os import listdir
from os.path import join, isdir, basename, dirname, isfile
import sys

Mypaths = ['C:\\Felix\\29_THESIS\\MODEL_B\\Chapter_3_Diagnostics\\04_Data\\CWD\\Damage\\AE', 'C:\\Felix\\29_THESIS\\MODEL_B\\Chapter_3_Diagnostics\\04_Data\\CWD\\NoDamage\\AE']

for mypath, name in zip(Mypaths, Names):
	os.system('python Long_Analysis2.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode new_long_analysis_features_wfm --name AE_0_' + name)
	
	os.system('python Long_Analysis2.py --mypath ' + mypath + ' --channel AE_1 --fs 1.e6 --mode new_long_analysis_features_wfm --name AE_1_' + name)
	
	os.system('python Long_Analysis2.py --mypath ' + mypath + ' --channel AE_2 --fs 1.e6 --mode new_long_analysis_features_wfm --name AE_2_' + name)
	

# Mypaths = ['C:\\Felix\\29_THESIS\\MODEL_B\\Chapter_3_Diagnostics\\04_Data\\CWD\\Damage\\AC', 'C:\\Felix\\29_THESIS\\MODEL_B\\Chapter_3_Diagnostics\\04_Data\\CWD\\NoDamage\\AC']

# for mypath, name in zip(Mypaths, Names):
	# os.system('python Long_Analysis2.py --mypath ' + mypath + ' --channel AC_0 --fs 1.e6 --mode new_long_analysis_features_wfm --name AC_0_' + name)
	
	# os.system('python Long_Analysis2.py --mypath ' + mypath + ' --channel AC_1 --fs 1.e6 --mode new_long_analysis_features_wfm --name AC_1_' + name)
	
	# os.system('python Long_Analysis2.py --mypath ' + mypath + ' --channel AC_2 --fs 1.e6 --mode new_long_analysis_features_wfm --name AC_2_' + name)


	


