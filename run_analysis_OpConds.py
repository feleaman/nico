import os
from os import listdir
from os.path import join, isdir, basename, dirname, isfile
import sys
import pandas as pd

for i in range(1):
	k = i+8
	mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180306_Messung_E\\Test_' + str(k)
	# mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_O\\Test_' + str(k)
	
	extension = 'dms'
	Filepaths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) if f[-3:] == extension if f[1] == 'E']	

	mess_filepath_str = ''
	for fnm in Filepaths:
		mess_filepath_str += ' ' + fnm
	
	os.system('python AVG_Main_Analysis.py --mypaths' + mess_filepath_str + ' --channel AE_0 --fs 1.e6 --mode 1_avg_spectrum --name AvgEnvFft_MessE_Test_' + str(k) + ' --db_out 37 --units mv --sqr_envelope OFF --demodulation ON --filter OFF --freq_lp 95.e3 --freq_hp 140.e3 --wv_deco WPD --wv_mother db6 --wv_crit kurt --wv_approx OFF --level 5')
sys.exit()
for i in range(12):
	k = i+1
	# mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180306_Messung_E\\Test_' + str(k)
	mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_O\\Test_' + str(k)
	
	extension = 'dms'
	Filepaths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) if f[-3:] == extension if f[1] == 'E']	

	mess_filepath_str = ''
	for fnm in Filepaths:
		mess_filepath_str += ' ' + fnm
	
	os.system('python AVG_Main_Analysis.py --mypaths' + mess_filepath_str + ' --channel AE_0 --fs 1.e6 --mode 1_avg_spectrum --name AvgEnvFft_MessO_Test_' + str(k) + ' --db_out 37 --units mv --sqr_envelope OFF --demodulation ON --filter OFF --freq_lp 95.e3 --freq_hp 140.e3 --wv_deco OFF --wv_mother db6 --wv_crit kurt --wv_approx OFF --level 5')
sys.exit()	
# myp++++++++++++++++++

for i in range(12):
	k = i+1
	mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180306_Messung_E\\Test_' + str(k)
	# mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_O\\Test_' + str(k)
	
	extension = 'dms'
	Filepaths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) if f[-3:] == extension if f[1] == 'E']	

	mess_filepath_str = ''
	for fnm in Filepaths:
		mess_filepath_str += ' ' + fnm
	
	os.system('python AVG_Main_Analysis.py --mypaths' + mess_filepath_str + ' --channel AE_0 --fs 1.e6 --mode 1_avg_spectrum --name AvgEnvFft_MessE_Test_' + str(k) + ' --db_out 37 --units mv --sqr_envelope OFF --demodulation ON --filter OFF --freq_lp 95.e3 --freq_hp 140.e3 --wv_deco WPD --wv_mother db6 --wv_crit kurt_sen --wv_approx OFF --level 5')

for i in range(12):
	k = i+1
	# mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180306_Messung_E\\Test_' + str(k)
	mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_O\\Test_' + str(k)
	
	extension = 'dms'
	Filepaths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) if f[-3:] == extension if f[1] == 'E']	

	mess_filepath_str = ''
	for fnm in Filepaths:
		mess_filepath_str += ' ' + fnm
	
	os.system('python AVG_Main_Analysis.py --mypaths' + mess_filepath_str + ' --channel AE_0 --fs 1.e6 --mode 1_avg_spectrum --name AvgEnvFft_MessO_Test_' + str(k) + ' --db_out 37 --units mv --sqr_envelope OFF --demodulation ON --filter OFF --freq_lp 95.e3 --freq_hp 140.e3 --wv_deco WPD --wv_mother db6 --wv_crit kurt_sen --wv_approx OFF --level 5')


