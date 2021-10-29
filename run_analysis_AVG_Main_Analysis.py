import os
from os import listdir
from os.path import join, isdir, basename, dirname, isfile
import sys
import pandas as pd


# mypath = 'C:\\Felix\\60_Comparison_AE_ACC\\04_Data\\CWD\\NoDamage\\AE'
mypath = 'C:\\Felix\\60_Comparison_AE_ACC\\04_Data\\CWD\\Damage\\AE'

extension = 'dms'
Filepaths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) if f[-3:] == extension]
print(Filepaths)

mess_filepath_str = ''
for fnm in Filepaths:
	mess_filepath_str += ' ' + fnm
print(mess_filepath_str)
# sys.exit()
os.system('python AVG_Main_Analysis.py --mypaths' + mess_filepath_str + ' --channel AE_0 --fs 1.e6 --mode 1_avg_spectrum --name CWD_Damage_WPD_lvl7_db6_SQR_MPR_GearMesh4tol_Window_10_AE0 --db_out 37 --units mv --sqr_envelope ON --demodulation ON --filter OFF --freq_lp 95.e3 --freq_hp 140.e3 --wv_deco WPD --wv_mother db6 --wv_crit mpr --wv_approx OFF --level 7')

os.system('python AVG_Main_Analysis.py --mypaths' + mess_filepath_str + ' --channel AE_1 --fs 1.e6 --mode 1_avg_spectrum --name CWD_Damage_WPD_lvl7_db6_SQR_MPR_GearMesh4tol_Window_10_AE1 --db_out 37 --units mv --sqr_envelope ON --demodulation ON --filter OFF --freq_lp 95.e3 --freq_hp 140.e3 --wv_deco WPD --wv_mother db6 --wv_crit mpr --wv_approx OFF --level 7')

os.system('python AVG_Main_Analysis.py --mypaths' + mess_filepath_str + ' --channel AE_2 --fs 1.e6 --mode 1_avg_spectrum --name CWD_Damage_WPD_lvl7_db6_SQR_MPR_GearMesh4tol_Window_10_AE2 --db_out 37 --units mv --sqr_envelope ON --demodulation ON --filter OFF --freq_lp 95.e3 --freq_hp 140.e3 --wv_deco WPD --wv_mother db6 --wv_crit mpr --wv_approx OFF --level 7')

sys.exit()

mypath = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\04_Data\\AE_Data\\Extended_End'

extension = 'dms'
Filepaths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) if f[-3:] == extension]
Filepaths_A = [f for f in Filepaths if f.find('20180227') != -1]
Filepaths_G = [f for f in Filepaths if f.find('20180314') != -1]
Filepaths_J = [f for f in Filepaths if f.find('20180316') != -1]
Filepaths_L = [f for f in Filepaths if f.find('20180322') != -1]
Filepaths_N = [f for f in Filepaths if f.find('20180511') != -1]
Filepaths_Q = [f for f in Filepaths if f.find('20181102') != -1]
Mess_Filepaths = []
Mess_Filepaths.append(Filepaths_A)
Mess_Filepaths.append(Filepaths_G)
Mess_Filepaths.append(Filepaths_J)
Mess_Filepaths.append(Filepaths_L)
Mess_Filepaths.append(Filepaths_N)
Mess_Filepaths.append(Filepaths_Q)


# Mess_Filepaths = Filepaths_L
# print(Mess_Filepaths)

# WV_Mothers = ['db1', 'db6', 'db12', 'db18', 'sym6', 'sym12', 'sym18']
WV_Mothers = ['db6']
idx_levels = '22 23 24 25'
# wv_mother = 'db1'
# Levels = ['3', '5', '7', '9']
count = 888
level = '8'

for wv_mother in WV_Mothers:
	for mess_filepath in Mess_Filepaths:
		mess_filepath_str = ''
		for fnm in mess_filepath:
			mess_filepath_str += ' ' + fnm
		# print(mess_filepath_str)
		# sys.exit()
		
		os.system('python AVG_Main_Analysis.py --mypaths' + mess_filepath_str + ' --channel AE_0 --fs 1.e6 --mode 1_avg_spectrum --name eq2Idx_' + str(count) + ' --db_out 37 --units mv --sqr_envelope OFF --demodulation ON --filter OFF --freq_lp 95.e3 --freq_hp 140.e3 --wv_deco iDWT --wv_mother ' + wv_mother + ' --wv_crit kurt_sen --wv_approx OFF --idx_levels ' + idx_levels + ' --level ' + level)
		
		# os.system('python AVG_Main_Analysis.py --channel AE_0 --fs 1.e6 --mode 1_avg_spectrum --name Idx_' + str(count) + ' --db_out 37 --units mv --sqr_envelope OFF --demodulation ON --filter OFF --freq_lp 95.e3 --freq_hp 140.e3 --wv_deco DWT --wv_mother ' + wv_mother + ' --wv_crit kurt_sen --wv_approx OFF --idx_levels ' + idx_levels + ' --level ' + level)
		
		# os.system('python AVG_Main_Analysis.py --mypaths' + mess_filepath_str + ' --channel ' + channel + ' --fs 1.e6 --mode 1_avg_spectrum --name ' + channel + '_10s_Idx_' + str(count) + ' --db_out 49 --range 0 10 --units mv --sqr_envelope OFF --demodulation ON --filter OFF --freq_lp 320.e3 --freq_hp 400.e3 --wv_deco OFF --wv_mother ' + wv_mother + ' --wv_crit inverse_levels --wv_approx OFF --level ' + level + ' --idx_levels ' + idx_levels)
		
		
	count += 1
sys.exit()




# mypath = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\04_Data\\Bochum\\B4c'
# extension = 'dms'
# Filepaths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) if f[-3:] == extension]

# Channels = ['AE_1', 'AE_2', 'AE_3']
# # Channels = ['AE_3']


# count = 1
# level = '5'
# wv_mother = 'db6'
# idx_levels = '22 23 24 25'
# # for wv_mother in WV_Mothers:
# # for level in Levels:

# mess_filepath_str = ''
# for filepath in Filepaths:
	# mess_filepath_str += ' ' + filepath
# for channel in Channels:
	# os.system('python AVG_Main_Analysis.py --mypaths' + mess_filepath_str + ' --channel ' + channel + ' --fs 1.e6 --mode 1_avg_spectrum --name ' + channel + '_Idx_' + str(count) + ' --db_out 43 --range 0 10 --units mv --sqr_envelope OFF --demodulation ON --filter OFF --freq_lp 320.e3 --freq_hp 400.e3 --wv_deco OFF --wv_mother ' + wv_mother + ' --wv_crit inverse_levels --wv_approx OFF --level ' + level + ' --idx_levels ' + idx_levels)
# count += 1


# sys.exit()



mypath = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\04_Data\\Schottland\\S1pai'

extension = 'dms'
Filepaths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) if f[-3:] == extension]

Channels = ['AE_1', 'AE_2', 'AE_3']
# Channels = ['AE_3']


count = 1
level = '5'
wv_mother = 'db6'
# idx_levels = str(24)
idx_levels = '22 23 24 25'

# for wv_mother in WV_Mothers:
# for level in Levels:

# mess_filepath_str = ''
# for filepath in Filepaths:
	# mess_filepath_str += ' ' + filepath
# for channel in Channels:
	# os.system('python AVG_Main_Analysis.py --mypaths' + mess_filepath_str + ' --channel ' + channel + ' --fs 1.e6 --mode 1_avg_spectrum --name ' + channel + '_30s_Idx_' + str(count) + ' --db_out 37 --range 0 30 --units mv --sqr_envelope OFF --demodulation ON --filter OFF --freq_lp 320.e3 --freq_hp 400.e3 --wv_deco WPD --wv_mother ' + wv_mother + ' --wv_crit coef_one_level --wv_approx OFF --level ' + level + ' --idx_levels ' + idx_levels)

mess_filepath_str = ''
for filepath in Filepaths:
	mess_filepath_str += ' ' + filepath
for channel in Channels:
	os.system('python AVG_Main_Analysis.py --mypaths' + mess_filepath_str + ' --channel ' + channel + ' --fs 1.e6 --mode 1_avg_spectrum --name ' + channel + '_10s_Idx_' + str(count) + ' --db_out 49 --range 0 10 --units mv --sqr_envelope OFF --demodulation ON --filter OFF --freq_lp 320.e3 --freq_hp 400.e3 --wv_deco OFF --wv_mother ' + wv_mother + ' --wv_crit inverse_levels --wv_approx OFF --level ' + level + ' --idx_levels ' + idx_levels)
count += 1
# inverse_levels
# coef_one_level
