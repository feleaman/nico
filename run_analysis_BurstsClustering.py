import os
from os import listdir
from os.path import join, isdir, basename, dirname, isfile
import sys

# # GEARBOX AMT FULL+++++++++++++++++++++++++++++++++++++++++++++++++











# Mypaths1 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180314_Messung_G\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180315_Messung_H\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_I\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_J\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180322_Messung_L\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_N\\Warm']



# Mypaths2 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180314_Messung_G\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180315_Messung_H\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_I\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_J\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long1', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long2', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180322_Messung_L\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_N\\Long']


Mypaths = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180314_Messung_G\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180315_Messung_H\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_I\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_J\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180322_Messung_L\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_N\\Warm']

Mypaths2 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180314_Messung_G\\Warm\\Tacho', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180315_Messung_H\\Warm\\Tacho', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_I\\Warm\\Tacho', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_J\\Warm\\Tacho', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Warm\\Tacho', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180322_Messung_L\\Warm\\Tacho', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Warm\\Tacho', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_N\\Warm\\Tacho']

# Mypaths = Mypaths1 + Mypaths2 + Mypaths3
# # Mypaths = Mypaths2 + Mypaths3



# Names1 = ['G_0_Warm_Bursts_db6_lvl7_thr08_NoFilt_ALT', 'H_0_Warm_Bursts_db6_lvl7_thr08_NoFilt_ALT','I_0_Warm_Bursts_db6_lvl7_thr08_NoFilt_ALT','J_0_Warm_Bursts_db6_lvl7_thr08_NoFilt_ALT','K0_0_Warm_Bursts_db6_lvl7_thr08_NoFilt_ALT','L_0_Warm_Bursts_db6_lvl7_thr08_NoFilt_ALT','M_0_Warm_Bursts_db6_lvl7_thr08_NoFilt_ALT','N_0_Warm_Bursts_db6_lvl7_thr08_NoFilt_ALT']

# Names2 = ['G_1_Long_Bursts_db6_lvl7_thr08_NoFilt_ALT', 'H_1_Long_Bursts_db6_lvl7_thr08_NoFilt_ALT','I_1_Long_Bursts_db6_lvl7_thr08_NoFilt_ALT','J_1_Long_Bursts_db6_lvl7_thr08_NoFilt_ALT','K1_1_Long_Bursts_db6_lvl7_thr08_NoFilt_ALT', 'K2_1_Long_Bursts_db6_lvl7_thr08_NoFilt_ALT', 'L_1_Long_Bursts_db6_lvl7_thr08_NoFilt_ALT','M_1_Long_Bursts_db6_lvl7_thr08_NoFilt_ALT','N_1_Long_Bursts_db6_lvl7_thr08_NoFilt_ALT']

Names = ['G_0_Warm_Bursts_Teeth', 'H_0_Warm_Bursts_Teeth','I_0_Warm_Bursts_Teeth','J_0_Warm_Bursts_Teeth','K_0_Warm_Bursts_Teeth', 'L_0_Warm_Bursts_Teeth','M_0_Warm_Bursts_Teeth','N_0_Warm_Bursts_Teeth']



# # Names1 = ['G_0_Warm_Bursts_Feat_thr08_NoFilt', 'H_0_Warm_Bursts_Feat_thr08_NoFilt','I_0_Warm_Bursts_Feat_thr08_NoFilt','J_0_Warm_Bursts_Feat_thr08_NoFilt','K0_0_Warm_Bursts_Feat_thr08_NoFilt','L_0_Warm_Bursts_Feat_thr08_NoFilt','M_0_Warm_Bursts_Feat_thr08_NoFilt','N_0_Warm_Bursts_Feat_thr08_NoFilt']

# # Names2 = ['G_1_Long_Bursts_Feat_thr08_NoFilt', 'H_1_Long_Bursts_Feat_thr08_NoFilt','I_1_Long_Bursts_Feat_thr08_NoFilt','J_1_Long_Bursts_Feat_thr08_NoFilt','K1_1_Long_Bursts_Feat_thr08_NoFilt', 'K2_1_Long_Bursts_Feat_thr08_NoFilt', 'L_1_Long_Bursts_Feat_thr08_NoFilt','M_1_Long_Bursts_Feat_thr08_NoFilt','N_1_Long_Bursts_Feat_thr08_NoFilt']

# # Names3 = ['G_2_Last_Bursts_Feat_thr08_NoFilt', 'H_2_Last_Bursts_Feat_thr08_NoFilt','I_2_Last_Bursts_Feat_thr08_NoFilt','J_2_Last_Bursts_Feat_thr08_NoFilt','K1_2_Last_Bursts_Feat_thr08_NoFilt', 'K2_2_Last_Bursts_Feat_thr08_NoFilt', 'L_2_Last_Bursts_Feat_thr08_NoFilt','M_2_Last_Bursts_Feat_thr08_NoFilt','N_2_Last_Bursts_Feat_thr08_NoFilt']

# Names = Names1 + Names2 + Names3
# # Names = Names2 + Names3





count = 7
for mypath, mypath2, name in zip(Mypaths, Mypaths2, Names):
	os.system('python Bursts_Clustering.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --plot OFF --save ON --mode burst_detection_per_tacho --thr_value 0.7 --thr_mode fixed_value --name ' + name + str(count) + ' --stella 300 --lockout 300 --window_time 0.001 --save ON --fs_tacho 50.e3 --mypath2 ' + mypath2)
	


sys.exit()


# # GEARBOX AMT+++++++++++++++++++++++++++++++++++++++++++++++++
Mypaths = ['C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\04_Data\\AE_Data\\Extended_End\\N']
# Names = ['TestBench_WPD_db6_lvl7_Bursts_0p7mv_300_300_0p001_mvin_NoFilter_ALTERED']
Names = ['TestBench_Bursts_0p7mv_300_300_0p001_mvin_NoFilter']

# Channels = ['AE_1', 'AE_2']

# Channels = ['AE_3']
# highpass 5.e3 3
# detector_wpd_thr_nocorr
count = 0
for mypath, name in zip(Mypaths, Names):
	count += 1
	os.system('python M_Bursts_Detector.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --plot ON --save_plot OFF --mode detector_wpd_thr_nocorr --thr_value 0.7 --thr_mode fixed_value --name ' + name + str(count) + ' --filter highpass 5.e3 3 --stella 300 --lockout 300 --db_out 37 --unit mv --window_time 0.001 --level 7 --save ON --wv_mother db6 --inverse_wv OFF --idx_fp1 0 --idx_fp2 0 --short_burst OFF')


sys.exit()






# # SCHOTTLAND+++++++++++++++++++++++++++++++++++++++++++++++++
Mypaths = ['C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\04_Data\\Schottland\\S1pai']
Names = ['S1pai_12rms_600_600_0p002_mvin']
# 
Channels = ['AE_1', 'AE_2', 'AE_3']
# Channels = ['AE_1', 'AE_2']

# Channels = ['AE_3']

count = 0
for mypath, name in zip(Mypaths, Names):
	for channel in Channels:
		count += 1
		os.system('python M_Bursts_Detector.py --mypath ' + mypath + ' --channel ' + channel +' --fs 1.e6 --plot OFF --save_plot ON --mode detector_thr --thr_value 12 --thr_mode factor_rms --name ' + name + '_' + channel + '_' + str(count) + ' --filter OFF --stella 600 --lockout 600 --range 0 30 --db_out 49 --unit mv --window_time 0.002 --level 0 --save ON --wv_mother AAA --inverse_wv OFF --idx_fp1 7 --idx_fp2 24 --short_burst OFF')


sys.exit()


#BOCHUM+++++++++++++++++++++++++++++++++++++++++++++++++
Mypaths = ['C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\04_Data\\Bochum\\B4c']
Names = ['B4c_Short_Bursts_PDT_left_12rms_600_600_200_0p002_mvin_WPD_db6_lvl5']

Channels = ['AE_1', 'AE_2', 'AE_3']
# Channels = ['AE_1']

count = 0
for mypath, name in zip(Mypaths, Names):
	for channel in Channels:
		count += 1
		os.system('python M_Bursts_Detector.py --mypath ' + mypath + ' --channel ' + channel +' --fs 1.e6 --plot OFF --save_plot ON --mode detector_wpd_thr --thr_value 12 --thr_mode factor_rms --name ' + name + '_' + channel + '_' + str(count) + ' --filter OFF --stella 600 --lockout 600 --range 0 10 --db_out 43 --unit mv --window_time 0.002 --level 5 --save ON --wv_mother db6 --inverse_wv OFF  --idx_fp1 7 --idx_fp2 24 --short_burst PDT_left')


sys.exit()
# 
# detector_thr_7exp
		
		
# # #CWD+++++++++++++++++++++++++++++++++++++++++++++++++
# Mypaths = ['C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\04_Data\\CWD\\Damage\\AE']
# Names = ['CWD_Damage_Bursts_7rms_600_600_0p002_mvin_5exp']
# Channels = ['AE_0']
# count = 0
# for mypath, name in zip(Mypaths, Names):
	# for channel in Channels:
		# count += 1
		# os.system('python M_Bursts_Detector.py --mypath ' + mypath + ' --channel ' + channel +' --fs 1.e6 --plot ON --save_plot OFF --mode detector_thr --thr_value 7 --thr_mode factor_rms --name ' + name + '_' + channel + '_' + str(count) + ' --filter OFF --stella 600 --lockout 600 --range 0 5 --db_out 37 --unit mv --window_time 0.002')
# sys.exit()
		
# #AMT+++++++++++++++++++++++++++++++++++++++++++++++++
# Mypaths = ['C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\04_Data\\AE_Data\\Extended_End\\G']
# Names = ['MessG_Bursts_7rms_300_300_0p001_mvin_5exp']
# Channels = ['AE_0']
# count = 0
# for mypath, name in zip(Mypaths, Names):
	# for channel in Channels:
		# count += 1
		# os.system('python M_Bursts_Detector.py --mypath ' + mypath + ' --channel ' + channel +' --fs 1.e6 --plot OFF --save_plot ON --mode detector_thr --thr_value 7 --thr_mode factor_rms --name ' + name + '_' + channel + '_' + str(count) + ' --filter highpass 20.e3 3 --stella 300 --lockout 300 --range 0 5 --db_out 37 --unit mv --window_time 0.001')


# Mypaths = ['C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\04_Data\\AE_Data\\Extended_End\\A']
# Names = ['MessA_Bursts_7rms_300_300_0p001_mvin_5exp']
# Channels = ['AE_0']
# count = 0
# for mypath, name in zip(Mypaths, Names):
	# for channel in Channels:
		# count += 1
		# os.system('python M_Bursts_Detector.py --mypath ' + mypath + ' --channel ' + channel +' --fs 1.e6 --plot OFF --save_plot ON --mode detector_thr --thr_value 7 --thr_mode factor_rms --name ' + name + '_' + channel + '_' + str(count) + ' --filter highpass 20.e3 3 --stella 300 --lockout 300 --range 0 5 --db_out 37 --unit mv --window_time 0.001')
	
# Mypaths = ['C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\04_Data\\AE_Data\\Extended_End\\N']
# Names = ['MessN_Bursts_7rms_300_300_0p001_mvin_5exp']
# Channels = ['AE_0']
# count = 0
# for mypath, name in zip(Mypaths, Names):
	# for channel in Channels:
		# count += 1
		# os.system('python M_Bursts_Detector.py --mypath ' + mypath + ' --channel ' + channel +' --fs 1.e6 --plot OFF --save_plot ON --mode detector_thr --thr_value 7 --thr_mode factor_rms --name ' + name + '_' + channel + '_' + str(count) + ' --filter highpass 20.e3 3 --stella 300 --lockout 300 --range 0 5 --db_out 37 --unit mv --window_time 0.001')
# sys.exit()
Mypaths = ['C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\04_Data\\AE_Data\\Extended_End\\Q']
Names = ['MessQ_Bursts_7rms_300_300_0p001_mvin_5exp']
Channels = ['AE_0']
count = 0
for mypath, name in zip(Mypaths, Names):
	for channel in Channels:
		count += 1
		os.system('python M_Bursts_Detector.py --mypath ' + mypath + ' --channel ' + channel +' --fs 1.e6 --plot OFF --save_plot ON --mode detector_thr --thr_value 7 --thr_mode factor_rms --name ' + name + '_' + channel + '_' + str(count) + ' --filter highpass 20.e3 3 --stella 300 --lockout 300 --range 0 5 --db_out 37 --unit mv --window_time 0.001')