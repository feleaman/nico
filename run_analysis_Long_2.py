import os
from os import listdir
from os.path import join, isdir, basename, dirname, isfile
import sys
import pandas as pd



Mypaths1 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180314_Messung_G\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180315_Messung_H\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_I\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_J\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180322_Messung_L\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_N\\Warm']



Mypaths2 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180314_Messung_G\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180315_Messung_H\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_I\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_J\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long1', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long2', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180322_Messung_L\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_N\\Long']



Mypaths3 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180314_Messung_G\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180315_Messung_H\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_I\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_J\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long1\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long2\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180322_Messung_L\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_N\\Long\\Last_720']


 


Mypaths = Mypaths1 + Mypaths2 + Mypaths3



Names1 = ['G_0_Warm_AE_EnvFft', 'H_0_Warm_AE_EnvFft','I_0_Warm_AE_EnvFft','J_0_Warm_AE_EnvFft','K0_0_Warm_AE_EnvFft','L_0_Warm_AE_EnvFft','M_0_Warm_AE_EnvFft','N_0_Warm_AE_EnvFft']


Names2 = ['G_1_Long_AE_EnvFft', 'H_1_Long_AE_EnvFft','I_1_Long_AE_EnvFft','J_1_Long_AE_EnvFft','K1_1_Long_AE_EnvFft', 'K2_1_Long_AE_EnvFft', 'L_1_Long_AE_EnvFft','M_1_Long_AE_EnvFft','N_1_Long_AE_EnvFft']


Names3 = ['G_2_Last_AE_EnvFft', 'H_2_Last_AE_EnvFft','I_2_Last_AE_EnvFft','J_2_Last_AE_EnvFft','K1_2_Last_AE_EnvFft', 'K2_2_Last_AE_EnvFft', 'L_2_Last_AE_EnvFft','M_2_Last_AE_EnvFft','N_2_Last_AE_EnvFft']


Names = Names1 + Names2 + Names3


Mypaths = ['M:\\Betriebsmessungen\\WEA-Getriebe-Eickhoff\\Durchführung\\Messdaten\\PXI\\Warmlauf_Trigger_Defekt\\tdms\\AE', 'M:\\Betriebsmessungen\\WEA-Getriebe-Eickhoff\\Durchführung\\Messdaten\\PXI\\Warmlauf_Trigger_Defekt_2\\tdms\\AE']

# Mypaths = ['M:\\Betriebsmessungen\\WEA-Getriebe-Eickhoff\\Durchführung\\Messdaten\\PXI\\Warmlauf_Trigger_Defekt_2\\tdms\\AE']

# Mypaths = ['M:\\Betriebsmessungen\\WEA-Getriebe-Eickhoff\\Durchführung\\Messdaten\\PXI\\Warmlauf_Ohne_Trigger\\tdms\\AE']

Names = ['Bochum_Temp_Ramp_A', 'Bochum_Temp_Ramp_B']


wv_mother = 'db6'
Channels = ['AE_3']

level = '8'
count = 7
# for wv_mother in WV_Mothers:
for channel in Channels:
	for mypath, name in zip(Mypaths, Names):
		os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel ' + channel + ' --fs 1.e6 --mode overall_features_select --name ' + name + '_' + channel + ' --db_out 49 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 OFF --features_g2 OFF --features_g3 OFF --features_g4 OFF --features_g5 OFF --wv_deco OFF --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit kurt_sen --level ' + level)
		
		# os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode magnitude_freq_comps --name ' + name + '_Idx_' + str(count) + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation ON --filter OFF --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 OFF --features_g2 OFF --features_g3 OFF --features_g4 OFF --features_g5 OFF --wv_deco ON --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit kurt_sen --level ' + level)
	


sys.exit()













# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ RUN AUTOMATIC WAVELET, KURTOGRAM, BP FILTER, RAW +++++++++++++ +++++++++++++ +++++++++++++ +++++++++++++ +++++++++++++ +++++++++++++ ++++++++++
# mypath = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\04_Data\\AE_Data\\End_EMD'

print('comienza!!!!!!!!')
mypath = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\04_Data\\Schottland\\S1'

print('+++++++ WAVELETS ++++++++++++++++++++++++')
wv_mother = 'db6'
Channels = ['AE_1', 'AE_2', 'AE_3']
level = '5'
count = 2
for channel in Channels:
	s_channel = channel.replace('_', '')
	os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel ' + channel + ' --fs 1.e6 --mode overall_features --name S1_' + s_channel + '_Wfm_Idx_' + str(count) + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco WPD --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit coef_one_level --idx_levels 24 --range 0 30 --level ' + level)



mypath = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\04_Data\\Bochum\\B4c'

print('+++++++ WAVELETS ++++++++++++++++++++++++')
wv_mother = 'db6'
Channels = ['AE_1', 'AE_2', 'AE_3']
level = '5'
count = 2
for channel in Channels:
	s_channel = channel.replace('_', '')
	os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel ' + channel + ' --fs 1.e6 --mode overall_features --name B4c_' + s_channel + '_Wfm_Idx_' + str(count) + ' --db_out 43 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco WPD --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit coef_one_level --idx_levels 24 --range 0 10 --level ' + level)




mypath = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\04_Data\\Schottland\\S1'

print('+++++++ iWAVELETS ++++++++++++++++++++++++')
wv_mother = 'db6'
Channels = ['AE_1', 'AE_2', 'AE_3']
level = '5'
count = 3
for channel in Channels:
	s_channel = channel.replace('_', '')
	os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel ' + channel + ' --fs 1.e6 --mode overall_features --name S1_' + s_channel + '_Wfm_Idx_' + str(count) + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco WPD --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit inverse_levels --idx_levels 23 24 25 --range 0 30 --level ' + level)


mypath = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\04_Data\\Bochum\\B4c'

print('+++++++ iWAVELETS ++++++++++++++++++++++++')
wv_mother = 'db6'
Channels = ['AE_1', 'AE_2', 'AE_3']
level = '5'
count = 3
for channel in Channels:
	s_channel = channel.replace('_', '')
	os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel ' + channel + ' --fs 1.e6 --mode overall_features --name B4c_' + s_channel + '_Wfm_Idx_' + str(count) + ' --db_out 43 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco WPD --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit inverse_levels --idx_levels 23 24 25 --range 0 10 --level ' + level)