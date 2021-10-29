import os
from os import listdir
from os.path import join, isdir, basename, dirname, isfile
import sys
import pandas as pd




mypath = 'C:\\Felix\\63_AE_AC_Fusion\\04_Data\\CWD\\channel_0\\damage\\ocult'
os.system('python Long_Analysis.py --channel xxx --mypath ' + mypath + ' --fs 1.e6 --mode overall_features --name CWD_Ch0_CC_nmax_Damage --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std ON')

mypath = 'C:\\Felix\\63_AE_AC_Fusion\\04_Data\\CWD\\channel_0\\nodamage\\ocult'
os.system('python Long_Analysis.py --channel xxx --mypath ' + mypath + ' --fs 1.e6 --mode overall_features --name CWD_Ch0_CC_nmax_NoDamage --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std ON')



mypath = 'C:\\Felix\\63_AE_AC_Fusion\\04_Data\\CWD\\channel_1\\damage\\ocult'
os.system('python Long_Analysis.py --channel xxx --mypath ' + mypath + ' --fs 1.e6 --mode overall_features --name CWD_Ch1_CC_nmax_Damage --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std ON')

mypath = 'C:\\Felix\\63_AE_AC_Fusion\\04_Data\\CWD\\channel_1\\nodamage\\ocult'
os.system('python Long_Analysis.py --channel xxx --mypath ' + mypath + ' --fs 1.e6 --mode overall_features --name CWD_Ch1_CC_nmax_NoDamage --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std ON')





mypath = 'C:\\Felix\\63_AE_AC_Fusion\\04_Data\\CWD\\channel_2\\damage\\ocult'
os.system('python Long_Analysis.py --channel xxx --mypath ' + mypath + ' --fs 1.e6 --mode overall_features --name CWD_Ch2_CC_nmax_Damage --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std ON')

mypath = 'C:\\Felix\\63_AE_AC_Fusion\\04_Data\\CWD\\channel_2\\nodamage\\ocult'
os.system('python Long_Analysis.py --channel xxx --mypath ' + mypath + ' --fs 1.e6 --mode overall_features --name CWD_Ch2_CC_nmax_NoDamage --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std ON')
sys.exit()


# #LUEGO CAMBIAR A MESS O EN EL NOMBRE Y PATH!!!!!!!!
# mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\CC_MM\\MessE\\CC_ENV'
# os.system('python Long_Analysis.py --channel xxx --mypath ' + mypath + ' --fs 1.e6 --mode overall_features --name CC_ENV_nmax_OvFeatures_MessE --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std ON')

# mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\CC_MM\\MessE\\CC_SQR'
# os.system('python Long_Analysis.py --channel xxx --mypath ' + mypath + ' --fs 1.e6 --mode overall_features --name CC_SQR_nmax_OvFeatures_MessE --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std ON')

# mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\CC_MM\\MessE\\MM_ENV'
# os.system('python Long_Analysis.py --channel xxx --mypath ' + mypath + ' --fs 1.e6 --mode overall_features --name MM_ENV_nmax_OvFeatures_MessE --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std ON')

# mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\CC_MM\\MessE\\MM_SQR'
# os.system('python Long_Analysis.py --channel xxx --mypath ' + mypath + ' --fs 1.e6 --mode overall_features --name MM_SQR_nmax_OvFeatures_MessE --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std ON')
# sys.exit()




# sys.exit() + 7 + 1
for i in range(12):
	k = i+1
	mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180306_Messung_E\\Test_' + str(k)
	# mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_O\\Test_' + str(k)
	
	# os.system('python Cross_Correlation_Analysis.py --mode auto_cross_correlation --save OFF --channel OFF --mypath ' + mypath)	
	
	os.system('python Long_Analysis.py --channel AE_0 --mypath ' + mypath + ' --fs 1.e6 --mode overall_features --name AE_ENV_nmax_MessE_Test_' + str(k) + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation ON --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std OFF --norm_max ON')
	
	
	os.system('python Long_Analysis.py --channel ACC_0 --mypath ' + mypath + ' --fs 50.e3 --mode overall_features --name AC_ENV_nmax_MessE_Test_' + str(k) + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation ON --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std OFF --norm_max ON')
	

# for i in range(12):
	# k = i+1
	# # mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180306_Messung_E\\Test_' + str(k)
	# mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_O\\Test_' + str(k)
	
	# # os.system('python Cross_Correlation_Analysis.py --mode auto_cross_correlation --save OFF --channel OFF --mypath ' + mypath)	
	
	# os.system('python Long_Analysis.py --channel AE_0 --mypath ' + mypath + ' --fs 1.e6 --mode overall_features --name AE_ENV_nmax_MessO_Test_' + str(k) + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation ON --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std OFF --norm_max ON')
	
	
	# os.system('python Long_Analysis.py --channel ACC_0 --mypath ' + mypath + ' --fs 50.e3 --mode overall_features --name AC_ENV_nmax_MessO_Test_' + str(k) + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation ON --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std OFF --norm_max ON')
	
sys.exit()
for i in range(12):
	k = i+6
	mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_O\\Test_' + str(k)
	os.system('python Long_Analysis.py --channel AE_0 --mypath ' + mypath + ' --fs 1.e6 --mode overall_features --name AE_0_OvFeatures_MessO_Test_' + str(k) + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std ON')
	os.system('python Long_Analysis.py --channel ACC_0 --mypath ' + mypath + ' --fs 50.e3 --mode overall_features --name AC_0_OvFeatures_MessO_Test_' + str(k) + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7 --add_mean_std ON')
# os.system('python Long_Analysis.py --channel AE_0 --mypath ' + mypath + ' --fs 1.e6 --mode overall_features --name AE_0_OvFeatures_MessE_Test1 --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7')
# os.system('python Long_Analysis.py --channel AC_0 --mypath ' + mypath + ' --fs 50.e3 --mode overall_features --name AC_0_OvFeatures_MessE_Test1 --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother db6 --wv_approx OFF --wv_crit kurt --level 7')





sys.exit()


#+++++Calculo MessA: AE-AC
# mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180227_Messung_A\\Long\\Last_720'
# name = 'AE_0_WPD_db6_kurt_lvl_5_OvFeatures_MessA'
# level = '5'
# wv_mother = 'db6'

# os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode overall_features --name ' + name + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco WPD --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit kurt --level ' + level)

# name = 'AC_0_WPD_db6_kurt_lvl_5_OvFeatures_MessA'
# os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel ACC_0 --fs 50.e3 --mode overall_features --name ' + name + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco WPD --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit kurt --level ' + level)

#+++++Calculo MessQ: AE
mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181102_Messung_Q\\Long\\Last_720'
# name = 'AE_0_WPD_db6_kurt_lvl_5_OvFeatures_MessQ'
# level = '5'
# wv_mother = 'db6'
# os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode overall_features --name ' + name + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco WPD --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit kurt --level ' + level)

#+++++Calculo MessP: AC
# mypath = 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181031_Messung_P\\Long\\Last_720'
name = 'ACC_0_WPD_db6_kurt_lvl_5_OvFeatures_MessQ'
level = '5'
wv_mother = 'db6'
os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel ACC_0 --fs 50.e3 --mode overall_features --name ' + name + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco WPD --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit kurt --level ' + level)


name = 'ACC_0_Kurto_lvl_7_OvFeatures_MessQ'
level = '7'
wv_mother = 'db6'
os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel ACC_0 --fs 50.e3 --mode overall_features --name ' + name + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter kurtogram --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit kurt --level ' + level)

name = 'ACC_0_OvFeatures_MessQ'
level = '7'
wv_mother = 'db6'
os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel ACC_0 --fs 50.e3 --mode overall_features --name ' + name + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit kurt --level ' + level)

# name = 'ACC_2_WPD_db6_kurt_lvl_5_OvFeatures_MessP'
# os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel ACC_2 --fs 50.e3 --mode overall_features --name ' + name + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco WPD --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit kurt --level ' + level)


# name = 'AC_2_Kurto_lvl_7_OvFeatures_MessP'
# os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel ACC_2 --fs 50.e3 --mode overall_features --name ' + name + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter kurtogram --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco OFF --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit kurt --level ' + level)



sys.exit()

Names1 = ['G_0_Warm_AE_EnvFft', 'H_0_Warm_AE_EnvFft','I_0_Warm_AE_EnvFft','J_0_Warm_AE_EnvFft','K0_0_Warm_AE_EnvFft','L_0_Warm_AE_EnvFft','M_0_Warm_AE_EnvFft','N_0_Warm_AE_EnvFft']


Names2 = ['G_1_Long_AE_EnvFft', 'H_1_Long_AE_EnvFft','I_1_Long_AE_EnvFft','J_1_Long_AE_EnvFft','K1_1_Long_AE_EnvFft', 'K2_1_Long_AE_EnvFft', 'L_1_Long_AE_EnvFft','M_1_Long_AE_EnvFft','N_1_Long_AE_EnvFft']

# Names2 = ['M_1_Long_AE_EnvFft','N_1_Long_AE_EnvFft']


Names3 = ['G_2_Last_AE_EnvFft', 'H_2_Last_AE_EnvFft','I_2_Last_AE_EnvFft','J_2_Last_AE_EnvFft','K1_2_Last_AE_EnvFft', 'K2_2_Last_AE_EnvFft', 'L_2_Last_AE_EnvFft','M_2_Last_AE_EnvFft','N_2_Last_AE_EnvFft']




Names = Names1 + Names2 + Names3
# Names = Names2 + Names3



# Mypaths = ['C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\04_Data\\AE_Data\\Extended_End']
# Names = ['AE_Wfm']

# wv_mother = 'db6'

# WV_Mothers = ['db6', 'db12', 'db18', 'sym6', 'sym12', 'sym18']
WV_Mothers = ['db6']


level = '8'
count = 7
# for wv_mother in WV_Mothers:
for wv_mother in WV_Mothers:
	for mypath, name in zip(Mypaths, Names):
		# os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode overall_features_select --name ' + name + '_Idx_' + str(count) + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 OFF --features_g2 OFF --features_g3 OFF --features_g4 OFF --features_g5 OFF --wv_deco ON --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit kurt_sen --level ' + level)
		
		os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode overall_features_CORR --name ' + name + '_Idx_CORR_' + str(count) + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation OFF --filter OFF --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 ON --features_g2 ON --features_g3 ON --features_g4 ON --features_g5 ON --wv_deco DWT --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit kurt_sen --level ' + level) #OJO CON CRITERIO
		
		# os.system('python Long_Analysis.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode magnitude_freq_comps --name ' + name + '_Idx_' + str(count) + ' --db_out 37 --units mv --plot OFF --sqr_envelope OFF --demodulation ON --filter OFF --freq_lp 90.e3 --freq_hp 150.e3 --extension dms --features_g1 OFF --features_g2 OFF --features_g3 OFF --features_g4 OFF --features_g5 OFF --wv_deco ON --wv_mother ' + wv_mother + ' --wv_approx OFF --wv_crit kurt_sen --level ' + level)
	
	count += 1

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