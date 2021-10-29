import os
import sys
os.system('cd C:\Felix\Code\nico')
from numpy import arange






#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++AUTO ANN
Processings = ['butter_demod', 'OFF']
Denois = ['OFF']
Alphas = ['1.e2', '1.e1', '1.e-1']
Data_Norms = ['per_rms']
# Layers = ['2000', '500', '750 250', '1000']
Layers = ['15', '10', '5', '15 6', '10 4', '5 2']
Functions = ['tanh', 'logistic']
Classes = ['2n_2noclass']
Solvers = ['adam']
Diffs = ['OFF', '1']
Demods_LPs = ['5000.', '2000.']
# Features = ['sortint20_stats_nsnk', 'int20_stats_nsnk', 'sortint10_stats_nsnk', 'si20statsnsnk_LRstdmean', 'sortint25_stats_nsnk', 'i10statsnsnk_lrstdmean', 'i10statsnsnk_lrstd', 'i10statsnmnsnk_lrstd', 'i10statsnmnsnknmin_lrstd', 'i10maxminrms_lrrms', 'i10maxminstd_lrrmsstd']
Features = ['i10maxminrms_lrrms', 'i10maxminstd_lrrmsstd']
# Demods_Pres = ['highpass 80.e3 3', 'bandpass 80.e3 200.e3 3']
Demods_Pres = ['bandpass 80.e3 200.e3 3']



# Processings = ['OFF']
# Denois = ['OFF']
# Alphas = ['1.e0']
# Data_Norms = ['OFF']
# Layers = ['500']
# Functions = ['logistic']
# Classes = ['2n_2noclass']
# Solvers = ['adam', 'lbfgs']


count = 5776
for dem_pre in Demods_Pres:
	for feature in Features:
		for dem_lp in Demods_LPs:
			for diff in Diffs:
				for solver in Solvers:
					for function in Functions:
						for processing in Processings:
							for denois in Denois:
								for alpha in Alphas:
									for data_norm in Data_Norms:
										for layer in Layers:
											for classe in Classes:
											# try:
												os.system('python Reco_Signal_Training.py --channel AE_Signal --fs 1.e6 --save ON --features ' + feature + ' --rs 1 --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20171007_155215_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl --tol 1.e-4 --activation ' + function + ' --data_norm ' + data_norm + ' --layers ' + layer + ' --alpha ' + alpha + ' --class2 1 --solver ' + solver + ' --denois ' + denois + ' --processing ' + processing + ' --demod_filter lowpass ' + dem_lp + ' 3 --demod_prefilter ' + dem_pre + ' --NN_name ' + str(count) + ' --classes ' + classe + ' --diff ' + diff)


												# +++++++++++VALID 1500 80
												os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features ' + feature + ' --method NN --NN_model C:\\Felix\\Code\\nico\\clf_' + str(count) + '.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\Fault\\V3_9_n1500_M80_AE_Signal_20160928_154159.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\Fault\\classification_20170921_103023_V3_9_n1500_M80_AE_Signal_20160928_154159.pkl --clf_check ON --class2 1 --data_norm ' + data_norm + ' --denois ' + denois + ' --processing ' + processing + ' --demod_filter lowpass ' + dem_lp + ' 3 --demod_prefilter ' + dem_pre + ' --plot OFF' + ' --classes ' + classe + ' --save_plot OFF --diff ' + diff)

													
													
												# except:
													# print('exception')
												
												count = count + 1
# termina



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++AUTO EDG
