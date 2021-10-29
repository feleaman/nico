import os
import sys


os.system('cd C:\Felix\Code\nico')


# Demod_Filters = ['lowpass 2000. 3', 'lowpass 5000. 3', 'lowpass 8000. 3']
# Delays = ['0.0002', '0.0001', '0.00005']

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++MANUAL
# # # +++++++++++TRAIN-DIRECT 1500 80
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features DataSorted --method WIN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171018_180003.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\V1_9_n1500_M80_AE_Signal_20160928_144737.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl --clf_check ON --class2 1 --data_norm OFF --processing OFF --demod_filter lowpass 2000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff OFF --classes 2n_2noclass --thr_mode factor_rms --thr_value 1.595339 --save_plot OFF --window_delay 0 --save_name TRAIN --overlap 0 --rms_change 6.0 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode fixed_value')


# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features DataSorted --method WIN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171018_180003.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\V1_9_n1500_M80_AE_Signal_20160928_144737.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl --clf_check ON --class2 1 --data_norm OFF --processing OFF --demod_filter lowpass 2000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff OFF --classes 2n_2noclass --thr_mode factor_rms --thr_value 1.595339 --save_plot OFF --window_delay 0 --save_name TRAIN --overlap 0 --rms_change 5.5 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode fixed_value')

# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features DataSorted --method WIN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171018_180003.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\V1_9_n1500_M80_AE_Signal_20160928_144737.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl --clf_check ON --class2 1 --data_norm OFF --processing OFF --demod_filter lowpass 2000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff OFF --classes 2n_2noclass --thr_mode factor_rms --thr_value 1.595339 --save_plot OFF --window_delay 0 --save_name TRAIN --overlap 0 --rms_change 5.0 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode fixed_value')


# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features DataSorted --method WIN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171018_180003.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\V1_9_n1500_M80_AE_Signal_20160928_144737.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl --clf_check ON --class2 1 --data_norm OFF --processing OFF --demod_filter lowpass 2000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff OFF --classes 2n_2noclass --thr_mode factor_rms --thr_value 1.595339 --save_plot OFF --window_delay 0 --save_name TRAIN --overlap 0 --rms_change 4.5 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode fixed_value')



# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features DataSorted --method WIN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171018_180003.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\V1_9_n1500_M80_AE_Signal_20160928_144737.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl --clf_check ON --class2 1 --data_norm OFF --processing OFF --demod_filter lowpass 2000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff OFF --classes 2n_2noclass --thr_mode factor_rms --thr_value 1.595339 --save_plot OFF --window_delay 0 --save_name TRAIN --overlap 0 --rms_change 6.5 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode fixed_value')



# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features DataSorted --method WIN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171018_180003.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\V1_9_n1500_M80_AE_Signal_20160928_144737.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl --clf_check ON --class2 1 --data_norm OFF --processing OFF --demod_filter lowpass 2000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff OFF --classes 2n_2noclass --thr_mode factor_rms --thr_value 1.595339 --save_plot OFF --window_delay 0 --save_name TRAIN --overlap 0 --rms_change 7.0 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode fixed_value')


# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features DataSorted --method WIN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171018_180003.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\V1_9_n1500_M80_AE_Signal_20160928_144737.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl --clf_check ON --class2 1 --data_norm OFF --processing OFF --demod_filter lowpass 2000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff OFF --classes 2n_2noclass --thr_mode factor_rms --thr_value 1.595339 --save_plot OFF --window_delay 0 --save_name TRAIN --overlap 0 --rms_change 4.0 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode fixed_value')

# sys.exit()




# # EDG*******************************************************************

# #+++++++++++TEST-DIRECT 1500 80
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint25_stats_nsnk --method EDG --NN_model C:\\Felix\\019_Burst_Detection\\20180112_New_Results_Class2_1\\ANN\\Train_Valid\\batch3\\Part2\\Models\\clf_4391.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\Fault\\V2_9_n1500_M80_AE_Signal_20160928_151441.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\Fault\\classification_20170922_134842_V2_9_n1500_M80_AE_Signal_20160928_151441.pkl --clf_check ON --class2 1 --data_norm OFF --processing butter_demod --demod_filter lowpass 2000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff 1 --classes 2n_2noclass --thr_mode factor_rms --thr_value 1.451 --save_plot OFF --window_delay 0.0002 --save_name TEST --overlap 0 --rms_change 0.268 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode factor_rms')


# #+++++++++++TEST-DIRECT 1000 80
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint25_stats_nsnk --method EDG --NN_model C:\\Felix\\019_Burst_Detection\\20180112_New_Results_Class2_1\\ANN\\Train_Valid\\batch3\\Part2\\Models\\clf_4391.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\Fault\\V2_8_n1000_M80_AE_Signal_20160928_151108.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\Fault\\classification_20170922_122221_V2_8_n1000_M80_AE_Signal_20160928_151108.pkl --clf_check ON --class2 1 --data_norm OFF --processing butter_demod --demod_filter lowpass 2000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff 1 --classes 2n_2noclass --thr_mode factor_rms --thr_value 1.451 --save_plot OFF --window_delay 0.0002 --save_name TEST --overlap 0 --rms_change 0.268 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode factor_rms')


# #+++++++++++TEST-DIRECT  1500 40
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint25_stats_nsnk --method EDG --NN_model C:\\Felix\\019_Burst_Detection\\20180112_New_Results_Class2_1\\ANN\\Train_Valid\\batch3\\Part2\\Models\\clf_4391.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\Fault\\V1_6_n1500_M40_AE_Signal_20160928_143502.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\Fault\\classification_20170921_151654_V1_6_n1500_M40_AE_Signal_20160928_143502.pkl --clf_check ON --class2 1 --data_norm OFF --processing butter_demod --demod_filter lowpass 2000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff 1 --classes 2n_2noclass --thr_mode factor_rms --thr_value 1.451 --save_plot OFF --window_delay 0.0002 --save_name TEST --overlap 0 --rms_change 0.268 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode factor_rms')


# #+++++++++++TEST-DIRECT  1000 40
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint25_stats_nsnk --method EDG --NN_model C:\\Felix\\019_Burst_Detection\\20180112_New_Results_Class2_1\\ANN\\Train_Valid\\batch3\\Part2\\Models\\clf_4391.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_40\\Fault\\V2_5_n1000_M40_AE_Signal_20160928_145532.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_40\\Fault\\classification_20171018_133246_V2_5_n1000_M40_AE_Signal_20160928_145532.pkl --clf_check ON --class2 1 --data_norm OFF --processing butter_demod --demod_filter lowpass 2000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff 1 --classes 2n_2noclass --thr_mode factor_rms --thr_value 1.451 --save_plot OFF --window_delay 0.0002 --save_name TEST --overlap 0 --rms_change 0.268 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode factor_rms')







# # WIN*******************************************************************

#+++++++++++TEST-DIRECT 1500 80
os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint25_stats_nsnk --method WIN --NN_model C:\\Felix\\019_Burst_Detection\\20180112_New_Results_Class2_1\\ANN\\Train_Valid\\batch3\\Part2\\Models\\clf_4391.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\Fault\\V2_9_n1500_M80_AE_Signal_20160928_151441.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\Fault\\classification_20170922_134842_V2_9_n1500_M80_AE_Signal_20160928_151441.pkl --clf_check ON --class2 1 --data_norm OFF --processing OFF --demod_filter lowpass 5000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff OFF --classes 2n_2noclass --thr_mode factor_rms --thr_value 3.577 --save_plot OFF --window_delay 0 --save_name TEST --overlap 0 --rms_change 0.322 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode factor_rms')


#+++++++++++TEST-DIRECT 1000 80
os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint25_stats_nsnk --method WIN --NN_model C:\\Felix\\019_Burst_Detection\\20180112_New_Results_Class2_1\\ANN\\Train_Valid\\batch3\\Part2\\Models\\clf_4391.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\Fault\\V2_8_n1000_M80_AE_Signal_20160928_151108.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\Fault\\classification_20170922_122221_V2_8_n1000_M80_AE_Signal_20160928_151108.pkl --clf_check ON --class2 1 --data_norm OFF --processing OFF --demod_filter lowpass 5000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff OFF --classes 2n_2noclass --thr_mode factor_rms --thr_value 3.577 --save_plot OFF --window_delay 0 --save_name TEST --overlap 0 --rms_change 0.322 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode factor_rms')


#+++++++++++TEST-DIRECT  1500 40
os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint25_stats_nsnk --method WIN --NN_model C:\\Felix\\019_Burst_Detection\\20180112_New_Results_Class2_1\\ANN\\Train_Valid\\batch3\\Part2\\Models\\clf_4391.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\Fault\\V1_6_n1500_M40_AE_Signal_20160928_143502.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\Fault\\classification_20170921_151654_V1_6_n1500_M40_AE_Signal_20160928_143502.pkl --clf_check ON --class2 1 --data_norm OFF --processing OFF --demod_filter lowpass 5000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff OFF --classes 2n_2noclass --thr_mode factor_rms --thr_value 3.577 --save_plot OFF --window_delay 0 --save_name TEST --overlap 0 --rms_change 0.322 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode factor_rms')


#+++++++++++TEST-DIRECT  1000 40
os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint25_stats_nsnk --method WIN --NN_model C:\\Felix\\019_Burst_Detection\\20180112_New_Results_Class2_1\\ANN\\Train_Valid\\batch3\\Part2\\Models\\clf_4391.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_40\\Fault\\V2_5_n1000_M40_AE_Signal_20160928_145532.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_40\\Fault\\classification_20171018_133246_V2_5_n1000_M40_AE_Signal_20160928_145532.pkl --clf_check ON --class2 1 --data_norm OFF --processing OFF --demod_filter lowpass 5000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff OFF --classes 2n_2noclass --thr_mode factor_rms --thr_value 3.577 --save_plot OFF --window_delay 0 --save_name TEST --overlap 0 --rms_change 0.322 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode factor_rms')


















# # THR*******************************************************************

# #+++++++++++TEST-DIRECT 1500 80
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint25_stats_nsnk --method THR --NN_model C:\\Felix\\019_Burst_Detection\\20180112_New_Results_Class2_1\\ANN\\Train_Valid\\batch3\\Part2\\Models\\clf_4391.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\Fault\\V2_9_n1500_M80_AE_Signal_20160928_151441.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\Fault\\classification_20170922_134842_V2_9_n1500_M80_AE_Signal_20160928_151441.pkl --clf_check ON --class2 1 --data_norm OFF --processing OFF --demod_filter lowpass 5000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff OFF --classes 2n_2noclass --thr_mode factor_rms --thr_value 4.292 --save_plot OFF --window_delay 0 --save_name TEST_clf4391_batch3 --overlap 0 --rms_change 0.246 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode factor_rms')


# #+++++++++++TEST-DIRECT 1000 80
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint25_stats_nsnk --method THR --NN_model C:\\Felix\\019_Burst_Detection\\20180112_New_Results_Class2_1\\ANN\\Train_Valid\\batch3\\Part2\\Models\\clf_4391.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\Fault\\V2_8_n1000_M80_AE_Signal_20160928_151108.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\Fault\\classification_20170922_122221_V2_8_n1000_M80_AE_Signal_20160928_151108.pkl --clf_check ON --class2 1 --data_norm OFF --processing OFF --demod_filter lowpass 5000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff OFF --classes 2n_2noclass --thr_mode factor_rms --thr_value 4.292 --save_plot OFF --window_delay 0 --save_name TEST_clf4391_batch3 --overlap 0 --rms_change 0.246 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode factor_rms')


# #+++++++++++TEST-DIRECT  1500 40
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint25_stats_nsnk --method THR --NN_model C:\\Felix\\019_Burst_Detection\\20180112_New_Results_Class2_1\\ANN\\Train_Valid\\batch3\\Part2\\Models\\clf_4391.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\Fault\\V1_6_n1500_M40_AE_Signal_20160928_143502.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\Fault\\classification_20170921_151654_V1_6_n1500_M40_AE_Signal_20160928_143502.pkl --clf_check ON --class2 1 --data_norm OFF --processing OFF --demod_filter lowpass 5000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff OFF --classes 2n_2noclass --thr_mode factor_rms --thr_value 4.292 --save_plot OFF --window_delay 0 --save_name TEST_clf4391_batch3 --overlap 0 --rms_change 0.246 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode factor_rms')


# #+++++++++++TEST-DIRECT  1000 40
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint25_stats_nsnk --method THR --NN_model C:\\Felix\\019_Burst_Detection\\20180112_New_Results_Class2_1\\ANN\\Train_Valid\\batch3\\Part2\\Models\\clf_4391.pkl --n_files 1 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_40\\Fault\\V2_5_n1000_M40_AE_Signal_20160928_145532.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_40\\Fault\\classification_20171018_133246_V2_5_n1000_M40_AE_Signal_20160928_145532.pkl --clf_check ON --class2 1 --data_norm OFF --processing OFF --demod_filter lowpass 5000. 3 --demod_prefilter highpass 80.e3 3 --plot OFF --diff OFF --classes 2n_2noclass --thr_mode factor_rms --thr_value 4.292 --save_plot OFF --window_delay 0 --save_name TEST_clf4391_batch3 --overlap 0 --rms_change 0.246 --denois OFF --freq_hp 80.e3 --order_hp 3 --rms_change_mode factor_rms')




