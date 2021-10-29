import os
import sys
import time

start_time = time.time()

os.system('cd C:\Felix\Code\nico')



# os.system('python ML_Model.py --save ON --file C:\\Felix\\Eickhoff\\Burst_Features\\thr_0_4_wtime_1ms_400khz\\thr_0_4_wtime_1ms_400khz.pkl --mode mode2 --file_test C:\\Felix\Eickhoff\\Burst_Features\\thr_0_4_wtime_1ms_400khz\\test\\box\\features_Eickhoff_WEA_AE_20171122_224324_thr_0.4_wtime_0.001_400khzHP.pkl')

# os.system('python ML_Model.py --save ON --file C:\\Felix\\Eickhoff\\Burst_Features\\thr_1_0_wtime_1ms_350khz_441khz\\thr_1_0_wtime_1ms_350khz_441khz.pkl --mode mode2')


os.system('python ML_Model.py --save ON --file C:\\Felix\\Eickhoff\\Burst_Features\\thr_1_0_wtime_1ms_300khz_441_khz\\thr_1_0_wtime_1ms_300khz_441_khz.pkl --mode mode2')



# os.system('python ML_Model.py --save ON --file C:\\Felix\\27_Materialerkennung\\Data\\thr_03_wtime_1_nofilt\\thr_03_wtime_1_nofilt.pkl')

print("--- %s seconds ---" % (time.time() - start_time))

sys.exit()