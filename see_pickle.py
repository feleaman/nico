import os
import sys
import pickle
sys.path.insert(0, './lib') #to open user-defined functions
# import pandas as pd

from m_open_extension import *
import matplotlib.pyplot as plt

from tkinter import filedialog
from tkinter import Tk
plt.rcParams['agg.path.chunksize'] = 1000
from os import chdir
from os.path import basename
plt.rcParams['savefig.directory'] = chdir(os.path.dirname('C:'))
# dict1 = {'perro':2, 'gato':6, 'pollo':0}
# dict2 = {'perro':6, 'gato':1, 'pollo':2}

# dict1 = {'perro':[2], 'gato':[6], 'pollo':[0]}
# dict2 = {'perro':[6], 'gato':[1], 'pollo':[2]}

# print({k: dict1.get(k, 0) + dict2.get(k, 0) for k in set(dict1) & set(dict2)})
# sys.exit()



root = Tk()
root.withdraw()
root.update()
filename = filedialog.askopenfilename()
root.destroy()

pik = read_pickle(filename)
print(pik)
sys.exit()

for filename in filenames:
	pik = read_pickle(filename)
	if basename(filename).find('lvl_3') != -1:
		save_pickle(basename(filename), pik)
	
	print(pik)
	
	# print(pik['MSE_Valid'])
	# print(pik['MSE_Train'][20])
	# plt.plot(pik['PEN2_Train'])
	# plt.show()
	# print('Train++++++++++++++')
	# print(pik['MSE_Train'][-1])
	# print(pik['CRC_Train'][-1])
	# print(pik['DME_Train'][-1])
	# print(pik['DST_Train'][-1])
	# print(pik['DKU_Train'][-1])
	# print('\nValid++++++++++++++')
	# print(pik['MSE_Valid'][-1])
	# print(pik['CRC_Valid'][-1])
	# print(pik['DME_Valid'][-1])
	# print(pik['DST_Valid'][-1])
	# print(pik['DKU_Valid'][-1])
	# plt.plot(pik)
	# plt.show()
	# opt = input('divide por 1000?...')
	# if opt == 'y':
		# new = pik/1000.
		# del pik
		# save_pickle(filename, new)
		# del new
	
sys.exit()


# sum = 0
# for element in pik['scores_cv']:
	# sum += element['mcc'] 
# mcc_mean = sum/10.

# sum2 = 0
# for element in pik['scores_cv']:
	# sum2 += (element['mcc'] - mcc_mean)**2.0
# mcc_sd = (sum2/10.)**0.5
# print('\n mcc mean cv', mcc_mean)
# print('mcc std cv', mcc_sd)


# sum = 0
# for element in pik['scores_cv']:
	# sum += element['recall'] 
# recall_mean = sum/10.

# sum2 = 0
# for element in pik['scores_cv']:
	# sum2 += (element['recall'] - recall_mean)**2.0
# recall_sd = (sum2/10.)**0.5
# print('\n recall mean cv', recall_mean)
# print('recall std cv', recall_sd)


# sum = 0
# for element in pik['scores_cv']:
	# sum += element['fpr'] 
# fpr_mean = sum/10.

# sum2 = 0
# for element in pik['scores_cv']:
	# sum2 += (element['fpr'] - fpr_mean)**2.0
# fpr_sd = (sum2/10.)**0.5
# print('\n fpr mean cv', fpr_mean)
# print('fpr std cv', fpr_sd)



# print(pik['scores_test'])

# sys.exit()
# # print(pik['config']['alpha'])
# # print(pik['config']['activation'])
# # print(pik['config']['diff'])
# # print(pik['config']['solver'])
# # pik = {'a':[1, 2], 'b':[2, 3]}
# DataFr = pd.DataFrame([pik])
# # DataFr = pd.DataFrame.from_dict(pik)
# writer = pd.ExcelWriter('caca' + '.xlsx')

# DataFr.to_excel(writer, sheet_name='Sheet1')
# writer.close()



# sys.exit()

# vec = pik['classification']
# pos = 0
# neg = 0
# for i in range(len(vec)):
	# if vec[i] == 0:
		# neg += 1
	# else:
		# pos += 1
# print('pos')
# print(pos)
# print('neg')
# print(neg)
# print('pos+neg')
# print(pos+neg)


# sys.exit()
# # print('rms')
# # print(pik['feat_rms'])
# # print('rise')

# # print(pik['feat_rise'])
# # print('count')

# # print(pik['feat_count'])

# # print(len(pik['classification']))
# # print(pik[0])
# print('\n activation+++++++++ ')
# print(pik[0]['activation'])
# print('\n solver+++++++++ ')
# print(pik[0]['solver'])
# print('\n layers+++++++++ ')
# print(pik[0]['layers'])
# print('\n alpha+++++++++ ')
# print(pik[0]['alpha'])

# print('\n DIFF++++++++ ')
# print(pik[0]['diff'])

# # print(pik)

# # plt.plot(pik)
# # plt.show()
# sys.exit()

# # pik_new['filename'] = 'V1_9_n1500_M80_AE_Signal_20160928_144737.mat'

# classification = pik['classification']


# n_pos = 0
# n_neg = 0
# n_ind = 0
# for i in classification:
	# if i == 1:
		# n_pos += 1
	# elif i == 0:
		# n_neg += 1
	# elif i == 2:
		# n_ind += 1
	# else:
		# print('error')
		# sys.exit()
# print('length')
# print(len(classification))
# print('n_pos', n_pos)
# print('n_neg', n_neg)
# print('n_ind', n_ind)


# # print(classification_mod[360])



# # classification_mod[695] = 0
# # classification_mod[410] = 2

# # classification_mod[64] = 1
# # classification_mod[63] = 0
















# # pik_new['classification'] = classification_mod
# # save_pickle('MODclassification_20171018_133246_V2_5_n1000_M40_AE_Signal_20160928_145532.pkl', pik_new)

# # sys.exit()