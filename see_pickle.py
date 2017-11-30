import os
import sys
import pickle
sys.path.insert(0, './lib') #to open user-defined functions

from m_open_extension import *
import matplotlib.pyplot as plt

from tkinter import filedialog
from tkinter import Tk
root = Tk()
root.withdraw()
root.update()
filename = filedialog.askopenfilename()
root.destroy()

pik = read_pickle(filename)
print(pik)
plt.plot(pik)
plt.show()
sys.exit()

# pik_new['filename'] = 'V1_9_n1500_M80_AE_Signal_20160928_144737.mat'

classification = pik['classification']


n_pos = 0
n_neg = 0
n_ind = 0
for i in classification:
	if i == 1:
		n_pos += 1
	elif i == 0:
		n_neg += 1
	elif i == 2:
		n_ind += 1
	else:
		print('error')
		sys.exit()
print('length')
print(len(classification))
print('n_pos', n_pos)
print('n_neg', n_neg)
print('n_ind', n_ind)


# print(classification_mod[360])



# classification_mod[695] = 0
# classification_mod[410] = 2

# classification_mod[64] = 1
# classification_mod[63] = 0
















# pik_new['classification'] = classification_mod
# save_pickle('MODclassification_20171018_133246_V2_5_n1000_M40_AE_Signal_20160928_145532.pkl', pik_new)

# sys.exit()