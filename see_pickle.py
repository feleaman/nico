import os
import sys
import pickle
sys.path.insert(0, './lib') #to open user-defined functions

from m_open_extension import *

from tkinter import filedialog
from tkinter import Tk
root = Tk()
root.withdraw()
root.update()
filename = filedialog.askopenfilename()
root.destroy()

pik = read_pickle(filename)
print(pik)

# pik_new = pik
# pik_new['filename'] = 'V1_9_n1500_M80_AE_Signal_20160928_144737.mat'

# save_pickle('MODclassification_20171007_155215_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl', pik_new)


sys.exit()