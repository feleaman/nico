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
Filepaths = filedialog.askopenfilenames()
root.destroy()

# print(Filepaths[0])
# print(os.path.basename(Filepaths[0]))


Filenames = [os.path.basename(filepath) for filepath in Filepaths]
print(Filenames)
# sys.exit()

# os.rename(Filepaths[0], 'Doctor_Bill.JPG')

Places = ['S3', 'S1', 'S2', 'Z1', 'Z2', 'Z3']
# Places = ['Z1', 'Z2', 'Z3']
Repetitions = ['R1', 'R2', 'R3', 'R4', 'R5']

# for filepath in Filepaths:
	# os.rename(filepath, 'V1_' + place + '_' + repetition + '_' + os.path.basename(filepath))
Filepaths = list(Filepaths)
print(len(Filepaths))

count = 0
for place in Places:
	for repetition in Repetitions:
		newname = 'V3_' + place + '_' + repetition + '_' + os.path.basename(Filepaths[count])
		# name = os.path.basename(Filepaths[count])
		# print(count)
		# print(Filepaths[count])
		# print(count)
		# print(os.path.dirname(Filepaths[count]))
		os.rename(Filepaths[count], newname)
		# newpath = 'C:\\Felix\\Code\\nico'
		# os.rename(Filepaths[count], os.path.join(newpath, newname))
		count = count + 1

# sys.exit()