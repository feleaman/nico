import scipy
import numpy as np
from tkinter import filedialog
from tkinter import Tk
from argparse import ArgumentParser
import sys
sys.path.insert(0, './lib')
from m_open_extension import *
from os.path import basename

Inputs = ['channel', 'fs']
InputsOpt_Defaults = {'range':None}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	channel = config['channel']
	fs = config['fs']
	
	root = Tk()
	root.withdraw()
	root.update()
	Filepaths = filedialog.askopenfilenames()
	root.destroy()
	
	# Filenames = [basename(filepath) for filepath in Filepaths]
	
	for filepath in Filepaths:
		filename = basename(filepath)

		try:
			x = load_signal_2(filepath, channel=config['channel'])
		except:
			# x = f_open_mat(filename, channel=config['channel'])
			# x, channel = f_open_mat_2(filename, channel=config['channel'])
			# x = read_pickle(filename)
			x = f_open_tdms(filepath, channel)

		
		if config['range'] != None:
			x = x[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
		
		x = x*1000./70.8 #37 dB schottland
		# x = x*1000./70.8
		# x = x*1000./141.25 #43 dB bochum laststufen
		
		
		x = list(x)
		x = np.array(x)
		mydict={config['channel']:x}
		scipy.io.savemat(channel + '_' + filename[:-5] + '.mat', mydict)
		print('Signal was saved as .mat. Analysis finalizes')
	
	return 



def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	parser = ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'range':
			parser.add_argument('--' + element, nargs='+')
		else:
			parser.add_argument('--' + element, nargs='?')
	
	args = parser.parse_args()
	config = {}
	for element in Inputs:
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Required:', element)
			sys.exit()

	for element, value in zip(Inputs_opt, Defaults):
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Default ' + element + ' = ', value)
			config[element] = value
	

	config['fs'] = float(config['fs'])
	
	if config['range'] != None:
		config['range'][0] = float(config['range'][0])
		config['range'][1] = float(config['range'][1])
	
	
	return config

if __name__ == '__main__':
	main(sys.argv)
