#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import scipy.signal as si
import logging
import matplotlib.pyplot as plt
import sys
from tkinter import filedialog
from tkinter import Tk


sys.path.insert(0, './lib')
from m_open_extension import *
from m_fft import *
# from m_demodulation import *
# from m_denois import *
# from m_det_features import *
# from m_processing import *
# from decimal import Decimal

def main():
	n = 1000
	dt = 0.1
	x = np.sin([i*dt for i in range(n)])
	
	
	root = Tk()
	root.withdraw()
	root.update()
	filename = filedialog.askopenfilename()
	root.destroy()
	
	x = load_signal(filename, channel='AE_0')
	
	fs = 1.e6
	levels = 5
	kurtogram(x, fs, levels)
	
	return

if __name__ == "__main__":
	main()