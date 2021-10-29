import numpy as np
# from scipy.integrate import odeint
# from scipy import signal
# from scipy import stats
# import scipy
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'png'
from m_open_extension import *


def plot3v_thesis(three_signals, style):
	#Modules and global properties
	from matplotlib import font_manager
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.subplots_adjust(hspace=0.175, left=0.15, right=0.9, bottom=0.125, top=0.95)
	
	#Values Fixed
	font_big = 17
	font_little = 14
	font_label = 12
	
	#Plot
	fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, sharey=True)
	keys = list(three_signals)
	ax[0].plot(three_signals['dom'], three_signals[keys[0]], label=keys[0])
	ax[1].plot(three_signals['dom'], three_signals[keys[1]], label=keys[1])
	ax[2].plot(three_signals['dom'], three_signals[keys[2]], label=keys[2])
	
	#Axis X
	if style['dom'] == 'time':	
		ax[2].set_xlabel('Time [s]', fontsize=font_big)
	elif style['dom'] == 'frequency':
		if style['kHz'] == 'ON':
			ax[2].set_xlabel('Frequency [kHz]', fontsize=font_big)
		else:
			ax[2].set_xlabel('Frequency [Hz]', fontsize=font_big)
	
	#Axis Y
	if style['mV'] == 'ON':
		ax[1].set_ylabel('Amplitude [mV]', fontsize=font_big)
	
	#Size labels from axis
	ax[0].tick_params(axis='both', labelsize=font_little)
	ax[1].tick_params(axis='both', labelsize=font_little)	
	ax[2].tick_params(axis='both', labelsize=font_little)	
		
	#Scientific notation
	lim = 2
	ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)
	
	
	

	#Figure Size
	fig.set_size_inches(6, 6)
	
	#Eliminate line from label
	ax[0].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)	
	ax[1].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
	ax[2].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
	
	#Size from offset text
	ax[0].yaxis.offsetText.set_fontsize(font_little-2)
	ax[1].yaxis.offsetText.set_fontsize(font_little-2)
	ax[2].yaxis.offsetText.set_fontsize(font_little-2)
	
	#Hidde labels from Axis X in upper plots
	ax[0].tick_params(labelbottom=False) 
	ax[1].tick_params(labelbottom=False) 
	
	#Set Ticks in Axis Y
	ax[0].set_yticks([-20, -10, 0, 10, 20]) 
	ax[1].set_yticks([-20, -10, 0, 10, 20]) 
	ax[2].set_yticks([-20, -10, 0, 10, 20]) 
	
	
	#Set Limits in Axis X
	ax[0].set_xlim(left=0, right=2)

	
	plt.tight_layout()
	plt.show()
	return

def plot3h_thesis(three_signals, style):
	#Modules and global properties
	from matplotlib import font_manager
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	fig, ax = plt.subplots(ncols=3, nrows=1, sharex=style['sharex'], sharey=style['sharey'])
	
	# #Values Fixed
	# font_big = 17+2
	# font_little = 15+2
	# font_label = 13+2
	# font_offset = 15+2
	# font_autolabel = 15+2
	# lim = 3
	# plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.175, top=0.915)
	# fig.set_size_inches(14.2, 4.0)
	
	#Values Fixed Auto
	font_caption = (23+3)*0.4316
	font_big = (17+3)*0.4316
	font_little = (15+3)*0.4316
	font_label = (13+3)*0.4316
	font_offset = (15+3)*0.4316
	font_autolabel = (15+3)*0.4316
	lim = 3

	if style['caption'] == 'lower':
		plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.26, top=0.89)
		fig.set_size_inches(6.1, 1.85)
		fig.text(0.182, 0.02, '(a)', fontsize=font_caption)
		fig.text(0.510, 0.02, '(b)', fontsize=font_caption)
		fig.text(0.840, 0.02, '(c)', fontsize=font_caption)
	elif style['caption'] == 'lower left':
		# plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.21, top=0.89)
		plt.subplots_adjust(wspace=0.32, left=0.07, right=0.98, bottom=0.213, top=0.89)
		fig.set_size_inches(6.1, 1.71)
		fig.text(0.059, 0.03, '(a)', fontsize=font_caption)
		fig.text(0.387, 0.03, '(b)', fontsize=font_caption)
		fig.text(0.717, 0.03, '(c)', fontsize=font_caption)
	else:
		plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.26, top=0.89)
		fig.set_size_inches(6.1, 1.71)
	#Axis X
	if style['dom'] == 'time':
		fact = 1.
		ax[0].set_xlabel('Time [s]', fontsize=font_big)
		ax[1].set_xlabel('Time [s]', fontsize=font_big)
		ax[2].set_xlabel('Time [s]', fontsize=font_big)
	elif style['dom'] == 'frequency':
		if style['kHz'] == 'ON':
			fact = 1000.
			ax[0].set_xlabel('Frequency [kHz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [kHz]', fontsize=font_big)
			ax[2].set_xlabel('Frequency [kHz]', fontsize=font_big)
		else:
			fact = 1.
			ax[0].set_xlabel('Frequency [Hz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [Hz]', fontsize=font_big)
			ax[2].set_xlabel('Frequency [Hz]', fontsize=font_big)
	elif style['dom'] == 'other':
		fact = 1.
		ax[0].set_xlabel(style['xtitle'][0], fontsize=font_big)
		ax[1].set_xlabel(style['xtitle'][1], fontsize=font_big)
		ax[2].set_xlabel(style['xtitle'][2], fontsize=font_big)
	
	
	#Plot	
	keys = list(three_signals)
	if style['type'] == 'plot':
		ax[0].plot(three_signals['dom']/fact, three_signals[keys[0]], label=keys[0])
		ax[1].plot(three_signals['dom']/fact, three_signals[keys[1]], label=keys[1])
		ax[2].plot(three_signals['dom']/fact, three_signals[keys[2]], label=keys[2])
	elif style['type'] == 'bar':
		bar0 = ax[0].bar(three_signals['dom']/fact, three_signals[keys[0]], label=keys[0])
		bar1 = ax[1].bar(three_signals['dom']/fact, three_signals[keys[1]], label=keys[1])
		bar2 = ax[2].bar(three_signals['dom']/fact, three_signals[keys[2]], label=keys[2])
	
	#Axis Y
	if style['dom'] == 'time':
		ax[0].set_ylabel('Amplitude [mV]', fontsize=font_big)
		ax[1].set_ylabel('Amplitude [mV]', fontsize=font_big)
		ax[2].set_ylabel('Amplitude [mV]', fontsize=font_big)		
		
	elif style['dom'] == 'frequency':
		ax[0].set_ylabel('Magnitude [mV]', fontsize=font_big)
		ax[1].set_ylabel('Magnitude [mV]', fontsize=font_big)
		ax[2].set_ylabel('Magnitude [mV]', fontsize=font_big)
	
	elif style['dom'] == 'other':
		ax[0].set_ylabel(style['ytitle'][0], fontsize=font_big)
		ax[1].set_ylabel(style['ytitle'][1], fontsize=font_big)
		ax[2].set_ylabel(style['ytitle'][2], fontsize=font_big)
		
		# params = {'mathtext.default': 'regular' }          
		# plt.rcParams.update(params)
		# ax[0].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[1].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[2].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)	
		
		# ax[0].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[1].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[2].set_title('Envelope spectrum', fontsize=font_offset)
	
	#Size labels from axis
	ax[0].tick_params(axis='both', labelsize=font_little)
	ax[1].tick_params(axis='both', labelsize=font_little)	
	ax[2].tick_params(axis='both', labelsize=font_little)	
		
	#Scientific notation	
	ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)

	#Eliminate line from label
	if style['legend'] == True:
		ax[0].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)	
		ax[1].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
		ax[2].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
	
	#Title
	if style['title'] == True:
		ax[0].set_title(keys[0], fontsize=font_big, y=0.97)	
		ax[1].set_title(keys[1], fontsize=font_big, y=0.97)
		ax[2].set_title(keys[2], fontsize=font_big, y=0.97)
	
	#Size from offset text
	ax[0].yaxis.offsetText.set_fontsize(font_offset)
	ax[1].yaxis.offsetText.set_fontsize(font_offset)
	ax[2].yaxis.offsetText.set_fontsize(font_offset)
	
	
	#Set Ticks in Axis Y
	# ax[0].set_yticks([-20, -10, 0, 10, 20]) 
	# ax[1].set_yticks([0, 4, 8, 12, 16]) 
	# ax[2].set_yticks([-20, -10, 0, 10, 20])
	
	#Set Ticks in Axis X
	if style['dom'] == 'other':
		ax[0].set_xticks(three_signals['dom']) 
		ax[1].set_xticks(three_signals['dom']) 
		ax[2].set_xticks(three_signals['dom'])
		
		ax[0].set_xticklabels(style['xticklabels']) 
		ax[1].set_xticklabels(style['xticklabels']) 
		ax[2].set_xticklabels(style['xticklabels'])
		
		
	
	#Set Limits in Axis X
	if style['ymax'] != None:
		ax[0].set_ylim(bottom=0, top=style['ymax'][0])
		ax[1].set_ylim(bottom=0, top=style['ymax'][1])
		ax[2].set_ylim(bottom=0, top=style['ymax'][2])
	
	if style['xmax'] != None:
		ax[0].set_xlim(left=0, right=style['xmax'][0])
		ax[1].set_xlim(left=0, right=style['xmax'][1])
		ax[2].set_xlim(left=0, right=style['xmax'][2])
	
	
	blw = 0.475
	ax[0].spines['top'].set_linewidth(blw)
	ax[0].spines['right'].set_linewidth(blw)
	ax[0].spines['left'].set_linewidth(blw)
	ax[0].spines['bottom'].set_linewidth(blw)
	ax[0].xaxis.set_tick_params(width=blw)
	ax[0].yaxis.set_tick_params(width=blw)
	
	ax[1].spines['top'].set_linewidth(blw)
	ax[1].spines['right'].set_linewidth(blw)
	ax[1].spines['left'].set_linewidth(blw)
	ax[1].spines['bottom'].set_linewidth(blw)
	ax[1].xaxis.set_tick_params(width=blw)
	ax[1].yaxis.set_tick_params(width=blw)
	
	ax[2].spines['top'].set_linewidth(blw)
	ax[2].spines['right'].set_linewidth(blw)
	ax[2].spines['left'].set_linewidth(blw)
	ax[2].spines['bottom'].set_linewidth(blw)
	ax[2].xaxis.set_tick_params(width=blw)
	ax[2].yaxis.set_tick_params(width=blw)
	

		
	if style['autolabel'] == True:
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				height2 = height*10.
				ax[0].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=font_autolabel)
		autolabel(bar0)
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				height2 = height*100.
				ax[1].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=font_autolabel)
		autolabel(bar1)
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				height2 = height*100.
				ax[2].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=font_autolabel)
		autolabel(bar2)
	
	# plt.tight_layout()
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
		
		
	return

def plot3h_thesis_big(three_signals, style):
	#Modules and global properties
	from matplotlib import font_manager
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	fig, ax = plt.subplots(ncols=3, nrows=1, sharex=style['sharex'], sharey=style['sharey'])
	
	#Values Fixed
	font_big = 17+3
	font_little = 15+3
	font_label = 13+3
	font_offset = 15+3
	font_autolabel = 15+3
	font_caption = 23+3
	lim = 3
	# plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.175, top=0.915)
	# fig.set_size_inches(14.2, 4.0)
	
	# #Values Fixed Auto
	# font_caption = (23+3)*0.4316
	# font_big = (17+3)*0.4316
	# font_little = (15+3)*0.4316
	# font_label = (13+3)*0.4316
	# font_offset = (15+3)*0.4316
	# font_autolabel = (15+3)*0.4316
	# lim = 3

	if style['caption'] == 'lower':
		plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.26, top=0.89)
		fig.set_size_inches(14.2, 4.0)
		fig.text(0.182, 0.02, '(a)', fontsize=font_caption)
		fig.text(0.510, 0.02, '(b)', fontsize=font_caption)
		fig.text(0.840, 0.02, '(c)', fontsize=font_caption)
	elif style['caption'] == 'lower left':
		# plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.21, top=0.89)
		plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.213, top=0.89)
		fig.set_size_inches(14.2, 4.0)
		fig.text(0.059, 0.03, '(a)', fontsize=font_caption)
		fig.text(0.387, 0.03, '(b)', fontsize=font_caption)
		fig.text(0.717, 0.03, '(c)', fontsize=font_caption)
	else:
		plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.26, top=0.89)
	
	fig.set_size_inches(14.2, 4.0)
	
	# ax[0].grid(axis='y')
	# ax[1].grid(axis='y')
	# ax[0].set_axisbelow(True)
	#Axis X
	
	
	if style['dom'] == 'time':
		fact = 1.
		ax[0].set_xlabel('Time [s]', fontsize=font_big)
		ax[1].set_xlabel('Time [s]', fontsize=font_big)
		ax[2].set_xlabel('Time [s]', fontsize=font_big)
	elif style['dom'] == 'frequency':
		if style['kHz'] == 'ON':
			fact = 1000.
			ax[0].set_xlabel('Frequency [kHz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [kHz]', fontsize=font_big)
			ax[2].set_xlabel('Frequency [kHz]', fontsize=font_big)
		else:
			fact = 1.
			ax[0].set_xlabel('Frequency [Hz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [Hz]', fontsize=font_big)
			ax[2].set_xlabel('Frequency [Hz]', fontsize=font_big)
	elif style['dom'] == 'other':
		fact = 1.
		ax[0].set_xlabel(style['xtitle'][0], fontsize=font_big)
		ax[1].set_xlabel(style['xtitle'][1], fontsize=font_big)
		ax[2].set_xlabel(style['xtitle'][2], fontsize=font_big)
	
	
	#Plot	
	keys = list(three_signals)
	if style['type'] == 'plot':
		ax[0].plot(three_signals['dom']/fact, three_signals[keys[0]], label=keys[0])
		ax[1].plot(three_signals['dom']/fact, three_signals[keys[1]], label=keys[1])
		ax[2].plot(three_signals['dom']/fact, three_signals[keys[2]], label=keys[2])
	elif style['type'] == 'bar':
		bar0 = ax[0].bar(three_signals['dom']/fact, three_signals[keys[0]], label=keys[0])
		bar1 = ax[1].bar(three_signals['dom']/fact, three_signals[keys[1]], label=keys[1])
		bar2 = ax[2].bar(three_signals['dom']/fact, three_signals[keys[2]], label=keys[2])
	
	#Axis Y
	if style['dom'] == 'time':
		ax[0].set_ylabel('Amplitude [mV]', fontsize=font_big)
		ax[1].set_ylabel('Amplitude [mV]', fontsize=font_big)
		ax[2].set_ylabel('Amplitude [mV]', fontsize=font_big)		
		
	elif style['dom'] == 'frequency':
		ax[0].set_ylabel('Magnitude [mV]', fontsize=font_big)
		ax[1].set_ylabel('Magnitude [mV]', fontsize=font_big)
		ax[2].set_ylabel('Magnitude [mV]', fontsize=font_big)
	
	elif style['dom'] == 'other':
		ax[0].set_ylabel(style['ytitle'][0], fontsize=font_big)
		ax[1].set_ylabel(style['ytitle'][1], fontsize=font_big)
		ax[2].set_ylabel(style['ytitle'][2], fontsize=font_big)
		
		# params = {'mathtext.default': 'regular' }          
		# plt.rcParams.update(params)
		# ax[0].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[1].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[2].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)	
		
		# ax[0].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[1].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[2].set_title('Envelope spectrum', fontsize=font_offset)
	
	#Size labels from axis
	ax[0].tick_params(axis='both', labelsize=font_little)
	ax[1].tick_params(axis='both', labelsize=font_little)	
	ax[2].tick_params(axis='both', labelsize=font_little)	
		
	#Scientific notation	
	ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)

	#Eliminate line from label
	if style['legend'] == True:
		ax[0].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)	
		ax[1].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
		ax[2].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
	
	#Title
	if style['title'] == True:
		ax[0].set_title(keys[0], fontsize=font_big)	
		ax[1].set_title(keys[1], fontsize=font_big)
		ax[2].set_title(keys[2], fontsize=font_big)
	
	#Size from offset text
	ax[0].yaxis.offsetText.set_fontsize(font_offset)
	ax[1].yaxis.offsetText.set_fontsize(font_offset)
	ax[2].yaxis.offsetText.set_fontsize(font_offset)
	
	
	#Set Ticks in Axis Y
	# ax[0].set_yticks([-20, -10, 0, 10, 20]) 
	# ax[1].set_yticks([0, 4, 8, 12, 16]) 
	# ax[2].set_yticks([-20, -10, 0, 10, 20])
	
	#Set Ticks in Axis X
	if style['dom'] == 'other':
		ax[0].set_xticks(three_signals['dom']) 
		ax[1].set_xticks(three_signals['dom']) 
		ax[2].set_xticks(three_signals['dom'])
		
		ax[0].set_xticklabels(style['xticklabels']) 
		ax[1].set_xticklabels(style['xticklabels']) 
		ax[2].set_xticklabels(style['xticklabels'])
		
		
	
	#Set Limits in Axis X
	if style['ymax'] != None:
		ax[0].set_ylim(bottom=0, top=style['ymax'][0])
		ax[1].set_ylim(bottom=0, top=style['ymax'][1])
		ax[2].set_ylim(bottom=0, top=style['ymax'][2])
	
	if style['xmax'] != None:
		ax[0].set_xlim(left=0, right=style['xmax'][0])
		ax[1].set_xlim(left=0, right=style['xmax'][1])
		ax[2].set_xlim(left=0, right=style['xmax'][2])
	
	
	blw = 1.
	ax[0].spines['top'].set_linewidth(blw)
	ax[0].spines['right'].set_linewidth(blw)
	ax[0].spines['left'].set_linewidth(blw)
	ax[0].spines['bottom'].set_linewidth(blw)
	ax[0].xaxis.set_tick_params(width=blw)
	ax[0].yaxis.set_tick_params(width=blw)
	
	ax[1].spines['top'].set_linewidth(blw)
	ax[1].spines['right'].set_linewidth(blw)
	ax[1].spines['left'].set_linewidth(blw)
	ax[1].spines['bottom'].set_linewidth(blw)
	ax[1].xaxis.set_tick_params(width=blw)
	ax[1].yaxis.set_tick_params(width=blw)
	
	ax[2].spines['top'].set_linewidth(blw)
	ax[2].spines['right'].set_linewidth(blw)
	ax[2].spines['left'].set_linewidth(blw)
	ax[2].spines['bottom'].set_linewidth(blw)
	ax[2].xaxis.set_tick_params(width=blw)
	ax[2].yaxis.set_tick_params(width=blw)
	
	
	# ax[0].yaxis.set_major_formatter(FixedOrderFormatter(-2))
	# ax[1].yaxis.set_major_formatter(FixedOrderFormatter(-2))
	# ax[2].yaxis.set_major_formatter(FixedOrderFormatter(-2))
	
	# ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	# ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	# ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	
	if style['autolabel'] == True:
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				height2 = height*100.
				ax[0].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=font_autolabel)
		autolabel(bar0)
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				height2 = height*100.
				ax[1].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height2, '%.1f' % height2, ha='center', va='bottom', fontsize=font_autolabel)
		autolabel(bar1)
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				height2 = height*100.
				ax[2].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=font_autolabel)
		autolabel(bar2)
	
	
	# plt.tight_layout()
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
		
		
	return

def plot6B_thesis_big(data, style):
	#Modules and global properties
	from matplotlib import font_manager
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	
	# params = {'mathtext.default': 'regular' }          
	# plt.rcParams.update(params)
	# plt.rc_context({'axes.autolimit_mode': 'round_numbers'})
	
	fig, ax = plt.subplots(ncols=3, nrows=2, sharex=style['sharex'], sharey=style['sharey'])
	
	
	lim = 3
	font_big = 17+3
	font_little = 15+3
	font_label = 13+3
	font_offset = 15+3
	font_autolabel = 15+3
	font_caption = 23+3
	if style['n_data'] == 1:
		plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.15, top=0.95, hspace=0.52)
		fig.set_size_inches(14.2, 7)
	else:
		plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.15, top=0.89, hspace=0.52)
		fig.set_size_inches(14.2, 7.6)
	
	# 6.5
	fig.text(0.053-0.015, 0.04, '(d)', fontsize=font_caption)
	fig.text(0.385-0.015, 0.04, '(e)', fontsize=font_caption)
	fig.text(0.717-0.015, 0.04, '(f)', fontsize=font_caption)
	
	fig.text(0.053-0.015, 0.528, '(a)', fontsize=font_caption)
	fig.text(0.385-0.015, 0.528, '(b)', fontsize=font_caption)
	fig.text(0.717-0.015, 0.528, '(c)', fontsize=font_caption)
	# 0.522
	marker = ['o', 's', 'v', '^']
	# keys = list(six_signals)
	count = 0
	for i in range(2):
		for j in range(3):
			ax[i][j].set_xlabel(style['xtitle'], fontsize=font_big)
			
			if style['n_data'] == 1:
				ax[i][j].bar(data['data_x'], data['data_y'][count])
			else:
				for k in range(style['n_data']):
					# print(data['data_x'])
					# print(data['data_y'][k][count])
					ax[i][j].plot(data['data_x'], data['data_y'][count][k], label=style['legend'][k], marker=marker[k])
				
	
			ax[i][j].set_ylabel(style['ytitle'][count], fontsize=font_big)
			
			ax[i][j].tick_params(axis='both', labelsize=font_little)
			
			ax[i][j].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
			
			# if style['legend'] != None:
				# ax[i][j].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
			
			ax[i][j].set_title(style['title'][count], fontsize=font_big)
			
			ax[i][j].yaxis.offsetText.set_fontsize(font_offset)
			
			ax[i][j].set_xticks(data['data_x'])
		
			ax[i][j].set_xticklabels(style['xticklabels']) 
			
			ax[i][j].grid(axis='both')
			count += 1
	

		

	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)



		
	if style['ylim'] != None:	
		ax[0][0].set_ylim(bottom=style['ylim'][0][0], top=style['ylim'][0][1])	
		ax[0][1].set_ylim(bottom=style['ylim'][1][0], top=style['ylim'][1][1])
		ax[0][2].set_ylim(bottom=style['ylim'][2][0], top=style['ylim'][2][1])
		
		ax[1][0].set_ylim(bottom=style['ylim'][3][0], top=style['ylim'][3][1])	
		ax[1][1].set_ylim(bottom=style['ylim'][4][0], top=style['ylim'][4][1])
		ax[1][2].set_ylim(bottom=style['ylim'][5][0], top=style['ylim'][5][1])
	
	if style['xlim'] != None:	
		ax[1][0].set_xlim(bottom=style['xlim'][1][0], top=style['xlim'][1][1])	
		ax[1][1].set_xlim(bottom=style['xlim'][1][0], top=style['xlim'][1][1])
		ax[1][2].set_xlim(bottom=style['xlim'][1][0], top=style['xlim'][1][1])

	

	
	
	# blw = 1.
	# ax[0].spines['top'].set_linewidth(blw)
	# ax[0].spines['right'].set_linewidth(blw)
	# ax[0].spines['left'].set_linewidth(blw)
	# ax[0].spines['bottom'].set_linewidth(blw)
	# ax[0].xaxis.set_tick_params(width=blw)
	# ax[0].yaxis.set_tick_params(width=blw)
	
	# ax[1].spines['top'].set_linewidth(blw)
	# ax[1].spines['right'].set_linewidth(blw)
	# ax[1].spines['left'].set_linewidth(blw)
	# ax[1].spines['bottom'].set_linewidth(blw)
	# ax[1].xaxis.set_tick_params(width=blw)
	# ax[1].yaxis.set_tick_params(width=blw)
	
	# ax[2].spines['top'].set_linewidth(blw)
	# ax[2].spines['right'].set_linewidth(blw)
	# ax[2].spines['left'].set_linewidth(blw)
	# ax[2].spines['bottom'].set_linewidth(blw)
	# ax[2].xaxis.set_tick_params(width=blw)
	# ax[2].yaxis.set_tick_params(width=blw)
	if style['n_data'] != 1:
		if style['n_data'] == 4:
			# ax[2][0].legend(fontsize=font_label, loc=(-1.71,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
			ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.7,1.17), handletextpad=0.3, labelspacing=.3)
			
			
		elif style['n_data'] == 3:
			# if style['legend']['first'][0].find('db') != -1:
			print('++++++++++ legend')
			
			if style['legend'][0].find('db') != -1:
				ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.4,1.17), handletextpad=0.3, labelspacing=.3)
			else:
				ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.45,1.17), handletextpad=0.3, labelspacing=.3)
			
			# else:
				# ax[2][0].legend(fontsize=font_label, loc=(-1.5,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
		elif style['n_data'] == 2:
			ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.30,1.17), handletextpad=0.3, labelspacing=.3)
	
	# plt.tight_layout()
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
		
		
	return

def plot6_3v(f, Filepaths):
	
	
	names = ['AE_0', 'AE_0', 'AE_1', 'AE_1', 'AE_2', 'AE_2']
	names2 = ['xxxNODamage', 'Damage', 'xxxNODamage', 'Damage', 'xxxNODamage', 'Damage']
	names3 = []
	signals = {}
	for name, name2 in zip(names, names2):
		for filepath in Filepaths:
			if filepath.find(name) != -1:
				if name not in signals.keys():
					signals[name + '_' + name2] = read_pickle(filepath)
					names3.append(name + '_' + name2)
	
	font_big = 17
	font_little = 15
	font_label = 13
	
	from matplotlib import font_manager
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	
	fig, ax = plt.subplots(ncols=2, nrows=3, sharex=True, sharey=True)
	plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.11, top=0.94, hspace=0.35)
	fig.set_size_inches(12.2, 6.8)		

	ax[0][0].plot(f/1000., signals[names3[0]], label='AE-0')
	
	ax[0][1].plot(f/1000., signals[names3[1]], label='AE-0')
	
	ax[1][0].plot(f/1000., signals[names3[2]], label='AE-1')		
	
	ax[1][1].plot(f/1000., signals[names3[3]], label='AE-1')
	
	ax[2][0].plot(f/1000., signals[names3[4]], label='AE-2')
	
	ax[2][1].plot(f/1000., signals[names3[5]], label='AE-2')

	
	name_ylabel = r'$\Sigma$ SC [-]'
	ax[0][0].set_ylabel(name_ylabel, fontsize=font_big)
	ax[0][1].set_ylabel(name_ylabel, fontsize=font_big)
	
	ax[1][0].set_ylabel(name_ylabel, fontsize=font_big)	
	ax[1][1].set_ylabel(name_ylabel, fontsize=font_big)
	
	ax[2][0].set_ylabel(name_ylabel, fontsize=font_big)
	ax[2][1].set_ylabel(name_ylabel, fontsize=font_big)
	
	name_xlabel = 'Frequency [kHz]'
	# ax[0][0].set_xlabel(name_xlabel, fontsize=font_big)
	# ax[0][1].set_xlabel(name_xlabel, fontsize=font_big)
	
	# ax[1][0].set_xlabel(name_xlabel, fontsize=font_big)	
	# ax[1][1].set_xlabel(name_xlabel, fontsize=font_big)
	
	ax[2][0].set_xlabel(name_xlabel, fontsize=font_big)
	ax[2][1].set_xlabel(name_xlabel, fontsize=font_big)
	
	ax[0][0].legend(fontsize=font_label, loc='best')
	ax[0][1].legend(fontsize=font_label, loc='best')
	
	ax[1][0].legend(fontsize=font_label, loc='best')	
	ax[1][1].legend(fontsize=font_label, loc='best')
	
	ax[2][0].legend(fontsize=font_label, loc='best')
	ax[2][1].legend(fontsize=font_label, loc='best')
	
	#Eliminate line from label
	ax[2][0].set_xlabel(name_xlabel, fontsize=font_big)
	ax[2][1].set_xlabel(name_xlabel, fontsize=font_big)
	
	ax[0][0].legend(fontsize=font_label, loc='best', handlelength=0, handletextpad=0, fancybox=True)
	ax[0][1].legend(fontsize=font_label, loc='best', handlelength=0, handletextpad=0, fancybox=True)
	
	ax[1][0].legend(fontsize=font_label, loc='best', handlelength=0, handletextpad=0, fancybox=True)	
	ax[1][1].legend(fontsize=font_label, loc='best', handlelength=0, handletextpad=0, fancybox=True)
	
	ax[2][0].legend(fontsize=font_label, loc='best', handlelength=0, handletextpad=0, fancybox=True)
	ax[2][1].legend(fontsize=font_label, loc='best', handlelength=0, handletextpad=0, fancybox=True)

	
	
	plt.rcParams['mathtext.fontset'] = 'cm'
	ax[0][0].set_title('No fault', fontsize=font_big)
	ax[0][1].set_title('Fault', fontsize=font_big)
	
	# ax[1][0].set_title('AE-1', fontsize=font_big)	
	# ax[1][1].set_title('AE-1', fontsize=font_big)
	
	# ax[2][0].set_title('AE-2', fontsize=font_big)
	# ax[2][1].set_title('AE-2', fontsize=font_big)
	
	
	ax[0][0].set_xlim(left=0, right=500)
	ax[0][1].set_xlim(left=0, right=500)
	
	ax[1][0].set_xlim(left=0, right=500)	
	ax[1][1].set_xlim(left=0, right=500)
	
	ax[2][0].set_xlim(left=0, right=500)
	ax[2][1].set_xlim(left=0, right=500)
	

	valtop = 0.4
	ax[0][0].set_ylim(bottom=0, top=valtop)
	ax[0][1].set_ylim(bottom=0, top=valtop)
	
	ax[1][0].set_ylim(bottom=0, top=valtop)	
	ax[1][1].set_ylim(bottom=0, top=valtop)
	
	ax[2][0].set_ylim(bottom=0, top=valtop)
	ax[2][1].set_ylim(bottom=0, top=valtop)

	
	
	# ax[0][0].set_yticks([0, 100, 200, 300, 400, 500])
	# ax[0][1].set_yticks([0, 100, 200, 300, 400, 500])
	# ax[0][2].set_yticks([0, 100, 200, 300, 400, 500])
	
	# ax[1][0].set_yticks([0, 100, 200, 300, 400, 500])
	# ax[1][1].set_yticks([0, 100, 200, 300, 400, 500])
	# ax[1][2].set_yticks([0, 100, 200, 300, 400, 500])
	

	
	
	ax[0][0].tick_params(axis='both', labelsize=font_little)
	ax[0][1].tick_params(axis='both', labelsize=font_little)
	
	ax[1][0].tick_params(axis='both', labelsize=font_little)	
	ax[1][1].tick_params(axis='both', labelsize=font_little)
	
	ax[2][0].tick_params(axis='both', labelsize=font_little)
	ax[2][1].tick_params(axis='both', labelsize=font_little)

	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)
	
	plt.show()

	return

def plot2h_thesis(three_signals, style):
	#Modules and global properties
	from matplotlib import font_manager
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	fig, ax = plt.subplots(ncols=2, nrows=1, sharex=style['sharex'], sharey=style['sharey'])
	
	#Values Fixed
	amp_factor = 1.1923
	font_big = 17*amp_factor
	font_little = 15*amp_factor
	font_label = 13*amp_factor
	font_offset = 15*amp_factor
	lim = 2
	plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.175, top=0.90)
	fig.set_size_inches(10.2, 4.0)
	# 0.92
	
	
	#Axis X
	if style['dom'] == 'time':
		fact = 1.
		ax[0].set_xlabel('Time [s]', fontsize=font_big)
		ax[1].set_xlabel('Time [s]', fontsize=font_big)
	elif style['dom'] == 'frequency':
		if style['kHz'] == 'ON':
			fact = 1000.
			ax[0].set_xlabel('Frequency [kHz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [kHz]', fontsize=font_big)
		else:
			fact = 1.
			ax[0].set_xlabel('Frequency [Hz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [Hz]', fontsize=font_big)
	elif style['dom'] == 'other':
		fact = 1.
		ax[0].set_xlabel(style['xtitle'][0], fontsize=font_big)
		ax[1].set_xlabel(style['xtitle'][1], fontsize=font_big)
	
	
	#Plot	
	keys = list(three_signals)
	if style['type'] == 'plot':
		ax[0].plot(three_signals['dom']/fact, three_signals[keys[0]], label=keys[0])
		ax[1].plot(three_signals['dom']/fact, three_signals[keys[1]], label=keys[1])
	elif style['type'] == 'bar':
		bar0 = ax[0].bar(three_signals['dom']/fact, three_signals[keys[0]], label=keys[0])
		bar1 = ax[1].bar(three_signals['dom']/fact, three_signals[keys[1]], label=keys[1])
	
	#Axis Y
	if style['dom'] == 'time':
		ax[0].set_ylabel('Amplitude [mV]', fontsize=font_big)
		ax[1].set_ylabel('Amplitude [mV]', fontsize=font_big)
		
	elif style['dom'] == 'frequency':
		if style['mV'] == 'ON':
			ax[0].set_ylabel('Magnitude [mV]', fontsize=font_big)
			ax[1].set_ylabel('Magnitude [mV]', fontsize=font_big)
		else:
			ax[0].set_ylabel('Magnitude [g]', fontsize=font_big)
			ax[1].set_ylabel('Magnitude [g]', fontsize=font_big)
	
	elif style['dom'] == 'other':
		ax[0].set_ylabel(style['ytitle'][0], fontsize=font_big)
		ax[1].set_ylabel(style['ytitle'][1], fontsize=font_big)
		
		# params = {'mathtext.default': 'regular' }          
		# plt.rcParams.update(params)
		# ax[0].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[1].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[2].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)	
		
		# ax[0].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[1].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[2].set_title('Envelope spectrum', fontsize=font_offset)
	
	#Size labels from axis
	ax[0].tick_params(axis='both', labelsize=font_little)
	ax[1].tick_params(axis='both', labelsize=font_little)	
		
	#Scientific notation	
	ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)

	#Eliminate line from label
	if style['legend'] == True:
		ax[0].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)	
		ax[1].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
	
	#Title
	if style['title'] == True:
		plt.rcParams['mathtext.fontset'] = 'cm'
		ax[0].set_title(keys[0], fontsize=font_big)	
		ax[1].set_title(keys[1], fontsize=font_big)
	
	#Size from offset text
	ax[0].yaxis.offsetText.set_fontsize(font_offset)
	ax[1].yaxis.offsetText.set_fontsize(font_offset)
	
	
	#Set Ticks in Axis Y
	# ax[0].set_yticks([-20, -10, 0, 10, 20]) 
	# ax[1].set_yticks([0, 4, 8, 12, 16]) 
	# ax[2].set_yticks([-20, -10, 0, 10, 20])
	
	#Set Ticks in Axis X
	if style['dom'] == 'other':
		ax[0].set_xticks(three_signals['dom']) 
		ax[1].set_xticks(three_signals['dom']) 
		
		ax[0].set_xticklabels(style['xticklabels']) 
		ax[1].set_xticklabels(style['xticklabels']) 
		
		
	
	#Set Limits in Axis X
	if style['ymax'] != None:
		ax[0].set_ylim(bottom=0, top=style['ymax'][0])
		ax[1].set_ylim(bottom=0, top=style['ymax'][1])
	
	if style['xmax'] != None:
		ax[0].set_xlim(left=0, right=style['xmax'][0])
		ax[1].set_xlim(left=0, right=style['xmax'][1])
		
	if style['autolabel'] == True:
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				height2 = height*10.
				ax[0].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=15)
		autolabel(bar0)
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				height2 = height*100.
				ax[1].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=15)
		autolabel(bar1)

	
	# plt.tight_layout()
	plt.show()
	return

def plot2v_thesis(three_signals, style):
	#Modules and global properties
	from matplotlib import font_manager
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	fig, ax = plt.subplots(ncols=1, nrows=2, sharex=style['sharex'], sharey=style['sharey'])
	
	#Values Fixed
	amp_factor = 1.1923
	font_big = 17*amp_factor
	font_little = 15*amp_factor
	font_label = 13*amp_factor
	font_offset = 15*amp_factor
	lim = 2
	# plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.175, top=0.90)
	# fig.set_size_inches(10.2, 4.0)
	# 0.92
	
	plt.subplots_adjust(left=0.15, right=0.95, bottom=0.165, top=0.92)
	# fig.set_size_inches(6.2, 4.2)
	fig.set_size_inches(5.92, 5)
	
	
	#Axis X
	if style['dom'] == 'time':
		fact = 1.
		ax[0].set_xlabel('Time [s]', fontsize=font_big)
		ax[1].set_xlabel('Time [s]', fontsize=font_big)
	elif style['dom'] == 'frequency':
		if style['kHz'] == 'ON':
			fact = 1000.
			# ax[0].set_xlabel('Frequency [kHz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [kHz]', fontsize=font_big)
		else:
			fact = 1.
			# ax[0].set_xlabel('Frequency [Hz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [Hz]', fontsize=font_big)
	elif style['dom'] == 'other':
		fact = 1.
		# ax[0].set_xlabel(style['xtitle'][0], fontsize=font_big)
		ax[1].set_xlabel(style['xtitle'][1], fontsize=font_big)
	
	
	#Plot	
	keys = list(three_signals)
	if style['type'] == 'plot':
		ax[0].plot(three_signals['dom']/fact, three_signals[keys[0]], label=keys[0])
		ax[1].plot(three_signals['dom']/fact, three_signals[keys[1]], label=keys[1])
	elif style['type'] == 'bar':
		bar0 = ax[0].bar(three_signals['dom']/fact, three_signals[keys[0]], label=keys[0])
		bar1 = ax[1].bar(three_signals['dom']/fact, three_signals[keys[1]], label=keys[1])
	
	# #Axis Y
	# if style['dom'] == 'time':
		# ax[0].set_ylabel('Amplitude [mV]', fontsize=font_big)
		# ax[1].set_ylabel('Amplitude [mV]', fontsize=font_big)
		
	# elif style['dom'] == 'frequency':
		# if style['mV'] == 'ON':
			# ax[0].set_ylabel('Magnitude [mV]', fontsize=font_big)
			# ax[1].set_ylabel('Magnitude [mV]', fontsize=font_big)
		# else:
			# ax[0].set_ylabel('Magnitude [g]', fontsize=font_big)
			# ax[1].set_ylabel('Magnitude [g]', fontsize=font_big)
	
	# elif style['dom'] == 'other':
		# ax[0].set_ylabel(style['ytitle'][0], fontsize=font_big)
		# ax[1].set_ylabel(style['ytitle'][1], fontsize=font_big)
		
		# params = {'mathtext.default': 'regular' }          
		# plt.rcParams.update(params)
		# ax[0].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[1].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[2].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)	
		
		# ax[0].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[1].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[2].set_title('Envelope spectrum', fontsize=font_offset)
	
	#Size labels from axis
	ax[0].tick_params(axis='both', labelsize=font_little)
	ax[1].tick_params(axis='both', labelsize=font_little)	
		
	#Scientific notation	
	ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)

	#Eliminate line from label
	if style['legend'] == True:
		ax[0].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)	
		ax[1].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
	
	#Title
	if style['title'] == True:
		plt.rcParams['mathtext.fontset'] = 'cm'
		ax[0].set_title('AE-1', fontsize=font_big)	
		# ax[1].set_title(keys[1], fontsize=font_big)
	
	
	# #Eliminate line from label
	# ax[0].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label, loc='upper center')	
	# ax[1].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label, loc='upper center')
	
	
	#Size from offset text
	ax[0].yaxis.offsetText.set_fontsize(font_offset)
	ax[1].yaxis.offsetText.set_fontsize(font_offset)
	
	
	
	
	
	#Set Ticks in Axis X
	if style['dom'] == 'other':
		ax[0].set_xticks(three_signals['dom']) 
		ax[1].set_xticks(three_signals['dom']) 
		
		ax[0].set_xticklabels(style['xticklabels']) 
		ax[1].set_xticklabels(style['xticklabels']) 
	
	#Hidde labels from Axis X in upper plots
	ax[0].tick_params(labelbottom=False) 
	# ax[1].tick_params(labelbottom=False) 
		
	
	#Set Limits in Axis X
	if style['ymax'] != None:
		ax[0].set_ylim(bottom=0, top=style['ymax'])
		ax[1].set_ylim(bottom=0, top=style['ymax'])
	
	if style['xmax'] != None:
		ax[0].set_xlim(left=0, right=style['xmax'])
		ax[1].set_xlim(left=0, right=style['xmax'])
		
	if style['autolabel'] == True:
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				height2 = height*10.
				ax[0].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=15)
		autolabel(bar0)
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				height2 = height*100.
				ax[1].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=15)
		autolabel(bar1)
	# ax.text(2, 0.65, , fontsize=font_little)
	#Set Ticks in Axis Y
	ax[0].set_yticks([0., 0.0003, 0.0006, 0.0009]) 
	ax[1].set_yticks([0., 0.0003, 0.0006, 0.0009]) 
	# ax[2].set_yticks([-20, -10, 0, 10, 20])
	
	fig.text(0.015, 0.5, 'Magnitude [mV$^{2}$]', va='center', rotation='vertical', fontsize=font_big)
	# plt.tight_layout()
	plt.show()
	return

def plot9_thesis_big(data, style):
	#Modules and global properties
	from matplotlib import font_manager
	
	# del font_manager.weight_dict['roman']
	# font_manager._rebuild()	
	# plt.rcParams['font.family'] = 'Times New Roman'
	
	params = {'font.family':'Times New Roman'}
	plt.rcParams.update(params)
	
	params = {'mathtext.fontset': 'stix' }
	
	
	# params = {'mathtext.default': 'regular' }
	plt.rcParams.update(params)
	
	fig, ax = plt.subplots(ncols=3, nrows=3, sharey='row')
	lim = 2
	
	
	font_big = 17+3
	font_title = 17+1
	font_little = 15+3
	font_label = 13+3
	font_offset = 15+3
	font_autolabel = 15+3
	font_caption = 23+3
	plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.1, top=0.95, hspace=0.635)
	# hspace=0.47
	fig.set_size_inches(14.2, 9) #7
	# 6.5
	fig.text(0.053-0.015, 0.0275, '(g)', fontsize=font_caption)
	fig.text(0.385-0.015, 0.0275, '(h)', fontsize=font_caption)
	fig.text(0.717-0.015, 0.0275, '(i)', fontsize=font_caption)
	
	fig.text(0.053-0.015, 0.35, '(d)', fontsize=font_caption)
	fig.text(0.385-0.015, 0.35, '(e)', fontsize=font_caption)
	fig.text(0.717-0.015, 0.35, '(f)', fontsize=font_caption)
	
	fig.text(0.053-0.015, 0.678, '(a)', fontsize=font_caption)
	fig.text(0.385-0.015, 0.678, '(b)', fontsize=font_caption)
	fig.text(0.717-0.015, 0.678, '(c)', fontsize=font_caption)
	# 0.522


	count = 0
	for i in range(3):
		for j in range(3):
			if style['dbae'] == 'ON':
				data_y = 20*np.log10(1000*data['y'][count])
			else:
				print(count)
				data_y = data['y'][count]
			ax[i][j].plot(data['x'][count], data_y, label=style['legend'][count])
			
			# if i == 0:
				# ax[i][j].set_yticks([-10, 10, 30, 50])
			
			ax[i][j].set_xlabel(style['xlabel'], fontsize=font_big)
			ax[i][j].set_ylabel(style['ylabel'], fontsize=font_big)
			ax[i][j].tick_params(axis='both', labelsize=font_little)
			ax[i][j].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
			if style['legend'][count] != None:
				if style['legend_line'] == 'OFF':
					ax[i][j].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
				else:
					ax[i][j].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
			if style['title'][count] != None:
				ax[i][j].set_title(style['title'][count], fontsize=font_title)
			ax[i][j].yaxis.offsetText.set_fontsize(font_offset)
			if style['customxlabels'] != None:
				# ax[i][j].set_xticklabels(style['xticklabels'])
				# ax[i][j].set_xticks(style['xticklabels'])
				ax[i][j].set_xticklabels(style['customxlabels'])
				ax[i][j].set_xticks(style['customxlabels'])
			if style['ylim'] != None:
				ax[i][j].set_ylim(bottom=style['ylim'][count][0], top=style['ylim'][count][1])		
			if style['xlim'] != None:
				ax[i][j].set_xlim(left=style['xlim'][count][0], right=style['xlim'][count][1])
			ax[i][j].grid(axis='both')
			count += 1
			# ax[2].set_xticks(three_signals['dom'])
		
			# ax[0].set_xticklabels(style['xticklabels']) 
	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)




	# ax[0][0].set_yticklabels([-15, 0, 15, 30])
	# ax[0][0].set_yticks([0, 0.8e-4, 1.6e-4, 2.4e-4])
	# ax[0][1].set_yticks([0, 0.8e-4, 1.6e-4, 2.4e-4])
	# ax[0][2].set_yticks([0, 0.8e-4, 1.6e-4, 2.4e-4])
	
	
	#Set Limits in Axis X
	
		
	#Set Vertical Lines
	if style['vlines'] != None:
		ax.vlines(style['vlines'], ymax=style['range_lines'][1], ymin=style['range_lines'][0], linestyles='dashed')
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		pos_v = 57.5 #for temp
		pos_v = 130. #for fix
		pos_v = 0.235 #for fix
		ax.annotate(s='End $1^{st}$ MC', xy=[style['vlines'][1]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $2^{nd}$ MC', xy=[style['vlines'][2]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $3^{rd}$ MC', xy=[style['vlines'][3]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $4^{th}$ MC', xy=[style['vlines'][4]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $5^{th}$ MC', xy=[style['vlines'][5]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $6^{th}$ MC', xy=[style['vlines'][6]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $7^{th}$ MC', xy=[style['vlines'][7]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $8^{th}$ MC', xy=[style['vlines'][8]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		
	# ax[i][j].set_xticklabels(style['customxlabels'])
	# ax[0][0].set_yticks([-20, -10, 0, 10, 20])
	# ax[0][1].set_yticks([-20, -10, 0, 10, 20])
	# ax[0][2].set_yticks([-20, -10, 0, 10, 20])
	
	# plt.tight_layout()
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return
	
	
def plot6_thesis_big(data, style):
	#Modules and global properties
	from matplotlib import font_manager
	
	# del font_manager.weight_dict['roman']
	# font_manager._rebuild()	
	# plt.rcParams['font.family'] = 'Times New Roman'
	
	params = {'font.family':'Times New Roman'}
	plt.rcParams.update(params)
	
	params = {'mathtext.fontset': 'stix' }
	
	
	# params = {'mathtext.default': 'regular' }
	plt.rcParams.update(params)
	
	fig, ax = plt.subplots(ncols=3, nrows=2, sharey='row')
	lim = 2
	
	
	font_big = 17+3
	font_little = 15+3
	font_label = 13+3
	font_offset = 15+3
	font_autolabel = 15+3
	font_caption = 23+3
	plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.15, top=0.95, hspace=0.52)
	# hspace=0.47
	fig.set_size_inches(14.2, 7)
	# 6.5
	fig.text(0.053-0.015, 0.04, '(d)', fontsize=font_caption)
	fig.text(0.385-0.015, 0.04, '(e)', fontsize=font_caption)
	fig.text(0.717-0.015, 0.04, '(f)', fontsize=font_caption)
	
	fig.text(0.053-0.015, 0.528, '(a)', fontsize=font_caption)
	fig.text(0.385-0.015, 0.528, '(b)', fontsize=font_caption)
	fig.text(0.717-0.015, 0.528, '(c)', fontsize=font_caption)
	# 0.522


	count = 0
	for i in range(2):
		for j in range(3):
			if style['dbae'] == 'ON':
				data_y = 20*np.log10(1000*data['y'][count])
			else:
				data_y = data['y'][count]
			ax[i][j].plot(data['x'][count], data_y, label=style['legend'][count])
			
			if i == 0:
				ax[i][j].set_yticks([-10, 10, 30, 50])
			
			ax[i][j].set_xlabel(style['xlabel'], fontsize=font_big)
			ax[i][j].set_ylabel(style['ylabel'], fontsize=font_big)
			ax[i][j].tick_params(axis='both', labelsize=font_little)
			ax[i][j].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
			if style['legend'][count] != None:
				if style['legend_line'] == 'OFF':
					ax[i][j].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
				else:
					ax[i][j].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
			if style['title'][count] != None:
				ax[i][j].set_title(style['title'][count], fontsize=font_big)
			ax[i][j].yaxis.offsetText.set_fontsize(font_offset)
			if i == 1:
				if style['customxlabels'] != None:
					# ax[i][j].set_xticklabels(style['xticklabels'])
					# ax[i][j].set_xticks(style['xticklabels'])
					ax[i][j].set_xticklabels(style['customxlabels'])
					ax[i][j].set_xticks(style['customxlabels'])
			if style['ylim'] != None:
				ax[i][j].set_ylim(bottom=style['ylim'][count][0], top=style['ylim'][count][1])		
			if style['xlim'] != None:
				ax[i][j].set_xlim(left=style['xlim'][count][0], right=style['xlim'][count][1])
			ax[i][j].grid(axis='both')
			count += 1
			# ax[2].set_xticks(three_signals['dom'])
		
			# ax[0].set_xticklabels(style['xticklabels']) 
	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)




	# ax[0][1].set_yticklabels([-15, 0, 15, 30])
	# ax[0][1].set_yticks([-15, 0, 15, 30])
	
	
	#Set Limits in Axis X
	
		
	#Set Vertical Lines
	if style['vlines'] != None:
		ax.vlines(style['vlines'], ymax=style['range_lines'][1], ymin=style['range_lines'][0], linestyles='dashed')
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		pos_v = 57.5 #for temp
		pos_v = 130. #for fix
		pos_v = 0.235 #for fix
		ax.annotate(s='End $1^{st}$ MC', xy=[style['vlines'][1]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $2^{nd}$ MC', xy=[style['vlines'][2]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $3^{rd}$ MC', xy=[style['vlines'][3]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $4^{th}$ MC', xy=[style['vlines'][4]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $5^{th}$ MC', xy=[style['vlines'][5]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $6^{th}$ MC', xy=[style['vlines'][6]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $7^{th}$ MC', xy=[style['vlines'][7]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $8^{th}$ MC', xy=[style['vlines'][8]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		
	# ax[i][j].set_xticklabels(style['customxlabels'])
	ax[0][0].set_yticks([-20, -10, 0, 10, 20])
	ax[0][1].set_yticks([-20, -10, 0, 10, 20])
	ax[0][2].set_yticks([-20, -10, 0, 10, 20])
	
	# plt.tight_layout()
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return

	

def plot3_thesis_new(data, style):
	#Modules and global properties
	from matplotlib import font_manager
	
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	
	params = {'mathtext.default': 'regular' }          
	plt.rcParams.update(params)
	
	fig, ax = plt.subplots(ncols=3, nrows=1, sharey='row')
	# fig, ax = plt.subplots(ncols=3, nrows=1)
	lim = 2
	
	
	font_big = 17+3
	font_little = 15+3
	font_label = 13+3
	font_offset = 15+3
	font_autolabel = 15+3
	font_caption = 23+3
	# plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.15, top=0.95, hspace=0.52)
	# # hspace=0.47
	# fig.set_size_inches(14.2, 3.6)
	# # 6.5
	# fig.text(0.053-0.015, 0.04, '(d)', fontsize=font_caption)
	# fig.text(0.385-0.015, 0.04, '(e)', fontsize=font_caption)
	# fig.text(0.717-0.015, 0.04, '(f)', fontsize=font_caption)
	
	# fig.text(0.053-0.015, 0.528, '(a)', fontsize=font_caption)
	# fig.text(0.385-0.015, 0.528, '(b)', fontsize=font_caption)
	# fig.text(0.717-0.015, 0.528, '(c)', fontsize=font_caption)
	# # 0.522
	
	
	# # plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.213, top=0.89)
	# plt.subplots_adjust(wspace=0.32, left=0.066, right=0.98, bottom=0.213, top=0.81)
	# # fig.set_size_inches(14.2, 4.0)
	# fig.set_size_inches(14.2, 4.6)

	# fig.text(0.059, 0.05, '(a)', fontsize=font_caption)
	# fig.text(0.387, 0.05, '(b)', fontsize=font_caption)
	# fig.text(0.717, 0.05, '(c)', fontsize=font_caption)

	
	
	# plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.21, top=0.89)
	plt.subplots_adjust(wspace=0.32+0.02, left=0.065+0.01, right=0.98-0.01, bottom=0.213, top=0.89)
	fig.set_size_inches(14.2, 4.0)
	fig.text(0.059, 0.03, '(a)', fontsize=font_caption)
	fig.text(0.387, 0.03, '(b)', fontsize=font_caption)
	fig.text(0.717, 0.03, '(c)', fontsize=font_caption)

	

	count = 0
	
	for j in range(3):
		if style['dbae'] == 'ON':
			data_y = 20*np.log10(1000*data['y'][count])
		else:
			data_y = data['y'][count]
		# ax[j].set_yticks([-30, -15, 0, 15, 30])
		
		if style['ylog'] == 'ON':
			ax[j].semilogy(data['x'][count], data_y, label=style['legend'][count])
		else:
			ax[j].plot(data['x'][count], data_y, label=style['legend'][count], marker='D', ls='', color='r')
			# ax[j].plot(data['x'][count], data_y, label=style['legend'][count], marker='o', ls='')
			# ax[j].plot(data['x'][count], data_y, label=style['legend'][count])
			# ax[j].bar(data['x'][count], data_y, label=style['legend'][count])
		
		ax[j].set_xlabel(style['xlabel'], fontsize=font_big)
		ax[j].set_ylabel(style['ylabel'], fontsize=font_big)
		ax[j].tick_params(axis='both', labelsize=font_little)
		# ax[j].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
		if style['legend'][count] != None:
			if style['legend_line'] == 'OFF':
				ax[j].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
			else:
				ax[j].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
		if style['title'][count] != None:
			ax[j].set_title(style['title'][count], fontsize=font_big)
		ax[j].yaxis.offsetText.set_fontsize(font_offset)
		if j >= 0:
			if style['customxlabels'] != None:
				# ax[i][j].set_xticklabels(style['xticklabels'])
				# ax[i][j].set_xticks(style['xticklabels'])
				ax[j].set_xticklabels(style['customxlabels'])
				ax[j].set_xticks(style['customxlabels'])
		if style['ylim'] != None:
			ax[j].set_ylim(bottom=style['ylim'][count][0], top=style['ylim'][count][1])		
		if style['xlim'] != None:
			ax[j].set_xlim(left=style['xlim'][count][0], right=style['xlim'][count][1])
		ax[j].grid(axis='both')
		count += 1
		# ax[2].set_xticks(three_signals['dom'])
	
		# ax[0].set_xticklabels(style['xticklabels']) 
	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)




	# ax[0][1].set_yticklabels([-15, 0, 15, 30])
	# ax[0][1].set_yticks([-15, 0, 15, 30])
	
	
	#Set Limits in Axis X
	
		
	#Set Vertical Lines
	if style['vlines'] != None:
		ax.vlines(style['vlines'], ymax=style['range_lines'][1], ymin=style['range_lines'][0], linestyles='dashed')
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		pos_v = 57.5 #for temp
		pos_v = 130. #for fix
		pos_v = 0.235 #for fix
		ax.annotate(s='End $1^{st}$ MC', xy=[style['vlines'][1]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $2^{nd}$ MC', xy=[style['vlines'][2]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $3^{rd}$ MC', xy=[style['vlines'][3]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $4^{th}$ MC', xy=[style['vlines'][4]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $5^{th}$ MC', xy=[style['vlines'][5]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $6^{th}$ MC', xy=[style['vlines'][6]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $7^{th}$ MC', xy=[style['vlines'][7]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End $8^{th}$ MC', xy=[style['vlines'][8]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		
	# ax[0].set_yticks([1.e4, 1.e5, 1.e6, 1.e7])
	# ax[1].set_yticks([1.e4, 1.e5, 1.e6, 1.e7])
	# ax[2].set_yticks([1.e4, 1.e5, 1.e6, 1.e7])
	
	# ax[0].set_yticks([-50, -30, -10, 10, 30])
	# ax[1].set_yticks([-50, -30, -10, 10, 30])
	# ax[2].set_yticks([-50, -30, -10, 10, 30])
	
	
	
	# ax[0].set_xticks([1, 2, 3, 4, 5, 6])
	# ax[1].set_xticks([1, 2, 3, 4, 5, 6])
	# ax[2].set_xticks([1, 2, 3, 4, 5, 6])
	
	# ax[0].set_xticks([0, 100, 200, 300, 400, 500])
	# ax[1].set_xticks([0, 100, 200, 300, 400, 500])
	# ax[2].set_xticks([0, 100, 200, 300, 400, 500])
	
	# ax[0].set_xticklabels([1, 2, 4, 7, 9, 10])
	# ax[1].set_xticklabels([1, 2, 4, 7, 9, 10])
	# ax[2].set_xticklabels([1, 2, 4, 7, 9, 10])
	
	
	# plt.tight_layout()
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return

def plot3_thesis_new_varb(data1, data2, style):
	#Modules and global properties
	from matplotlib import font_manager
	
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	
	params = {'mathtext.default': 'regular' }          
	plt.rcParams.update(params)
	
	fig, ax = plt.subplots(ncols=3, nrows=1, sharey='row')
	lim = 2
	
	
	font_big = 17+3
	font_little = 15+3
	font_label = 13+3
	font_offset = 15+3
	font_autolabel = 15+3
	font_caption = 23+3
	# plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.15, top=0.95, hspace=0.52)
	# # hspace=0.47
	# fig.set_size_inches(14.2, 3.6)
	# # 6.5
	# fig.text(0.053-0.015, 0.04, '(d)', fontsize=font_caption)
	# fig.text(0.385-0.015, 0.04, '(e)', fontsize=font_caption)
	# fig.text(0.717-0.015, 0.04, '(f)', fontsize=font_caption)
	
	# fig.text(0.053-0.015, 0.528, '(a)', fontsize=font_caption)
	# fig.text(0.385-0.015, 0.528, '(b)', fontsize=font_caption)
	# fig.text(0.717-0.015, 0.528, '(c)', fontsize=font_caption)
	# # 0.522
	
	
	# # plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.213, top=0.89)
	# plt.subplots_adjust(wspace=0.32, left=0.066, right=0.98, bottom=0.213, top=0.81)
	# # fig.set_size_inches(14.2, 4.0)
	# fig.set_size_inches(14.2, 4.6)

	# fig.text(0.059, 0.05, '(a)', fontsize=font_caption)
	# fig.text(0.387, 0.05, '(b)', fontsize=font_caption)
	# fig.text(0.717, 0.05, '(c)', fontsize=font_caption)

	
	
	# plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.21, top=0.89)
	plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.213, top=0.82)
	fig.set_size_inches(14.2, 4.7)
	fig.text(0.059, 0.03, '(a)', fontsize=font_caption)
	fig.text(0.387, 0.03, '(b)', fontsize=font_caption)
	fig.text(0.717, 0.03, '(c)', fontsize=font_caption)

	

	count = 0
	
	for j in range(3):

		
		if style['ylog'] == 'ON':
			ax[j].semilogy(data['x'][count], data_y, label=style['legend'][count])
		else:
			ax[j].plot(data1['x'][count], data1['y'][count], label='Measurement without crack', marker='o', ls='', color='blue', alpha=0.5)
			ax[j].plot(data2['x'][count], data2['y'][count], label='Measurement with crack', marker='s', ls='', color='red', alpha=0.5)
		
		ax[j].set_xlabel(style['xlabel'], fontsize=font_big)
		ax[j].set_ylabel(style['ylabel'], fontsize=font_big)
		ax[j].tick_params(axis='both', labelsize=font_little)
		# ax[j].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
		
		# if style['legend'][count] != None:
			# if style['legend_line'] == 'OFF':
				# ax[j].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
			# else:
				# ax[j].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
				
		if style['title'][count] != None:
			ax[j].set_title(style['title'][count], fontsize=font_big)
		ax[j].yaxis.offsetText.set_fontsize(font_offset)
		if j >= 0:
			if style['customxlabels'] != None:
				# ax[i][j].set_xticklabels(style['xticklabels'])
				# ax[i][j].set_xticks(style['xticklabels'])
				ax[j].set_xticklabels(style['customxlabels'])
				ax[j].set_xticks(style['customxlabels'])
		if style['ylim'] != None:
			ax[j].set_ylim(bottom=style['ylim'][count][0], top=style['ylim'][count][1])		
		if style['xlim'] != None:
			ax[j].set_xlim(left=style['xlim'][count][0], right=style['xlim'][count][1])
		ax[j].grid(axis='both')
		count += 1
		# ax[2].set_xticks(three_signals['dom'])
	
		# ax[0].set_xticklabels(style['xticklabels']) 
	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)




	# ax[0][1].set_yticklabels([-15, 0, 15, 30])
	# ax[0][1].set_yticks([-15, 0, 15, 30])
	
	
	#Set Limits in Axis X
	
	

	# if len(style['legend']['first']) == 4:
	ax[2].legend(fontsize=font_label, loc=(-1.75,1.13), handletextpad=0.3, labelspacing=.3, ncol=2)
	# elif len(style['legend']['first']) == 3:
		# if style['legend']['first'][0].find('db') != -1:
			# ax[2].legend(fontsize=font_label, loc=(-1.4,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
		# else:
			# ax[2].legend(fontsize=font_label, loc=(-1.5,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
	
		
	#Set Vertical Lines
	
		
	# ax[0].set_yticks([1.e4, 1.e5, 1.e6, 1.e7, 1.e8, 1.e9])
	# ax[1].set_yticks([1.e4, 1.e5, 1.e6, 1.e7, 1.e8, 1.e9])
	# ax[2].set_yticks([1.e4, 1.e5, 1.e6, 1.e7, 1.e8, 1.e9])
	
	# ax[0].set_xticks([0, 100, 200, 300, 400, 500])
	# ax[1].set_xticks([0, 100, 200, 300, 400, 500])
	# ax[2].set_xticks([0, 100, 200, 300, 400, 500])
	
	
	# plt.tight_layout()
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return


def plot12_thesis_testbench(data, style):
	#Modules and global properties
	from matplotlib import font_manager
	
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	
	# params = {'mathtext.default': 'regular' }          
	# plt.rcParams.update(params)
	# plt.rc_context({'axes.autolimit_mode': 'round_numbers'})
	
	fig, ax = plt.subplots(ncols=3, nrows=4, sharey=False, sharex=True)
	lim = 2
	
	from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
	
	
	class FixedOrderFormatter(ScalarFormatter):
		def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
			self._order_of_mag = order_of_mag
			ScalarFormatter.__init__(self, useOffset=useOffset, useMathText=useMathText)
		def _set_orderOfMagnitude(self, range):
			"""Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
			self.orderOfMagnitude = self._order_of_mag
	
	
	font_big = 17+3
	font_little = 15+3
	font_label = 13+3
	font_offset = 15+3
	font_autolabel = 15+3
	font_caption = 23+3
	if style['n_data'] == 1:
		plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.050, top=0.96, hspace=0.5)
		fig.set_size_inches(14.2, 9)
	else:
		# plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.080, top=0.90, hspace=0.42)
		plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.060, top=0.90, hspace=0.475)
		fig.set_size_inches(14.2, 9.6)
	
	
	
	
	# font_big = 17+3
	# font_little = 15+3
	# font_label = 13+3
	# font_offset = 15+3
	# font_autolabel = 15+3
	# font_caption = 23+3
	# if style['n_data'] == 1:
		# plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.080, top=0.96, hspace=0.42)
		# fig.set_size_inches(14.2, 9)
	# else:
		# plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.080, top=0.90, hspace=0.42)
		# fig.set_size_inches(14.2, 9.6)

	# fig.text(0.053-0.015, 0.04, '(d)', fontsize=font_caption)
	# fig.text(0.385-0.015, 0.04, '(e)', fontsize=font_caption)
	# fig.text(0.717-0.015, 0.04, '(f)', fontsize=font_caption)
	
	from matplotlib.ticker import ScalarFormatter

	class ScalarFormatterForceFormat(ScalarFormatter):
		def _set_format(self,vmin,vmax):  # Override function that finds format to use.
			self.format = "%1.2f"  # Give format here
	
	class ScalarFormatterForceFormat1(ScalarFormatter):
		def _set_format(self,vmin,vmax):  # Override function that finds format to use.
			self.format = "%1.1f"  # Give format here
	class ScalarFormatterForceFormat2(ScalarFormatter):
		def _set_format(self,vmin,vmax):  # Override function that finds format to use.
			self.format = "%1.2f"  # Give format here
	class ScalarFormatterForceFormat3(ScalarFormatter):
		def _set_format(self,vmin,vmax):  # Override function that finds format to use.
			self.format = "%1.3f"  # Give format here
	class ScalarFormatterForceFormat0(ScalarFormatter):
		def _set_format(self,vmin,vmax):  # Override function that finds format to use.
			self.format = "%1.0f"  # Give format here
			
	# fig.text(0.053-0.015, 0.528, '(a)', fontsize=font_caption)
	# fig.text(0.385-0.015, 0.528, '(b)', fontsize=font_caption)
	# fig.text(0.717-0.015, 0.528, '(c)', fontsize=font_caption)

	marker = ['o', 's', 'v', '^']
	count = 0
	for i in range(4):
		for j in range(3):
			if style['n_data'] == 1:
				ax[i][j].bar(data['x'][count], data['y'][count], label=style['legend'][count])
			else:
				for k in range(style['n_data']):
					# print(k)
					# print(data['y'][k][count])
					ax[i][j].plot(data['x'][count], data['y'][k][count], label=style['legend'][k], marker=marker[k])
				
			# if i == 3:
				# ax[i][j].set_xlabel(style['xlabel'], fontsize=font_big)
			
			if style['ylabel'][count] != None:
				ax[i][j].set_ylabel(style['ylabel'][count], fontsize=font_big)
			ax[i][j].tick_params(axis='both', labelsize=font_little)
			# ax[i][j].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
			if style['n_data'] == 1:
				if style['legend'][count] != None:
					if style['legend_line'] == 'OFF':
						ax[i][j].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
					else:
						ax[i][j].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
			if style['title'][count] != None:
				ax[i][j].set_title(style['title'][count], fontsize=font_big)
			ax[i][j].yaxis.offsetText.set_fontsize(font_offset)
			
			# if i == 1:
			if style['customxlabels'] != None:
				ax[i][j].set_xticks(data['x'][count])
	
				ax[i][j].set_xticklabels(style['customxlabels']) 
					
					
					# ax[i][j].set_xticklabels(style['customxlabels'])
					# ax[i][j].set_xticks(style['customxlabels'])
			if style['ylim'] != None:
				ax[i][j].set_ylim(bottom=style['ylim'][count][0], top=style['ylim'][count][1])		
			if style['xlim'] != None:
				ax[i][j].set_xlim(left=style['xlim'][0], right=style['xlim'][1])
			ax[i][j].grid(axis='both')
			
			# ax[i][j].legend()
			count += 1
			
			# yfmt = ScalarFormatterForceFormat()
			# yfmt.set_powerlimits((-lim,lim))
			# ax[i][j].yaxis.set_major_formatter(yfmt)   
	
	#MAX
	lim = 2
	ax[0][0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat1()
	yfmt.set_powerlimits((-lim,lim))
	ax[0][0].yaxis.set_major_formatter(yfmt)
	# yticks_ = [0, 600 , 1100] #kurt
	# yticks_ = [300, 900 , 1500] #cyclo
	# yticks_ = [0, 600, 1200] #emd
	# yticks_ = [800, 1400 , 2000] #raw
	yticks_ = [0, 1500 , 3000] #daubechies
	ax[0][0].set_yticks(yticks_)
	ax[0][0].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#RMS
	lim = 4
	ax[0][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat0()
	yfmt.set_powerlimits((-lim,lim))
	ax[0][1].yaxis.set_major_formatter(yfmt)
	# yticks_ = [0, 50, 100] #kurt
	# yticks_ = [20, 70, 120] #cyclo
	# yticks_ = [0, 25, 50] #emd
	# yticks_ = [100, 125, 150] #raw
	yticks_ = [0, 125, 250] #db
	ax[0][1].set_yticks(yticks_)
	ax[0][1].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	# #AVG
	# lim = 4
	# ax[0][2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat0()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[0][2].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [-0.00004, 0.0, 0.00004] #kurt
	# # yticks_ = [-0.000015, 0.0, 0.000015] #cyclo
	# yticks_ = [-10, 50, 110] #raw
	# ax[0][2].set_yticks(yticks_)
	# ax[0][2].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#CRF
	lim = 4
	ax[0][2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat0()
	yfmt.set_powerlimits((-lim,lim))
	ax[0][2].yaxis.set_major_formatter(yfmt)
	# yticks_ = [4, 17, 30] #kurto, cyclo
	# yticks_ = [0, 50, 100] #emd
	# yticks_ = [6, 10, 14] #raw
	yticks_ = [0, 15, 30] #db
	ax[0][2].set_yticks(yticks_)
	ax[0][2].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	
	#STD
	lim = 4
	ax[1][0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat0()
	yfmt.set_powerlimits((-lim,lim))
	ax[1][0].yaxis.set_major_formatter(yfmt)
	# yticks_ = [0, 50, 100] #kurto
	# yticks_ = [20, 70, 120] #cyclo
	# yticks_ = [0, 25, 50] #emd
	# yticks_ = [90, 115, 140] #raw
	yticks_ = [0, 125, 250] #db
	ax[1][0].set_yticks(yticks_)
	ax[1][0].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#AVS
	lim = 2
	ax[1][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat1()
	yfmt.set_powerlimits((-lim,lim))
	ax[1][1].yaxis.set_major_formatter(yfmt)
	# yticks_ = [0.0, 0.015, 0.03] #kurt, cyclo, emd
	# yticks_ = [0.04, 0.06, 0.08] #raw
	yticks_ = [0., 0.5, 1.0] #db
	ax[1][1].set_yticks(yticks_)
	ax[1][1].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#KUR
	lim = 4
	ax[1][2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat0()
	yfmt.set_powerlimits((-lim,lim))
	ax[1][2].yaxis.set_major_formatter(yfmt)
	# yticks_ = [0, 15, 30] #kurt
	# yticks_ = [0, 7, 14] #cyclo
	# yticks_ = [0, 750, 1500] #emd
	# yticks_ = [2.5, 3., 3.5] #raw
	# yticks_ = [0, 5., 10] #db
	yticks_ = [0, 8., 16] #sym
	ax[1][2].set_yticks(yticks_)
	ax[1][2].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	# #VAR
	# lim = 2
	# ax[2][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat1()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[2][1].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [0.e3, 5.e3, 10.e3] #kurto, cyclo
	# yticks_ = [0.8e4, 1.4e4, 2.e4] #raw
	# ax[2][1].set_yticks(yticks_)
	# ax[2][1].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#CEF
	lim = 4
	ax[2][0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat0()
	yfmt.set_powerlimits((-lim,lim))
	ax[2][0].yaxis.set_major_formatter(yfmt)
	# yticks_ = [100, 125, 150] #kurt
	# yticks_ = [100, 120, 140] #cyclo
	# yticks_ = [50, 200, 350] #emd
	# yticks_ = [170, 185, 200] #raw
	yticks_ = [0, 35, 70] #db
	ax[2][0].set_yticks(yticks_)
	ax[2][0].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#SEN
	lim = 4
	ax[2][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat0()
	yfmt.set_powerlimits((-lim,lim))
	ax[2][1].yaxis.set_major_formatter(yfmt)
	# yticks_ = [20, 21, 22] #kurt, cyclo
	# yticks_ = [14, 19, 23] #emd
	# yticks_ = [21, 21.3, 21.6] #raw
	yticks_ = [12, 16, 20] #db
	ax[2][1].set_yticks(yticks_)
	ax[2][1].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	# #VAS
	# lim = 2
	# ax[2][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat1()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[2][1].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [0., 0.0025, 0.005]  #kurt
	# # yticks_ = [0., 0.005, 0.01] #cyclo
	# yticks_ = [0.005, 0.01, 0.015] #raw
	# ax[2][1].set_yticks(yticks_)
	# ax[2][1].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#p17
	lim = 4
	ax[2][2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat0()
	yfmt.set_powerlimits((-lim,lim))
	ax[2][2].yaxis.set_major_formatter(yfmt)
	# yticks_ = [0.0, 1.5, 3.0] #kurt
	# yticks_ = [1.6, 2.4, 3.2] #cyclo
	# yticks_ = [0, 15, 30] #emd
	# yticks_ = [25, 30, 35] #raw
	yticks_ = [0, 8, 16] #db
	ax[2][2].set_yticks(yticks_)
	ax[2][2].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#p21
	lim = 2
	ax[3][0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat1()
	yfmt.set_powerlimits((-lim,lim))
	ax[3][0].yaxis.set_major_formatter(yfmt)
	# yticks_ = [0.0, 0.015, 0.03] #kurt
	# yticks_ = [0.01, 0.02, 0.03] #cyclo
	# yticks_ = [0.0, 0.05, 0.1] #emd
	# yticks_ = [0.14, 0.16, 0.18] #raw
	yticks_ = [0.1, 0.3, 0.5] #db
	ax[3][0].set_yticks(yticks_)
	ax[3][0].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#STS
	lim = 4
	ax[3][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat0()
	yfmt.set_powerlimits((-lim,lim))
	ax[3][1].yaxis.set_major_formatter(yfmt)
	# yticks_ = [4., 15, 26] #kurt
	# yticks_ = [10., 16, 22] #cyclo
	# yticks_ = [0., 75, 150] #emd
	# yticks_ = [110., 125, 140] #raw
	yticks_ = [10, 40, 70] #db
	ax[3][1].set_yticks(yticks_)
	ax[3][1].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#P24
	lim = 2
	ax[3][2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat2()
	yfmt.set_powerlimits((-lim,lim))
	ax[3][2].yaxis.set_major_formatter(yfmt)	
	# yticks_ = [0.0, 0.025, 0.05] #kurto
	# yticks_ = [0.0, 0.04, 0.08] #cyclo
	# yticks_ = [0.0, 0.035, 0.07] #emd
	# yticks_ = [0.08, 0.10, 0.12] #raw
	yticks_ = [0.0, 0.05, 0.1] #db
	ax[3][2].set_yticks(yticks_)
	ax[3][2].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	
	#Visibility
	count = 0
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		if count < 12:
			for tk in ax_it.get_xticklabels():
				tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)
		count +=1
	# plt.setp(ax[1][1].get_xticklabels(), visible=False)
	
	
	if style['n_data'] != 1:
		if style['n_data'] == 4:
			# ax[2][0].legend(fontsize=font_label, loc=(-1.71,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
			ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.7,1.35), handletextpad=0.3, labelspacing=.3)
		
		if style['n_data'] == 2:
			# ax[2][0].legend(fontsize=font_label, loc=(-1.71,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
			ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.3,1.35), handletextpad=0.3, labelspacing=.3)
			# 1.35
		elif style['n_data'] == 3:
			# if style['legend']['first'][0].find('db') != -1:
			print('++++++++++ legend')
			
			if style['legend'][0].find('db') != -1:
				ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.4,1.35), handletextpad=0.3, labelspacing=.3)
			else:
				ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.45,1.35), handletextpad=0.3, labelspacing=.3)
			
			# else:
				# ax[2][0].legend(fontsize=font_label, loc=(-1.5,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
		
	
	
	#Set Limits in Axis X
	
	

	# plt.rcParams['axes.xmargin'] = 0
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	

	# ax[1][1].set_aspect('equal')
	
	# plt.tight_layout()
	if style['output'] == 'plot':
		
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return
	
def plot15_thesis_big(data, style):
	#Modules and global properties
	from matplotlib import font_manager
	
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	
	# params = {'mathtext.default': 'regular' }          
	# plt.rcParams.update(params)
	# plt.rc_context({'axes.autolimit_mode': 'round_numbers'})
	
	fig, ax = plt.subplots(ncols=3, nrows=5, sharey=False, sharex=True)
	lim = 2
	
	from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
	
	
	class FixedOrderFormatter(ScalarFormatter):
		def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
			self._order_of_mag = order_of_mag
			ScalarFormatter.__init__(self, useOffset=useOffset, useMathText=useMathText)
		def _set_orderOfMagnitude(self, range):
			"""Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
			self.orderOfMagnitude = self._order_of_mag
	
	
	# font_big = 17+3
	# font_little = 15+3
	# font_label = 13+3
	# font_offset = 15+3
	# font_autolabel = 15+3
	# font_caption = 23+3
	# if style['n_data'] == 1:
		# plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.080, top=0.96, hspace=0.42)
		# fig.set_size_inches(14.2, 9)
	# else:
		# plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.080, top=0.90, hspace=0.42)
		# fig.set_size_inches(14.2, 9.6)
	
	
	font_big = 17+3
	font_little = 15+3
	font_label = 13+3
	font_offset = 15+3
	font_autolabel = 15+3
	font_caption = 23+3
	if style['n_data'] == 1:
		plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.050, top=0.96, hspace=0.5)
		fig.set_size_inches(14.2, 9)
	else:
		plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.080, top=0.90, hspace=0.42)
		fig.set_size_inches(14.2, 9.6)

	# fig.text(0.053-0.015, 0.04, '(d)', fontsize=font_caption)
	# fig.text(0.385-0.015, 0.04, '(e)', fontsize=font_caption)
	# fig.text(0.717-0.015, 0.04, '(f)', fontsize=font_caption)
	from matplotlib.ticker import ScalarFormatter

	class ScalarFormatterForceFormat(ScalarFormatter):
		def _set_format(self,vmin,vmax):  # Override function that finds format to use.
			self.format = "%1.2f"  # Give format here
	
	class ScalarFormatterForceFormat1(ScalarFormatter):
		def _set_format(self,vmin,vmax):  # Override function that finds format to use.
			self.format = "%1.1f"  # Give format here
	class ScalarFormatterForceFormat2(ScalarFormatter):
		def _set_format(self,vmin,vmax):  # Override function that finds format to use.
			self.format = "%1.2f"  # Give format here
	class ScalarFormatterForceFormat3(ScalarFormatter):
		def _set_format(self,vmin,vmax):  # Override function that finds format to use.
			self.format = "%1.3f"  # Give format here
	class ScalarFormatterForceFormat0(ScalarFormatter):
		def _set_format(self,vmin,vmax):  # Override function that finds format to use.
			self.format = "%1.0f"  # Give format here
			
	# fig.text(0.053-0.015, 0.528, '(a)', fontsize=font_caption)
	# fig.text(0.385-0.015, 0.528, '(b)', fontsize=font_caption)
	# fig.text(0.717-0.015, 0.528, '(c)', fontsize=font_caption)

	marker = ['o', 's', 'v', '^']
	count = 0
	for i in range(5):
		for j in range(3):
			if style['n_data'] == 1:
				ax[i][j].bar(data['x'][count], data['y'][count], label=style['legend'][count])
			else:
				for k in range(style['n_data']):
					ax[i][j].plot(data['x'][count], data['y'][k][count], label=style['legend'][k], marker=marker[k])
				
			if i == 4:
				ax[i][j].set_xlabel(style['xlabel'], fontsize=font_big)
			if style['ylabel'][count] != None:
				ax[i][j].set_ylabel(style['ylabel'][count], fontsize=font_big)
			ax[i][j].tick_params(axis='both', labelsize=font_little)
			# ax[i][j].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
			if style['n_data'] == 1:
				if style['legend'][count] != None:
					if style['legend_line'] == 'OFF':
						ax[i][j].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
					else:
						ax[i][j].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
			if style['title'][count] != None:
				ax[i][j].set_title(style['title'][count], fontsize=font_big)
			ax[i][j].yaxis.offsetText.set_fontsize(font_offset)
			if i == 1:
				if style['customxlabels'] != None:
					ax[i][j].set_xticks(data['x'][count])
		
					ax[i][j].set_xticklabels(style['customxlabels']) 
					
					
					# ax[i][j].set_xticklabels(style['customxlabels'])
					# ax[i][j].set_xticks(style['customxlabels'])
			if style['ylim'] != None:
				ax[i][j].set_ylim(bottom=style['ylim'][count][0], top=style['ylim'][count][1])		
			if style['xlim'] != None:
				ax[i][j].set_xlim(left=style['xlim'][0], right=style['xlim'][1])
			ax[i][j].grid(axis='both')
			
			# ax[i][j].legend()
			count += 1
			
			# yfmt = ScalarFormatterForceFormat()
			# yfmt.set_powerlimits((-lim,lim))
			# ax[i][j].yaxis.set_major_formatter(yfmt)   
	
	# #MAX
	# lim = 2
	# ax[0][0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat1()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[0][0].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [0, 600 , 1100] #kurt
	# # yticks_ = [300, 900 , 1500] #cyclo
	# yticks_ = [800, 1400 , 2000] #raw
	# ax[0][0].set_yticks(yticks_)
	# ax[0][0].set_ylim(bottom=yticks_[0], top=yticks_[2])
	# #RMS
	# lim = 4
	# ax[0][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat0()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[0][1].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [0, 50, 100] #kurt
	# # yticks_ = [20, 70, 120] #cyclo
	# yticks_ = [100, 125, 150] #raw
	# ax[0][1].set_yticks(yticks_)
	# ax[0][1].set_ylim(bottom=yticks_[0], top=yticks_[2])
	# #AVG
	# lim = 4
	# ax[0][2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat0()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[0][2].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [-0.00004, 0.0, 0.00004] #kurt
	# # yticks_ = [-0.000015, 0.0, 0.000015] #cyclo
	# yticks_ = [-10, 50, 110] #raw
	# ax[0][2].set_yticks(yticks_)
	# ax[0][2].set_ylim(bottom=yticks_[0], top=yticks_[2])
	# #CRF
	# lim = 4
	# ax[1][0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat0()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[1][0].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [4, 17, 30] #kurto, cyclo
	# yticks_ = [6, 10, 14] #raw

	# ax[1][0].set_yticks(yticks_)
	# ax[1][0].set_ylim(bottom=yticks_[0], top=yticks_[2])
	# #STD
	# lim = 4
	# ax[1][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat0()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[1][1].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [0, 50, 100] #kurto
	# # yticks_ = [20, 70, 120] #cyclo
	# yticks_ = [90, 115, 140] #raw
	# ax[1][1].set_yticks(yticks_)
	# ax[1][1].set_ylim(bottom=yticks_[0], top=yticks_[2])
	# #AVS
	# lim = 2
	# ax[1][2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat1()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[1][2].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [0.0, 0.015, 0.03] #kurt, cyclo
	# yticks_ = [0.04, 0.06, 0.08] #raw
	# ax[1][2].set_yticks(yticks_)
	# ax[1][2].set_ylim(bottom=yticks_[0], top=yticks_[2])
	# #KUR
	# lim = 4
	# ax[2][0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat1()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[2][0].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [0, 15, 30] #kurt
	# # yticks_ = [0, 7, 14] #cyclo
	# yticks_ = [2.5, 3., 3.5] #raw

	# ax[2][0].set_yticks(yticks_)
	# ax[2][0].set_ylim(bottom=yticks_[0], top=yticks_[2])
	# #VAR
	# lim = 2
	# ax[2][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat1()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[2][1].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [0.e3, 5.e3, 10.e3] #kurto, cyclo
	# yticks_ = [0.8e4, 1.4e4, 2.e4] #raw
	# ax[2][1].set_yticks(yticks_)
	# ax[2][1].set_ylim(bottom=yticks_[0], top=yticks_[2])
	# #CEF
	# lim = 4
	# ax[2][2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat0()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[2][2].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [100, 125, 150] #kurt
	# # yticks_ = [100, 120, 140] #cyclo
	# yticks_ = [170, 185, 200] #raw

	# ax[2][2].set_yticks(yticks_)
	# ax[2][2].set_ylim(bottom=yticks_[0], top=yticks_[2])
	# #SEN
	# lim = 4
	# ax[3][0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat1()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[3][0].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [20, 21, 22] #kurt, cyclo
	# yticks_ = [21, 21.3, 21.6] #raw
	# ax[3][0].set_yticks(yticks_)
	# ax[3][0].set_ylim(bottom=yticks_[0], top=yticks_[2])
	# #VAS
	# lim = 2
	# ax[3][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat1()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[3][1].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [0., 0.0025, 0.005]  #kurt
	# # yticks_ = [0., 0.005, 0.01] #cyclo
	# yticks_ = [0.005, 0.01, 0.015] #raw

	# ax[3][1].set_yticks(yticks_)
	# ax[3][1].set_ylim(bottom=yticks_[0], top=yticks_[2])
	# #p17
	# lim = 4
	# ax[3][2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat1()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[3][2].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [0.0, 1.5, 3.0] #kurt
	# # yticks_ = [1.6, 2.4, 3.2] #cyclo
	# yticks_ = [25, 30, 35] #raw
	# ax[3][2].set_yticks(yticks_)
	# ax[3][2].set_ylim(bottom=yticks_[0], top=yticks_[2])
	# #p21
	# lim = 2
	# ax[4][0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat2()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[4][0].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [0.0, 0.015, 0.03] #kurt
	# # yticks_ = [0.01, 0.02, 0.03] #cyclo
	# yticks_ = [0.14, 0.16, 0.18] #raw
	# ax[4][0].set_yticks(yticks_)
	# ax[4][0].set_ylim(bottom=yticks_[0], top=yticks_[2])
	# #STS
	# lim = 4
	# ax[4][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat0()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[4][1].yaxis.set_major_formatter(yfmt)
	# # yticks_ = [4., 15, 26] #kurt
	# # yticks_ = [10., 16, 22] #cyclo
	# yticks_ = [110., 125, 140] #raw
	# ax[4][1].set_yticks(yticks_)
	# ax[4][1].set_ylim(bottom=yticks_[0], top=yticks_[2])
	# #P24
	# lim = 2
	# ax[4][2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	# yfmt = ScalarFormatterForceFormat2()
	# yfmt.set_powerlimits((-lim,lim))
	# ax[4][2].yaxis.set_major_formatter(yfmt)	
	# # yticks_ = [0.0, 0.025, 0.05] #kurto
	# # yticks_ = [0.0, 0.04, 0.08] #cyclo
	# yticks_ = [0.08, 0.10, 0.12] #raw
	# ax[4][2].set_yticks(yticks_)
	# ax[4][2].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	
	#Visibility
	count = 0
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		if count < 12:
			for tk in ax_it.get_xticklabels():
				tk.set_visible(False)
		ax_it.yaxis.offsetText.set_visible(True)
		count +=1
	# plt.setp(ax[1][1].get_xticklabels(), visible=False)
	
	
	if style['n_data'] != 1:
		if style['n_data'] == 4:
			# ax[2][0].legend(fontsize=font_label, loc=(-1.71,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
			ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.7,1.35), handletextpad=0.3, labelspacing=.3)
		
		if style['n_data'] == 2:
			# ax[2][0].legend(fontsize=font_label, loc=(-1.71,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
			ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.3,1.43), handletextpad=0.3, labelspacing=.3)
			# 1.35
		elif style['n_data'] == 3:
			# if style['legend']['first'][0].find('db') != -1:
			print('++++++++++ legend')
			
			if style['legend'][0].find('db') != -1:
				ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.4,1.35), handletextpad=0.3, labelspacing=.3)
			else:
				ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.45,1.35), handletextpad=0.3, labelspacing=.3)
			
			# else:
				# ax[2][0].legend(fontsize=font_label, loc=(-1.5,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
		
	
	
	#Set Limits in Axis X
	
	

	# plt.rcParams['axes.xmargin'] = 0
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	

	# ax[1][1].set_aspect('equal')
	
	# plt.tight_layout()
	if style['output'] == 'plot':
		
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return



def plot12_thesis_new(data, style):
	#Modules and global properties
	from matplotlib import font_manager
	
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	
	# params = {'mathtext.default': 'regular' }          
	# plt.rcParams.update(params)
	# plt.rc_context({'axes.autolimit_mode': 'round_numbers'})
	
	fig, ax = plt.subplots(ncols=3, nrows=4, sharey=False, sharex=True)
	lim = 2
	
	from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
	
	
	class FixedOrderFormatter(ScalarFormatter):
		def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
			self._order_of_mag = order_of_mag
			ScalarFormatter.__init__(self, useOffset=useOffset, useMathText=useMathText)
		def _set_orderOfMagnitude(self, range):
			"""Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
			self.orderOfMagnitude = self._order_of_mag
	
	
	font_big = 17+3
	font_little = 15+3
	font_label = 13+3
	font_offset = 15+3
	font_autolabel = 15+3
	font_caption = 23+3
	if style['n_data'] == 1:
		plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.050, top=0.96, hspace=0.5)
		fig.set_size_inches(14.2, 9)
	else:
		plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.080, top=0.90, hspace=0.42)
		fig.set_size_inches(14.2, 9.6)

	# fig.text(0.053-0.015, 0.04, '(d)', fontsize=font_caption)
	# fig.text(0.385-0.015, 0.04, '(e)', fontsize=font_caption)
	# fig.text(0.717-0.015, 0.04, '(f)', fontsize=font_caption)
	from matplotlib.ticker import ScalarFormatter

	class ScalarFormatterForceFormat(ScalarFormatter):
		def _set_format(self,vmin,vmax):  # Override function that finds format to use.
			self.format = "%1.2f"  # Give format here
	
	class ScalarFormatterForceFormat1(ScalarFormatter):
		def _set_format(self,vmin,vmax):  # Override function that finds format to use.
			self.format = "%1.1f"  # Give format here
	class ScalarFormatterForceFormat2(ScalarFormatter):
		def _set_format(self,vmin,vmax):  # Override function that finds format to use.
			self.format = "%1.2f"  # Give format here
	class ScalarFormatterForceFormat3(ScalarFormatter):
		def _set_format(self,vmin,vmax):  # Override function that finds format to use.
			self.format = "%1.3f"  # Give format here
	class ScalarFormatterForceFormat0(ScalarFormatter):
		def _set_format(self,vmin,vmax):  # Override function that finds format to use.
			self.format = "%1.0f"  # Give format here
			
	# fig.text(0.053-0.015, 0.528, '(a)', fontsize=font_caption)
	# fig.text(0.385-0.015, 0.528, '(b)', fontsize=font_caption)
	# fig.text(0.717-0.015, 0.528, '(c)', fontsize=font_caption)

	marker = ['o', 's', 'v', '^']
	count = 0
	for i in range(4):
		for j in range(3):
			if style['n_data'] == 1:
				# yerror_ = np.ones(len(data['x'][count]))
				# ax[i][j].bar(data['x'][count], data['y'][count], label=style['legend'][count], yerr=yerror_, capsize=10)
				ax[i][j].bar(data['x'][count], data['y'][count], label=style['legend'][count])
			else:
				for k in range(style['n_data']):
					ax[i][j].plot(data['x'][count], data['y'][k][count], label=style['legend'][k], marker=marker[k])
				
			if i == 3:
				if style['xlabel'] != None:
					ax[i][j].set_xlabel(style['xlabel'], fontsize=font_big)
			if style['ylabel'][count] != None:
				ax[i][j].set_ylabel(style['ylabel'][count], fontsize=font_big)
			ax[i][j].tick_params(axis='both', labelsize=font_little)
			# ax[i][j].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
			if style['n_data'] == 1:
				if style['legend'][count] != None:
					if style['legend_line'] == 'OFF':
						ax[i][j].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
					else:
						ax[i][j].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
			if style['title'][count] != None:
				ax[i][j].set_title(style['title'][count], fontsize=font_big)
			ax[i][j].yaxis.offsetText.set_fontsize(font_offset)
			if i == 1:
				if style['customxlabels'] != None:
					ax[i][j].set_xticks(data['x'][count])
		
					ax[i][j].set_xticklabels(style['customxlabels']) 
					
					
					# ax[i][j].set_xticklabels(style['customxlabels'])
					# ax[i][j].set_xticks(style['customxlabels'])
			if style['ylim'] != None:
				ax[i][j].set_ylim(bottom=style['ylim'][count][0], top=style['ylim'][count][1])		
			if style['xlim'] != None:
				ax[i][j].set_xlim(left=style['xlim'][0], right=style['xlim'][1])
			ax[i][j].grid(axis='both')
			
			# ax[i][j].legend()
			count += 1
			
			# yfmt = ScalarFormatterForceFormat()
			# yfmt.set_powerlimits((-lim,lim))
			# ax[i][j].yaxis.set_major_formatter(yfmt)   
	
	#MAX
	lim = 2
	ax[0][0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat1()
	yfmt.set_powerlimits((-lim,lim))
	ax[0][0].yaxis.set_major_formatter(yfmt)
	# yticks_ = [0, 600 , 1100] #kurt
	# yticks_ = [0, 20 , 40] #b4c
	# yticks_ = [10, 14 , 18] #s1
	yticks_ = [0, 9 , 18] #s1pai
	yticks_ = [0.8, 1.4, 2.0] #s1pai
	ax[0][0].set_yticks(yticks_)
	ax[0][0].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#RMS
	lim = 4
	ax[0][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat1()
	yfmt.set_powerlimits((-lim,lim))
	ax[0][1].yaxis.set_major_formatter(yfmt)
	# yticks_ = [0, 50, 100] #kurt
	# yticks_ = [0.2, 0.4, 0.6] #b4c
	# yticks_ = [0.1, 0.2, 0.3] #s1
	yticks_ = [0.1, 0.2, 0.3] #s1pai
	ax[0][1].set_yticks(yticks_)
	ax[0][1].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#CRF
	lim = 4
	ax[0][2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat0()
	yfmt.set_powerlimits((-lim,lim))
	ax[0][2].yaxis.set_major_formatter(yfmt)
	# yticks_ = [40, 60, 80] #b4c
	yticks_ = [20, 50, 80] #s1

	ax[0][2].set_yticks(yticks_)
	ax[0][2].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#STD
	lim = 4
	ax[1][0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat1()
	yfmt.set_powerlimits((-lim,lim))
	ax[1][0].yaxis.set_major_formatter(yfmt)
	# yticks_ = [0, 50, 100] #kurto
	# yticks_ = [0.2, 0.4, 0.6] #b4c
	# yticks_ = [0.1, 0.2, 0.3] #s1
	yticks_ = [0.1, 0.2, 0.3] #s1pai
	ax[1][0].set_yticks(yticks_)
	ax[1][0].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#AVS
	lim = 2
	ax[1][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat1()
	yfmt.set_powerlimits((-lim,lim))
	ax[1][1].yaxis.set_major_formatter(yfmt)
	# yticks_ = [0.0, 2.e-4, 4.e-4] #b4c
	# yticks_ = [5.e-5, 7.5e-5, 1.e-4] #s1
	yticks_ = [0.e-5, 5.e-5, 1.e-4] #s1pai
	ax[1][1].set_yticks(yticks_)
	ax[1][1].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#KUR
	lim = 4
	ax[1][2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat0()
	yfmt.set_powerlimits((-lim,lim))
	ax[1][2].yaxis.set_major_formatter(yfmt)
	# yticks_ = [0, 15, 30] #kurt
	# yticks_ = [0, 70, 140] #b4c
	yticks_ = [0, 20, 40] #s1

	ax[1][2].set_yticks(yticks_)
	ax[1][2].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#CEF
	lim = 4
	ax[2][0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat0()
	yfmt.set_powerlimits((-lim,lim))
	ax[2][0].yaxis.set_major_formatter(yfmt)
	# yticks_ = [100, 125, 150] #kurt
	# yticks_ = [160, 200, 240] #b4c
	# yticks_ = [160, 190, 220] #s1
	yticks_ = [150, 185, 220] #s1pai

	ax[2][0].set_yticks(yticks_)
	ax[2][0].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#SEN
	lim = 4
	ax[2][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat1()
	yfmt.set_powerlimits((-lim,lim))
	ax[2][1].yaxis.set_major_formatter(yfmt)
	# yticks_ = [19, 20.5, 22] #b4c
	# yticks_ = [21, 21.5, 22] #s1
	yticks_ = [21.5, 22.5, 23.5] #s1pai
	ax[2][1].set_yticks(yticks_)
	ax[2][1].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#p17
	lim = 4
	ax[2][2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat1()
	yfmt.set_powerlimits((-lim,lim))
	ax[2][2].yaxis.set_major_formatter(yfmt)
	# yticks_ = [0.0, 1.5, 3.0] #kurt
	# yticks_ = [0.7, 1.4, 2.1] #b4c
	# yticks_ = [0.8, 1, 1.2] #s1
	yticks_ = [0.4, 0.8, 1.2] #s1pai
	ax[2][2].set_yticks(yticks_)
	ax[2][2].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#p21
	lim = 2
	ax[3][0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat1()
	yfmt.set_powerlimits((-lim,lim))
	ax[3][0].yaxis.set_major_formatter(yfmt)
	# yticks_ = [0.0, 0.015, 0.03] #kurt
	# yticks_ = [0.004, 0.006, 0.008] #b4c
	# yticks_ = [0.004, 0.005, 0.006] #s1
	yticks_ = [0.002, 0.004, 0.006] #s1pai
	ax[3][0].set_yticks(yticks_)
	ax[3][0].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#STS
	lim = 4
	ax[3][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat0()
	yfmt.set_powerlimits((-lim,lim))
	ax[3][1].yaxis.set_major_formatter(yfmt)
	# yticks_ = [4., 15, 26] #kurt
	# yticks_ = [80., 100, 120] #b4c
	# yticks_ = [100., 110, 120] #s1
	yticks_ = [80., 100, 120] #s1
	ax[3][1].set_yticks(yticks_)
	ax[3][1].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	#P24
	lim = 2
	ax[3][2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim), useOffset=True)
	yfmt = ScalarFormatterForceFormat1()
	yfmt.set_powerlimits((-lim,lim))
	ax[3][2].yaxis.set_major_formatter(yfmt)	
	# yticks_ = [0.0, 0.025, 0.05] #kurto
	# yticks_ = [0.0, 1.e-3, 2.e-3] #b4c
	# yticks_ = [5.e-4, 7.5e-4, 1.e-3] #s1
	yticks_ = [0.e-4, 5.e-4, 1.e-3] #s1pai
	ax[3][2].set_yticks(yticks_)
	ax[3][2].set_ylim(bottom=yticks_[0], top=yticks_[2])
	
	
	#Visibility
	count = 0
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		if count >= 0:
			for tk in ax_it.get_xticklabels():
				tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)
		count +=1
	# plt.setp(ax[1][1].get_xticklabels(), visible=False)
	
	
	if style['n_data'] != 1:
		if style['n_data'] == 4:
			# ax[2][0].legend(fontsize=font_label, loc=(-1.71,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
			ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.7,1.35), handletextpad=0.3, labelspacing=.3)
		
		if style['n_data'] == 2:
			# ax[2][0].legend(fontsize=font_label, loc=(-1.71,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
			ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.3,1.43), handletextpad=0.3, labelspacing=.3)
			# 1.35
		elif style['n_data'] == 3:
			# if style['legend']['first'][0].find('db') != -1:
			print('++++++++++ legend')
			
			if style['legend'][0].find('db') != -1:
				ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.4,1.35), handletextpad=0.3, labelspacing=.3)
			else:
				ax[0][2].legend(fontsize=font_label, ncol=style['n_data'], loc=(-1.45,1.35), handletextpad=0.3, labelspacing=.3)
			
			# else:
				# ax[2][0].legend(fontsize=font_label, loc=(-1.5,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
		
	
	
	#Set Limits in Axis X
	
	

	# plt.rcParams['axes.xmargin'] = 0
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	

	# ax[1][1].set_aspect('equal')
	
	# plt.tight_layout()
	if style['output'] == 'plot':
		
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return

def plot1_thesis(data, style):
	#Modules and global properties
	
	# from matplotlib import font_manager	
	# del font_manager.weight_dict['roman']
	# font_manager._rebuild()
	# plt.rcParams['font.family'] = 'Times New Roman'	
	
	from matplotlib import font_manager		
	params = {'font.family':'Times New Roman'}
	plt.rcParams.update(params)	
	params = {'mathtext.fontset': 'stix' }	
	plt.rcParams.update(params)
	
	
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	
	
	fig, ax = plt.subplots(ncols=1, nrows=1)
	
	#Values Fixed
	amp_factor = 1
	font_big = (17+2+1)*amp_factor
	font_little = (15+2+1)*amp_factor
	font_label = (13+2+1)*amp_factor
	font_offset = (15+2+1)*amp_factor
	lim = 3
	
	# left 0.15 right 094
	plt.subplots_adjust(left=0.16, right=0.95, bottom=0.165, top=0.92)
	
	
	
	ax.set_xlabel(style['xlabel'], fontsize=font_big)
	
	
	
	for i in range(len(data['x'])):
		ax.plot(data['x'][i], data['y'][i], label=style['legend'][i], color=style['color'][i])

	
	
	ax.set_ylabel(style['ylabel'], fontsize=font_big)

	
	#Size labels from axis
	ax.tick_params(axis='both', labelsize=font_little)
		
	#Scientific notation	
	ax.ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	# #Visibility
	# for ax_it in ax.flatten():
		# for tk in ax_it.get_yticklabels():
			# tk.set_visible(True)
		# for tk in ax_it.get_xticklabels():
			# tk.set_visible(True)
		# ax_it.yaxis.offsetText.set_visible(True)

	#Eliminate line from label
	if style['legend'] != None:
		if style['legend_line'] == 'OFF':
			ax.legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
		else:
			ax.legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
	
	#Title
	if style['title'] != None:
		# plt.rcParams['mathtext.fontset'] = 'cm'
		# ax.set_title(keys[0], fontsize=font_big)	
		ax.set_title(style['title'], fontsize=font_big-2)
	#Size from offset text
	ax.yaxis.offsetText.set_fontsize(font_offset)
	

	
	#Set Ticks in Axis X
	if style['customxlabels'] == 'ON':
		# ax.set_xticks(data['dom']) 		
		ax.set_xticklabels(style['xticklabels']) 
		
	
	
	#Set Limits in Axis X
	if style['ylim'] != None:
		ax.set_ylim(bottom=style['ylim'][0], top=style['ylim'][1])
	
	if style['xlim'] != None:
		ax.set_xlim(left=style['xlim'][0], right=style['xlim'][1])
		
	#Set Vertical Lines
	if style['vlines'] != None:
		ax.vlines(style['vlines'], ymax=style['range_lines'][1], ymin=style['range_lines'][0], linestyles='dashed')
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		pos_v = 57.5 #for temp
		# pos_v = 157.5 #for temp
		pos_v = 78 #for temp
		pos_v = 0.34 #for p21 old
		# pos_v = 240 #for temp
		# pos_v = 390 #for temp
		# pos_v = 134. #for fix
		# pos_v = 0.24 #for fix
		# pos_v = 0.235 #for rel
		rest_x = 0.35
		ax.annotate(s='End MC 2', xy=[style['vlines'][1]-rest_x,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End MC 3', xy=[style['vlines'][2]-rest_x,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End MC 4', xy=[style['vlines'][3]-rest_x,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End MC 5', xy=[style['vlines'][4]-rest_x,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End MC 6', xy=[style['vlines'][5]-rest_x,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End MC 7', xy=[style['vlines'][6]-rest_x,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End MC 8', xy=[style['vlines'][7]-rest_x,pos_v], rotation=90, fontsize=font_label-3)
		ax.annotate(s='End MC 9', xy=[style['vlines'][8]-rest_x,pos_v], rotation=90, fontsize=font_label-3)
		# ax.annotate(s='End $9^{th}$ MC 9', xy=[style['vlines'][8]-0.5,pos_v], rotation=90, fontsize=font_label-3)
		
	
	ax.grid(axis='both')
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	
	return

def plot2_thesis_new(data, style):
	print('!!!!!!!!!!!')
	#Modules and global properties
	from matplotlib import font_manager
	
	params = {'font.family':'Times New Roman'}
	plt.rcParams.update(params)	
	params = {'mathtext.fontset': 'stix' }	
	plt.rcParams.update(params)
	
	
	# del font_manager.weight_dict['roman']
	# font_manager._rebuild()
	# plt.rcParams['font.family'] = 'Times New Roman'	
	
	fig, ax = plt.subplots(ncols=2, nrows=1, sharex=False)
	
	#Values Fixed
	amp_factor = 1
	font_big = (17+2+1)*amp_factor+1
	font_little = (15+2+1)*amp_factor+1
	font_label = (13+2+1)*amp_factor+1
	font_offset = (15+2+1)*amp_factor+1
	lim = 3
	font_caption = font_big

	
	# fig.set_size_inches(12, 4.5)
	# plt.subplots_adjust(left=0.075, right=0.95, bottom=0.165, top=0.9, wspace=0.225)
	
	fig.set_size_inches(11.5, 4.)
	plt.subplots_adjust(left=0.085, right=0.96, bottom=0.21, top=0.9, wspace=0.24)
	fig.text(0.059, 0.022, '(a)', fontsize=font_caption+4)
	fig.text(0.547, 0.022, '(b)', fontsize=font_caption+4)
	
	
	
	for i in range(len(data['x'])):
		# ax[i].plot(data['x'][i], data['y'][i], label=style['legend'][i], color=style['color'][i])
		ax[i].plot(data['x'][i], data['y'][i])

	
	
		ax[i].set_ylabel(style['ylabel'][i], fontsize=font_big)
		ax[i].set_xlabel(style['xlabel'][i], fontsize=font_big)
	
	#Size labels from axis
		ax[i].tick_params(axis='both', labelsize=font_little)
		
	#Scientific notation	
		ax[i].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	# #Visibility
	# for ax_it in ax.flatten():
		# for tk in ax_it.get_yticklabels():
			# tk.set_visible(True)
		# for tk in ax_it.get_xticklabels():
			# tk.set_visible(True)
		# ax_it.yaxis.offsetText.set_visible(True)

		#Eliminate line from label
		if style['legend'] != None:
			if style['legend_line'] == 'OFF':
				ax.legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
			else:
				ax.legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
		
		#Title
		if style['title'] != None:
			# plt.rcParams['mathtext.fontset'] = 'cm'
			# ax.set_title(keys[0], fontsize=font_big)	
			ax[i].set_title(style['title'][i], fontsize=font_big)
		#Size from offset text
		ax[i].yaxis.offsetText.set_fontsize(font_offset)
		

		
		#Set Ticks in Axis X
		if style['customxlabels'] == 'ON':
			# ax.set_xticks(data['dom']) 		
			ax.set_xticklabels(style['xticklabels']) 
			
		
		
		#Set Limits in Axis X
		if style['ylim'] != None:
			ax[i].set_ylim(bottom=style['ylim'][i][0], top=style['ylim'][i][1])
		
		if style['xlim'] != None:
			ax[i].set_xlim(left=style['xlim'][i][0], right=style['xlim'][i][1])
		

		
	
		ax[i].grid(axis='both')
	
	# #Ticks
	# ax[0].set_yticks([-3, -1.5, 0, 1.5, 3]) 
	
	# #TicksLabels
	# ax[0].set_xticklabels([0, 0.05, 0.1, 0.15, 0.2, 0.25])
	# ax[1].set_xticklabels([0, 0.05, 0.1, 0.15, 0.2, 0.25]) 
	
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	
	return

def plot6_thesis_new_emd(data, style):
	print('!!!!!!!!!!!')
	#Modules and global properties
	from matplotlib import font_manager	
	params = {'font.family':'Times New Roman'}
	plt.rcParams.update(params)	
	params = {'mathtext.fontset': 'stix' }	
	plt.rcParams.update(params)
	
	
	
	
	
	# from matplotlib import font_manager	
	# del font_manager.weight_dict['roman']
	# font_manager._rebuild()
	# plt.rcParams['font.family'] = 'Times New Roman'	
	
	
	fig, ax = plt.subplots(ncols=2, nrows=3, sharex=True, sharey=False)
	
	#Values Fixed
	amp_factor = 1
	font_big = (17+2+1)*amp_factor+1-2
	font_little = (15+2+1)*amp_factor+1
	font_label = (13+2+1)*amp_factor+1
	font_offset = (15+2+1)*amp_factor+1
	lim = 3
	font_caption = font_big

	
	# fig.set_size_inches(12, 4.5)
	# plt.subplots_adjust(left=0.075, right=0.95, bottom=0.165, top=0.9, wspace=0.225)
	
	fig.set_size_inches(12., 6.5)
	plt.subplots_adjust(hspace=0.5, left=0.1, right=0.96, bottom=0.13, top=0.93, wspace=0.28)
	px = 0.033
	fig.text(0.073-px, 0.025, '(e)', fontsize=font_caption+4)
	fig.text(0.56-px, 0.025, '(f)', fontsize=font_caption+4)
	
	fig.text(0.073-px, 0.025+0.34, '(c)', fontsize=font_caption+4)
	fig.text(0.56-px, 0.025+0.34, '(d)', fontsize=font_caption+4)
	
	fig.text(0.073-px, 0.025+0.33+0.31, '(a)', fontsize=font_caption+4)
	fig.text(0.56-px, 0.025+0.33+0.31, '(b)', fontsize=font_caption+4)
	
	count = 0
	for j in range(2):
		for i in range(3):
			# ax[i].plot(data['x'][i], data['y'][i], label=style['legend'][i], color=style['color'][i])
			ax[i][j].plot(data['x'][count], data['y'][count])

		
		
			ax[i][j].set_ylabel(style['ylabel'], fontsize=font_big)
			
			if i == 2:
				ax[i][j].set_xlabel(style['xlabel'], fontsize=font_big)
		
		#Size labels from axis
			ax[i][j].tick_params(axis='both', labelsize=font_little)
			
		#Scientific notation	
			ax[i][j].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
		
		# #Visibility
		# for ax_it in ax.flatten():
			# for tk in ax_it.get_yticklabels():
				# tk.set_visible(True)
			# for tk in ax_it.get_xticklabels():
				# tk.set_visible(True)
			# ax_it.yaxis.offsetText.set_visible(True)

			#Eliminate line from label
			if style['legend'] != None:
				if style['legend_line'] == 'OFF':
					ax.legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
				else:
					ax.legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
			
			#Title
			if style['title'] != None:
				# plt.rcParams['mathtext.fontset'] = 'cm'
				# ax.set_title(keys[0], fontsize=font_big)	
				ax[i][j].set_title(style['title'][count], fontsize=font_big)
			#Size from offset text
			ax[i][j].yaxis.offsetText.set_fontsize(font_offset)
			

			
			#Set Ticks in Axis X
			if style['customxlabels'] == 'ON':
				# ax.set_xticks(data['dom']) 		
				ax.set_xticklabels(style['xticklabels']) 
				
			
			
			#Set Limits in Axis X
			if style['ylim'] != None:
				ax[i][j].set_ylim(bottom=style['ylim'][i][0], top=style['ylim'][i][1])
			
			if style['xlim'] != None:
				ax[i][j].set_xlim(left=style['xlim'][0], right=style['xlim'][1])
			

			
		
			ax[i][j].grid(axis='both')
			count += 1
		#Ticks
			ax[i][j].set_yticks([-3, -1.5, 0, 1.5, 3]) 
			# ax[i][j].set_yticks([40, 70, 100]) 
		
		#TicksLabels
		# ax[0].set_xticklabels([0, 0.05, 0.1, 0.15, 0.2, 0.25])
		# ax[1].set_xticklabels([0, 0.05, 0.1, 0.15, 0.2, 0.25]) 
	
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	
	return

def plot1_scalo_thesis(map, style):
	#Modules and global properties
	from matplotlib import font_manager
	
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	fig, ax = plt.subplots(ncols=1, nrows=1)
	
	#Values Fixed
	amp_factor = 1
	font_big = (17+2+1)*amp_factor
	font_little = (15+2+1)*amp_factor
	font_label = (13+2+1)*amp_factor
	font_offset = (15+2+1)*amp_factor
	lim = 3

	plt.subplots_adjust(left=0.15, right=0.93, bottom=0.165, top=0.92)
	
	
	
	ax.set_xlabel(style['xlabel'], fontsize=font_big)
	

	extent_ = style['extent']
	contour_ = ax.contourf(map, extent=extent_, cmap=style['colormap'])
	
	
	cbar = fig.colorbar(contour_, ax=ax)
	
	# cbar.set_label('SCD [V$^{2}$]', fontsize=font_big)
	cbar.set_label(r'$S^{\alpha}_{x}(f)$', fontsize=font_big)
	
	# cbar.set_label('SC [-]', fontsize=font_big)
	# cbar.set_label('Energy [V$^{2}$]', fontsize=font_big)
	# cbar.set_label('Magnitude [V]', fontsize=font_big)
	# cbar.set_ticks([0, 200, 400, 600, 800])
	# cbar.set_ticks([0, 1200, 2400, 3600, 4800])
	cbar.ax.tick_params(labelsize=font_little) 
	
	cbar.ax.yaxis.get_offset_text().set(size=font_little)
	
	
	ax.set_ylabel(style['ylabel'], fontsize=font_big)

	
	#Size labels from axis
	ax.tick_params(axis='both', labelsize=font_little)
		
	#Scientific notation	
	ax.ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	# #Visibility
	# for ax_it in ax.flatten():
		# for tk in ax_it.get_yticklabels():
			# tk.set_visible(True)
		# for tk in ax_it.get_xticklabels():
			# tk.set_visible(True)
		# ax_it.yaxis.offsetText.set_visible(True)

	#Eliminate line from label
	if style['legend'] != None:
		if style['legend_line'] == 'OFF':
			ax.legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
		else:
			ax.legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
	
	#Title
	if style['title'] != None:
		# plt.rcParams['mathtext.fontset'] = 'cm'
		# ax.set_title(keys[0], fontsize=font_big)	
		ax.set_title(style['title'], fontsize=font_big-2)
	#Size from offset text
	ax.yaxis.offsetText.set_fontsize(font_offset)
	

	
	
		
	
	
	#Set Limits in Axis X
	if style['ylim'] != None:
		ax.set_ylim(bottom=style['ylim'][0], top=style['ylim'][1])
	
	if style['xlim'] != None:
		ax.set_xlim(left=style['xlim'][0], right=style['xlim'][1])
		
	#Set Vertical Lines

	#Set Ticks in Axis X
	if style['xticklabels'] != None:
		print('+++++++++++++++')
		ax.set_xticks(style['xticklabels']) 		
		ax.set_xticklabels(style['xticklabels']) 
	
	ax.grid(axis='both')
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return

def plot3h_thesis_multi(three_signals, style):
	#Modules and global properties
	from matplotlib import font_manager
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	fig, ax = plt.subplots(ncols=3, nrows=1, sharex=style['sharex'], sharey=style['sharey'])
	
	# #Values Fixed
	# font_big = 17
	# font_little = 15
	# font_label = 13
	# font_offset = 15
	# lim = 3
	# plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.175, top=0.92)
	# fig.set_size_inches(14.2, 4.0)
	
	#Values Fixed
	font_big = 17
	font_little = 15
	font_label = 13
	font_offset = 15
	lim = 3
	plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.175, top=0.92)
	fig.set_size_inches(2.36, 0.665)
	

	
	
	#Axis X
	if style['dom'] == 'time':
		fact = 1.
		ax[0].set_xlabel('Time [s]', fontsize=font_big)
		ax[1].set_xlabel('Time [s]', fontsize=font_big)
		ax[2].set_xlabel('Time [s]', fontsize=font_big)
	elif style['dom'] == 'frequency':
		if style['kHz'] == 'ON':
			fact = 1000.
			ax[0].set_xlabel('Frequency [kHz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [kHz]', fontsize=font_big)
			ax[2].set_xlabel('Frequency [kHz]', fontsize=font_big)
		else:
			fact = 1.
			ax[0].set_xlabel('Frequency [Hz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [Hz]', fontsize=font_big)
			ax[2].set_xlabel('Frequency [Hz]', fontsize=font_big)
	elif style['dom'] == 'other':
		fact = 1.
		ax[0].set_xlabel(style['xtitle'][0], fontsize=font_big)
		ax[1].set_xlabel(style['xtitle'][1], fontsize=font_big)
		ax[2].set_xlabel(style['xtitle'][2], fontsize=font_big)
	
	
	#Plot	
	keys = list(three_signals)
	markers = ['s', 'o', '^', 'v']
	print(len(three_signals[keys[0]]))
	print(style['legend']['first'])
	if style['type'] == 'plot':
		for i, label, marker in zip(range(len(three_signals[keys[0]])), style['legend']['first'], markers):
			ax[0].plot(three_signals['dom']/fact, three_signals[keys[0]][i], label=label, marker=marker)
		for i, label, marker in zip(range(len(three_signals[keys[1]])), style['legend']['second'], markers):
			ax[1].plot(three_signals['dom']/fact, three_signals[keys[1]][i], label=label, marker=marker)
		for i, label, marker in zip(range(len(three_signals[keys[2]])), style['legend']['third'], markers):
			ax[2].plot(three_signals['dom']/fact, three_signals[keys[2]][i], label=label, marker=marker)
		# for i in range(len(three_signals[keys[0]])):
			# ax[0].plot(three_signals['dom']/fact, three_signals[keys[0]][i], label='caca')
		# for i in range(len(three_signals[keys[1]])):
			# ax[1].plot(three_signals['dom']/fact, three_signals[keys[1]][i], label=keys[1])
		# for i in range(len(three_signals[keys[2]])):
			# ax[2].plot(three_signals['dom']/fact, three_signals[keys[2]][i], label=keys[2])
	elif style['type'] == 'bar':
		bar0 = ax[0].bar(three_signals['dom']/fact, three_signals[keys[0]], label=keys[0])
		bar1 = ax[1].bar(three_signals['dom']/fact, three_signals[keys[1]], label=keys[1])
		bar2 = ax[2].bar(three_signals['dom']/fact, three_signals[keys[2]], label=keys[2])
	
	#Axis Y
	if style['dom'] == 'time':
		ax[0].set_ylabel('Amplitude [mV]', fontsize=font_big)
		ax[1].set_ylabel('Amplitude [mV]', fontsize=font_big)
		ax[2].set_ylabel('Amplitude [mV]', fontsize=font_big)		
		
	elif style['dom'] == 'frequency':
		ax[0].set_ylabel('Magnitude [mV]', fontsize=font_big)
		ax[1].set_ylabel('Magnitude [mV]', fontsize=font_big)
		ax[2].set_ylabel('Magnitude [mV]', fontsize=font_big)
	
	elif style['dom'] == 'other':
		ax[0].set_ylabel(style['ytitle'][0], fontsize=font_big)
		ax[1].set_ylabel(style['ytitle'][1], fontsize=font_big)
		ax[2].set_ylabel(style['ytitle'][2], fontsize=font_big)
		
		# params = {'mathtext.default': 'regular' }          
		# plt.rcParams.update(params)
		# ax[0].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[1].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[2].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)	
		
		# ax[0].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[1].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[2].set_title('Envelope spectrum', fontsize=font_offset)
	
	#Size labels from axis
	ax[0].tick_params(axis='both', labelsize=font_little)
	ax[1].tick_params(axis='both', labelsize=font_little)	
	ax[2].tick_params(axis='both', labelsize=font_little)	
		
	#Scientific notation	
	ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)

	#Eliminate line from label
	if style['legend'] != False:
		loc = 'best'
		ax[0].legend(fontsize=font_label, loc = loc)	
		ax[1].legend(fontsize=font_label, loc = loc)
		ax[2].legend(fontsize=font_label, loc = loc)
		# , loc = 'upper left'
		# ax[0].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)	
		# ax[1].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
		# ax[2].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
	
	#Title
	if style['title'] == True:
		ax[0].set_title(keys[0], fontsize=font_big)	
		ax[1].set_title(keys[1], fontsize=font_big)
		ax[2].set_title(keys[2], fontsize=font_big)
		# ax[0].title.set_visible(False)
		# ax[1].title.set_visible(False)
		# ax[2].title.set_visible(False)
	
	#Size from offset text
	ax[0].yaxis.offsetText.set_fontsize(font_offset)
	ax[1].yaxis.offsetText.set_fontsize(font_offset)
	ax[2].yaxis.offsetText.set_fontsize(font_offset)
	
	
	#Set Ticks in Axis Y
	# ax[0].set_yticks([0, 3, 6, 9, 12, 15]) 
	# ax[1].set_yticks([0, 3, 6, 9, 12, 15]) 
	# ax[2].set_yticks([0, 3, 6, 9, 12, 15])
	
	#Set Ticks in Axis X
	if style['dom'] == 'other':
		ax[0].set_xticks(three_signals['dom']) 
		ax[1].set_xticks(three_signals['dom']) 
		ax[2].set_xticks(three_signals['dom'])
		
		ax[0].set_xticklabels(style['xticklabels']) 
		ax[1].set_xticklabels(style['xticklabels']) 
		ax[2].set_xticklabels(style['xticklabels'])
		
		
	
	#Set Limits in Axis X
	if style['ymax'] != None:
		ax[0].set_ylim(bottom=style['ymin'][0], top=style['ymax'][0])
		ax[1].set_ylim(bottom=style['ymin'][1], top=style['ymax'][1])
		ax[2].set_ylim(bottom=style['ymin'][2], top=style['ymax'][2])
	
	# ax[2].set_xlim(left=1.01, right=1.06)
	# ax[1].set_xlim(left=1.29, right=1.34)
	# ax[0].set_xlim(left=1.58, right=1.63)
	if style['autolabel'] == True:
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				# height2 = height*100.
				ax[0].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=15)
		autolabel(bar0)
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				ax[1].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=15)
		autolabel(bar1)
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				ax[2].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=15)
		autolabel(bar2)
	
	# plt.tight_layout()
	plt.show()
	return

def plot6_thesis(three_signals, style):
	
	# width = 1
	# hrange = (0/norm, 15/norm)
	# nbins = 15
	# font_big = 17
	# font_little = 15
	# font_label = 13
	
	# from matplotlib import font_manager
	# del font_manager.weight_dict['roman']
	# font_manager._rebuild()
	# plt.rcParams['font.family'] = 'Times New Roman'	
	
	# fig, ax = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True)
	# plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.125, top=0.92, hspace=0.75)
	# fig.set_size_inches(14.2, 7.2)		
	# ax[0][0].plot(distr['c10_w005_p2'])
	# ax[0][1].plot(distr['c10_w01_p2'])
	# ax[0][2].plot(distr['c10_w015_p2'])
	
	# ax[1][0].plot(distr['c10_w005_p4'])
	# ax[1][1].plot(distr['c10_w01_p4'])
	# ax[1][2].plot(distr['c10_w015_p4'])
	
	# ax[2][0].plot(distr['c10_w005_p7'])
	# ax[2][1].plot(distr['c10_w01_p7'])
	# ax[2][2].plot(distr['c10_w015_p7'])	

	# ax[0][0].set_ylabel('Occurrence', fontsize=font_big)
	# ax[0][1].set_ylabel('Occurrence', fontsize=font_big)
	# ax[0][2].set_ylabel('Occurrence', fontsize=font_big)
	
	# ax[1][0].set_ylabel('Occurrence', fontsize=font_big)
	# ax[1][1].set_ylabel('Occurrence', fontsize=font_big)
	# ax[1][2].set_ylabel('Occurrence', fontsize=font_big)
	
	# ax[2][0].set_ylabel('Occurrence', fontsize=font_big)
	# ax[2][1].set_ylabel('Occurrence', fontsize=font_big)
	# ax[2][2].set_ylabel('Occurrence', fontsize=font_big)
	
	# ax[0][0].set_xlabel('Monotonicity', fontsize=font_big)
	# ax[0][1].set_xlabel('Monotonicity', fontsize=font_big)
	# ax[0][2].set_xlabel('Monotonicity', fontsize=font_big)
	
	# ax[1][0].set_xlabel('Monotonicity', fontsize=font_big)
	# ax[1][1].set_xlabel('Monotonicity', fontsize=font_big)
	# ax[1][2].set_xlabel('Monotonicity', fontsize=font_big)
	
	# ax[2][0].set_xlabel('Monotonicity', fontsize=font_big)
	# ax[2][1].set_xlabel('Monotonicity', fontsize=font_big)
	# ax[2][2].set_xlabel('Monotonicity', fontsize=font_big)
	
	# plt.rcParams['mathtext.fontset'] = 'cm'
	# ax[0][0].set_title('$C$=10, $W$=0.05, $P$=2', fontsize=font_big)
	# ax[0][1].set_title('$C$=10, $W$=0.1, $P$=2', fontsize=font_big)
	# ax[0][2].set_title('$C$=10, $W$=0.15, $P$=2', fontsize=font_big)
	
	# ax[1][0].set_title('$C$=10, $W$=0.05, $P$=4', fontsize=font_big)
	# ax[1][1].set_title('$C$=10, $W$=0.1, $P$=4', fontsize=font_big)
	# ax[1][2].set_title('$C$=10, $W$=0.15, $P$=4', fontsize=font_big)
	
	# ax[2][0].set_title('$C$=10, $W$=0.05, $P$=7', fontsize=font_big)
	# ax[2][1].set_title('$C$=10, $W$=0.1, $P$=7', fontsize=font_big)
	# ax[2][2].set_title('$C$=10, $W$=0.15, $P$=7', fontsize=font_big)	
	
	# ax[0][0].set_xlim(left=0, right=1)
	# ax[0][1].set_xlim(left=0, right=1)
	# ax[0][2].set_xlim(left=0, right=1)
	
	# ax[1][0].set_xlim(left=0, right=1)
	# ax[1][1].set_xlim(left=0, right=1)
	# ax[1][2].set_xlim(left=0, right=1)
	
	
	# ax[2][0].set_xlim(left=0, right=1)
	# ax[2][1].set_xlim(left=0, right=1)
	# ax[2][2].set_xlim(left=0, right=1)
	
	# valtop = 9
	# ax[0][0].set_ylim(bottom=0, top=valtop)
	# ax[0][1].set_ylim(bottom=0, top=valtop)
	# ax[0][2].set_ylim(bottom=0, top=valtop)
	
	# ax[1][0].set_ylim(bottom=0, top=valtop)
	# ax[1][1].set_ylim(bottom=0, top=valtop)
	# ax[1][2].set_ylim(bottom=0, top=valtop)
	
	# ax[2][0].set_ylim(bottom=0, top=valtop)
	# ax[2][1].set_ylim(bottom=0, top=valtop)
	# ax[2][2].set_ylim(bottom=0, top=valtop)
	
	
	# ax[0][0].set_yticks([0, 3, 6, 9])
	# ax[0][1].set_yticks([0, 3, 6, 9])
	# ax[0][2].set_yticks([0, 3, 6, 9])
	
	# ax[1][0].set_yticks([0, 3, 6, 9])
	# ax[1][1].set_yticks([0, 3, 6, 9])
	# ax[1][2].set_yticks([0, 3, 6, 9])
	
	# ax[2][0].set_yticks([0, 3, 6, 9])
	# ax[2][1].set_yticks([0, 3, 6, 9])
	# ax[2][2].set_yticks([0, 3, 6, 9])
	
	
	
	# ax[0][0].tick_params(axis='both', labelsize=font_little)
	# ax[0][1].tick_params(axis='both', labelsize=font_little)
	# ax[0][2].tick_params(axis='both', labelsize=font_little)
	
	# ax[1][0].tick_params(axis='both', labelsize=font_little)
	# ax[1][1].tick_params(axis='both', labelsize=font_little)
	# ax[1][2].tick_params(axis='both', labelsize=font_little)
	
	# ax[2][0].tick_params(axis='both', labelsize=font_little)
	# ax[2][1].tick_params(axis='both', labelsize=font_little)
	# ax[2][2].tick_params(axis='both', labelsize=font_little)
	
	# for ax_it in ax.flatten():
		# for tk in ax_it.get_yticklabels():
			# tk.set_visible(True)
		# for tk in ax_it.get_xticklabels():
			# tk.set_visible(True)
		# ax_it.yaxis.offsetText.set_visible(True)
	
	# plt.show()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	#Modules and global properties
	from matplotlib import font_manager
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	fig, ax = plt.subplots(ncols=3, nrows=1, sharex=style['sharex'], sharey=style['sharey'])
	
	#Values Fixed
	font_big = 17
	font_little = 15
	font_label = 13
	font_offset = 15
	lim = 2
	plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.175, top=0.92)
	fig.set_size_inches(14.2, 4.0)
	
	
	
	#Axis X
	if style['dom'] == 'time':
		fact = 1.
		ax[0].set_xlabel('Time [s]', fontsize=font_big)
		ax[1].set_xlabel('Time [s]', fontsize=font_big)
		ax[2].set_xlabel('Time [s]', fontsize=font_big)
	elif style['dom'] == 'frequency':
		if style['kHz'] == 'ON':
			fact = 1000.
			ax[0].set_xlabel('Frequency [kHz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [kHz]', fontsize=font_big)
			ax[2].set_xlabel('Frequency [kHz]', fontsize=font_big)
		else:
			fact = 1.
			ax[0].set_xlabel('Frequency [Hz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [Hz]', fontsize=font_big)
			ax[2].set_xlabel('Frequency [Hz]', fontsize=font_big)
	elif style['dom'] == 'other':
		fact = 1.
		ax[0].set_xlabel(style['xtitle'][0], fontsize=font_big)
		ax[1].set_xlabel(style['xtitle'][1], fontsize=font_big)
		ax[2].set_xlabel(style['xtitle'][2], fontsize=font_big)
	
	
	#Plot	
	keys = list(three_signals)
	markers = ['s', 'o', '^', 'v']
	if style['type'] == 'plot':
		for i, label, marker in zip(range(len(three_signals[keys[0]])), style['legend']['first'], markers):
			ax[0].plot(three_signals['dom']/fact, three_signals[keys[0]][i], label=label, marker=marker)
		for i, label, marker in zip(range(len(three_signals[keys[1]])), style['legend']['second'], markers):
			ax[1].plot(three_signals['dom']/fact, three_signals[keys[1]][i], label=label, marker=marker)
		for i, label, marker in zip(range(len(three_signals[keys[2]])), style['legend']['third'], markers):
			ax[2].plot(three_signals['dom']/fact, three_signals[keys[2]][i], label=label, marker=marker)
		# for i in range(len(three_signals[keys[0]])):
			# ax[0].plot(three_signals['dom']/fact, three_signals[keys[0]][i], label='caca')
		# for i in range(len(three_signals[keys[1]])):
			# ax[1].plot(three_signals['dom']/fact, three_signals[keys[1]][i], label=keys[1])
		# for i in range(len(three_signals[keys[2]])):
			# ax[2].plot(three_signals['dom']/fact, three_signals[keys[2]][i], label=keys[2])
	elif style['type'] == 'bar':
		bar0 = ax[0].bar(three_signals['dom']/fact, three_signals[keys[0]], label=keys[0])
		bar1 = ax[1].bar(three_signals['dom']/fact, three_signals[keys[1]], label=keys[1])
		bar2 = ax[2].bar(three_signals['dom']/fact, three_signals[keys[2]], label=keys[2])
	
	#Axis Y
	if style['dom'] == 'time':
		ax[0].set_ylabel('Amplitude [mV]', fontsize=font_big)
		ax[1].set_ylabel('Amplitude [mV]', fontsize=font_big)
		ax[2].set_ylabel('Amplitude [mV]', fontsize=font_big)		
		
	elif style['dom'] == 'frequency':
		ax[0].set_ylabel('Magnitude [mV]', fontsize=font_big)
		ax[1].set_ylabel('Magnitude [mV]', fontsize=font_big)
		ax[2].set_ylabel('Magnitude [mV]', fontsize=font_big)
	
	elif style['dom'] == 'other':
		ax[0].set_ylabel(style['ytitle'][0], fontsize=font_big)
		ax[1].set_ylabel(style['ytitle'][1], fontsize=font_big)
		ax[2].set_ylabel(style['ytitle'][2], fontsize=font_big)
		
		# params = {'mathtext.default': 'regular' }          
		# plt.rcParams.update(params)
		# ax[0].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[1].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[2].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)	
		
		# ax[0].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[1].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[2].set_title('Envelope spectrum', fontsize=font_offset)
	
	#Size labels from axis
	ax[0].tick_params(axis='both', labelsize=font_little)
	ax[1].tick_params(axis='both', labelsize=font_little)	
	ax[2].tick_params(axis='both', labelsize=font_little)	
		
	#Scientific notation	
	ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)

	#Eliminate line from label
	if style['legend'] != False:
		ax[0].legend(fontsize=font_label)	
		ax[1].legend(fontsize=font_label)
		ax[2].legend(fontsize=font_label, loc='lower right')
		# ax[0].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)	
		# ax[1].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
		# ax[2].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
	
	#Title
	if style['title'] == True:
		ax[0].set_title(keys[0], fontsize=font_big)	
		ax[1].set_title(keys[1], fontsize=font_big)
		ax[2].set_title(keys[2], fontsize=font_big)
		# ax[0].title.set_visible(False)
		# ax[1].title.set_visible(False)
		# ax[2].title.set_visible(False)
	
	#Size from offset text
	ax[0].yaxis.offsetText.set_fontsize(font_offset)
	ax[1].yaxis.offsetText.set_fontsize(font_offset)
	ax[2].yaxis.offsetText.set_fontsize(font_offset)
	
	
	#Set Ticks in Axis Y
	# ax[0].set_yticks([-20, -10, 0, 10, 20]) 
	# ax[1].set_yticks([-20, -10, 0, 10, 20]) 
	# ax[2].set_yticks([-20, -10, 0, 10, 20])
	
	#Set Ticks in Axis X
	if style['dom'] == 'other':
		ax[0].set_xticks(three_signals['dom']) 
		ax[1].set_xticks(three_signals['dom']) 
		ax[2].set_xticks(three_signals['dom'])
		
		ax[0].set_xticklabels(style['xticklabels']) 
		ax[1].set_xticklabels(style['xticklabels']) 
		ax[2].set_xticklabels(style['xticklabels'])
		
		
	
	#Set Limits in Axis X
	if style['ymax'] != None:
		ax[0].set_ylim(bottom=0, top=style['ymax'][0])
		ax[1].set_ylim(bottom=0, top=style['ymax'][1])
		ax[2].set_ylim(bottom=0, top=style['ymax'][2])
	
	# ax[2].set_xlim(left=1.01, right=1.06)
	# ax[1].set_xlim(left=1.29, right=1.34)
	# ax[0].set_xlim(left=1.58, right=1.63)
	if style['autolabel'] == True:
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				# height2 = height*100.
				ax[0].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=15)
		autolabel(bar0)
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				ax[1].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=15)
		autolabel(bar1)
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				ax[2].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=15)
		autolabel(bar2)
	
	# plt.tight_layout()
	plt.show()
	return

def plot3h_thesis_big_multi(three_signals, style):
	#Modules and global properties
	from matplotlib import font_manager
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	fig, ax = plt.subplots(ncols=3, nrows=1, sharex=style['sharex'], sharey=style['sharey'])
	
	#Values Fixed
	font_big = 17+3
	font_little = 15+3
	font_label = 14+3
	font_offset = 15+3
	font_autolabel = 15+3
	font_caption = 23+3
	lim = 3
	# plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.175, top=0.915)
	# fig.set_size_inches(14.2, 4.0)
	
	# #Values Fixed Auto
	# font_caption = (23+3)*0.4316
	# font_big = (17+3)*0.4316
	# font_little = (15+3)*0.4316
	# font_label = (13+3)*0.4316
	# font_offset = (15+3)*0.4316
	# font_autolabel = (15+3)*0.4316
	# lim = 3

	if style['caption'] == 'lower':
		# plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.26, top=0.89)
		plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.26, top=0.89)
		fig.set_size_inches(14.2, 4.0)
		fig.text(0.182, 0.02, '(a)', fontsize=font_caption)
		fig.text(0.510, 0.02, '(b)', fontsize=font_caption)
		fig.text(0.840, 0.02, '(c)', fontsize=font_caption)
	elif style['caption'] == 'lower left':
		# plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.213, top=0.89)
		plt.subplots_adjust(wspace=0.32, left=0.066, right=0.98, bottom=0.213, top=0.81)
		# fig.set_size_inches(14.2, 4.0)
		fig.set_size_inches(14.2, 4.6)

		fig.text(0.059, 0.05, '(a)', fontsize=font_caption)
		fig.text(0.387, 0.05, '(b)', fontsize=font_caption)
		fig.text(0.717, 0.05, '(c)', fontsize=font_caption)
	else:
		plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.26, top=0.89)
		fig.set_size_inches(14.2, 4.0)
	#Axis X
	if style['dom'] == 'time':
		fact = 1.
		ax[0].set_xlabel('Time [s]', fontsize=font_big)
		ax[1].set_xlabel('Time [s]', fontsize=font_big)
		ax[2].set_xlabel('Time [s]', fontsize=font_big)
	elif style['dom'] == 'frequency':
		if style['kHz'] == 'ON':
			fact = 1000.
			ax[0].set_xlabel('Frequency [kHz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [kHz]', fontsize=font_big)
			ax[2].set_xlabel('Frequency [kHz]', fontsize=font_big)
		else:
			fact = 1.
			ax[0].set_xlabel('Frequency [Hz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [Hz]', fontsize=font_big)
			ax[2].set_xlabel('Frequency [Hz]', fontsize=font_big)
	elif style['dom'] == 'other':
		fact = 1.
		ax[0].set_xlabel(style['xtitle'][0], fontsize=font_big)
		ax[1].set_xlabel(style['xtitle'][1], fontsize=font_big)
		ax[2].set_xlabel(style['xtitle'][2], fontsize=font_big)
	
	
	# #Plot	
	# keys = list(three_signals)
	# if style['type'] == 'plot':
		# ax[0].plot(three_signals['dom']/fact, three_signals[keys[0]], label=keys[0])
		# ax[1].plot(three_signals['dom']/fact, three_signals[keys[1]], label=keys[1])
		# ax[2].plot(three_signals['dom']/fact, three_signals[keys[2]], label=keys[2])
	# elif style['type'] == 'bar':
		# bar0 = ax[0].bar(three_signals['dom']/fact, three_signals[keys[0]], label=keys[0])
		# bar1 = ax[1].bar(three_signals['dom']/fact, three_signals[keys[1]], label=keys[1])
		# bar2 = ax[2].bar(three_signals['dom']/fact, three_signals[keys[2]], label=keys[2])
	keys = list(three_signals)

	markers = ['s', 'o', '^', 'v']
	if style['type'] == 'plot':
		for i, label, marker in zip(range(len(three_signals[keys[0]])), style['legend']['first'], markers):

			ax[0].plot(three_signals['dom']/fact, three_signals[keys[0]][i], label=label, marker=marker)
		for i, label, marker in zip(range(len(three_signals[keys[1]])), style['legend']['second'], markers):
			ax[1].plot(three_signals['dom']/fact, three_signals[keys[1]][i], label=label, marker=marker)
		for i, label, marker in zip(range(len(three_signals[keys[2]])), style['legend']['third'], markers):
			ax[2].plot(three_signals['dom']/fact, three_signals[keys[2]][i], label=label, marker=marker)
	#Axis Y
	if style['dom'] == 'time':
		ax[0].set_ylabel('Amplitude [mV]', fontsize=font_big)
		ax[1].set_ylabel('Amplitude [mV]', fontsize=font_big)
		ax[2].set_ylabel('Amplitude [mV]', fontsize=font_big)		
		
	elif style['dom'] == 'frequency':
		ax[0].set_ylabel('Magnitude [mV]', fontsize=font_big)
		ax[1].set_ylabel('Magnitude [mV]', fontsize=font_big)
		ax[2].set_ylabel('Magnitude [mV]', fontsize=font_big)
	
	elif style['dom'] == 'other':
		ax[0].set_ylabel(style['ytitle'][0], fontsize=font_big)
		ax[1].set_ylabel(style['ytitle'][1], fontsize=font_big)
		ax[2].set_ylabel(style['ytitle'][2], fontsize=font_big)
		
		# params = {'mathtext.default': 'regular' }          
		# plt.rcParams.update(params)
		# ax[0].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[1].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[2].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)	
		
		# ax[0].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[1].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[2].set_title('Envelope spectrum', fontsize=font_offset)
	
	#Size labels from axis
	ax[0].tick_params(axis='both', labelsize=font_little)
	ax[1].tick_params(axis='both', labelsize=font_little)	
	ax[2].tick_params(axis='both', labelsize=font_little)	
		
	#Scientific notation	
	ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)

	#Eliminate line from label
	if style['legend'] == True:
		ax[0].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)	
		ax[1].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
		ax[2].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
	
	#Title
	if style['title'] == True:
		ax[0].set_title(keys[0], fontsize=font_big)	
		ax[1].set_title(keys[1], fontsize=font_big)
		ax[2].set_title(keys[2], fontsize=font_big)
	
	#Size from offset text
	ax[0].yaxis.offsetText.set_fontsize(font_offset)
	ax[1].yaxis.offsetText.set_fontsize(font_offset)
	ax[2].yaxis.offsetText.set_fontsize(font_offset)
	
	
	#Set Ticks in Axis Y
	# ax[0].set_yticks([-20, -10, 0, 10, 20]) 
	# ax[1].set_yticks([0, 4, 8, 12, 16]) 
	# ax[2].set_yticks([-20, -10, 0, 10, 20])
	
	#Set Ticks in Axis X
	if style['dom'] == 'other':
		ax[0].set_xticks(three_signals['dom']) 
		ax[1].set_xticks(three_signals['dom']) 
		ax[2].set_xticks(three_signals['dom'])
		
		ax[0].set_xticklabels(style['xticklabels']) 
		ax[1].set_xticklabels(style['xticklabels']) 
		ax[2].set_xticklabels(style['xticklabels'])
		
		
	
	#Set Limits in Axis X
	if style['ymax'] != None:
		ax[0].set_ylim(bottom=style['ymin'][0], top=style['ymax'][0])
		ax[1].set_ylim(bottom=style['ymin'][1], top=style['ymax'][1])
		ax[2].set_ylim(bottom=style['ymin'][2], top=style['ymax'][2])

	
	if style['xmax'] != None:
		ax[0].set_xlim(left=0, right=style['xmax'][0])
		ax[1].set_xlim(left=0, right=style['xmax'][1])
		ax[2].set_xlim(left=0, right=style['xmax'][2])
	
	#Legend
	
	print('!!!', len(style['legend']['first']))
	if style['legend'] != False:
		if len(style['legend']['first']) == 4:
			ax[2].legend(fontsize=font_label, loc=(-1.71,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
		elif len(style['legend']['first']) == 3:
			if style['legend']['first'][0].find('db') != -1:
				ax[2].legend(fontsize=font_label, loc=(-1.4,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
			else:
				ax[2].legend(fontsize=font_label, loc=(-1.5,1.15), handletextpad=0.3, labelspacing=.3, ncol=len(style['legend']['first']))
	
	blw = 1.
	ax[0].spines['top'].set_linewidth(blw)
	ax[0].spines['right'].set_linewidth(blw)
	ax[0].spines['left'].set_linewidth(blw)
	ax[0].spines['bottom'].set_linewidth(blw)
	ax[0].xaxis.set_tick_params(width=blw)
	ax[0].yaxis.set_tick_params(width=blw)
	
	ax[1].spines['top'].set_linewidth(blw)
	ax[1].spines['right'].set_linewidth(blw)
	ax[1].spines['left'].set_linewidth(blw)
	ax[1].spines['bottom'].set_linewidth(blw)
	ax[1].xaxis.set_tick_params(width=blw)
	ax[1].yaxis.set_tick_params(width=blw)
	
	ax[2].spines['top'].set_linewidth(blw)
	ax[2].spines['right'].set_linewidth(blw)
	ax[2].spines['left'].set_linewidth(blw)
	ax[2].spines['bottom'].set_linewidth(blw)
	ax[2].xaxis.set_tick_params(width=blw)
	ax[2].yaxis.set_tick_params(width=blw)
		
	if style['autolabel'] == True:
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				height2 = height*10.
				ax[0].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=font_autolabel)
		autolabel(bar0)
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				height2 = height*100.
				ax[1].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=font_autolabel)
		autolabel(bar1)
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				height2 = height*100.
				ax[2].text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%.1f' % height, ha='center', va='bottom', fontsize=font_autolabel)
		autolabel(bar2)
	
	# plt.tight_layout()
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
		
		
	return

def plot3h_paper(three_signals, style):
	# from matplotlib import font_manager
	# del font_manager.weight_dict['roman']
	# font_manager._rebuild()
	# plt.rcParams['font.family'] = 'Times New Roman'	
	fig, ax = plt.subplots(ncols=3, nrows=1, sharex=True, sharey=True)
	
	#Values Fixed
	font_big = 13
	font_little = 11.5
	font_label = 12
	font_offset = 11.5
	font_title = 13
	lim = 2
	plt.subplots_adjust(wspace=0.275, left=0.075, right=0.98, bottom=0.15, top=0.9)
	fig.set_size_inches(14.2, 4.0)
	
	
	
	#Axis X
	if style['dom'] == 'time':
		fact = 1.
		ax[0].set_xlabel('Time [s]', fontsize=font_big)
		ax[1].set_xlabel('Time [s]', fontsize=font_big)
		ax[2].set_xlabel('Time [s]', fontsize=font_big)
	elif style['dom'] == 'frequency':
		if style['kHz'] == 'ON':
			fact = 1000.
			ax[0].set_xlabel('Frequency [kHz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [kHz]', fontsize=font_big)
			ax[2].set_xlabel('Frequency [kHz]', fontsize=font_big)
		else:
			fact = 1.
			ax[0].set_xlabel('Frequency [Hz]', fontsize=font_big)
			ax[1].set_xlabel('Frequency [Hz]', fontsize=font_big)
			ax[2].set_xlabel('Frequency [Hz]', fontsize=font_big)
	#Plot	
	label_imf = 'IMF-3'
	keys = list(three_signals)
	ax[0].plot(three_signals['dom']/fact, three_signals[keys[0]], label=label_imf)
	ax[1].plot(three_signals['dom']/fact, three_signals[keys[1]], label=label_imf)
	ax[2].plot(three_signals['dom']/fact, three_signals[keys[2]], label=label_imf)
	
	#Axis Y
	if style['dom'] == 'time':
		ax[0].set_ylabel('Amplitude [mV]', fontsize=font_big)
		ax[1].set_ylabel('Amplitude [mV]', fontsize=font_big)
		ax[2].set_ylabel('Amplitude [mV]', fontsize=font_big)
		
		
		
	elif style['dom'] == 'frequency':
		ax[0].set_ylabel('Amplitude [mV]', fontsize=font_big)
		ax[1].set_ylabel('Amplitude [mV]', fontsize=font_big)
		ax[2].set_ylabel('Amplitude [mV]', fontsize=font_big)
		
		# params = {'mathtext.default': 'regular' }          
		# plt.rcParams.update(params)
		# ax[0].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[1].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)
		# ax[2].set_ylabel('Magnitude [m$V^{2}$]', fontsize=font_big)	
		
		# ax[0].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[1].set_title('Envelope spectrum', fontsize=font_offset)
		# ax[2].set_title('Envelope spectrum', fontsize=font_offset)
	
	#Size labels from axis
	ax[0].tick_params(axis='both', labelsize=font_little)
	ax[1].tick_params(axis='both', labelsize=font_little)	
	ax[2].tick_params(axis='both', labelsize=font_little)	
		
	#Scientific notation	
	ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	#Title
	ax[0].set_title('No Fault', fontsize=font_title)
	ax[1].set_title('Initial Faulty Condition', fontsize=font_title)
	ax[2].set_title('Developed Faulty Condition', fontsize=font_title)
	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)

	# #Eliminate line from label
	# ax[0].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label, loc='upper center')	
	# ax[1].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label, loc='upper center')
	# ax[2].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label, loc='upper center')
	
	#Size from offset text
	ax[0].yaxis.offsetText.set_fontsize(font_offset)
	ax[1].yaxis.offsetText.set_fontsize(font_offset)
	ax[2].yaxis.offsetText.set_fontsize(font_offset)
	
	
	#Set Ticks in Axis Y
	# ax[0].set_yticks([-20, -10, 0, 10, 20]) 
	# ax[1].set_yticks([-20, -10, 0, 10, 20]) 
	# ax[2].set_yticks([-20, -10, 0, 10, 20]) 	
	
	# Set Limits in Axis X
	# ax[0].set_xlim(left=0, right=500)
	
	# ax[2].set_xlim(left=1.01, right=1.06)
	# ax[1].set_xlim(left=1.29, right=1.34)
	# ax[0].set_xlim(left=1.58, right=1.63)

	
	# plt.tight_layout()
	plt.show()
	return