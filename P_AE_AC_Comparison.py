#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk

import os.path
import sys

sys.path.insert(0, './lib')
from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *
from m_plots import *
from decimal import Decimal

plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes


#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
from argparse import ArgumentParser


#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['mode']
InputsOpt_Defaults = {'channel':'OFF', 'fs':'OFF', 'name':'auto', 'color':None, 'marker':'D', 'output':'OFF'}

def main(argv):
    config = read_parser(argv, Inputs, InputsOpt_Defaults)

    if config['mode'] == '2_avg_spectra':
        root = Tk()
        root.withdraw()
        root.update()
        Filepaths = filedialog.askopenfilenames()
        root.destroy()    
        for filepath in Filepaths:
            print(basename(filepath))
        dict1 = read_pickle(Filepaths[0])
        dict2 = read_pickle(Filepaths[1])

        factor_f = 1.
        # /1.023077
        #*1.023077

        # f_1 = dict1['f']*1.023077
        # f_2 = dict2['f']*1.023077
        
        #ae
        f_1 = dict1['f']/1.023077
        f_2 = dict2['f']
        
        magX_1 = dict1['fft']*1000.
        magX_2 = dict2['fft']*1000. 

        
        # GearMesh
        # HarmonicsFc
        name = 'CaseA_AE_HarmonicsFc'
        path_1 = 'C:\\Felix\\60_Comparison_AE_ACC\\03_Figures\\'
        path_2 = 'C:\\Felix\\60_Comparison_AE_ACC\\12_Latex\\'        
        path_1b = path_1 + name + '.svg'
        path_2b = path_2 + name + '.pdf'
        

        # 'Magnitude [dB$_{AE}$]'
        # '$X_{\psi}^{2}$ [mV$^{2}$]'
        
        # [1.e3, 1.e4, 1.e5, 1.e6, 1.e7, 1.e8, 1.e9]
        # [1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e7, 1.e8, 1.e9]
        # r'$X_{AE}$ [dB$_{AE}$]' $\mu$V
        
        style = {'xlabel':r'$f$ [Hz]', 'ylabel':r'$X_{AE}$ [$\mu$V]', 'legend':[None, None, None, None, None, None], 'title':['AE-1', 'AE-2', 'AE-3: envelope'], 'xlim':[0,80], 'ylim':[0,8], 'color':config['color'], 'loc_legend':'upper right', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200], 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b, 'dbae':'OFF', 'ylog':'OFF', 'marker':config['marker'], 'scatter':'OFF', 'customxlabels':None, 'customylabels':None}
        # [-30, -15, 0, 15, 30]
        data = {'x':[f_1, f_2], 'y':[magX_1, magX_2]}

        # plt.rcParams['mathtext.fontset'] = 'cm'
        myplot_scatter_2h(data, style)
    
    elif config['mode'] == '6_plot_cwd':
        root = Tk()
        root.withdraw()
        root.update()
        Filepaths = filedialog.askopenfilenames()
        root.destroy()    
        for filepath in Filepaths:
            print(basename(filepath))
        dict1 = read_pickle(Filepaths[0])
        dict2 = read_pickle(Filepaths[1])
        dict3 = read_pickle(Filepaths[2])
        dict4 = read_pickle(Filepaths[3])
        dict5 = read_pickle(Filepaths[4])
        dict6 = read_pickle(Filepaths[5])
        f_1 = dict1['f']
        f_2 = dict2['f']
        f_3 = dict3['f']
        f_4 = dict4['f']
        f_5 = dict5['f']
        f_6 = dict6['f']
        factor = 1.
        factor = 1.e3
        magX_1 = dict1['fft']*factor
        magX_2 = dict2['fft']*factor
        magX_3 = dict3['fft']*factor
        magX_4 = dict4['fft']*factor
        magX_5 = dict5['fft']*factor
        magX_6 = dict6['fft']*factor
        
        # name = basename(filepath)[:-5]
        # name = name.replace('Meaning', '\\TestBench_' + config['calculation'])
        

        name = 'CaseB_AC'
        path_1 = 'C:\\Felix\\60_Comparison_AE_ACC\\03_Figures\\'
        path_2 = 'C:\\Felix\\60_Comparison_AE_ACC\\12_Latex\\'        
        path_1b = path_1 + name + '.svg'
        path_2b = path_2 + name + '.pdf'
        
        min_a = -40
        max_a = 40
        min_b = -40
        max_b = 40
        
        
        
        # min_a = 0
        # max_a = 0.005
        # min_b = 0
        # max_b = 0.02
        
        ['MC 1', 'MC 7', 'MC 10', 'MC 1', 'MC 7', 'MC 10']
        [None, None, None, None, None, None]
        # [[min_a, max_a], [min_a, max_a], [min_a, max_a], [min_b, max_b], [min_b, max_b], [min_b, max_b]]
        
        # r'Magnitude [dB$_{AE}$]'
        # [[min_y,max_y], [min_y,max_y], [min_y,max_y], [min_y,max_y], [min_y,max_y], [min_y,max_y]]
        max_y = 0.04
        max_y = 8.
        
        min_y = 0
        max_x = 175
        # [None, None, None, None, None, None]
        # ['No damage: AE-1', 'Damage: AE-1', 'No damage: AE-2', 'Damage: AE-2', 'No damage: AE-3', 'Damage: AE-3']
        # r'$X_{AE}$ [$\mu$V]'
        style = {'xlabel':'$f$ [Hz]', 'ylabel':r'$X_{AC}$ [g]', 'legend':['No damage: AC-1', 'Damage: AC-1', 'No damage: AC-2', 'Damage: AC-2', 'No damage: AC-3', 'Damage: AC-3'], 'title':None, 'customxlabels':None, 'xlim':[[0, max_x], [0, max_x], [0, max_x], [0, max_x], [0, max_x], [0, max_x]], 'ylim':[[min_y,max_y], [min_y,max_y], [min_y,max_y], [min_y,max_y], [min_y,max_y], [min_y,max_y]], 'color':[None, None, None, None, None, None], 'loc_legend':'upper left', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200], 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b, 'dbae':'OFF', 'color':config['color']}
        
        data = {'x':[f_1, f_2, f_3, f_4, f_5, f_6], 'y':[magX_1, magX_2, magX_3, magX_4, magX_5, magX_6]}
        
        myplot_6_cwd(data, style)
        
    elif config['mode'] == '3_plot_cwd':
        root = Tk()
        root.withdraw()
        root.update()
        Filepaths = filedialog.askopenfilenames()
        root.destroy()    
        for filepath in Filepaths:
            print(basename(filepath))
        dict1 = read_pickle(Filepaths[0])
        dict2 = read_pickle(Filepaths[1])
        dict3 = read_pickle(Filepaths[2])
        f_1 = dict1['f']
        f_2 = dict2['f']
        f_3 = dict3['f']
        factor = 1.
        magX_1 = dict1['fft']*factor
        magX_2 = dict2['fft']*factor
        magX_3 = dict3['fft']*factor
        
        # name = basename(filepath)[:-5]
        # name = name.replace('Meaning', '\\TestBench_' + config['calculation'])
        

        name = 'CaseB_AC_zoomfull'
        path_1 = 'D:\\2017_2019\\60_Comparison_AE_ACC\\03_Figures\\'
        path_2 = 'D:\\2017_2019\\60_Comparison_AE_ACC\\12_Latex\\'        
        path_1b = path_1 + name + '.svg'
        path_2b = path_2 + name + '.pdf'
        
        min_a = -40
        max_a = 40
        min_b = -40
        max_b = 40
        
        
        
        # min_a = 0
        # max_a = 0.005
        # min_b = 0
        # max_b = 0.02
        
        ['MC 1', 'MC 7', 'MC 10', 'MC 1', 'MC 7', 'MC 10']
        [None, None, None, None, None, None]
        # [[min_a, max_a], [min_a, max_a], [min_a, max_a], [min_b, max_b], [min_b, max_b], [min_b, max_b]]
        
        # r'Magnitude [dB$_{AE}$]'
        # [[min_y,max_y], [min_y,max_y], [min_y,max_y], [min_y,max_y], [min_y,max_y], [min_y,max_y]]
        max_y = 0.03
        # max_y = 8.
        
        min_y = 0
        max_x = 800
        min_x = 400.
        
        max_x = 1000.
        min_x = 0.
        
        # [None, None, None, None, None, None]
        # ['No damage: AE-1', 'Damage: AE-1', 'No damage: AE-2', 'Damage: AE-2', 'No damage: AE-3', 'Damage: AE-3']
        # r'$X_{AE}$ [$\mu$V]'
        style = {'xlabel':'$f$ [Hz]', 'ylabel':r'$X_{AC}$ [g]', 'legend':['Damage: AC-1', 'Damage: AC-2', 'Damage: AC-3'], 'title':None, 'customxlabels':None, 'xlim':[[min_x, max_x], [min_x, max_x], [min_x, max_x]], 'ylim':[[min_y,max_y], [min_y,max_y], [min_y,max_y]], 'color':[None, None, None], 'loc_legend':'upper right', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200], 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b, 'dbae':'OFF', 'color':config['color']}
        
        data = {'x':[f_1, f_2, f_3], 'y':[magX_1, magX_2, magX_3]}
        
        myplot_3_cwd(data, style)
    
    
    else:
        print('error mode')
        
        
        
        
    return

# plt.show()
def read_parser(argv, Inputs, InputsOpt_Defaults):
    Inputs_opt = [key for key in InputsOpt_Defaults]
    Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
    parser = ArgumentParser()
    for element in (Inputs + Inputs_opt):
        print(element)
        if element == 'no_element':
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
    
    #Type conversion to float
    # if config['power2'] != 'auto' and config['power2'] != 'OFF':
        # config['power2'] = int(config['power2'])
    # config['fs_tacho'] = float(config['fs_tacho'])
    # config['fs_signal'] = float(config['fs_signal'])
    # config['fscore_min'] = float(config['fscore_min'])
    #Type conversion to int    
    # Variable conversion
    return config

def myplot_scatter_2h(data, style):
    # from matplotlib import font_manager    
    # del font_manager.weight_dict['roman']
    # font_manager._rebuild()
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'Times New Roman'    
    
    
    # fig, ax = plt.subplots(ncols=3, nrows=1, sharey='row')
    fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
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
    plt.subplots_adjust(wspace=0.33, left=0.1, right=0.965, bottom=0.213, top=0.89)
    fig.set_size_inches(10.2, 4.0)
    fig.text(0.05, 0.03, '(a)', fontsize=font_caption)
    fig.text(0.55, 0.03, '(b)', fontsize=font_caption)

    

    count = 0
    
    for j in range(2):
        if style['dbae'] == 'ON':
            data_y = 20*np.log10(1000*data['y'][count])
        else:
            data_y = data['y'][count]
        # ax[j].set_yticks([-30, -15, 0, 15, 30])
        
        if style['ylog'] == 'ON':
            ax[j].semilogy(data['x'][count], data_y, label=style['legend'][count], color=style['color'])
        else:
            if style['scatter'] == 'ON':
                ax[j].plot(data['x'][count], data_y, label=style['legend'][count], marker=style['marker'], ls='', color=style['color'])
            # ax[j].plot(data['x'][count], data_y, label=style['legend'][count], marker='o', ls='')
            # ax[j].plot(data['x'][count], data_y, label=style['legend'][count])
            # ax[j].bar(data['x'][count], data_y, label=style['legend'][count])
            else:
                ax[j].plot(data['x'][count], data_y, label=style['legend'][count], color=style['color'])
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
            # ax[j].set_ylim(bottom=style['ylim'][count][0], top=style['ylim'][count][1])
            ax[j].set_ylim(bottom=style['ylim'][0], top=style['ylim'][1])                
        if style['xlim'] != None:
            # ax[j].set_xlim(left=style['xlim'][count][0], right=style['xlim'][count][1])
            ax[j].set_xlim(left=style['xlim'][0], right=style['xlim'][1])
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
    
        
    
        
    # ax[0].set_yticks([1.e3, 1.e4, 1.e5, 1.e6, 1.e7, 1.e8, 1.e9])
    # ax[1].set_yticks([1.e3, 1.e4, 1.e5, 1.e6, 1.e7, 1.e8, 1.e9])
    # ax[2].set_yticks([1.e4, 1.e5, 1.e6, 1.e7])
    
    # ax[0].set_yticks([-10, 10, 30, 50])    
    # ax[1].set_yticks([-10, 10, 30, 50])
    
    # ax[0].set_yticks([-10 ,0, 10, 20, 30, 40, 50])
    # ax[1].set_yticks([-10 ,0, 10, 20, 30, 40, 50])
    
    
    # ax[2].set_yticks([-50, -30, -10, 10, 30])    
    if style['customylabels'] != None:
        ax[0].set_yticks(style['customylabels'])
        ax[1].set_yticks(style['customylabels'])
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

def myplot_6_cwd(data, style):
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
    
    fig, ax = plt.subplots(ncols=2, nrows=3, sharey=True, sharex=True)
    lim = 2
    
    
    font_big = 17+3
    font_little = 15+3
    font_label = 13+3
    font_offset = 15+3
    font_autolabel = 15+3
    font_caption = 23+3+4
    plt.subplots_adjust(wspace=0.25, left=0.075, right=0.975, bottom=0.11, top=0.965, hspace=0.55)
    # hspace=0.47
    fig.set_size_inches(12.0, 9.0)
    # 6.5
    
    
    
    fig.text(0.028, 0.665+0.025, '(a)', fontsize=font_caption)
    fig.text(0.028, 0.360, '(c)', fontsize=font_caption)
    fig.text(0.028, 0.04, '(e)', fontsize=font_caption)
    
    fig.text(0.521, 0.665+0.025, '(b)', fontsize=font_caption)
    fig.text(0.521, 0.360, '(d)', fontsize=font_caption)
    fig.text(0.521, 0.04, '(f)', fontsize=font_caption)
    
    # 0.522


    count = 0
    for i in range(3):
        for j in range(2):
            print(count)
            if style['dbae'] == 'ON':
                data_y = 20*np.log10(1000*data['y'][count])
            else:
                data_y = data['y'][count]
            ax[i][j].plot(data['x'][count], data_y, label=style['legend'][count], color=style['color'])
            
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
            if style['title'] != None:
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
    
    
def myplot_3_cwd(data, style):
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
    
    fig, ax = plt.subplots(ncols=1, nrows=3, sharey=True, sharex=False)
    lim = 2
    
    
    font_big = 17+3
    font_little = 15+3
    font_label = 13+3
    font_offset = 15+3
    font_autolabel = 15+3
    font_caption = 23+3+4
    plt.subplots_adjust(wspace=0.25, left=0.075, right=0.975, bottom=0.11, top=0.965, hspace=0.55)
    # hspace=0.47
    fig.set_size_inches(10.0, 9.0)
    # 6.5
    
    
    
    fig.text(0.028, 0.665+0.025, '(a)', fontsize=font_caption)
    fig.text(0.028, 0.360, '(b)', fontsize=font_caption)
    fig.text(0.028, 0.04, '(c)', fontsize=font_caption)
    
    # fig.text(0.521, 0.665+0.025, '(b)', fontsize=font_caption)
    # fig.text(0.521, 0.360, '(d)', fontsize=font_caption)
    # fig.text(0.521, 0.04, '(f)', fontsize=font_caption)
    
    # 0.522


    count = 0
    for i in range(3):
        print(count)
        if style['dbae'] == 'ON':
            data_y = 20*np.log10(1000*data['y'][count])
        else:
            data_y = data['y'][count]
        ax[i].plot(data['x'][count], data_y, label=style['legend'][count], color=style['color'])
        
        # if i == 0:
            # ax[i][j].set_yticks([-10, 10, 30, 50])
        
        ax[i].set_xlabel(style['xlabel'], fontsize=font_big)
        ax[i].set_ylabel(style['ylabel'], fontsize=font_big)
        ax[i].tick_params(axis='both', labelsize=font_little)
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
        if style['legend'][count] != None:
            if style['legend_line'] == 'OFF':
                ax[i].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
            else:
                ax[i].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
        if style['title'] != None:
            ax[i].set_title(style['title'][count], fontsize=font_big)
        ax[i].yaxis.offsetText.set_fontsize(font_offset)
        if i == 1:
            if style['customxlabels'] != None:
                # ax[i][j].set_xticklabels(style['xticklabels'])
                # ax[i][j].set_xticks(style['xticklabels'])
                ax[i].set_xticklabels(style['customxlabels'])
                ax[i].set_xticks(style['customxlabels'])
        if style['ylim'] != None:
            ax[i].set_ylim(bottom=style['ylim'][count][0], top=style['ylim'][count][1])        
        if style['xlim'] != None:
            ax[i].set_xlim(left=style['xlim'][count][0], right=style['xlim'][count][1])
        ax[i].grid(axis='both')
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

if __name__ == '__main__':
    main(sys.argv)
